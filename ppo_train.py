import datetime
import json
import os
import sys
from typing import List

import fire
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig

from utils.datasets import load_ppo_data
from utils.models import get_transformers_tokenizer, get_ppo_model, get_transformers_model

import warnings

# disable warnings
warnings.filterwarnings("ignore")


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    rm_model: str = "",
    dataset_name: str = "dsp",
    data_dir: str = "data/dsp/dsp_academy_pairs.train.json",
    output_dir: str = "./logs",
    # lora hyperparams
    lora_r: int = 0,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # training hyperparams
    cutoff_len: int = 512,
    add_eos_token: bool = False,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    total_steps: int = 300,
    learning_rate: float = 3e-4,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    # Initialize & preprocess configs
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        torch.cuda.set_device(local_rank)  # Set the current process to use the correct GPU

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    if use_wandb:
        wandb_run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M") if len(wandb_run_name) == 0 else f"{wandb_run_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # output_dir = os.path.join(output_dir, wandb_run_name if len(wandb_run_name) > 0 else datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        configs = dict(**locals())
        print(f"Training Q-Adapter model with params:\n" + "\n".join([f"{k}: {v}" for k, v in configs.items()]) + "\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    _, train_data, val_data = load_ppo_data(dataset_name=dataset_name, data_dir=data_dir, tokenizer=tokenizer, cutoff_len=cutoff_len, add_eos_token=add_eos_token)

    model, ref_model = get_ppo_model(base_model, tokenizer, lora_r, lora_alpha, lora_dropout, device_map=device_map, load_in_8bit=lora_r > 0)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        json.dump(configs, open(os.path.join(output_dir, "training_config.json"), "w"), indent=4, ensure_ascii=False)

    reward_model = AutoModelForSequenceClassification.from_pretrained(rm_model, torch_dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=device_map)
    sentiment_pipe = transformers.pipeline("text-classification", model=reward_model, tokenizer=tokenizer, device_map=device_map)
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if sentiment_pipe.tokenizer.pad_token_id is None:
        sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
    
    learning_rate = learning_rate if lora_r > 0 else 5e-5
    ppo_config = PPOConfig(
        exp_name=wandb_run_name,
        task_name=wandb_run_name,
        steps=total_steps,
        learning_rate=learning_rate,
        batch_size=micro_batch_size*gradient_accumulation_steps,
        mini_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
    }

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_data,
        data_collator=collator
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        if lora_r == 0:
            ref_model = torch.compile(ref_model)
    
    print(f"Epoch length: {len(ppo_trainer.dataloader)}")
    exp_end = False
    for _ in range(num_epochs):
        for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = [input_ids.to(ppo_trainer.model.current_device) for input_ids in batch["input_ids"]]

            # Get response from gpt2
            response_tensors, ref_response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            rewards = [torch.tensor(output[1]["score"]).to(ppo_trainer.model.current_device) for output in pipe_outputs]
            ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
            ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
            ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
            batch["ref_rewards"] = ref_rewards

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

            if _epoch >= total_steps - 1:
                exp_end = True
                break
        
        if exp_end:
            break

    ppo_trainer.save_pretrained(output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
