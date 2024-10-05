import datetime
import json
import os
import sys
from typing import List

import fire
import torch
import transformers

from transformers.integrations.integration_utils import (
    TensorBoardCallback,
    WandbCallback,
)

from utils.datasets import DataCollatorPreference, load_rm_data
from utils.models import get_transformers_tokenizer, get_qadapter_model


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    dataset_name: str = "dsp",
    data_dir: str = "data/dsp/dsp_academy_pairs.train.json",
    output_dir: str = "./logs",
    # Q-Adapter hyperparams
    alpha_tilde=1.0,
    alpha_0=0.01,
    gamma=1.0,
    beta=0.1,
    # training hyperparams
    batch_size: int = 512,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    cutoff_len: int = 512,
    logging_steps: int = 10,
    eval_steps: int = 200,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
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
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

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
    tokenizer = get_transformers_tokenizer(base_model, padding_side="right")

    # Load data
    _, train_data, val_data = load_rm_data(
        dataset_name=dataset_name,
        data_dir=data_dir,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        train_on_inputs=train_on_inputs,
        add_eos_token=add_eos_token,
    )
    data_collator = DataCollatorPreference(tokenizer=tokenizer)

    # Create model
    model, callback = get_qadapter_model(
        model_path=base_model, 
        tokenizer=tokenizer, 
        alpha_tilde=alpha_tilde, 
        alpha_0=alpha_0, 
        gamma=gamma, 
        beta=beta, 
        lora_r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        device_map=device_map, 
        load_in_8bit=True, 
        register_callback=True
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    callbacks = [callback, TensorBoardCallback()]
    if use_wandb:
        callbacks.append(WandbCallback())

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        json.dump(configs, open(os.path.join(output_dir, "training_config.json"), "w"), indent=4, ensure_ascii=False)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=callbacks,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            group_by_length=group_by_length,
            report_to="none",
            run_name=wandb_run_name if use_wandb else None,
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
