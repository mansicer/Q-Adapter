import os
import sys

import fire
import torch
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig, pipeline, AutoModelForSequenceClassification

from utils.models import get_transformers_tokenizer, get_transformers_model, get_qadapter_model
from utils.prompter import *

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def evaluate(model, tokenizer, texts, max_new_tokens=256, temperature=0.1, do_sample=False, **kwargs):
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        stop_strings=["<|eot_id|>"], # assistant stop token for Llama-3.1
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            tokenizer=tokenizer,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    results = generation_output.sequences
    results = [tokenizer.decode(s, skip_special_tokens=True) for s in results]
    return results


def remove_duplicate_questions(data):
    seen_questions = set()
    filtered_data = []

    for item in data:
        user_question = item["chosen"][0]["content"]
        if user_question not in seen_questions:
            seen_questions.add(user_question)
            filtered_data.append(item)

    return filtered_data


def main(
    base_model: str = "",
    lora_path: str = "",
    dataset_name: str = "",
    data_path: str = "",
    output_dir: str = "data/replay",
    inference_batch_size: int = 8,
    num_samples: int = -1,
    max_new_tokens=256,
):
    train_data = load_dataset("json", data_files={"train": data_path}, split="train")
    num_samples = len(train_data) if num_samples == -1 else num_samples
    pure_data = remove_duplicate_questions(train_data)
    pure_data = pure_data[:num_samples]
    
    tokenizer = get_transformers_tokenizer(base_model, padding_side="left")
    model = get_transformers_model(base_model=base_model, tokenizer=tokenizer, lora_path=lora_path, device_map=device, load_in_8bit=True)
    
    model = model.eval()
    model.config.use_cache = True
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # texts = [tokenizer.apply_chat_template(sample["chosen"][:-1], tokenize=False, add_generation_prompt=True) for sample in tqdm(pure_data)]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    pipe_in = [item["chosen"][:-1] for item in pure_data]
    outputs = []
    for output in tqdm(pipe(
            pipe_in, 
            batch_size=inference_batch_size,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            stop_strings=["<|eot_id|>"],
        )):
        outputs.append(output[0]["generated_text"])

    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    for i, out in tqdm(enumerate(outputs), desc="Add replay data to dataset"):
        chat_messages = deepcopy(pure_data[i])
        chat_messages["chosen"] = out
        train_data = train_data.add_item(chat_messages)
        # train_data.append(chat_messages)
    train_data.to_json(os.path.join(output_dir, os.path.basename(data_path)))
    # with open(os.path.join(output_dir, os.path.basename(data_path)), "w") as f:
    #     for item in train_data:
    #         f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
