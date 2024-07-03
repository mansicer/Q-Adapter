from datasets import load_dataset
from utils.prompter import *


class DataCollatorPreference:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        data = {k: [f[k] for f in data] for k in data[0]}
        batch = dict(input_ids=data["input_ids_chosen"] + data["input_ids_rejected"], attention_mask=data["attention_mask_chosen"] + data["attention_mask_rejected"], labels=data["labels_chosen"] + data["labels_rejected"])

        # Pad labels
        labels = batch["labels"]
        max_label_length = max(len(l) for l in labels)
        padding_side = self.tokenizer.padding_side
        for i, label in enumerate(labels):
            remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(label))
            labels[i] = labels[i] + remainder if padding_side == "right" else remainder + labels[i]

        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        return batch


def tokenize(tokenizer, prompt, cutoff_len: int = 256, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result


def dsp_preprocess(data_sample):
    texts = data_sample["text"]
    scores = data_sample["score"]
    if scores[0] > scores[1]:
        chosen, rejected = texts
    else:
        rejected, chosen = texts
    return chosen, rejected, DSPPrompter.get_instruction(chosen)


def hh_rlhf_preprocess(data_sample):
    chosen, rejected = data_sample["chosen"], data_sample["rejected"]
    return chosen, rejected, HHRLHFPrompter.get_instruction(chosen)


def rm_preprocess(preprocess_fn, tokenizer, cutoff_len: int = 256, add_eos_token=True, train_on_inputs=True):

    def process_fn(data_sample):
        chosen, rejected, instruction = preprocess_fn(data_sample)
        chosen_tokens = tokenize(tokenizer, chosen, cutoff_len=cutoff_len, add_eos_token=add_eos_token)
        rejected_tokens = tokenize(tokenizer, rejected, cutoff_len=cutoff_len, add_eos_token=add_eos_token)
        instruction_tokens = tokenize(tokenizer, instruction, add_eos_token=add_eos_token)
        instruction_len = (len(instruction_tokens["input_ids"]) - 1) if add_eos_token else len(instruction_tokens["input_ids"])

        if not train_on_inputs:
            chosen_tokens["labels"] = [-100] * instruction_len + chosen_tokens["labels"][instruction_len:]
            rejected_tokens["labels"] = [-100] * instruction_len + rejected_tokens["labels"][instruction_len:]

        return dict(
            input_ids_chosen=chosen_tokens["input_ids"],
            attention_mask_chosen=chosen_tokens["attention_mask"],
            labels_chosen=chosen_tokens["labels"],
            input_ids_rejected=rejected_tokens["input_ids"],
            attention_mask_rejected=rejected_tokens["attention_mask"],
            labels_rejected=rejected_tokens["labels"],
        )

    return process_fn


def sft_preprocess(preprocess_fn, tokenizer, cutoff_len: int = 256, add_eos_token=True, train_on_inputs=True):

    def process_fn(data_sample):
        chosen, rejected, instruction = preprocess_fn(data_sample)
        chosen_tokens = tokenize(tokenizer, chosen, cutoff_len=cutoff_len, add_eos_token=add_eos_token)
        instruction_tokens = tokenize(tokenizer, instruction, add_eos_token=add_eos_token)
        instruction_len = (len(instruction_tokens["input_ids"]) - 1) if add_eos_token else len(instruction_tokens["input_ids"])
        if not train_on_inputs:
            chosen_tokens["labels"] = [-100] * instruction_len + chosen_tokens["labels"][instruction_len:]
        return dict(input_ids=chosen_tokens["input_ids"], attention_mask=chosen_tokens["attention_mask"], labels=chosen_tokens["labels"])

    return process_fn


def dpo_preprocess(preprocess_fn):

    def process_fn(data_sample):
        chosen, rejected, instruction = preprocess_fn(data_sample)
        chosen_text = chosen[len(instruction):]
        rejected_text = rejected[len(instruction):]

        return dict(prompt=instruction, chosen=chosen_text, rejected=rejected_text)

    return process_fn


def ppo_preprocess(preprocess_fn, tokenizer, cutoff_len: int = 256, add_eos_token=True):

    def process_fn(data_sample):
        chosen, rejected, instruction = preprocess_fn(data_sample)
        input_ids = tokenizer.encode(
            instruction, 
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )
        query = tokenizer.decode(input_ids)
        return dict(query=query, input_ids=input_ids)

    return process_fn


def load_sft_data(dataset_name, data_dir, tokenizer, cutoff_len: int = 256, train_on_inputs=False, add_eos_token=True):
    if "dsp" in dataset_name.lower():
        test_data_path = data_dir.replace("train", "test")
        train_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="train")
        test_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="test")
        process_fn = sft_preprocess(preprocess_fn=dsp_preprocess, tokenizer=tokenizer, cutoff_len=cutoff_len, add_eos_token=add_eos_token, train_on_inputs=train_on_inputs)
    elif "hh-rlhf" in dataset_name.lower():
        train_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="train")
        test_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="test")
        process_fn = sft_preprocess(preprocess_fn=hh_rlhf_preprocess, tokenizer=tokenizer, cutoff_len=cutoff_len, add_eos_token=add_eos_token, train_on_inputs=train_on_inputs)
    
    train_data = train_data.shuffle().map(process_fn, remove_columns=train_data.column_names)
    test_data = test_data.shuffle().map(process_fn, remove_columns=test_data.column_names)
    return None, train_data, test_data


def load_rm_data(dataset_name, data_dir, tokenizer, cutoff_len: int = 256, train_on_inputs=False, add_eos_token=True):
    if "dsp" in dataset_name.lower():
        test_data_path = data_dir.replace("train", "test")
        train_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="train")
        test_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="test")
        process_fn = rm_preprocess(preprocess_fn=dsp_preprocess, tokenizer=tokenizer, cutoff_len=cutoff_len, add_eos_token=add_eos_token, train_on_inputs=train_on_inputs)
    elif "hh-rlhf" in dataset_name.lower():
        train_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="train")
        test_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="test")
        process_fn = rm_preprocess(preprocess_fn=hh_rlhf_preprocess, tokenizer=tokenizer, cutoff_len=cutoff_len, add_eos_token=add_eos_token, train_on_inputs=train_on_inputs)
    
    train_data = train_data.shuffle().map(process_fn, remove_columns=train_data.column_names)
    test_data = test_data.shuffle().map(process_fn, remove_columns=test_data.column_names)
    return None, train_data, test_data


def load_dpo_data(dataset_name, data_dir):
    if "dsp" in dataset_name.lower():
        test_data_path = data_dir.replace("train", "test")
        train_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="train")
        test_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="test")
        process_fn = dpo_preprocess(preprocess_fn=dsp_preprocess)
    elif "hh-rlhf" in dataset_name.lower():
        train_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="train")
        test_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="test")
        process_fn = dpo_preprocess(preprocess_fn=hh_rlhf_preprocess)

    train_data = train_data.shuffle().map(process_fn, remove_columns=train_data.column_names)
    test_data = test_data.shuffle().map(process_fn, remove_columns=test_data.column_names)
    return None, train_data, test_data


def load_ppo_data(dataset_name, data_dir, tokenizer, cutoff_len: int = 256, add_eos_token=True):
    if "dsp" in dataset_name.lower():
        test_data_path = data_dir.replace("train", "test")
        train_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="train")
        test_data = load_dataset("json", data_files={"train": data_dir, "test": test_data_path}, split="test")
        process_fn = ppo_preprocess(preprocess_fn=dsp_preprocess, tokenizer=tokenizer, cutoff_len = 256, add_eos_token=True)
    elif "hh-rlhf" in dataset_name.lower():
        train_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="train")
        test_data = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="test")
        process_fn = ppo_preprocess(preprocess_fn=hh_rlhf_preprocess, tokenizer=tokenizer, cutoff_len = 256, add_eos_token=True)

    train_data = train_data.shuffle().map(process_fn, remove_columns=train_data.column_names)
    test_data = test_data.shuffle().map(process_fn, remove_columns=test_data.column_names)
    train_data.set_format(type="torch")
    test_data.set_format(type="torch")
    return None, train_data, test_data
