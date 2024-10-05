import datetime
import json
import os
import sys
from typing import List

import fire
import torch
import transformers

from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead

from utils.q_adapter import modify_forward, modify_prepare_inputs_for_generation


def get_transformers_tokenizer(tokenizer_path, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def init_transformers_model(model_class, model_path, tokenizer, device_map, load_in_8bit, dtype=torch.float16):
    if load_in_8bit:
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
        )
    else:
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    if "llama" in model_path.lower():
        model.config.pad_token_id = tokenizer.pad_token_id
    return model

def get_transformers_model(base_model, tokenizer, model_class=AutoModelForCausalLM, lora_path="", lora_r=0, lora_alpha=16, lora_dropout=0.05, lora_target_modules=None, device_map="auto", load_in_8bit=False):
    dtype = torch.bfloat16 if lora_path == "" and lora_r == 0 and not load_in_8bit else torch.float16
    model = init_transformers_model(model_class, base_model, tokenizer, device_map, load_in_8bit, dtype=dtype)
    
    if len(lora_path) > 0:
        model = PeftModel.from_pretrained(model, lora_path)
    
    if lora_r > 0:
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    return model

def get_dpo_model(model_path, tokenizer, lora_r=0, lora_alpha=16, lora_dropout=0.05, lora_target_modules=None, device_map="auto", load_in_8bit=False):
    dtype = torch.bfloat16 if lora_r == 0 and not load_in_8bit else torch.float16
    ref_model = init_transformers_model(AutoModelForCausalLM, model_path, tokenizer, device_map, load_in_8bit, dtype=dtype)
    
    if lora_r > 0:
        ref_model = prepare_model_for_kbit_training(ref_model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(ref_model, config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        return model, None
    else:
        model = init_transformers_model(AutoModelForCausalLM, model_path, tokenizer, device_map, load_in_8bit=False) # Trainable model without 8bit quantization
        return model, ref_model


def get_ppo_model(model_path, tokenizer, lora_r=0, lora_alpha=16, lora_dropout=0.05, device_map="auto", load_in_8bit=False):
    if lora_r > 0:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_path,
            peft_config=config,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        return model, None
    else:
        ref_model = init_transformers_model(AutoModelForCausalLMWithValueHead, model_path, tokenizer, device_map, load_in_8bit)
        model = init_transformers_model(AutoModelForCausalLMWithValueHead, model_path, tokenizer, device_map, load_in_8bit=False) # Trainable model without 8bit quantization
        return model, ref_model


def get_qadapter_model(model_path, tokenizer, alpha_tilde, alpha_0, gamma, beta, lora_path="", lora_r=0, lora_alpha=16, lora_dropout=0.05, lora_target_modules=None, device_map="auto", load_in_8bit=False, register_callback=False):
    model = init_transformers_model(AutoModelForCausalLM, model_path, tokenizer, device_map, load_in_8bit, dtype=torch.float16)
    
    assert len(lora_path) > 0 or lora_r > 0, f"Q-Adapter must have `len(lora_path) ({len(lora_path)}) > 0 or lora_r ({lora_r}) > 0` "
    
    if len(lora_path) > 0:
        model = PeftModel.from_pretrained(model, lora_path)
    
    if lora_r > 0:
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
    model = modify_prepare_inputs_for_generation(model)
    result = modify_forward(model, alpha_tilde=alpha_tilde, alpha_0=alpha_0, gamma=gamma, beta=beta, register_callback=register_callback)
    return result
