import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.peft_model import PeftModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import TrainerCallback, TrainingArguments


class MetricCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
        self.log_dict = {}

    def on_step_begin(self, args, state, control, **kwargs):
        self.log_dict.clear()

    def on_log(self, args, state, control, logs, **kwargs):
        log_dict = {k: np.array(v).mean() for k, v in self.log_dict.items()}
        logs.update(log_dict)


def qadapter_loss(q_values_ai, values, mask, gamma, beta=0.1):
    loss = 0.0
    log_dict = {}

    reward = q_values_ai - gamma * values
    reward_sum = (reward * mask).sum(dim=1)
    bs = reward_sum.shape[0] // 2
    logits_chosen, logits_rejected = reward_sum[:bs], reward_sum[bs:]
    logits = torch.stack([logits_chosen, logits_rejected], dim=-1)
    logits /= reward.shape[1]
    labels = torch.tensor([1] * bs).to(reward_sum.device)
    pref_loss = F.cross_entropy(logits, labels)
    loss += pref_loss

    reg_loss = beta * (reward**2 * mask).sum() / mask.sum()
    loss += reg_loss

    log_dict["pref_loss"] = pref_loss.item()
    log_dict["chi2_loss"] = reg_loss.item()
    log_dict["q_values_mean"] = ((q_values_ai * mask).sum() / mask.sum()).item()
    log_dict["values_mean"] = ((values * mask).sum() / mask.sum()).item()
    log_dict["reward_mean"] = ((reward * mask).sum() / mask.expand_as(reward).sum()).item()
    return loss, log_dict


def modify_forward(peft_model: PeftModelForCausalLM, alpha_tilde, alpha_0, gamma, beta, register_callback=False):
    if register_callback:
        log_callback = MetricCallback()

    def forward(input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            past_key_values = (None, None)

        adapter_model_output = peft_model.__original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values[0], **kwargs)
        with peft_model.disable_adapter():
            base_model_output = peft_model.__original_forward(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values[1], **kwargs)
        adapter_model_logits = adapter_model_output.logits.float().contiguous()
        base_model_logits = base_model_output.logits.float().contiguous()

        q_values = adapter_model_logits
        base_log_pi = F.log_softmax(base_model_logits, dim=-1).detach()
        
        if attention_mask is not None:
            if labels is not None:
                input_mask = attention_mask.bool() & (labels != -100)
            else:
                input_mask = attention_mask.bool()
            values = alpha_tilde * torch.logsumexp((q_values + alpha_0 * base_log_pi) / alpha_tilde, dim=-1)
            q_values_ai = torch.gather(q_values[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            loss, log_dict = qadapter_loss(q_values_ai=q_values_ai, values=values[:, :-1], mask=input_mask[:, :-1], gamma=gamma, beta=beta)

            logits = (q_values + alpha_0 * base_log_pi) / alpha_tilde
            log_dict["base_log_pi"] = base_log_pi.mean().item()
            log_dict["logits_mean"] = logits.mean().item()

            if register_callback:
                for k, v in log_dict.items():
                    arr = log_callback.log_dict.get(k, [])
                    arr.append(v)
                    log_callback.log_dict[k] = arr
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=(adapter_model_output.past_key_values, base_model_output.past_key_values), hidden_states=None, attentions=None)
        else:
            logits = (q_values + alpha_0 * base_log_pi) / alpha_tilde
            return CausalLMOutputWithPast(logits=logits, past_key_values=(adapter_model_output.past_key_values, base_model_output.past_key_values), hidden_states=None, attentions=None)

    peft_model.__original_forward = peft_model.forward
    peft_model.forward = forward
    if register_callback:
        return peft_model, log_callback
    else:
        return peft_model


def modify_prepare_inputs_for_generation(peft_model: PeftModelForCausalLM):

    def prepare_inputs_for_generation(input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        past_key_values_0, past_key_values_1 = past_key_values if past_key_values is not None else (None, None)
        inputs_embeds_0, inputs_embeds_1 = inputs_embeds if inputs_embeds is not None else (None, None)
        inputs = peft_model.__original_prepare_inputs_for_generation(input_ids, past_key_values=past_key_values_0, attention_mask=attention_mask, inputs_embeds=inputs_embeds_0, **kwargs)
        if "past_key_values" in inputs:
            inputs["past_key_values"] = (past_key_values_0, past_key_values_1)
        if "inputs_embeds" in inputs:
            inputs["inputs_embeds"] = (inputs_embeds_0, inputs_embeds_1)
        return inputs

    peft_model.__original_prepare_inputs_for_generation = peft_model.prepare_inputs_for_generation
    peft_model.prepare_inputs_for_generation = prepare_inputs_for_generation
    return peft_model
