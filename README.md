# Q-Adapter

The implementation of the manuscript "Q-Adapter: Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation".

## Installation

You may need a cuda installation to run the code with DeepSpeed. You can create a conda virtual environment with the following command:

```bash
conda create -n qadapter python=3.10
conda activate qadapter
```

The rest of the dependencies can be installed with the following command:

```bash
pip install -r requirements.txt
```

## Training Q-Adapter

To train the Q-Adapter with the default configuration in the DSP dataset, you can run the following command:

```bash
data_class=academy # data_class can be academy, business, entertainment, or literature
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml qadapter_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=dsp --data_dir=data/chat/dsp/dsp_${data_class}_pairs.train.json --lora_r=8 --logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=1 --wandb_project=Q-Adapter --wandb_run_name=QAdapter-${data_class} --output_dir=logs/qadapter-dsp-${data_class}
```

We use `accelerate` to launch an experiment with 4 GPUs. You can adjust the accelerate config by running `accelerate config` to create a config for your training platform. The other arguments can also be modified according to the machine capabilities and the dataset you want to train on. We enable `wandb` tracking by default, where the input information can be modified through command line arguments. All the scripts listed here should work well with 4x NVIDIA GeForce RTX 4090 GPUs. 

Alternatively, we can train Q-Adapter in the HH-RLHF dataset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml qadapter_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=hh-rlhf --data_dir=data/chat/hh-rlhf/harmless.train.json --lora_r=8 --logging_steps=20 --eval_steps=100 --num_epochs=2 --micro_batch_size=1 --wandb_project=Q-Adapter --wandb_run_name=QAdapter-harmless --output_dir=logs/qadapter-hh-harmless
```

### Baselines

We also provide scripts to run our baselines. For example, the following script runs SFT in the academy dataset of DSP:

```bash
data_class=academy # data_class can be academy, business, entertainment, or literature
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml sft_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=dsp --data_dir=data/chat/dsp/dsp_${data_class}_pairs.train.json --lora_r=8 --logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=4 --wandb_project=Q-Adapter --wandb_run_name=SFT-${data_class} --output_dir=logs/sft-dsp-${data_class}
```

DPO: 

```bash
data_class=academy # data_class can be academy, business, entertainment, or literature
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml dpo_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=dsp --data_dir=data/chat/dsp/dsp_${data_class}_pairs.train.json --lora_r=8 --logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=2 --wandb_project=Q-Adapter --wandb_run_name=DPO-${data_class} --output_dir=logs/dpo-dsp-${data_class}
```

Replay: This baseline requires us to generate additional replay data before training. 

```bash
# data generate
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python make_replay_data.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=dsp --data_path=data/chat/dsp/dsp_academy_pairs.train.json --inference_batch_size=32 --max_new_tokens=256
# SFT
data_class=academy # data_class can be academy, business, entertainment, or literature
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml sft_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=dsp --data_dir=data/replay/dsp/dsp_${data_class}_pairs.train.json --lora_r=8 --logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=4 --wandb_project=Q-Adapter --wandb_run_name=Replay-${data_class} --output_dir=logs/replay-dsp-${data_class}
```

PPO: We find that there may encounter bugs with TRL when executing multi-GPU experiments with PPOTrainer, so we provide the script with single GPU.
```bash
# Train RM
data_class=academy # data_class can be academy, business, entertainment, or literature
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_zero3.yaml rm_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name=dsp --data_dir=data/chat/dsp/dsp_${data_class}_pairs.train.json --logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=8 --wandb_project=Q-Adapter-Tuned --wandb_run_name=RM-${data_class} --output_dir=logs/rm-dsp-${data_class}
# Train PPO
data_class=academy # data_class can be academy, business, entertainment, or literature
CUDA_VISIBLE_DEVICES=0 python ppo_train.py --base_model=meta-llama/Meta-Llama-3.1-8B-Instruct --rm_model=logs/rm/rm-dsp-${data_class} --dataset_name=dsp --data_dir=data/chat/dsp/dsp_${data_class}_pairs.train.json --lora_r=8 --total_steps=500 --num_epochs=5 --micro_batch_size=8 --wandb_project=Q-Adapter-Tuned --wandb_run_name=PPO-${data_class} --output_dir=logs/ppo-dsp-${data_class}
```

## Evaluate Q-Adapter

We use `lm-eval` to evaluate the performance of all methods in a modified repo to implement the inference process of Q-Adapter. You should first install the library: 

```bash
pip install -e lm-evaluation-harness
```

Afterward, you can evaluate any model with the following script:

```bash
# MMLU
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,load_in_8bit=True,peft=your/peft/model/path --tasks mmlu --device=cuda --batch_size=auto --trust_remote_code --show_config
# Other benchmarks
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,load_in_8bit=True,peft=your/peft/model/path --tasks gsm8k_cot,leaderboard_mmlu_pro,leaderboard_bbh,leaderboard_ifeval --device=cuda --batch_size=auto --trust_remote_code --show_config --apply_chat_template --fewshot_as_multiturn
```
