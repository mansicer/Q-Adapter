# Q-Adapter

The implementation of the paper "Q-Adapter: Training Your LLM Adapter as a Residual Q-Function", [arXiv link](https://arxiv.org/abs/2407.03856).

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
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml qadapter_train.py \
--base_model=meta-llama/Meta-Llama-3-8B-Instruct --dataset_name=dsp --data_dir=data/dsp/dsp_${data_class}_pairs.train.json \
--logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=1 \
--wandb_project=Q-Adapter --wandb_run_name=QAdapter-${data_class} --output_dir=logs/qadapter-${data_class}
```

We use `accelerate` to launch an experiment with 4 GPU chips. You can adjust the accelerate config by running `accelerate config` to create a config for your training platform. The other arguments can also be modified according to the machine capabilities and the dataset you want to train on. We enable `wandb` tracking by default, where the input information can be modified through command line arguments.

To train the Q-Adapter in the HH-RLHF dataset, you can run the following command:

```bash
data_class=helpful-base # data_class can be helpful-base or harmless-base
CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=true accelerate launch --config_file accelerate_config.yaml qadapter_train.py \
--base_model=meta-llama/Meta-Llama-3-8B --dataset_name=hh-rlhf --data_dir=${data_class} \
--logging_steps=20 --eval_steps=100 --num_epochs=3 --micro_batch_size=1 \
--wandb_project=Q-Adapter --wandb_run_name=QAdapter-${data_class} --output_dir=logs/qadapter-${data_class}
```
## Citation

```tex
@misc{li2024qadaptertrainingllmadapter,
      title={Q-Adapter: Training Your LLM Adapter as a Residual Q-Function}, 
      author={Yi-Chen Li and Fuxiang Zhang and Wenjie Qiu and Lei Yuan and Chengxing Jia and Zongzhang Zhang and Yang Yu},
      year={2024},
      eprint={2407.03856},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.03856}, 
}
```
