# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=cifar10_resnet_mcd trainer.min_epochs=1 trainer.max_epochs=1 data.train_subset=1000 logger=wandb_csv
```

## Train a sweep of experiments SLURM configuration

```bash
python src/train.py -m trainer.devices=4 experiment=cifar10_resnet_mcd_sweep logger=wandb_csv
```

## Train ResNet with SAM

```bash
python src/train.py -m experiment=cifar10_resnet_sam_mcd data.train_subset=12500,25000,37500,50000 model.net_config.arch=resnet18,resnet34,resnet50 trainer=ddp trainer.devices=4 logger=wandb_csv
```

## Train ViT

```bash
python src/train.py experiment=cifar10_vit_mcd logger=csv
```

### Train ViT with sweep (no SLURM) using DDP strategy with multi-GPUs

Note that `batch_size` must be divisible by the number of GPUs

```bash
python src/train.py -m experiment=cifar10_vit_mcd data.train_subset=12500,25000,37500,50000 trainer=ddp trainer.devices=4 data.batch_size=512 logger=wandb_csv
```

### Train ViT with sweep (no SLURM) with hydra-joblib-launcher (like SLURM array)

```bash
python src/train.py -m experiment=cifar10_vit_mcd data.train_subset=12500,25000,37500,50000 logger=wandb_csv
```

### Evaluate a model trained with MCD:

We pass the same configuration used during training

```bash
python src/eval.py \
  --config-path "$(pwd)/logs/train/multiruns/2026-01-14_05-45-18/0/.hydra" \
  --config-name config.yaml \
  ckpt_path="'$(pwd)/logs/train/multiruns/2026-01-14_05-45-18/0/csv/version_0/checkpoints/epoch=199-step=5000.ckpt'" \
  trainer.devices=1 \
  trainer.accelerator=gpu
```

### Train and evaluate a deep ensemble of WideResNet models

```bash
python src/train.py -m experiment=cifar10_resnet data.train_subset=12500,25000,37500,50000 seed='range(1234, 1244)' trainer=ddp trainer.devices=4 data.batch_size=512 logger=wandb_csv
```

### Train ViT of different sizes

```bash
python src/train.py -m experiment=cifar10_vit_mcd data.train_subset=50000 model.net_config.depth=6,12,24 model.net_config.heads=8,16,24  model.net_config.patch=16,32 trainer=ddp trainer.devices=4 logger=wandb_csv
```
