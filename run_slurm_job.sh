#!/bin/bash
#SBATCH --job-name=exp_cifar10
#SBATCH --output=logs/slurm/%x_%j.out   # Log output to logs/<job-name>_<job-id>.out
#SBATCH --error=logs/slurm/%x_%j.err    # Error logs (optional)
#SBATCH --time=6:00:00            # Max walltime
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=16G                 # Memory


# Load your environment
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run training script with Hydra config
srun -c 4 python3 src/train.py -m \
    trainer=ddp \
    trainer.devices=2 \
    experiment=exp_cifar10_resnet \
    data.train_subset=25,50,75,100 \
    data.num_workers=4 \
    logger=wandb
