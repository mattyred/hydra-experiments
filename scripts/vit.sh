#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# CONFIG
# -----------------------
NUM_GPUS=4
EXPERIMENT="cifar10_vit_mcd"
TRAIN_SUBSETS=(12500 25000 37500 50000)
BATCH_SIZE=128

COMMON_ARGS="trainer.devices=1 logger=wandb_csv"

# -----------------------
# LAUNCH
# -----------------------
echo "Launching ${#TRAIN_SUBSETS[@]} jobs on ${NUM_GPUS} GPUs"

for i in "${!TRAIN_SUBSETS[@]}"; do
  GPU_ID=$(( i % NUM_GPUS ))
  SUBSET=${TRAIN_SUBSETS[$i]}

  echo "Job $i → GPU $GPU_ID (train_subset=$SUBSET)"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
    python src/train.py \
      experiment=$EXPERIMENT \
      data.train_subset=$SUBSET \
      data.batch_size=$BATCH_SIZE \
      trainer.devices=1 \
      logger=wandb_csv \
      &

done

wait
echo "All jobs completed."
