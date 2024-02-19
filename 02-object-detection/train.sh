#!/bin/bash
PARTITION="short"
IDUN_TIME=00:09:00
JOB_NAME=$(date "+%Y-%m-%d-%H:%M:%S")-testing-dataset
CURRENT_PATH=$(pwd)
OUTPUT_FOLDER=output

sbatch --partition=$PARTITION \
    --account=ie-idi \
    --time=$IDUN_TIME \
    --nodes=1 \
    --ntasks-per-node=1 \
    --mem=50G \
    --gres=gpu:1 \
    --job-name=$JOB_NAME \
    --output=$OUTPUT_FOLDER/$JOB_NAME \
    --export=IDUN_TIME=$IDUN_TIME \
              train.slurm
