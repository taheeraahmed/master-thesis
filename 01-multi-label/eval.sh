#!/bin/sh
echo "Running eval.sh"

sbatch --partition=GPUQ \
    --account=share-ie-idi \
    --time=00:10:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=50G \
    --gres=gpu:1 \
    --job-name=pred \
    --output=/cluster/home/taheeraa/code/master-thesis/01-multi-label/output/2024-04-07-12:55:58-alexnet-focal-multi-label-e35-bs32-lr0.0005-t45:00:00/pred_out.out \
    /cluster/home/taheeraa/code/master-thesis/01-multi-label/eval.slurm