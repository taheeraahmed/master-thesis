#!/bin/sh
echo "Running eval.sh"


sbatch --partition=GPUQ \
    --account=ie-idi \
    --time=00:10:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=50G \
    --gres=gpu:1 \
    --job-name=pred \
    --output=eval_out.out \
    /cluster/home/taheeraa/code/master-thesis/01-multi-label/eval.slurm