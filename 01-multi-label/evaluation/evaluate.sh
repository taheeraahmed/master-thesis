#!/bin/bash
PARTITION="CPUQ"
ACCOUNT=share-ie-idi
NUM_CORES=8

if [ "$PARTITION" = "GPUQ" ]; then
    echo "starting gpu gig"
    echo ""
    sbatch --partition=$PARTITION \
    --account=share-ie-idi \
    --time=01:00:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$NUM_CORES \
    --mem=128G \
    --gres=gpu:1 \
    --job-name="evaluate-inference-gpu" \
    --output=results/gpu_idun.out \
    --export=ALL,PARTITION=$PARTITION,NUM_CORES=$NUM_CORES \
    evaluate.slurm
fi

if [ "$PARTITION" = "CPUQ" ]; then
    echo "starting cpu gig"
    echo ""
    sbatch --partition=$PARTITION \
    --account=$ACCOUNT \
    --time=01:00:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$NUM_CORES \
    --mem=128G \
    --job-name="evaluate-inference-cpu" \
    --output=results/cpu_idun.out \
    --export=ALL,PARTITION=$PARTITION,NUM_CORES=$NUM_CORES \
    evaluate.slurm
fi
