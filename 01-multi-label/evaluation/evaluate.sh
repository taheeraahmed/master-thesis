#!/bin/bash
PARTITION="GPUQ"
ACCOUNT=share-ie-idi
NUM_CORES=8

if [ "$PARTITION" = "GPUQ" ]; then
    echo "starting gpu gig"
    echo ""
    sbatch --partition=$PARTITION \
    --account=$ACCOUNT \
    --time=00:60:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$NUM_CORES \
    --mem=128G \
    --gres=gpu:1 \
    --job-name="calculating-iou-all-with-label" \
    --output=logs/calculating-iou-all.out \
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
    --output=logs/cpu_idun.out \
    --export=ALL,PARTITION=$PARTITION,NUM_CORES=$NUM_CORES \
    evaluate.slurm
fi
