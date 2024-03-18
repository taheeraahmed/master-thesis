#!/bin/bash
TEST_MODE=true

MODELS=("swin")
LOSSES=("wfl" "ce" "wce")
TASK=multi-label

BATCH_SIZE=28
LEARNING_RATE=0.001
NUM_EPOCHS=25

IDUN_TIME=45:00:00

#    ======= DO NOT EDIT THIS SCRIPT =======

DATE=$(date "+%Y-%m-%d-%H:%M:%S")
USER=$(whoami)
CURRENT_PATH=$(pwd)
ROOT_OUTPUT_FOLDER="/cluster/home/$USER/code/master-thesis/01-multi-label/output"


for MODEL in "${MODELS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
        PARTITION="GPUQ"
        EXPERIMENT_NAME=${DATE}-${MODEL}-$LOSS-$TASK-pl
        echo "Current EXPERIMENT_NAME is: $EXPERIMENT_NAME"

        if [ "$TEST_MODE" = true ]; then
            EXPERIMENT_NAME="TEST-${EXPERIMENT_NAME}"
            IDUN_TIME=00:09:00
            BATCH_SIZE=28
            LEARNING_RATE=0.001
            NUM_EPOCHS=3
            PARTITION="short"
        fi
        if [ "$TEST_MODE" = false ]; then
            EXPERIMENT_NAME="${EXPERIMENT_NAME}-e$NUM_EPOCHS-bs$BATCH_SIZE-lr$LEARNING_RATE-t$IDUN_TIME"
        fi

        mkdir -p $ROOT_OUTPUT_FOLDER/$EXPERIMENT_NAME/model_checkpoints # Stores logs and checkpoints
        mkdir -p $ROOT_OUTPUT_FOLDER/$EXPERIMENT_NAME/images            # Store images

        echo "Made directory: $ROOT_OUTPUT_FOLDER/$EXPERIMENT_NAME"
        OUTPUT_FILE="$ROOT_OUTPUT_FOLDER/$EXPERIMENT_NAME/idun_out.out"


        CODE_PATH="/cluster/home/$USER/runs/code/${EXPERIMENT_NAME}"

        echo "Copying code to $CODE_PATH"
        mkdir -p $CODE_PATH
        rsync -av \
            --exclude='.venv' \
            --exclude='idun' \
            --exclude='images' \
            --exclude='runs' \
            --exclude='notebooks' \
            --exclude='output' \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='utils/__pycache__' \
            --exclude='trainers/__pycache__' \
            --exclude='mlruns/' \
            /cluster/home/$USER/code/master-thesis/01-multi-label/ $CODE_PATH

        echo "Current user is: $USER"
        echo "Current path is: $CURRENT_PATH"
        echo "Current job name is: $EXPERIMENT_NAME"
        echo "Running slurm job from $CODE_PATH"

        sbatch --partition=$PARTITION \
            --account=ie-idi \
            --time=$IDUN_TIME \
            --nodes=1 \
            --ntasks-per-node=1 \
            --mem=50G \
            --gres=gpu:1 \
            --job-name=$EXPERIMENT_NAME \
            --output=$OUTPUT_FILE \
            --export=TEST_MODE=$TEST_MODE,EXPERIMENT_NAME=$EXPERIMENT_NAME,CODE_PATH=$CODE_PATH,IDUN_TIME=$IDUN_TIME,MODEL=$MODEL,BATCH_SIZE=$BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE,NUM_EPOCHS=$NUM_EPOCHS,LOSS=$LOSS \
            $CODE_PATH/train.slurm
    done
done