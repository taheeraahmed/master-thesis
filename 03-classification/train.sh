#!/bin/bash
TEST_MODE=true

MODELS=("densenet" "swin")
LOSS=wfl # wfl or wbce

BATCH_SIZE=32
LEARNING_RATE=0.001
NUM_EPOCHS=25

IDUN_TIME=45:00:00

#    ======= DO NOT EDIT THIS SCRIPT =======

DATE=$(date "+%Y-%m-%d-%H:%M:%S")
USER=$(whoami)
CURRENT_PATH=$(pwd)

for MODEL in "${MODELS[@]}"; do
    PARTITION="GPUQ"
    JOB_NAME=${DATE}-${MODEL}-$LOSS-classification

    if [ "$TEST_MODE" = true ]; then
        JOB_NAME="TEST-${JOB_NAME}"
        IDUN_TIME=00:09:00
        BATCH_SIZE=2
        LEARNING_RATE=0.001
        NUM_EPOCHS=2
        PARTITION="short"
    fi
    if [ "$TEST_MODE" = false ]; then
        JOB_NAME="${JOB_NAME}-e$NUM_EPOCHS-bs$BATCH_SIZE-lr$LEARNING_RATE-t$IDUN_TIME"
    fi

    OUTPUT_FOLDER=${JOB_NAME}

    mkdir -p /cluster/home/$USER/code/master-thesis/01-classification/output/$OUTPUT_FOLDER/model_checkpoints # Stores logs and checkpoints
    mkdir -p /cluster/home/$USER/code/master-thesis/01-classification/output/$OUTPUT_FOLDER/images            # Store images
    mkdir -p /cluster/home/$USER/code/master-thesis/01-classification/runs/$OUTPUT_FOLDER                     # Stores tensorboard info

    echo "Made directory: /cluster/home/$USER/code/master-thesis/01-classification/output/$OUTPUT_FOLDER"
    OUTPUT_FILE="/cluster/home/$USER/code/master-thesis/01-classification/output/$OUTPUT_FOLDER/idun_out.out"
    echo "Current OUTPUT_FOLDER is: $OUTPUT_FOLDER"

    # Define the destination path for the code
    CODE_PATH="/cluster/home/$USER/runs/code/${OUTPUT_FOLDER}"

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
        /cluster/home/$USER/code/master-thesis/01-classification/ $CODE_PATH

    echo "Current user is: $USER"
    echo "Current path is: $CURRENT_PATH"
    echo "Current job name is: $JOB_NAME"
    echo "Running slurm job from $CODE_PATH"

    sbatch --partition=$PARTITION \
        --account=ie-idi \
        --time=$IDUN_TIME \
        --nodes=1 \
        --ntasks-per-node=1 \
        --mem=50G \
        --gres=gpu:1 \
        --job-name=$JOB_NAME \
        --output=$OUTPUT_FILE \
        --export=TEST_MODE=$TEST_MODE,OUTPUT_FOLDER=$OUTPUT_FOLDER,CODE_PATH=$CODE_PATH,IDUN_TIME=$IDUN_TIME,MODEL=$MODEL,BATCH_SIZE=$BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE,NUM_EPOCHS=$NUM_EPOCHS,LOSS=$LOSS \
        $CODE_PATH/train.slurm
done