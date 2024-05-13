#!/bin/bash

FAST_DEV_RUN_ENABLED=false 
TEST_TIME_AUGMENTATION=false
# Resnet FL, DenseNet MLSM, AlexNet MLSM
MODELS=("swin")
LOSSES=("bce")

SCHEDULER=cosineannealinglr
OPTIMIZER=adam
BATCH_SIZE=64
LEARNING_RATE=0.0005
NUM_EPOCHS=35

ADD_TRANSFORMS=true
DESCRIPTION=new-dataset-fine-tuning-benchmarking-transformers-cxr-normalization-tta
CKPT_PATH=/cluster/home/taheeraa/code/master-thesis/01-multi-label/checkpoints/simmim_swinb_ImageNet_Xray926k.pth

TASK=14-multi-label
ACCOUNT=share-ie-idi
NUM_CORES=8
IDUN_TIME=10:00:00
PARTITION="GPUQ"
EVAL_MODE=false

echo "Starting training :)"

#    ======= DO NOT EDIT THIS SCRIPT =======

DATE=$(date "+%Y-%m-%d-%H:%M:%S")
USER=$(whoami)
CURRENT_PATH=$(pwd)
ROOT_OUTPUT_FOLDER="/cluster/home/$USER/code/master-thesis/01-multi-label/output"

for MODEL in "${MODELS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do        
        EXPERIMENT_NAME=${DATE}-${MODEL}-$LOSS-$TASK
        echo "Current EXPERIMENT_NAME is: $EXPERIMENT_NAME"

        if [ "$FAST_DEV_RUN_ENABLED" = true ]; then
            EXPERIMENT_NAME="FAST_DEV_RUN-${EXPERIMENT_NAME}"
            IDUN_TIME=00:05:00
            PARTITION="short"
            NUM_CORES=4
            NUM_EPOCHS=1
        fi

        EXPERIMENT_NAME="${EXPERIMENT_NAME}-e$NUM_EPOCHS-bs$BATCH_SIZE-lr$LEARNING_RATE-$DESCRIPTION"

        if [ "$EVAL_MODE" = true ]; then
            IDUN_TIME=00:15:00
            EXPERIMENT_NAME="${EXPERIMENT_NAME}-evaluation"
        fi
        if [ "$ADD_TRANSFORMS" = true ]; then
            EXPERIMENT_NAME="${EXPERIMENT_NAME}-aug"
        fi
        if [ "$TEST_TIME_AUGMENTATION" = true ]; then
            EXPERIMENT_NAME="${EXPERIMENT_NAME}-tta"
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
            --exclude='notebook'\
            --exclude='mlruns/' \
            --exclude='checkpoints/'\
            /cluster/home/$USER/code/master-thesis/01-multi-label/ $CODE_PATH

        echo "Current user is: $USER"
        echo "Current path is: $CURRENT_PATH"
        echo "Current job name is: $EXPERIMENT_NAME"
        echo "Running slurm job from $CODE_PATH"

        sbatch --partition=$PARTITION \
            --account=$ACCOUNT \
            --time=$IDUN_TIME \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=$NUM_CORES \
            --mem=50G \
            --gres=gpu:1 \
            --job-name=$EXPERIMENT_NAME \
            --output=$OUTPUT_FILE \
            --export=EVAL_MODE=$EVAL_MODE,EXPERIMENT_NAME=$EXPERIMENT_NAME,CODE_PATH=$CODE_PATH,IDUN_TIME=$IDUN_TIME,MODEL=$MODEL,BATCH_SIZE=$BATCH_SIZE,LEARNING_RATE=$LEARNING_RATE,NUM_EPOCHS=$NUM_EPOCHS,LOSS=$LOSS,ADD_TRANSFORMS=$ADD_TRANSFORMS,OPTIMIZER=$OPTIMIZER,SCHEDULER=$SCHEDULER,NUM_CORES=$NUM_CORES,TEST_TIME_AUGMENTATION=$TEST_TIME_AUGMENTATION,FAST_DEV_RUN_ENABLED=$FAST_DEV_RUN_ENABLED,CKPT_PATH=$CKPT_PATH \
            $CODE_PATH/train.slurm
    done
done