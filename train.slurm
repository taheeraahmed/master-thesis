#!/bin/sh
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total time at IDUN $IDUN_TIME"

echo "nvidia-smi"
nvidia-smi
module load Python/3.10.8-GCCcore-12.2.0

echo "installing requirements"
pip install -q -r requirements.txt

echo "running python-scrip"
python $CODE_PATH/src/train.py \
  --test_mode $TEST_MODE \
  --output_folder $OUTPUT_FOLDER \
  --idun_time $IDUN_TIME \
  --model $MODEL \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_epochs $NUM_EPOCHS \
  --class_imbalance $CLASS_IMBALANCE \

echo "done running train.py"