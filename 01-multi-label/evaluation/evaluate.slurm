#!/bin/sh
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Using $NUM_CORES"
echo "Total of $SLURM_NTASKS cores"
echo "Total time at IDUN $IDUN_TIME"

echo ""
echo "----------------------------------------------"
echo ">> nvidia-smi"
nvidia-smi
echo ""

echo ">> module load Python/3.10.8-GCCcore-12.2.0"
module load Python/3.10.8-GCCcore-12.2.0
echo ""

echo ">> source /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/.venv/bin/activate"
source /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/.venv/bin/activate
echo ""

echo ">> pip freeze"
pip freeze
echo ""

echo ">> which python"
which python
echo ""

echo ">> python run_inference.py"
echo ""
python /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/main.py \
    --partition $PARTITION \
    --num_workers $NUM_CORES \
    --compare_xai_bbox \

echo ">> done"