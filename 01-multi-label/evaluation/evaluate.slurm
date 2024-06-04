#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128G
#SBATCH --job-name="running-eval-inference"
#SBATCH --gres=gpu:1

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

echo ">> module load Python/3.10.8-GCCcore-12.2.0"
module load Python/3.10.8-GCCcore-12.2.0

echo ">> source /cluster/home/taheeraa/code/master-thesis/01-multi-label/.venv/bin/activate"
source /cluster/home/taheeraa/code/master-thesis/01-multi-label/.venv/bin/activate

echo ">> which python"
which python

echo ">> python run_inference.py"
echo ""
python /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/main.py \

echo ">> done"