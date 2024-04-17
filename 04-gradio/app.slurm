#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:15:00         # 0 days and 15 minutes limit
#SBATCH --nodes=1                 # 1 compute nodes
#SBATCH --cpus-per-task=4         # 2 CPU cores
#SBATCH --mem=16G                 # 5 gigabytes memory
#SBATCH --output=output_log.out   # Log file

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total time at IDUN $IDUN_TIME"

echo ""
echo "----------------------------------------------"
echo ">> nvidia-smi"
nvidia-smi

echo ">> module load Python/3.10.8-GCCcore-12.2.0"
module load Python/3.10.8-GCCcore-12.2.0

echo ">> source /cluster/home/taheeraa/code/master-thesis/04-gradio/.venv/bin/activate"
source /cluster/home/taheeraa/code/master-thesis/04-gradio/.venv/bin/activate

echo ">> which python"
which python

echo ">> python train.py"
echo ""
python app.py \

echo ">> done"