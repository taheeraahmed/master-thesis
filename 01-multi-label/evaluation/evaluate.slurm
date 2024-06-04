#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G
#SBATCH --job-name="running-eval-inference"
#SBATCH --output="idun_out/output_cpu.out"

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

echo ">> module load Anaconda3/2022.10"
module load Anaconda3/2022.10    

echo ">> which python"
which python

echo ">> installing requirements"
pip install -q -r /cluster/home/taheeraa/code/master-thesis/01-multi-label/requirements.txt

echo ">> python run_inference.py"
echo ""
python /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/main.py \
    --num_workers $NUM_CORES \
    --batch_size 32 \
    --test_augment False \
    --xai False \

echo ">> done"