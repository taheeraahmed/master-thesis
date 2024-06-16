#!/bin/bash
#SBATCH --job-name=inference_test_cpu
#SBATCH --output=out_inference.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=CPUQ

# Load necessary modules (e.g., Python)
module load Python/3.10.8-GCCcore-12.2.0


echo ">> source /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/.venv/bin/activate"
source /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/.venv/bin/activate
echo ""

echo "running python script"

# Run the Python script
python /cluster/home/taheeraa/code/master-thesis/01-multi-label/evaluation/test-cpu-inference.py
