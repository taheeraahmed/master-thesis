#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --job-name="look-at-data"
#SBATCH --output=jupyter.out
#SBATCH --gres=gpu:1
module load Anaconda3/2022.05
jupyter notebook --no-browser