#!/bin/bash -l
#SBATCH --partition=GPU
#SBATCH --gpus=1
#SBATCH --array=1-10
#SBATCH --output=
#SBATCH --time=
#SBATCH --ntasks-per-node=
#SBATCH --mem=
#SBATCH --job-name=
#SBATCH --mail-user=
#SBATCH --mail-type=

export PYTHONUNBUFFERED=TRUE
bash ./homo/trainingJobs/train$SLURM_ARRAY_TASK_ID.sh > ./homo/logs/trainModel$SLURM_ARRAY_TASK_ID.log