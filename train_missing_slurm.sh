#!/bin/bash
#SBATCH --job-name=transformers      # Specify job name
#SBATCH --partition=gpu        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --mem=0                # Use entire memory of node
#SBATCH --gres=gpu:1           # Use one GPU
#SBATCH --time=100:00:00        # Set a limit on the total run time
#SBATCH --mail-type=FAIL       # Notify user by email in case of job failure
#SBATCH --account=sc-users     # Charge resources on this project account
#SBATCH --output=/home/rohrerc/jobs/transformers.o%A_%a    # File name for standard output
#SBATCH --error=/home/rohrerc/jobs/transformers.e%A_%a     # File name for standard error output
#SBATCH --array=1-14%14
 
n=$SLURM_ARRAY_TASK_ID
model_config=`sed -n "${n} p" ~/projects/pytorch-image-models/train_missing_commands.txt`
 
source /opt/miniforge/etc/profile.d/conda.sh
export PATH=/opt/miniforge/bin:$PATH
conda activate hpc
 
$model_config
