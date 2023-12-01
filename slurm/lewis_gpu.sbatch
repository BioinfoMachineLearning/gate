#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition gpu4
#SBATCH --cpus-per-task=8  # cores per task
#SBATCH --mem=60G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes time 
#SBATCH --account=engineering-gpu  # investors in gpu4 will replace this (e.g. engineering-gpu)
#SBATCH --gres gpu:1
#SBATCH --job-name=JOBNAME
#SBATCH --output=JOBNAME-%j.out  # %j is the unique jobID

module load cuda/11.1.0
module load cudnn/8.0.4.30-11.0-linux-x64

source /home/jl4mc/data/anaconda3/bin/activate
conda activate gate
export PYTHONPATH=/home/jl4mc/data/gate

cd /home/jl4mc/data/gate

