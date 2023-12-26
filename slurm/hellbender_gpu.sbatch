#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition gpu
#SBATCH --ntasks-per-node=16  # cores per task
#SBATCH --mem=50G  # memory per core (default is 1GB/core)
#SBATCH --time 2-00:00     # days-hours:minutes time 
#SBATCH --gres gpu:A100
#SBATCH --job-name=JOBNAME
#SBATCH --output=JOBNAME-%j.out  # %j is the unique jobID
#SBATCH --mail-type=all
#SBATCH --mail-user=jl4mc@umsystem.edu

module load cuda/11.8.0

source /home/jl4mc/data/mambaforge/bin/activate

conda activate gate

export PYTHONPATH=/home/jl4mc/data/gate/

cd /home/jl4mc/data/gate/

