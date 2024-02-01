#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition=chengji-lab-gpu
#SBATCH --account=chengji-lab
#SBATCH --ntasks-per-node=4  # cores per task
#SBATCH --mem=60G  # memory per core (default is 1GB/core)
#SBATCH --time 3-00:00     # days-hours:minutes time
#SBATCH --gres gpu:A100
#SBATCH --job-name=directed_kmeans_v8_t2000_sim0.5_fold0
#SBATCH --output=directed_kmeans_v8_t2000_sim0.5_fold0-%j.out  # %j is the unique jobID
#SBATCH --mail-type=all
#SBATCH --mail-user=jl4mc@umsystem.edu

module load cuda/11.8.0

source /home/jl4mc/data/mambaforge/bin/activate

conda activate gate

export PYTHONPATH=/home/jl4mc/data/gate/

cd /home/jl4mc/data/gate/

python gate/network/edge_directed_kmeans_node_pairwise_loss/train_single_fold_seed_v4_mango.py --datadir dataset/CASP15_scores/sample/kmeans_v7/kmeans_sil_t2000/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v7/0.5/kmeans_sil_t2000/ --project directed_kmeans_v8_t2000_sim0.5 --dbdir experiments/ --labeldir dataset/CASP15_scores/label/ --log_val_mse true --fold 0