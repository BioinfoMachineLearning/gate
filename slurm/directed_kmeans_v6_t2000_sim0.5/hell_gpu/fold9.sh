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
#SBATCH --job-name=directed_kmeans_v6_t2000_sim0.5_fold9
#SBATCH --output=directed_kmeans_v6_t2000_sim0.5_fold9-%j.out  # %j is the unique jobID

module load cuda/11.8.0

source /home/jl4mc/data/mambaforge/bin/activate

conda activate gate

export PYTHONPATH=/home/jl4mc/data/gate/

cd /home/jl4mc/data/gate/

python gate/network/edge_directed_kmeans_node/train_single_fold_seed_v3_mango.py --datadir dataset/CASP15_scores/sample/kmeans_v6/kmeans_sil_t2000/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v6/0.5/kmeans_sil_t2000/ --project directed_kmeans_v6_t2000_sim0.5 --dbdir experiments/ --labeldir dataset/CASP15_scores/label/ --log_val_mse true --fold 9