#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition=chengji-lab-gpu
#SBATCH --account=chengji-lab
#SBATCH --ntasks-per-node=16  # cores per task
#SBATCH --mem=50G  # memory per core (default is 1GB/core)
#SBATCH --time 5-00:00     # days-hours:minutes time
#SBATCH --gres gpu:A100
#SBATCH --job-name=directed_kmeans_af_v7_t3000_sim0.2_fold0
#SBATCH --output=directed_kmeans_af_v7_t3000_sim0.2_fold0-%j.out  # %j is the unique jobID

module load cuda/11.8.0

source /home/jl4mc/data/mambaforge/bin/activate

conda activate gate

export PYTHONPATH=/home/jl4mc/data/gate/

cd /home/jl4mc/data/gate/

python gate/network/edge_directed_kmeans_node_pairwise_loss/train_single_fold_seed_v4_af_mango.py --datadir dataset/CASP15_inhouse_full/sample/kmeans_v6/kmeans_sil_t3000/ --scoredir dataset/CASP15_inhouse_full/ --outdir dataset/CASP15_inhouse_full/processed_dataset_directed_kmeans_v6/0.2/kmeans_sil_t3000/ --project directed_kmeans_af_v7_t3000_sim0.2 --dbdir experiments/ --labeldir dataset/CASP15_inhouse_full/label/ --log_val_mse true --fold 0