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
#SBATCH --job-name=directed_kmeans_af_v7_t3000_sim0.5_fold1
#SBATCH --output=directed_kmeans_af_v7_t3000_sim0.5_fold1-%j.out  # %j is the unique jobID

module load cuda/11.1.0
module load cudnn/8.0.4.30-11.0-linux-x64

source /home/jl4mc/data/anaconda3/bin/activate
conda activate gate
export PYTHONPATH=/home/jl4mc/data/gate

cd /home/jl4mc/data/gate

python gate/network/edge_directed_kmeans_node_pairwise_loss/train_single_fold_seed_v4_af_mango.py --datadir dataset/CASP15_inhouse_full/sample/kmeans_v6/kmeans_sil_t3000/ --scoredir dataset/CASP15_inhouse_full/ --outdir dataset/CASP15_inhouse_full/processed_dataset_directed_kmeans_v6/0.5/kmeans_sil_t3000/ --project directed_kmeans_af_v7_t3000_sim0.5 --dbdir experiments/ --labeldir dataset/CASP15_inhouse_full/label/ --log_val_mse true --fold 1