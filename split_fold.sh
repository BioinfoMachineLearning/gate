#python gate/network/edge_directed_kmeans_node/split_fold.py --datadir dataset/CASP15_scores/sample/kmeans_v3_c2/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v3_c2_sim/0.7 --sim_threshold 0.7

#python gate/network/edge_directed_kmeans_node/split_fold.py --datadir dataset/CASP15_scores/sample/kmeans_v3_c2/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v3_c2_sim/0.5 --sim_threshold 0.5

#python gate/network/edge_directed_kmeans_node/split_fold.py --datadir dataset/CASP15_scores/sample/kmeans_v3_c2/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v3_c2_sim/0.3 --sim_threshold 0.3

#python gate/network/edge_directed_kmeans_node/split_fold.py --datadir dataset/CASP15_scores/sample/kmeans_v2_c3/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v2_c3_sim/0 --sim_threshold 0

#python gate/network/edge_directed_kmeans_node/split_fold.py --datadir dataset/CASP15_scores/sample/kmeans_v2_c2/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v2_c2_sim/0 --sim_threshold 0

#python gate/network/edge_directed_kmeans_node/split_fold.py --datadir dataset/CASP15_scores/sample/kmeans_v3_c2/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v3_c2_sim/0 --sim_threshold 0

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/0 --sim_threshold 0

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/0.3 --sim_threshold 0.2

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/0.3_true --sim_threshold 0.3

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/0.5 --sim_threshold 0.5

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/0.7 --sim_threshold 0.7

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/0.8 --sim_threshold 0.8

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/auto_global --auto_sim_threshold true --auto_sim_threshold_global true

python gate/network/edge_directed_kmeans_node/split_fold_usalign.py --datadir dataset/CASP15_scores/sample/kmeans_v5/ --scoredir dataset/CASP15_scores/ --outdir dataset/CASP15_scores/processed_dataset_directed_kmeans_v5/auto_local --auto_sim_threshold true --auto_sim_threshold_global false