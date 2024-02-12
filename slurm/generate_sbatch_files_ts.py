import os, sys, argparse, time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--threshold', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    for fold in range(10):

        cmd = f"python gate/network/edge_directed_kmeans_node_pairwise_loss/train_single_fold_seed_ts_mango.py "\
            f"--datadir dataset/CASP15_TS/sample/kmeans_v6/kmeans_sil_t2000/ " \
            f"--scoredir dataset/CASP15_TS/ " \
            f"--outdir dataset/CASP15_TS/processed_dataset_directed_kmeans_v1/{args.threshold}/kmeans_sil_t2000/ " \
            f"--project directed_kmeans_ts_v1_t2000_sim{args.threshold} " \
            f"--dbdir experiments/ " \
            f"--labeldir dataset/CASP15_TS/label/ " \
            f"--log_val_mse true " \
            f"--fold {fold}"

        jobname = f"directed_kmeans_ts_v1_t2000_sim{args.threshold}_fold{fold}"
        with open(f"{args.outdir}/fold{fold}.sh", 'w') as fw:
            for line in open(args.template):
                line = line.replace("JOBNAME", jobname)
                fw.write(line)
            fw.write(cmd)

        

