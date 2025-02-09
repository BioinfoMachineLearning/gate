import os, argparse
from multiprocessing import Pool

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--ckptdir', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    for line in open(args.infile):
        line = line.rstrip('\n')
        foldname, ckptname, ensemble_mode = line.split()

        print(foldname)

        configs = ckptname.split('_')

        ckpt_dir = os.path.join(args.ckptdir, foldname, 'ckpt', ckptname)

        if len([ckptfile for ckptfile in os.listdir(ckpt_dir) if ckptfile.find('.ckpt') >= 0]) > 1:
            raise Exception(f"multiple check points in {ckpt_dir}")

        for ckptfile in os.listdir(ckpt_dir):
            if ckptfile.find('.ckpt') < 0:
                continue
            ckptname = ckptfile
            break

        os.system(f"cp {ckpt_dir}/{ckptfile} {args.outdir}/{args.prefix}_{foldname}.ckpt")

        config_names = ["node_input_dim", "edge_input_dim", "num_heads", "num_layer", "dp_rate", "layer_norm", "batch_norm", "residual", "hidden_dim", "mlp_dp_rate", "loss_fun", "pairwise_loss_weight", "opt", "lr", "weight_decay", "batch_size"]
        
        for config_name, config_value in zip(config_names, configs):
            print(f"'{config_name}': {config_value},")

        




