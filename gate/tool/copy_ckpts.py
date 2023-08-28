import os, sys, argparse, time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    
    for infile in os.listdir(args.indir):
        if infile.find('fold') < 0:
            continue
        
        os.system(f"cp -r {args.indir}/{infile}/ckpt {args.outdir}/{infile}")