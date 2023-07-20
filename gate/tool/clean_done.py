import os, sys, argparse, time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)

    args = parser.parse_args()
    
    for infile in os.listdir(args.indir):
        if infile.find('.done') < 0:
            continue

        if os.path.exists(args.indir + '/' + infile.replace('.done', '')):
            continue
        
        os.system(f"rm {args.indir}/{infile}")