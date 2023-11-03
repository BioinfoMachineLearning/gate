import os, sys, argparse, time
import difflib
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir1', type=str, required=True)
    parser.add_argument('--indir2', type=str, required=True)
    args = parser.parse_args()
    
    for infolder1 in os.listdir(args.indir1):
        print("checking " + infolder1)
        for infile1 in os.listdir(args.indir1 + '/' + infolder1):
            inpath1 = args.indir1 + '/' + infolder1 + '/' + infile1
            inpath2 = args.indir2 + '/' + infolder1 + '/' + infile1
            cmd = f"diff {inpath1} {inpath2}"
            contents = os.popen(cmd).read().split('\n')
            if len(contents[0]) > 0:
                print(inpath1 + ',' + inpath2)

