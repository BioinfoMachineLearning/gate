import os, sys, argparse, time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)

    args = parser.parse_args()

    for yamlfile in sorted(os.listdir(args.indir)):
        if yamlfile.find('.yaml') < 0:
            continue

        os.system(f"kubectl apply -f {args.indir}/{yamlfile}")

