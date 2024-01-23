
import os, sys, argparse, time
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import json

def run_command(inparams):
    workdir, inpdb, refpdb, outfile = inparams
    cmd = "docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) " \
          f"-v {workdir}:/home " \
          f"registry.scicore.unibas.ch/schwede/openstructure compare-structures " \
          f"-m /home/{inpdb} -r /home/{refpdb}  --output scores.json --cad-score --residue-number-alignment"

    print(cmd)
    os.system(cmd)
    os.system(f"mv {workdir}/scores.json {outfile}")
    
def run_pairwise(indir, scoredir, outfile):

    pdbs = sorted(os.listdir(indir))

    process_list = []
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            resultfile = f"{scoredir}/{pdb1}_{pdb2}.json"
            if os.path.exists(resultfile) and len(open(resultfile).readlines()) > 15:
                continue
            workdir = f"{scoredir}/{pdb1}_{pdb2}"
            os.makedirs(workdir, exist_ok=True)
            os.system(f"cp {indir}/{pdb1} {workdir}/{pdb1}.pdb")
            os.system(f"cp {indir}/{pdb2} {workdir}/{pdb2}.pdb")
            process_list.append([workdir, pdb1 + '.pdb', pdb2 + '.pdb', resultfile])

    pool = Pool(processes=120)
    results = pool.map(run_command, process_list)
    pool.close()
    pool.join()

    scores_dict = {}
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            json_file = f"{scoredir}/{pdb1}_{pdb2}.json"
            if not os.path.exists(json_file):
                raise Exception(f"cannot find {json_file}")
            data_json = json.load(open(json_file))
            cadscore = data_json['cad_score']
            scores_dict[f"{pdb1}_{pdb2}"] = cadscore

    data_dict = {}
    for i in range(len(pdbs)):
        pdb1 = pdbs[i]
        cadscores = []
        for j in range(len(pdbs)):
            pdb2 = pdbs[j]
            cadscore = 1
            if pdb1 != pdb2:
                cadscore = scores_dict[f"{pdb1}_{pdb2}"]
            cadscores += [cadscore]

        data_dict[pdb1] = cadscores

    pd.DataFrame(data_dict).to_csv(outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):

        scoredir = args.outdir + '/' + target
        os.makedirs(scoredir, exist_ok=True)

        outfile = args.outdir + '/' + target + '.csv'
        run_pairwise(args.indir + '/' + target, scoredir, outfile)
