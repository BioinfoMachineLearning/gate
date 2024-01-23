
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
          f"-m /home/{inpdb} -r /home/{refpdb}  --output scores.json --dockq"

    print(cmd)
    os.system(cmd)
    os.system(f"mv {workdir}/scores.json {outfile}")
    
def run_pairwise(indir, scoredir, outfile1, outfile2):

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

    dockq_wave_scores_dict = {}
    dockq_ave_scores_dict = {}
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
            dockq_wave = data_json['dockq_wave']
            dockq_ave = data_json['dockq_ave']
            dockq_wave_scores_dict[f"{pdb1}_{pdb2}"] = dockq_wave
            dockq_ave_scores_dict[f"{pdb1}_{pdb2}"] = dockq_ave

    data_dict_wave, data_dict_ave = {}, {}
    for i in range(len(pdbs)):
        pdb1 = pdbs[i]
        dockq_waves, dockq_aves = [], []
        for j in range(len(pdbs)):
            pdb2 = pdbs[j]
            dockq_wave, dockq_ave = 1, 1
            if pdb1 != pdb2:
                dockq_wave = dockq_wave_scores_dict[f"{pdb1}_{pdb2}"]
                dockq_ave = dockq_ave_scores_dict[f"{pdb1}_{pdb2}"]
            dockq_waves += [dockq_wave]
            dockq_aves += [dockq_aves]

        data_dict_wave[pdb1] = dockq_waves
        data_dict_ave[pdb1] = dockq_aves

    pd.DataFrame(data_dict_wave).to_csv(outfile1)
    pd.DataFrame(data_dict_ave).to_csv(outfile2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    for target in os.listdir(args.indir):

        scoredir = args.outdir + '/' + target
        os.makedirs(scoredir, exist_ok=True)

        outfile1 = args.outdir + '/' + target + '_wave.csv'
        outfile2 = args.outdir + '/' + target + '_ave.csv'

        run_pairwise(args.indir + '/' + target, scoredir, outfile1, outfile2)
