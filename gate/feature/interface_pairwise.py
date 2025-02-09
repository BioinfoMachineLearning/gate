import argparse
import os
import json
import time
import sys
from multiprocessing import Pool
from ost import io
from ost.mol.alg import scoring
import pandas as pd

def _parse_args():
    desc = ("Computes QS-scores based on chain mappings from MMalign as "
            "computed by the predictioncenter. You need to run the following "
            "data collection scripts as described in README: "
            "collect_assemblies.py, collect_targets.py and "
            "collect_mmalign_results.py. Many of the scores will be None."
            "Reason for that are 1) models with empty chain names, "
            "2) Models with chains where the residue numbers are not strictly "
            "increasing (requirenent of ChainMapper), 3) MMalign maps chains "
            "with non-equal sequences. 3 is what happens most often.") 
    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--procnum", default=150, required=False)
    return parser.parse_args() 

def _crash_out(error, out_path):
    """ Produces output JSON containing error and exits with return code 1
    """
    with open(out_path, 'w') as fh:
        json.dump({"errors": [error]}, fh)
    sys.exit(1)


def _local_scores_to_json_dict(score_dict):
    """ Convert ResNums to str for JSON serialization
    """
    json_dict = dict()
    for ch, ch_scores in score_dict.items():
        for num, s in ch_scores.items():
            ins_code = num.ins_code.strip("\u0000")
            json_dict[f"{ch}.{num.num}.{ins_code}"] = s
    return json_dict

def _patch_scores_to_json_list(interface_dict, score_dict):
    """ Creates List of patch scores that are consistent with interface residue
    lists
    """
    json_list = list()
    for ch, ch_nums in interface_dict.items():
        json_list += score_dict[ch]
    return json_list

def _interface_residues_to_json_list(interface_dict):
    """ Convert ResNums to str for JSON serialization.

    Changes in this function will affect _PatchScoresToJSONList
    """
    json_list = list()
    for ch, ch_nums in interface_dict.items():
        for num in ch_nums:
            ins_code = num.ins_code.strip("\u0000")
            json_list.append(f"{ch}.{num.num}.{ins_code}")
    return json_list

def _cal_interface_score(inparams):

    t0 = time.time()
    modeldir, pdb1, pdb2, outdir = inparams
    out_file = os.path.join(outdir, f"{pdb1}_{pdb2}.json")

    if os.path.exists(out_file):
        return out_file
        
    try:
        mdl = io.LoadPDB(os.path.join(modeldir, pdb1))
        trg = io.LoadPDB(os.path.join(modeldir, pdb2))
    except Exception as e:
        if str(e).startswith("duplicate atom 'OXT' in residue "):
            _crash_out(str(e), out_file)
        else:
            raise

    try:
        scorer = scorer = scoring.Scorer(mdl, trg, resnum_alignments=True)
    except Exception as e:       
        _crash_out(str(e), out_file)

    data = dict()
    try:
        data["trg_file"] = pdb2
        data["mdl_file"] = pdb1
        data["qs_global"] = round(scorer.qs_global, 3)
        data["qs_best"] = round(scorer.qs_best, 3)
        data["gdtts"] = round(scorer.gdtts, 3)
        data["gdtts_transform"] = scorer.transform.data
        data["rmsd"] = round(scorer.rmsd, 3)
        data["dockq_ave"] = scorer.dockq_ave
        data["dockq_wave"] = scorer.dockq_wave
        data["dockq_ave_full"] = scorer.dockq_ave_full
        data["dockq_wave_full"] = scorer.dockq_wave_full
        data["lddt"] = round(scorer.lddt, 3)
        # data["local_lddt"] = _local_scores_to_json_dict(scorer.local_lddt)
        data["cad_score"] = scorer.cad_score
        # data["local_cad_score"] = _local_scores_to_json_dict(scorer.local_cad_score)
        # N = len(scorer.model.atoms) - len(scorer.stereochecked_model.atoms)
        # data["stereocheck_atoms_removed_mdl"] = N
        # N = len(scorer.target.atoms) - len(scorer.stereochecked_target.atoms)
        # data["stereocheck_atoms_removed_trg"] = N
        # data["interface_residues"] = _interface_residues_to_json_list(scorer.model_interface_residues)
        # data["target_interface_residues"] = _interface_residues_to_json_list(scorer.target_interface_residues)
        # data["patch_qs"] = _patch_scores_to_json_list(scorer.model_interface_residues, scorer.patch_qs)
        # data["patch_dockq"] = _patch_scores_to_json_list(scorer.model_interface_residues, scorer.patch_dockq)
        data["mapping"] = scorer.mapping.GetFlatMapping(mdl_as_key=True)
        data["runtime"] = time.time() - t0
    except Exception as e:       
        _crash_out(str(e), out_file)

    with open(out_file, 'w') as fh:
        json.dump(data, fh)

    return out_file

def main():

    args = _parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    scoredir = os.path.join(args.outdir, 'scores')
    os.makedirs(scoredir, exist_ok=True)

    pdbs = sorted(os.listdir(args.indir))
    process_list = []
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            outfile = os.path.join(scoredir, f"{pdb1}_{pdb2}.json")
            if os.path.exists(outfile):
                continue
            process_list.append([args.indir, pdb1, pdb2, scoredir])

    pool = Pool(processes=int(args.procnum))
    results = pool.map(_cal_interface_score, process_list)
    pool.close()
    pool.join()

    scores_dict = {'dockq_wave': {}, 'dockq_ave': {}, 'cad_score': {}}
    for i in range(len(pdbs)):
        for j in range(len(pdbs)):
            pdb1 = pdbs[i]
            pdb2 = pdbs[j]
            if pdb1 == pdb2:
                continue
            jsonfile = os.path.join(scoredir, f"{pdb1}_{pdb2}.json")
            if not os.path.exists(jsonfile):
                raise Exception(f"cannot find {jsonfile}")
            
            try:
                with open(jsonfile) as f:
                    # print(jsonfile)
                    data = json.loads(f.read())
                    scores_dict['dockq_wave'][f"{pdb1}_{pdb2}"] = data["dockq_wave_full"]
                    scores_dict['dockq_ave'][f"{pdb1}_{pdb2}"] = data["dockq_ave_full"]
                    scores_dict['cad_score'][f"{pdb1}_{pdb2}"] = data["cad_score"]
            except Exception as e:
                print(jsonfile)
                print(e)

    dockq_wave_dict, dockq_ave_dict, cad_score_dict = {}, {}, {}
    for i in range(len(pdbs)):
        pdb1 = pdbs[i]
        dockq_waves, dockq_aves, cad_scores = [], [], []
        for j in range(len(pdbs)):
            pdb2 = pdbs[j]
            dockq_wave, dockq_ave, cad_score = 1, 1, 1
            if pdb1 != pdb2:
                if f"{pdb1}_{pdb2}" not in scores_dict['dockq_wave']:
                    print(f"Cannot find {pdb1}_{pdb2}!")

                if f"{pdb1}_{pdb2}" not in scores_dict['dockq_ave']:
                    print(f"Cannot find {pdb1}_{pdb2}!")

                if f"{pdb1}_{pdb2}" not in scores_dict['cad_score']:
                    print(f"Cannot find {pdb1}_{pdb2}!")

                dockq_wave = scores_dict['dockq_wave'][f"{pdb1}_{pdb2}"]
                dockq_ave = scores_dict['dockq_ave'][f"{pdb1}_{pdb2}"]
                cad_score = scores_dict['cad_score'][f"{pdb1}_{pdb2}"]

            dockq_waves += [dockq_wave]
            dockq_aves += [dockq_ave]
            cad_scores += [cad_score]

        dockq_wave_dict[pdb1] = dockq_waves
        dockq_ave_dict[pdb1] = dockq_aves
        cad_score_dict[pdb1] = cad_scores

    pd.DataFrame(dockq_wave_dict).to_csv(os.path.join(args.outdir, 'dockq_wave.csv'))
    pd.DataFrame(dockq_ave_dict).to_csv(os.path.join(args.outdir, 'dockq_ave.csv'))
    pd.DataFrame(cad_score_dict).to_csv(os.path.join(args.outdir, 'cad_score.csv'))


if __name__ == '__main__':
    main()

