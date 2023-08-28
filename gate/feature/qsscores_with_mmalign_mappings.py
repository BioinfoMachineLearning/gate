import argparse
import os
import subprocess
import json
import traceback
from multiprocessing import Pool
from ost import io
from ost import conop
from ost.mol.alg.chain_mapping import ChainMapper
from ost.mol.alg.qsscore import QSScorer
from ost.io import ReadStereoChemicalPropsFile
from ost.mol.alg import CheckStructure, Molck, MolckSettings
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
    parser.add_argument("--indir")
    parser.add_argument("--mmalign_dir")
    parser.add_argument("--outdir")
    parser.add_argument("--procnum")
    return parser.parse_args() 

def read_qsscore(infile):
    with open(infile) as f:
        data = json.load(f)
        if data['QS_best'] is None:
            return 0.0
        return float(data['QS_best'])


class MMalignResult:
    """ helper object to hold MMalign results for a single model

    :param trg_filename: Name of PDB file with target coordinates
    :type trg_filename: :class:`str`
    :param mdl_filename: Name of PDB file with model coordinates
    :type mdl_filename: :class:`str`
    :param tmscore: TM-score as computed by MMalign
    :type tmscore: :class:`float`
    :param flat_mapping: Dictionary with target chain names as key and the
                         mapped mdl chain names as value. Only mapped pairs!
                         Non mapped chains can be left out.
    :type flat_mapping: :class:`dict` with :class:`str` as key/value
    """
    def __init__(self, trg_filename, mdl_filename, tmscore, flat_mapping):
        self.trg_filename = trg_filename
        self.mdl_filename = mdl_filename
        self.tmscore = tmscore
        self.flat_mapping = flat_mapping

    @staticmethod
    def FromFile(infile):
        """ Static function to create object of type MMAlignResult from line in
        predictioncenter SUMMARY file

        :param line: Single line of MMalign SUMMARY file
        :type line: :class:`str`
        """
        trg_filename, mdl_filename, trg_mapping_str, mdl_mapping_str = "", "", "" , ""
        tmscore = 0
        for line in open(infile):
          line = line.rstrip('\n')
          if line.find('Name of Structure_2:') == 0:
            contents = line.split(':', maxsplit=2)
            trg_filename = contents[1].split('/')[-1]
            trg_mapping_str = contents[2]
          elif line.find('Name of Structure_1:') == 0:
            contents = line.split(':', maxsplit=2)
            mdl_filename = contents[1].split('/')[-1]
            mdl_mapping_str = contents[2]
          elif line.find('TM-score=') == 0 and line.find('Structure_1') > 0:
            tmscore = float(line.split()[1])

        print(infile)
        split_mdl_mapping_str = mdl_mapping_str.split()[0].split(':')
        split_trg_mapping_str = trg_mapping_str.split()[0].split(':')
        assert(len(split_mdl_mapping_str) > 0 and \
               len(split_mdl_mapping_str) == len(split_trg_mapping_str))
        flat_mapping = dict()
        for trg_ch, mdl_ch in zip(split_trg_mapping_str, split_mdl_mapping_str):
            if trg_ch != '' and mdl_ch != '':
                # USalign result from the predictioncenter come as
                # 1,A instead of A
                if ',' in trg_ch and ',' in mdl_ch:
                    flat_mapping[trg_ch.split(',')[1]] = mdl_ch.split(',')[1]
                else:
                    flat_mapping[trg_ch] = mdl_ch
        return MMalignResult(trg_filename, mdl_filename.replace('_filtered', ''), tmscore, flat_mapping)

def _process_model(trg, mdl, mmalign_result):
    """ Computes QS-score for one single model

    Has grown into a function to identify all the weirdness we get
    from CASP models...
    Returns a dict with keys "score", "errors" and "n_atoms_removed".
    If any error occurs, score and n_atoms_removed will be None and
    errors are listed in "errors"

    Currently we're dealing with:

    - empty chain names
    - requirement of the ChainMapper: residue numbers in a chain must be
      strictly increasing
    - mappings of mmalign which are invalid, i.e. mapping between chains
      with different sequences... well, in the end thats what MMalign is
      supposed to do, sequence independent superposition. But in this case
      we should filter it out anyways.
    """

    # catch models which have empty chain names
    empty_ch = False
    for ch in mdl.chains:
        if ch.GetName().strip() == "":
            empty_ch = True
            break
    if empty_ch:
        return {"QS_global": None,
                "QS_best": None,
                "tm_score": mmalign_result.tmscore,
                "errors": "empty chain name observed"}

    # check whether the flat mapping from mmalign is actually valid,
    # i.e. whether only chains with equal sequence are mapped together
    mapper = ChainMapper(trg, resnum_alignments=True)
    chem_groups = mapper.chem_groups
    trg_chem_group_mapper = dict()
    for chem_group_idx, chem_group in enumerate(chem_groups):
        for cname in chem_group:
            trg_chem_group_mapper[cname] = chem_group_idx
    try:
        chem_mapping, _, _ = mapper.GetChemMapping(mdl)
    except RuntimeError as e:
        if str(e).startswith("Residue numbers in input structures must be "):
            return {"QS_global": None,
                    "QS_best": None,
                    "tm_score": mmalign_result.tmscore,
                    "errors": ["Model does not fulfull requirement of "
                               "strictly increasing residue numbers in "
                               "each chain"]}
        else:
            raise
    mdl_chem_group_mapper = dict()
    for chem_group_idx, mdl_chem_group in enumerate(chem_mapping):
        for cname in mdl_chem_group:
            mdl_chem_group_mapper[cname] = chem_group_idx

    print(mmalign_result.flat_mapping)
    mismatches = list()
    for k,v in mmalign_result.flat_mapping.items():
        if k not in trg_chem_group_mapper or v not in mdl_chem_group_mapper:
            return {"QS_global": None,
                    "QS_best": None,
                    "tm_score": mmalign_result.tmscore,
                    "errors": [f"Trg ch {k} or mdl ch {v} not present... "
                               f"possibly removed because of terrible stereo-"
                               f"chemistry?"]}
        if trg_chem_group_mapper[k] != mdl_chem_group_mapper[v]:
            mismatches.append((k, v))

    if len(mismatches) > 0:
        return {"QS_global": None,
                "QS_best": None,
                "tm_score": mmalign_result.tmscore,
                "errors": [f"invalid chain mapping (trg {x[0]}, mdl {x[1]})" \
                           for x in mismatches]}

    # make sure that we only have residues with all required backbone atoms
    trg, _, _ = mapper.ProcessStructure(trg)
    mdl, _, _ = mapper.ProcessStructure(mdl)

    # setup QSScorer and score
    alns = dict()
    remove_from_flat_mapping = list()
    flat_mapping = dict(mmalign_result.flat_mapping)
    for k, v in flat_mapping.items():
        trg_ch = trg.Select(f"cname={k}")
        mdl_ch = mdl.Select(f"cname={v}")
        if len(trg_ch.residues) == 0 or len(mdl_ch.residues) == 0:
            remove_from_flat_mapping.append(k)
            continue
        trg_s = ''.join([r.one_letter_code for r in trg_ch.residues])
        trg_s = seq.CreateSequence(k, trg_s)
        trg_s.AttachView(trg_ch)
        mdl_s = ''.join([r.one_letter_code for r in mdl_ch.residues])
        mdl_s = seq.CreateSequence(v, mdl_s)
        mdl_s.AttachView(mdl_ch)
        aln = mapper.Align(trg_s, mdl_s, mol.ChemType.AMINOACIDS)
        alns[(k,v)] = aln
        #print(aln.ToString())

    for k in remove_from_flat_mapping:
        del flat_mapping[k]

    qs_scorer = QSScorer(trg, mapper.chem_groups, mdl, alns)
    score_result = qs_scorer.FromFlatMapping(flat_mapping)

    return {"QS_global": score_result.QS_global,
            "QS_best": score_result.QS_best,
            "tm_score": mmalign_result.tmscore,
            "errors": list()}

def _cal_qsscore(inparams):

    modeldir, pdb1, pdb2, mmalign_out_dir, outdir = inparams
    
    mmalign_data = MMalignResult.FromFile(f"{mmalign_out_dir}/{pdb1}_{pdb2}.mmalign")

    pdb1_ent = io.LoadPDB(modeldir + '/' + pdb1)
    # do cleaning and stereochemistry checks on target
    ms1 = MolckSettings(rm_unk_atoms=True,
                       rm_non_std=True,
                       rm_hyd_atoms=True,
                       rm_oxt_atoms=True,
                       rm_zero_occ_atoms=False,
                       colored=False,
                       map_nonstd_res=True,
                       assign_elem=True)
    Molck(pdb1_ent, conop.GetDefaultLib(), ms1)
    stereo_param = ReadStereoChemicalPropsFile()
    pdb1_ent = pdb1_ent.CreateFullView()
    
    pdb2_ent = io.LoadPDB(modeldir + '/' + pdb2)
    ms2 = MolckSettings(rm_unk_atoms=True,
                       rm_non_std=True,
                       rm_hyd_atoms=True,
                       rm_oxt_atoms=True,
                       rm_zero_occ_atoms=False,
                       colored=False,
                       map_nonstd_res=True,
                       assign_elem=True)
    Molck(pdb2_ent, conop.GetDefaultLib(), ms2)
    stereo_param = ReadStereoChemicalPropsFile()
    pdb2_ent = pdb2_ent.CreateFullView()

    results = _process_model(pdb1_ent, pdb2_ent, mmalign_data)
    
    with open(f"{outdir}/{pdb1}_{pdb2}.qsscore", 'w') as fh:
        json.dump(results, fh)

    print(results)

    return results

def main():
    args = _parse_args()
    results = {}
    for target in os.listdir(args.indir):
        if target.find('.csv') > 0:
            continue
        print(f"Processing {target}")
        pdbs = sorted(os.listdir(args.indir + '/' + target))
        process_list = []
        for i in range(len(pdbs)):
            for j in range(len(pdbs)):
                pdb1 = pdbs[i]
                pdb2 = pdbs[j]
                if pdb1 == pdb2:
                    continue
                if os.path.exists(f"{args.outdir}/{target}/{pdb1}_{pdb2}.qsscore"):
                    continue
                process_list.append([args.indir + '/' + target, pdb1, pdb2, args.mmalign_dir + '/' + target, args.outdir + '/' + target])

        if not os.path.exists(args.outdir + '/' + target):
            os.makedirs(args.outdir + '/' + target)

        pool = Pool(processes=int(args.procnum))
        results = pool.map(_cal_qsscore, process_list)
        pool.close()
        pool.join()

        scores_dict = {}
        for i in range(len(pdbs)):
            for j in range(len(pdbs)):
                pdb1 = pdbs[i]
                pdb2 = pdbs[j]
                if pdb1 == pdb2:
                    continue
                qsscore_file = f"{args.outdir}/{target}/{pdb1}_{pdb2}.qsscore"
                if not os.path.exists(qsscore_file):
                    raise Exception(f"cannot find {qsscore_file}")
                scores_dict[f"{pdb1}_{pdb2}"] = read_qsscore(qsscore_file)

        data_dict = {}
        for i in range(len(pdbs)):
            pdb1 = pdbs[i]
            tmscores = []
            for j in range(len(pdbs)):
                pdb2 = pdbs[j]
                tmscore = 1
                if pdb1 != pdb2:
                    tmscore = scores_dict[f"{pdb1}_{pdb2}"]
                tmscores += [tmscore]
            data_dict[pdb1] = tmscores

        pd.DataFrame(data_dict).to_csv(args.outdir + '/' + target + '.csv')

if __name__ == '__main__':
    main()