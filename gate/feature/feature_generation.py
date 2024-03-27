import os, sys, argparse, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from gate.tool.utils import *
import re, subprocess
import itertools
from gate.tool.protein import *
from gate.tool.alignment import *
import copy
import json

def complex_feature_generation(fasta_path, input_model_dir, output_dir, config, features_multimer, use_alphafold_features=False, pkldir=""):

    feature_file_dict = {}

    os.makedirs(output_dir, exist_ok=True)

    aligned_model_dir = os.path.join(output_dir, 'aligned_models')
    os.makedirs(aligned_model_dir, exist_ok=True)

    # filter models by sequences
    print("################## 1. Aligning models by sequence #########################")
    cmd = f"python {config.scripts.align_model_script} --clustalw_program {config.tools.clustalw_program} --fasta_path {fasta_path} --modeldir {input_model_dir} --outdir {aligned_model_dir}"
    print(cmd)
    os.system(cmd)

    num_input_models = len(os.listdir(input_model_dir))
    num_aligned_models = len(os.listdir(aligned_model_dir))
    print(f"Successfully aligned models: {num_aligned_models} out of {num_input_models}!")

    feature_dir = os.path.join(output_dir, 'feature')
    os.makedirs(feature_dir, exist_ok=True)

    # calculate pairwise similarity scores
    workdir = os.path.join(feature_dir, 'mmalign_pairwise')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.pairwise_mmalign = os.path.join(workdir, 'pairwise_mmalign.csv')
    print("#### 2. Generating pairwise similarity scores using MMAlign ####")
    if not os.path.exists(features_multimer.pairwise_mmalign):
        os.system(f"python {config.scripts.mmalign_pairwise_script} --indir {aligned_model_dir} --outdir {workdir} --mmalign_program {config.tools.mmalign_program}")
    else:
        print("Pairwise similarity scores using MMAlign has been generated!")

    workdir = os.path.join(feature_dir, 'usalign_pairwise')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.pairwise_usalign = os.path.join(workdir, 'pairwise_usalign.csv')
    print("#### 3. Generating pairwise similarity scores using USAlign ####")
    if not os.path.exists(features_multimer.pairwise_usalign):
        os.system(f"python {config.scripts.usalign_pairwise_script} --indir {aligned_model_dir} --outdir {workdir} --usalign_program {config.tools.usalign_program}")
    else:
        print("Pairwise similarity scores using USAlign has been generated!")

    print("#### 4. Generating interface pairwise similarity scores (DockQ_wave, DockQ_ave, CAD-score) #### ")
    workdir = os.path.join(feature_dir, 'interface_pairwise')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.pairwise_dockq_wave = os.path.join(workdir, 'dockq_wave.csv')
    features_multimer.pairwise_dockq_ave = os.path.join(workdir, 'dockq_ave.csv')
    features_multimer.pairwise_cad_score = os.path.join(workdir, 'cad_score.csv')

    if not os.path.exists(features_multimer.pairwise_dockq_wave):
        script_name = os.path.basename(config.scripts.interface_pairwise_script)
        cmd = f"docker run --rm -v {config.scripts.interface_pairwise_script}:/home/{script_name} -v {output_dir}:/home " + '-u $(id -u ${USER}):$(id -g ${USER}) ' + config.envs.openstructure
        cmd += f" /home/{script_name} --indir /home/{os.path.basename(aligned_model_dir)} --outdir /home/feature/{os.path.basename(workdir)}"
        print(cmd)
        os.system(cmd)
    else:
        print("Interface pairwise similarity scores (DockQ_wave, DockQ_ave, CAD-score) has been generated!")

    print("#### 5. Generating interface pairwise similarity scores (QS-Score) #### ")
    workdir = os.path.join(feature_dir, 'qsscore_pairwise')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.pairwise_qsscore = os.path.join(workdir, 'qsscore.csv')
    if not os.path.exists(features_multimer.pairwise_qsscore):
        script_name = os.path.basename(config.scripts.qsscore_pairwise_script)
        cmd = f"docker run --rm -v {config.scripts.qsscore_pairwise_script}:/home/{script_name} -v {output_dir}:/home " + '-u $(id -u ${USER}):$(id -g ${USER}) ' + config.envs.openstructure
        cmd += f" /home/{script_name} --indir /home/{os.path.basename(aligned_model_dir)} --outdir /home/feature/{os.path.basename(workdir)} --mmalign_score_dir /home/feature/mmalign_pairwise/scores"
        print(cmd)
        os.system(cmd)
    else:
        print("Interface pairwise similarity score (QS-Score) has been generated!")

    print("#### 6. Generating icps scores (CDPred) #### ")
    workdir = os.path.join(feature_dir, 'icps')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.icps = os.path.join(workdir, 'icps.csv')
    if not os.path.exists(features_multimer.icps):
        cmd = f"python {config.scripts.icps_script} --fasta_path {fasta_path} --model_dir {aligned_model_dir} --outdir {workdir} " \
              f"--pairwise_score_csv {features_multimer.pairwise_mmalign} " \
              f"--hhblits_databases {config.databases.hhblits_bfd_database} --hhblits_binary_path {config.tools.hhblits_binary_path} " \
              f"--jackhmmer_database {config.databases.jackhmmer_database} --jackhmmer_binary {config.tools.jackhmmer_binary_path} " \
              f"--clustalw_program {config.tools.clustalw_program} --cdpred_env_path {config.envs.cdpred} --cdpred_program_path {config.tools.cdpred}"
        print(cmd)
        os.system(cmd)
    else:
        print("icps scores has been generated!")

    features_multimer.model_size = os.path.join(workdir, 'model_size.csv')

    print("#### 7. Generating alphafold plddt scores ####")
    workdir = os.path.join(feature_dir, 'plddt')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.plddt = os.path.join(workdir, 'plddt.csv')
    if not os.path.exists(features_multimer.plddt):
        cmd = f"python {config.scripts.plddt_script} --indir {aligned_model_dir} --outdir {workdir} --model_size_csv {features_multimer.model_size}"
        os.system(cmd)
    else:
        print("alphafold plddt scores has been generated!")

    print("#### 8. Generating EnQA scores ####")
    workdir = os.path.join(feature_dir, 'enqa')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.enqa = os.path.join(workdir, 'enqa.csv')

    if not os.path.exists(features_multimer.enqa):
        cmd = f"python {config.scripts.enqa_script} --fasta_path {fasta_path} --indir {aligned_model_dir} --outdir {workdir} --model_size_csv {features_multimer.model_size} " \
              f"--enqa_env_path={config.envs.enqa} --enqa_program_path={config.tools.enqa}"
        print(cmd)
        os.system(cmd)
    else:
        print("EnQA scores has been generated!")

    print("#### 9. Generating DProQA scores ####")
    workdir = os.path.join(feature_dir, 'dproqa')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.dproqa = os.path.join(workdir, 'dproqa.csv')

    if not os.path.exists(features_multimer.dproqa):
        cmd = f"python {config.scripts.dproqa_script} --indir {aligned_model_dir} --outdir {workdir} --model_size_csv {features_multimer.model_size} " \
              f"--dproqa_env_path={config.envs.dproqa} --dproqa_program_path={config.tools.dproqa}"
        print(cmd)
        os.system(cmd)
    else:
        print("DProQA scores has been generated!")

    print("#### 10. Generating GCPNET-EMA scores ####")
    workdir = os.path.join(feature_dir, 'gcpnet_ema')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.gcpnet_ema = os.path.join(workdir, 'esm_plddt.csv')
    if not os.path.exists(features_multimer.gcpnet_ema):
        cmd = f"python {config.scripts.gcpnet_script} --indir {aligned_model_dir} --outdir {workdir} --model_size_csv {features_multimer.model_size} " \
              f"--gcpnet_ema_env_path={config.envs.gcpnet_ema} --gcpnet_ema_program_path={config.tools.gcpnet_ema}"
        print(cmd)
        os.system(cmd)
    else:
        print("GCPNET-EMA scores has been generated!")

    print("#### 11. Generating Voro scores ####")
    workdir = os.path.join(feature_dir, 'voro')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.voro = os.path.join(workdir, 'voro.csv')
    if not os.path.exists(features_multimer.voro):
        cmd = f"python {config.scripts.voro_script} --indir {aligned_model_dir} --outdir {workdir} --model_size_csv {features_multimer.model_size} --voro_program_path={config.tools.ftdmp} --voro_env_path={config.envs.ftdmp}"
        os.system(cmd)
    else:
        print("Voro scores has been generated!")

    print("#### 12. Generating edge features ####")
    workdir = os.path.join(feature_dir, 'edge_feature')
    os.makedirs(workdir, exist_ok=True)

    features_multimer.common_interface = os.path.join(workdir, 'common_interface.csv')
    if not os.path.exists(features_multimer.common_interface):
        cmd = f"python {config.scripts.edge_script} --indir {aligned_model_dir} --outdir {workdir} --cdpreddir {feature_dir}/icps"
        os.system(cmd)
    else:
        print("edge features has been generated!")

    if use_alphafold_features:

        if not os.path.exists(pkldir):
            raise Exception(f"Cannot find {pkldir} to generate alphafold features!")

        print("#### 13. Generating Alphafold feature scores ####")
        workdir = os.path.join(feature_dir, 'alphafold')
        os.makedirs(workdir, exist_ok=True)

        features_multimer.af_features = os.path.join(workdir, 'af_features.csv')
        if not os.path.exists(features_multimer.af_features):
            cmd = f"python {config.scripts.alphafold_feature_script} --fasta_path {fasta_path} --pdbdir {aligned_model_dir} --pkldir {pkldir} --outfile {features_multimer.af_features}"
            os.system(cmd)
        else:
            print("Alphafold feature scores has been generated!")

def monomer_feature_generation(targetname, fasta_path, input_model_dir, output_dir, config, contact_map_file, dist_map_file, features_monomer):

    os.makedirs(output_dir, exist_ok=True)

    aligned_model_dir = os.path.join(output_dir, 'aligned_models')
    if os.path.exists(aligned_model_dir):
        os.system(f"rm -rf {aligned_model_dir}")
    os.system(f"cp -r {input_model_dir} {aligned_model_dir}")

    feature_dir = os.path.join(output_dir, 'feature')
    os.makedirs(feature_dir, exist_ok=True)

    # calculate pairwise similarity scores
    workdir = os.path.join(feature_dir, 'tmscore_pairwise')
    os.makedirs(workdir, exist_ok=True)

    features_monomer.pairwise_tmscore = os.path.join(workdir, 'pairwise_tmscore.csv')
    features_monomer.pairwise_gdtscore = os.path.join(workdir, 'pairwise_tmscore.csv')
    print("#### 1. Generating pairwise similarity scores using TMscore ####")
    if not os.path.exists(features_monomer.pairwise_tmscore):
        os.system(f"python {config.scripts.tmscore_pairwise_script} --indir {aligned_model_dir} --outdir {workdir} --tmscore_program {config.tools.tmscore_program}")
    else:
        print("pairwise similarity scores using TMscore has been generated!")

    print("#### 2. Generating interface pairwise similarity scores (lddt, cad-score) ####")
    workdir = os.path.join(feature_dir, 'interface_pairwise')
    os.makedirs(workdir, exist_ok=True)

    features_monomer.pairwise_lddt = os.path.join(workdir, 'lddt.csv')
    features_monomer.pairwise_cad_score = os.path.join(workdir, 'cad_score.csv')
    if not os.path.exists(features_monomer.pairwise_lddt):
        script_name = os.path.basename(config.scripts.interface_pairwise_ts_script)
        cmd = f"docker run --rm -v {config.scripts.interface_pairwise_ts_script}:/home/{script_name} -v {output_dir}:/home " + '-u $(id -u ${USER}):$(id -g ${USER}) ' + config.envs.openstructure
        cmd += f" /home/{script_name} --indir /home/{os.path.basename(aligned_model_dir)} --outdir /home/feature/{os.path.basename(workdir)} "
        os.system(cmd)
    else:
        print("interface pairwise similarity scores using TMscore has been generated!")

    print("#### 3. Generating alphafold plddt scores ####")
    workdir = os.path.join(feature_dir, 'plddt')
    os.makedirs(workdir, exist_ok=True)

    features_monomer.plddt = os.path.join(workdir, 'plddt.csv')
    if not os.path.exists(features_monomer.plddt):
        cmd = f"python {config.scripts.plddt_script} --indir {aligned_model_dir} --outdir {workdir}"
        print(cmd)
        os.system(cmd)
    else:
        print("alphafold plddt scores has been generated!")

    print("#### 4. Generating EnQA scores ####")
    workdir = os.path.join(feature_dir, 'enqa')
    os.makedirs(workdir, exist_ok=True)

    features_monomer.enqa = os.path.join(workdir, 'enqa.csv')
    if not os.path.exists(features_monomer.enqa):
        cmd = f"python {config.scripts.enqa_script} --fasta_path {fasta_path} --indir {aligned_model_dir} --outdir {workdir} " \
              f"--enqa_env_path={config.envs.enqa} --enqa_program_path={config.tools.enqa}"
        print(cmd)
        os.system(cmd)
    else:
        print("EnQA scores has been generated!")

    print("#### 5. Generating GCPNET-EMA scores ####")
    workdir = os.path.join(feature_dir, 'gcpnet_ema')
    os.makedirs(workdir, exist_ok=True)

    features_monomer.gcpnet_ema = os.path.join(workdir, 'esm_plddt.csv')
    if not os.path.exists(features_monomer.gcpnet_ema):
        cmd = f"python {config.scripts.gcpnet_script} --indir {aligned_model_dir} --outdir {workdir} " \
              f"--gcpnet_ema_env_path={config.envs.gcpnet_ema} --gcpnet_ema_program_path={config.tools.gcpnet_ema}"
        print(cmd)
        os.system(cmd)
    else:
        print("GCPNET-EMA scores has been generated!")

    print("#### 6. Generating DeepRank3 scores ####")
    workdir = os.path.join(feature_dir, 'DeepRank3')
    os.makedirs(workdir, exist_ok=True)

    features_monomer.deeprank3_cluster = os.path.join(workdir, 'DeepRank3_Cluster.txt')
    if not os.path.exists(features_monomer.deeprank3_cluster):
        cmd = f"sh {config.scripts.deeprank3_cluster_script} {targetname} {fasta_path} {aligned_model_dir} {workdir} {contact_map_file} {dist_map_file}"
        print(cmd)
        os.system(cmd)
    else:
        print("DeepRank3_Cluster scores has been generated!")

    features_monomer.deeprank3_singleqa = os.path.join(workdir, 'DeepRank3_SingleQA.txt')
    if not os.path.exists(features_monomer.deeprank3_singleqa):
        cmd = f"sh {config.scripts.deeprank3_singleqa_script} {targetname} {fasta_path} {aligned_model_dir} {workdir} {contact_map_file} {dist_map_file}"
        print(cmd)
        os.system(cmd)
    else:
        print("DeepRank3_SingleQA scores has been generated!")

    features_monomer.deeprank3_singleqa_lite = os.path.join(workdir, 'DeepRank3_SingleQA_lite.txt')
    if not os.path.exists(features_monomer.deeprank3_singleqa_lite):
        cmd = f"sh {config.scripts.deeprank3_singleqa_lite_script} {targetname} {fasta_path} {aligned_model_dir} {workdir} {contact_map_file} {dist_map_file}"
        print(cmd)
        os.system(cmd)
    else:   
        print("DeepRank3_SingleQA_lite scores has been generated!")

    features_monomer.deeprank3_features = os.path.join(workdir, 'ALL_14_scores')

