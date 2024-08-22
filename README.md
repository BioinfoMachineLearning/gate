# GATE: Graph Transformers for Estimating Protein Model Accuracy

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
   - [Multimer Structure Estimation](#1-multimer-structure-estimation)
   - [Monomer Structure Estimation](#2-monomer-structure-estimation)

## Introduction
GATE is a tool designed for the estimation of protein model accuracy using advanced graph transformers. This repository contains the code, models, and instructions for running the tool.

## Installation
To install GATE, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/GATE.git
   cd GATE
2. **Install Mamba:**
    ```
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh 
    rm Mambaforge-$(uname)-$(uname -m).sh
    source ~/.bashrc  
    ```
3. **Install tools:**

    ```
    cd tools

    # Install GCPNet-EMA
    git clone https://github.com/BioinfoMachineLearning/GCPNet-EMA
    mkdir GCPNet-EMA/checkpoints
    wget -P GCPNet-EMA/checkpoints/ https://zenodo.org/record/10719475/files/structure_ema_finetuned_gcpnet_i2d5t9xh_best_epoch_106.ckpt

    # Install EnQA
    git clone https://github.com/BioinfoMachineLearning/EnQA
    chmod -R 755 EnQA/utils

    # Install DProQA
    git clone https://github.com/jianlin-cheng/DProQA

    # Install Venclovas QAs
    git clone https://github.com/kliment-olechnovic/ftdmp

    # Install CDPred
    git clone https://github.com/BioinfoMachineLearning/CDPred

    # Install openstructure
    docker pull registry.scicore.unibas.ch/schwede/openstructure:latest
    # or
    singularity pull docker://registry.scicore.unibas.ch/schwede/openstructure:latest
    ```
4. **Install additional tools for estimating protein monomer structures (optional)**

    Follow the instructions in https://github.com/jianlin-cheng/DeepRank3
    
5. **Install python environments**

    ```
    mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    mamba install -c dglteam dgl-cuda11.0
    mamba install pandas biopython

    mamba env create -f envs/gcpnet_ema.yaml
    mamba env create -f envs/enqa.yaml
    mamba env create -f envs/dproqa.yaml

    # mamba env create -f envs/ftdmp.yaml
    # install ftdmp (https://github.com/kliment-olechnovic/ftdmp/issues/4)
    mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
    mamba install pyg pytorch-cluster pytorch-scatter pytorch-sparse pytorch-spline-conv -c pyg
    mamba install -c conda-forge pandas
    mamba install -c conda-forge r-base

    # install CDPred
    # mamba create -n cdpred python=3.6
    # pip install -r tools/CDPred/requirments.txt
    # mamba install -y -c bioconda hmmer hhsuite==3.3.0 
    mamba env create -f envs/cdpred.yaml

    ```

6. **Download databases (~2.5T)**

    ```
    mkdir databases
    sh scripts/download_bfd.sh databases/
    sh scripts/download_uniref90.sh databases/
    ```

## **Configuration**
    
    * Replace the contents for the ROOTDIR in gate/feature/config.py with your installation path

    * Change use_docker to False if using Singularity

## Usage

### **1. Multimer Structure Estimation**
To run the GATE tool for estimating protein multimer structure accuracy, use the `inference_multimer.py` script with the following arguments:

#### Required Arguments:
* --fasta_path FASTA_PATH

    The path to the input FASTA file containing the protein sequences.

* --input_model_dir INPUT_MODEL_DIR
    
    The directory containing the input protein models.

* --output_dir OUTPUT_DIR
    
    The directory where the output results will be saved.

#### Optional Arguments:
* --pkldir PKLDIR

    The directory where intermediate pickle files will be stored.

* --use_af_feature USE_AF_FEATURE

    Specify whether to use AlphaFold features. Accepts True or False. Default is False.

* --sample_times SAMPLE_TIMES
    Number of times to sample the models. Default is 5.

#### Example Command:

Here are examples of how to use the `inference_multimer.py` script with different settings:

1. **Not using AlphaFold Features (default)**

   ```bash
   python inference_multimer.py --fasta_path $FASTA_PATH --input_model_dir $INPUT_MODEL_DIR --output_dir $OUTPUT_DIR

2. **Using AlphaFold Features**
    ```bash
    python inference_multimer.py --fasta_path $FASTA_PATH --input_model_dir $INPUT_MODEL_DIR --output_dir $OUTPUT_DIR --pkldir $PKLDIR --use_af_feature True
    ```

### **2. Monomer Structure Estimation**
To run the GATE tool for estimating protein multimer structure accuracy, use the `inference_monomer.py` script with the following arguments:

#### Required Arguments:
* --fasta_path FASTA_PATH

    The path to the input FASTA file containing the protein sequences.

* --input_model_dir INPUT_MODEL_DIR
    
    The directory containing the input protein models.

* --output_dir OUTPUT_DIR
    
    The directory where the output results will be saved.

#### Optional Arguments: 

* --contact_map_file CONTACT_MAP_FILE
    
    The path to the contact map file.

* --dist_map_file DIST_MAP_FILE
    
    The path to the distance map file.

* --sample_times SAMPLE_TIMES
    Number of times to sample the models. Default is 5.

#### Example Command:

```bash
python inference_monomer.py --fasta_path $FASTA_PATH --input_model_dir $INPUT_MODEL_DIR --output_dir $OUTPUT_DIR
```