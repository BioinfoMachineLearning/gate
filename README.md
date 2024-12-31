# GATE: Graph Transformers for Estimating Protein Model Accuracy

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
   - [Clone the Repository](#clone-the-repository)
   - [Install Mamba](#install-mamba)
   - [Install Tools](#install-tools)
   - [Set Up Python Environments](#set-up-python-environments)
   - [Download Databases](#download-databases)
3. [Configuration](#configuration)
4. [Usage](#usage)
   - [Required Arguments](#required-arguments)
   - [Optional Arguments](#optional-arguments)
   - [Example Commands](#example-commands)
5. [Contact](#contact)

---

## Introduction

GATE is a tool designed for estimating protein model accuracy using advanced graph transformers. This repository contains the code, pre-trained models, and instructions for setup and usage.

---

## Installation

### Clone the Repository

```bash
git clone -b public https://github.com/BioinfoMachineLearning/gate
cd gate
```


### Install Mamba

```
wget "https://github.com/conda-forge/miniforge/releases/download/23.1.0-3/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh 
rm Mambaforge-$(uname)-$(uname -m).sh
source ~/.bashrc  
```

### Install tools

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

### Set Up Python Environments

``` 
# Install python enviorment for gate
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -c dglteam dgl-cuda11.0
mamba install pandas biopython

# Install python enviorment for GCPNet-EMA
mamba env create -f tools/GCPNet-EMA/environment.yaml
mamba activate GCPNet-EMA
pip3 install -e tools/GCPNet-EMA
pip3 install prody==2.4.1
pip3 uninstall protobuf
mamba deactivate

# Install python enviorment for EnQA
mamba env create -f envs/enqa.yaml

# Install python enviorment for DProQA
mamba env create -f envs/dproqa.yaml

# Install python enviorment for VoroMQA
mamba env create -f envs/ftdmp.yaml

# Install python enviorment for CDPred
mamba env create -f envs/cdpred.yaml

```

### Download databases (~2.5T)

```
mkdir databases

# Create virtual links if the databases are stored elsewhere
sh scripts/download_bfd.sh databases/
sh scripts/download_uniref90.sh databases/
```

### **Configuration**
    
    * Replace the contents for the ROOTDIR in gate/feature/config.py with your installation path

    * Set use_docker to False if using Singularity instead of Docker.

## Usage

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

#### Example Commands:

Here are examples of how to use the `inference_multimer.py` script with different settings:

1. **Not using AlphaFold Features (default)**

   ```bash
   python inference_multimer.py --fasta_path $FASTA_PATH --input_model_dir $INPUT_MODEL_DIR --output_dir $OUTPUT_DIR

2. **Using AlphaFold Features**
    ```bash
    python inference_multimer.py --fasta_path $FASTA_PATH --input_model_dir $INPUT_MODEL_DIR --output_dir $OUTPUT_DIR --pkldir $PKLDIR --use_af_feature True
    ```
