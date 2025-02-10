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
5. [Citing This Work](#citing-this-work)

---

## Introduction

GATE is a tool designed for estimating protein model accuracy using advanced graph transformers. This repository contains the code, pre-trained models, and instructions for setup and usage.

![Program workflow](imgs/Gate_workflow_v2.jpg)

### The overall performance of GATE (MULTICOM_GATE) in CASP16 EMA competition in terms of Z-scores

![CASP16 result](imgs/global_SCORE_summed_zscore_ranking.png)

### The overall performance of GATE (MULTICOM_GATE) in CASP16 EMA competition in terms of per-target average

| **Predictor Name**      | **Pearson's correlation** | **Spearman's correlation** | **Ranking loss** | **AUC**  |
|-------------------------|------------|------------|------------------|----------|
| GuijunLab-QA            | 0.6479     | 0.4149     | **0.1195**       | 0.6328   |
| MULTICOM                | 0.6156     | 0.4380     | _0.1207_         | 0.6660   |
| **MULTICOM_GATE**       | **0.7076** | _**0.4514**_ | _**0.1221**_     | _**0.6680**_ |
| MULTICOM_LLM            | _0.6836_   | **0.4808** | 0.1230           | _0.6685_ |
| MIEnsembles-Server      | 0.6072     | 0.4498     | 0.1325           | 0.6670   |
| GuijunLab-PAthreader    | 0.5309     | 0.3744     | 0.1331           | 0.6237   |
| ModFOLDdock2            | _**0.6542**_ | _0.4640_   | 0.1371           | **0.6859** |
| ModFOLDdock2R           | 0.5724     | 0.3867     | 0.1375           | 0.6518   |
| VifChartreuse           | 0.2921     | 0.2777     | 0.1440           | 0.6149   |
| MQA_base                | 0.4331     | 0.2897     | 0.1462           | 0.6085   |
| MQA_server              | 0.4326     | 0.2913     | 0.1468           | 0.6120   |
| GuijunLab-Human         | 0.6327     | 0.4148     | 0.1477           | 0.6368   |
| MULTICOM_human          | 0.5897     | 0.4260     | 0.1518           | 0.6576   |
| ChaePred                | 0.4548     | 0.3971     | 0.1580           | 0.6534   |
| AF_unmasked             | 0.4015     | 0.2731     | 0.1595           | 0.6052   |
| VifChartreuseJaune      | 0.3421     | 0.1756     | 0.1630           | 0.5951   |
| GuijunLab-Assembly      | 0.5439     | 0.3280     | 0.1636           | 0.6191   |
| Guijunlab-Complex       | 0.4889     | 0.3019     | 0.1792           | 0.6054   |
| ModFOLDdock2S           | 0.5285     | 0.3116     | 0.1806           | 0.6084   |
| MULTICOM_AI             | 0.3281     | 0.2623     | 0.1913           | 0.6057   |
| COAST                   | 0.3840     | 0.2297     | 0.2091           | 0.6072   |
| MQA                     | 0.4410     | 0.2425     | 0.2183           | 0.5858   |
| PIEFold_human           | 0.1929     | 0.1451     | 0.2306           | 0.5497   |



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

## Citing This Work
If you find this work useful, please cite: 

Liu, J., Neupane, P., & Cheng, J. (2025). Estimating Protein Complex Model Accuracy Using Graph Transformers and Pairwise Similarity Graphs. bioRxiv, 2025-02 (https://doi.org/10.1101/2025.02.04.63656)

```bibtex
@article {Liu2025.02.04.636562,
	author = {Liu, Jian and Neupane, Pawan and Cheng, Jianlin},
	title = {Estimating Protein Complex Model Accuracy Using Graph Transformers and Pairwise Similarity Graphs},
	elocation-id = {2025.02.04.636562},
	year = {2025},
	doi = {10.1101/2025.02.04.636562},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://doi.org/10.1101/2025.02.04.636562},
	journal = {bioRxiv}
}
```