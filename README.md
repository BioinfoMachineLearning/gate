# GATE
Graph transformers for estimating protein model accuracy

# **Installation**

Install Mamba

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh 
rm Mambaforge-$(uname)-$(uname -m).sh
source ~/.bashrc  
```

## Install dependencies ##

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

## Install python environments ##

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


