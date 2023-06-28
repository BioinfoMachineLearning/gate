conda create -n gate python=3.8
conda activate gate
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install -c dglteam dgl-cuda11.0
conda install chardet scikit-learn pandas

pip install wandb
wandb login