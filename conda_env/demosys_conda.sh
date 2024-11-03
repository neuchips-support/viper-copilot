#!/bin/bash

# please make sure you already installed conda environment to ~/anaconda3 path first
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda env remove -n demo_sys
conda env create -f llamacpp_demo_env.yml
if [ -z "$(which pip3)" ]; then
   sudo apt install python3-pip
fi

conda activate demo_sys

pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



echo "============== Demo sys anaconda env setup Succeed =============="

