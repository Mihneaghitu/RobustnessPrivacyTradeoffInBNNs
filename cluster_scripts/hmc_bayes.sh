#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=grid_search_bayes.out
export PATH=/vol/bitbucket/mg2720/fypvenv/bin/:$PATH
source activate
which python
source /vol/cuda/12.2.0/setup.sh
cd ~/fyp/RobustnessPrivacyTradeoffInBNNs/cluster_scripts
python run_hmc_grid_search.py $1
