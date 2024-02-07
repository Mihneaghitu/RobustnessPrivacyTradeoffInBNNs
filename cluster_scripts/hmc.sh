#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
export PATH=/vol/bitbucket/mg2720/fypvenv/bin/:$PATH
source activate
which python
source /vol/cuda/12.2.0/setup.sh
cd ~/fyp/RobustnessPrivacyTradeoffInBNNs/cluster_scripts
python run_hmc.py $1