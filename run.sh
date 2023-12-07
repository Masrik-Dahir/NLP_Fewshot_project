#!/bin/bash

#SBATCH --output=output.log
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --job-name=charlie_is_sad
#SBATCH --time=14-00:00
#SBATCH --qos=short

module load python/3.10
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
python main.py --train_path ../preprocessed_data/train/ --test_path ../preprocessed_data/test/ --dev_path ../preprocessed_data/dev/ --k 5 --train
