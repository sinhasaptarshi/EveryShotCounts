#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-24:00:00


source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

cd Deep-Temporal-Repetition-Counting/
python main.py --pretrain_path=pretrained_models/resnext-101-kinetics.pth