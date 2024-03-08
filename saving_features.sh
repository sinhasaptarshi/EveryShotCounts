#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=devel
#SBATCH --time=0-03:00:00


source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
# python save_swim_features.py --dataset RepCount --model VideoMAE --num_gpus 4 --save_exemplar_encodings True
python save_swim_features.py --dataset Countix --model 3D-ResNeXt101 --num_gpus 4 --save_exemplar_encodings True
# python save_swim_features.py --dataset RepCount --model VideoSwin --num_gpus 4 --save_exemplar_encodings True
# python save_egoloops_tokens.py --dataset EgoLoops --model VideoMAE --num_gpus 4 --save_exemplar_encodings True
# python convert_tokens.py
