#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --time=1-24:00:00

# cp -r exemplar_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/
# cp -r saved_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/
cp -r exemplar_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/
cp -r saved_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/
source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --dataset RepCount --tokens_dir saved_swintokens_repcount --exemplar_dir exemplar_swintokens_repcount --save_path saved_models_peaks1.0_multishot_fullrepcount_lr5e-6_swinencoder --token_pool_ratio 1.0 --multishot --iterative_shots --lr 5e-6
