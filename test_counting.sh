#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=0-10:00:00


source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --only_test --tokens_dir saved_VideoMAEtokens_RepCount --exemplar_dir exemplar_VideoMAEtokens_RepCount --trained_model saved_models_repcount_videomae_attndecoder477_lr5e-5_threshold0.4///epoch_220.pyth --token_pool_ratio 0.4 --encodings mae --multishot --iterative_shots --full_attention --window_size 4 7 7
