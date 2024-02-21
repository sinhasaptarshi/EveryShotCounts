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
python exemplar_counting_train.py --num_gpus 1 --only_test --dataset EgoLoops --trained_model saved_models_peaks1.0_multishot_fullrepcount_lr2e-6_countnormalized/epoch_180.pyth --token_pool_ratio 0.4 --encodings mae --multishot --iterative_shots 
