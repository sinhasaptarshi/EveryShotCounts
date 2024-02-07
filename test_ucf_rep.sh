#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --job-name=vtf
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --partition=devel
#SBATCH --time=0-03:00:00


source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --only_test --trained_model saved_models_peaks1.0_multishot_fullrepcount_lr2e-6_countnormalized/epoch_180.pyth --tokens_dir saved_VideoMAEtokens_UCFRep --exemplar_dir exemplar_VideoMAEtokens_UCFRep --dataset UCFRep --multishot --iterative_shots --token_pool_ratio 0.6