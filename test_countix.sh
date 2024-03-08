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
python exemplar_counting_train.py --num_gpus 1 --only_test --trained_model saved_models_countix_multishot/epoch_270.pyth --tokens_dir saved_tokens_countix --exemplar_dir exemplar_tokens_countix --dataset Countix --multishot --iterative_shots --token_pool_ratio 0.6 --encodings mae --window_size 3 3 3 