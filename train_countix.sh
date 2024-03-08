#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-24:00:00

cp -r exemplar_tokens_countix /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r saved_tokens_countix /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID

source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --tokens_dir saved_tokens_countix --exemplar_dir exemplar_tokens_countix --save_path ./saved_models_videomae_countix_attndecoder --multishot --iterative_shots --dataset Countix --density_peak_width 1.0 --token_pool_ratio 0.6 --encodings mae --full_attention --lr 5e-5 --slurm_job_id $SLURM_JOB_ID --window_size 7 3 3
