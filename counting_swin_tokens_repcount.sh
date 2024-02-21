#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-24:00:00

# mkdir /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r exemplar_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r saved_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r exemplar_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r saved_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --dataset RepCount --tokens_dir saved_swintokens_repcount --exemplar_dir exemplar_swintokens_repcount --save_path saved_models_repcountfull_swinencoded_fullattention_threshold0.0_v2_lr1e-5 --token_pool_ratio 0.8 --multishot --iterative_shots --lr 8e-6 --encodings swin --slurm_job_id $SLURM_JOB_ID --threshold 0.0 --full_attention
# rm -rf raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID/exemplar_tokens_reencoded
# rm -rf raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID/saved_tokens_reencoded