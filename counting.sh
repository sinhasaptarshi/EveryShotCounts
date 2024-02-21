#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-00:00:00

# mkdir /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r exemplar_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r saved_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r exemplar_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/
# cp -r saved_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/
source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --dataset RepCount --tokens_dir saved_tokens_reencoded --exemplar_dir exemplar_tokens_reencoded --save_path saved_models_repcountfull_maeencoded_fullattention_threshold0.1_v2_lr5e-5 --token_pool_ratio 0.4 --multishot --iterative_shots --lr 1e-5 --encodings mae --slurm_job_id $SLURM_JOB_ID --threshold 0.1 --full_attention
rm -rf raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID/exemplar_tokens_reencoded
rm -rf raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID/saved_tokens_reencoded
