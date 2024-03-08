#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-12:00:00

# mkdir /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r exemplar_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r saved_tokens_reencoded/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r exemplar_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r saved_swintokens_repcount/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r exemplar_VideoMAEtokens_RepCount/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r saved_VideoMAEtokens_RepCount/ /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --dataset RepCount --tokens_dir saved_VideoMAEtokens_RepCount --exemplar_dir exemplar_VideoMAEtokens_RepCount --save_path saved_models_repcount_videomae_attndecoder477_lr5e-5_threshold0.4_nomae --token_pool_ratio 0.4 --multishot --iterative_shots --lr 5e-5 --encodings mae --slurm_job_id $SLURM_JOB_ID --threshold 0.4 --full_attention --window_size 4 7 7
# python exemplar_counting_train.py --num_gpus 1 --dataset RepCount --tokens_dir saved_swintokens_repcount --exemplar_dir exemplar_swintokens_repcount --save_path saved_models_repcount_swin_attndecoder333_lr5e-5 --token_pool_ratio 0.9 --multishot --iterative_shots --lr 5e-5 --encodings swin --slurm_job_id $SLURM_JOB_ID --threshold 0.0 --full_attention --window_size 3 3 3
# rm -rf raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID/exemplar_tokens_reencoded
# rm -rf raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID/saved_tokens_reencoded
