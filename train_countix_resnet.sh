#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-24:00:00

cp -r exemplar_3D-ResNeXt101tokens_Countix /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r saved_3D-ResNeXt101tokens_Countix /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID

source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --tokens_dir saved_3D-ResNeXt101tokens_Countix --exemplar_dir exemplar_3D-ResNeXt101tokens_Countix --save_path ./saved_models_countix_multishot_countnormalized_fullattention_resnet --lr 1e-5 --multishot --iterative_shots --dataset Countix --density_peak_width 1.0 --token_pool_ratio 1.0 --encodings resnext --full_attention --slurm_job_id $SLURM_JOB_ID
