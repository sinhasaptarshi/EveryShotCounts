#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --time=1-12:00:00

# cp -r exemplar_VideoMAEtokens_UCFRep /raid/local_scratch/sxs63-wwp01/
# cp -r saved_VideoMAEtokens_UCFRep /raid/local_scratch/sxs63-wwp01/

# cp -r exemplar_VideoSwintokens_UCFRep /raid/local_scratch/sxs63-wwp01/
# cp -r saved_VideoSwintokens_UCFRep /raid/local_scratch/sxs63-wwp01/

# cp -r exemplar_3D-ResNeXt101tokens_UCFRep_ucfpretrained /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r saved_3D-ResNeXt101tokens_UCFRep_ucfpretrained /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_finetune_encoder.py --num_gpus 1  --save_path ./saved_ucfrep_finetune_models --multishot --iterative_shots --dataset UCFRep --encodings resnext --slurm_job_id $SLURM_JOB_ID --threshold 0.0
