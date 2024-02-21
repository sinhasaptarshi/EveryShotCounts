#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=1-00:00:00

# cp -r exemplar_VideoMAEtokens_UCFRep /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
# cp -r saved_VideoMAEtokens_UCFRep /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID

# cp -r exemplar_VideoSwintokens_UCFRep /raid/local_scratch/sxs63-wwp01/
# cp -r saved_VideoSwintokens_UCFRep /raid/local_scratch/sxs63-wwp01/

cp -r exemplar_3D-ResNeXt101tokens_UCFRep_ucfpretrained /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
cp -r saved_3D-ResNeXt101tokens_UCFRep_ucfpretrained /raid/local_scratch/sxs63-wwp01/$SLURM_JOB_ID
source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --tokens_dir saved_3D-ResNeXt101tokens_UCFRep_ucfpretrained --exemplar_dir exemplar_3D-ResNeXt101tokens_UCFRep_ucfpretrained --save_path ./saved_models_mae_embeddings_ucfrep_multishot_ucfpretrained_v3 --multishot --iterative_shots --dataset UCFRep --density_peak_width 1.0 --token_pool_ratio 1.0 --encodings resnext --slurm_job_id $SLURM_JOB_ID --threshold 0.0 --full_attention --lr 1e-5
