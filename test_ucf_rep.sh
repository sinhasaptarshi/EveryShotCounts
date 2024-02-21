#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=vtf
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --time=0-03:00:00


source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate repcount

export WANDB_MODE=offline
python exemplar_counting_train.py --num_gpus 1 --only_test --tokens_dir saved_3D-ResNeXt101tokens_UCFRep_contextpretrained --exemplar_dir exemplar_3D-ResNeXt101tokens_UCFRep_contextpretrained --trained_model saved_models_resnextencodings_ucfrep_multishot_countnormalized_contextpretrained/epoch_044.pyth --dataset UCFRep --multishot --iterative_shots --token_pool_ratio 1.0 --encodings resnext
