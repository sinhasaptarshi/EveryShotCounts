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
python exemplar_counting_train.py --num_gpus 1 --only_test --tokens_dir saved_3D-ResNeXt101tokens_UCFRep_ucfpretrained --exemplar_dir exemplar_3D-ResNeXt101tokens_UCFRep_ucfpretrained --trained_model saved_models_resnext_embeddings_ucfrep_multishot_ucfpretrained//epoch_298.pyth --dataset UCFRep --multishot --iterative_shots --token_pool_ratio 0.6 --encodings mae
