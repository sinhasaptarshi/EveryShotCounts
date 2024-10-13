This is the codebase for the paper "Every Shot Counts: Using Exemplars for Repetition Counting in Videos" 

[Saptarshi Sinha](https://sinhasaptarshi.github.io), [Alexandros Stergiou](https://alexandrosstergiou.github.io) and [Dima Damen](https://dimadamen.github.io)


[Proceedings of the Asian conference on computer vision (ACCV) 2024](https://accv2024.org)

Links: [[arXiv]](https://arxiv.org/abs/2403.18074) [[webpage]](https://sinhasaptarshi.github.io/escounts/)

![supported versions](https://img.shields.io/badge/python-3.x-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue/?style=flat&logo=pytorch&color=informational)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)

# Abstract

Video repetition counting infers the number of repetitions of recurring actions or motion within a video. We propose an exemplar-based approach that discovers visual correspondence of video exemplars across repetitions within target videos. Our proposed **E**very **S**hot **Counts** (ESCounts) model is an attention-based encoder-decoder that encodes videos of varying lengths alongside exemplars from the same and different videos. In training, ESCounts regresses locations of high correspondence to the exemplars within the video. In tandem, our method learns a latent that encodes representations of general repetitive motions, which we use for exemplar-free, zero-shot inference. Extensive experiments over commonly used datasets (RepCount, Countix, and UCFRep) showcase ESCounts obtaining state-of-the-art performance across all three datasets. On RepCount, ESCounts increases the off-by-one from 0.39 to 0.56 and decreases the mean absolute error from 0.38 to 0.21. Detailed ablations further demonstrate the effectiveness of our method. 


<p align="center">
<img src="./figs/landing_figure.png" width="700" height="320" />
</p>



# Install environment

Create a conda environment and activate it.

`conda create -n repcount python=3.8`

`conda activate repcount`

Install the required packages

`pip install -r requirements.txt`

# Dataset Download

Download the Repcount dataset from [here](https://svip-lab.github.io/dataset/RepCount_dataset.html) under `data/RepCount` 

Get Countix dataset from [here](https://sites.google.com/view/repnet) and get it under `data/Countix`


For UCF101, download the dataset from [here](https://www.crcv.ucf.edu/data/UCF101.php) and get it under `data/UCFRep`

# Extract VideoMAE encodings

We use a pretrained VideoMAE-v2 encoder to extract spatio-temporal tokens from videos.

Download the pretrained encoder weights from [here](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth) and put it in `pretrained_models/`

Extract spatio-temporal tokens for each video using

```python save_swin_features.py --dataset RepCount --model VideoMAE --num_gpus 1 --data_path data/RepCount```

This will create a folder `saved_VideoMAEtokens_RepCount` with tokens for all videos in the train, val and test set.

Next, extract spatio-temporal tokens for each exemplars using 

```python save_swin_features.py --dataset RepCount --model VideoMAE --num_gpus 1 --save_exemplar_encodings True --data_path data/RepCount```

Again, this will create the folder `exemplar_VideoMAEtokens_RepCount` with tokens from repetitions in each video. For each video, the shape will be `N x 3 x 8 x 14 x 14`, where N is the number of repetitions in the video.

# Train ESCounts

To train with ESCounts on the encoded tokens, use

`python exemplar_counting_train.py --num_gpus 1 --dataset RepCount --tokens_dir saved_VideoMAEtokens_RepCount --exemplar_dir exemplar_VideoMAEtokens_RepCount --save_path saved_models_repcount --token_pool_ratio 0.4 --multishot --iterative_shots --lr 5e-5 --encodings mae --threshold 0.4`

This will save checkpoints in the `save_path`. `--threshold 0.4` uses exemplars from different videos of same actions with probability of 0.4. `--token_pool_ratio 0.4` downsamples encoded spatio-temporal tokens by spatial average pooling in order to fit in memory. `--token_pool_ratio 1.0` uses no spatial average pooling. Modify this during inference appropriately.


# Testing


To run inference with trained checkpoint, run

`python exemplar_counting_train.py --dataset RepCount --tokens_dir saved_VideoMAEtokens_RepCount --exemplar_dir exemplar_VideoMAEtokens_RepCount --trained_model xxxxx.pyth --multishot --iterative_shots --get_overlapping_segments`

Replace `xxxxx.pyth` with the trained checkpoints.

You can download our trained model from [here](https://drive.google.com/file/d/1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48/view?usp=drive_link) 


# Run Demo
To run demo on any video `data/xxxx.mp4` with our trained model, use the following:

`python demo.py --video_name data/xxxx.mp4 --resource 'cpu'`

Change the resource accordingly. 
The output should be like `The number of repetitions is $x$`.


# Citation
If you find ESCounts helpful, please consider citing our paper as 

```
@InProceedings{sinha2024every,
title = {Every Shot Counts: Using Exemplars for Repetition Counting in Videos},
author = {Sinha, Saptarshi and Stergiou, Alexandros and Damen, Dima},
booktitle={Proceedings of the Asian conference on computer vision (ACCV)},
year = {2024},
}
```
