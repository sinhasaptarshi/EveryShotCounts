This is the codebase for the paper "Every Shot Counts: Using Exemplars for Repetition Counting in Videos"

# Install environment

Create a conda environment and activate it.

`conda create -n repcount python=3.8`

`conda activate repcount`

Install the required packages

`pip install -r requirements.txt`



# Extract VideoMAE encodings

We use a pretrained VideoMAE-v2 encoder to extract spatio-temporal tokens from videos.

Download the pretrained encoder weights from [here](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth) and put it in `pretrained_models/`

Extract spatio-temporal tokens for each video using

```python save_swin_features.py --dataset RepCount --model VideoMAE --num_gpus 1```

This will create a folder `saved_VideoMAEtokens_RepCount` with tokens for all videos in the train, val and test set.

Next, extract spatio-temporal tokens for each exemplars using 

```python save_swin_features.py --dataset RepCount --model VideoMAE --num_gpus 1 --save_exemplar_encodings True```

Again, this will create the folder `exemplar_VideoMAEtokens_RepCount` with repetition tokens for each video. For each video, the shape will be `N x 3 x 8 x 14 x 14`, where N is the number of repetitions in the video.

# Train ESCounts