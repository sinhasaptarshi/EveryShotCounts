This is the codebase for the paper "Every Shot Counts: Using Exemplars for Repetition Counting in Videos"


# Extract VideoMAE encodings

We use a pretrained VideoMAE-v2 encoder to extract spatio-temporal tokens from videos.

Download the pretrained encoder weights from [here](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth) and put it in `pretrained_models/`

Extract spatio-temporal tokens for each video using

```python save_swin_features.py --dataset RepCount --model VideoMAE --num_gpus 1```