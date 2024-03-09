This is the codebase for the paper "Every Shot Counts: Using Exemplars for Repetition Counting in Videos"


# Extract VideoMAE encodings

We use a pretrained VideoMAE-v2 encoder to extract spatio-temporal tokens from videos.

Download the pretrained encoder weights from [here](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth) and put it in `pretrained_models/`

 First extract video-wise
 encodings from the pretrained VideoMAE encoder
`````` 