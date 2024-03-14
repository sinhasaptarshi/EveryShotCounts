import torch
import torch.nn as nn
torch.cuda.empty_cache()
import numpy as np
import os, sys
import einops
import cv2
from video_mae_cross_full_attention import SupervisedMAE
from slowfast.utils.parser import load_config
import argparse
import av
import math
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import create_video_transform
from itertools import cycle, islice
import gdown

def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=False)
    parser.add_argument('--video_name', default='data/stu2_39.mp4', type=str,help='Demo video to infer on')
    parser.add_argument('--resource', default='cpu', type=str, help='choose compute resource to use, e.g `cpu`,`cuda:0`,etc')
    parser.add_argument('--pool_tokens', default=0.4, type=float)
    parser.add_argument('--pretrained_encoder', default =None, type=str)
    parser.add_argument('--scale_counts', default=100)
    parser.add_argument('--dataset', default='Repcount', help='choose from [Repcount, Countix, UCFRep]', type=str)
    parser.add_argument('--model', default='VideoMAE', help="VideoMAE, VideoSwin")
    parser.add_argument('--encodings', default='mae', help="mae, swin, resnext")
    return parser

def extract_tokens(video, model, args, num_frames=16):
    C, T, H, W = video.shape
    padding = torch.zeros([C, 64, H, W])
    video = torch.cat([video, padding], 1)
    clip_list = []
    for j in range(0, T, 16):
        idx = np.linspace(j, j+64, num_frames+1)[:num_frames].astype(int)
        clips = video[:, idx]
        clip_list.append(clips)
    data = torch.stack(clip_list).to(args.resource)
    dtype = 'cuda' if 'cuda' in args.resource else 'cpu'
    with torch.autocast(enabled='cuda' in args.resource, device_type=dtype):
        with torch.no_grad():
            encoded, thw = model(data)  ## extract encodings
            encoded = encoded.transpose(1,2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2])
    del data
    return encoded

def read_video_timestamps(video_filename, timestamps,  duration=0):
    """ 
    summary

    Args:
        video_filename (string): full filepath of the video
        timestamps (list): list of ints for the temporal points to load from the file

    Returns:
        frames: tensor of shape C x T x H x W
        totfps: float for the video segment length (in secs)
    """
    try:
        assert os.path.isfile(video_filename), f"VideoLoader: {video_filename} does not exist"
    except:
        print(f"{video_filename} does not exist")
    
    frames = []

    container = av.open(video_filename)
    
    min_t = min(timestamps)
    max_t = max(timestamps)
    
    for i, f in enumerate(islice(container.decode(video=0), min_t, max_t+1)):
        c = i + min_t
        if c in timestamps:
            for _ in range(timestamps.tolist().count(c)): # for multiple occurrences
                frames.append(f)
        
    video_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in frames] # list of length T with items size [H x W x 3] 
    video_frames = thwc_to_cthw(torch.stack(video_frames).to(torch.float32))

    container.close()
    return video_frames, timestamps[-1]





def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None
    video = args.video_name
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    transform =  create_video_transform(mode="test",
                                        convert_to_float=False,
                                        min_size = 224,
                                        crop_size = 224,
                                        num_samples = None,
                                        video_mean = [0.485,0.456,0.406], 
                                        video_std = [0.229,0.224,0.225])
    frame_idx = np.arange(0, num_frames, 1)

    ### read frames from video
    vid_frames, _ = read_video_timestamps(video, frame_idx)
    vid_frames = transform(vid_frames/255.)

    ### load encoder checkpoint pretrained on Kinetics
    
    cfg = load_config(args, path_to_config='configs/pretrain_config.yaml')
    encoder = SupervisedMAE(cfg=cfg, just_encode=True, use_precomputed=False, encodings=args.encodings)
    encoder = encoder.to(args.resource)
    if args.pretrained_encoder:
        state_dict = torch.load(args.pretrained_encoder)['model_state']
    else:
        state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth', map_location=args.resource)['model_state']   ##pretrained on Kinetics
    
    for name in encoder.state_dict().keys():
            if 'decode' in name:
                continue
            matched = 0

            for name_, param in state_dict.items():
                if name_ == name:
                    encoder.state_dict()[name].copy_(param)
                    matched = 1
                    break
                elif '.qkv.' in name and 'blocks' in name:
                    q_name = name.replace('.qkv.', '.q.').replace('module.', '')
                    k_name = name.replace('.qkv.', '.k.').replace('module.', '')
                    v_name = name.replace('.qkv.', '.v.').replace('module.', '')
                    params = torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]])
                    encoder.state_dict()[name].copy_(params)
                    matched = 1
                    break
            # if matched == 0:
            #     print(f"parameters {name} not found")
    
    ### encode video
    encoder.eval()
    encoded = extract_tokens(vid_frames, encoder, args)
    encoded = encoded[0::4]
    encoded = einops.rearrange(encoded, 'S C T H W -> C (S T) H W')
    del encoder, state_dict
    if args.pool_tokens < 1.0:
        factor = math.ceil(encoded.shape[-1] * args.pool_tokens)
        tokens = torch.nn.functional.adaptive_avg_pool3d(encoded, (encoded.shape[-3], factor, factor))

    decoder = SupervisedMAE(cfg=cfg,use_precomputed=True, token_pool_ratio=0.4, iterative_shots=True, encodings='mae', no_exemplars=False, window_size=(4,7,7))
    decoder = decoder.to(args.resource)
    
    print('---- loading pretrained model ----')
    output = 'repcount_trained.pyth'
    if not os.path.isfile(output):
        if args.model=='VideoMAE':
            url = 'https://drive.google.com/uc?id=1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48'
        else:
            url = 'https://drive.google.com/uc?id=1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48'
        gdown.download(url, output, quiet=False)
    decoder.load_state_dict(torch.load(output, map_location=args.resource)['model_state_dict'])

    ##placeholder exemplar
    tokens = tokens.unsqueeze(0)
    shapes = tokens.shape[-3:]
    tokens = einops.rearrange(tokens, 'B C T H W -> B (T H W) C')
    # print(tokens.shape)
    dtype = 'cuda' if 'cuda' in args.resource else 'cpu'
    with torch.autocast(enabled='cuda' in args.resource, device_type=dtype):
        with torch.no_grad():
            predicted_density_map = decoder(tokens, thw=[shapes,], shot_num=0)
    # print(pred.shape)
    predicted_counts = predicted_density_map.sum().item()/args.scale_counts
    print(f'The number of repetitions is {round(predicted_counts)}')
    


if __name__ == '__main__':
    main()

