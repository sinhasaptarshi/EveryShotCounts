import torch
import numpy as np 
from mmaction.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from Rep_count import Rep_count
from Countix import Countix
from UCF_Rep import UCFRep

from video_mae_cross_full_attention import SupervisedMAE
from slowfast.utils.parser import load_config
import argparse
from resnext_models import resnext
import pdb
import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_gpus', default=4, type=int)
    parser.add_argument('--pretrained_encoder', default ='pretrained_models/VIT_B_16x4_MAE_PT.pyth', type=str)
    parser.add_argument('--save_exemplar_encodings', default=False, type=bool)
    parser.add_argument('--dataset', default='Repcount', help='choose from [Repcount, Countix, UCFRep]', type=str)
    parser.add_argument('--model', default='VideoMAE', help="VideoMAE, VideoSwin")
    parser.add_argument('--encodings', default='mae', help="mae, swin, resnext")
    parser.add_argument('--data_path', default='', help='data path for the dataset')
    parser.add_argument('--use_v1', action='store_true', help='use the v1 variant of the encoder')
    return parser

def save_exemplar(dataloaders, model, args):
    '''
     This function extracts the encodings for every repetition in each video by uniformly sampling 16 frames
     within the repetition segments and saves these encodings as npz format. The input to the encoder is 
     B*3xTxHxW, where B is the total number of repetitions in the selected video. The output is spatio-temporal 
     tokens of shape Bx(T'H'W')xC. We save these encodigns as BxCxT'xH'xW'.
     inputs: a dict consisting of 'train', 'val' and 'test' dataloaders, 
             the pretrained model,
             other parameters needed 
    '''

    if args.model == '3D-ResNeXt101':
        num_frames = 64
    else:
        num_frames = 16
    if args.dataset == 'UCFRep': ###UCFRep has train and val splits only
        splits = ['train', 'val']
    else:
        splits = ['train', 'val', 'test']
    target_dir = f'exemplar_{args.model}tokens_{args.dataset}'
    for split in splits:
        for item in tqdm.tqdm(dataloaders[split],total=len(dataloaders[split])):
            video = item[0].squeeze(0)
            starts = item[-3][0]
            ends = item[-2][0]
            video_name = item[-1][0]
            C, T, H, W = video.shape
            

            clip_list = []
            num_exemplars = len(starts)
            for j in range(num_exemplars):
                s = starts[j].item()  ## start times of each repetition
                e = ends[j].item()  ## end times of each repetition
                if s==e:
                    continue
                idx = np.linspace(s, min(e, video.shape[1]-1), num_frames+1)[:num_frames].astype(int) ###sample 16 frames from the repetition segment defined by the start and end
                clips = video[:, idx]
                clip_list.append(clips)
            data = torch.stack(clip_list).cuda()  ### batch of repetitions
            with torch.no_grad():
                if args.model == 'VideoMAE':
                    encoded, thw = model(data) ## extract encodings
                    encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2]) # reshape to B x C x T x H x W
                else:
                    encoded = model(data)

            
            enc_np = encoded.cpu().numpy()
            del encoded, data
            torch.cuda.empty_cache()
            
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
                
            np.savez('{}/{}.npz'.format(target_dir, video_name), enc_np) ##save as npz

def save_tokens(dataloaders, model, args):
    '''
     This function extracts the encodings for each video using windows of 64 frames and then sampling 16 frames uniformly
      from these windows. We save the encodings npz format. The input to the encoder is B*3x16xHxW, where B is the batch size 
      and each batch comprises of overlapping windows in each videp. The output is spatio-temporal tokens of shape Bx(T'H'W')xC. We 
     save these encodigns as BxCxT'xH'xW'.
     inputs: a dict consisting of 'train', 'val' and 'test' dataloaders, 
             the pretrained model,
             other parameters needed 
    '''

    if args.model == '3D-ResNeXt101':
        num_frames = 64
    else:
        num_frames = 16
    if args.dataset == 'UCFRep': ### UCFRep has train and val splits only
        splits = ['train', 'val']
    else:
        splits = ['train', 'val', 'test']
    
    target_dir = f'saved_{args.model}tokens_{args.dataset}'

    
    for split in splits:
        for item in tqdm.tqdm(dataloaders[split],total=len(dataloaders[split])):
            video = item[0].squeeze(0)
            video_name = item[-1][0]
            C, T, H, W = video.shape
            padding = torch.zeros([C, 64, H, W]) ### add padding of zeros at the end
            video = torch.cat([video, padding], 1)

            clip_list = []
            n_frames = T
            for j in range(0, T, 16): #### 75% overlap
                idx = np.linspace(j, j+64, num_frames+1)[:num_frames].astype(int) ### sample 16 frames from windows of 64 frames
                clips = video[:,idx]
                clip_list.append(clips)
            data = torch.stack(clip_list).cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    if args.model == 'VideoMAE':
                        encoded, thw = model(data)  ### extract encodings
                        encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2]) # reshape to B x C x T x H x W
                    else:
                        encoded = model(data)
            enc_np = encoded.cpu().numpy()
            del encoded, data
            torch.cuda.empty_cache()
            
            if not os.path.isdir(target_dir):
                print('Creating folder')
                os.makedirs(target_dir)
            
            np.savez('{}/{}.npz'.format(target_dir, video_name), enc_np) ### saving as npz


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None
    args.save_video_encodings = not args.save_exemplar_encodings
    if args.use_v1:
        cfg = load_config(args, path_to_config='configs/pretrain_config_v1.yaml')
    else:
        cfg = load_config(args, path_to_config='configs/pretrain_config.yaml')
    if args.model == 'VideoMAE': ### for videomae-based encoder (recommended)
        
        model = SupervisedMAE(cfg=cfg, just_encode=True, use_precomputed=False, encodings=args.encodings).cuda()
        if args.pretrained_encoder:
            state_dict = torch.load(args.pretrained_encoder)
            if 'model_state' in state_dict.keys():
                state_dict = state_dict['model_state']
            else:
                state_dict = state_dict['model']
        else:
            state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth')['model_state']   ##pretrained on Kinetics
        # print(model)
        
        model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
        # pdb.set_trace()
        # print(model)
        # print(state_dict.keys())
        for name in model.state_dict().keys():
            if 'decoder' in name or 'decode_heads' in name:
                continue

                
            matched = 0

            for name_, param in state_dict.items():
                # if args.num_gpus > 1:
                if 'encoder.' in name_:
                    name_ = name_.replace('encoder.', '')
                name_ = f'module.{name_}'


                # pdb.set_trace()
                if name_ == name:

                    model.state_dict()[name].copy_(param)
                    matched = 1
                    break
            if matched == 0 and '.qkv.' in name:
                if not args.use_v1:
                    q_name = name.replace('.qkv.', '.q.').replace('module.', '')
                    k_name = name.replace('.qkv.', '.k.').replace('module.', '')
                    v_name = name.replace('.qkv.', '.v.').replace('module.', '')
                    params = torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]])
                    model.state_dict()[name].copy_(params)
                    matched = 1
                    break
                else:
                    if '.qkv.bias' in name:
                        q_name = name.replace('.qkv.', '.q_').replace('module.', 'encoder.')
                        v_name = name.replace('.qkv.', '.v_').replace('module.', 'encoder.')
                        params = torch.cat([state_dict[q_name], torch.zeros_like(state_dict[v_name], requires_grad=False), state_dict[v_name]])
                        model.state_dict()[name].copy_(params)
                        matched = 1
                        break
            if matched == 0:
                print(f"parameters {name} not found")

    elif args.model == 'VideoSwin':   ###for swin-based encoder
        config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
        checkpoint = './pretrained_models/swin_tiny_patch244_window877_kinetics400_1k.pth'
        model_cfg = Config.fromfile(config)
        model = build_model(model_cfg.model, train_cfg=model_cfg.get('train_cfg'), test_cfg=model_cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')

        backbone = model.backbone
        backbone = backbone.cuda()
        model = nn.DataParallel(backbone, device_ids=[i for i in range(args.num_gpus)])

    elif args.model == '3D-ResNeXt101':  ## for resnext based encoder
        model = resnext.resnet101(num_classes=400, sample_size=224, sample_duration=64, last_fc=False)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
        model.load_state_dict(torch.load('pretrained_models/resnext-101-64f-kinetics.pth')['state_dict'])
        

    model.eval()
    if args.dataset == 'RepCount':
        dataset_train = Rep_count(cfg=cfg,split="train",data_dir=args.data_path,sampling_interval=1,encode_only=True)
        dataset_val = Rep_count(cfg=cfg,split="valid",data_dir=args.data_path,sampling_interval=1,encode_only=True)
        dataset_test = Rep_count(cfg=cfg,split="test",data_dir=args.data_path,sampling_interval=1,encode_only=True)
    elif args.dataset == 'Countix':
        dataset_train = Countix(cfg=cfg,split="train",sampling_interval=1,encode_only=True)
        dataset_val = Countix(cfg=cfg,split="val",sampling_interval=1,encode_only=True)
        dataset_test = Countix(cfg=cfg,split="test",sampling_interval=1,encode_only=True)
    elif args.dataset == 'UCFRep':
        dataset_train = UCFRep(cfg=cfg,split="train",sampling_interval=1,encode_only=True)
        dataset_val = UCFRep(cfg=cfg,split="val",sampling_interval=1,encode_only=True)
        dataset_test = UCFRep(cfg=cfg,split="val",sampling_interval=1,encode_only=True)

    dataloaders = {'train':torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size,
                                                       num_workers=8,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=False),
                   'val':torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=args.batch_size,
                                                     num_workers=8,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     drop_last=False),
                   'test':torch.utils.data.DataLoader(dataset_test,
                                                     batch_size=args.batch_size,
                                                     num_workers=8,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     drop_last=False)}
    
    if args.save_video_encodings:
        save_tokens(dataloaders, model, args)
    elif args.save_exemplar_encodings:
        save_exemplar(dataloaders, model, args)

    

if __name__ == '__main__':
    main()
    

