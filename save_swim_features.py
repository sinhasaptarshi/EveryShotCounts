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

from video_mae_cross import SupervisedMAE
from slowfast.utils.parser import load_config
import argparse
from resnext_models import resnext

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
    return parser

def save_exemplar(dataloaders, model, args):
    if args.model == '3D-ResNeXt101':
        num_frames = 64
    else:
        num_frames = 16
    if args.dataset == 'UCFRep':
        splits = ['train', 'val']
    elif args.dataset == 'Quva':
        splits = ['test']
    else:
        splits = ['train', 'test']
    target_dir = f'exemplar_{args.model}tokens_{args.dataset}'
    for split in splits:
        for item in tqdm.tqdm(dataloaders[split],total=len(dataloaders[split])):
            video = item[0].squeeze(0)
            # print(video.shape)
            # print(video.max())
            # print(video.min())
            # print(video.mean())
            starts = item[-3][0]
            ends = item[-2][0]
            # print(starts)
            video_name = item[-1][0]
            print(video_name)
            # if os.path.exists('exemplar_maetokens_repcount/{}.npz'.format(video_name)):
            #     continue
            C, T, H, W = video.shape
            

            clip_list = []
            num_exemplars = len(starts)
            # print(num_exemplars)
            # if split == 'train':
            for j in range(num_exemplars):
                # peak = (starts[j].item() + ends[j].item())//2
                # length = (ends[j].item() - starts[j].item())
                # s, e = peak - length // 4, peak + length // 4
                s = starts[j].item()
                e = ends[j].item()
                if s==e:
                    continue
                idx = np.linspace(s, min(e, video.shape[1]-1), num_frames+1)[:num_frames].astype(int)
                clips = video[:, idx]
                clip_list.append(clips)
            # else:
            #     idx = np.linspace(starts[0].item(), ends[0].item(), 16)
            #     clips = video[:, idx]
            #     clip_list.append(clips)
            
            data = torch.stack(clip_list).cuda()
            # print(data.shape)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    '''
                    if data.shape[0] > 64:
                        encoded, thw = [], []
                        ids = [ix for ix in range(64,data.shape[0],64)]
                        if data.shape[0] - ids[-1] > 0:
                            ids.append(data.shape[0])
                        ids_s = [ix for ix in range(0,data.shape[0],64)]
                        for id_s,id_f in zip(ids_s,ids):
                            enc,thw_ = model(data[id_s:id_f,...])
                            encoded.append(enc)
                            thw.append(thw_)
                        encoded =  torch.cat(encoded,dim=0)
                        thw = thw[0]
                    else:
                    '''
                    if args.model == 'VideoMAE':
                        encoded, thw = model(data)
                        encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2]) # reshape to B x C x T x H x W
                    else:
                        encoded = model(data)
                        # thw = encoded.shape[-3:]
                    
            # print(encoded.shape)
            
            enc_np = encoded.cpu().numpy()
            del encoded, data
            torch.cuda.empty_cache()
            
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
                
            np.savez('{}/{}.npz'.format(target_dir, video_name), enc_np)

def save_tokens(dataloaders, model, args):
    if args.model == '3D-ResNeXt101':
        num_frames = 64
    else:
        num_frames = 16
    if args.dataset == 'UCFRep':
        splits = ['train', 'val']
    elif args.dataset == 'Quva':
        splits = ['test']
    else:
        splits = ['test']
    
    target_dir = f'saved_{args.model}tokens_{args.dataset}'

    
    for split in splits:
        for item in tqdm.tqdm(dataloaders[split],total=len(dataloaders[split])):
            video = item[0].squeeze(0)
            video_name = item[-1][0]
            print(video_name)
            # if os.path.exists('saved_maetokens_repcount/{}.npz'.format(video_name)):
            #     continue
            # if video_name != 'stu8_4.mp4':
            #     pass
            # print(video_name)
            C, T, H, W = video.shape
            padding = torch.zeros([C, 64, H, W])
            video = torch.cat([video, padding], 1)

            clip_list = []
            n_frames = T
            for j in range(0, T, 16): #### 75% overlap
                idx = np.linspace(j, j+64, num_frames+1)[:num_frames].astype(int)
                clips = video[:,idx]
                clip_list.append(clips)
            data = torch.stack(clip_list).cuda()
            # print(data)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    if args.model == 'VideoMAE':
                        encoded, thw = model(data)
                        encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2]) # reshape to B x C x T x H x W
                    else:
                        encoded = model(data)
                        # print(encoded.shape)
                        # thw = encoded.shape[-3:]
                    
            # print(encoded.shape)
            enc_np = encoded.cpu().numpy()
            del encoded, data
            torch.cuda.empty_cache()
            
            if not os.path.isdir(target_dir):
                print('Creating folder')
                os.makedirs(target_dir)
            
            np.savez('{}/{}.npz'.format(target_dir, video_name), enc_np)


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None
    args.save_video_encodings = not args.save_exemplar_encodings
    print(args.save_exemplar_encodings)
    print(args.save_video_encodings)
    args.data_path = '/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP'
    
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    if args.model == 'VideoMAE':
        
        model = SupervisedMAE(cfg=cfg, just_encode=True, use_precomputed=False, encodings=args.encodings).cuda()
        # if args.pretrained_encoder:
        state_dict = torch.load(args.pretrained_encoder)['model_state']
        #state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth')['model_state']
        model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
        # for name, param in state_dict.items():
        # #     # print(name)
        #     if args.num_gpus > 1:
        #         name = f'module.{name}'
        #     if name in model.state_dict().keys():
        #         continue
        #     else:
        #         print(name)
        #         # if 'decoder' not in name:
        #             print(name)
        #             # new_name = name.replace('quantizer.', '')
        #             try:
        #                 model.state_dict()[name].copy_(param)
        #             except:
        #                 print(f"parameters {name} not found")
        print(state_dict.keys())
        for name in model.state_dict().keys():
            if 'decoder' in name:
                continue
            matched = 0

            for name_, param in state_dict.items():
                if args.num_gpus > 1:
                    name_ = f'module.{name_}'
                if name_ == name:
                    model.state_dict()[name].copy_(param)
                    matched = 1
                    break
                elif '.qkv.' in name:
                    q_name = name.replace('.qkv.', '.q.').replace('module.', '')
                    k_name = name.replace('.qkv.', '.k.').replace('module.', '')
                    v_name = name.replace('.qkv.', '.v.').replace('module.', '')
                    params = torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]])
                    model.state_dict()[name].copy_(params)
                    matched = 1
                    break
            if matched == 0:
                print(f"parameters {name} not found")
            # else:
            #     print(f"parameters {name} found")

    elif args.model == 'VideoSwin':
        config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
        checkpoint = './pretrained_models/swin_tiny_patch244_window877_kinetics400_1k.pth'
        model_cfg = Config.fromfile(config)
        model = build_model(model_cfg.model, train_cfg=model_cfg.get('train_cfg'), test_cfg=model_cfg.get('test_cfg'))
        # model = nn.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
        # print(model.backbone.layers[0].blocks[0].attn.qkv.weight.data)

        load_checkpoint(model, checkpoint, map_location='cpu')
        # print(model.backbone.layers[0].blocks[0].attn.qkv.weight.data)

        backbone = model.backbone
        backbone = backbone.cuda()
        model = nn.DataParallel(backbone, device_ids=[i for i in range(args.num_gpus)])
        # print(model.module.layers[0].blocks[0].attn.qkv.weight.data)

    elif args.model == '3D-ResNeXt101':
        model = resnext.resnet101(num_classes=400, sample_size=224, sample_duration=64, last_fc=False)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
        model.load_state_dict(torch.load('pretrained_models/resnext-101-64f-kinetics.pth')['state_dict'])
        
        # model.load_state_dict(torch.load('results/0208-23:40_2stream/save_62.pth')['state_dict'], strict=False)
        

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
    















# backbone.eval()
# x = torch.rand(1, 3, 16, 224, 224).cuda()
# y = backbone(x)
# print(y.shape)

