import pathlib
from random import randint
import torch.utils.data
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import einops

class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 add_noise= False,
                 num_frames=512,
                 tokens_dir = "saved_tokens",
                 exemplar_dir = "exemplar_tokens",
                 density_maps_dir = "gt_density_maps"):
        
        self.num_frames=num_frames
        self.tokens_dir = tokens_dir
        self.exemplar_dir = exemplar_dir
        self.density_maps_dir = density_maps_dir
        self.split = split # set the split to load
        self.add_noise = add_noise # add noise to frames (augmentation)
        if self.split == 'train':
            csv_path = f"datasets/repcount/{self.split}_with_fps.csv"
        else:
            csv_path = f"datasets/repcount/validtest_with_fps.csv"
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['count'].notna()]
        # self.df = self.df[self.df['count'] < 5] ### remove videos with more than 5 repetitions
        # self.df = self.df[self.df['fps'] >= 10]
        self.df = self.df[self.df['num_frames'] > 64]
        self.df = self.df.drop(self.df.loc[self.df['name']=='stu1_10.mp4'].index)
        self.df = self.df[self.df['count'] > 0] # remove no reps
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )
    
    
        
    def load_tokens(self,path,is_exemplar):
        tokens = np.load(path)['arr_0'] # Load in format C x t x h x w
        if is_exemplar:
            N = tokens.shape[0]
            if self.split == 'train':
                idx = np.random.randint(N)
            else:
                idx = 0
            return torch.from_numpy(tokens[idx:idx+1]) ### return the encoding for a selected example per video instance
        else:
            tokens = tokens[0::4] # non overlapping segments
        return tokens


    def load_density_map(self,path,count):
        gt_density_map = np.load(path)['arr_0'][0::4]
        return gt_density_map/gt_density_map.sum() * count  ##scale by count to make the sum consistent
      
      
    
    def __getitem__(self, index):
        video_name = self.df.iloc[index]['name'].replace('.mp4', '.npz')
        row = self.df.iloc[index]
        mean_duration = 512
        clc = np.array([int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])])
        starts = clc[0::2]
        ends = clc[1::2]
        
        # --- Exemplar tokens loading ---
        examplar_path = f"{self.exemplar_dir}/{self.split}/{video_name}"
        example_rep = self.load_tokens(examplar_path,True) 

        # --- Density map loading ---
        density_map_path = f"{self.density_maps_dir}/{video_name}"
        gt_density = self.load_density_map(density_map_path,row['count'])  
        
        # --- Video tokens loading ---
        video_path = f"{self.tokens_dir}/{self.split}/{video_name}"
        vid_tokens = self.load_tokens(video_path,False) 
        
        T = row['num_frames'] ### number of frames in the video
        if T <= self.num_frames:
            start, end = 0, T
        else:
            start = random.choice(np.arange(0, T-self.num_frames, 64))
            end = start + self.num_frames  ## for taking 8 segments

        sampled_segments = torch.from_numpy(vid_tokens[(start//64) : (end//64)])
        sampled_segments = einops.rearrange(sampled_segments, 'S C t h w -> (S t h w) C')
        #n, c, t, h, w = sampled_segments.shape
        #sampled_segments = sampled_segments.permute(0, 2, 3, 4, 1).reshape(-1, c)
        # sampled_segment = torch.stack(sampled_segments)
        gt = gt_density[(start//4): (end//4)]
        # print(gt.sum())

        return sampled_segments, example_rep, gt, gt.sum(), self.df.iloc[index]['name'][:-4]
        

    def __len__(self):
        return len(self.df)


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = Rep_count()
    print('--- dataset created ---')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,batch_size=1,num_workers=10,shuffle=False,pin_memory=False,drop_last=True)
    
    sum_clip_dur = []
    sum_tot_dur = []
    sum_clip_counts = []
    sum_tot_counts = []
    
    density_maps_sum = {}
    counts = {}
    
    fps = []
    
    for i, item in enumerate(tqdm(dataloader)):
        print(f"It. {i} | vid tokens: {item[0].shape} | exem tokens: {item[1].shape} | density map: {item[2].shape}:{item[3]}")
        #if int(item[3].item())!=int(item[5].item()):
        #    print(item[3].item(),int(item[5].item()))
        #if int(item[3].item()) not in density_maps_sum.keys():
        #    density_maps_sum[int(item[3].item())] = 1
        #else:
        #    density_maps_sum[int(item[3].item())] += 1
            
        #if int(item[5]) not in counts.keys():
        #    counts[int(item[5])] = 1
        #else:
        #    counts[int(item[5])] += 1
        #sum_clip_dur.append(item[0])
        #sum_tot_dur.append(item[1])
        #sum_clip_counts.append(item[2])
        #sum_tot_counts.append(item[3])
        # print(sum(sum_tot_counts)/len(sum_tot_counts))
        
        #fps.append(item[4])
        #print(item[0].shape)
        # print(i, item[1].shape)
        # print(i, item[2].shape)
        # print(item[2])
    
    #for i in range(7):
    #    if i in counts.keys() and i in density_maps_sum.keys():
    #        print(i,counts[i],density_maps_sum[i])

    # print(f"Avg clip dur: {sum(sum_clip_dur)/len(sum_clip_dur)} | Avg vid dur: {sum(sum_tot_dur)/len(sum_tot_dur)}")
    # print(f"Avg clip reps: {sum(sum_clip_counts)/len(sum_clip_counts)} | Avg vid counts: {sum(sum_tot_counts)/len(sum_tot_counts)}")
    # print(sum(fps)/len(fps))
    
    
        
    
