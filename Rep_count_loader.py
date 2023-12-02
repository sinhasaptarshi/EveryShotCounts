import pathlib
from random import randint
import torch.utils.data
import os, sys, math
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
                 density_maps_dir = "gt_density_maps",
                 select_rand_segment=True,
                 compact=False,
                 pool_tokens_factor=1.0):
        
        self.num_frames=num_frames
        self.tokens_dir = tokens_dir
        self.exemplar_dir = exemplar_dir
        self.density_maps_dir = density_maps_dir
        self.compact = compact
        self.select_rand_segment = select_rand_segment
        self.pool_tokens = pool_tokens_factor
        self.split = split # set the split to load
        self.add_noise = add_noise # add noise to frames (augmentation)
        if self.split == 'train':
            csv_path = f"datasets/repcount/{self.split}_balanced_new.csv"
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
    
    
        
    def load_tokens(self,path,is_exemplar,bounds=None, lim_constraint=20):
        tokens = np.load(path)['arr_0'] # Load in format C x t x h x w
            
        if is_exemplar:
            N = tokens.shape[0]
            if self.select_rand_segment:
                idx = np.random.randint(N)
            else:
                idx = 0
            tokens = tokens[idx:idx+1] ### return the encoding for a selected example per video instance
        else:
            if bounds is not None:
                low_bound = bounds[0]//64
                up_bound = bounds[1]//64
            # else:
            #     low_bound = 0
            #     up_bound = None
            
            # print(tokens.shape[0])
            tokens = tokens[0::4] # non overlapping segments
            tokens = tokens[low_bound:min(up_bound, low_bound+lim_constraint)] ## non overlapping segments
                
        
        tokens = torch.from_numpy(tokens)
        if self.pool_tokens < 1.0:
            factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
            tokens = torch.nn.functional.adaptive_avg_pool3d(tokens, (tokens.shape[-3], factor, factor))
            
        tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
        
        # if bounds is not None:
        #     start = bounds[0] // 8 ## Sampling every 4 frames and MViT temporally downsample T=16 -> 8 
        #     end = bounds[1] // 8
        #     tokens = tokens[:,start:end,:,:]
            
        return tokens


    def load_density_map(self,path,count, bound, lim_constraint=20):
        gt_density_map = np.load(path)['arr_0']#[0::4]
        low = bound[0] // 64
        up = bound[1] //64
        # gt_density_map = gt_density_map/gt_density_map.sum() * count 
        gt_density_map = gt_density_map[(low * 64):(min(up, low + lim_constraint)  * 64)]
        # return gt_density_map
        return  gt_density_map##scale by count to make the sum consistent
      
      
    
    def __getitem__(self, index):
        video_name = self.df.iloc[index]['name'].replace('.mp4', '.npz')
        row = self.df.iloc[index]
        
        segment_start = row['segment_start']
        segment_end = row['segment_end']        
        
        # --- Exemplar tokens loading ---
        # examplar_path = f"{self.exemplar_dir}/{self.split}/{video_name}"
        examplar_path = f"{self.exemplar_dir}/{video_name}"
        example_rep = self.load_tokens(examplar_path,True) 

        # --- Density map loading ---
        density_map_path = f"{self.density_maps_dir}/{video_name}"
        gt_density = self.load_density_map(density_map_path,row['count'],(segment_start,segment_end), lim_constraint=20)  
        # gt_density = gt_density[segment_start:(segment_end//64 * 64)]
        
        # --- Video tokens loading ---
        # video_path = f"{self.tokens_dir}/{self.split}/{video_name}"
        video_path = f"{self.tokens_dir}/{video_name}"
        vid_tokens = self.load_tokens(video_path,False, (segment_start,segment_end), lim_constraint=20) ###lim_constraint for memory issues
        
        if not self.select_rand_segment:
            vid_tokens = vid_tokens
            gt_density = torch.from_numpy(gt_density)
            
            return vid_tokens, example_rep, gt_density, gt_density.sum(), self.df.iloc[index]['name'][:-4] 
        
        T = row['num_frames'] ### number of frames in the video
        if T <= self.num_frames:
            start, end = 0, T
        else:
            start = random.choice(np.arange(0, T-self.num_frames, 64))
            end = start + self.num_frames  ## for taking 8 segments

        sampled_segments = vid_tokens[(start//64) : (end//64)]
        sampled_segments = einops.rearrange(sampled_segments, 'C t h w -> (t h w) C')
        #n, c, t, h, w = sampled_segments.shape
        #sampled_segments = sampled_segments.permute(0, 2, 3, 4, 1).reshape(-1, c)
        # sampled_segment = torch.stack(sampled_segments)
        gt = gt_density[(start//4): (end//4)]
        # print(gt.sum())

        return sampled_segments, example_rep, gt, gt.sum(), self.df.iloc[index]['name'][:-4]
        

    def __len__(self):
        return len(self.df)


    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        
        # [1 x T1 x .... ], [1 x T2 x ....] => [2 x T2 x ....] (T2 > T1)    
        vids = pad_sequence([einops.rearrange(x[0],'C T H W -> T C H W') for x in batch])
        if self.compact:
            vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
        else:
            vids = einops.rearrange(vids, 'T B C H W -> B C T H W')
        exemplars = torch.stack([x[1] for x in batch]).squeeze(1)
        if self.compact:
            exemplars = einops.rearrange(exemplars,'B C T H W -> B (T H W) C')
        gt_density = einops.rearrange(pad_sequence([x[2] for x in batch]), 'S B -> B S')
        gt_density_sum =  torch.tensor([x[3] for x in batch], dtype=torch.float)
        names = [x[4] for x in batch]
        
        # return padded video, exemplar, padded density map,
        return vids, exemplars, gt_density, gt_density_sum, names


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = Rep_count(select_rand_segment=False, compact=False, pool_tokens_factor=0.5)
    print('--- dataset created ---')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,
                                             batch_size=8,
                                             num_workers=10,
                                             shuffle=False,
                                             pin_memory=False,
                                             drop_last=True,
                                             collate_fn=dat.collate_fn)
    sum_clip_dur = []
    sum_tot_dur = []
    sum_clip_counts = []
    sum_tot_counts = []
    
    density_maps_sum = {}
    counts = {}
    
    fps = []
    
    for i, item in enumerate(tqdm(dataloader)):
        print(f"It. {i} \n vid tokens: {item[0].shape} \n exem tokens: {item[1].shape} \n density map: {item[2].shape}:{item[3]} \n \n")
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
    
    
        
    
