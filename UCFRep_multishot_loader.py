import pathlib
from random import randint
import torch.utils.data
import os, sys, math
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy import integrate
from scipy import ndimage

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import einops

class UCFRep(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 add_noise= False,
                 num_frames=512,
                 tokens_dir = "saved_VideoMAEtokens_UCFRep",
                 exemplar_dir = "exemplar_VideoMAEtokens_UCFRep",
                 density_maps_dir = "gt_density_maps_recreated",
                 select_rand_segment=True,
                 compact=False,
                 lim_constraint=np.inf,
                 pool_tokens_factor=1.0,
                 peak_at_random_location=False,
                 get_overlapping_segments=False,
                 multishot=True,
                 density_peak_width=1.0,
                 threshold=0.0):
        
        self.num_frames=num_frames
        self.lim_constraint = lim_constraint
        self.tokens_dir = tokens_dir
        self.exemplar_dir = exemplar_dir
        self.density_maps_dir = density_maps_dir
        self.compact = compact
        self.select_rand_segment = select_rand_segment
        self.pool_tokens = pool_tokens_factor
        self.split = split # set the split to load
        self.add_noise = add_noise # add noise to frames (augmentation)
        self.peak_at_random_location = peak_at_random_location
        self.get_overlapping_segments = get_overlapping_segments
        self.multishot = multishot
        self.threshold = threshold
        self.density_peak_width = density_peak_width
        self.temporal_downsample = 16 if '3D-ResNeXt101' in self.tokens_dir else 8
        if self.split == 'train':
            csv_path = f"datasets/ucf-rep/new_train.csv"
        else:
            csv_path = f"datasets/ucf-rep/new_val.csv"
        self.df = pd.read_csv(csv_path)
        self.df['density_map_sum'] = 0
        self.df['type'] = self.df['name'].apply(lambda x: x.split('/')[-2])
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )
    
        
    def load_tokens(self,path,is_exemplar,bounds=None, lim_constraint=np.inf, id=None, cycle_start_id=0, count=None, shot_num=1, get_overlapping_segments=False, segment_id=0):
        """
        loading video or exemplar tokens. 
        input: path -> the path for the saved video/exemplar tokens
               is_exemplar -> True/False for encoding exemplar tokens or not.
               bounds -> (st, end) to trim video given the start and end timestamps. 
               lim_constraint -> for memory issues, lim_constraint trims the video till this value. 
               shot_num = (1,2,3) how many exemplar tokens to return

        output:
               video/exemplar tokens
        """
        
        try:
            tokens = np.load(path)['arr_0'] # Load in format C x t x h x w
        except:
            print(f'Could not load {path}')
            

            
        if is_exemplar:
            if shot_num == 0:
                shot_num = 1
            if count is not None:
                if count > 0.6:
                    N = round(count)
                else:
                    N = 1
            else:
                N = tokens.shape[0]
            if self.select_rand_segment or self.split == 'train':
                # print(N)
                shot_num = min(shot_num, N)
                
                idx = cycle_start_id + np.random.choice(np.arange(N), size=shot_num, replace=False)   ### select random exemplars from the video
            else:
                shot_num = min(shot_num, N)
                if shot_num == 1:
                    idx = [N//2]
                elif shot_num == 2:
                    idx = [N//4, 3*N//4]
                elif shot_num == 3:
                    idx = [N//4, N//2, 3*N//4]
            new_tokens = []
            for id in idx:
                new_tokens.append(tokens[id])
            tokens = np.stack(new_tokens)

            if tokens.shape[0] == 0:
                print(path)
            tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
            tokens = torch.from_numpy(tokens)
        else:
            if bounds is not None:
                low_bound = int(bounds[0]//self.temporal_downsample)
                up_bound = int(min(math.ceil(bounds[1]/self.temporal_downsample), lim_constraint))
            if get_overlapping_segments:
                if self.split != 'test':
                    tokens1 = tokens[segment_id::4]   ### concatenating tokens for non-overlapping windows
                    tokens1 = einops.rearrange(tokens1,'S C T H W -> C (S T) H W')
                    tokens1 = tokens1[:, max(low_bound-(2*segment_id), 0):max(up_bound-(2*segment_id), 0)]
                    tokens1 = torch.from_numpy(tokens1)
                    tokens2 = None
                else:
                    tokens1 = tokens[0::4]
                    tokens2 = tokens[2::4]
                
                    tokens1 = einops.rearrange(tokens1,'S C T H W -> C (S T) H W')
                    tokens2 = einops.rearrange(tokens2,'S C T H W -> C (S T) H W')   ### getting multiple overlapping segments
                    tokens1 = tokens1[:, low_bound:up_bound]
                    tokens2 = tokens2[:, max(low_bound-4, 0) : max(up_bound-4, 0)]
                    tokens1 = torch.from_numpy(tokens1)
                    tokens2 = torch.from_numpy(tokens2)
                if self.pool_tokens < 1.0 and not is_exemplar:
                    factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
                    tokens1 = torch.nn.functional.adaptive_avg_pool3d(tokens1, (tokens1.shape[-3], factor, factor))
                    if tokens2 is not None:
                        tokens2 = torch.nn.functional.adaptive_avg_pool3d(tokens2, (tokens2.shape[-3], factor, factor))
                if self.split != 'test':
                    tokens = tokens1
                else:
                    tokens = (tokens1, tokens2)
            else:
                tokens = tokens[0::4] # non overlapping segments
                tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
                tokens = tokens[:, low_bound:up_bound]
                tokens = torch.from_numpy(tokens)
                if self.pool_tokens < 1.0:
                    factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
                    tokens = torch.nn.functional.adaptive_avg_pool3d(tokens, (tokens.shape[-3], factor, factor))
            
        if is_exemplar:
            return tokens, shot_num
        else:
            return tokens
    
    def __getitem__(self, index):
        
        video_name = self.df.iloc[index]['name'].split('/')[-1].replace('.avi', '.npz')
        type = self.df.iloc[index]['name'].split('/')[-2]
        row = self.df.iloc[index]
        if self.get_overlapping_segments and self.split=='train':
            segment_id = np.random.randint(4)
        else:
            segment_id = 0
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not math.isnan(row[key])]
        try:
            cycle_start_id = row['cycle_start_id']
        except:
            cycle_start_id = 0

        if self.multishot:
            if self.split == 'train':
                shot_num_ = np.random.randint(0,3)  ### number of examples
            else:
                shot_num_ = 0
        else:
            shot_num_ = 1

        #### with probability==threshold, randomly sample exemplars from videos of same class

        if np.random.rand() < self.threshold and type != 'other' and self.split == 'train':
            select_videos = self.df['name'][self.df['type'] == type].values
            select_example_video = np.random.choice(select_videos)
            exemplar_video_name = select_example_video.split('/')[-1].replace('.avi', '.npz')
        else:
            exemplar_video_name = video_name
  
        
        if self.split in ['val', 'test']:
            lim_constraint = np.inf
        else:
            lim_constraint = np.inf   #### to fit into gpus, you can adjust this to lesser value.
        cycle = np.array(cycle)
        cycle = cycle - row['start_frame']
        if self.split == 'train':
            segment_start = row['start_frame'] - row['start_frame']
            segment_end = row['end_frame'] - row['start_frame']
        else:
            segment_start = row['start_frame'] - row['start_frame']
            segment_end = row['end_frame'] - row['start_frame']
        num_frames = math.ceil(row['num_frames']/self.temporal_downsample)* self.temporal_downsample  
        
       
        frame_ids = np.arange(int(segment_end) + 100)
        low = int((segment_start // self.temporal_downsample) + (segment_id * 2)) * self.temporal_downsample
        up = int(min(math.ceil(segment_end / self.temporal_downsample ), lim_constraint))* self.temporal_downsample

        ### create density maps

        select_frame_ids = frame_ids[low:up][0::self.temporal_downsample]
        density_map_alt = np.zeros(len(select_frame_ids))
        actual_counts = 0
        for i in range(0,len(cycle),2):
            if cycle[i] == cycle[i+1]:
                continue
            actual_counts += 1
            st, end = (cycle[i]//self.temporal_downsample) * self.temporal_downsample, min(np.ceil(cycle[i+1]/self.temporal_downsample) * self.temporal_downsample, select_frame_ids[-1])
            if st in select_frame_ids and end in select_frame_ids:
                start_id = np.where(select_frame_ids == st)[0][0]
                end_id = np.where(select_frame_ids == end)[0][0]
                mid = (start_id + end_id)//2  ### middle of the repetition
                density_map_alt[mid] = 1  ### assign 1 to middle 
        gt_density = ndimage.gaussian_filter1d(density_map_alt, sigma=self.density_peak_width, order=0)  ### gaussian smoothing
        count = gt_density.sum()

        actual_counts = gt_density.sum()
        try:
            assert count == actual_counts
        except:
            print(count)
            print(actual_counts)

        starts = np.array(cycle[0::2])
        ends = np.array(cycle[1::2])
        durations = ends - starts
        durations = durations.astype(np.float32)
        durations[durations == 0] = 0
        select_exemplar = durations.argmax()

        ### load exemplar tokens
        examplar_path = f"{self.exemplar_dir}/{exemplar_video_name}"

        if self.split == 'train':
            example_rep, shot_num = self.load_tokens(examplar_path,True, cycle_start_id=cycle_start_id, count=None, shot_num=shot_num_) 
        else:
            example_rep, shot_num = self.load_tokens(examplar_path,True, id = None, count=None, shot_num=shot_num_)
        if shot_num_ == 0:
            shot_num = 0
        if example_rep.shape[1] == 0:
            print(row)
        
        ### load video tokens

        video_path = f"{self.tokens_dir}/{video_name}"
        vid_tokens = self.load_tokens(video_path,False, (segment_start,segment_end), lim_constraint=lim_constraint, segment_id=segment_id, get_overlapping_segments=self.get_overlapping_segments) ###lim_constraint for memory issues
        

        if not self.select_rand_segment:
            vid_tokens = vid_tokens
            gt_density = torch.from_numpy(gt_density).half() 
            return vid_tokens, example_rep, gt_density, actual_counts, self.df.iloc[index]['name'].split('/')[-1][:-4], list(vid_tokens[0].shape[-3:]), shot_num 
        
        T = row['num_frames'] ### number of frames in the video
        if T <= self.num_frames:
            start, end = 0, T
        else:
            start = random.choice(np.arange(0, T-self.num_frames, 64))
            end = start + self.num_frames  ## for taking 8 segments

        sampled_segments = vid_tokens[(start//64) : (end//64)]
        thw = sampled_segments.shape()[-3:]
        sampled_segments = einops.rearrange(sampled_segments, 'C t h w -> (t h w) C')

        gt = gt_density[(start//4): (end//4)]


        return sampled_segments, example_rep, gt, gt.sum(), self.df.iloc[index]['name'][:-4], thw, shot_num
        

    def __len__(self):
        return len(self.df)


    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        
        # [1 x T1 x .... ], [1 x T2 x ....] => [2 x T2 x ....] (T2 > T1) 
        if len(batch[0][0]) == 2:
            vids = pad_sequence([einops.rearrange(x[0][0],'C T H W -> T C H W') for x in batch])
            vids1 = pad_sequence([einops.rearrange(x[0][1],'C T H W -> T C H W') for x in batch])
            if self.compact:
                vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
                vids1 = einops.rearrange(vids1, 'T B C H W -> B (T H W) C')
            else:
                vids = einops.rearrange(vids, 'T B C H W -> B C T H W')
                vids1 = einops.rearrange(vids1, 'T B C H W -> B C T H W')
            vids = (vids, vids1)
        else:
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
        thw = [x[5] for x in batch]
        shot_num = [x[6] for x in batch]
        
        # return padded video, exemplar, padded density map,
        return vids, exemplars, gt_density, gt_density_sum, names, thw, shot_num


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = UCFRep(select_rand_segment=False, compact=False, pool_tokens_factor=0.5, get_overlapping_segments=False)
    print('--- dataset created ---')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,
                                             batch_size=1,
                                             num_workers=1,
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
    density_map_sum = []
    
    fps = []
    
    for i, item in enumerate(tqdm(dataloader)):
        print(f"It. {i} \n vid tokens: {item[0][0].shape} \n exem tokens: {item[1].shape} \n density map: {item[2].shape}:{item[3]} \n \n")
        density_map_sum.append(item[3][0].item())

