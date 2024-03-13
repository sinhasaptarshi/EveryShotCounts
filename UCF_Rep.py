import pathlib
from random import randint
import torch.utils.data
import os, sys
import numpy as np
import cv2
import math
import av
import pandas as pd
from tqdm import tqdm
import random

from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import create_video_transform
from itertools import cycle, islice
from scipy import ndimage

import pytorchvideo


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def read_video_timestamps(video_filename, timestamps, duration=0):
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
        frames.append(f)
        
    video_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in frames] # list of length T with items size [H x W x 3] 
    video_frames = thwc_to_cthw(torch.stack(video_frames).to(torch.float32))

    container.close()
    return video_frames, timestamps[-1]

class UCFRep(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 cfg = None,
                 jittering=False,
                 add_noise= False,
                 sampling='uniform',
                 encode_only=False,
                 sampling_interval=4,
                 data_dir = "data/UCF101"):
        
        self.sampling = sampling
        self.sampling_interval = sampling_interval
        self.encode_only = encode_only
        self.data_dir = data_dir
        self.split = split # set the split to load
        self.jittering = jittering # temporal jittering (augmentation)
        self.add_noise = add_noise # add noise to frames (augmentation)
        csv_path = f"datasets/ucf-rep/new_{self.split}.csv"
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['counts'].notna()]
        self.df = self.df[self.df['num_frames'] > 0]
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )
        if cfg is not None:
            self.num_frames = cfg.DATA.NUM_FRAMES
        else:
            self.num_frames = 16 

        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
        
        self.transform =  create_video_transform(mode="test",
                                        convert_to_float=False,
                                        min_size = 224,
                                        crop_size = 224,
                                        num_samples = None,
                                        video_mean = mean, 
                                        video_std = std)
    
    
    def get_vid_clips(self,vid_length, start=None, end=None):
        
        """
        get_vid_clips.

        Samples `num_frames` frames clips from the given video. 

        Args:
            vid_length (int, Optional): number of frames in the entire video. If None, it will take the end of the last repetition as the end of video
            num_frames (int, optional): number of frames to be sampled. Default is 16
            sampling_interval (int, optional): sample one frame every N frames. Default is 4
        """
        if self.encode_only:
            if start is not None and end is not None:
                return np.asarray([d for d in range(start,end+1,self.sampling_interval)])
            else:
                return np.asarray([d for d in range(0,vid_length+1,self.sampling_interval)])
        
        if self.sampling=='uniform':
            self.sampling_interval = int(vid_length/self.num_frames)
        
        clip_duration = int(self.num_frames * self.sampling_interval)  ### clip duration 
        
        start = 0
        idx = np.linspace(0, clip_duration, self.num_frames+1).astype(int)[:self.num_frames]
        
        frame_idx = start + idx
        
        if frame_idx[-1] > vid_length:
            frame_idx = frame_idx[frame_idx<=vid_length] # remove indices that are grater than the length
            frame_idx = list(islice(cycle(frame_idx), self.num_frames )) # repeat frames
            frame_idx.sort()
            frame_idx = np.asarray(frame_idx)
        
        return frame_idx


    def __getitem__(self, index):
         
        video_name = f"{self.df.iloc[index]['name']}"  ### video name
        start_frame = int(self.df.iloc[index]['start_frame'])   ### start frame index
        end_frame = int(self.df.iloc[index]['end_frame'])    ### end frame index

        if self.split == 'train':
            lim_constraint = np.inf
        else:
            lim_constraint = np.inf
        
        row = self.df.iloc[index]
        duration = int(row['num_frames'])
        clc = np.array([int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])])
        starts = clc[0::2]   #### start time stamps of repetitions
        ends = clc[1::2]  ### end timestamps of repetitions
        cycle = clc

        segment_start = 0
        segment_end = row['end_frame'] - row['start_frame']   ### end index relative to the start , so if start index is 10 and end index is 40, this shifts the start and end to 0 and 30 on the trimmed clip
        num_frames = math.ceil(row['num_frames']/8)* 8

        # num_frames = num_frames #+ 15
        frame_ids = np.arange(num_frames)
        low = ((segment_start // 8) ) * 8
        up = (min(math.ceil(segment_end / 8 ), lim_constraint))* 8
        
        ### density map creation. will be removed
        select_frame_ids = frame_ids[low:up][0::8]
        density_map_alt = np.zeros(len(select_frame_ids))
        actual_counts = 0
        for i in range(0,len(cycle),2):
            if cycle[i] == cycle[i+1]:
                continue
            actual_counts += 1
            st, end = (cycle[i]//16) * 16, min(np.ceil(cycle[i+1]/16) * 16, select_frame_ids[-1])
            if st in select_frame_ids and end in select_frame_ids:
                start_id = np.where(select_frame_ids == st)[0][0]
                end_id = np.where(select_frame_ids == end)[0][0]
                mid = (start_id + end_id)//2
                density_map_alt[mid] = 1
        gt_density = ndimage.gaussian_filter1d(density_map_alt, sigma=0.25, order=0)
        count = gt_density.sum()


        
        frame_idx = self.get_vid_clips(duration-1, start=start_frame, end=end_frame)   ### get frame indices

        vid,  num_frames = read_video_timestamps(video_name, frame_idx,  duration=duration-1)  #### returns the frames at the passed indices frame_idx

        vid = self.transform(vid/255.)  

        starts = starts - start_frame   ### shifts the timestamps as appropriate for the trimmed clip
        ends = ends - start_frame
        
        return vid, gt_density, count, starts, ends, self.df.iloc[index]['name'].split('/')[-1][:-4]

    def __len__(self):
        return len(self.df)


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = UCFRep(split='train', encode_only=True, sampling_interval=1)
    print('dataset created')
    device = torch.device("cpu") 
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,batch_size=1,num_workers=1,shuffle=False,pin_memory=False,drop_last=True)
    
    sum_clip_dur = []
    sum_tot_dur = []
    sum_clip_counts = []
    sum_tot_counts = []
    
    fps = []
    
    for i, item in enumerate(tqdm(dataloader)):
        sum_clip_dur.append(item[0])
        sum_tot_dur.append(item[1])
        sum_clip_counts.append(item[2])
        sum_tot_counts.append(item[3])
        
        fps.append(item[4])
        
        print(item[0].shape)
        print(item[-1])

    
    print(f"Avg clip dur: {sum(sum_clip_dur)/len(sum_clip_dur)} | Avg vid dur: {sum(sum_tot_dur)/len(sum_tot_dur)}")
    print(f"Avg clip reps: {sum(sum_clip_counts)/len(sum_clip_counts)} | Avg vid counts: {sum(sum_tot_counts)/len(sum_tot_counts)}")
    print(sum(fps)/len(fps))
    
    
        
    
