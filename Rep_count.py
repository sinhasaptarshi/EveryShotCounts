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

import pytorchvideo


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


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
    
    #video = EncodedVideo.from_path(video_filename) # load video with pytorchvideo dataset
    frames = []
    #fs = video._container.decode(**{"video":0}) # get a stream of frames
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

class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 cfg = None,
                 jittering=False,
                 add_noise= False,
                 sampling='uniform',
                 encode_only=False,
                 sampling_interval=4,
                 data_dir = "data/RepCount/"):
        
        self.sampling = sampling
        self.sampling_interval = sampling_interval
        self.encode_only = encode_only
        self.data_dir = data_dir
        self.split = split # set the split to load
        self.jittering = jittering # temporal jittering (augmentation)
        self.add_noise = add_noise # add noise to frames (augmentation)
        csv_path = f"datasets/repcount/{self.split}_with_fps.csv"
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['count'].notna()]
        self.df = self.df[self.df['num_frames'] > 64]

        self.df = self.df.drop(self.df.loc[self.df['name']=='stu1_10.mp4'].index)
        self.df = self.df[self.df['count'] > 0] # remove no reps
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )
        if cfg is not None:
            self.num_frames = cfg.DATA.NUM_FRAMES
        else:
            self.num_frames = 16 
        
        self.transform =  create_video_transform(mode="test",
                                                convert_to_float=False,
                                                min_size = 224,
                                                crop_size = 224,
                                                num_samples = None,
                                                video_mean = [0.485,0.456,0.406], 
                                                video_std = [0.229,0.224,0.225])

    def get_vid_clips(self,vid_length):
        
        """
        get_vid_clips.

        Samples `num_frames` frames clips from the given video. 

        Args:
            vid_length (int, Optional): number of frames in the entire video. If None, it will take the end of the last repetition as the end of video
            num_frames (int, optional): number of frames to be sampled. Default is 16
            sampling_interval (int, optional): sample one frame every N frames. Default is 4
        """
        if self.encode_only:
            return np.asarray([d for d in range(0,vid_length+1,self.sampling_interval)])
        
        if self.sampling=='uniform':
            self.sampling_interval = int(vid_length/self.num_frames)
        
        clip_duration = int(self.num_frames * self.sampling_interval)  ### clip duration 
        
        start = randint(0, max(vid_length-clip_duration, 0))  ### sample a start frame randomly
        idx = np.linspace(0, clip_duration, self.num_frames+1).astype(int)[:self.num_frames]
        
        frame_idx = start + idx
        
        if frame_idx[-1] > vid_length:
            frame_idx = frame_idx[frame_idx<=vid_length] # remove indices that are grater than the length
            frame_idx = list(islice(cycle(frame_idx), self.num_frames )) # repeat frames
            frame_idx.sort()
            frame_idx = np.asarray(frame_idx)
        
        return frame_idx

    def __getitem__(self, index):
         
        video_name = f"{self.data_dir}/{self.split}/{self.df.iloc[index]['name']}"
        cap = cv2.VideoCapture(video_name)
        
        row = self.df.iloc[index]
        duration = row['num_frames']
        clc = [int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])]
        starts = clc[0::2]
        ends = clc[1::2]
        
        frame_idx = self.get_vid_clips(duration-1)   ### get frame indices
        vid,  num_frames = read_video_timestamps(video_name, frame_idx, duration=duration-1)  ## return frames at the passed indices
        vdur = (frame_idx[-1] - frame_idx[0]) / row['fps']
        vid = self.transform(vid/255.)  
        return vid, starts, ends, self.df.iloc[index]['name'][:-4]
            

    def __len__(self):
        return len(self.df)


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = Rep_count(data_dir = "data/LLSP/")
    print('dataset created')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,batch_size=1,num_workers=10,shuffle=False,pin_memory=False,drop_last=True)
    
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
        
        # print(i, item[1].shape)
        # print(i, item[2].shape)
        # print(item[2])
    
    print(f"Avg clip dur: {sum(sum_clip_dur)/len(sum_clip_dur)} | Avg vid dur: {sum(sum_tot_dur)/len(sum_tot_dur)}")
    print(f"Avg clip reps: {sum(sum_clip_counts)/len(sum_clip_counts)} | Avg vid counts: {sum(sum_tot_counts)/len(sum_tot_counts)}")
    print(sum(fps)/len(fps))
    
    
        
    
