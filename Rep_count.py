import pathlib
from random import randint
import torch.utils.data
import os, sys
import numpy as np
import cv2
import collections
import skimage.draw
import math
import csv
import pandas as pd
from tqdm import tqdm
import random
from label_norm import normalize_label

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

import pytorchvideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Div255
)
from pytorchvideo.data.encoded_video import EncodedVideo




def read_video(video_filename, start, end, exemplar_start=None, exemplar_end=None):
    """ 
    summary

    Args:
        video_filename (string): full filepath of the video
        start (int/float): the start time in secs of the video segment
        end (int/float): the end time in secs of the video segment

    Returns:
        frames: tensor of shape C x T x H x W
        totfps: float for the video segment length (in secs)
    """
    video_present = False
    try:
        assert os.path.isfile(video_filename), f"VideoLoader: {video_filename} does not exist"
        video_present = True
    except:
        print(f"{video_filename} does not exist")
    video = EncodedVideo.from_path(video_filename) # load video with pytorchvideo dataset 
    frames = video.get_clip(start_sec=start, end_sec=end)['video']
    if exemplar_start is not None and exemplar_end is not None:
        exemplar_frames = video.get_clip(start_sec=exemplar_start, end_sec=exemplar_end)['video']
    else:
        exemplar_frames = None
    # print(frames)
    # cap = cv2.VideoCapture(video_filename)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # totfs = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    try:
        assert len(frames.shape) == 4
    except:
        if video_present:
            print(f'Issue with {video_filename} using {start} and {end} len {totfs}')
            return False, False
    return frames, exemplar_frames#, totfs


class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 cfg = None,
                 jittering=False,
                 add_noise= False,
                 data_dir = "/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/"):
        
        self.data_dir = data_dir
        self.split = split # set the split to load
        self.jittering = jittering # temporal jittering (augmentation)
        self.add_noise = add_noise # add noise to frames (augmentation)
        csv_path = f"/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/{self.split}_with_fps.csv"
        self.df = pd.read_csv(csv_path)
        if cfg is not None:
            self.num_frames = cfg.DATA.NUM_FRAMES
        else:
            self.num_frames = 16
        
        # to_remove = []
        # for i,row in tqdm(self.df.iterrows()): # iteration to remove:
            
        #     video_name = f"{self.data_dir}/{row['video_id']}.mp4"
            
        #     if not os.path.isfile(video_name): # 1. videos that were not downloaded
        #         to_remove.append(i)
        #     else:
        #         size = os.path.getsize(video_name)
        #         if size < 500: # 2. corrupted videos
        #             to_remove.append(i)
        #         cap = cv2.VideoCapture(video_name)
        #         fps = cap.get(cv2.CAP_PROP_FPS)
        #         totfs = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        #         if totfs - float(row['repetition_end']) < 0 : # 3. videos with wrong end timestamp annotations
        #             to_remove.append(i) 
        
        # print(f">> Removed {len(to_remove)} videos that were not found!")
        # self.df = self.df.drop(to_remove)  
    
    def preprocess(self,video_frame_length, time_points, num_frames):
        """
        process label(.csv) to density map label
        Args:
            video_frame_length: video total frame number, i.e 1024frames
            time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
            num_frames: 64
        Returns: for example [0.1,0.8,0.1, .....]
        """
        new_crop = []
        for i in range(len(time_points)):  # frame_length -> 64
            item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
            new_crop.append(item)
        new_crop = np.sort(new_crop)
        label = normalize_label(new_crop, num_frames)
        # print(label)

        return label

    def __getitem__(self, index):
        
        video_name = f"{self.data_dir}/{self.split}/{self.df.iloc[index]['name']}"
        row = self.df.iloc[index]
        fps = row['fps']
        duration = row['duration']
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])]
        starts = cycle[0::2]
        ends = cycle[1::2]
        exemplar_index = random.randint(0, len(starts)-1)
        exemplar_start = starts[exemplar_index]/fps  ## divide by fps tp convert to secs
        exemplar_end = ends[exemplar_index]/fps
        # print(video_name)
        # start = abs(float(self.df.iloc[index]['repetition_start']))#-float(s))
        # end = abs(float(self.df.iloc[index]['repetition_end']))#-float(s))
        
        count = float(self.df.iloc[index]['count'])
            
        if self.jittering:
            jitter = max(math.floor((end - start) / (2*count)),0)
            start += jitter 
        try:
            vid, exemplar = read_video(video_name,0,duration, exemplar_start=exemplar_start, exemplar_end=exemplar_end)
            # exemplar = read_video(video_name, exemplar_start, exemplar_end)
            # print(vid)
        
            transform =  pytorchvideo.transforms.create_video_transform(mode='train',
                                                                        convert_to_float=False,
                                                                        min_size = 224,
                                                                        crop_size = 224,
                                                                        num_samples = self.num_frames,
                                                                        video_mean = [0.485,0.456,0.406], 
                                                                        video_std = [0.229,0.224,0.225])
            transform_exemplar =  pytorchvideo.transforms.create_video_transform(mode='train',
                                                                        convert_to_float=False,
                                                                        min_size = 224,
                                                                        crop_size = 224,
                                                                        num_samples = 3,
                                                                        video_mean = [0.485,0.456,0.406], 
                                                                        video_std = [0.229,0.224,0.225])
            vid = transform(vid/255.)  
            if exemplar is not None:
                exemplar = transform_exemplar(exemplar/255) 
            # print(vid)
            density_label = torch.Tensor(self.preprocess(duration*fps, cycle, 64))
            
        except Exception:
            return (False)
        return vid, exemplar, density_label, count, video_name
            

    def __len__(self):
        return len(self.df)


## testing
if __name__=='__main__':
    from tqdm import tqdm
    dat = Rep_count()
    print('dataset created')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,batch_size=1,num_workers=2,shuffle=False,pin_memory=False,drop_last=True)
    
    for i, item in enumerate(tqdm(dataloader)):
        print(i,item[0].shape)
        print(i, item[1].shape)
        print(i, item[2].shape)
        print(item[2])
    
        
    