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




def read_video(video_filename, start, end):
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
    # print(frames)
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    totfs = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    try:
        assert len(frames.shape) == 4
    except:
        if video_present:
            print(f'Issue with {video_filename} using {start} and {end} len {totfs}')
            return False, False
    return frames, totfs


class Countix(torch.utils.data.Dataset):
    def __init__(self,
                 split="val",
                 cfg = None,
                 jittering=False,
                 add_noise= False,
                 data_dir = "/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/countix/countix_videos"):
        
        self.data_dir = data_dir
        self.split = split # set the split to load
        self.jittering = jittering # temporal jittering (augmentation)
        self.add_noise = add_noise # add noise to frames (augmentation)
        csv_path = f"/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/countix/countix_{self.split}_modified.csv"
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


    def __getitem__(self, index):
        
        video_name = f"{self.data_dir}/{self.df.iloc[index]['video_id']}.mp4"
        print(video_name)
        start = abs(float(self.df.iloc[index]['repetition_start']))#-float(s))
        end = abs(float(self.df.iloc[index]['repetition_end']))#-float(s))
        
        count = float(self.df.iloc[index]['count'])
            
        if self.jittering:
            jitter = max(math.floor((end - start) / (2*count)),0)
            start += jitter 
        try:
            vid, totfs = read_video(video_name,start,end)
        
            transform =  pytorchvideo.transforms.create_video_transform(mode='val',
                                                                        convert_to_float=False,
                                                                        min_size = 224,
                                                                        crop_size = 224,
                                                                        num_samples = self.cfg.NUM_FRAMES,
                                                                        video_mean = [0.485,0.456,0.406], 
                                                                        video_std = [0.229,0.224,0.225])
            vid = transform(vid/255.)   
            print(vid.shape)
        except Exception:
            return (False)
        return vid, count, video_name, self.df.iloc[index]['video_id'], start, end
            

    def __len__(self):
        return len(self.df)



if __name__=='__main__':
    from tqdm import tqdm
    dat = Countix()
    print('dataset created')
    device = torch.device("cpu")
    print(f'Device: {device}')
    dataloader = torch.utils.data.DataLoader(dat,batch_size=24,num_workers=12,shuffle=False,pin_memory=False,drop_last=True)
    
    for i,item in enumerate(tqdm(dataloader)):
        print(i,item[0].shape)
    
        
    