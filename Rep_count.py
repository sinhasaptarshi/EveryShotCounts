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
from pytorchvideo.data.utils import thwc_to_cthw

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

def read_video_timestamps(video_filename, timestamps, exemplar_timestamps):
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
    video = EncodedVideo.from_path(video_filename) # load video with pytorchvideo dataset
    frames = {}
    exemplar_frames = []
    fs = video._container.decode(**{"video":0}) # get a stream of frames


    
    # print(timestamps)
    for iter, f in enumerate(fs):
        # print(f.pts)
        # if f.pts in timestamps: # check if timestamp is within given list
        #     frames[f.pts] = f
        if iter in exemplar_timestamps:
            exemplar_frames.append(f)
        if iter in timestamps:
            frames[iter] = f
        elif iter > timestamps[-1]:
            break
        # elif f.pts > timestamps[-1]:
        #     break

    result = [frames[pts] for pts in sorted(frames)] # rearrange
    # print(torch.from_numpy(result[1].to_ndarray(format='rgb24')).shape)
    

    video_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in result] # list of length T with items size [H x W x 3] 
    video_frames = thwc_to_cthw(torch.stack(video_frames).to(torch.float32))
    # print(video_frames.shape)
    exemplar_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in exemplar_frames]  ### example reps
    exemplar_frames = thwc_to_cthw(torch.stack(exemplar_frames).to(torch.float32))
    # frames = thwc_to_cthw(torch.stack(video_frames)).to(torch.float32) # C x T x H x W
    # print(video_frames.shape)
    return video_frames, exemplar_frames, timestamps[-1]


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
        self.df = self.df[self.df['count'].notna()]
        self.df = self.df[self.df['count'] < 5] ### remove videos with more than 5 repetitions
        print(len(self.df))
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
    
    def get_vid_segment(self, time_points, min_frames=3, num_frames=16, sample_breaks=False, use_softmax=True):
        """ 
        get_vid_segment. 
        
        Find the keyframes to sample from given: the number of frames to sample, a minimum number of frames per repetition,
        and if to also sample over video segments that no repetitions occur.

        Args:
            time_points (list): list containing all the start and end iteration kepoints in the format [<start loop 1> , <end loop 1> , <start loop 2>, etc.]
            min_frames (int, optional): The minimum number of frames to sample per repetition. Defaults to 3.
            num_frames (int, optional): The number of frames to sample. Defaults to 64.
            sample_empty (bool, optional): Boolean flag to either sample over segments without repetitins (between end and start).
            use_softmax (bool, option): Bolean flag to either weight loop segments by their softmax. Is set to `True` the sum of each loop segment within the density map should be 1.

        Returns:
            tuple: containing the list of frame indices, the new count based on the sample segment, and a (unscaled) density matrix based on the start and end indices sampled.
        """
        
        if sample_breaks: # Flag for sampling in transitioning segments between repetitions
            reps = {}
            for idx,i in enumerate(range(0,len(time_points),1)) :
                if time_points[i] < time_points[i+1]:
                    if i % 2 == 0: # Segment is between a start and end of a repetition
                        reps[idx] = {'start':time_points[i], 'end':time_points[i+1], 'is_rep':True}
                    else:
                        reps[idx] = {'start':time_points[i], 'end':time_points[i+1], 'is_rep':False}
        else:
            reps = {idx:{'start':time_points[i], 'end':time_points[i+1], 'is_rep':True} for idx,i in enumerate(range(0,len(time_points),2))}
        
        if int(time_points[-1]) - int(time_points[0]) > num_frames : # video has >= 64 frames
            counts = num_frames//min_frames # get the maximum number of repetitions that can be observed based on min_frames
            
            if len(reps) > counts : # the number of repetitions that can be sampled is less than the ground truth reps
                start_idx = random.randint(0, len(reps) - counts) # get (a random) start index
                n_reps = {i:reps[i] for i  in range(start_idx,start_idx+counts)}
            else: # the number of repetitions that can be sampled is more than the ground truth reps
                start_idx = 0
                min_frames = num_frames // len(reps)
                n_reps = reps
        else:
            n_reps = reps
            counts = len(n_reps)
        
        rep_counts = 0
        for idx in n_reps: # Adjust the counts based on the number of observable repetitions
            if n_reps[idx]['is_rep']:
                rep_counts += 1

        indices = [] # frame indices to sample

        for val in n_reps.values():
            if val['start'] == val['end']:
                val['end'] += 2
            for v in np.sort(np.random.randint(int(val['start']),int(val['end']),size=min_frames)):
                indices.append(v)
                
        prox_to_mean = [] # list to store gaussians based on repetitions start/end for density map
        for t in range(time_points[0],time_points[-1]):
            is_rep = False # flag to keep track if the timestamp falls within a loop
            for key in reps.keys():
                if reps[key]['start'] <= t and reps[key]['end'] >= t and  reps[key]['is_rep'] : # only assign a >= 0 value if the timestamp is within a loop
                    is_rep =True
                    mean = float(reps[key]['start']) + (float(reps[key]['end']) - float(reps[key]['start']))/2
                    prox_to_mean.append(abs(abs(float(t)-mean)/(mean-reps[key]['start'])-1)) # calculate the absolute distance to the mean (regularised)
                    break
            if not is_rep: # add zeros for not repetitions
                prox_to_mean.append(0)
                
        if use_softmax: # weight by softmax at each segment locations
            for key in reps.keys():
                tot = 0
                for idx in range(reps[key]['start']-time_points[0],reps[key]['end']-time_points[0]): # first for loop to calculate exponent per item
                    prox_to_mean[idx] = math.exp(prox_to_mean[idx])
                    tot += prox_to_mean[idx]
                for idx in range(reps[key]['start']-time_points[0],reps[key]['end']-time_points[0]): # second for loop for softmax
                    prox_to_mean[idx] /= tot
                  
        return indices, rep_counts, prox_to_mean[indices[0]:indices[-1]]
    

    def __getitem__(self, index):
        
        video_name = f"{self.data_dir}/{self.split}/{self.df.iloc[index]['name']}"
        row = self.df.iloc[index]
        fps = row['fps']
        duration = row['duration']
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])]
        starts = cycle[0::2]
        ends = cycle[1::2]
        # assert len(starts) != 0
        exemplar_index = random.randint(0, len(starts)-1)
        exemplar_start = starts[exemplar_index]  ## divide by fps tp convert to secs
        exemplar_end = ends[exemplar_index]
        exemplar_frameidx = np.random.randint(exemplar_start, exemplar_end, 3)
        # print(video_name)
        # start = abs(float(self.df.iloc[index]['repetition_start']))#-float(s))
        # end = abs(float(self.df.iloc[index]['repetition_end']))#-float(s))
        
        original_count = float(self.df.iloc[index]['count'])
        
            
        if self.jittering:
            jitter = max(math.floor((end - start) / (2*count)),0)
            start += jitter 
        # try:
        # print(cycle)
        frame_idx, count, density = self.get_vid_segment(cycle, sample_breaks=False)
        # print(frame_idx)
        vid, exemplar, num_frames = read_video_timestamps(video_name, frame_idx, exemplar_frameidx)
        # print(vid.shape)
        # print(exemplar.shape)
        # print(num_frames)
        
        label = normalize_label(cycle, cycle[-1]) ## computing density map over entire video
        # print(original_count, label.sum())
        density = label[frame_idx] ## sampling the density map at the selected frame indices
        density = density / density.sum() * count  ### normalizing density to sum up to count
        print(density)
        print(count, density.sum())
        # print(density)
        # print(count)

        # vid, exemplar = read_video(video_name,0,duration, exemplar_start=exemplar_start, exemplar_end=exemplar_end)
        # exemplar = read_video(video_name, exemplar_start, exemplar_end)
        # print(vid)
    
        transform =  pytorchvideo.transforms.create_video_transform(mode=self.split,
                                                                    convert_to_float=False,
                                                                    min_size = 224,
                                                                    crop_size = 224,
                                                                    num_samples = None,
                                                                    video_mean = [0.485,0.456,0.406], 
                                                                    video_std = [0.229,0.224,0.225])
        transform_exemplar =  pytorchvideo.transforms.create_video_transform(mode=self.split,
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
        density_label = torch.Tensor(self.preprocess(duration*fps, cycle, self.num_frames))
        
        # except Exception:
        #     return (False)
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
        print(i)
        # print(i,item[0].shape)
        # print(i, item[1].shape)
        # print(i, item[2].shape)
        # print(item[2])
    
        
    
