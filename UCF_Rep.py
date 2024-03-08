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
from label_norm import normalize_label
from pytorchvideo.data.utils import thwc_to_cthw
from pytorchvideo.transforms import create_video_transform
from itertools import cycle, islice
from scipy import ndimage

import pytorchvideo


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def read_video_timestamps(video_filename, timestamps, exemplar_timestamps, duration=0):
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
    exemplar_frames = []
    #fs = video._container.decode(**{"video":0}) # get a stream of frames
    container = av.open(video_filename)
    
    # min_t = min((min(timestamps),min(exemplar_timestamps)))
    # max_t = max((max(timestamps),max(exemplar_timestamps)))
    min_t = min(timestamps)
    max_t = max(timestamps)
    # max_t = 1000000000000000
    
    for i, f in enumerate(islice(container.decode(video=0), min_t, max_t+1)):
        c = i + min_t
        if c in exemplar_timestamps:
            for _ in range(exemplar_timestamps.tolist().count(c)): # for multiple occurrences
                exemplar_frames.append(f)
        frames.append(f)
        # if c in timestamps:
        #     for _ in range(timestamps.tolist().count(c)): # for multiple occurrences
        #         frames.append(f)
        
    video_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in frames] # list of length T with items size [H x W x 3] 
    video_frames = thwc_to_cthw(torch.stack(video_frames).to(torch.float32))
    # print(video_frames.shape)
    exemplar_frames = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in exemplar_frames]  ### example reps
    exemplar_frames = thwc_to_cthw(torch.stack(exemplar_frames).to(torch.float32))
    container.close()
    return video_frames, exemplar_frames, timestamps[-1]

'''
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
    try:
        assert len(frames.shape) == 4
    except:
        if video_present:
            print(f'Issue with {video_filename} using {start} and {end} len {totfs}')
            return False, False
    return frames, exemplar_frames#, totfs
'''

class UCFRep(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 cfg = None,
                 jittering=False,
                 add_noise= False,
                 sampling='uniform',
                 encode_only=False,
                 sampling_interval=4,
                 data_dir = "/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/UCF-101"):
        
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
        # self.df = self.df[self.df['count'] < 5] ### remove videos with more than 5 repetitions
        # self.df = self.df[self.df['fps'] >= 10]
        # self.df = self.df[(self.df['name'] == 'stu9_71.mp4')]# or (self.df['name'] =='stu9_7.mp4')]
        # self.df = self.df[self.df['num_frames'] > 64]
        # self.df = self.df[self.df['name'] == 'test118.mp4']
        # self.df = pd.concat([self.df, self.df])

        # self.df = self.df.drop(self.df.loc[self.df['name']=='stu1_10.mp4'].index)
        # self.df = self.df[self.df['count'] > 0] # remove no reps
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
        self.transform_exemplar =  create_video_transform(mode=self.split,
                                                        convert_to_float=False,
                                                        min_size = 224,
                                                        crop_size = 224,
                                                        num_samples = 3,
                                                        video_mean = mean, 
                                                        video_std = std)
    
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

        return label
    
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
        
        # start = randint(0, max(vid_length-clip_duration, 0))  ### sample a start frame randomly
        start = 0
        idx = np.linspace(0, clip_duration, self.num_frames+1).astype(int)[:self.num_frames]
        
        frame_idx = start + idx
        
        if frame_idx[-1] > vid_length:
            frame_idx = frame_idx[frame_idx<=vid_length] # remove indices that are grater than the length
            frame_idx = list(islice(cycle(frame_idx), self.num_frames )) # repeat frames
            frame_idx.sort()
            frame_idx = np.asarray(frame_idx)
        
        return frame_idx


    '''
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
    '''
    

    def __getitem__(self, index):
         
        # video_name = f"{self.data_dir}//{self.df.iloc[index]['video_id']}"
        video_name = f"{self.df.iloc[index]['name']}"
        #print(os.path.isfile(video_name))
        cap = cv2.VideoCapture(video_name)
        start_frame = int(self.df.iloc[index]['start_frame'])
        end_frame = int(self.df.iloc[index]['end_frame'])

        if self.split == 'train':
            lim_constraint = np.inf
        else:
            lim_constraint = np.inf
        
        row = self.df.iloc[index]
        duration = int(row['num_frames'])
        clc = np.array([int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])])
        # print(len(clc))
        starts = clc[0::2]
        ends = clc[1::2]
        # clc = clc - start_frame
        cycle = clc

        segment_start = 0
        segment_end = row['end_frame'] - row['start_frame']  
        num_frames = math.ceil(row['num_frames']/8)* 8
        
        # 
        # segment_start = 0
        # segment_end = row['num_frames']   
        # --- Alternate density map loading ---
        # density_map = self.preprocess(num_frames, cycle, num_frames)
        num_frames = num_frames #+ 15
        frame_ids = np.arange(num_frames)
        low = ((segment_start // 8) ) * 8
        up = (min(math.ceil(segment_end / 8 ), lim_constraint))* 8
        # density_map = np.array(density_map[low: up])
        # select_frame_ids = frame_ids[0::16]
        # print(len(select_frame_ids))
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
        # print(density_map_alt.sum())
        # gt_density = density_map_alt
        gt_density = ndimage.gaussian_filter1d(density_map_alt, sigma=0.25, order=0)
        count = gt_density.sum()
        # print(count)
        # print(duration)
        # exemplar_index = random.randint(0, len(starts)-1)
        exemplar_index = 0
        exemplar_start = starts[exemplar_index]  ## divide by fps tp convert to secs
        exemplar_end = ends[exemplar_index]
        exemplar_frameidx = np.linspace(exemplar_start, exemplar_end, 3).astype(int)
        # print(exemplar_frameidx)
        
        frame_idx = self.get_vid_clips(duration-1, start=start_frame, end=end_frame)
        # frame_idx = self.get_vid_clips(duration)
        # frame_idx, count, density = self.get_vid_segment(cycle, sample_breaks=False)
        # print(frame_idx)
        vid, exemplar, num_frames = read_video_timestamps(video_name, frame_idx, exemplar_frameidx, duration=duration-1)
        
        # label = normalize_label(clc, duration+10) ## computing density map over entire video

        # count = label[frame_idx[0]: frame_idx[-1]].sum()  ### calculating the actual count in the trimmed clip
        # density = label[frame_idx - start_frame] ## sampling the density map at the selected frame indices
        

        # density = torch.Tensor(density / (density.sum() + 1e-10) * count)  ### normalizing density to sum up to count
        # vdur = (frame_idx[-1] - frame_idx[0]) / row['fps']
        #print(f"""Sampled frames [count: {count:.2f} | duration: {vdur:.2f}] full video [count: {row['count']} | duration: {row['duration']:.2f}] """)
        
        # print(vid)
        # print(vid.shape[1])
        vid = self.transform(vid/255.)  
        if exemplar is not None:
            exemplar = self.transform_exemplar(exemplar/255) 
        

        starts = starts - start_frame
        ends = ends - start_frame
        
        return vid, gt_density, count, starts, ends, self.df.iloc[index]['name'].split('/')[-1][:-4]#, segment_start, segment_end

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
        # print(i, item[1].shape)
        # print(i, item[2].shape)
        # print(item[2])
    
    print(f"Avg clip dur: {sum(sum_clip_dur)/len(sum_clip_dur)} | Avg vid dur: {sum(sum_tot_dur)/len(sum_tot_dur)}")
    print(f"Avg clip reps: {sum(sum_clip_counts)/len(sum_clip_counts)} | Avg vid counts: {sum(sum_tot_counts)/len(sum_tot_counts)}")
    print(sum(fps)/len(fps))
    
    
        
    