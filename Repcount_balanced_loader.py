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

class Rep_count(torch.utils.data.Dataset):
    def __init__(self,
                 split="train",
                 add_noise= False,
                 num_frames=512,
                 tokens_dir = "saved_tokens_reencoded",
                 exemplar_dir = "exemplar_tokens_reencoded",
                 density_maps_dir = "gt_density_maps_recreated",
                 select_rand_segment=True,
                 compact=False,
                 lim_constraint=np.inf,
                 pool_tokens_factor=1.0,
                 peak_at_random_location=True,
                 get_overlapping_segments=False,
                 multishot=True,
                 density_peak_width=1.0):
        
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
        self.density_peak_width = density_peak_width
        # print(self.get_overlapping_segments)
        if self.split == 'train':
            # csv_path = f"datasets/repcount/{self.split}_balanced_new.csv"
            # csv_path = f"datasets/repcount/{self.split}_less_than_6.csv"
            # csv_path = f"datasets/repcount/{self.split}_balanced_new.csv"
            csv_path = f"datasets/repcount/{self.split}_with_fps.csv"
        else:
            csv_path = f"datasets/repcount/validtest_with_fps.csv"
        self.df = pd.read_csv(csv_path)
        self.df['density_map_sum'] = 0
        self.df = self.df[self.df['count'].notna()]
        # self.df = self.df[self.df['count'] < 5] ### remove videos with more than 5 repetitions
        # self.df = self.df[self.df['fps'] >= 10]
        self.df = self.df[self.df['num_frames'] > 64]
        self.df = self.df.drop(self.df.loc[self.df['name']=='stu1_10.mp4'].index)
        self.df = self.df[self.df['count'] > 0] # remove no reps
        # self.df = self.df.drop(self.df.loc[self.df['name']=='stu9_69.mp4'].index)
        # self.df = self.df[self.df['num_frames'] < 1800]
        print(f"--- Loaded: {len(self.df)} videos for {self.split} --- " )
    

    def PDF(self, x, u, sig):
        # f(x)
        return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

# integral f(x)
    def get_integrate(self, x_1, x_2, avg, sig):
        y, err = integrate.quad(self.PDF, x_1, x_2, args=(avg, sig))
        return y

    def normalize_label(self, y_frame, y_length):
    # y_length: total frames
    # return: normalize_label  size:nparray(y_length,)
        y_label = [0 for i in range(y_length)]  # 坐标轴长度，即帧数
        for i in range(0, len(y_frame), 2):
            x_a = y_frame[i]
            x_b = y_frame[i + 1]
            avg = (x_b + x_a) / 2
            sig = (x_b - x_a) / 6
            num = x_b - x_a + 1  # 帧数量 update 1104
            if num != 1:
                for j in range(num):
                    x_1 = x_a - 0.5 + j
                    x_2 = x_a + 0.5 + j
                    y_ing = self.get_integrate(x_1, x_2, avg, sig)
                    y_label[x_a + j] = y_ing
            else:
                y_label[x_a] = 1
        return y_label
    
        
    def load_tokens(self,path,is_exemplar,bounds=None, lim_constraint=np.inf, id=None, cycle_start_id=0, count=None, get_overlapping_segments=True, merge_early=False, shot_num=1, segment_id=0):
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
                
                idx = cycle_start_id + np.random.choice(np.arange(N), size=shot_num, replace=False)
                # idx = cycle_start_id + np.random.randint(N - shot_num + 1, size=shot_num)
                # print(idx)
                # idx = 0
                
                # shot_num = 1
            else:
                
                # idx = [id] if id is not None else [0]
                
                # shot_num = min(1, N)
                # shot_num = 1
                shot_num = min(shot_num, N)
                idx = np.arange(shot_num)
                print(idx)
                # print(shot_num)
                # if id is not None:
                #     # idx = np.random.randint(N)
                #     idx = 0
                # else:
                #     idx = 0
            # print(idx)
            new_tokens = []
            for id in idx:
                new_tokens.append(tokens[id])
            tokens = np.stack(new_tokens)


            # tokens = tokens[idx:idx+shot_num] ### return the encoding for a selected example per video instance
            # print(tokens.shape)
            if tokens.shape[0] == 0:
                print(path)
            tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
            tokens = torch.from_numpy(tokens)
            # if self.pool_tokens < 1.0:
            #     factor = math.ceil(tokens.shape[-1] * 0.65)
            #     tokens = torch.nn.functional.adaptive_avg_pool3d(tokens, (tokens.shape[-3], factor, factor))
            

        else:
            if bounds is not None:
                low_bound = bounds[0]//8
                # up_bound = bounds[1]//8 
                up_bound = min(math.ceil(bounds[1]/8), lim_constraint)
            # else:
            #     low_bound = 0
            #     up_bound = None
            if get_overlapping_segments:
                # tokens = tokens[0::2]
                ### merging overlapping segments by averaging the tokens over the overlapping frames
                if merge_early:
                    num_frames_per_segment = 64
                    overlap_between_segments = 0.75
                    downsampling_factor = 8
                    temporal_tokens_per_segment = num_frames_per_segment // downsampling_factor
                    non_overlapping_tokens_per_segment = int((1 - overlap_between_segments) * temporal_tokens_per_segment)
                    # print(tokens.shape)
                    vid_dur = int(num_frames_per_segment + ((tokens.shape[0] - 1) * (1-overlap_between_segments) * num_frames_per_segment)) // downsampling_factor ### computing the total number of frames encoded by the segments. 
                    freq = np.zeros(vid_dur)
                    merged_tokens = np.zeros((tokens.shape[1], vid_dur, tokens.shape[3], tokens.shape[4]))
                    for segment_idx in range(tokens.shape[0]):
                        freq[(non_overlapping_tokens_per_segment * segment_idx): (non_overlapping_tokens_per_segment * segment_idx + temporal_tokens_per_segment)] += 1 #### 1 temporal token represents 8 frames. So 64 frames are encoded in 8 temporal tokens. freq calculates the number of times one temporal position appears during the summing of overlapping segments
                        merged_tokens[:, (non_overlapping_tokens_per_segment * segment_idx): (non_overlapping_tokens_per_segment * segment_idx + temporal_tokens_per_segment)] += tokens[segment_idx]   ### adding tokens over overlapping segments
                    freq = np.expand_dims(freq, (0, 2, 3))
                    merged_tokens = merged_tokens/ freq
                    tokens = merged_tokens
                else:
                    tokens1 = tokens[0::4]
                    tokens2 = tokens[2::4]
                    
                    tokens1 = einops.rearrange(tokens1,'S C T H W -> C (S T) H W')
                    tokens2 = einops.rearrange(tokens2,'S C T H W -> C (S T) H W')
                    # print(low_bound)
                    tokens1 = tokens1[:, low_bound:up_bound]
                    tokens2 = tokens2[:, max(low_bound-4, 0) : max(up_bound-4, 0)]
                    if tokens2.shape[1] == 0:
                        tokens2 = tokens1  ### incase we get empty tokens
                    # print(tokens1.shape)
                    # print(tokens2.shape)
                    tokens1 = torch.from_numpy(tokens1)
                    tokens2 = torch.from_numpy(tokens2)
                    if segment_id == 1:
                        tokens1 = tokens2
                    if self.pool_tokens < 1.0 and not is_exemplar:
                        factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
                        tokens1 = torch.nn.functional.adaptive_avg_pool3d(tokens1, (tokens1.shape[-3], factor, factor))
                        tokens2 = torch.nn.functional.adaptive_avg_pool3d(tokens2, (tokens2.shape[-3], factor, factor))
                    tokens = (tokens1, tokens2)
                    

                
                # print(tokens.shape)
            else:
                tokens = tokens[0::4] # non overlapping segments
                tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
            # if self.split == 'train':
                tokens = tokens[:, low_bound:up_bound]
            # tokens = tokens[low_bound:min(up_bound, low_bound+lim_constraint)] ## non overlapping segments
                
        
                tokens = torch.from_numpy(tokens)
                if self.pool_tokens < 1.0 and not is_exemplar:
                    factor = math.ceil(tokens.shape[-1] * self.pool_tokens)
                    tokens = torch.nn.functional.adaptive_avg_pool3d(tokens, (tokens.shape[-3], factor, factor))
        
        # tokens = einops.rearrange(tokens,'S C T H W -> C (S T) H W')
        
        # if bounds is not None:
        #     start = bounds[0] // 8 ## Sampling every 4 frames and MViT temporally downsample T=16 -> 8 
        #     end = bounds[1] // 8
        #     tokens = tokens[:,start:end,:,:]
        if is_exemplar:
            return tokens, shot_num
        else:
            return tokens

    def preprocess(self, video_frame_length, time_points, num_frames):
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
        label = self.normalize_label(new_crop, num_frames)

        return label


    def load_density_map(self,path,count, bound, lim_constraint=20):
        gt_density_map = np.load(path)['arr_0']#[0::4]
        low = bound[0] // 8 * 8
        up = (bound[1] // 8 ) * 8
        # gt_density_map = gt_density_map/gt_density_map.sum() * count 
        # print(gt_density_map.shape)
        # gt_density_map = gt_density_map[(low * 64):(min(up, low + lim_constraint)  * 64)] #multiply 60 if needed
        gt_density_map = gt_density_map[low: up]
        # gt_density_map = gt_density_map[bound[0]:bound[1]+1]
        # return gt_density_map
        return  gt_density_map ##scale by count to make the sum consistent
      
      
    
    def __getitem__(self, index):
        video_name = self.df.iloc[index]['name'].replace('.mp4', '.npz')
        row = self.df.iloc[index]
        # no_repetitions = row['count']
        
        cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and not math.isnan(row[key])]
        rep_locs = [int(float(row[key])) for key in row.keys() if 'M' in key and not math.isnan(row[key])]
        cycle_start_id = row['cycle_start_id']
        rep_starts = np.array(cycle[0::2])  ### getting the start of repetitions
        rep_ends = np.array(cycle[1::2])   ### getting the ending of the repetitions
        if self.multishot:
            if self.split == 'train':
                shot_num_ = np.random.randint(0,3)  ### number of examples
            else:
                shot_num_ = 2
        else:
            shot_num_ = 1
        rep_dur = np.array(rep_ends - rep_starts)
        if rep_dur.min() == 0:
            rm_id = np.where(rep_dur == 0)[0]
            rep_starts = np.delete(rep_starts, rm_id)
            rep_ends = np.delete(rep_ends, rm_id)
        assert ((len(rep_locs) == len(rep_starts)) & (len(rep_locs) == len(rep_ends)))
        no_repetitions = len(rep_starts)
        # print(row['count'])
        if self.split == 'train':
            lim_constraint = np.inf
        else:
            lim_constraint = np.inf
        
        
        # if self.split in ['val', 'test']:
        #     lim_constraint = np.inf
        # else:
        #     lim_constraint = np.inf

        segment_start = row['segment_start']
        segment_end = row['segment_end']  
        num_frames = row['num_frames']   
        if no_repetitions > 6:
            c = np.random.randint(1, 7) ### randomly drawing count from 1 to 6
            # print(c)
            cycle_start_id = np.random.randint(no_repetitions - c + 1)
            # segment_start = cycle[2*cycle_start_id]
            # segment_end = cycle[2*(cycle_start_id + c - 1) + 1]
            segment_start = rep_starts[cycle_start_id]
            segment_end = rep_ends[cycle_start_id + c - 1]
            # segment_start = rep_ends[cycle_start_id - 1]
            # segment_end = rep_starts[cycle_start_id + c ]
        # 
        # segment_start = 0
        # segment_end = row['num_frames']   
        # --- Alternate density map loading ---
        # density_map = self.preprocess(num_frames, cycle, num_frames)
        # frame_ids = np.arange(range(0, num_frames, 16)[-1] + 64)
        frame_ids = np.arange(num_frames)
        segment_id = 0
        if self.split == 'train':
            if self.get_overlapping_segments:
                if np.random.rand() < 0.5:
                    segment_id = 1
                else:
                    segment_id = 0
        low = (segment_start // 8) * 8
        up = min(math.ceil(segment_end / 8 ), lim_constraint) * 8
        # density_map = np.array(density_map[low: up])
        # select_frame_ids = frame_ids[0::8]
        if segment_id == 0:
            select_frame_ids = frame_ids[low:up][0::8]
        else:
            low = max(low, 32)
            select_frame_ids = frame_ids[low:up][0::8]
        density_map_alt = np.zeros(len(select_frame_ids))
        actual_counts = 0
        if self.peak_at_random_location:
            for i in range(0, len(rep_locs)):
                rep_loc = rep_locs[i]
                rep_loc = round(rep_loc / 8) * 8
                if rep_loc in select_frame_ids:
                    mid = np.where(select_frame_ids == rep_loc)[0][0]
                    density_map_alt[mid] = 1
        else:
            for i in range(0,len(cycle),2):
                if cycle[i] == cycle[i+1]:
                    continue
                
                st, end = (cycle[i]//8) * 8, min(np.ceil(cycle[i+1]/8) * 8, select_frame_ids[-1])
                if st in select_frame_ids and end in select_frame_ids:
                    start_id = np.where(select_frame_ids == st)[0][0]
                    end_id = np.where(select_frame_ids == end)[0][0]
                    mid = (start_id + end_id)//2
                    density_map_alt[mid] = 1
                # actual_counts += 1
        # print(density_map_alt.sum())
        # gt_density = density_map_alt
        gt_density = ndimage.gaussian_filter1d(density_map_alt, sigma=self.density_peak_width, order=0)
        count = gt_density.sum()
        # if round(count) > 6:
        #     print(count)
        #     print(c)
        # print(count)
        


        # gt_counts = density_map.sum()
        # density_map = density_map[0::8]
        # gt_density = density_map / density_map.sum() * gt_counts

        
        # --- Exemplar tokens loading ---
        # examplar_path = f"{self.exemplar_dir}/{self.split}/{video_name}"
        starts = np.array(cycle[0::2])
        ends = np.array(cycle[1::2])
        durations = ends - starts
        durations = durations.astype(np.float32)
        durations[durations == 0] = np.inf
        select_exemplar = durations.argmin()
        examplar_path = f"{self.exemplar_dir}/{video_name.replace('.npz', '_new.npz')}"
        # examplar_path = f"{self.exemplar_dir}/{self.df.iloc[(index + np.random.randint(100)) % self.__len__()]['name'].replace('.mp4', '_new.npz')}"
        if self.split == 'train':
            example_rep, shot_num = self.load_tokens(examplar_path,True, cycle_start_id=cycle_start_id, count=count, shot_num=shot_num_) 
        else:
            example_rep, shot_num = self.load_tokens(examplar_path,True, id = None, count=count, shot_num=shot_num_) 
        if shot_num_ == 0:
            shot_num = 0
        if example_rep.shape[1] == 0:
            print(row)
        # example_rep = self.load_tokens(examplar_path, True)
        
        
        # --- Density map loading ---
        # density_map_path = f"{self.density_maps_dir}/{video_name}"
        # gt_density = self.load_density_map(density_map_path,row['count'],(segment_start,segment_end), lim_constraint=lim_constraint)  
        # gt_density = gt_density[segment_start:(segment_end//64 * 64)]
        
        # --- Video tokens loading ---
        # video_path = f"{self.tokens_dir}/{self.split}/{video_name}"
        video_path = f"{self.tokens_dir}/{video_name}"
        vid_tokens = self.load_tokens(video_path,False, (segment_start,segment_end), lim_constraint=lim_constraint, get_overlapping_segments=self.get_overlapping_segments, segment_id=segment_id) ###lim_constraint for memory issues
        
        # self.df['density_map_sum'].iloc[index] = gt_density.sum()
        # print(self.get_overlapping_segments)

        if not self.select_rand_segment:
            vid_tokens = vid_tokens
            gt_density = torch.from_numpy(gt_density).half() 
            # if row['count'] > gt_density.sum():
            #     print(row['count'],gt_density.sum(),self.df.iloc[index]['name'][:-4])
            # print(shot_num)
            if not self.get_overlapping_segments:
                return vid_tokens, example_rep, gt_density, gt_density.sum(), self.df.iloc[index]['name'][:-4], list(vid_tokens.shape[-3:]), shot_num,
            else:
                return vid_tokens, example_rep, gt_density, gt_density.sum(), self.df.iloc[index]['name'][:-4], list(vid_tokens[0].shape[-3:]), shot_num,
        
        T = row['num_frames'] ### number of frames in the video
        if T <= self.num_frames:
            start, end = 0, T
        else:
            start = random.choice(np.arange(0, T-self.num_frames, 64))
            end = start + self.num_frames  ## for taking 8 segments

        sampled_segments = vid_tokens[(start//64) : (end//64)]
        thw = sampled_segments.shape()[-3:]
        sampled_segments = einops.rearrange(sampled_segments, 'C t h w -> (t h w) C')
        #n, c, t, h, w = sampled_segments.shape
        #sampled_segments = sampled_segments.permute(0, 2, 3, 4, 1).reshape(-1, c)
        # sampled_segment = torch.stack(sampled_segments)
        gt = gt_density[(start//4): (end//4)]
        # print(gt.sum())

        return sampled_segments, example_rep, gt, gt.sum(), self.df.iloc[index]['name'][:-4], thw
        

    def __len__(self):
        return len(self.df)


    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        
        # [1 x T1 x .... ], [1 x T2 x ....] => [2 x T2 x ....] (T2 > T1) 
        # print(self.get_overlapping_segments) 
        if self.get_overlapping_segments:
            vids1 = pad_sequence([einops.rearrange(x[0][0],'C T H W -> T C H W') for x in batch])   
            vids2 = pad_sequence([einops.rearrange(x[0][1],'C T H W -> T C H W') for x in batch]) 
            if self.compact:
               vids1 = einops.rearrange(vids1, 'T B C H W -> B (T H W) C')
               vids2 = einops.rearrange(vids2, 'T B C H W -> B (T H W) C')
            else:
               vids1 = einops.rearrange(vids1, 'T B C H W -> B C T H W')
               vids2 = einops.rearrange(vids2, 'T B C H W -> B C T H W')
            vids = (vids1, vids2)
        else:
            vids = pad_sequence([einops.rearrange(x[0],'C T H W -> T C H W') for x in batch])

            if self.compact:
                vids = einops.rearrange(vids, 'T B C H W -> B (T H W) C')
            else:
                vids = einops.rearrange(vids, 'T B C H W -> B C T H W')
        # min_examplars = min([x[1].shape[1] for x in batch])
        # exemplars = torch.stack([x[1][:, :min_examplars] for x in batch]).squeeze(1)
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
    dat = Rep_count(select_rand_segment=False, compact=False, pool_tokens_factor=0.5, peak_at_random_location=False, get_overlapping_segments=True)
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
        print(f"It. {i} \n vid tokens: {item[0][0].shape} \n exem tokens: {item[1].shape} \n density map: {item[2].shape}:{item[3]} \n \n shot_num: {item[6]}")
        density_map_sum.append(item[3][0].item())
        # print(item[0][0].shape)
        # print(item[0][1].shape)
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
    # df = pd.read_csv('datasets/repcount/validtest_with_fps.csv')
    # df['density_map_sum'] = density_map_sum
    # df.to_csv('datasets/repcount/validtest_with_fps_new1.csv')
    #for i in range(7):
    #    if i in counts.keys() and i in density_maps_sum.keys():
    #        print(i,counts[i],density_maps_sum[i])

    # print(f"Avg clip dur: {sum(sum_clip_dur)/len(sum_clip_dur)} | Avg vid dur: {sum(sum_tot_dur)/len(sum_tot_dur)}")
    # print(f"Avg clip reps: {sum(sum_clip_counts)/len(sum_clip_counts)} | Avg vid counts: {sum(sum_tot_counts)/len(sum_tot_counts)}")
    # print(sum(fps)/len(fps))
    
    
        
    
