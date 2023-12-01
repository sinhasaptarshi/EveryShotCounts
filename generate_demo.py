import pandas as pd
import numpy as np
import os, sys
import random

max_data = 100
train_df = pd.read_csv('datasets/repcount/train_with_fps.csv')
train_df = train_df[train_df['count'].notna()]

# self.df = self.df[self.df['count'] < 5] ### remove videos with more than 5 repetitions
# self.df = self.df[self.df['fps'] >= 10]
print(f"Length of the original dataframe: {train_df.shape[0]}")
train_df = train_df[train_df['num_frames'] > 64]
print(f"Length of the dataframe when videos with less than 64 frames are removed: {train_df.shape[0]}")
train_df = train_df[train_df['name']!='stu1_10.mp4']
train_df = train_df[train_df['count'] > 0] # remove no reps
print(f"Length of the dataframe videos with no counts are further dropped: {train_df.shape[0]}")

### selecting videos with count <=6
selected_videos = train_df[train_df['count']<=6]
selected_videos['segment_start'] = 0
selected_videos['segment_end'] = selected_videos['num_frames']
rem_videos = train_df[(train_df['count'] > 6)] ## videos with counts more than 6
rem_videos.reset_index()
counts = selected_videos['count']
unique_counts, freq = np.unique(counts, return_counts=True)

for count, freq in zip(unique_counts.astype(int), freq):
    to_add = max_data - freq
    print(f"Repetition counts: {count} number of videos: {freq} | num videos to be added {to_add} ")
    # if count <= 3:
    #     min_rep = 64
    #     potential_clips = []
    # select = rem_videos.sample(to_add)
    cnt = 0
    rand_indices = list(range(rem_videos.shape[0])) # list of indices in random order
    random.shuffle(rand_indices)
    idx = 0
    #select_rep = 1
    while cnt < to_add:
        row = rem_videos.iloc[rand_indices[idx]]
        
        # get all the start and ends timestamps
        clc = np.array([int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])])
        starts = clc[0::2]
        ends = clc[1::2]
        if np.random.rand() > 0.4:
            if (starts[1:] - ends[0:-1]).max() <= 0:  ### checking if the selected video has interruptions
                continue
        
        # get the duration of the repetition
        rep_durations = ends - starts
        
        if rep_durations.min() > 64:
            new_row = row.copy()
            for key in new_row.keys():
                if 'L' in key:
                    new_row[key] = np.nan
            
            select_start = random.choice(list(range(0,len(starts) - count))) # select repetition start randomly
            
            select_starts = starts[select_start: select_start + count] # get random segment with `count` repetitions
            select_ends = ends[select_start: select_start + count]
            
            if (select_ends[-1]-select_starts[0] < 64):
                print(f'Selected random segment has less than {64} frames -> dropping (segment stats shown below)')
                print(f"count: {count} \n starts: {select_starts} \n ends: {select_ends} \n frames: {select_ends[-1]-select_starts[0]}")
                continue
            new_row['segment_start'] = (select_starts[0] // 64) * 64
            new_row['segment_end'] = (select_ends[-1] // 64 + 1) * 64  ### getting segment start and end
            for i in range(count):
                new_row[f"L{2*i + 1}"] = select_starts[i]
                new_row[f"L{2*i + 2}"] = select_ends[i]
            new_row['count'] = int(count)
            cnt += 1
            selected_videos = pd.concat([selected_videos, new_row.to_frame().T], ignore_index=True)
        
        idx = (idx + 1)%len(rem_videos)
        
        #if idx == 0:
        #    select_rep += 1
            
selected_videos.reset_index()
print(selected_videos.columns)
counts = selected_videos['count']
print(np.unique(counts, return_counts=True))   
selected_videos.to_csv('demo.csv')