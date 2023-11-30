import pandas as pd
import numpy as np
import os


max_data = 100

train_df = pd.read_csv('datasets/repcount/train_with_fps.csv')
train_df = train_df[train_df['count'].notna()]
# self.df = self.df[self.df['count'] < 5] ### remove videos with more than 5 repetitions
# self.df = self.df[self.df['fps'] >= 10]
train_df = train_df[train_df['num_frames'] > 64]
train_df = train_df[train_df['name']!='stu1_10.mp4']
train_df = train_df[train_df['count'] > 0] # remove no reps

### selecting videos with count <=6
selected_videos = train_df[train_df['count']<=6]
rem_videos = train_df[(train_df['count'] > 6)] ## videos with counts more than 6
rem_videos.reset_index()







counts = selected_videos['count']

unique_counts, freq = np.unique(counts, return_counts=True)
for count, freq in zip(unique_counts.astype(int), freq):
    to_add = max_data - freq
    # if count <= 3:
    #     min_rep = 64
    #     potential_clips = []
    # select = rem_videos.sample(to_add)

    cnt = 0
    idx = 0
    select_rep = 1
    while cnt <= to_add:
        row = rem_videos.iloc[idx]
        # print(row)
    # for iter, row in rem_videos.iterrows():

        clc = np.array([int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])])
        starts = clc[0::2]
        ends = clc[1::2]
        rep_durations = ends - starts
        if rep_durations.min() > 64:
            if cnt == to_add:
                break
            new_row = row
            for key in new_row.keys():
                if 'L' in key:
                    new_row[key] = np.nan
            # select_rep = np.random.choice(len(starts) - count + 1, replace=False)
            # select_rep = len(starts) // 2
            select_starts = starts[select_rep: select_rep + count]
            select_ends = ends[select_rep: select_rep + count]
            for i in range(count):
                new_row['L{}'.format(2*i + 1)] = select_starts[i]
                new_row['L{}'.format(2*i + 2)] = select_ends[i]

            new_row['count'] = int(count)
            cnt += 1
            selected_videos = pd.concat([selected_videos, new_row.to_frame().T], ignore_index=True)
        idx = (idx + 1)%len(rem_videos)
        if idx == 0:
            select_rep += 1
            
selected_videos.reset_index()
print(selected_videos.columns)
counts = selected_videos['count']
print(np.unique(counts, return_counts=True))   
selected_videos.to_csv('demo.csv')




# print(freq.mean())

# print(freq)
# print(((counts == 0) & (num_frames >= 512)).sum())
# new_df = pd.DataFrame([])
# new_df['name'] = []
# selected_names = []


# for counts in range(1,7):
#     select = train_df[train_df['count'] == counts]
#     for iter, row in select.iterrows():

#         print(row['name'])
#         # new_df['name'].append(row['name'])