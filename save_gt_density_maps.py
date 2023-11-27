import pandas as pd
import numpy  as np
import scipy.ndimage as ndimage

import os

for split in ['train', 'validtest']:
    df = pd.read_csv(f"datasets/repcount/{split}_with_fps.csv")
    for index in range(len(df)):
        row = df.iloc[index]
        clc = np.array([int(float(row[key])) for key in row.keys() if 'L' in key and not np.isnan(row[key])])
        starts = clc[0::2]
        ends = clc[1::2]
        count = row['count']
        num_frames = row['num_frames']
        video_name = row['name'].replace('.mp4','.npz')
        print(video_name)

    
        gt_density = np.zeros(num_frames)

        for i in range(len(starts)):
            if starts[i] == ends[i]:
                continue
            segment = np.zeros(ends[i]-starts[i])
            segment[len(segment)//2] = 1
            segment = ndimage.gaussian_filter1d(segment, sigma=len(segment)/6, order = 0)
            gt_density[starts[i]:(ends[i])] = segment
        
        if not os.path.isdir(f"gt_density_maps"):
            os.makedirs("gt_density_maps")
        
        np.savez(f"gt_density_maps/{video_name}", gt_density)

