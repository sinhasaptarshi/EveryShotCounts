import numpy as np 
import pandas as pd
import scipy.io 
import os

annotation_path = '../annotations'
video_path = '/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/UCF-101'

for split in ['train', 'val']:
    names = []
    
    columns = ['name', 'counts', 'num_frames', 'start_frame', 'end_frame']
    columns.extend([f'L{i}' for i in range(1, 301)])
    
    # print(df.columns)
    # train_df[c] = columns 

    annotations = os.path.join(annotation_path, split)
    annotation_files = os.listdir(annotations)
    df = pd.DataFrame(np.nan, index=range(len(annotation_files)), columns=columns)
    print
    for i, filename in enumerate(annotation_files):
        folder = filename.split('_')[1]
        filepath = os.path.join(annotations, filename)
        mat = scipy.io.loadmat(filepath)
        ann = mat['label'][0,0]
        num_frames = ann['duration'][0, 0] ## duration of video
        bound = ann['temporal_bound'][:,0] #### start and end of repetitions
        start_frame = ann['start_frame'][0,0]
        end_frame = ann['end_frame'][0,0]
        print(bound)
        count = len(bound) - 1
        df['counts'][i] = count
        df['start_frame'][i] = start_frame
        df['end_frame'][i] = end_frame
        video_id = filename.replace('.mat','.avi')
        video_loc = os.path.join(video_path, folder, video_id)
        df['name'][i] = video_loc
        df['num_frames'][i] = num_frames
        for j in range(count):
            df[f'L{2*j + 1}'][i] = bound[j]
            df[f'L{2*j + 2}'][i] = bound[j+1]
        # print(df)
        if not os.path.exists(video_loc):
            print(video_loc)
    df.to_csv(f'datasets/ucf-rep/new_{split}.csv')