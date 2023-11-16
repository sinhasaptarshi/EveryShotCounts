import pandas as pd
import cv2
import numpy as np
import os

for t in ['train', 'valid', 'test']:
    print(f'starting {t}')
    df = pd.read_csv('~/repetition_counting/CounTR/datasets/repcount/{}_with_fps.csv'.format(t))
    df['num_frames'] = 0
    video_names = df['name'].values
    for i, name in enumerate(video_names):
        # print(os.path.exists(f"/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/{t}/{name}"))
        cap = cv2.VideoCapture(f"/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/{t}/{name}")
        df['num_frames'].iloc[i] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    df.to_csv('~/repetition_counting/CounTR/datasets/repcount/{}_with_fps.csv'.format(t))