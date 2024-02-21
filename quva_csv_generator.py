import pandas as pd
import numpy as np 
import os
import cv2


video_path = os.path.join('QUVARepetitionDataset/videos')
annotations_path = os.path.join('QUVARepetitionDataset/annotations')

videos = os.listdir(video_path)
print(len(videos))

columns = ['name', 'counts', 'num_frames']
columns.extend([f"L{i}" for i in range(1, 301)])


df = pd.DataFrame(columns=columns, index=range(len(videos)))
# df.to_csv('datasets/quva/new_test.csv')
for i, video in enumerate(videos):
    if video.startswith('._'):
        continue
    # print(video)
    video_read = os.path.join(video_path, video)
    cap = cv2.VideoCapture(video_read)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anno_pth = os.path.join(annotations_path, video.replace('.mp4', '.npy'))
    annotations = np.load(anno_pth)
    print(annotations)
    df['name'][i] = video
    df['counts'][i] = len(annotations)
    df['num_frames'][i] = length
    annotations = np.concatenate([[0], annotations])
    for k in range(len(annotations)-1):
        df[f'L{2 * k + 1}'][i] = annotations[k]
        df[f'L{2 * k + 2}'][i] = annotations[k+1]
df.to_csv('datasets/quva/new_test.csv')



# print(df)
print(len(df))


