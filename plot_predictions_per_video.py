import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

df = pd.read_csv('datasets/repcount/validtest_with_fps.csv')
videos = df['name']
for vid in videos:
    vid_ = f'../LLSP/validtest/{vid}'


    cap = cv2.VideoCapture(vid_)
    
    npzfile = vid.replace('.mp4', '.npz')
    predictions = np.load(f'predictions/{npzfile}')['arr_0'] / 60

    gt = np.load(f'gt_density_maps_recreated/{npzfile}')['arr_0']

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(length)

    print(predictions.shape)
# print(length - predictions.shape[1])

    predictions = np.concatenate([predictions[0], np.zeros(length - len(predictions[0]))])
    gt = np.concatenate([gt, np.zeros(length - len(gt))])

    for i in range(length):
        plt.plot(range(length), predictions)
        plt.ylabel('prediction')
        plt.xlabel('frame number')
        plt.axvline(x = i, color = 'r', label = 'axvline - full height')
        plt.axvline(x = i-1, color = 'w', label = 'axvline - full height')
        plt.savefig('plots/plot_'+str(i).zfill(4)+'.png')
    os.system('ffmpeg -r {} -f image2 -s 224x224 -i plots/plot_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p outputs/prediction_{}.mp4'.format(fps, vid[:-4]))
    os.system('rm -rf plots/*')

    for i in range(length):
        plt.plot(range(length), gt)
        plt.ylabel('gt')
        plt.xlabel('frame number')
        plt.axvline(x = i, color = 'r', label = 'axvline - full height')
        plt.axvline(x = i-1, color = 'w', label = 'axvline - full height')
        plt.savefig('plots/plot_'+str(i).zfill(4)+'.png')
    os.system('ffmpeg -r {} -f image2 -s 224x224 -i plots/plot_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p outputs/gt_{}.mp4'.format(fps, vid[:-4]))
    os.system('rm -rf plots/*')