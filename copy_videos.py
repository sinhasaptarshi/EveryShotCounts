import os
import pandas as pd 

for split in ['train']:
    data = pd.read_csv(f'datasets/ucf-rep/new_{split}.csv')
    names = data['name']
    os.makedirs('../ucf526/train', exist_ok=True)
    for name in names:
        
        activity = name.split('/')[-2]
        video = name.split('/')[-1][:-4]
        path = os.path.join('../ucf526',activity, video)
        dest = os.path.join('../ucf526/train',activity)
        # print(name)
        os.makedirs(dest, exist_ok=True)
        os.system(f'mv {path} {dest}')