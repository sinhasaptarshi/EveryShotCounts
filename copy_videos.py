import os
import pandas as pd 

for split in ['val']:
    data = pd.read_csv(f'datasets/ucf-rep/new_{split}.csv')
    names = data['name']
    os.makedirs('../ucf526/val', exist_ok=True)
    for name in names:
        
        activity = name.split('/')[-2]
        video = name.split('/')[-1][:-4]
        path = os.path.join('../ucf526',activity, video)
        dest = os.path.join('../ucf526/val',activity)
        # print(name)
        os.makedirs(dest, exist_ok=True)
        os.system(f'mv {path} {dest}')