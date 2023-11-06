import torch
import torch.nn as nn
import numpy as np
import os
from Rep_count import Rep_count
from tqdm import tqdm
from video_mae_cross import SupervisedMAE
from slowfast.utils.parser import load_config
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    args = parser.parse_args()
    args.opts = None
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    dataset = Rep_count(cfg=cfg)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=1,shuffle=False,pin_memory=False,drop_last=True)
    
    model = SupervisedMAE(cfg=cfg).cuda()

    for i, item in enumerate(tqdm(dataloader)):
        # print(item[0].shape)
        # print(item[1].shape)
        data = item[0].cuda()
        example = item[1].cuda()
        y = model(data, example)


if __name__=='__main__':
    main()
