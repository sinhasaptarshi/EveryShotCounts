import torch
import torch.nn as nn
import numpy as np
import os
from Rep_count import Rep_count
from tqdm import tqdm
from video_mae_cross import SupervisedMAE
from slowfast.utils.parser import load_config
import timm.optim.optim_factory as optim_factory
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    args = parser.parse_args()
    args.opts = None
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    dataset = Rep_count(cfg=cfg)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=6,num_workers=2,shuffle=True,pin_memory=False,drop_last=True)
    
    model = SupervisedMAE(cfg=cfg).cuda()
    param_groups = optim_factory.add_weight_decay(model, 5e-2)
    optimizer = torch.optim.AdamW(param_groups, lr=1e-6, betas=(0.9, 0.95))
    state_dict = torch.load('pretrained_models/VIT_B_16x4_MAE_PT.pyth')['model_state']
    for name, param in state_dict.items():
        if name in model.state_dict().keys():
            if 'decoder' not in name:
                print(name)
                # new_name = name.replace('quantizer.', '')
                model.state_dict()[name].copy_(param)
    # model.load_state_dict(state_dict, strict=False)
    # print(state_dict.keys())

    loss1 = nn.MSELoss().cuda()
    for i, item in enumerate(tqdm(dataloader)):
        # print(item[0].shape)
        # print(item[1].shape)
        data = item[0].cuda()
        example = item[1].cuda()
        optimizer.zero_grad()
        y = model(data, example)
        loss = loss1(y, item[2].cuda())
        loss.backward()
        optimizer.step()
        print(loss)


if __name__=='__main__':
    main()
