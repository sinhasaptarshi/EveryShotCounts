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
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=12, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--use_mae', action='store_true', help='Use mean absolute error as a loss function')

    parser.set_defaults(use_mae=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/FSC147/', type=str,
                        help='dataset path')
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str,
                     help='annotation json file')
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str,
                        help='data split json file')
    parser.add_argument('--class_file', default='ImageClasses_FSC147.txt', type=str,
                        help='class json file')
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str,
                        help='images directory')
    parser.add_argument('--gt_dir', default='gt_density_map_adaptive_384_VarV2', type=str,
                        help='ground truth directory')
    parser.add_argument('--output_dir', default='./data/out/fim6_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/CounTR/data/out/pre_4_dir/checkpoint__pretraining_199.pth',
                        help='resume from checkpoint')

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--num_gpus', default=4, type=int, help='number of gpus')

    # Logging parameters
    parser.add_argument('--log_dir', default='./logs/fim6_dir',
                        help='path where to tensorboard log')
    parser.add_argument("--title", default="CounTR_finetuning", type=str)
    parser.add_argument("--wandb", default="counting", type=str)
    parser.add_argument("--team", default="repetition_counting", type=str)
    parser.add_argument("--wandb_id", default=None, type=str)

    return parser



def main():
    # parser = argparse.ArgumentParser(
    #     description="Provide SlowFast video training and testing pipeline."
    # )
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None
    wandb_run = wandb.init(
                config=args,
                resume="allow",
                project=args.wandb,
                name=args.title,
                entity=args.team,
                tags=["CounTR", "finetuning"],
                id=args.wandb_id,
            )
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    dataset = Rep_count(cfg=cfg)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,num_workers=4,shuffle=True,pin_memory=False,drop_last=True)
    
    model = SupervisedMAE(cfg=cfg).cuda()
    model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
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
    model.train()
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    
    for epoch in range(args.epochs):
        train_loss = 0
        sample_size = 0
        for i, item in enumerate(tqdm(dataloader)):
            # print(item[0].shape)
            # print(item[1].shape)
            iter = (epoch * len(dataloader)) + i
            data = item[0].cuda()
            example = item[1].cuda()
            actual_counts = item[3].cuda()
            optimizer.zero_grad()
            y = model(data, example)
            predict_count = torch.sum(y, dim=1).type(torch.FloatTensor).cuda()
            loss2 = lossSL1(predict_count, actual_counts)  ###L1 loss between count and predicted count
            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts), actual_counts + 1e-1)) / \
                            predict_count.flatten().shape[0]    #### reduce the mean absolute error
            loss = lossMSE(y, item[2].cuda()) 
            if args.use_mae:
                loss = loss + loss3
            else:
                loss = loss + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss * data.shape[0]
            sample_size += data.shape[0]
            if iter % 10 == 0:
                print('Epoch {}  Batch {}/{} Train loss {}'.format(epoch, i, len(dataloader), train_loss/ sample_size))
        wandb_run.log({'epoch': epoch, 'train_loss': train_loss/sample_size})
    wandb_run.finish()

if __name__=='__main__':
    main()
