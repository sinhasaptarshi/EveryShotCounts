import torch
import torch.nn as nn
import numpy as np
import os
from Rep_count_loader import Rep_count
from tqdm import tqdm
from video_mae_cross import SupervisedMAE
from video_memae import RepMem
from slowfast.utils.parser import load_config
import timm.optim.optim_factory as optim_factory
import argparse
import wandb
import torch.optim as optim

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--save_path', default='./saved_models', type=str, help="Path to save the model")

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
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--eval_freq', default=1, type=int)

    # Dataset parameters
    parser.add_argument('--data_path', default='/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/', type=str,
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
    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pyth', type=str)

    # Training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
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
    parser.add_argument("--use_wandb", default=True, type=lambda x: (str(x).lower() == 'true'))
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
    if args.use_wandb:
        wandb_run = wandb.init(
                    config=args,
                    resume="allow",
                    project=args.wandb,
                    entity=args.team,
                    id=args.wandb_id,
                )
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    
    
    
    dataset_train = Rep_count(cfg=cfg,split="train",data_dir=args.data_path)
    dataset_val = Rep_count(cfg=cfg,split="valid",data_dir=args.data_path)
    dataset_test = Rep_count(cfg=cfg,split="test",data_dir=args.data_path)
    
    # Create dict of dataloaders for train and val
    dataloaders = {'train':torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=args.batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=False,
                                                       drop_last=True),
                   'val':torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,
                                                     shuffle=False,
                                                     pin_memory=False,
                                                     drop_last=True)}
              
    scaler = torch.cuda.amp.GradScaler() # use mixed percision for efficiency
    
    model = SupervisedMAE(cfg=cfg).cuda()
    #model = RepMem().cuda()
    
    model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
    param_groups = optim_factory.add_weight_decay(model, 5e-2)
    
    
    # if args.pretrained_encoder:
    #     state_dict = torch.load(args.pretrained_encoder)['model_state']
    # else:
    #     state_dict = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth')['model_state']

    # loaded=0
    # for name, param in state_dict.items():
    #     if 'module.' not in name: # fix for dataparallel
    #         name = 'module.'+name
    #     if name in model.state_dict().keys():
    #         if 'decoder' not in name:
    #             loaded += 1
    #             # new_name = name.replace('quantizer.', '')
    #             model.state_dict()[name].copy_(param)
    # print(f"--- Loaded {loaded} params from statedict ---")
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.blr, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    
    train_step = 0
    val_step = 0
    
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        print(f"Epoch: {epoch:02d}")
        for phase in ['train', 'val']:
            if phase == 'val':
                if epoch % args.eval_freq != 0:
                    continue
                model.eval()
                ground_truth = list()
                predictions = list()
            else:
                model.train()
            
            with torch.set_grad_enabled(phase == 'train'):
                total_loss_all = 0
                total_loss1 = 0
                total_loss2 = 0
                total_loss3 = 0
                off_by_one = 0
                count = 0
                
                bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
                dataloader = dataloaders[phase]
                with tqdm(total=len(dataloader),bar_format=bformat,ascii='░▒█') as pbar:
                    for i, item in enumerate(dataloader):
                        if phase == 'train':
                            train_step+=1
                        elif phase == 'val':
                            val_step+=1
                        with torch.cuda.amp.autocast(enabled=True):
                            data = item[0].cuda() # B x C x T x H x W
                            print(data.shape)
                            example = item[1].cuda() # B x C x T' x H x W
                            actual_counts = item[3].cuda() # B x 1
            
                            optimizer.zero_grad()
                            y = model(data, example)
                            predict_count = torch.sum(y, dim=1).type(torch.FloatTensor).cuda() # sum density map
                            if phase == 'val':
                                ground_truth.append(actual_counts.detach().cpu().numpy())
                                predictions.append(predict_count.detach().cpu().numpy())
                            
                            loss2 = lossSL1(predict_count, actual_counts)  ###L1 loss between count and predicted count
                            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts), actual_counts + 1e-1)) / \
                            predict_count.flatten().shape[0]    #### reduce the mean absolute error
                            
                            loss1 = lossMSE(y, item[2].cuda()) 
                            
                            if args.use_mae:
                                loss = loss1 + loss3
                                loss2 = 0 # Set to 0 for clearer logging
                            else:
                                loss = loss1 + loss2
                                loss3 = 0 # Set to 0 for clearer logging
                            if phase=='train':
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            
                            epoch_loss = loss.item()
                            count += data.shape[0]
                            total_loss_all += loss.item() * data.shape[0]
                            total_loss1 += loss1.item() * data.shape[0]
                            total_loss2 += loss2 * data.shape[0]
                            total_loss3 += loss3.item() * data.shape[0]
                            off_by_one += (torch.abs(actual_counts - predict_count) <=1).sum().item()
                            
                            if args.use_wandb:
                                if phase == 'train':
                                    wandb_run.log({
                                        "train_step": train_step,
                                        "train_loss_per_step": loss,
                                        "train_loss1_per_step": loss1,
                                        "train_loss2_per_step": loss2,
                                        "train_loss3_per_step": loss3
                                    })
                                if phase == 'val':
                                    wandb.log({
                                        "val_step": val_step,
                                        "val_loss_per_step": loss,
                                        "val_loss1_per_step": loss1,
                                        "val_loss2_per_step": loss2,
                                        "val_loss3_per_step": loss3
                                    })
                            
                            pbar.set_description(f"EPOCH: {epoch:02d} | PHASE: {phase} ")
                            pbar.set_postfix_str(f" LOSS: {total_loss_all/count:.2f} | MAE:{total_loss3/count:.2f} | LOSS ITER: {loss.item():.2f} | OBO: {off_by_one/count:.2f}")
                            pbar.update()
                             
                
                if args.use_wandb:
                    if phase == 'train':
                        wandb.log({"epoch": epoch, 
                            "train_loss": total_loss_all/float(count), 
                            "train_loss1": total_loss1/float(count), 
                            "train_loss2": total_loss2/float(count), 
                            "train_loss3": total_loss3/float(count), 
                        })
                    
                    if phase == 'val':
                        if not os.path.isdir(args.save_path):
                            os.makedirs(args.save_path)
                        wandb.log({"epoch": epoch, 
                            "val_loss": total_loss_all/float(count), 
                            "val_loss1": total_loss1/float(count), 
                            "val_loss2": total_loss2/float(count), 
                            "val_loss3": total_loss3/float(count), 
                            "obo": off_by_one/count, 
                            "mae": total_loss3/count, 
                        })
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, os.path.join(args.save_path, 'checkpoint_epoch_{}.pyth'.format(str(epoch).zfill(5))))
    
    if args.use_wandb:                                   
        wandb_run.finish()

if __name__=='__main__':
    main()
