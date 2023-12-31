import torch
import torch.nn as nn
import numpy as np
import os, sys
from Rep_count_loader import Rep_count
from tqdm import tqdm
from video_mae_cross import SupervisedMAE
from video_memae import RepMem
from slowfast.utils.parser import load_config
import timm.optim.optim_factory as optim_factory
import argparse
import wandb
import torch.optim as optim
import math
import random
from util.lr_sched import adjust_learning_rate
import pandas as pd
from util.misc import NativeScalerWithGradNormCount as NativeScaler

torch.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--only_test', action='store_true',
                        help='Only testing')
    parser.add_argument('--trained_model', default='', type=str,
                        help='path to a trained model')
    parser.add_argument('--scale_counts', default=100, type=int, help='scaling the counts')



    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--save_path', default='./saved_models_fulldata', type=str, help="Path to save the model")

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--use_mae', action='store_true', help='Use mean absolute error as a loss function')

    parser.set_defaults(use_mae=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=5e-7, metavar='LR',
                        help='learning rate (peaklr)')
    parser.add_argument('--init_lr', type=float, default=8e-6, metavar='LR',
                        help='learning rate (initial lr)')
    parser.add_argument('--peak_lr', type=float, default=8e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--decay_milestones', type=list, default=[80, 160], help='milestones to decay for step decay function')
    parser.add_argument('--eval_freq', default=2, type=int)
    parser.add_argument('--cosine_decay', default=True, type=bool)

    # Dataset parameters
    parser.add_argument('--precomputed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='flag to specify if precomputed tokens will be loaded')
    parser.add_argument('--data_path', default='/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/LLSP/', type=str,
                        help='dataset path')
    parser.add_argument('--tokens_dir', default='saved_tokens_reencoded', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--exemplar_dir', default='exemplar_tokens_reencoded', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--gt_dir', default='gt_density_maps_recreated', type=str,
                        help='ground truth density map directory')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    # Training parameters    
    parser.add_argument('--seed', default=0, type=int)
    
    #parser.add_argument('--resume', default='/jmain02/home/J2AD001/wwp01/sxs63-wwp01/repetition_counting/CounTR/data/out/pre_4_dir/checkpoint__pretraining_199.pth', help='resume from checkpoint')
    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pyth', type=str)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
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
    
    parser.add_argument("--token_pool_ratio", default=0.4, type=float)

    return parser


def main():
    # parser = argparse.ArgumentParser(
    #     description="Provide SlowFast video training and testing pipeline."
    # )
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None
    g = torch.Generator()
    g.manual_seed(args.seed)
    if args.use_wandb:
        wandb_run = wandb.init(
                    config=args,
                    resume="allow",
                    project=args.wandb,
                    entity=args.team,
                    id=args.wandb_id,
                )
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    
    
    
    if args.precomputed:
        dataset_train = Rep_count(split="train",
                                  tokens_dir = args.tokens_dir,
                                  exemplar_dir = args.exemplar_dir,
                                  density_maps_dir = args.gt_dir,
                                  select_rand_segment=False, 
                                  compact=True, 
                                  pool_tokens_factor=args.token_pool_ratio)
        
        dataset_valid = Rep_count(split="valid",
                                  tokens_dir = args.tokens_dir,
                                  exemplar_dir = args.exemplar_dir,
                                  density_maps_dir = args.gt_dir,
                                  select_rand_segment=False, 
                                  compact=True, 
                                  pool_tokens_factor=args.token_pool_ratio)
    
        #dataset_test = Rep_count(cfg=cfg,split="test",data_dir=args.data_path)
    
        # Create dict of dataloaders for train and val
        dataloaders = {'train':torch.utils.data.DataLoader(dataset_train,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           pin_memory=False,
                                                           drop_last=False,
                                                           collate_fn=dataset_train.collate_fn,
                                                           worker_init_fn=seed_worker,
                                                           generator=g),
                       'val':torch.utils.data.DataLoader(dataset_valid,
                                                         batch_size=args.batch_size,
                                                         num_workers=args.num_workers,
                                                         shuffle=False,
                                                         pin_memory=False,
                                                         drop_last=False,
                                                         collate_fn=dataset_valid.collate_fn,
                                                         worker_init_fn=seed_worker,
                                                         generator=g),
                        'test':torch.utils.data.DataLoader(dataset_valid,
                                                         batch_size=1,
                                                         num_workers=args.num_workers,
                                                         shuffle=False,
                                                         pin_memory=False,
                                                         drop_last=False,
                                                         collate_fn=dataset_valid.collate_fn,
                                                         worker_init_fn=seed_worker,
                                                         generator=g)}
              
    scaler = torch.cuda.amp.GradScaler() # use mixed percision for efficiency
    # scaler = NativeScaler()
    
    model = SupervisedMAE(cfg=cfg,use_precomputed=args.precomputed).cuda()
    #model = RepMem().cuda()
    if args.num_gpus > 1:
        model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
    # param_groups = optim_factory.add_weight_decay(model, 5e-2)
    
    
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
    
    
    
    train_step = 0
    val_step = 0
    if args.only_test:
        model.load_state_dict(torch.load(args.trained_model)['model_state_dict'])
        model.eval()
        print(f"Testing")
        dataloader = dataloaders['test']
        gt_counts = list()
        predictions = list()
        predict_mae = list()
        predict_mse = list()
        clips = list()
        
        bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
        with tqdm(total=len(dataloader),bar_format=bformat,ascii='░▒█') as pbar:
            for i, item in enumerate(dataloader):
                data = item[0].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                example = item[1].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                actual_counts = item[3].cuda() # B x 1
                video_name = item[4]
                thw = item[5]
                with torch.no_grad():
                    y = model(data, example, thw, shot_num=1)
                    # y = torch.nn.functional.relu(y)
                # print(video_name[0])
                mse = ((y - density_map)**2).sum(-1)
                np.savez('predictions_smallsubset_noupsampling/'+video_name[0]+'.npz', y[0].cpu().numpy())
                np.savez('gt_smallsubset_noupsampling/'+video_name[0]+'.npz', density_map[0].cpu().numpy())
                predict_counts = torch.sum(y, dim=1).type(torch.FloatTensor).cuda() / args.scale_counts
                predictions.extend(predict_counts.detach().cpu().numpy())
                gt_counts.extend(actual_counts.detach().cpu().numpy())
                mae = torch.div(torch.abs(predict_counts - actual_counts), actual_counts + 1e-1)
                predict_mae.extend(mae.cpu().numpy())
                clips.append(data.shape[1]//(8*7*7))
                # print(predict_mae)

        predict_mae = np.array(predict_mae)
        predictions = np.array(predictions)
        gt_counts = np.array(gt_counts)
        # df = pd.read_csv('datasets/repcount/validtest_with_fps.csv')
        # df['predictions'] = predictions
        # df.to_csv('datasets/repcount/validtest_with_fps.csv')
        clips = np.array(clips)
        min = clips.min()
        max = clips.max()

        print(gt_counts)
        diff = np.abs(predictions - gt_counts)
        diff_z = np.abs(predictions.round() - gt_counts.round())
        print(f'Overall MAE: {predict_mae.mean()}')
        print(f'OBO: {(diff<=1).sum()/ len(diff)}')
        print(f'OBZ: {(diff_z==0).sum()/ len(diff)}')
        print(f'RMSE: {np.sqrt((diff**2).mean())}')
        for counts in range(1,7):
            print(f"MAE for count {counts} is {predict_mae[gt_counts <= counts].mean()}")
        for duration in np.linspace(min, max, 10)[1:]:
            print(f"MAE for duration less that {duration} is {predict_mae[clips <= duration].mean()}")

        return

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, betas=(0.9, 0.95))
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    milestones = [i for i in range(0, args.epochs, 40)]
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.8)
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    best_rmse = np.inf

    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epochs):
        # if epoch <= args.warmup_epochs:
        #     lr = args.init_lr + ((args.peak_lr - args.init_lr) *  epoch / args.warmup_epochs)  ### linear warmup
        # else:
        #     if args.cosine_decay:
        #         lr = args.end_lr + (args.peak_lr - args.end_lr) * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))) / 2  ##cosine annealing
        #     else:
        #         if epoch in args.decay_milestones:
        #             lr = lr * 0.1
        
        # print(lr)
        scheduler.step()
        

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
                off_by_zero = 0
                off_by_one = 0
                mse = 0
                count = 0
                mae = 0
                
                bformat='{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
                dataloader = dataloaders[phase]
                with tqdm(total=len(dataloader),bar_format=bformat,ascii='░▒█') as pbar:
                    for i, item in enumerate(dataloader):
                        if phase == 'train':
                            train_step+=1
                            # if (i+1) % args.accum_iter == 0:
                            #     lr = adjust_learning_rate(optimizer, (i + 1) / len(dataloader) + epoch, args)
                            #     print(lr)
                        elif phase == 'val':
                            val_step+=1
                        with torch.cuda.amp.autocast(enabled=True):
                            data = item[0].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                            # print(data.shape)
                            example = item[1].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                            density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                            actual_counts = item[3].cuda() # B x 1
                            thw = item[5]
                            # print(actual_counts)
            
                            
                            y = model(data, example, thw, shot_num=1)
                            # y = torch.nn.functional.relu(y)
                            if phase == 'train':
                                mask = np.random.binomial(n=1, p=0.8, size=[1,density_map.shape[1]])
                            else:
                                mask = np.ones([1, density_map.shape[1]])
                            
                            masks = np.tile(mask, (density_map.shape[0], 1))
                            
                            
                            masks = torch.from_numpy(masks).cuda()
                            loss = (y - density_map) ** 2
                            
                            # loss = (loss * masks / density_map.sum(1, keepdims=True)).sum() / density_map.shape[0]
                            loss = ((loss * masks) / density_map.shape[1]).sum() / density_map.shape[0]
                            # loss = (loss * masks).sum() / density_map.shape[0]
                            
                            #print(item[4],data.shape,y.shape,density_map.shape)
                            predict_count = torch.sum(y, dim=1).type(torch.FloatTensor).cuda() / args.scale_counts # sum density map
                            # print(predict_count)
                            if phase == 'val':
                                ground_truth.append(actual_counts.detach().cpu().numpy())
                                predictions.append(predict_count.detach().cpu().numpy())
                            
                            loss2 = lossSL1(predict_count, actual_counts)  ###L1 loss between count and predicted count
                            # actual_counts /= 60
                            # predict_count /= 60
                            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts), actual_counts + 1e-1)) / \
                            predict_count.flatten().shape[0]    #### reduce the mean absolute error
                            # loss1 = lossMSE(y, density_map) 
                            loss1 = loss
                            
                            # if args.use_mae:
                            #     loss = loss1 + loss3
                            #     loss2 = 0 # Set to 0 for clearer logging
                            # else:
                            #     loss = loss1 + loss2
                            #     loss3 = 0 # Set to 0 for clearer logging
                            # loss = loss1
                            if phase=='train':
                                loss /= args.accum_iter
                                # scaler(loss, optimizer, parameters=model.parameters(), update_grad=(i + 1) % args.accum_iter == 0)

                                # scaler.scale(loss).backward()
                                loss.backward()
                                if (i + 1) % args.accum_iter == 0: ### accumulate gradient
                                    # scaler.step(optimizer)
                                    # scaler.update()
                                    optimizer.step()
                                    optimizer.zero_grad()
                            
                            epoch_loss = loss.item()
                            count += data.shape[0]
                            total_loss_all += loss.item() * data.shape[0]
                            total_loss1 += loss.item() * data.shape[0]
                            total_loss2 += loss2 * data.shape[0]
                            total_loss3 += loss3.item() * data.shape[0]
                            # actual_counts = actual_counts / 60
                            # predict_count = predict_count / 60
                            off_by_zero += (torch.abs(actual_counts.round() - predict_count.round()) ==0).sum().item()  ## off by zero
                            off_by_one += (torch.abs(actual_counts - predict_count) <=1 ).sum().item()   ## off by one
                            mse += ((actual_counts - predict_count)**2).sum().item()
                            mae += torch.sum(torch.div(torch.abs(predict_count - actual_counts), (actual_counts) + 1e-1)).item()
                            
                            # if args.use_wandb:
                            #     if phase == 'train':
                            #         wandb_run.log({
                            #             "train_step": train_step,
                            #             "train_loss_per_step": loss,
                            #             "train_loss1_per_step": loss1,
                            #             "train_loss2_per_step": loss2,
                            #             "train_loss3_per_step": loss3,
                            #             # "lr": lr
                            #         })
                            #     if phase == 'val':
                            #         wandb.log({
                            #             "val_step": val_step,
                            #             "val_loss_per_step": loss,
                            #             "val_loss1_per_step": loss1,
                            #             "val_loss2_per_step": loss2,
                            #             "val_loss3_per_step": loss3
                            #         })
                            
                            pbar.set_description(f"EPOCH: {epoch:02d} | PHASE: {phase} ")
                            pbar.set_postfix_str(f" LOSS: {total_loss_all/count:.2f} | MAE:{mae/count:.2f} | LOSS ITER: {loss.item():.2f} | OBZ: {off_by_zero/count:.2f} | OBO: {off_by_one/count:.2f}")
                            pbar.update()
                             
                
                if args.use_wandb:
                    if phase == 'train':
                        wandb.log({"epoch": epoch,
                            # "lr": lr,
                            "train_loss": total_loss_all/float(count), 
                            "train_loss1": total_loss1/float(count), 
                            "train_loss2": total_loss2/float(count), 
                            "train_loss3": total_loss3/float(count), 
                            "train_obz": off_by_zero/count,
                            "train_obo": off_by_one/count,
                            "train_rmse": np.sqrt(mse/count),
                            "train_mae": mae/count
                        })
                    
                    if phase == 'val':
                        if not os.path.isdir(args.save_path):
                            os.makedirs(args.save_path)
                        wandb.log({"epoch": epoch, 
                            "val_loss": total_loss_all/float(count), 
                            "val_loss1": total_loss1/float(count), 
                            "val_loss2": total_loss2/float(count), 
                            "val_loss3": total_loss3/float(count), 
                            "val_obz": off_by_zero/count, 
                            "val_obo": off_by_one/count,
                            "val_mae": mae/count, 
                            "val_rmse": np.sqrt(mse/count)
                        })
                        if np.sqrt(mse/count) < best_rmse:
                            best_rmse = np.sqrt(mse/count)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'best.pyth'))
                        torch.save({
                                # 'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'current.pyth'))
    
    
    if args.use_wandb:                                   
        wandb_run.finish()

if __name__=='__main__':
    main()