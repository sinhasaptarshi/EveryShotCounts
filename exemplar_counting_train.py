import torch
import torch.nn as nn
torch.cuda.empty_cache()
import numpy as np
import os, sys
# from Rep_count_loader import Rep_count
from Repcount_multishot_loader import Rep_count

from Countix_multishot_loader import Countix
from UCFRep_multishot_loader import UCFRep
from tqdm import tqdm
from video_mae_cross import SupervisedMAE as SupervisedMAE
from video_mae_cross_full_attention import SupervisedMAE as SupervisedMAE_fullattention
from slowfast.utils.parser import load_config
import timm.optim.optim_factory as optim_factory
from util.pos_embed import get_2d_sincos_pos_embed
import argparse
import wandb
import torch.optim as optim
import math
import random
from util.lr_sched import adjust_learning_rate
import pandas as pd
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 
from matplotlib import rc

torch.manual_seed(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--encodings', default='swin', type=str, help=['swin','mae'])
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--only_test', action='store_true',
                        help='Only testing')
    parser.add_argument('--trained_model', default='', type=str,
                        help='path to a trained model')
    parser.add_argument('--scale_counts', default=100, type=int, help='scaling the counts')

    parser.add_argument('--dataset', default='RepCount', type=str, help='Repcount, Countix, UCFRep')

    parser.add_argument('--get_overlapping_segments', default=False, type=bool, help='whether to get overlapping segments')

    parser.add_argument('--peak_at_random_locations', default=False, type=bool, help='whether to have density peaks at random locations')

    parser.add_argument('--multishot', action='store_true')

    parser.add_argument('--full_attention', action='store_true')

    parser.add_argument('--iterative_shots', action='store_true', help='will show the examples one by one')

    parser.add_argument('--no_exemplars', action='store_true', help='to not use exemplars')

    parser.add_argument('--density_peak_width', default=1.0, type=float, help='sigma for the peak of density maps, lesser sigma gives sharp peaks')


    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--save_path', default='./saved_models_repcountfull', type=str, help="Path to save the model")

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
    parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
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
    parser.add_argument('--data_path', default='/raid/local_scratch/sxs63-wwp01/', type=str,
                        help='dataset path')
    parser.add_argument('--slurm_job_id', default=None, type=str,
                        help='job id')
    parser.add_argument('--tokens_dir', default='saved_tokens_reencoded', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--exemplar_dir', default='exemplar_tokens_reencoded', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--gt_dir', default='gt_density_maps_recreated', type=str,
                        help='ground truth density map directory')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='p, cut off to decide if select exemplar from different video')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    # Training parameters    
    parser.add_argument('--seed', default=0, type=int)
    
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
    parser.add_argument("--title", default="", type=str)
    parser.add_argument("--use_wandb", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--wandb", default="", type=str)
    parser.add_argument("--team", default="", type=str)
    parser.add_argument("--wandb_id", default='', type=str)
    
    parser.add_argument("--token_pool_ratio", default=0.6, type=float)
    parser.add_argument("--rho", default=0.7, type=float)
    parser.add_argument("--window_size", default=(4,7,7), type=int, nargs='+', help='window size for windowed self attention')

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    args.window_size = tuple(args.window_size)
    args.opts = None
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    
    '''
    create dataloaders
    '''
    if args.precomputed:
        if args.dataset == 'Countix':
            dataset_train = Countix(split="train",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings)
            
            dataset_valid = Countix(split="val",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings)
            dataset_test = Countix(split="test",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    encodings=args.encodings)
        elif args.dataset == 'RepCount':
            dataset_train = Rep_count(split="train",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    threshold=args.threshold)
            
            dataset_valid = Rep_count(split="valid",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)
            dataset_test = Rep_count(split="test",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)

        elif args.dataset == 'UCFRep':
            dataset_train = UCFRep(split="train",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    threshold=args.threshold)
            
            dataset_valid = UCFRep(split="valid",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)
            dataset_test = UCFRep(split="test",
                                    tokens_dir = args.tokens_dir,
                                    exemplar_dir = args.exemplar_dir,
                                    density_maps_dir = args.gt_dir,
                                    select_rand_segment=False, 
                                    compact=True, 
                                    pool_tokens_factor=args.token_pool_ratio,
                                    peak_at_random_location=args.peak_at_random_locations,
                                    get_overlapping_segments=args.get_overlapping_segments,
                                    multishot=args.multishot,
                                    density_peak_width = args.density_peak_width)
        # Create dict of dataloaders for train and val
        dataloaders = {'train':torch.utils.data.DataLoader(dataset_train,
                                                           batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           pin_memory=False,
                                                           drop_last=False,
                                                           collate_fn=dataset_train.collate_fn,
                                                           worker_init_fn=seed_worker,
                                                           persistent_workers=True,
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
                        'test':torch.utils.data.DataLoader(dataset_test,
                                                         batch_size=1,
                                                         num_workers=args.num_workers,
                                                         shuffle=False,
                                                         pin_memory=False,
                                                         drop_last=False,
                                                         collate_fn=dataset_valid.collate_fn,
                                                         worker_init_fn=seed_worker,
                                                         generator=g)}
              
    # scaler = torch.cuda.amp.GradScaler() # use mixed percision for efficiency
    # scaler = NativeScaler()
    if args.full_attention:
        model = SupervisedMAE_fullattention(cfg=cfg,use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio, iterative_shots=args.iterative_shots, encodings=args.encodings, no_exemplars=args.no_exemplars, window_size=args.window_size).cuda() 
    else:
        model = SupervisedMAE(cfg=cfg,use_precomputed=args.precomputed, token_pool_ratio=args.token_pool_ratio, iterative_shots=args.iterative_shots, encodings=args.encodings, no_exemplars=args.no_exemplars).cuda()
    if args.num_gpus > 1:
        model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])
    
    train_step = 0
    val_step = 0
    if args.only_test:   #### only for testing
        model.load_state_dict(torch.load(args.trained_model)['model_state_dict'])  ### load trained model
        videos = []
        loss = []
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
                if args.get_overlapping_segments:
                    data, data2 = item[0][0], item[0][1]
                else:
                    data = item[0].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                example = item[1].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                actual_counts = item[3].cuda() # B x 1
                video_name = item[4]
                
                videos.append(video_name[0])

                
                shot_num = item[6][0]
                b, n, c = data.shape
                
                thw = item[5]
                with torch.no_grad():
                    if args.get_overlapping_segments:
                        data = data.cuda().type(torch.cuda.FloatTensor)
                        data2 = data2.cuda().type(torch.cuda.FloatTensor)
                        pred1 = model(data, example, thw, shot_num=shot_num)
                        pred2 = model(data2, example, thw, shot_num=shot_num)
                        if pred1.shape != pred2.shape:
                            pred2 = torch.cat([torch.zeros(1, 4).cuda(), pred2], 1)
                        else:
                            print('equal')
                        pred = (pred1 + pred2) / 2
                    else:
                        pred = model(data, example, thw, shot_num=shot_num)  ### predict the density maps


                
                mse = ((pred - density_map)**2).mean(-1)
                predict_counts = torch.sum(pred, dim=1).type(torch.FloatTensor).cuda() / args.scale_counts   #### scaling down by args.scale_counts
                predict_counts = predict_counts.round()
                predictions.extend(predict_counts.detach().cpu().numpy())
                gt_counts.extend(actual_counts.detach().cpu().numpy())
                mae = torch.div(torch.abs(predict_counts - actual_counts), actual_counts + 1e-1)
                predict_mae.extend(mae.cpu().numpy())
                predict_mse.extend(np.sqrt(mse.cpu().numpy()))
                loss.append(mse.cpu().numpy())
                
        predict_mae = np.array(predict_mae)
        predictions = np.array(predictions).round()
        gt_counts = np.array(gt_counts)
        predict_mse = np.array(predict_mse)
        diff = np.abs(predictions.round() - gt_counts)
        diff_z = np.abs(predictions.round() - gt_counts.round())
        print(f'Overall MAE: {predict_mae.mean()}')  ### calculating mae
        print(f'OBO: {(diff<=1).sum()/ len(diff)}')   ### calculating obo
        print(f'OBZ: {(diff_z==0).sum()/ len(diff)}') ### calculating obz
        print(f'RMSE: {np.sqrt((diff**2).mean())}')  ### calculating rmse
        return

    if args.use_wandb:
            wandb_run = wandb.init(
                        config=args,
                        resume="allow",
                        project=args.wandb,
                        entity=args.team,
                        id=f"{args.wandb_id}_{args.dataset}_{args.encodings}_{args.lr}_fullattention{args.full_attention}_{args.threshold}_{args.slurm_job_id}",
                    )
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    milestones = [i for i in range(0, args.epochs, 60)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.8)  ### reduce learning rate by 0.8 every 60 epochs
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    best_loss = np.inf

    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
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
                        elif phase == 'val':
                            val_step+=1
                        with torch.cuda.amp.autocast(enabled=True):
                            data = item[0].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                            example = item[1].cuda().type(torch.cuda.FloatTensor) # B x (THW) x C
                            density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts  ###scaling up args.scale_counts.This helps in magnifying the loss
                            actual_counts = item[3].cuda() # B x 1
                            thw = item[5]
                            shot_num = item[6][0] ## number of shots
                            b,n,c = data.shape
                            y = model(data, example, thw, shot_num=shot_num)
                            if phase == 'train':
                                mask = np.random.binomial(n=1, p=0.8, size=[1,density_map.shape[1]])  ### random masking of 20% density map
                            else:
                                mask = np.ones([1, density_map.shape[1]])
                            
                            masks = np.tile(mask, (density_map.shape[0], 1))
                            
                            
                            masks = torch.from_numpy(masks).cuda()
                            loss = ((y - density_map) ** 2)
                            loss = ((loss * masks) / density_map.shape[1]).sum() / density_map.shape[0]  ### mse

                            predict_count = torch.sum(y, dim=1).type(torch.cuda.FloatTensor) / args.scale_counts # sum density map
                            # loss_mse = torch.mean((predict_count - actual_counts)**2)
                            if phase == 'val':
                                ground_truth.append(actual_counts.detach().cpu().numpy())
                                predictions.append(predict_count.detach().cpu().numpy())
                            
                            loss2 = lossSL1(predict_count, actual_counts)  ###L1 loss between count and predicted count
                            loss3 = torch.sum(torch.div(torch.abs(predict_count - actual_counts), actual_counts + 1e-1)) / \
                            predict_count.flatten().shape[0]    #### reduce the mean absolute error (mae loss)
                            if phase=='train':

                                loss1 = (loss + 1.0 * loss3) / args.accum_iter  ### mse between density maps + mae loss (loss3)
                                loss1.backward()    ### call backward
                                if (i + 1) % args.accum_iter == 0: ### accumulate gradient
                                    optimizer.step() ##update parameters
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()
                            
                            epoch_loss = loss.item()
                            count += b
                            total_loss_all += loss.item() * b
                            total_loss1 += loss.item() * b
                            total_loss2 += loss2.item() * b
                            total_loss3 += loss3.item() * b
                            off_by_zero += (torch.abs(actual_counts.round() - predict_count.round()) ==0).sum().item()  ## off by zero
                            off_by_one += (torch.abs(actual_counts.round() - predict_count.round()) <=1 ).sum().item()   ## off by one
                            mse += ((actual_counts - predict_count.round())**2).sum().item()
                            mae += torch.sum(torch.div(torch.abs(predict_count.round() - actual_counts), (actual_counts) + 1e-1)).item()  ##mean absolute error
                            

                            
                            pbar.set_description(f"EPOCH: {epoch:02d} | PHASE: {phase} ")
                            pbar.set_postfix_str(f" LOSS: {total_loss_all/count:.2f} | MAE:{mae/count:.2f} | LOSS ITER: {loss.item():.2f} | OBZ: {off_by_zero/count:.2f} | OBO: {off_by_one/count:.2f} | RMSE: {np.sqrt(mse/count):.3f}")
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

                        ### Savind checkpoints
                        if total_loss_all/float(count) < best_loss:
                            best_loss = total_loss_all/float(count)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'best_1.pyth'))
                        torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'epoch_{}.pyth'.format(str(epoch).zfill(3))))
    
    
    if args.use_wandb:                                   
        wandb_run.finish()

if __name__=='__main__':
    main()