import torch
import torch.nn as nn
torch.cuda.empty_cache()
import numpy as np
import os, sys
# from Rep_count_loader import Rep_count
from UCF_Rep import UCFRep as UCFRepraw
from Repcount_multishot_loader import Rep_count
# from Repcount_balanced_loader import Rep_count
from Countix_multishot_loader import Countix
from UCFRep_multishot_loader import UCFRep
from egoloops_multishot_loader import EgoLoops
from tqdm import tqdm
from video_mae_cross import SupervisedMAE as SupervisedMAE
from video_mae_cross_finetune import SupervisedMAE as SupervisedMAE_finetune
from video_mae_cross_full_attention import SupervisedMAE as SupervisedMAE_fullattention
from video_memae import RepMem
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
                        help='cut off to decide if select exemplar from different video')
    
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
    parser.add_argument("--wandb_id", default='counTR_exp', type=str)
    
    parser.add_argument("--token_pool_ratio", default=0.6, type=float)

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
    
    cfg = load_config(args, path_to_config='pretrain_config.yaml')
    if not args.only_test:
        if args.slurm_job_id is not None:
            args.data_path = os.path.join(args.data_path, args.slurm_job_id)
        args.tokens_dir = os.path.join(args.data_path, args.tokens_dir)
        args.exemplar_dir = os.path.join(args.data_path, args.exemplar_dir)
    
    
    model = SupervisedMAE_finetune(cfg=cfg,use_precomputed=False, token_pool_ratio=1.0, iterative_shots=args.iterative_shots, encodings='resnext', finetune_encoder=True).cuda()

    print(model)
    model = model.cuda()
    # model = nn.DataParallel(model)
    # state_dict = torch.load('pretrained_models/resnext-101-64f-kinetics-ucf101_split1.pth')['state_dict']
    state_dict = torch.load('resnext101.pth')['state_dict']
    # print(state_dict.keys())
    for name in model.state_dict().keys():
        if name.startswith('blocks.'):
            # print(name.replace('blocks.', 'module.'))
            n_name = name.replace('blocks.', '')
            if n_name not in state_dict.keys():
                # print(n_name)
                try:
                    param = state_dict[n_name]
                    model.state_dict()[name].copy_(param)
                except:
                    print(n_name)
                    pass
            # print(state_dict[name.replace('blocks.', 'module.')])

    # model.blocks.load_state_dict(state_dict)
    train_dataset = UCFRepraw(split='train', sampling_interval=1, encode_only=True)
    val_dataset = UCFRepraw(split='val', sampling_interval=1, encode_only=True)
    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=1, num_workers=4, pin_memory=True),
                   'val': torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=4, pin_memory=True)}
    
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
        videos = []
        loss = []
        # model.shot_token.data.fill_(0)
        # model.decoder_spatial_pos_embed.data = torch.from_numpy(get_2d_sincos_pos_embed(512, 8).astype(np.float32)).cuda()
        # model.decoder_blocks[0].attn.wq.reset_parameters()
        # model.decoder_blocks[0].attn.wk.reset_parameters()
        # model.decoder_blocks[1].attn.wk.reset_parameters()
        # model.decoder_blocks[0].attn.wv.reset_parameters()
        # model.decoder_blocks[1].attn.wv.reset_parameters()
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
                # print(example.shape)
                density_map = item[2].cuda().type(torch.cuda.FloatTensor).half() * args.scale_counts
                actual_counts = item[3].cuda() # B x 1
                video_name = item[4]
                
                videos.append(video_name[0])

                
                shot_num = item[6][0]
                # print(shot_num)
                b, n, c = data.shape
                
                # print(data.shape)
                # data_ = data.reshape(b, -1, 36, c)
                # pred = torch.zeros(b, 0).cuda()

                thw = item[5]
                with torch.no_grad():
                    if args.get_overlapping_segments:
                        data = data.cuda().type(torch.cuda.FloatTensor)
                        # print(data.shape)
                        data2 = data2.cuda().type(torch.cuda.FloatTensor)
                        # print(data2.shape)
                        pred1 = model(data, example, thw, shot_num=shot_num)
                        pred2 = model(data2, example, thw, shot_num=shot_num)
                        if pred1.shape != pred2.shape:
                            pred2 = torch.cat([torch.zeros(1, 4).cuda(), pred2], 1)
                        else:
                            print('equal')
                        pred = (pred1 + pred2) / 2
                    else:
                        # print(shot_num)
                        pred = model(data, example, thw, shot_num=shot_num)

                    # for i in range(pred.shape[0]):
                    # num_segments = pred.shape[1] // 8
                    # total_dur = 8 + 2 * (num_segments - 1)
                    # freq = torch.zeros(total_dur).cuda()
                    # pred_density_map = torch.zeros(total_dur).cuda()
                    # for j in range((total_dur-8)//2):
                    #     freq[2*j: 2*j + 8] += 1
                    #     pred_density_map[2*j : 2*j + 8] += pred[0][8*j:8*j + 8]
                    # pred_density_map = pred_density_map / freq
                    # pred = pred_density_map.unsqueeze(0)


                    # # pred[i] = density_map
                    # del pred_density_map
                    # del freq

                    # for i in range(0, data_.shape[1], 48):
                    #     data = data_[:, i: min(i+48, data.shape[1])]
                    #     k = data.shape[1]
                    #     if data.shape[1] < 48:
                    #         data = torch.cat([data, torch.zeros(b, 48-data.shape[1], 36, c).cuda()], 1)
                    #     data = data.reshape(b, -1, c)
                    #     y = model(data, example, thw)
                    #     y = y[:, :k]
                    #     pred = torch.cat([pred, y], 1)
                        # print(pred.shape)
                    # y = torch.nn.functional.relu(y)
                # print(video_name[0])
                mse = ((pred - density_map)**2).mean(-1)
                np.savez('countix_density_maps_peaks1.0/'+video_name[0]+'_preds_1shot.npz', pred[0].cpu().numpy())
                np.savez('countix_density_maps_peaks1.0/'+video_name[0]+'_countix_gt.npz', density_map[0].cpu().numpy())
                predict_counts = torch.sum(pred, dim=1).type(torch.FloatTensor).cuda() / args.scale_counts
                predict_counts = predict_counts.round()
                predictions.extend(predict_counts.detach().cpu().numpy())
                gt_counts.extend(actual_counts.detach().cpu().numpy())
                mae = torch.div(torch.abs(predict_counts - actual_counts), actual_counts + 1e-1)
                predict_mae.extend(mae.cpu().numpy())
                predict_mse.extend(np.sqrt(mse.cpu().numpy()))
                loss.append(mse.cpu().numpy())
                clips.append(data.shape[1]//(8*9*9))
                # print(predict_mae)
        print(videos)
        predict_mae = np.array(predict_mae)
        predictions = np.array(predictions)
        gt_counts = np.array(gt_counts)
        predict_mse = np.array(predict_mse)
        videos = pd.DataFrame(videos, columns=['name'])
        loss = pd.DataFrame(loss, columns=['1shot_loss'])
        # df = pd.read_csv('datasets/ucf_aug_val_new.csv')
        # df['predictions'] = predictions
        # df['gt_counts'] = gt_counts
        # df.to_csv('datasets/ucf_aug_val_new.csv')
        df = pd.concat([videos, loss], axis=1)
        df.to_csv('countix_results_2.csv')
        clips = np.array(clips)
        min_ = clips.min()
        max_ = clips.max()

        print(gt_counts)
        print(predictions.round())
        diff = np.abs(predictions.round() - gt_counts)
        # diff_rmse = np.abs(predictions - gt_counts)
        diff_z = np.abs(predictions.round() - gt_counts.round())
        print(f'Overall MAE: {predict_mae.mean()}')
        print(f'OBO: {(diff<=1).sum()/ len(diff)}')
        print(f'OBZ: {(diff_z==0).sum()/ len(diff)}')
        print(f'RMSE: {np.sqrt((diff**2).mean())}')
        for counts in range(1,round(gt_counts.max())+1):
            print(f"MAE for count {counts} is {predict_mae[gt_counts <= counts].mean()}")
        for counts in range(1,round(gt_counts.max())+1):
            print(f"RMSE for count {counts} is {np.sqrt((diff**2)[gt_counts <= counts].mean())}")
        for duration in np.linspace(min_, max_, 10)[1:]:
            print(f"MAE for duration less that {duration} is {predict_mae[clips <= duration].mean()}")
        for duration in np.linspace(min_, max_, 10)[1:]:
            print(f"RMSE for duration less that {duration} is {np.sqrt((diff**2)[clips <= duration].mean())}")

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
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, betas=(0.9, 0.95))
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    milestones = [i for i in range(0, args.epochs, 60)]
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.8)
    lossMSE = nn.MSELoss().cuda()
    lossSL1 = nn.SmoothL1Loss().cuda()
    best_loss = np.inf

    os.makedirs(args.save_path, exist_ok=True)
    wandb_run.watch(model, log='gradients')
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
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
                            # if args.get_overlapping_segments:
                            #     data, data2 = item[0][0], item[0][1]
                            # else:
                            data = item[0]#.cuda()#.type(torch.cuda.FloatTensor) # B x (THW) x C
                            density_map = item[1].cuda()
                            actual_counts = item[2].cuda()
                            segment_start = item[-2]
                            segment_end = item[-1]
                            # print(data.shape)
                            # print(data.shape)
                            b, c, t, h, w = data.shape
                            data = torch.cat([data, torch.zeros(b, c, 64, h, w)], 2)
                            clips = []
                            for j in range(0, t, 64):
                                clips.append(data[:,:,j:j+64,:,:].squeeze(0))

                            batched_clips = torch.stack(clips)#.cuda()
                            # print(batched_clips.shape)
                            starts = item[3][0]
                            ends = item[4][0]
                            
                            if phase == 'train':
                                shot_num = np.random.randint(0,3)
                                # shot_num = 2
                                if shot_num > 0:
                                    select_exemplar = np.random.choice(range(len(starts)), shot_num, replace=False)
                                    for shot in range(shot_num):
                                        exemplar_start = starts[shot]
                                        exemplar_end = ends[shot]
                                        exemplar_frames = np.linspace(exemplar_start, exemplar_end-1, 65)[:64].astype(int)
                                        # print(exemplar_frames)
                                        exemplar = data[:, :, exemplar_frames]
                                        batched_clips = torch.cat([batched_clips, exemplar], 0)
                            

                            data = batched_clips.cuda()

                            # shot_num = item[6][0]
                            
                            # if args.get_overlapping_segments:
                            #     # padding = data.shape[1] - data2.shape[1]
                            #     # data2 = torch.cat([data2, torch.zeros(data.shape[0], padding, data2.shape[-1])], 1)
                            #     # data = torch.cat([data, data2])
                            #     data = data.cuda().type(torch.cuda.FloatTensor)
                            #     # data = data.repeat(2,1,1)
                            #     # example = example.repeat(2,1,1)
                            #     # print(example.shape)
                            #     # data2 = data2.cuda().type(torch.cuda.FloatTensor)
                            #     # pred2 = model(data2, example, thw)
                            #     # del data2
                            #     # data = data.cuda().type(torch.cuda.FloatTensor)
                            #     b,n,c = data.shape
                            #     # data2 = data2.cuda().type(torch.cuda.FloatTensor)
                            #     # pred1 = model(data, example, thw)
                            #     # pred = model(data, example, thw)
                            #     y = model(data, example, thw)
                            #     # del data
                            #     # pred1 = pred[0]
                            #     # if padding != 0:
                            #     #     pred2 = pred[1][:-(padding//(thw[0][-1] * thw[0][-2]))]
                            #     # else:
                            #     #     pred2 = pred[1]
                            #     # print(pred1.shape)
                            #     # print(pred2.shape)
                            #     # if pred1.shape != pred2.shape:
                            #     #     pred2 = torch.cat([torch.zeros(abs(len(pred1) - len(pred2))).cuda(), pred2])
                            #     # if pred1.shape[1] - pred2.shape[1] == 2:
                            #     #     print(pred1.shape)
                            #     #     print(pred2.shape)
                            #     # y = (pred1[0] + pred1[1]) / 2
                            #     # y = y.unsqueeze(0)
                            #     # y = (pred1 + pred2) / 2
                            #     # y = y.unsqueeze(0)
                            # else:
                            # b,n,c = data.shape
                            y = model(data, shot_num=shot_num, start=segment_start[0], end=segment_end[0])
                            # print(y.shape)
                            # print(y.shape)
                            # print(density_map.shape)
                            # y = model(data, example, thw, shot_num=shot_num)
                            # y = torch.nn.functional.relu(y)
                            if phase == 'train':
                                mask = np.random.binomial(n=1, p=0.8, size=[1,density_map.shape[1]])
                            else:
                                mask = np.ones([1, density_map.shape[1]])
                            
                            masks = np.tile(mask, (density_map.shape[0], 1))
                            
                            
                            masks = torch.from_numpy(masks).cuda()
                            loss = ((y - density_map) ** 2) #+ (0.2 * (y / args.scale_counts).abs()) 
                            counts = density_map.sum(1, keepdims=True) / args.scale_counts
                            counts[counts == 0] = 1
                            # loss = (loss * masks / counts / density_map.shape[1]).sum() / density_map.shape[0]   ## normalized by counts
                            loss = ((loss * masks) / density_map.shape[1]).sum() / density_map.shape[0]  ### normalized by duration
                            # loss = (loss * masks).sum() / density_map.shape[0]  ###non-normalized
                            
                            #print(item[4],data.shape,y.shape,density_map.shape)
                            predict_count = torch.sum(y, dim=1).type(torch.cuda.FloatTensor) / args.scale_counts # sum density map
                            # loss_mae = torch.mean(torch.abs(predict_count - actual_counts)/actual_counts)
                            loss_mse = torch.mean((predict_count - actual_counts)**2)
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
                            # loss1 = loss
                            
                            # if args.use_mae:
                            #     loss = loss1 + loss3
                            #     loss2 = 0 # Set to 0 for clearer logging
                            # else:
                            #     loss = loss1 + loss2
                            #     loss3 = 0 # Set to 0 for clearer logging
                            # loss = loss1
                            if phase=='train':
                                # loss1 = loss / args.accum_iter
                                loss1 = (loss + loss3) / args.accum_iter
                                # loss1 = loss
                                # scaler(loss, optimizer, parameters=model.parameters(), update_grad=(i + 1) % args.accum_iter == 0)

                                # scaler.scale(loss1).backward()
                                loss1.backward()
                                if (i + 1) % args.accum_iter == 0: ### accumulate gradient
                                    # scaler.step(optimizer)
                                    # scaler.update()
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    torch.cuda.empty_cache()
                            
                            epoch_loss = loss.item()
                            count += b
                            total_loss_all += loss.item() * b
                            total_loss1 += loss.item() * b
                            total_loss2 += loss2.item() * b
                            total_loss3 += loss3.item() * b
                            # actual_counts = actual_counts / 60
                            # predict_count = predict_count / 60
                            off_by_zero += (torch.abs(actual_counts.round() - predict_count.round()) ==0).sum().item()  ## off by zero
                            off_by_one += (torch.abs(actual_counts.round() - predict_count.round()) <=1 ).sum().item()   ## off by one
                            mse += ((actual_counts - predict_count.round())**2).sum().item()
                            mae += torch.sum(torch.div(torch.abs(predict_count.round() - actual_counts), (actual_counts) + 1e-1)).item()
                            
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
                        if total_loss_all/float(count) < best_loss:
                            best_loss = total_loss_all/float(count)
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'best_1.pyth'))
                        torch.save({
                                # 'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(args.save_path, 'epoch_{}.pyth'.format(str(epoch).zfill(3))))
    
    
    if args.use_wandb:                                   
        wandb_run.finish()

if __name__=='__main__':
    main()