import time
from functools import partial
import math
import random
from slowfast.models.attention import MultiScaleBlock

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from slowfast.models import head_helper, operators, stem_helper 

from timm.models.vision_transformer import PatchEmbed, Block
from models_crossvit import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed
import logging



def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    # if verbose:
    #     logger.info(f"min width {min_width}")
    #     logger.info(f"width {width} divisor {divisor}")
    #     logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

class SupervisedMAE(nn.Module):
    def __init__(self, cfg=None, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, just_encode=False, 
                 use_precomputed=True):

        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        embed_dim = cfg.MVIT.EMBED_DIM
        self.just_encode = just_encode
        self.use_precomputed = use_precomputed
        # Prepare backbone
        pool_first = cfg.MVIT.POOL_FIRST
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.input_dims = [temporal_size, spatial_size, spatial_size]

        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # num_patches = self.patch_embed.num_patches

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        self.num_patches = math.prod(self.patch_dims)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        input_size = self.patch_dims
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]


        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        if not self.use_precomputed:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)]) 
        
        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = self.num_patches + 1
        else:
            pos_embed_dim = self.num_patches

        
        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )


 
        
        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                separate_qkv=cfg.MVIT.SEPARATE_QKV,
            )

            # if cfg.MODEL.ACT_CHECKPOINT:
            #     attention_block = checkpoint_wrapper(attention_block)
            
            if not self.use_precomputed:
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

            embed_dim = dim_out

            self.norm = norm_layer(embed_dim)


        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.shot_token = nn.Parameter(torch.zeros(512))

        # Exemplar encoder with CNN
        # self.decoder_proj1 = nn.Sequential(
        #     nn.Conv3d(768, 512, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((1,2,2)) #[3,64,64]->[64,32,32]
        # )
        # self.decoder_proj2 = nn.Sequential(
        #     nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((1, 2, 2)) #[64,32,32]->[128,16,16]
        # )
        # self.decoder_proj3 = nn.Sequential(
        #     nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d((1, 2, 2)) # [128,16,16]->[256,8,8]
        # )
        # self.decoder_proj4 = nn.Sequential(
        #     nn.Conv3d(256, decoder_embed_dim, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm3d(512),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool3d((1,1,1))
        #     # [256,8,8]->[512,1,1]
        # )


        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Density map regresssion module
        # self.decode_head0 = nn.Sequential(
        #     nn.Conv2d(decoder_embed_dim, 256, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 256),
        #     nn.ReLU(inplace=True)
        # )
        # self.decode_head1 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 256),
        #     nn.ReLU(inplace=True)
        # )
        # self.decode_head2 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 256),
        #     nn.ReLU(inplace=True)
        # )
        # self.decode_head3 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 1, kernel_size=1, stride=1)
        # )  

        # --------------------------------------------------------------------------

        self.decode_head0 = nn.Sequential(
            nn.Conv3d(decoder_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            # nn.MaxPool3d((1,2,2))
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2))
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(256, 1, kernel_size=1, stride=1)
        )  
        self.temporal_map = nn.Linear(self.patch_dims[0], cfg.DATA.NUM_FRAMES)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    @torch.no_grad()
    def sincos_pos_embed(self, max_len=1000, embed_dim=768, n=10000):
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(n) / embed_dim))
        k = torch.arange(0, max_len).unsqueeze(1)     
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0) 
        return pe

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # print(self.num_patches)
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        # print(pos_embed.shape)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.shot_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    def _mae_forward_encoder(self, x,):
        x, bcthw = self.patch_embed(x, keep_spatial=False)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        s = 1 if self.cls_embed_on else 0
        B, N, C = x.shape

        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # 0: no cls token

        # if self.cfg.MASK.PER_FRAME_MASKING:
        #     x = x.reshape([B * T, H * W, C])
        # x, mask, ids_restore, ids_keep = self._mae_random_masking(
        #     x, mask_ratio, None if self.cfg.MASK.MAE_RND_MASK else mask
        # )
        # if self.cfg.MASK.PER_FRAME_MASKING:
        #     x = x.view([B, -1, C])

        if self.cls_embed_on:
            # append cls token
            cls_token = self.cls_token  #
            if self.use_fixed_sincos_pos:
                cls_token = cls_token + self.pos_embed[:, :s, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)
        # apply Transformer blocks
        B, N, C = x.shape
        thw = [T, H, W]
        for _, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
        x = self.norm(x)

        return x, thw


    # def forward_decoder(self, x, y_, shot_num=3):
    #     # embed tokens
    #     x = self.decoder_embed(x)
    #     # add pos embed
    #     x = x + self.decoder_pos_embed

    #     # Exemplar encoder
    #     y_ = y_.transpose(0,1) # y_ [N,3,3,64,64]->[3,N,3,64,64]
    #     y1=[]
    #     C=0
    #     N=0
    #     cnt = 0
    #     for yi in y_:
    #         cnt+=1
    #         if cnt > shot_num:
    #             break
    #         yi = self.decoder_proj1(yi)
    #         yi = self.decoder_proj2(yi)
    #         yi = self.decoder_proj3(yi)
    #         yi = self.decoder_proj4(yi)
    #         N, C,_,_ = yi.shape
    #         y1.append(yi.squeeze(-1).squeeze(-1)) # yi [N,C,1,1]->[N,C]       
            
    #     if shot_num > 0:
    #         y = torch.cat(y1,dim=0).reshape(shot_num,N,C).to(x.device)
    #     else:
    #         y = self.shot_token.repeat(y_.shape[1],1).unsqueeze(0).to(x.device)
    #     y = y.transpose(0,1) # y [3,N,C]->[N,3,C]
        
    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x, y)
    #     x = self.decoder_norm(x)
        
    #     # Density map regression
    #     n, hw, c = x.shape
    #     h = w = int(math.sqrt(hw))
    #     x = x.transpose(1, 2).reshape(n, c, h, w)

    #     x = F.interpolate(
    #                     self.decode_head0(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
    #     x = F.interpolate(
    #                     self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
    #     x = F.interpolate(
    #                     self.decode_head2(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
    #     x = F.interpolate(
    #                     self.decode_head3(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
    #     x = x.squeeze(-3)

    #     return x

    def forward(self, vid, yi=None, boxes=None, shot_num=0):
        # if boxes.nelement() > 0:
        #     torchvision.utils.save_image(boxes[0], f"data/out/crops/box_{time.time()}_{random.randint(0, 99999):>5}.png")
        y1 = []
        if not self.use_precomputed:
            with torch.no_grad():
                latent, thw = self._mae_forward_encoder(vid)  ##temporal dimension preserved 1568 tokens
                latent = latent[:, 1:]
                if self.just_encode:
                    return latent, torch.tensor(thw).to(latent.device)
        else:
            latent = vid
            # print(_)
        x = self.decoder_embed(latent)
        # x = x + self.decoder_pos_embed
        x = x + self.sincos_pos_embed(x.shape[1], x.shape[2]).to(x.device)
        yi = self.decoder_embed(yi)
        # print(yi.shape)
        yi = yi.mean(1).squeeze(1)
        
        # print(yi.shape)

        # print(x.shape)
        # yi = self.decoder_proj1(yi)
        # yi = self.decoder_proj2(yi)
        # yi = self.decoder_proj3(yi)
        # yi = self.decoder_proj4(yi)  ### (B, 512, 1, 1, 1)
        # print(yi.shape)

        # N, C, _, _, _ = yi.shape
        # y1.append(yi.squeeze(-1).squeeze(-1).squeeze(-1)) 
        N, C = yi.shape
        y1.append(yi)
        y = torch.cat(y1,dim=0).reshape(1,N,C).to(x.device)
        y = y.transpose(0,1)
        for blk in self.decoder_blocks:
            x = blk(x, y)  ### feature interaction model
        x = self.decoder_norm(x)
        # print(x.shape)
        n, thw, c = x.shape
        
        # t = self.patch_dims[0]
        
        # h = w = int(math.sqrt(thw/t))
        h = w = 7
        t = int(thw / (h*w))
        x = x.transpose(1, 2).reshape(n, c, t, h, w)
        # print(x.shape)
        x = self.decode_head0(x)  
        x = F.interpolate(x, scale_factor=(4,1,1), mode='trilinear')
        x = self.decode_head1(x)
        x = F.upsample(x, scale_factor=(2,1,1), mode='trilinear')
        x = self.decode_head2(x)  ### (B, 1, 8, 1, 1)
        print(x.shape)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.temporal_map(x)

        # print(x.shape)
        # pred = self.forward_decoder(latent, boxes, shot_num)  # [N, 384, 384]
        return x.squeeze(1)


# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = SupervisedMAE(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = SupervisedMAE(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = SupervisedMAE(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def mae_vit_base_patch16_fim4(**kwargs):
#     model = SupervisedMAE(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def mae_vit_base_patch16_fim6(**kwargs):
#     model = SupervisedMAE(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  
# mae_vit_base4_patch16 = mae_vit_base_patch16_fim4 # decoder: 4 blocks
# mae_vit_base6_patch16 = mae_vit_base_patch16_fim6 # decoder: 6 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  
