import time
from functools import partial
import math
import random
from slowfast.models.attention import MultiScaleBlock
import math 

import numpy as np
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from slowfast.models import head_helper, operators, stem_helper 
from slowfast.models import stem_helper

from timm.models.vision_transformer import PatchEmbed, Block
from model_crossvit_window_attention import CrossAttentionBlock, WindowedSelfAttention, compute_mask, get_window_size

from util.pos_embed import get_2d_sincos_pos_embed
import logging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from models_crossvit import Attention

class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


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
                 use_precomputed=True, token_pool_ratio=1.0, iterative_shots=False, encodings='swin', no_exemplars=False, window_size=(3,3,3)):

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
        self.iterative_shots = iterative_shots

        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

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


        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

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
                if self.use_fixed_sincos_pos:
                    self.pos_embed = self.get_sinusoid_encoding_table(pos_embed_dim, embed_dim)
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
        embed_dim = 2048 if encodings=='resnext' else 768
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        # print('Embed Dim', embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        spatial_tokens = math.ceil(token_pool_ratio * 14) if encodings == 'mae' else math.ceil(token_pool_ratio * 7)
        self.decoder_spatial_pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(decoder_embed_dim, spatial_tokens).astype(np.float32)), requires_grad=False)
        self.example_spatial_pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(decoder_embed_dim, 14).astype(np.float32)), requires_grad=False)
        trunc_normal_(self.decoder_spatial_pos_embed, std=.02)
        self.shot_token = nn.Parameter(torch.zeros(1568, decoder_embed_dim))

 
        
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=0, iterative_shots=self.iterative_shots, no_exemplars=no_exemplars)
            for i in range(decoder_depth)])      ### cross-attention blocks

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in self.window_size)
        self.downsample = None 
        self.decode_heads = nn.ModuleList([
            WindowedSelfAttention(
                dim=decoder_embed_dim,
                num_heads=8,
                window_size=self.window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=norm_layer,
                use_checkpoint=False,
            )
            for i in range(3)])       ##### windowed self-attention blocks

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.map = nn.Linear(decoder_embed_dim, 1, bias=True)

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
    
    @torch.no_grad()
    def get_sinusoid_encoding_table(self, n_position, d_hid): 
        ''' Sinusoid position encoding table ''' 
        # TODO: make it with torch instead of numpy 
        def get_position_angle_vec(position): 
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

        return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

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

    def forward(self, vid, yi=None, thw=None, boxes=None, shot_num=1):
        
        y1 = []

        ### extract latent representations
        if not self.use_precomputed:
            with torch.no_grad():
                latent, thw = self._mae_forward_encoder(vid)  ##temporal dimension preserved 1568 tokens
                latent = latent[:, 1:]
                if self.just_encode:
                    return latent, torch.tensor(thw).to(latent.device)
        else:
            latent = vid
        t,h,w = thw[0]
        x = self.decoder_embed(latent)

        t = x.shape[1] // (h*w)

        ### temporal position embedding
        decoder_pos_embed_temporal = self.sincos_pos_embed(t, x.shape[2]).to(x.device)
        ### spatial position embedding  + temporal position embedding
        pos_embed = self.decoder_spatial_pos_embed.repeat(
                    1, t, 1
                ) + torch.repeat_interleave(
                    decoder_pos_embed_temporal,
                    h*w,
                    dim=1,
                )
        
        x = x + pos_embed
 
        if shot_num > 0:
            yi = self.decoder_embed(yi)
            N,_,C = yi.shape
            
            y = yi 

        else:  ### use 0-shot token if shot_num>0
            
            y = self.shot_token.unsqueeze(0).repeat(x.shape[0],1, 1).to(x.device)  ## zero-shot token repeat

        for blk in self.decoder_blocks:  ### cross-attention blocks
            x = blk(x, y, shot_num=max(shot_num,1))  
        x = self.decoder_norm(x)

        n, thw, c = x.shape

        t = int(thw / (h*w))
        x = x.reshape(n,t,h,w,c)
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        x = x.view(B, D, H, W, -1)

        for i, decode_head in enumerate(self.decode_heads):  ### self-attention blocks with windowed self-attention
            x = decode_head(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.map(x)    ### linear mapping
        x = einops.rearrange(x, 'b d h w c -> b c d h w')

        x = x.squeeze(1).reshape(-1, t, h, w)
        x = self.pool(x)  ### spatial average pooling
        x = x.squeeze(-1).squeeze(-1)

        return x
