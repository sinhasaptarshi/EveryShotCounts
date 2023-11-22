import argparse
from functools import partial
import math
import einops
from slowfast.models.attention import MultiScaleBlock

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils


from timm.models.vision_transformer import PatchEmbed, Block
from models_crossvit import CrossAttentionBlock

from memvit.models import build_model
from memvit.utils.parser import load_config

from util.pos_embed import get_2d_sincos_pos_embed

def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

class RepMem(nn.Module):
    def __init__(self, 
                 cfg=None,
                 enc_cfg='configs/AVA/MeMViT_16_K400.yaml',
                 exp_enc='x3d_xs',
                 decoder_embed_dim=512,
                 feat_ineract_depth=1,  
                 decoder_depth=4, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MeMViT encoder
        cfg_args = {'shard_id':0,
                    'num_shards':0,
                    'init_method':"tcp://localhost:9999",
                    'cfg_file':enc_cfg,
                    'opts':None}
        args = argparse.Namespace(**cfg_args)
        enc_cfg = load_config(args)
        
        self.encoder = build_model(enc_cfg)
        self.encoder.head = nn.Identity()
        enc_embed_dim = enc_cfg.MVIT.EMBED_DIM * int(math.prod([dim[1] for dim in enc_cfg.MVIT.DIM_MUL]))
        # --------------------------------------------------------------------------
        
        
        # --------------------------------------------------------------------------
        # exemplar encoder
        self.exp_encoder = torch.hub.load("facebookresearch/pytorchvideo", model=exp_enc, pretrained=True)
        self.exp_encoder.blocks[5] = nn.Identity()
        for _,param in self.exp_encoder.named_parameters():
            enc_exp_embed_dim = param.shape[0]
        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # Feature interaction module specifics
        self.projx = nn.Linear(enc_embed_dim, decoder_embed_dim, bias=True)
        self.projex = nn.Linear(enc_exp_embed_dim, decoder_embed_dim, bias=True)
        
        self.zero_shot_token = nn.Parameter(torch.zeros(enc_exp_embed_dim))

        self.feat_interact_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(feat_ineract_depth)])

        self.feat_interact_norm = norm_layer(decoder_embed_dim)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)

        #self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # --------------------------------------------------------------------------
        
        self.temporal_map = nn.Linear(decoder_embed_dim, 1, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        torch.nn.init.normal_(self.zero_shot_token, std=.02)
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
            


    def forward(self, x, ex):
        
        # Video encoder
        with torch.no_grad():
            x = self.encoder(x)  ##temporal dimension preserved 1568 tokens
            x = einops.reduce(x, 'b h w t c -> b t c', 'mean')
        x = x.detach() # detach tensor from computational graph
        torch.cuda.empty_cache() # empty memory cache (for better memory utilisation)
        
        # Exemplar encoder
        if ex is not None:
            with torch.no_grad():
                ex = self.exp_encoder(ex)
                ex = einops.reduce(ex, 'b c t h w -> b t c', 'mean')
            ex = ex.detach() 
            torch.cuda.empty_cache() 
        else:
            ex = self.zero_shot_token
        
        # projections to joint space
        x, ex = (self.projx(x),self.projex(ex))
        
        for blk in self.feat_interact_blocks: # feature interaction module
            x = blk(x, ex) 
        x = self.feat_interact_norm(x)
        
        for blk in self.decoder_blocks: # decoder blocks
            x = blk(x) 
            x = F.interpolate(einops.rearrange(x, 'b t c -> b c t'), size=x.shape[-1]*2, mode='linear', align_corners=False)
            x = einops.rearrange(x, 'b c t -> b t c')
        x = self.decoder_norm(x)
        
        return self.temporal_map(x)



if __name__ == '__main__':
    model = RepMem().cuda()
    
    x = torch.rand(3,3,64,224,224).cuda()
    ex = torch.rand(3,3,4,224,224).cuda()
    
    with torch.autocast(device_type="cuda"):
        x = model(x,ex)
    
    print(x.shape)
    
    
