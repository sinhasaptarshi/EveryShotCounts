import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# import slowfast.utils.logging as logging
from common import TwoStreamFusion
from slowfast.models.reversible_mvit import ReversibleMViT
from slowfast.models.attention import MultiScaleBlock
import slowfast.utils.misc as misc
from slowfast.models import head_helper
from slowfast.models.attention import attention_pool
from losses import MultipleMSELoss
from slowfast.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    # validate_checkpoint_wrapper_import,
)


from slowfast.models import head_helper, operators, stem_helper  # noqa
# from .build import MODEL_REGISTRY

# logger = logging.get_logger(__name__)


class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
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
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

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

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

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

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(cfg, self)

            embed_dim = round_width(
                embed_dim, dim_mul.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                cfg.MVIT.REV.RESPATH_FUSE, dim=2 * embed_dim
            )

            if "concat" in self.cfg.MVIT.REV.RESPATH_FUSE:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

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

                if cfg.MODEL.ACT_CHECKPOINT:
                    attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                2 * embed_dim
                if ("concat" in cfg.MVIT.REV.RESPATH_FUSE and self.enable_rev)
                else embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                cfg=cfg,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

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

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def forward(self, x, bboxes=None, return_attn=False):
        x = x[0]
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
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

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, thw = blk(x, thw)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.cls_embed_on:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                x = self.head([x], bboxes)

            else:
                if self.use_mean_pooling:
                    if self.cls_embed_on:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.cls_embed_on:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)
                x = self.head(x)

        return x



class MaskMViT(MViT):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pretrain_depth = cfg.MASK.PRETRAIN_DEPTH
        if self.pretrain_depth[-1] + 1 < cfg.MVIT.DEPTH:
            del self.blocks[self.pretrain_depth[-1] + 1 :]
        del self.norm
        del self.head
        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

        self.head_type = cfg.MASK.HEAD_TYPE.split("_")
        feat_sz = [self.feat_size[depth] for depth in self.pretrain_depth]
        self.multimse_loss = MultipleMSELoss()
        if self.head_type[0] == "separate":
            if not cfg.MASK.PRED_HOG:
                pred_t_sz = (
                    1
                    if self.cfg.MASK.TIME_STRIDE_LOSS
                    else self.patch_stride[0]
                )
                num_classes = [
                    pred_t_sz * (self.feat_stride[depth][-1] ** 2) * 3
                    for depth in self.pretrain_depth
                ]
                self.pred_head = head_helper.MSSeparateHead(
                    self.blocks, cfg, num_classes, feat_sz
                )
            else:
                self.hogs = nn.ModuleList()
                self.nbins = 9
                self.cell_sz = 8
                self.hogs.append(
                    operators.HOGLayerC(
                        nbins=self.nbins,
                        pool=self.cell_sz,
                    )
                )
                self.ncells = [
                    (self.feat_stride[depth][-1] // self.cell_sz) ** 2
                    for depth in self.pretrain_depth
                ]
                pred_hog_classes = [self.nbins * ncell for ncell in self.ncells]
                pred_hog_classes = [
                    pred_hog_class * 3  # 3 color channels
                    for pred_hog_class in pred_hog_classes
                ]
                self.pred_head = head_helper.MSSeparateHead(
                    self.blocks, cfg, pred_hog_classes, feat_sz
                )
                self.hog_loss = "mse"
        else:
            raise NotImplementedError

        embed_dim = cfg.MVIT.EMBED_DIM
        decoder_embed_dim = cfg.MASK.DECODER_EMBED_DIM
        self.sep_pos_embed_decoder = cfg.MASK.DECODER_SEP_POS_EMBED
        self.counter = 0
        if cfg.MASK.MAE_ON:
            # ----------------------------------------------------------------
            # MAE decoder specifics
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            dim_in = self.blocks[-1].dim_out
            self.norm = norm_layer(dim_in)
            self.decoder_embed = nn.Linear(dim_in, decoder_embed_dim, bias=True)
            num_patches = math.prod(self.patch_dims)
            if self.use_abs_pos:
                if self.sep_pos_embed_decoder:
                    self.dec_pos_embed_spatial = nn.Parameter(
                        torch.zeros(
                            1,
                            self.patch_dims[1] * self.patch_dims[2],
                            decoder_embed_dim,
                        )
                    )
                    self.dec_pos_embed_temporal = nn.Parameter(
                        torch.zeros(1, self.patch_dims[0], decoder_embed_dim)
                    )
                    if self.cls_embed_on:
                        self.dec_pos_embed_class = nn.Parameter(
                            torch.zeros(1, 1, decoder_embed_dim)
                        )
                else:
                    self.decoder_pos_embed = nn.Parameter(
                        torch.zeros(
                            1,
                            num_patches + 1
                            if self.cls_embed_on
                            else num_patches,
                            decoder_embed_dim,
                        )
                    )
        self.mask_token = nn.Parameter(
            torch.zeros(
                1, 1, decoder_embed_dim if cfg.MASK.MAE_ON else embed_dim
            )
        )
        trunc_normal_(self.mask_token, std=0.02)
        if self.use_abs_pos and cfg.MASK.MAE_ON:
            if self.sep_pos_embed_decoder:
                trunc_normal_(self.dec_pos_embed_spatial, std=0.02)
                trunc_normal_(self.dec_pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.dec_pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.decoder_pos_embed, std=0.02)

        if cfg.MASK.SCALE_INIT_BY_DEPTH:
            self.fix_init_weight()

        self.pred_pixel_wt = 0.0 if cfg.MASK.PRED_HOG else 1.0
        self.pred_hog_wt = 1.0 if cfg.MASK.PRED_HOG else 0.0

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed_decoder:
                    names.extend(
                        [
                            "dec_pos_embed_spatial",
                            "dec_pos_embed_temporal",
                            "dec_pos_embed_class",
                        ]
                    )
                else:
                    names.extend(["pos_embed_decoder"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        for trans in self.pred_head.transforms:
            for layer_id, layer in enumerate(trans):
                if hasattr(layer, "attn"):
                    rescale(
                        layer.attn.proj.weight.data,
                        layer_id + 1 + len(self.blocks),
                    )  # or + len(self.blocks)
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _get_multiscale_mask(self, mask):
        if self.use_2d_patch:
            mask = mask.unsqueeze(0)
        output_masks = []
        for depth in self.pretrain_depth:
            size = self.feat_size[depth][-1]
            output_mask = F.interpolate(mask, size=size)
            if self.use_2d_patch:
                output_mask = output_mask[0]
            output_mask = output_mask.flatten(1).to(torch.bool)
            output_masks.append(output_mask)
        return output_masks

    def _patchify(self, imgs, p=16, time_stride_loss=True):
        N, _, T, H, W = imgs.shape
        u = 1 if time_stride_loss else self.patch_stride[0]
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u
        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def _unpatchify(self, x):
        N, T, H, W, p, u, t, h, w = self.patch_info
        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs

    def _get_pixel_label_2d(self, input_img, output_masks, norm=True):
        input_img = input_img.permute(0, 2, 3, 1)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            size = self.feat_stride[depth][-1]
            label = input_img.unfold(1, size, size).unfold(2, size, size)
            label = label.flatten(1, 2).flatten(2)
            label = label[output_mask]
            if norm:
                mean = label.mean(dim=-1, keepdim=True)
                var = label.var(dim=-1, keepdim=True)
                label = (label - mean) / (var + 1.0e-6) ** 0.5
            labels.append(label)
        return labels

    def _get_pixel_label_3d(
        self, input_frames, output_masks, time_stride_loss=True, norm=True
    ):
        if time_stride_loss:
            input_frames = input_frames[
                :, :, :: self.cfg.MVIT.PATCH_STRIDE[0], :, :
            ]
        imgs = input_frames
        input_frames = input_frames.permute(0, 2, 3, 4, 1)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            size = self.feat_stride[depth][-1]
            label = self._patchify(
                imgs, p=size, time_stride_loss=time_stride_loss
            )
            label = label[output_mask]

            if norm:  # self.norm_pix_loss:
                mean = label.mean(dim=-1, keepdim=True)
                var = label.var(dim=-1, keepdim=True)
                label = (label - mean) / (var + 1.0e-6) ** 0.5
            labels.append(
                (label, self.pred_pixel_wt / len(self.pretrain_depth))
            )
        return labels

    def _get_hog_label_2d(self, input_frames, output_masks):
        # input_frames, B C H W
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            feat_size = self.feat_size[depth][-1]
            hog_list = []
            for hog in self.hogs:
                tmp_hog = hog(input_frames).flatten(1, 2)  # return B C H W
                unfold_size = tmp_hog.shape[-1] // feat_size
                tmp_hog = (
                    tmp_hog.permute(0, 2, 3, 1)
                    .unfold(1, unfold_size, unfold_size)
                    .unfold(2, unfold_size, unfold_size)
                    .flatten(1, 2)
                    .flatten(2)
                )
                tmp_hog = tmp_hog[output_mask]
                hog_list.append(tmp_hog)
            all_tlabel = torch.cat(hog_list, -1)
            labels.append((all_tlabel, self.pred_hog_wt, self.hog_loss))
        return labels

    def _get_hog_label_3d(self, input_frames, output_masks):
        input_frames = input_frames[
            :, :, :: self.cfg.MVIT.PATCH_STRIDE[0], :, :
        ]  # B C T H W
        input_frames = input_frames.transpose(1, 2)  # B T C H W
        B, T = input_frames.shape[:2]
        input_frames = input_frames.flatten(0, 1)  # BT C H W
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            feat_size = self.feat_size[depth][-1]
            hog_list = []
            for hog in self.hogs:
                tmp_hog = hog(input_frames).flatten(1, 2)  # BT C H W
                unfold_size = tmp_hog.shape[-1] // feat_size
                tmp_hog = (
                    tmp_hog.permute(0, 2, 3, 1)
                    .unfold(1, unfold_size, unfold_size)
                    .unfold(2, unfold_size, unfold_size)
                )  # BT h w C wh ww
                tmp_hog = tmp_hog.flatten(3).view(
                    B, T, feat_size, feat_size, -1
                )  # B T h w C (3 nbins h w)
                tmp_hog = tmp_hog.flatten(1, 3)  # B N C
                tmp_hog = tmp_hog[output_mask]
                hog_list.append(tmp_hog)
            all_tlabel = torch.cat(hog_list, -1)
            labels.append((all_tlabel, self.pred_hog_wt, self.hog_loss))
        return labels

    def _mae_random_masking(self, x, mask_ratio, mask_in=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if mask_in is None:
            if self.cfg.AUG.MASK_TUBE:
                noise = (
                    torch.rand(N, 1, self.H * self.W, device=x.device)
                    .repeat([1, self.T, 1])
                    .reshape(N, L)
                )  # noise in [0, 1]
            else:
                noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        else:
            noise = mask_in.flatten(1)
            mask_ratio = sum(noise.flatten()) / noise.numel()  # alrdy masked
        len_keep = int(L * (1 - mask_ratio))
        assert len_keep > 1
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def _mae_forward_encoder(self, x, mask_ratio, mask=None):
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

        if self.cfg.MASK.PER_FRAME_MASKING:
            x = x.reshape([B * T, H * W, C])
        x, mask, ids_restore, ids_keep = self._mae_random_masking(
            x, mask_ratio, None if self.cfg.MASK.MAE_RND_MASK else mask
        )
        if self.cfg.MASK.PER_FRAME_MASKING:
            x = x.view([B, -1, C])

        if self.cls_embed_on:
            # append cls token
            cls_token = self.cls_token  #
            if self.use_fixed_sincos_pos:
                cls_token = cls_token + self.pos_embed[:, :s, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos and not self.use_fixed_sincos_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                pos_embed = pos_embed.expand(x.shape[0], -1, -1)
                pos_embed = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(
                        1, 1, pos_embed.shape[2]
                    ),
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat(
                        [
                            self.pos_embed_class.expand(
                                pos_embed.shape[0], -1, -1
                            ),
                            pos_embed,
                        ],
                        1,
                    )
                x += pos_embed
            else:
                pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
                pos_embed_sampled = torch.gather(
                    pos_embed[:, s:, :],
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(
                        1, 1, self.pos_embed.shape[2]
                    ),
                )
                if self.cls_embed_on:
                    pos_embed_sampled = torch.cat(
                        [pos_embed[:, :s, :], pos_embed_sampled], 1
                    )
                x += pos_embed_sampled

        # apply Transformer blocks
        B, N, C = x.shape
        thw = [T, H, W]
        for _, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
        x = self.norm(x)

        return x, mask, ids_restore, thw

    def _mae_forward_decoder(self, x, ids_restore, mask, thw):
        # embed tokens
        x = self.decoder_embed(x)
        T, H, W = self.T, self.H, self.W
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            B, T * H * W + s - x.shape[1], 1
        )  # + s: no cls token
        x_ = torch.cat([x[:, s:, :], mask_tokens], dim=1)  # no cls token
        if self.cfg.MASK.PER_FRAME_MASKING:
            x_ = x_.view([B * T, H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        if self.cfg.MASK.PER_FRAME_MASKING:
            x_ = x_.view([B, T * H * W, C])
        x = torch.cat([x[:, :s, :], x_], dim=1)  # append cls token

        if self.sep_pos_embed_decoder:
            pos_embed = self.dec_pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.dec_pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            if self.cls_embed_on:
                pos_embed = torch.cat(
                    [
                        self.dec_pos_embed_class.expand(
                            pos_embed.shape[0], -1, -1
                        ),
                        pos_embed,
                    ],
                    1,
                )
            x += pos_embed
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        pixel_outputs = self.pred_head(
            [x],
            [mask.to(torch.bool)],
            return_all=self.cfg.VIS_MASK.ENABLE,
            thw=thw,
        )

        return pixel_outputs

    def _mae_forward(self, imgs, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore, thw = self._mae_forward_encoder(
            imgs, mask_ratio, mask
        )
        pred = self._mae_forward_decoder(latent, ids_restore, mask, thw)
        labels = []
        if self.pred_pixel_wt:
            if self.use_2d_patch:
                labels += self._get_pixel_label_2d(
                    imgs.detach(),
                    [mask.to(torch.bool)],
                    norm=self.cfg.MASK.NORM_PRED_PIXEL,
                )
            else:
                labels += self._get_pixel_label_3d(
                    imgs.detach(),
                    [mask.to(torch.bool)],
                    time_stride_loss=self.cfg.MASK.TIME_STRIDE_LOSS,
                    norm=self.cfg.MASK.NORM_PRED_PIXEL,
                )
        if self.pred_hog_wt:
            if self.use_2d_patch:
                labels += self._get_hog_label_2d(
                    imgs.detach(), [mask.to(torch.bool)]
                )
            else:
                labels += self._get_hog_label_3d(
                    imgs.detach(), [mask.to(torch.bool)]
                )

        self.counter += 1
        if self.cfg.VIS_MASK.ENABLE:
            return self._mae_visualize(imgs, pred, mask)
        loss, _ = self.multimse_loss(pred, labels)
        return loss

    def _mae_visualize(self, imgs, pred, mask):
        N, T, H, W, p, u, t, h, w = self.patch_info
        pred = pred[0]
        if self.cfg.MASK.TIME_STRIDE_LOSS:
            im_viz = imgs[:, :, :: self.cfg.MVIT.PATCH_STRIDE[0], :, :]
        else:
            im_viz = imgs
        reconstruct = self._unpatchify(
            pred * mask.reshape(N, t * h * w, 1)
            + self._patchify(
                im_viz, time_stride_loss=self.cfg.MASK.TIME_STRIDE_LOSS
            )
            * (1 - mask.reshape(N, t * h * w, 1))
        )
        masked = self._unpatchify(
            self._patchify(
                im_viz, time_stride_loss=self.cfg.MASK.TIME_STRIDE_LOSS
            )
            * (1 - mask.reshape(N, t * h * w, 1))
        )

        comparison = torch.stack(
            [im_viz, masked, reconstruct],
            dim=1,
        ).permute([0, 1, 3, 2, 4, 5])
        pfx = self.cfg.TEST.CHECKPOINT_FILE_PATH
        mr = self.cfg.AUG.MASK_RATIO
        for i in range(comparison.shape[0]):
            misc.plot_input_normed(
                comparison[i].cpu(),
                bboxes=(),
                texts=(),
                path=self.cfg.OUTPUT_DIR
                + "/vis_mask/vid/{}vis_video_in_mask_out_mr{}/vis_{}_{}.mp4".format(
                    pfx[pfx.rfind("/") + 1 : -5], mr, self.counter, i
                ),
                folder_path=self.cfg.OUTPUT_DIR
                + "/vis_mask/vid/{}vis_video_in_mask_out_mr{}".format(
                    pfx[pfx.rfind("/") + 1 : -5], mr
                ),
                make_grids=True,
                output_video=True,
            )
        return pred[0]

    def _maskfeat_forward(self, x, mask, return_all=False):
        x_embed, x_shape = self.patch_embed(x)
        if self.cfg.MASK.MAE_RND_MASK:
            _, mask, ids_restore, ids_keep = self._mae_random_masking(
                x_embed, self.cfg.AUG.MASK_RATIO, None
            )
            output_masks = [mask.to(torch.bool)]
        else:
            # take masks and labels from loader
            float_mask = mask.type_as(x)
            output_masks = self._get_multiscale_mask(float_mask)
        labels = []
        if self.pred_pixel_wt:
            if self.use_2d_patch:
                labels += self._get_pixel_label_2d(
                    x.detach(), output_masks, norm=self.cfg.MASK.NORM_PRED_PIXEL
                )
            else:
                labels += self._get_pixel_label_3d(
                    x.detach(), output_masks, norm=self.cfg.MASK.NORM_PRED_PIXEL
                )
        if self.pred_hog_wt:
            if self.use_2d_patch:
                labels += self._get_hog_label_2d(x.detach(), output_masks)
            else:
                labels += self._get_hog_label_3d(x.detach(), output_masks)

        x = x_embed
        T, H, W = self.T, self.H, self.W
        B, N, C = x.shape

        # switch input tokens by mask_token
        mask_tokens = self.mask_token.expand(B, N, -1)
        if self.cfg.MASK.MAE_RND_MASK:
            float_mask = mask.unsqueeze(-1)
        else:
            if self.use_2d_patch:
                float_mask = F.interpolate(
                    float_mask.unsqueeze(0), size=(H, W)
                )[0]
            else:
                float_mask = F.interpolate(float_mask, size=(H, W))
            float_mask = float_mask.flatten(1).unsqueeze(-1)
        x = x * (1 - float_mask) + mask_tokens * float_mask

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
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
                x = x + pos_embed
            else:
                x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        block_outputs = []
        for idx, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
            if idx in self.pretrain_depth:
                block_outputs.append(x)

        model_outputs = []
        if self.pred_pixel_wt:
            pixel_outputs = self.pred_head(
                block_outputs,
                output_masks,
                return_all=return_all,
                thw=thw,
            )
            model_outputs += pixel_outputs
        if self.pred_hog_wt:
            hog_outputs = self.pred_head(
                block_outputs,
                output_masks,
                return_all=return_all,
                thw=thw,
            )
            model_outputs += hog_outputs

        return model_outputs, labels

    def forward(self, x, return_all=False):
        if len(x) > 1:
            x, meta, mask = x
        else:
            x, mask = x[0], None

        # mask = None

        if self.cfg.MASK.MAE_ON:
            return self._mae_forward(
                x, mask_ratio=self.cfg.AUG.MASK_RATIO, mask=mask
            )
        else:
            return self._maskfeat_forward(x, mask, return_all)
        


# def video_mae_vit(cfg):
#     model = MaskMViT(cfg)
#     return model


# video_mae_vit_huge_patch14 = video_mae_vit_huge_patch14_dec512d8b
# 
