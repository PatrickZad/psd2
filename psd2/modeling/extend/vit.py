import math
import warnings
import torch
import torch.nn as nn
from psd2.layers import DropPath
import torch.utils.checkpoint as checkpoint
from copy import deepcopy
from functools import partial


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # 3, B, num_head, N, c
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        else:
            return x


class QMaskAttention(Attention):
    def __init__(self, len_q, attn_inst=None, *args, **kws):
        if attn_inst is None:
            super().__init__(*args, **kws)
        else:
            super(Attention, self).__init__()
            self.num_heads = attn_inst.num_heads
            self.scale = attn_inst.scale

            self.qkv = deepcopy(attn_inst.qkv)
            self.attn_drop = deepcopy(attn_inst.attn_drop)
            self.proj = deepcopy(attn_inst.proj)
            self.proj_drop = deepcopy(attn_inst.proj_drop)
        self.len_q = len_q

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # 3, B, num_head, N, c
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_mask = attn.new_zeros(attn.shape)
        attn_mask[..., -self.len_q :] = True
        attn.masked_fill_(attn_mask.bool(), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        else:
            return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class QMaskBlock(Block):
    def __init__(
        self,
        len_q,
        dim=None,
        num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        block_inst=None,
    ):
        super(Block, self).__init__()
        if block_inst is None:
            self.norm1 = norm_layer(dim)
            self.attn = QMaskAttention(
                len_q,
                attn_inst=None,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
        else:
            self.norm1 = deepcopy(block_inst.norm1)
            self.attn = QMaskAttention(len_q, block_inst.attn)
            # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
            self.drop_path = deepcopy(block_inst.drop_path)
            self.norm2 = deepcopy(block_inst.norm2)
            self.mlp = deepcopy(block_inst.mlp)
        self.len_q = len_q
        # hack impl
        self.qsa = None

    def forward(self, x, return_attention=False):
        img_seq, q_seq = torch.split(x, [x.shape[1] - self.len_q, self.len_q], dim=1)
        q_seq = self.qsa(q_seq)
        x = torch.cat([img_seq, q_seq], dim=1)
        return super().forward(x, return_attention)


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, org_img_size=224, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = (org_img_size, org_img_size)
        self.backbone = backbone
        with torch.no_grad():
            # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
            # map for all networks, the feature metadata has reliable channel and stride info, but using
            # stride to calc feature dim requires info about padding of each stage that isn't captured.
            training = backbone.training
            if training:
                backbone.eval()
            o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
            feature_size = o.shape[-2:]
            feature_dim = o.shape[1]
            backbone.train(training)
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        pretrain_size,
        patch_size,
        embed_dim,
        num_patches,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        is_distill=False,
    ):
        super().__init__()
        self.img_size = pretrain_size
        self.depth = depth
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.hidden_dim = (
            self.embed_dim
        )  # num_features for consistency with other models

        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if is_distill:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 2, self.embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, self.embed_dim)
            )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # set finetune flag
        self.has_mid_pe = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def finetune_det(
        self,
        img_size=[800, 1344],
        det_token_num=100,
        mid_pe_size=None,
    ):
        # import pdb;pdb.set_trace()

        import math

        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dim)).to(
            self.pos_embed.device
        )
        self.det_token = trunc_normal_(self.det_token, std=0.02)
        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = torch.zeros(
            1, det_token_num, self.embed_dim, device=self.pos_embed.device
        )
        det_pos_embed = trunc_normal_(det_pos_embed, std=0.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        if isinstance(self.patch_size, int):
            ph, pw = self.patch_size, self.patch_size
        else:
            ph, pw = self.patch_size
        P_H, P_W = (
            self.img_size[0] // ph,
            self.img_size[1] // pw,
        )
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // ph, W // pw
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(
            torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        )
        self.img_size = img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            print("No mid pe")
        else:
            print("Has mid pe")
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    1,
                    1
                    + (mid_pe_size[0] * mid_pe_size[1] // (ph*pw))
                    + det_token_num,
                    self.embed_dim,
                    device=self.pos_embed.device
                )
            )
            trunc_normal_(self.mid_pos_embed, std=0.02)
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "det_token"}

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.det_token_num :, :]
        patch_pos_embed = pos_embed[:, 1 : -self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        if isinstance(self.patch_size, int):
            ph, pw = self.patch_size, self.patch_size
        else:
            ph, pw = self.patch_size

        P_H, P_W = (
            self.img_size[0] // ph,
            self.img_size[1] // pw,
        )
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // ph, W // pw
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat(
            (cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1
        )
        return scale_pos_embed

    def InterpolateMidPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.det_token_num :, :]
        patch_pos_embed = pos_embed[:, :, 1 : -self.det_token_num, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        D, B, E, Q = patch_pos_embed.shape
        if isinstance(self.patch_size, int):
            ph, pw = self.patch_size, self.patch_size
        else:
            ph, pw = self.patch_size

        P_H, P_W = (
            self.mid_pe_size[0] // ph,
            self.mid_pe_size[1] // pw,
        )
        patch_pos_embed = patch_pos_embed.view(D * B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // ph, W // pw
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = (
            patch_pos_embed.flatten(2)
            .transpose(1, 2)
            .contiguous()
            .view(D, B, new_P_H * new_P_W, E)
        )
        scale_pos_embed = torch.cat(
            (cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2
        )
        return scale_pos_embed

    def forward_features(self, feat_seq, imgt_shape):
        # import pdb;pdb.set_trace()
        B, H, W = imgt_shape[0], imgt_shape[2], imgt_shape[3]
        x = feat_seq
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.pos_embed, img_size=(H, W)
            )
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i in range(len((self.blocks))):
            # x = self.blocks[i](x)
            x = checkpoint.checkpoint(self.blocks[i], x)  # saves mem, takes time
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)
        return x

        # return x[:, -self.det_token_num :, :]
        # cls_out, feat_out, det_out = torch.split(
        #    x, [self.cls_token.shape[1], h_pe * w_pe, self.det_token.shape[1]], dim=1
        # )
        # feat_out = feat_out.transpose(1, 2).reshape((B, -1, h_pe, w_pe))
        # return cls_out, feat_out, det_out

    def forward_return_all_selfattention(self, feat_seq, imgt_shape):
        B, H, W = imgt_shape[0], imgt_shape[2], imgt_shape[3]
        x = feat_seq
        # interpolate init pe
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(
                self.pos_embed, img_size=(H, W)
            )
        else:
            temp_pos_embed = self.pos_embed
        # interpolate mid pe
        if self.has_mid_pe:
            # temp_mid_pos_embed = []
            if (self.mid_pos_embed.shape[2] - 1 - self.det_token_num) != x.shape[1]:
                temp_mid_pos_embed = self.InterpolateMidPosEmbed(
                    self.mid_pos_embed, img_size=(H, W)
                )
            else:
                temp_mid_pos_embed = self.mid_pos_embed

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        det_token = self.det_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x, det_token), dim=1)
        x = x + temp_pos_embed
        x = self.pos_drop(x)
        output = []
        for i in range(len((self.blocks))):
            x, attn = self.blocks[i](x, return_attention=True)

            if i == len(self.blocks) - 1:
                output.append(attn)
            if self.has_mid_pe:
                if i < (self.depth - 1):
                    x = x + temp_mid_pos_embed[i]

        x = self.norm(x)

        return output

    def forward(self, feat_seq, imgt_shape, return_attention=False):
        if return_attention == True:
            # return self.forward_selfattention(x)
            return self.forward_return_all_selfattention(feat_seq, imgt_shape)
        else:
            x = self.forward_features(feat_seq, imgt_shape)
            return x


class QMaskVisionTransformer(VisionTransformer):
    def finetune_det(
        self,
        det_token_start=0,
        shared_qsa=True,
        img_size=[800, 1344],
        det_token_num=100,
        mid_pe_size=None,
    ):
        # import pdb;pdb.set_trace()

        import math

        g = math.pow(self.pos_embed.size(1) - 1, 0.5)
        if int(g) - g != 0:
            self.pos_embed = torch.nn.Parameter(self.pos_embed[:, 1:, :])

        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=0.02)
        cls_pos_embed = self.pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=0.02)
        patch_pos_embed = self.pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = (
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_P_H, new_P_W),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = torch.nn.Parameter(
            torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        )
        self.img_size = img_size
        if mid_pe_size == None:
            self.has_mid_pe = False
            print("No mid pe")
        else:
            print("Has mid pe")
            self.mid_pos_embed = nn.Parameter(
                torch.zeros(
                    self.depth - 1,
                    1,
                    1
                    + (mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2)
                    + det_token_num,
                    self.embed_dim,
                )
            )
            trunc_normal_(self.mid_pos_embed, std=0.02)
            self.has_mid_pe = True
            self.mid_pe_size = mid_pe_size
        qsa = deepcopy(self.blocks[torch.randint(0, self.depth, (1,)).item()])
        for i in range(det_token_start, self.depth):
            blk = QMaskBlock(det_token_num, block_inst=self.blocks[i])
            if shared_qsa:
                blk.qsa = qsa
            else:
                blk.qsa = deepcopy(qsa)
            self.blocks[i] = blk
        self.shared_qsa = shared_qsa


"""
vit_tiny
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


vit_small
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )



vit_smalldwr
    model = VisionTransformer(
        img_size=240,
        patch_size=16,
        embed_dim=330,
        depth=14,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


vit_base
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        is_distill=True,
    )
"""
