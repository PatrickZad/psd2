import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
from .utils import (
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
)
# detctron2 version mvitv2
logger = logging.getLogger(__name__)


__all__ = ["MViT"]


def attention_pool(x, pool, norm=None):
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H1, W1) -> (B, H1, W1, C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x

def prompt_attention_pool(x_prompt,pool:nn.Conv2d,norm=None):
    # use duplicate padding to be compatible
    B,N,C=x_prompt.shape
    x_prompt=x_prompt.permute(B*N,C,1,1)
    kh,kw=pool.kernel_size
    if kh%2!=0:
        h_side_padding=((kh-1)//2,(kh-1)//2)
    else:
        h_side_padding=(kh//2-1,kh//2)
    if kw%2!=0:
        w_side_padding=((kw-1)//2,(kw-1)//2)
    else:
        w_side_padding=(kw//2-1,kw//2)
    x_prompt=tF.pad(x_prompt,(w_side_padding[0],w_side_padding[1],h_side_padding[0],h_side_padding[1]),mode="replicate")
    pooled_x_prompt=tF.conv2d(x_prompt,weight=pool.weight,bias=pool.bias,stride=pool.stride,padding=(0,0),dilation=pool.dilation,groups=pool.groups)
    pooled_x_prompt.reshape(B,N,C)
    return pooled_x_prompt

class MultiScaleAttention(nn.Module):
    """Multiscale Multi-head Attention block."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # qkv pooling
        pool_padding = [k // 2 for k in pool_kernel]
        dim_conv = dim_out // num_heads
        self.pool_q = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_q,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = norm_layer(dim_conv)
        self.pool_k = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_k = norm_layer(dim_conv)
        self.pool_v = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_v = norm_layer(dim_conv)

        self.window_size = window_size
        if window_size:
            self.q_win_size = window_size // stride_q
            self.kv_win_size = window_size // stride_kv
        self.residual_pooling = residual_pooling

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            assert input_size[0] == input_size[1]
            size = input_size[0]
            rel_dim = 2 * max(size // stride_q, size // stride_kv) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H, W, C)
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5).contiguous()
        # q, k, v with shape (B * nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q)
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        ori_q = q
        if self.window_size:
            q, q_hw_pad = window_partition(q, self.q_win_size)
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw)

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(x, self.q_win_size, q_hw_pad, ori_q.shape[1:3])

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """Multiscale Transformer blocks"""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if stride_q > 1:
            kernel_skip = stride_q + 1
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_q, padding_skip, ceil_mode=False)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_block = self.attn(x_norm)

        if hasattr(self, "proj"):
            x = self.proj(x_norm)
        if hasattr(self, "pool_skip"):
            x = attention_pool(x, self.pool_skip)

        x = x + self.drop_path(x_block)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MViT(nn.Module):
    """
    This module implements Multiscale Vision Transformer (MViT) backbone in :paper:'mvitv2'.
    """

    def __init__(
        self,
        img_size=224,
        patch_stride=(4, 4),
        embed_dim=96,
        depth=16,
        num_heads=1,
        last_block_indexes=(0, 2, 11, 15),
        qkv_pool_kernel=(3, 3),
        adaptive_kv_stride=4,
        adaptive_window_size=56,
        residual_pooling=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=False,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_features=("scale2", "scale3", "scale4", "scale5"),
        with_cp=False,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_kernel (tuple): kernel size for patch embedding.
            patch_stride (tuple): stride size for patch embedding.
            patch_padding (tuple): padding size for patch embedding.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of MViT.
            num_heads (int): Number of base attention heads in each MViT block.
            last_block_indexes (tuple): Block indexes for last blocks in each stage.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            adaptive_kv_stride (int): adaptive stride size for kv pooling.
            adaptive_window_size (int): adaptive window size for window attention blocks.
            residual_pooling (bool): If true, enable residual pooling.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_features (tuple): name of the feature maps from each stage.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        if use_abs_pos:
            # Initialize absoluate positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_stride[0]) * (
                pretrain_img_size // patch_stride[1]
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dim_out = embed_dim
        stride_kv = adaptive_kv_stride
        window_size = adaptive_window_size
        input_size = (img_size // patch_stride[0], img_size // patch_stride[1])
        stage = 2
        stride = patch_stride[0]
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Multiply stride_kv by 2 if it's the last block of stage2 and stage3.
            if i == last_block_indexes[1] or i == last_block_indexes[2]:
                stride_kv_ = stride_kv * 2
            else:
                stride_kv_ = stride_kv
            # hybrid window attention: global attention in last three stages.
            window_size_ = 0 if i in last_block_indexes[1:] else window_size
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qkv_pool_kernel=qkv_pool_kernel,
                stride_q=2 if i - 1 in last_block_indexes else 1,
                stride_kv=stride_kv_,
                residual_pooling=residual_pooling,
                window_size=window_size_,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
            )
            self.blocks.append(block)
            self.with_cp=with_cp
            embed_dim = dim_out
            if i in last_block_indexes:
                name = f"scale{stage}"
                if name in out_features:
                    self._out_feature_channels[name] = dim_out
                    self._out_feature_strides[name] = stride
                    self.add_module(f"{name}_norm", norm_layer(dim_out))

                dim_out *= 2
                num_heads *= 2
                stride_kv = max(stride_kv // 2, 1)
                stride *= 2
                stage += 1
            if i - 1 in last_block_indexes:
                window_size = window_size // 2
                input_size = [s // 2 for s in input_size]

        self._out_features = out_features
        self._last_block_indexes = last_block_indexes

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, x.shape[1:3])

        outputs = {}
        stage = 2
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self._last_block_indexes:
                name = f"scale{stage}"
                if name in self._out_features:
                    x_out = getattr(self, f"{name}_norm")(x)
                    outputs[name] = x_out.permute(0, 3, 1, 2)
                stage += 1

        return outputs
from copy import deepcopy
from collections import OrderedDict
class SideMViT(MViT):
    def __init__(self, side_start_stage, *args, **kws):  # [1,2,3,4]
        super().__init__(*args, **kws)
        self.side_start_stage = side_start_stage
        self.init_side()
    def init_side(
        self
    ):
        self.side_blocks = nn.ModuleList()
        for i in range(self.side_start_stage - 1, len(self._last_block_indexes)):
            start_i=0 if i==0 else self._last_block_indexes[i-1]+1
            for bi in range(start_i,self._last_block_indexes[i]+1):
                side_block = deepcopy(self.blocks[bi])
                self.side_blocks.append(side_block)
            if hasattr(self, f"scale{i+2}_norm"):
                layer = deepcopy(getattr(self, f"scale{i+2}_norm"))
                layer_name = f"side_norm{i}"
                self.add_module(layer_name, layer)

    def load_side(self, state_dict):
        resume = False
        for k in state_dict:
            if "side_blocks" in k:
                resume = True
                break
        if not resume:
            block_params = OrderedDict()
            norm_params = {
                i: OrderedDict()
                for i in range(0, len(self._last_block_indexes))
                if hasattr(self, f"side_norm{i}")
            }
            for k, v in state_dict.items():
                if k.startswith("trans.blocks."):
                    k_kws = k.split(".")[1:]
                    idx = int(k_kws[1])
                    side_start_bi=0 if self.side_start_stage==0 else self._last_block_indexes[self.side_start_stage-2]+1
                    if idx >= side_start_bi:
                        n_idx = idx - side_start_bi
                        n_k = ".".join([str(n_idx)] + k_kws[2:])
                        block_params[n_k] = v
                elif k.startswith("trans.scale"): # norm
                    idx = int(k[len("trans.scale")])-2
                    if idx + 1 >= self.side_start_stage and hasattr(
                        self, f"side_norm{idx}"
                    ):
                        k_kws = k.split(".")
                        nk = ".".join(k_kws[2:])
                        norm_params[idx][nk] = v


            res = self.side_blocks.load_state_dict(block_params, strict=False)
            print("parameters of *side_blocks* haved been loaded:\n", str(res))
            for i, p in norm_params.items():
                res = getattr(self, f"side_norm{i}").load_state_dict(p, strict=False)
                print(f"parameters of *side_norm{i}* haved been loaded:\n", str(res))




#NOTE assume separately learned k-prompts and v-prompts 
class PromptedMultiScaleAttention(MultiScaleAttention):
    def forward(self, x,prompts):
        if prompts is None:
            return super().forward(x)
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H, W, C)
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        # q, k, v with shape (B * nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q) # Bh * H * W * c
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        L,_=prompts.shape[1:]
        kvp=prompts.reshape(B, L, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        kp,vp=kvp.reshape(2, B * self.num_heads, L, -1).unbind(0) # Bh * L * c

        ori_q = q
        if self.window_size:
            # Bh, H // w, w, W // w, w, c -> Bh, H//w, W // w, w, w, c -> (Bh * H * W // w // w) * w * w * c
            q, q_hw_pad = window_partition(q, self.q_win_size) # b * w * w * c
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
            # repeat prompts
            kp=kp.view(B * self.num_heads,1,1,L,-1).repeat(1,kv_hw_pad[0]//self.kv_win_size,kv_hw_pad[1]//self.kv_win_size, 1 , 1).contiguous().flatten(0,2)# (Bh * H * W // w // w) * L * c
            vp=vp.view(B * self.num_heads,1,1,L,-1).repeat(1,kv_hw_pad[0]//self.kv_win_size,kv_hw_pad[1]//self.kv_win_size, 1 , 1).contiguous().flatten(0,2)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1) # b * hw * c
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw)
        

        attn_img2prompt=(q * self.scale)@kp.transpose(-2, -1)
        attn=torch.cat([attn_img2prompt,attn],dim=-1)

        attn = attn.softmax(dim=-1)
        x = attn @ torch.cat([vp,v],dim=1)

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(x, self.q_win_size, q_hw_pad, ori_q.shape[1:3])

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        
        return x

class PromptedMultiScaleBlock(nn.Module):
    """Multiscale Transformer blocks"""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PromptedMultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if stride_q > 1:
            kernel_skip = stride_q + 1
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_q, padding_skip, ceil_mode=False)

    def forward(self, x, prompts):
        x_norm = self.norm1(x)
        #TODO test pre prompts norm 
        x_block = self.attn(x_norm,prompts)
        if hasattr(self, "proj"):
            x = self.proj(x_norm)
        if hasattr(self, "pool_skip"):
            x = attention_pool(x, self.pool_skip)

        x = x + self.drop_path(x_block)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PromptedMViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_kernel=(7, 7),
        patch_stride=(4, 4),
        patch_padding=(3, 3),
        in_chans=3,
        embed_dim=96,
        depth=16,
        num_heads=1,
        last_block_indexes=(0, 2, 11, 15),
        qkv_pool_kernel=(3, 3),
        adaptive_kv_stride=4,
        adaptive_window_size=56,
        residual_pooling=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=False,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_features=("scale2", "scale3", "scale4", "scale5"),
        with_cp=False,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_kernel (tuple): kernel size for patch embedding.
            patch_stride (tuple): stride size for patch embedding.
            patch_padding (tuple): padding size for patch embedding.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of MViT.
            num_heads (int): Number of base attention heads in each MViT block.
            last_block_indexes (tuple): Block indexes for last blocks in each stage.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            adaptive_kv_stride (int): adaptive stride size for kv pooling.
            adaptive_window_size (int): adaptive window size for window attention blocks.
            residual_pooling (bool): If true, enable residual pooling.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_features (tuple): name of the feature maps from each stage.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        if use_abs_pos:
            # Initialize absoluate positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_stride[0]) * (
                pretrain_img_size // patch_stride[1]
            )
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dim_out = embed_dim
        stride_kv = adaptive_kv_stride
        window_size = adaptive_window_size
        input_size = (img_size // patch_stride[0], img_size // patch_stride[1])
        stage = 2
        stride = patch_stride[0]
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Multiply stride_kv by 2 if it's the last block of stage2 and stage3.
            if i == last_block_indexes[1] or i == last_block_indexes[2]:
                stride_kv_ = stride_kv * 2
            else:
                stride_kv_ = stride_kv
            # hybrid window attention: global attention in last three stages.
            window_size_ = 0 if i in last_block_indexes[1:] else window_size
            block = PromptedMultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qkv_pool_kernel=qkv_pool_kernel,
                stride_q=2 if i - 1 in last_block_indexes else 1,
                stride_kv=stride_kv_,
                residual_pooling=residual_pooling,
                window_size=window_size_,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
            )
            self.blocks.append(block)
            self.with_cp=with_cp
            embed_dim = dim_out
            if i in last_block_indexes:
                name = f"scale{stage}"
                if name in out_features:
                    self._out_feature_channels[name] = dim_out
                    self._out_feature_strides[name] = stride
                    self.add_module(f"{name}_norm", norm_layer(dim_out))

                dim_out *= 2
                num_heads *= 2
                stride_kv = max(stride_kv // 2, 1)
                stride *= 2
                stage += 1
            if i - 1 in last_block_indexes:
                window_size = window_size // 2
                input_size = [s // 2 for s in input_size]

        self._out_features = out_features
        self._last_block_indexes = last_block_indexes

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        raise NotImplementedError

class SidePromptedMViT(SideMViT, PromptedMViT):
    def __init__(self, side_start_stage, *args, **kws):
        PromptedMViT.__init__(self, *args, **kws)
        self.side_start_stage = side_start_stage
        self.init_side()
class SidePromptedMViTLS(SideMViT, PromptedMViT):
    def __init__(self, side_start_stage, *args, **kws):
        PromptedMViT.__init__(self, *args, **kws)
        self.side_start_stage = side_start_stage
        self.blocks[self._last_block_indexes[-2]+1].attn.pool_q.stride = (1,1)
        self.blocks[self._last_block_indexes[-2]+1].attn.q_win_size*=2
        if hasattr(self.blocks[self._last_block_indexes[-2]+1],"pool_skip"):
                self.blocks[self._last_block_indexes[-2]+1].pool_skip=nn.Identity()
        self.init_side()
class SidePromptedMViTDC(SideMViT, PromptedMViT):
    def __init__(self, side_start_stage, *args, **kws):
        PromptedMViT.__init__(self, *args, **kws)
        self.side_start_stage = side_start_stage
        self.blocks[self._last_block_indexes[-2]+1].attn.pool_q.stride = (1,1)
        self.blocks[self._last_block_indexes[-2]+1].attn.pool_q.padding = (2,2)
        self.blocks[self._last_block_indexes[-2]+1].attn.pool_q.dilation=(2,2)
        if hasattr(self.blocks[self._last_block_indexes[-2]+1],"pool_skip"):
                self.blocks[self._last_block_indexes[-2]+1].pool_skip=nn.Identity()
        self.init_side()
