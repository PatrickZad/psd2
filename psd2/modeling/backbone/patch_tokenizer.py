""" Modified from timm

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.init import _calculate_fan_in_and_fan_out
import collections
import warnings
from itertools import repeat

try:
    from torch._six import container_abcs
except:
    import collections.abc as container_abcs
from .backbone import Backbone
from .build import BACKBONE_REGISTRY
from psd2.layers import Conv2d, ShapeSpec


class PatchEmbed_(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        pretrain_img_size = to_2tuple(pretrain_img_size)
        patch_size = to_2tuple(patch_size)
        self.pretrain_img_size = pretrain_img_size
        self.patch_size = patch_size
        self.pretrain_grid_size = (
            pretrain_img_size[0] // patch_size[0],
            pretrain_img_size[1] // patch_size[1],
        )
        self.num_patches = self.pretrain_grid_size[0] * self.pretrain_grid_size[1]
        self.flatten = flatten

        self.proj = Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        # TODO: consider padding
        assert (H % self.patch_size[0] == 0) and (W % self.patch_size[1] == 0)
        x = self.proj(x)

        _, C_new, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        if not self.flatten:
            x = x.transpose(1, 2).contiguous().view(B, C_new, H_new, W_new)
        return x


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise NotImplementedError("Unsupported mode: {}!".format(mode))

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        init.trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


# FPDETR
class PatchTokenizerMS(Backbone):
    """A multi-scale"""

    def __init__(
        self,
        pretrain_img_size=224,
        in_chans=3,
        embed_dim=768,
        embed_layer=PatchEmbed_,  # not in swin
        norm_layer=None,
        out_features=None,
        weight_init="",  # not in swin
        strides=[8, 16, 32],
        new_norms=False,
    ):
        """
        Args:
            pretrain_img_size (int, tuple): input image size during pretraining
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        assert len(strides) > 0
        if out_features is not None:
            strides = strides[: int(out_features[-1][1:])]
            self._out_features = out_features
        else:
            self._out_features = [f"t{len(strides)}"]
        self.new_norms = new_norms
        self.strides = strides
        embed_dim = parse(embed_dim, len(strides))
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self._out_feature_strides = {f"t{i+1}": strides[i] for i in range(len(strides))}
        self._out_feature_channels = {
            f"t{i+1}": embed_dim[i] for i in range(len(strides))
        }

        for idx, down_rate_abs in enumerate(self.strides):
            layer_name = f"patch_embed{idx}"
            down_rate_rel = (
                down_rate_abs // strides[idx - 1] if idx > 0 else down_rate_abs
            )
            in_chans = embed_dim[idx - 1] if idx > 0 else in_chans
            img_size = pretrain_img_size // down_rate_abs
            layer = embed_layer(
                pretrain_img_size=img_size,
                patch_size=down_rate_rel,
                in_chans=in_chans,
                embed_dim=embed_dim[idx],
                norm_layer=norm_layer,
                flatten=False,
            )
            self.add_module(layer_name, layer)

        self.weight_init = weight_init

        if self.new_norms:
            for i_layer in range(len(self.strides)):  # absolute layer index
                layer = norm_layer(self.embed_dim[i_layer])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)

    def init_weights(self):
        assert self.weight_init in ("jax", "jax_nlhb", "nlhb", "")
        if self.weight_init.startswith("jax"):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, jax_impl=True)
        else:
            self.apply(_init_vit_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward(self, x):
        outs = {}
        for idx, down_rate_abs in enumerate(self.strides):
            layer_name = f"patch_embed{idx}"
            layer = getattr(self, layer_name)
            x = layer(x)
            if self.new_norms:
                norm_layer = getattr(self, f"norm{idx}")
                # B, C, H, W = x.shape
                x_out = x.permute(0, 2, 3, 1).contiguous()
                x_out = norm_layer(x_out)
                x_out = x_out.permute(0, 3, 1, 2).contiguous()
            else:
                x_out = x
            out_name = f"t{idx+1}"
            if out_name in self._out_features:
                outs[out_name] = x_out

        return outs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def _init_vit_weights(m, n: str = "", jax_impl: bool = False):
    """ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith("head"):
            raise NotImplementedError
        elif n.startswith("pre_logits"):
            # removed for fine-tuning
            warnings.warn("Should have been removed for fine-tuning!")
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if "mlp" in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def parse(x, n):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, n))


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple(repeat(x, 2))


# YOLOS
class PatchEmbed(PatchEmbed_, Backbone):
    def __init__(self, *args, **kws):
        PatchEmbed_.__init__(self, *args, **kws)
        self._out_features = ["out"]

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return self.patch_size[0]

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self.embed_dim,
                stride=self.patch_size[0],
            )
            for name in self._out_features
        }

    def forward(self, x):
        ret = super().forward(x)
        return {self._out_features[-1]: ret}


@BACKBONE_REGISTRY.register()
def build_patch_tokenizerms(cfg, input_shape):
    pt_cfg = cfg.MODEL.PATCH_TOKENIZER
    return PatchTokenizerMS(
        pt_cfg.PRETRAIN_IMG_SIZE,
        input_shape.channels,
        pt_cfg.EMBED_DIM,
        PatchEmbed_,
        out_features=pt_cfg.OUT_FEATURES,
        strides=pt_cfg.STRIDES,
        weight_init=pt_cfg.WEIGHT_INIT,
        new_norms=pt_cfg.NEW_NORMS,
    )


@BACKBONE_REGISTRY.register()
def build_patch_embed(cfg, input_shape):
    pt_cfg = cfg.MODEL.PATCH_EMBED
    return PatchEmbed(
        pt_cfg.PRETRAIN_IMG_SIZE,
        pt_cfg.PATCH_SIZE,
        input_shape.channels,
        pt_cfg.EMBED_DIM,
        norm_layer=None,
        flatten=False,
    )
