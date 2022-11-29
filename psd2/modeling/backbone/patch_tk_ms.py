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
from torch.nn.init import _calculate_fan_in_and_fan_out

import warnings


from itertools import repeat

try:
    from torch._six import container_abcs
except:
    import collections.abc as container_abcs
import collections.abc
from .backbone import Backbone
from .build import BACKBONE_REGISTRY


def parse(x, n):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, n))


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
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
        self.pretrain_num_patches = (
            self.pretrain_grid_size[0] * self.pretrain_grid_size[1]
        )
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

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
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


class PatchTokenizerMS(Backbone):
    """A multi-scale"""

    def __init__(
        self,
        pretrain_img_size=224,
        in_chans=3,
        embed_dim=768,
        embed_layer=PatchEmbed,  # not in swin
        norm_layer=None,
        weight_init="",  # not in swin
        out_indices=(
            8,
            16,
            32,
        ),  # use (absolute) down-sample rates as out_indices
        new_norms=False,
        pretrained=None,
        init_cfg=None,
    ):
        """
        Args:
            pretrain_img_size (int, tuple): input image size during pretraining
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            pretrained: path to pretrained model (deprecated)
            init_cfg: configs for initialization (preferred)
        """
        super().__init__()
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        self.pretrained = pretrained
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is a deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                warnings.warn("Training from scratch, you should specify the init_cfg")
        else:
            raise TypeError("pretrained must be a str or None")

        self.out_indices = out_indices
        self.new_norms = new_norms
        embed_dim = parse(embed_dim, len(out_indices))
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        for idx, down_rate_abs in enumerate(self.out_indices):
            layer_name = f"patch_embed{down_rate_abs}"
            down_rate_rel = (
                down_rate_abs // self.out_indices[idx - 1] if idx > 0 else down_rate_abs
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
            for i_layer in range(len(out_indices)):  # absolute layer index
                layer = norm_layer(self.num_features[i_layer])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)

        self._out_features = ["d{}".format(o_id) for o_id in out_indices]
        self._out_feature_channels = {
            self._out_features[i]: embed_dim[i] for i in range(len(self._out_features))
        }
        self._out_feature_strides = {
            self._out_features[i]: out_indices[i]
            for i in range(len(self._out_features))
        }

    """
    def init_weights(self):
        assert self.weight_init in ("jax", "jax_nlhb", "nlhb", "")
        if self.weight_init.startswith("jax"):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, jax_impl=True)
        else:
            self.apply(_init_vit_weights)
        self._is_init = True

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}
    """

    def forward(self, x):
        outs = {}
        for idx, down_rate_abs in enumerate(self.out_indices):
            layer_name = f"patch_embed{down_rate_abs}"
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
            outs[self._out_features[idx]] = x_out

        return outs


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
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
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


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@BACKBONE_REGISTRY.register()
def build_ptkms_backbone(cfg, input_shape):
    bk_cfg = cfg.DETECTOR.MODEL.BACKBONE
    return PatchTokenizerMS(
        pretrain_img_size=bk_cfg.PRETRAIN_IMG_SIZE,
        in_chans=bk_cfg.IN_CHANNELS,
        embed_dim=bk_cfg.EMB_DIM,
        out_indices=bk_cfg.OUT_INDICES,
        new_norms=bk_cfg.NEW_NORMS,
    )
