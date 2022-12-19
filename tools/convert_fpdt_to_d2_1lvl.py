#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch
import numpy as np

if __name__ == "__main__":
    input = sys.argv[1]
    if "base" in input:
        n_heads = 10
        d_model = 480
    if "small" in input:
        n_heads = 8
        d_model = 384
    if "lite" in input:
        n_heads = 8
        d_model = 256
    n_points = 4
    obj = torch.load(input, map_location="cpu")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        for ti, t in enumerate([8, 16, 32]):
            k = k.replace("patch_embed{}".format(t), "patch_embed{}".format(ti))
        if k.startswith("neck"):
            k = k.replace("neck.convs", "input_proj")
            k = k.replace("neck.extra_convs.0", "input_proj.3")
            k = k.replace("conv", "0")
            k = k.replace("gn", "1")
        if k.startswith("bbox_head"):
            k = k[10:]
            k = k.replace("level_embeds", "level_embed")
            if "encoder1" in k:
                k = k.replace("norms.0", "norm1")
                k = k.replace("norms.1", "norm2")
            else:
                k = k.replace("norms.1", "norm1")
                k = k.replace("norms.2", "norm2")
            k = k.replace("encoder1", "encoder")
            k = k.replace("encoder2", "decoder")
            k = k.replace("attentions.0", "self_attn")
            k = k.replace("ffns.0.layers.0.0", "linear1")
            k = k.replace("ffns.0.layers.1", "linear2")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()
    newmodel["input_proj.0.0.weight"] = newmodel["input_proj.2.0.weight"]
    newmodel["input_proj.0.1.weight"] = newmodel["input_proj.2.1.weight"]
    newmodel["input_proj.0.1.bias"] = newmodel["input_proj.2.1.bias"]
    for i in range(1, 4):
        newmodel.pop("input_proj.{}.0.weight".format(i))
        newmodel.pop("input_proj.{}.1.weight".format(i))
        newmodel.pop("input_proj.{}.1.bias".format(i))
    for li in range(6):
        param = newmodel[
            "transformer.decoder.layers.{}.self_attn.attention_weights.bias".format(li)
        ]
        param = param.reshape(n_heads, 5, n_points)
        newmodel[
            "transformer.decoder.layers.{}.self_attn.attention_weights.bias".format(li)
        ] = np.concatenate([param[:, 2:3], param[:, -1:]], axis=1).reshape(-1)

        param = newmodel[
            "transformer.decoder.layers.{}.self_attn.attention_weights.weight".format(
                li
            )
        ]
        param = param.reshape(n_heads, 5, n_points, d_model)
        newmodel[
            "transformer.decoder.layers.{}.self_attn.attention_weights.weight".format(
                li
            )
        ] = np.concatenate([param[:, 2:3], param[:, -1:]], axis=1).reshape(-1, d_model)

        param = newmodel[
            "transformer.decoder.layers.{}.self_attn.sampling_offsets.bias".format(li)
        ]
        param = param.reshape(n_heads, 5, n_points * 2)
        newmodel[
            "transformer.decoder.layers.{}.self_attn.sampling_offsets.bias".format(li)
        ] = np.concatenate([param[:, 2:3], param[:, -1:]], axis=1).reshape(-1)

        param = newmodel[
            "transformer.decoder.layers.{}.self_attn.sampling_offsets.weight".format(li)
        ]
        param = param.reshape(n_heads, 5, n_points * 2, d_model)
        newmodel[
            "transformer.decoder.layers.{}.self_attn.sampling_offsets.weight".format(li)
        ] = np.concatenate([param[:, 2:3], param[:, -1:]], axis=1).reshape(-1, d_model)

        param = newmodel[
            "transformer.encoder.layers.{}.self_attn.attention_weights.bias".format(li)
        ]
        param = param.reshape(n_heads, 4, n_points)
        newmodel[
            "transformer.encoder.layers.{}.self_attn.attention_weights.bias".format(li)
        ] = param[:, -1:].reshape(-1)

        param = newmodel[
            "transformer.encoder.layers.{}.self_attn.attention_weights.weight".format(
                li
            )
        ]
        param = param.reshape(n_heads, 4, n_points, d_model)
        newmodel[
            "transformer.encoder.layers.{}.self_attn.attention_weights.weight".format(
                li
            )
        ] = param[:, -1:].reshape(-1, d_model)

        param = newmodel[
            "transformer.encoder.layers.{}.self_attn.sampling_offsets.bias".format(li)
        ]
        param = param.reshape(n_heads, 4, n_points * 2)
        newmodel[
            "transformer.encoder.layers.{}.self_attn.sampling_offsets.bias".format(li)
        ] = param[:, -1:].reshape(-1)

        param = newmodel[
            "transformer.encoder.layers.{}.self_attn.sampling_offsets.weight".format(li)
        ]
        param = param.reshape(n_heads, 4, n_points * 2, d_model)
        newmodel[
            "transformer.encoder.layers.{}.self_attn.sampling_offsets.weight".format(li)
        ] = param[:, -1:].reshape(-1, d_model)
        newmodel["transformer.level_embed"] = newmodel["transformer.level_embed"][
            -1:, :
        ]

    res = {"model": newmodel, "__author__": "fp_detr", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
