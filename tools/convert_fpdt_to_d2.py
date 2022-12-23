#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch


if __name__ == "__main__":
    input = sys.argv[1]

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

    res = {"model": newmodel, "__author__": "fp_detr", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
