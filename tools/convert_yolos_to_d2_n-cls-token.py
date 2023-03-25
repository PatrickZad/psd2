#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch


if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if "model" in obj:
        obj = obj["model"]
    if "state_dict" in obj:
        obj = obj["state_dict"]
    name_pop = []
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        k = k.replace("backbone", "transformer")
        if "cls_token" in k:
            continue
        if "mid_pos_embed" in k:
            param = obj.pop(old_k)
            param = param[:, :, 1:]
            newmodel[k] = param.detach().numpy()
        elif "pos_embed" in k:
            param = obj.pop(old_k)
            param = param[:, 1:]
            newmodel[k] = param.detach().numpy()

        elif "class_embed.layers.2" in k:
            param = obj.pop(old_k)
            param = torch.stack([param[1], param[-1]], dim=0)
            newmodel[k] = param.detach().numpy()
        else:
            newmodel[k] = obj.pop(old_k).detach().numpy()
        print(old_k, "->", k)
    res = {"model": newmodel, "__author__": "yolos", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
