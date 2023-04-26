#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch


if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if "teacher" in obj:
        obj = obj["teacher"]
    newmodel = {}
    for k in list(obj.keys()):
        old_k = k
        if "patch_embed" in k:
            k = k.replace("backbone.patch_embed", "backbone")
            k=k.replace("projection","proj")
        else:
            k = k.replace("backbone", "transformer")
            k = k.replace("stages", "layers")
            k = k.replace("attn.w_msa", "attn")
            k = k.replace("ffn.layers.0.0", "mlp.fc1")
            k = k.replace("ffn.layers.1", "mlp.fc2")
        newmodel[k] = obj.pop(old_k).detach().numpy()
        print(old_k, "->", k)
    res = {"model": newmodel, "__author__": "solider", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
