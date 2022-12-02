import logging
from typing import Dict, List, Optional
import torch
from torch.utils.data.sampler import Sampler

from psd2.utils import comm
import numpy as np
from collections import Counter
import itertools
import functools
import math
import copy
from scipy.optimize import linear_sum_assignment
import assign_cost_cuda

logger = logging.getLogger(__name__)


class APKSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        data_dicts: List[Dict],
        p: int,
        k: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: Optional[int] = None,
    ):
        self._size = len(data_dicts)
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.bs = batch_size
        self.p = p
        self.k = k
        self.nb = (
            self._size // self.ba if drop_last else math.ceil(self._size / self.bs)
        )
        self.org_imgs_pids = []
        max_p = 0
        for dd in data_dicts:
            pids = np.array([ann["person_id"] for ann in dd["annotations"]])
            pids = pids[pids > -1].tolist()
            self.org_imgs_pids.append(pids)
            if len(pids) > max_p:
                max_p = len(pids)
        # padding
        for img_pids in self.org_imgs_pids:
            while len(img_pids) < max_p:
                img_pids.append(-2)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        # sequence should match the default BatchSampler
        g = torch.Generator()
        g.manual_seed(self._seed)

        while True:
            all_imgs_idxs = list(range(self._size))  # indices to data dict
            len_diff = self.bs * self.nb - len(all_imgs_idxs)
            if len_diff > 0:
                append_idxs = torch.randint(
                    0, self._size, (len_diff,), generator=g
                ).tolist()
                all_imgs_idxs.extend(append_idxs)
            all_imgs_idxs = np.array(all_imgs_idxs)
            if self._shuffle:
                seq = torch.randperm(len(all_imgs_idxs), generator=g).tolist()
                all_imgs_idxs = all_imgs_idxs[seq]
            batch_idxs_built = all_imgs_idxs[: self.nb]
            batch_data_built = [
                [copy.deepcopy(self.org_imgs_pids[idx])] for idx in batch_idxs_built
            ]  # init batch
            batch_idxs_built = np.expand_dims(
                np.array(batch_idxs_built), axis=-1
            ).tolist()
            data_left_idxs = all_imgs_idxs[self.nb :].tolist()
            data_left = copy.deepcopy(
                [self.org_imgs_pids[idx] for idx in data_left_idxs]
            )
            for _ in range(self.bs - 1):
                cost_m = self._assign_cost(data_left, batch_data_built)
                indices = linear_sum_assignment(cost_m)
                assigned_data_indices = []
                for i, j in zip(*indices):
                    assigned_data_indices.append(i)
                    batch_data_built[j].append(data_left[i])
                    batch_idxs_built[j].append(data_left_idxs[i])
                data_left_new = []
                data_left_idxs_new = []
                for di in range(len(data_left)):
                    if di not in assigned_data_indices:
                        data_left_new.append(data_left[di])
                        data_left_idxs_new.append(data_left_idxs[di])
                data_left = data_left_new
                data_left_idxs = data_left_idxs_new
            batch_idxs_seq = itertools.chain(*batch_idxs_built)
            yield from batch_idxs_seq

    def _assign_cost(self, data_left, batch_built):
        data_left_arr = np.array(data_left)  # l x p
        batch_built_arr = np.array(batch_built)  # t x b x p
        # may reduce memory cost
        fl_batch_built_arr = np.resize(
            batch_built_arr,
            (
                batch_built_arr.shape[0],
                batch_built_arr.shape[1] * batch_built_arr.shape[2],
            ),
        )  # t x bp
        ac = assign_cost_cuda.assign_cost(
            data_left_arr, fl_batch_built_arr, self.p, self.k
        )
        return ac


def _assign_cost(data_left, batch_built, p, k):
    data_left_arr = np.array(data_left)  # l x p
    batch_built_arr = np.array(batch_built)  # t x b x p
    fl_batch_built_arr = np.resize(
        batch_built_arr,
        (
            batch_built_arr.shape[0],
            batch_built_arr.shape[1] * batch_built_arr.shape[2],
        ),
    )  # t x bp
    ac = assign_cost_cuda.assign_cost(data_left_arr, fl_batch_built_arr, p, k)
    return ac
