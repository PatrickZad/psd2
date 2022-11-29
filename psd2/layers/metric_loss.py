# encoding: utf-8
"""
based on
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
from psd2.config import configurable
from psd2.utils import comm


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels1, labels2, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [M, N]
      labels1: pytorch LongTensor, with shape [M]
      labels2: pytorch LongTensor
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    M, N = dist_mat.shape

    # shape [M, N]
    is_pos = labels1[:, None].expand(M, N).eq(labels2[:, None].expand(N, M).t())
    is_neg = labels1[:, None].expand(M, N).ne(labels2[:, None].expand(N, M).t())
    idxs_mat = torch.arange(N, device=dist_mat.device)[None].expand(M, N)
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [M, 1]
    pos_lens = is_pos.sum(dim=-1).tolist()
    pos_dists = dist_mat[is_pos].split(pos_lens)
    pos_dists_idx = idxs_mat[is_pos].split(pos_lens)
    dist_ap = []
    p_inds = []
    for dist_sub, idx_sub in zip(pos_dists, pos_dists_idx):
        val, r_idx = torch.max(dist_sub, dim=-1)
        idx_p = idx_sub[r_idx]
        dist_ap.append(val.view(-1))
        p_inds.append(idx_p.view(-1))
    dist_ap = torch.cat(dist_ap, dim=0)
    p_inds = torch.cat(p_inds, dim=0)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [M, 1]
    neg_lens = is_neg.sum(dim=-1).tolist()
    neg_dists = dist_mat[is_neg].split(neg_lens)
    neg_dists_idx = idxs_mat[is_neg].split(neg_lens)
    dist_an = []
    n_inds = []
    for dist_sub, idx_sub in zip(neg_dists, neg_dists_idx):
        val, r_idx = torch.min(dist_sub, dim=-1)
        idx_p = idx_sub[r_idx]
        dist_an.append(val.view(-1))
        n_inds.append(idx_p.view(-1))
    dist_an = torch.cat(dist_an, dim=0)
    n_inds = torch.cat(n_inds, dim=0)

    if return_inds:
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


# this is to support taking unlabeled persons as negtives only
class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    @configurable
    def __init__(self, margin=None, do_norm=False):
        self.margin = margin
        self.do_norm = do_norm
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction="none")
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction="none")

    @classmethod
    def from_config(cls, loss_cfg):
        assert hasattr(loss_cfg, "TRI")
        tri_cfg = loss_cfg.TRI
        return {"margin": tri_cfg.MARGIN, "do_norm": tri_cfg.NORM_FEAt}

    def __call__(self, feats1, feats2, labels1, labels2):
        # feats1 are anchors
        if self.norm:
            feats1 = normalize(feats1, axis=-1)
            feats2 = normalize(feats2, axis=-1)
        dist_mat = torch.cdist(feats1[None], feats2[None]).squeeze(
            0
        )  # euclidean_dist(feats1, feats2)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels1, labels2)
        y = torch.ones_like(dist_an)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        num_anchors = feats1.shape[0]
        comm.synchronize()
        all_num_a = comm.all_gather(num_anchors)
        num_a = sum([num.to("cpu") for num in all_num_a])
        num_a = torch.clamp(num_a / comm.get_world_size(), min=1).item()
        loss_val = loss.sum() / num_a
        return {
            "loss_triplet": loss_val,
            "dist_ap": dist_ap.mean(),
            "dist_an": dist_an.mean(),
        }
