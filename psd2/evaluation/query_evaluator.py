from copy import copy
from typing import List, OrderedDict

from PIL import Image

import psd2.utils.comm as comm

import torch
import logging

from psd2.utils.visualizer import VisImage, Visualizer
from .evaluator import DatasetEvaluator
import itertools
import os
import numpy as np
from torchvision.ops.boxes import box_iou

import copy

import random
import logging
from os.path import join as opj

import math
import psd2.utils.comm as comm
import random
import shutil
from torchvision.ops.boxes import box_iou
import pandas
import seaborn

logger = logging.getLogger(__name__)  # setup_logger()
import shutil


class QueryEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
        s_threds=[0.05, 0.2, 0.5, 0.7],
        vis=False,
        hist_only=False,
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        # {image name: torch concatenated [boxes, scores, reid features]
        inf_rts = torch.load(
            os.path.join(self._output_dir, "_gallery_gt_inf.pt" if "GT" not in self.dataset_name else "_gallery_gt_infgt.pt"),
            map_location=self._cpu_device,
        )
        self.gts = inf_rts["gts"]
        self.infs = inf_rts["infs"]
        self.gtfs = inf_rts["gt_fnames"]
        self.topks = [1, 5, 10]
        # statistics
        self.det_score_thresh = s_threds
        self.ignore_cam_id = False
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}
        # make vis dirs
        self.vis = vis
        if not vis:
            self._vis_search = _trivial_vis
            return
        vis_dir = opj(self._output_dir, "vis", "search")
        self.svis_dirs = {}
        lrk = comm.get_local_rank()
        for scr in s_threds:
            svis_dir = opj(vis_dir, str(scr))
            if lrk == 0:
                if not os.path.exists(svis_dir):
                    os.makedirs(svis_dir)
            self.svis_dirs[scr] = svis_dir
        comm.synchronize()
        self.hist_only = hist_only
        self.score_statistics = {st: {} for st in self.det_score_thresh}

    def reset(self):
        self.aps = {st: [] for st in self.det_score_thresh}
        self.accs = {st: [] for st in self.det_score_thresh}

    def _get_gallery_dets(self, img_id, score_thred):
        img_dets_all = self.infs[img_id][:, :5]
        return img_dets_all[img_dets_all[:, 4] >= score_thred]

    def _get_gallery_feats(self, img_id, score_thred):
        img_save_all = self.infs[img_id]
        return img_save_all[img_save_all[:, 4] >= score_thred][:, 5:]

    def _get_gt_boxs(self, img_id):
        return self.gts[img_id][:, :4]

    def _get_gt_ids(self, img_id):
        return self.gts[img_id][:, 4].long()

    def process(self, inputs, outputs):
        raise NotImplementedError

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            aps_all = comm.gather(self.aps, dst=0)
            accs_all = comm.gather(self.accs, dst=0)
            if self.vis:
                scores_all = comm.gather(self.score_statistics)
            if not comm.is_main_process():
                return {}
            aps = {}
            accs = {}
            score_statistics = {}
            for dst in self.det_score_thresh:
                aps[dst] = list(itertools.chain(*[ap[dst] for ap in aps_all]))
                accs[dst] = list(itertools.chain(*[acc[dst] for acc in accs_all]))
                if self.vis:
                    score_statistics[dst] = {"pos": [], "neg": []}
                    for sub_stat in scores_all:
                        score_statistics[dst]["pos"].extend(
                            list(
                                itertools.chain(
                                    *[
                                        sub_stat[dst][img_id]["pos"]
                                        for img_id in sub_stat[dst].keys()
                                    ]
                                )
                            )
                        )
                        score_statistics[dst]["neg"].extend(
                            list(
                                itertools.chain(
                                    *[
                                        sub_stat[dst][img_id]["neg"]
                                        for img_id in sub_stat[dst].keys()
                                    ]
                                )
                            )
                        )
        else:
            aps = self.aps
            accs = self.accs
            score_statistics = {}
            if self.vis:
                for dst in self.det_score_thresh:
                    score_statistics[dst] = {"pos": [], "neg": []}
                    score_statistics[dst]["pos"].extend(
                        list(
                            itertools.chain(
                                self.score_statistics[dst][img_id]["pos"]
                                for img_id in self.score_statistics[dst].keys()
                            )
                        )
                    )
                    score_statistics[dst]["neg"].extend(
                        list(
                            itertools.chain(
                                self.score_statistics[dst][img_id]["neg"]
                                for img_id in self.score_statistics[dst].keys()
                            )
                        )
                    )
                score_statistics = self.score_statistics
        search_result = OrderedDict()
        search_result["search"] = {}

        for dst in self.det_score_thresh:
            logger.info("Search eval_{:.2f} on {} queries. ".format(dst, len(aps[dst])))
            mAP = np.mean(aps[dst])
            search_result["search"].update({"mAP_{:.2f}".format(dst): mAP})
            acc = np.mean(np.array(accs[dst]), axis=0)
            # logger.info(str(acc))
            for i, v in enumerate(acc.tolist()):
                # logger.info("{:.2f} on {} acc. ".format(v, i))
                k = self.topks[i]
                search_result["search"].update({"top{:2d}_{:.2f}".format(k, dst): v})
        if self.vis:
            for st, scores in score_statistics.items():
                pos_scores = scores["pos"]
                neg_scores = random.sample(scores["neg"], len(pos_scores))
                vis_data = pandas.DataFrame(
                    {
                        "cos_scores": pos_scores + neg_scores,
                        "type": ["pos"] * len(pos_scores) + ["neg"] * len(neg_scores),
                    },
                )
                plt = seaborn.histplot(
                    data=vis_data, x="cos_scores", hue="type", binwidth=0.1
                )
                fig = plt.get_figure()
                fig.savefig(opj(self.svis_dirs[st], "cos_hist.png"), dpi=400)

        return copy.deepcopy(search_result)

    def _get_query_gallery_gts(self, query_dict):
        raise NotImplementedError

    def _vis_search(
        self,
        qimg_id,
        q_box,
        q_id,
        g_img_ids,
        qg_scores_dict,
        sthred,
        qgt_g_img_ids=None,
        vis_neg_nums=64,
    ):
        COLORS = ["r", "g", "b", "y", "c", "m"]
        T_COLORS_BG = {
            "r": "white",
            "g": "white",
            "b": "white",
            "y": "black",
            "c": "black",
            "m": "white",
        }
        id_dir = opj(self.svis_dirs[sthred], str(q_id))
        pos_dir = opj(id_dir, qimg_id[:-4], "pos")
        neg_dir = opj(id_dir, qimg_id[:-4], "neg")
        if not os.path.exists(pos_dir):
            os.makedirs(pos_dir)
        if not os.path.exists(neg_dir):
            os.makedirs(neg_dir)
        # org query
        q_path = self.gtfs[qimg_id]
        qimg = Image.open(q_path)
        vis_org = Visualizer(qimg)
        vis_org.draw_box(q_box.numpy(), edge_color="g")
        id_pos = q_box[:2]
        vis_org.draw_text(
            str(q_id), id_pos, horizontal_alignment="left", color="w", bg_color="g"
        )
        vis_org.get_output().save(opj(id_dir, qimg_id[:-4], qimg_id))
        if qgt_g_img_ids is not None:
            neg_g_img_ids = list(set(g_img_ids) - set(qgt_g_img_ids))
        else:
            qgt_g_img_ids = []
            neg_g_img_ids = []
            for gid in g_img_ids:
                tgt_ids = self._get_gt_ids(gid).tolist()
                if q_id in tgt_ids:
                    qgt_g_img_ids.append(gid)
                else:
                    neg_g_img_ids.append(gid)
        self.score_statistics[sthred][qimg_id] = {"pos": [], "neg": []}
        # vis gts
        # logger.info("Visualizing positives for pid {}".format(q_id))
        # with tqdm(total=len(qgt_g_img_ids)) as pbar:
        min_match_score = 1.0
        for gtgid in qgt_g_img_ids:
            g_img = Image.open(self.gtfs[gtgid])
            vis_tgt = Visualizer(g_img.copy())
            vis_det = Visualizer(g_img.copy())
            tgt_ids: List = self._get_gt_ids(gtgid).tolist()
            tgt_boxes = self._get_gt_boxs(gtgid).numpy()
            # det
            det_rts = self._get_gallery_dets(gtgid, sthred).numpy()
            det_boxes = det_rts[:, :4]
            det_scores = det_rts[:, 4]
            match_scores = qg_scores_dict[gtgid]
            max_match_score, max_match_idx = torch.max(match_scores, dim=0)
            pbox_in_gimg = tgt_boxes[tgt_ids.index(q_id)]
            pgdet_ious = box_iou(
                torch.from_numpy(pbox_in_gimg)[None, :], torch.from_numpy(det_boxes)
            ).squeeze(0)
            max_iou, max_iou_idx = torch.max(pgdet_ious, dim=0)
            if max_iou >= 0.5:
                det_match_score = match_scores[max_iou_idx].item()
                if min_match_score > det_match_score:
                    min_match_score = det_match_score
                self.score_statistics[sthred][qimg_id]["neg"].extend(
                    match_scores[:max_iou_idx].tolist()
                )
                self.score_statistics[sthred][qimg_id]["neg"].extend(
                    match_scores[max_iou_idx + 1 :].tolist()
                )
            else:  # gt not detected
                det_match_score = -1
                self.score_statistics[sthred][qimg_id]["neg"].extend(
                    match_scores.tolist()
                )
            self.score_statistics[sthred][qimg_id]["pos"].append(det_match_score)
            """if (
                    max_match_idx != max_iou_idx or det_match_score < 0.5
                ) and not self.hist_only:"""
            if not self.hist_only:  # vis all pos
                for bi, pbox in enumerate(det_boxes):
                    b_clr = COLORS[bi % len(COLORS)]
                    t_clr = T_COLORS_BG[b_clr]
                    vis_det.draw_box(pbox, edge_color=b_clr)
                    # match score
                    vis_det.draw_text(
                        "%.3f" % match_scores[bi],
                        pbox[:2],
                        horizontal_alignment="right",
                        color=t_clr,
                        bg_color=b_clr,
                    )
                    # det score
                    score_pos = pbox[2:]
                    vis_det.draw_text(
                        "%.3f" % det_scores[bi],
                        score_pos,
                        horizontal_alignment="left",
                        color=t_clr,
                        bg_color=b_clr,
                    )
                # tgt
                for i, pid in enumerate(tgt_ids):
                    pbox = tgt_boxes[i]
                    if pid == q_id:
                        vis_tgt.draw_box(pbox, edge_color="g")
                        id_pos = pbox[:2]
                        vis_tgt.draw_text(
                            str(pid),
                            id_pos,
                            horizontal_alignment="left",
                            color="w",
                            bg_color="g",
                        )
                    else:
                        vis_tgt.draw_box(pbox, edge_color="r")
                        id_pos = pbox[:2]
                        vis_tgt.draw_text(
                            str(pid),
                            id_pos,
                            horizontal_alignment="left",
                            color="w",
                            bg_color="r",
                        )
                img_tgt = vis_tgt.get_output().get_image()
                img_det = vis_det.get_output().get_image()
                img_vis = np.concatenate([img_tgt, img_det], axis=1)
                VisImage(img_vis).save(
                    opj(pos_dir, "{:.4f}".format(det_match_score) + gtgid)
                )
            # pbar.update(1)
        # return  # NOTE only update pos vis for now
        # vis negs
        # logger.info("Visualizing negatives for pid {}".format(q_id))
        vis_negs = math.ceil(vis_neg_nums / comm.get_world_size())

        vis_neg_g_img_ids = {}
        for gid in neg_g_img_ids:
            if gid not in qg_scores_dict:
                continue
            match_scores = qg_scores_dict[gid]
            self.score_statistics[sthred][qimg_id]["neg"].extend(match_scores.tolist())
            neg_match_score_max = match_scores.max().item()
            if neg_match_score_max > min_match_score:
                vis_neg_g_img_ids[gid] = neg_match_score_max
        if self.hist_only:
            return
        vl = min(vis_negs, len(vis_neg_g_img_ids))
        # with tqdm(total=vl) as pbar:
        for gid in random.sample(list(vis_neg_g_img_ids.keys()), vl):
            g_img = Image.open(self.gtfs[gid])
            neg_match_score_max = vis_neg_g_img_ids[gid]
            match_scores = qg_scores_dict[gid]
            vis_tgt = Visualizer(g_img.copy())
            vis_det = Visualizer(g_img.copy())
            # det
            det_rts = self._get_gallery_dets(gid, sthred).numpy()
            det_boxes = det_rts[:, :4]
            det_scores = det_rts[:, 4]
            for bi, pbox in enumerate(det_boxes):
                b_clr = COLORS[bi % len(COLORS)]
                t_clr = T_COLORS_BG[b_clr]
                vis_det.draw_box(pbox, edge_color=b_clr)
                # match score
                vis_det.draw_text(
                    "%.3f" % match_scores[bi],
                    pbox[:2],
                    horizontal_alignment="right",
                    color=t_clr,
                    bg_color=b_clr,
                )
                # det score
                score_pos = pbox[2:]
                vis_det.draw_text(
                    "%.3f" % det_scores[bi],
                    score_pos,
                    horizontal_alignment="left",
                    color=t_clr,
                    bg_color=b_clr,
                )
            # tgt
            tgt_ids = self._get_gt_ids(gid).tolist()
            tgt_boxes = self._get_gt_boxs(gid).numpy()
            for i, pid in enumerate(tgt_ids):
                pbox = tgt_boxes[i]
                vis_tgt.draw_box(pbox, edge_color="r")
                id_pos = pbox[:2]
                vis_tgt.draw_text(
                    str(pid),
                    id_pos,
                    horizontal_alignment="left",
                    color="w",
                    bg_color="r",
                )
            img_tgt = vis_tgt.get_output().get_image()
            img_det = vis_det.get_output().get_image()
            img_vis = np.concatenate([img_tgt, img_det], axis=1)
            VisImage(img_vis).save(
                opj(neg_dir, "{0:.4f}".format(neg_match_score_max) + gid)
            )
            # pbar.update(1)
        if len(os.listdir(pos_dir)) == 0:
            shutil.rmtree(pos_dir)
        if len(os.listdir(neg_dir)) == 0:
            shutil.rmtree(neg_dir)
        if len(os.listdir(opj(id_dir, qimg_id[:-4]))) == 1:
            shutil.rmtree(opj(id_dir, qimg_id[:-4]))
        if len(os.listdir(id_dir)) == 0:
            shutil.rmtree(id_dir)



def _trivial_vis(*args, **kw):
    pass
