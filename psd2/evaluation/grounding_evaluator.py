from copy import copy
from typing import List, OrderedDict
import torch
import logging
from .evaluator import DatasetEvaluator
import os
import numpy as np
from torchvision.ops.boxes import box_iou

import copy
import logging
import psd2.utils.comm as comm
from torchvision.ops.boxes import box_iou
from sklearn.metrics import average_precision_score
import time

logger = logging.getLogger(__name__)  # setup_logger()


class GroundingEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed,
        output_dir,
    ) -> None:
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.topks = [1, 5, 10]
        inf_rts = torch.load(
            os.path.join(
                self._output_dir,
                "_query_gt_inf.pt",
            ),
            map_location=self._cpu_device,
        )
        self.q_feats=inf_rts["qfeats"]
        self.qid=inf_rts["qids"]
        self.qfn=inf_rts["qfns"]
        self.gallery2qidx=inf_rts["g2qidxs"]
        # statistics
        self.ignore_cam_id = False
        n_query=self.qid.shape[0]
        self.grounding_trues=[[] for _ in range(n_query)]
        self.grounding_scores=[[] for _ in range(n_query)]
        self.num_gt=[0]*n_query
        self.num_pos=[0]*n_query
        comm.synchronize()

    def reset(self):
        n_query=self.qid.shape[0]
        self.grounding_trues=[[] for _ in range(n_query)]
        self.grounding_scores=[[] for _ in range(n_query)]
        self.num_gt=[0]*n_query
        self.num_pos=[0]*n_query

    def process(self, inputs, model):
        """
        Args:
            inputs:
                a batch of
                {
                    "image": augmented image tensor
                    "instances": an Instances object with attrs
                        {
                            image_size: hw (int, int)
                            file_name: full path string
                            image_id: filename string
                            gt_boxes: Boxes
                            gt_classes: tensor full with 0s
                            gt_pids: person identity tensor in [-1, max_id]
                            org_img_size: hw before augmentation (int, int)
                            org_gt_boxes: Boxes before augmentation
                        }
                }
            model:
                grounding model to run the specialized inference method
        """
        eval_start_time_stamp=time.perf_counter()
        query_idxs=[self.gallery2qidx[gdict[ "instances"].image_id] for gdict in inputs]
        query_feats=[self.q_feats[q_idxs] for q_idxs in query_idxs]
        compute_start_stamp=time.perf_counter()
        outputs=model(inputs,torch.stack(query_feats))
        """
        outputs:
                a batch of instances with attrs
                {
                    pred_boxes: Boxes in augmentation range (resizing/cropping/flipping/...)
                    pred_scores: tensor
                }
        """
        equal_compute_time_stamp=eval_start_time_stamp+ time.perf_counter()-compute_start_stamp

        # process
        for bi in range(len(inputs)):
            out_boxes=outputs[bi].pred_boxes.tensor.to(self._cpu_device)
            out_scores=outputs[bi].pred_scores.to(self._cpu_device)
            gt_boxes=inputs[bi].gt_boxes
            gt_pids=inputs[bi].gt_pids.numpy().tolist()
            for qidx in query_idxs[bi]:
                pred_qbox=out_boxes[qidx]
                pred_qscore=out_scores[qidx]
                q_pid=self.qid[qidx].item()
                if q_pid in gt_pids:
                    self.num_gt[qidx]+=1
                    gt_idx=gt_pids.index(q_pid)
                    gt_box=gt_boxes[gt_idx]
                    if box_iou(gt_box[None],pred_qbox[None])[0,0]>=0.5:
                        self.grounding_trues[qidx].append(1)
                        self.num_pos[qidx]+=1
                    else:
                        self.grounding_trues[qidx].append(0)
                    self.grounding_scores[qidx].append(pred_qscore)
                else:
                    self.grounding_trues[qidx].append(0)
                    self.grounding_scores[qidx].append(pred_qscore)

        return equal_compute_time_stamp



    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            raise NotImplementedError
        else:
            n_query=self.qid.shape[0]
            aps=[]
            accs=[]
            for i in range(n_query):
                recall_rate=self.num_pos[i]/self.num_gt[i]
                y_score = np.asarray(self.grounding_scores[i])
                y_true = np.asarray(self.grounding_trues[i])
                ap=(
                    0
                    if self.num_pos[i] == 0
                    else average_precision_score(y_true, y_score)
                    * recall_rate
                )
                inds = np.argsort(y_score)[::-1]
                y_true = y_true[inds]
                acc=[min(1, sum(y_true[:k]))  for k in self.topks]
                aps.append(ap)
                accs.append(acc)
            aps=np.array(aps)
            accs=np.array(accs)

        search_result = OrderedDict()
        search_result["search"] = {}

        logger.info("Search eval on {} queries. ".format( len(aps)))
        mAP = np.mean(aps)
        search_result["search"].update({"mAP": mAP})
        acc = np.mean(np.array(accs), axis=0)
        for i, v in enumerate(acc.tolist()):
            k = self.topks[i]
            search_result["search"].update({"top{:2d}".format(k): v})

        return copy.deepcopy(search_result)

