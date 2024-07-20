import copy
from typing import List, OrderedDict
import torch
import logging
from .evaluator import DatasetEvaluator
import itertools
import os
import logging
import psd2.utils.comm as comm
import itertools

logger = logging.getLogger(__name__)  # setup_logger()



class QueryInferencer(DatasetEvaluator):
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

        # save results
        self.q_feats=[]
        self.qid=[]
        self.qfn=[]
        self.gallery2qidx={}

    def reset(self):
        self.q_feats=[]
        self.qid=[]
        self.qfn=[]
        self.gallery2qidx={}

    def process(self, inputs, outputs):
        """
        Args:
            inputs:
                a batch of
                { "query":
                    {
                        "image": augmented image tensor
                        "instances": an Instances object with attrs
                            {
                                image_size: hw (int, int)
                                file_name: full path string
                                image_id: filename string
                                gt_boxes: Boxes (1 , 4)
                                gt_classes: tensor full with 0s
                                gt_pids: person identity tensor (1,)
                                org_img_size: hw before augmentation (int, int)
                                org_gt_boxes: Boxes before augmentation
                            }
                    }
                    "gallery": (optional)
                    {
                        "instances_list": a gallery of Instances objects
                        [
                            Instances object with attr:
                                file_name,
                                image_id,
                                gt_boxes: (1,4) box of the true positive
                                gt_pids: (1,) query pid
                            ...
                        ]
                    }
                }
            outputs:
                a batch of instances with attrs
                {
                    reid_feats: tensor (1,pfeat_dim)
                }
                or
                a batch of empty instances
        """

        for bi, in_dict in enumerate(inputs):
            query_dict = in_dict["query"]
            q_instances = query_dict["instances"].to(self._cpu_device)
            q_imgid = q_instances.image_id
            q_pid = q_instances.gt_pids
            q_feat=outputs[bi].reid_feats.to(self._cpu_device)
            self.q_feats.append(q_feat)
            self.qid.append(q_pid)
            self.qfn.append(q_imgid)
            self.gallery2qidx
            if "gallery" in in_dict:
                for inst in in_dict["gallery"]:
                    self.gallery2qidx.get(inst.image_id,[]).append(len(self.q_feats)-1)


    def evaluate(self):
        if self._distributed:
            all_q_feats= comm.gather(self.q_feats)
            all_qid= comm.gather(self.qid)
            all_qfn= comm.gather(self.qfn)
            all_g2qidx= comm.gather(self.gallery2qidx)
            if comm.is_main_process():
                raise NotImplementedError
                temp=all_g2qidx[0]
                offset=len(all_q_feats[0])
                for i in all_g2qidx[1:]:
                    pass
                all_q_feats=list(itertools.chain(*all_q_feats))
                all_qid=list(itertools.chain(*all_qid))
                all_qfn=list(itertools.chain(*all_qfn))
            comm.synchronize()
        else:
            q_feats=torch.cat(self.q_feats,dim=0)
            qid=torch.cat(self.qid)
            qfn= self.qfn
            g2qidx= self.gallery2qidx
        
        save_dict = {"qfeats": q_feats, "qids": qid, "qfns": qfn,"g2qidxs":g2qidx}
        save_path = os.path.join(
            self._output_dir,
            "_query_gt_inf.pt",
        )
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        torch.save(save_dict, save_path)
        comm.synchronize()
        # det eval
        if not comm.is_main_process():
            return {}
        infer_result = OrderedDict()
        infer_result["infer"] = {}

        return copy.deepcopy(infer_result)

