import psd2.utils.comm as comm

import torch
import logging

from .evaluator import DatasetEvaluator
import itertools

import copy
from collections import OrderedDict

logger = logging.getLogger(__name__)


# NOTE evaluation in augmented box range
class DomainIdEvaluator(DatasetEvaluator):
    def __init__(
        self,
        domain_names,
        dataset_name,
        distributed,
        output_dir,
    ) -> None:
        self.domain_names = {name: i for i, name in enumerate(domain_names)}
        self._distributed = distributed
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.pred_domain = []
        self.gt_domain = []

    def reset(self):
        self.pred_domain = []
        self.gt_domain = []

    def _gt_domains(self, gt_instances):
        rst = []
        for gti in gt_instances:
            if "PRW" in gti.file_name:
                rst.append(self.domain_names["PRW"])
            elif "cuhk" in gti.file_name:
                rst.append(self.domain_names["CUHK-SYSU"])
            elif "movienet" in gti.file_name:
                rst.append(self.domain_names["MovieNet"])
        return torch.tensor(rst, dtype=torch.long)

    def process(self, inputs, outputs):
        gt_instances = [gti["instances"].to(self._cpu_device) for gti in inputs]
        gt_dids = self._gt_domains(gt_instances)
        self.gt_domain.extend(gt_dids.cpu().tolist())
        pred_dids = torch.argmax(outputs, dim=1).cpu().tolist()
        self.pred_domain.extend(pred_dids)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            all_pred = comm.gather(self.pred_domain, dst=0)
            all_gt = comm.gather(self.gt_domain, dst=0)
            if comm.get_local_rank() != 0:
                comm.synchronize()
                return {}
            else:
                pred_domain = list(itertools.chain(*all_pred))
                gt_domain = list(itertools.chain(*all_gt))
        else:
            pred_domain = self.pred_domain
            gt_domain = self.gt_domain
        pred_domain = torch.tensor(pred_domain)
        gt_domain = torch.tensor(gt_domain)
        num_correct = (pred_domain == gt_domain).sum().item()
        acc = num_correct / len(gt_domain)
        result = OrderedDict()
        result["domain_identify"] = {"Acc": acc}
        return copy.deepcopy(result)
