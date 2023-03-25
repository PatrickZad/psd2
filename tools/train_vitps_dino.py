#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in psd2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use psd2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys
import os

sys.path.append("./")
sys.path.append("./tools")
sys.path.append("./assign_cost_cuda")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import logging
import torch
from psd2.engine import default_argument_parser, launch
import re
from train_ps_net import *
from psd2.engine import HookBase
import math
from psd2.engine import SimpleTrainer
import time
from psd2.utils.events import get_event_storage
from torch.nn.parallel import DistributedDataParallel


class VitPSTrainer(Trainer):
    def __init__(self, cfg, model_find_unused_parameters=True):
        super().__init__(cfg, model_find_unused_parameters)
        self._trainer = SimpleTrainerFreezeLastLayer(
            self._trainer.model,
            self._trainer.data_loader,
            self._trainer.optimizer,
            cfg.PERSON_SEARCH.DINO.FREEZE_LAST_LAYER_ITERS,
        )

    @classmethod
    def build_optimizer(cls, cfg, model):
        from psd2.solver.build import maybe_add_gradient_clipping

        logger = logging.getLogger("psd2.trainer")
        frozen_params = []
        learn_param_keys = []
        zero_wd_param_keys = []
        param_groups = [{"params": [], "lr": cfg.SOLVER.BASE_LR}] + [
            {"params": [], "lr": cfg.SOLVER.BASE_LR * lf}
            for lf in cfg.SOLVER.LR_FACTORS
        ]
        zero_wd_param_groups = [
            {"params": [], "lr": cfg.SOLVER.BASE_LR, "weight_decay": 0}
        ] + [
            {"params": [], "lr": cfg.SOLVER.BASE_LR * lf, "weight_decay": 0}
            for lf in cfg.SOLVER.LR_FACTORS
        ]
        zero_weight_decay_keys = ["pos_embed", "cls_token", "det_token"]
        freeze_regex = [re.compile(reg) for reg in cfg.SOLVER.FREEZE_PARAM_REGEX]
        lr_group_regex = [re.compile(reg) for reg in cfg.SOLVER.LR_GROUP_REGEX]

        def _find_match(pkey, prob_regs):
            match_idx = -1
            for mi, mreg in enumerate(prob_regs):
                if re.match(mreg, pkey):
                    assert match_idx == -1, "Ambiguous matching of {}".format(pkey)
                    match_idx = mi
            return match_idx

        for key, value in model.named_parameters():
            match_freeze = _find_match(key, freeze_regex)
            if match_freeze > -1:
                value.requires_grad = False
            if not value.requires_grad:
                frozen_params.append(key)
                continue
            match_learn = _find_match(key, lr_group_regex)
            is_zero_wd = False
            if "transformer" in key:
                if len(value.shape) == 1 or key.endswith(".bias"):
                    is_zero_wd = True
                else:
                    for k in zero_weight_decay_keys:
                        if k in key:
                            is_zero_wd = True
            if match_learn > 0:
                if is_zero_wd:
                    zero_wd_param_groups[match_learn]["params"].append(value)
                    zero_wd_param_keys.append(key)
                else:
                    param_groups[match_learn]["params"].append(value)
            else:
                if is_zero_wd:
                    zero_wd_param_groups[0]["params"].append(value)
                    zero_wd_param_keys.append(key)
                else:
                    param_groups[0]["params"].append(value)
            learn_param_keys.append(key)
        logger.info("Frozen parameters:\n{}".format("\n".join(frozen_params)))
        logger.info("Training parameters:\n{}".format("\n".join(learn_param_keys)))
        logger.info(
            "Zero Wieght-decay parameters:\n{}".format("\n".join(zero_wd_param_keys))
        )
        param_groups = [pg for pg in param_groups if len(pg["params"]) > 0] + [
            pg for pg in zero_wd_param_groups if len(pg["params"]) > 0
        ]
        optim = cfg.SOLVER.OPTIM
        if optim == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optim == "Adam":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optim == "AdamW":
            return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
                param_groups,
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError("Unsupported optimizer {}".format(optim))

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(
            WeightDecayScheduler(
                self.cfg.PERSON_SEARCH.DINO.WEIGHT_DECAY,
                self.cfg.PERSON_SEARCH.DINO.WEIGHT_DECAY_END,
            )
        )
        ret.append(
            MomentumUpdater(
                self.cfg.PERSON_SEARCH.DINO.MOMEMTUM,
                self.cfg.PERSON_SEARCH.DINO.MOMEMTUM_END,
            )
        )
        return ret


class WeightDecayScheduler(HookBase):
    def __init__(self, start=0.04, end=0.4):
        self._wd_start = start
        self._wd_end = end

    def before_step(self):
        cur_iter = self.trainer.iter
        max_iter = self.trainer.max_iter
        wd = self._wd_end + 0.5 * (self._wd_start - self._wd_end) * (
            1 + math.cos(math.pi * cur_iter / max_iter)
        )
        get_event_storage().put_scalar("weight_decay", wd)
        for param_group in self.trainer.optimizer.param_groups:
            if "weight_decay" not in param_group or param_group["weight_decay"] != 0:
                param_group["weight_decay"] = wd


class MomentumUpdater(HookBase):
    def __init__(self, start=0.996, end=1.0):
        self._mm_start = start
        self._mm_end = end

    def after_step(self):
        cur_iter = self.trainer.iter
        max_iter = self.trainer.max_iter
        mm = self._mm_end + 0.5 * (self._mm_start - self._mm_end) * (
            1 + math.cos(math.pi * cur_iter / max_iter)
        )
        get_event_storage().put_scalar("momentum", mm)
        if isinstance(self.trainer.model, DistributedDataParallel):
            self.trainer.model.module.ema_update(mm)
        else:
            self.trainer.model.ema_update(mm)


class SimpleTrainerFreezeLastLayer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, freeze_iters):
        super().__init__(model, data_loader, optimizer)
        self.f_iters = freeze_iters

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        # NOTE freeze last layer
        if self.iter < self.f_iters:
            for n, p in self.model.named_parameters():
                if "head_student.last_layer" in n:
                    p.grad = None

        self.optimizer.step()


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = VitPSTrainer(cfg, model_find_unused_parameters=not args.n_find_unparams)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    if args.auto_relaunch:
        re_launch = True
        while re_launch:
            try:
                launch(
                    main,
                    args.num_gpus,
                    num_machines=args.num_machines,
                    machine_rank=args.machine_rank,
                    dist_url=args.dist_url,
                    args=(args,),
                )
                re_launch = False
            except Exception as e:
                print(e)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
