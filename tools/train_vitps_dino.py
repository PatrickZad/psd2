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

sys.path.append("./")
sys.path.append("./tools")
sys.path.append("./assign_cost_cuda")
import logging
import torch
from psd2.engine import default_argument_parser, launch
import re
from train_ps_net import *
from psd2.engine import HookBase


class VitPSTrainer(Trainer):
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

        for key, value in model.named_parameters(recurse=True):
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


class WeightDecayScheduler(HookBase):
    pass


class MomentumScheduler(HookBase):
    pass


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
