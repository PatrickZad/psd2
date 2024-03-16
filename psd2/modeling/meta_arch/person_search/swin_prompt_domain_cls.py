# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from psd2.config import configurable


from .. import META_ARCH_REGISTRY
from psd2.modeling.extend.solider import SwinTransformer


from psd2.modeling.prompts import build_stage_prompt_pool


from .base import SearchBase
from copy import deepcopy
@META_ARCH_REGISTRY.register()
class PromptedSwinDomainCls(SearchBase):
    @configurable
    def __init__(
        self,
        prompt_pool,
        swin_org,
        swin_org_init_path,
        domain_names,
        *args,
        **kws,
    ):
        super().__init__(*args,**kws)
        self.prompt_pool=prompt_pool
        self.swin_org = swin_org
        self.swin_org_init_path = swin_org_init_path
        self.domain_names=domain_names
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.swin_org.parameters():
            p.requires_grad = False

    def load_state_dict(self, *args, **kws):
        out = super().load_state_dict(*args, **kws)
        state_dict = _load_file(self.swin_org_init_path)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        org_ks = list(state_dict.keys())
        for k in org_ks:
            v = state_dict.pop(k)
            if k.startswith("swin."):
                nk = k[len("swin.") :]
                if not isinstance(v, torch.Tensor):
                    state_dict[nk] = torch.tensor(v, device=self.device)
                else:
                    state_dict[nk]=v
        res = self.swin_org.load_state_dict(state_dict, strict=False)
        print("parameters of *swin_org* haved been loaded: \n")
        print(res)
        return out

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin_org.eval()
            self.prompt_pool.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret= super().form_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin = SwinTransformer(
            semantic_weight=tr_cfg.SEMANTIC_WEIGHT,
            pretrain_img_size=patch_embed.pretrain_img_size,
            patch_size=patch_embed.patch_size
            if isinstance(patch_embed.patch_size, int)
            else patch_embed.patch_size[0],
            embed_dims=patch_embed.embed_dim,
            depths=tr_cfg.DEPTH,
            num_heads=tr_cfg.NHEAD,
            window_size=tr_cfg.WIN_SIZE,
            mlp_ratio=tr_cfg.MLP_RATIO,
            qkv_bias=tr_cfg.QKV_BIAS,
            qk_scale=None,
            drop_rate=tr_cfg.DROPOUT,
            attn_drop_rate=tr_cfg.ATTN_DROPOUT,
            drop_path_rate=tr_cfg.DROP_PATH,
        )
        
        prompt_cfg = cfg.PERSON_SEARCH.PROMPT
        num_prompts = prompt_cfg.NUM_PROMPTS
        if isinstance(num_prompts, int):
            num_prompts = [num_prompts] * 4
        in_num_prompts = (
            [n * prompt_cfg.TOP_K for n in num_prompts]
            if "L2P" in prompt_cfg.PROMPT_TYPE and "Attn" not in prompt_cfg.PROMPT_TYPE
            else num_prompts
        )
        
        if isinstance(in_num_prompts, int):
            stage_num_prompts = in_num_prompts
        else:
            stage_num_prompts = in_num_prompts[0]
        prompt_pool=build_stage_prompt_pool(prompt_cfg,stage_num_prompts,swin.num_features[0],1,swin.num_features[-1],cfg.VIS_PERIOD)

        ret.update(
            {
                "swin_org": swin,
                "prompt_pool": prompt_pool,
            }
        )
        ret["swin_org_init_path"] = cfg.PERSON_SEARCH.QUERY_ENCODER_WEIGHTS
        return ret

    @torch.no_grad()
    def task_query(self, backbone_features):
        x = backbone_features[list(backbone_features.keys())[-1]]
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        if self.swin_org.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin_org.drop_after_pos(x)

        if self.swin_org.semantic_weight >= 0:
            w = torch.ones(x.shape[0], 1) * self.swin_org.semantic_weight
            w = torch.cat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()

        for i, stage in enumerate(self.swin_org.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin_org.semantic_weight >= 0:
                sw = self.swin_org.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin_org.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin_org.softplus(sw) + sb
        out = self.swin_org.norm3(out)
        out = (
            out.view(-1, *out_hw_shape, self.swin_org.num_features[i])
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x = self.swin_org.avgpool(out)
        x = torch.flatten(x, 1)
        return x

    def domain_cls(self, x):
        x = self.backbone(x)
        task_query = self.task_query(x)

        x = x[list(x.keys())[-1]]
        x = x.flatten(2).transpose(1, 2)

        task_query_x = task_query.unsqueeze(1)
        domain_ids,_, p_loss = self.prompt_pool(
                    task_query_x, 0, train=self.training,domain_id=True
                )
        if self.training:
            prompt_loss = {"loss_prompt": p_loss}
        return domain_ids, task_query, prompt_loss

    

    def forward_gallery(self, image_list, gt_instances):
        domain_cls_ids, _, prompt_loss = self.domain_cls(image_list.tensor)
        if self.training:
            return prompt_loss
        else:
            pred_instances=[]
            for i,gt_i in enumerate(gt_instances):
                inst=deepcopy(gt_i)
                inst.domain_cls=self.domain_names[domain_cls_ids[i]]
            return pred_instances
    def forward(self, input_list, infgt=False):
        """
        preds:
            a list of
            {
                "pred_boxes": XYXY_ABS Boxes in augmentation range (resizing/cropping/flipping/...) during training
                            XYXY_ABS Boxes in original range for test
                "pred_scores": tensor
                "assign_ids": assigned person identities (during training only)
                "reid_feats": tensor
            }
        """
        image_list = self.preprocess_input(input_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        if self.training:
            losses = self.forward_gallery(
                    image_list, gt_instances
                )
            return losses
        else:
            return self.forward_gallery(image_list, gt_instances)






def _load_file(filename):
    from psd2.utils.file_io import PathManager
    import pickle

    if filename.endswith(".pkl"):
        with PathManager.open(filename, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        if "model" in data and "__author__" in data:
            # file is in Detectron2 model zoo format
            return data
        else:
            # assume file is from Caffe2 / Detectron1 model zoo
            if "blobs" in data:
                # Detection models have "blobs", but ImageNet models don't
                data = data["blobs"]
            data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
            return {
                "model": data,
                "__author__": "Caffe2",
                "matching_heuristics": True,
            }
    elif filename.endswith(".pyth"):
        # assume file is from pycls; no one else seems to use the ".pyth" extension
        with PathManager.open(filename, "rb") as f:
            data = torch.load(f)
        assert (
            "model_state" in data
        ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
        model_state = {
            k: v
            for k, v in data["model_state"].items()
            if not k.endswith("num_batches_tracked")
        }
        return {
            "model": model_state,
            "__author__": "pycls",
            "matching_heuristics": True,
        }

    loaded = torch.load(
        filename, map_location=torch.device("cpu")
    )  # load native pth checkpoint
    if "model" not in loaded:
        loaded = {"model": loaded}
    return loaded
