from .base import SearchBase
from .. import META_ARCH_REGISTRY
from psd2.config import configurable
from psd2.modeling.extend.solider import SwinTransformer
import torch.nn as nn
import torch
import torch.nn.functional as tF
import psd2.utils.comm as comm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


@META_ARCH_REGISTRY.register()
class SwinLinearCDI(SearchBase):
    @configurable
    def __init__(
        self,
        domain_names,
        swin_org,
        swin_org_init_path,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.swin_org = swin_org
        self.swin_org_init_path = swin_org_init_path
        self.id_head = nn.Linear(768 if "small" in  self.swin_org_init_path else 1024, len(domain_names))
        self.domain_names = {name: i for i, name in enumerate(domain_names)}
        self.load_swin()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.swin_org.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin_org.eval()
            self.id_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin_org = SwinTransformer(
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
        ret["swin_org"] = swin_org
        ret["swin_org_init_path"] = cfg.PERSON_SEARCH.QUERY_ENCODER_WEIGHTS
        ret["domain_names"] = cfg.PERSON_SEARCH.DOMAIN_IDENTIFY.DOMAIN_NAMES
        return ret

    def load_swin(self):
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
        res = self.swin_org.load_state_dict(state_dict, strict=False)
        print("parameters of *swin_org* haved been loaded: \n")
        print(res)

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

    def _gt_domains(self, gt_instances):
        rst = []
        for gti in gt_instances:
            if "PRW" in gti.file_name:
                rst.append(self.domain_names["PRW"])
            elif "cuhk" in gti.file_name:
                rst.append(self.domain_names["CUHK-SYSU"])
            elif "movienet" in gti.file_name:
                rst.append(self.domain_names["MovieNet"])
        return torch.tensor(rst, dtype=torch.long, device=self.device)

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        backbone_features = self.backbone(image_list.tensor)
        img_embd = self.task_query(backbone_features)
        logits = self.id_head(img_embd)
        if self.training:
            gt_di = self._gt_domains(gt_instances)
            loss = tF.cross_entropy(logits, gt_di, reduction="mean")
            return {"loss_ce": loss}
        else:
            return tF.softmax(logits, dim=-1)


from psd2.structures import ImageList


@META_ARCH_REGISTRY.register()
class SwinLinearCDI_NPadding(SwinLinearCDI):
    def forward(self, input_list):
        image_list = []
        for in_dict in input_list:
            image = in_dict["image"].clone().to(self.device)
            image = (image - self.pixel_mean) / self.pixel_std
            img_list = ImageList.from_tensors([image], self.backbone.size_divisibility)
            image_list.append(img_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        img_embeds = []
        for imgl in image_list:
            backbone_features = self.backbone(imgl.tensor)
            img_embd = self.task_query(backbone_features)
            img_embeds.append(img_embd)
        img_embd = torch.cat(img_embeds, dim=0)
        logits = self.id_head(img_embd)
        if self.training:
            gt_di = self._gt_domains(gt_instances)
            loss = tF.cross_entropy(logits, gt_di, reduction="mean")
            return {"loss_ce": loss}
        else:
            return tF.softmax(logits, dim=-1)


@META_ARCH_REGISTRY.register()
class SwinFFNCDI(SwinLinearCDI):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        del self.id_head
        edim=768 if "small" in  self.swin_org_init_path else 1024
        self.id_head = nn.Sequential(
            Mlp(edim, edim * 2, edim), nn.Linear(edim, len(self.domain_names))
        )


@META_ARCH_REGISTRY.register()
class SwinLinearCDIOpen(SwinLinearCDI):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        for p in self.backbone.parameters():
            p.requires_grad = True
        for p in self.swin_org.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.train()
            self.swin_org.train()
            self.id_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)


@META_ARCH_REGISTRY.register()
class SwinKnnDI(SwinLinearCDI):
    @configurable
    def __init__(
        self,
        domain_feat_path,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        domain_clusters = []
        domain_ids = []
        for di, df in enumerate(domain_feat_path):
            domain_feats = torch.load(df, map_location=self.device)
            domain_feats = tF.normalize(domain_feats, dim=-1).cpu().numpy()
            clusters = (
                KMeans(n_clusters=5, random_state=0).fit(domain_feats).cluster_centers_
            )
            domain_clusters.append(clusters)
            domain_ids.append(np.zeros(5, dtype=np.int64) + di)
        X = np.concatenate(domain_clusters, 0)
        Y = np.concatenate(domain_ids, 0)
        self.neigh = KNeighborsClassifier(n_neighbors=1, metric="l1")
        self.neigh.fit(X, Y)

    @classmethod
    def from_config(cls, cfg):
        assert comm.get_world_size() == 1
        ret = super().from_config(cfg)
        ret["domain_feat_path"] = cfg.PERSON_SEARCH.QUERY_IMG_FEATS
        return ret

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        backbone_features = self.backbone(image_list.tensor)
        img_embd = self.task_query(backbone_features)
        img_embd = tF.normalize(img_embd, dim=-1)
        if self.training:
            raise NotImplementedError
        else:
            rst = []
            selection = self.neigh.predict(img_embd.detach().cpu().numpy())
            for di in selection:
                logits = torch.zeros(len(self.domain_names), device=self.device)
                logits[di] = 1
                rst.append(logits)
            return torch.stack(rst)


@META_ARCH_REGISTRY.register()
class SwinMsKnnDI(SwinKnnDI):
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
        outs = []
        for i, stage in enumerate(self.swin_org.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin_org.semantic_weight >= 0:
                sw = self.swin_org.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin_org.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin_org.softplus(sw) + sb
            out = getattr(self.swin_org, "norm{}".format(i))(out)
            out = (
                out.view(-1, *out_hw_shape, self.swin_org.num_features[i])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.swin_org.avgpool(out)
            out = torch.flatten(out, 1)
            outs.append(out)
        return torch.cat(outs, dim=-1)


@META_ARCH_REGISTRY.register()
class SwinExtracter(SearchBase):
    @configurable
    def __init__(
        self,
        swin_org,
        swin_org_init_path,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.swin_org = swin_org
        self.swin_org_init_path = swin_org_init_path
        self.load_swin()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.swin_org.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        self.training = mode
        if mode:
            # training:
            self.backbone.eval()
            self.swin_org.eval()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        patch_embed = ret["backbone"]
        tr_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        swin_org = SwinTransformer(
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
        ret["swin_org"] = swin_org
        ret["swin_org_init_path"] = cfg.PERSON_SEARCH.QUERY_ENCODER_WEIGHTS
        return ret

    def load_swin(self):
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
        res = self.swin_org.load_state_dict(state_dict, strict=False)
        print("parameters of *swin_org* haved been loaded: \n")
        print(res)

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

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        backbone_features = self.backbone(image_list.tensor)
        img_feats = self.task_query(backbone_features)
        return img_feats


@META_ARCH_REGISTRY.register()
class SwinMsExtracter(SwinExtracter):
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
        outs = []
        for i, stage in enumerate(self.swin_org.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin_org.semantic_weight >= 0:
                sw = self.swin_org.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin_org.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin_org.softplus(sw) + sb
            out = getattr(self.swin_org, "norm{}".format(i))(out)
            out = (
                out.view(-1, *out_hw_shape, self.swin_org.num_features[i])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.swin_org.avgpool(out)
            out = torch.flatten(out, 1)
            outs.append(out)
        return torch.cat(outs, dim=-1)


@META_ARCH_REGISTRY.register()
class SwinMsLinearCDI(SwinLinearCDI):
    @configurable
    def __init__(
        self,
        domain_names,
        swin_org,
        swin_org_init_path,
        *args,
        **kws,
    ):
        super(SwinLinearCDI, self).__init__(*args, **kws)
        self.swin_org = swin_org
        self.swin_org_init_path = swin_org_init_path
        self.id_head = nn.Linear(1440, len(domain_names))
        self.domain_names = {name: i for i, name in enumerate(domain_names)}
        self.load_swin()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.swin_org.parameters():
            p.requires_grad = False

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
        outs = []
        for i, stage in enumerate(self.swin_org.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin_org.semantic_weight >= 0:
                sw = self.swin_org.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin_org.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin_org.softplus(sw) + sb
            out = getattr(self.swin_org, "norm{}".format(i))(out)
            out = (
                out.view(-1, *out_hw_shape, self.swin_org.num_features[i])
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.swin_org.avgpool(out)
            out = torch.flatten(out, 1)
            outs.append(out)
        return torch.cat(outs, dim=-1)


@META_ARCH_REGISTRY.register()
class SwinMsFFNCDI(SwinMsLinearCDI):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        del self.id_head
        self.id_head = nn.Sequential(
            Mlp(1440, 1440 * 2, 1440), nn.Linear(1440, len(self.domain_names))
        )


@META_ARCH_REGISTRY.register()
class SwinCosCDI(SwinLinearCDI):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.topk = 4  # voting
        self.domain_keys = nn.Parameter(
            torch.FloatTensor(self.topk * len(self.domain_names), 768 if "small" in  self.swin_org_init_path else 1024)
        )
        nn.init.uniform_(self.domain_keys)

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        backbone_features = self.backbone(image_list.tensor)
        img_embd = self.task_query(backbone_features)
        cos_sim = torch.einsum(
            "bj,kj->bk",
            tF.normalize(img_embd, dim=-1),
            tF.normalize(self.domain_keys, dim=-1),
        )
        if self.training:
            gt_di = self._gt_domains(gt_instances)
            cos_dist = 1 - cos_sim
            pos_mask = torch.zeros_like(cos_dist)
            for i, di in enumerate(gt_di):
                pos_mask[i, di * self.topk : (di + 1) * self.topk] = 1
            pos_dist = cos_dist[pos_mask.bool()]
            loss = pos_dist.sum() / len(gt_di)
            return {"loss_cos": loss}
        else:
            domain_ids = torch.cat(
                [
                    torch.zeros(self.topk, device=self.device) + i
                    for i in range(len(self.domain_names))
                ]
            ).long()
            rst = []
            for cos_sim_i in cos_sim:
                topk_idx = torch.topk(cos_sim_i, k=self.topk)[1]
                di, ndi = torch.unique(domain_ids[topk_idx], return_counts=True)
                logits = torch.zeros(len(self.domain_names), device=self.device)
                logits[di[torch.argmax(ndi)]] = 1
                rst.append(logits)
            return torch.stack(rst)


@META_ARCH_REGISTRY.register()
class SwinMsCosCDI(SwinMsLinearCDI):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.topk = 6  # voting
        self.domain_keys = nn.Parameter(
            torch.FloatTensor(self.topk * len(self.domain_names), 1440)
        )
        nn.init.uniform_(self.domain_keys)

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        backbone_features = self.backbone(image_list.tensor)
        img_embd = self.task_query(backbone_features)
        cos_sim = torch.einsum(
            "bj,kj->bk",
            tF.normalize(img_embd, dim=-1),
            tF.normalize(self.domain_keys, dim=-1),
        )
        if self.training:
            gt_di = self._gt_domains(gt_instances)
            cos_dist = 1 - cos_sim
            pos_mask = torch.zeros_like(cos_dist)
            for i, di in enumerate(gt_di):
                pos_mask[i, di * self.topk : (di + 1) * self.topk] = 1
            pos_dist = cos_dist[pos_mask.bool()]
            loss = pos_dist.sum() / len(gt_di)
            return {"loss_cos": loss}
        else:
            domain_ids = torch.cat(
                [
                    torch.zeros(self.topk, device=self.device) + i
                    for i in range(len(self.domain_names))
                ]
            ).long()
            rst = []
            for cos_sim_i in cos_sim:
                topk_idx = torch.topk(cos_sim_i, k=self.topk)[1]
                di, ndi = torch.unique(domain_ids[topk_idx], return_counts=True)
                logits = torch.zeros(len(self.domain_names), device=self.device)
                logits[di[torch.argmax(ndi)]] = 1
                rst.append(logits)
            return torch.stack(rst)


@META_ARCH_REGISTRY.register()
class SwinMsCosDivCDI(SwinMsLinearCDI):
    @configurable
    def __init__(
        self,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.topk = 6  # voting
        self.domain_keys = nn.Parameter(
            torch.FloatTensor(self.topk * len(self.domain_names), 1440)
        )
        nn.init.uniform_(self.domain_keys)

    def diverse_loss(self, di):
        k_n = tF.normalize(self.domain_keys[di * self.topk : (di + 1) * self.topk])
        pair_wise_sim = torch.einsum("kj,bj->kb", k_n, k_n)
        mask = torch.logical_not(
            torch.eye(self.topk, device=self.domain_keys.device, dtype=torch.bool)
        )
        div_loss = pair_wise_sim[mask].sum() / ((self.topk) * (self.topk - 1))
        return div_loss

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        backbone_features = self.backbone(image_list.tensor)
        img_embd = self.task_query(backbone_features)
        cos_sim = torch.einsum(
            "bj,kj->bk",
            tF.normalize(img_embd, dim=-1),
            tF.normalize(self.domain_keys, dim=-1),
        )
        if self.training:
            gt_di = self._gt_domains(gt_instances)
            cos_dist = 1 - cos_sim
            pos_mask = torch.zeros_like(cos_dist)
            for i, di in enumerate(gt_di):
                pos_mask[i, di * self.topk : (di + 1) * self.topk] = 1
            pos_dist = cos_dist[pos_mask.bool()]
            loss = pos_dist.sum() / len(gt_di)
            return {"loss_cos": loss, "loss_div": self.diverse_loss(gt_di[0])}
        else:
            domain_ids = torch.cat(
                [
                    torch.zeros(self.topk, device=self.device) + i
                    for i in range(len(self.domain_names))
                ]
            ).long()
            rst = []
            for cos_sim_i in cos_sim:
                topk_idx = torch.topk(cos_sim_i, k=self.topk)[1]
                di, ndi = torch.unique(domain_ids[topk_idx], return_counts=True)
                logits = torch.zeros(len(self.domain_names), device=self.device)
                logits[di[torch.argmax(ndi)]] = 1
                rst.append(logits)
            return torch.stack(rst)


@META_ARCH_REGISTRY.register()
class SwinContraCDI(SwinCosCDI):
    @configurable
    def __init__(
        self,
        domain_feat_path,
        *args,
        **kws,
    ):
        super().__init__(*args, **kws)
        self.mem_len = 512
        self.mem_batch = 16
        self.temp = 0.07
        mem_samples = []
        for df in domain_feat_path:
            if len(df) > 0:
                domain_feats = torch.load(df, map_location=self.device)
                selected = torch.randperm(
                    domain_feats.shape[0], device=domain_feats.device
                )[: self.mem_len]
                mem_samples.append(domain_feats[selected])
            else:
                mem_samples.append(torch.zeros((512, 768 if "small" in  self.swin_org_init_path else 1024), device=self.device))
        self.register_buffer("memory", torch.stack(mem_samples))

    @classmethod
    def from_config(cls, cfg):
        assert comm.get_world_size() == 1
        ret = super().from_config(cfg)
        ret["domain_feat_path"] = cfg.PERSON_SEARCH.QUERY_IMG_FEATS
        return ret

    def sample_mem(self, task_id):
        sample = torch.randint(
            0, self.mem_len, (self.mem_batch,), device=self.memory.device
        )
        return self.memory[task_id][sample]

    def forward(self, input_list):
        image_list = self.preprocess_input(input_list)
        gt_instances = [gti["instances"].to(self.device) for gti in input_list]
        backbone_features = self.backbone(image_list.tensor)
        img_embd = self.task_query(backbone_features)
        cos_sim = torch.einsum(
            "bj,kj->bk",
            tF.normalize(img_embd, dim=-1),
            tF.normalize(self.domain_keys, dim=-1),
        )
        if self.training:
            gt_di = self._gt_domains(gt_instances)
            cur_di = gt_di[0]
            if cur_di == 0:
                pos_dist = 2 - 2 * cos_sim
                pos_mask = torch.zeros_like(cos_sim)
                for i, di in enumerate(gt_di):
                    pos_mask[i, di * self.topk : (di + 1) * self.topk] = 1
                pos_dist = pos_dist[pos_mask.bool()]
                loss = pos_dist.sum() / len(gt_di)
                return {"loss_cos": loss}
            else:
                losses = {}
                mem_samples = [self.sample_mem(i) for i in range(cur_di + 1)]
                # ref to SupCon, keys as query for contrastive
                # 1. cur domain
                # NOTE no self similarity
                logits_pos = (
                    cos_sim[
                        :,
                        cur_di * self.topk : (cur_di + 1) * self.topk,
                    ].T
                    / self.temp
                )
                q_neg_n = nn.functional.normalize(
                    torch.cat(mem_samples[:cur_di], dim=0), dim=1
                )
                logits_full = torch.cat(
                    [
                        logits_pos,
                        torch.einsum(
                            "kj,bj->kb",
                            tF.normalize(self.domain_keys, dim=-1)[
                                cur_di * self.topk : (cur_di + 1) * self.topk
                            ],
                            q_neg_n,
                        )
                        / self.temp,
                    ],
                    dim=1,
                )
                loss = (
                    torch.log(torch.exp(logits_full).sum(dim=1))
                    - logits_pos.mean(dim=1)
                ).sum()
                losses["loss_contrast_cur"] = loss
                # 2. previous domain
                for t in range(cur_di):
                    tn_K = nn.functional.normalize(
                        self.domain_keys[t * self.topk : (t + 1) * self.topk, :],
                        dim=1,
                    )
                    q_pos_n = nn.functional.normalize(
                        mem_samples[t],
                        dim=1,
                    )
                    q_neg_n = nn.functional.normalize(
                        torch.cat(mem_samples[:t] + mem_samples[t + 1 :], dim=0), dim=1
                    )
                    logits_pos = torch.einsum("kj,bj->kb", tn_K, q_pos_n) / self.temp
                    logits_full = torch.cat(
                        [
                            logits_pos,
                            torch.einsum("kj,bj->kb", tn_K, q_neg_n) / self.temp,
                        ],
                        dim=1,
                    )
                    loss = (
                        torch.log(torch.exp(logits_full).sum(dim=1))
                        - logits_pos.mean(dim=1)
                    ).sum()
                    losses["loss_contrast_mem{}".format(t)] = loss
                return losses
        else:
            domain_ids = torch.cat(
                [
                    torch.zeros(self.topk, device=self.device) + i
                    for i in range(len(self.domain_names))
                ]
            ).long()
            rst = []
            for cos_sim_i in cos_sim:
                topk_idx = torch.topk(cos_sim_i, k=self.topk)[1]
                di, ndi = torch.unique(domain_ids[topk_idx], return_counts=True)
                logits = torch.zeros(len(self.domain_names), device=self.device)
                logits[di[torch.argmax(ndi)]] = 1
                rst.append(logits)
            return torch.stack(rst)


@META_ARCH_REGISTRY.register()
class SwinGContraCDI(SwinLinearCDI):
    pass


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


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
