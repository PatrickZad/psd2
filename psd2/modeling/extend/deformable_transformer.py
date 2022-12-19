# ------------------------------------------------------------------------
# Based on Deformable DETR
# ------------------------------------------------------------------------
import torch.nn as nn
import copy
import math
from psd2.config.config import configurable
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
from psd2.layers.ms_deform_attn import MSDeformAttn


class DeformableTransformer(nn.Module):
    @configurable()
    def __init__(
        self,
        encoder,
        decoder,
        d_model=256,
        num_feature_levels=4,
        two_stage=False,
        two_stage_num_proposals=300,
    ):
        super().__init__()

        self.d_model = d_model
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        self.encoder = encoder

        self.decoder = decoder

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret.update(cls._init_encoder(cfg))
        ret.update(cls._init_decoder(cfg))
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        ret["d_model"] = dt_cfg.D_MODEL
        ret["num_feature_levels"] = dt_cfg.NUM_FEATURE_LEVELS
        ret["two_stage"] = dt_cfg.TWO_STAGE
        ret["two_stage_num_proposals"] = dt_cfg.TWO_STAGE_NUM_PROPOSALS
        return ret

    @classmethod
    def _init_encoder(cls, cfg):
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        encoder_layer = DeformableTransformerEncoderLayer(
            dt_cfg.D_MODEL,
            dt_cfg.DIM_FEEDFORWARD,
            dt_cfg.DROPOUT,
            dt_cfg.ACTIVATION,
            dt_cfg.NUM_FEATURE_LEVELS,
            dt_cfg.NHEAD,
            dt_cfg.ENC_N_POINTS,
        )
        encoder = DeformableTransformerEncoder(encoder_layer, dt_cfg.NUM_ENCODER_LAYERS)
        return {"encoder": encoder}

    @classmethod
    def _init_decoder(cls, cfg):
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        decoder_layer = DeformableTransformerDecoderLayer(
            dt_cfg.D_MODEL,
            dt_cfg.DIM_FEEDFORWARD,
            dt_cfg.DROPOUT,
            dt_cfg.ACTIVATION,
            dt_cfg.NUM_FEATURE_LEVELS,
            dt_cfg.NHEAD,
            dt_cfg.DEC_N_POINTS,
        )
        decoder = DeformableTransformerDecoder(
            decoder_layer, dt_cfg.NUM_DECODER_LAYERS, dt_cfg.RETURN_INTERMEDIATE_DEC
        )
        return {"decoder": decoder}

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
                N_, H_, W_, 1
            )
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H_ - 1, H_, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W_ - 1, W_, dtype=torch.float32, device=memory.device
                ),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
                N_, 1, 1, 2
            )
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
                output_memory
            )
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
                + output_proposals
            )

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact))
            )
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            lvl_pos_embed_flatten,
        )

        inter_references_out = inter_references
        input_info = (
            mask_flatten,
            lvl_pos_embed_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            memory,
        )
        if self.two_stage:
            return (
                input_info,
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return input_info, hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        src_pos=None,  # to be compatible with fp-detr
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + _inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + _inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# from fp-detr
class MultiHeadSelfAttentionModule(nn.MultiheadAttention):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__(d_model, n_heads, dropout=dropout)

    def forward(self, tgt, query_pos):
        if query_pos is not None:
            q = k = tgt + query_pos
        else:
            q = k = tgt
        tgt2 = (
            super()
            .forward(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0]
            .transpose(0, 1)
        )
        return tgt + tgt2


class DeformableTransformerQueryEncoderLayer(DeformableTransformerEncoderLayer):
    def __init__(self, mask_obj_queries, *args, **kws):
        super().__init__(*args, **kws)
        self.mask_obj_queries = mask_obj_queries
        self.norm0 = nn.LayerNorm(kws["d_model"])

    def forward(
        self,
        query_sa,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        obj_queries_start_idx = level_start_index[-1]
        obj_queries = src[:, obj_queries_start_idx:]
        query_pos = pos[:, obj_queries_start_idx:]
        obj_queries = query_sa(obj_queries, query_pos)
        src = torch.cat([src[:, :obj_queries_start_idx], obj_queries], dim=1)
        src = self.norm0(src)
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
            mask_last_lvl=self.mask_obj_queries,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerQueryEncoder(DeformableTransformerDecoder):
    def __init__(
        self,
        decoder_layer,
        query_sa_layer,
        query_saptial_shape,
        num_layers,
        return_intermediate=False,
    ):
        super().__init__(decoder_layer, num_layers, return_intermediate)
        self.query_sa = query_sa_layer
        self.query_saptial_shape = query_saptial_shape

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        src_pos=None,
    ):
        # image tokens
        src_reference_points = DeformableTransformerEncoder.get_reference_points(
            src_spatial_shapes, src_valid_ratios, src.device
        )
        # concat tokens
        all_tgt = torch.cat([src, tgt], dim=1)
        all_pos = torch.cat([src_pos, query_pos], dim=1)
        query_spatial_shapes = src_spatial_shapes.new_tensor(
            self.query_saptial_shape
        ).view(1, 2)
        all_spatial_shapes = torch.cat(
            [src_spatial_shapes, query_spatial_shapes], dim=0
        )
        all_level_start_index = torch.cat(
            (
                all_spatial_shapes.new_zeros((1,)),
                all_spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        mask_query = src_padding_mask.new_zeros((tgt.shape[0], tgt.shape[1]))
        all_padding_mask = torch.cat([src_padding_mask, mask_query], dim=1)

        output = all_tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            all_ref_pts = torch.cat(
                [src_reference_points, reference_points_input], dim=1
            )
            output = layer(
                self.query_sa,
                output,
                all_pos,
                all_ref_pts,
                all_spatial_shapes,
                all_level_start_index,
                all_padding_mask,
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                obj_output = output[:, all_level_start_index:]
                tmp = self.bbox_embed[lid](obj_output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + _inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + _inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DeformableQueryTransformer(DeformableTransformer):
    @classmethod
    def _init_decoder(cls, cfg):
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        qsa_layer = MultiHeadSelfAttentionModule(
            dt_cfg.D_MODEL, dt_cfg.NHEAD, dt_cfg.DROPOUT
        )
        qenc_layer = DeformableTransformerQueryEncoderLayer(
            dt_cfg.MASK_OBJ_QUERIES,
            d_model=dt_cfg.D_MODEL,
            d_ffn=dt_cfg.DIM_FEEDFORWARD,
            dropout=dt_cfg.DROPOUT,
            activation=dt_cfg.ACTIVATION,
            n_levels=dt_cfg.NUM_FEATURE_LEVELS + 1,
            n_heads=dt_cfg.NHEAD,
            n_points=dt_cfg.QENC_N_POINTS,
        )
        q_encoder = DeformableTransformerQueryEncoder(
            qenc_layer,
            qsa_layer,
            dt_cfg.QUERY_SPATIAL_SHAPE,
            dt_cfg.NUM_QENCODER_LAYERS,
            dt_cfg.RETURN_INTERMEDIATE_QENC,
        )

        return {"decoder": q_encoder}
