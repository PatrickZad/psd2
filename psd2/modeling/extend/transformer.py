# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from DETR with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from psd2.config.config import configurable


class Transformer(nn.Module):
    @configurable()
    def __init__(
        self,
        encoder,
        decoder,
        d_model,
    ):
        super().__init__()
        self._reset_parameters()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret.update(cls._init_encoder(cfg))
        ret.update(cls._init_decoder(cfg))
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        ret["d_model"] = dt_cfg.D_MODEL
        return ret

    @classmethod
    def _init_encoder(cls, cfg):
        # NOTE assume normalize_before is False
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        encoder_layer = TransformerEncoderLayer(
            dt_cfg.D_MODEL,
            dt_cfg.NHEAD,
            dt_cfg.DIM_FEEDFORWARD,
            dt_cfg.DROPOUT,
            dt_cfg.ACTIVATION,
            False,
        )
        encoder = TransformerEncoder(encoder_layer, dt_cfg.NUM_ENCODER_LAYERS, None)
        return {"encoder": encoder}

    @classmethod
    def _init_decoder(cls, cfg):
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        decoder_layer = TransformerDecoderLayer(
            dt_cfg.D_MODEL,
            dt_cfg.NHEAD,
            dt_cfg.DIM_FEEDFORWARD,
            dt_cfg.DROPOUT,
            dt_cfg.ACTIVATION,
            False,
        )
        decoder = TransformerDecoder(
            decoder_layer,
            dt_cfg.NUM_DECODER_LAYERS,
            nn.LayerNorm(dt_cfg.D_MODEL),
            return_intermediate=dt_cfg.RETURN_INTERMEDIATE_DEC,
        )
        return {"decoder": decoder}

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to NxHWxC
        bs, c, h, w = src.shape
        src = src.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        query_embed = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        input_info = (src, mask, pos_embed, memory)
        return input_info, hs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            value=src.transpose(0, 1),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            value=src2.transpose(0, 1),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            value=tgt.transpose(0, 1),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
            key=self.with_pos_embed(memory, pos).transpose(0, 1),
            value=memory.transpose(0, 1),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            value=tgt2.transpose(0, 1),
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos).transpose(0, 1),
            key=self.with_pos_embed(memory, pos).transpose(0, 1),
            value=memory.transpose(0, 1),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


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


class MultiHeadSelfAttentionModule(nn.MultiheadAttention):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__(d_model, n_heads, dropout=dropout)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, pre_norm=False):
        if pre_norm:
            tgt = self.norm0(tgt)
        if query_pos is not None:
            q = k = tgt + query_pos
        else:
            q = k = tgt
        tgt2 = (
            super()
            .forward(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0]
            .transpose(0, 1)
        )
        tgt = tgt + self.dropout0(tgt2)
        if not pre_norm:
            tgt = self.norm0(tgt)
        return tgt


class TransformerQueryEncoderLayer(TransformerEncoderLayer):
    def forward_post(
        self,
        query_sa,
        src,
        src_mask,
        src_key_padding_mask,
        pos,
        len_query,
    ):
        obj_queries = src[:, -len_query:]
        query_pos = pos[:, -len_query:]
        obj_queries = query_sa(obj_queries, query_pos, pre_norm=False)
        src = torch.cat([src[:, :-len_query], obj_queries], dim=1)
        # self attention
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            value=src.transpose(0, 1),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(
        self,
        query_sa,
        src,
        src_mask,
        src_key_padding_mask,
        pos,
        len_query,
    ):
        obj_queries = src[:, -len_query:]
        query_pos = pos[:, -len_query:]
        obj_queries = query_sa(obj_queries, query_pos, pre_norm=True)
        src = torch.cat([src[:, :-len_query], obj_queries], dim=1)
        # self attention
        src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            value=src.transpose(0, 1),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0].transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm2(src)

        # ffn
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src


class TransformerQueryEncoder(TransformerDecoder):
    def __init__(
        self,
        decoder_layer,
        query_sa_layer,
        mask_obj_queries,
        num_layers,
        return_intermediate=False,
    ):
        super().__init__(decoder_layer, num_layers, None, return_intermediate)
        self.query_sa = query_sa_layer
        self.mask_obj_queries = mask_obj_queries

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        len_query = tgt.shape[1]
        if memory_mask is not None:
            if memory_mask.dim() == 3:
                src_mask = memory_mask.new_zeros(
                    memory_mask.shape[0],
                    memory_mask.shape[1] + len_query,
                    memory_mask.shape[2] + len_query,
                )
            else:
                src_mask = memory_mask.new_zeros(
                    memory_mask.shape[0],
                    memory_mask.shape[1] + len_query,
                )
            if self.mask_obj_queries:
                src_mask[..., -len_query:] = 1.0
            src_mask = src_mask.to(memory_mask.dtype)
        else:
            src_mask = None
        if memory_key_padding_mask is not None:
            src_key_padding_mask = memory_key_padding_mask.new_zeros(
                memory_key_padding_mask.shape[0],
                memory_key_padding_mask.shape[1] + len_query,
            )
            if self.mask_obj_queries:
                src_key_padding_mask[..., -len_query:] = 1.0
            src_key_padding_mask = src_key_padding_mask.to(
                memory_key_padding_mask.dtype
            )
        else:
            src_key_padding_mask = None
        src = torch.cat([memory, tgt], dim=1)
        intermediate = []
        output = src
        for layer in self.layers:
            output = layer(
                self.query_sa,
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=torch.cat([pos, query_pos], dim=1),
                len_query=len_query,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class QueryTransformer(Transformer):
    @classmethod
    def _init_decoder(cls, cfg):
        dt_cfg = cfg.PERSON_SEARCH.DET.MODEL.TRANSFORMER
        qsa_layer = MultiHeadSelfAttentionModule(
            dt_cfg.D_MODEL, dt_cfg.NHEAD, dt_cfg.DROPOUT
        )
        qenc_layer = TransformerQueryEncoderLayer(
            d_model=dt_cfg.D_MODEL,
            dim_feedforward=dt_cfg.DIM_FEEDFORWARD,
            dropout=dt_cfg.DROPOUT,
            activation=dt_cfg.ACTIVATION,
            nhead=dt_cfg.NHEAD,
            normalize_before=False,
        )
        q_encoder = TransformerQueryEncoder(
            qenc_layer,
            qsa_layer,
            dt_cfg.MASK_OBJ_QUERIES,
            dt_cfg.NUM_QENCODER_LAYERS,
            dt_cfg.RETURN_INTERMEDIATE_QENC,
        )

        return {"decoder": q_encoder}
