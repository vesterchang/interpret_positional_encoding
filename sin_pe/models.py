# coding: utf-8

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

from fairseq.models.transformer import (
    TransformerModel, 
    TransformerEncoder, 
    TransformerDecoder, 
)
from .modules import PositionalEmbedding

@register_model("transformer_sin_pe")
class TransformerModel_sin_pe(TransformerModel):

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder_sin_pe(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder_sin_pe(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class TransformerEncoder_sin_pe(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.embed_positions = None if args.no_token_positional_embeddings else \
            PositionalEmbedding(
            args.max_source_positions + 1,
            embed_tokens.embedding_dim,
            self.padding_idx,
            learned=args.encoder_learned_pos,
        )


class TransformerDecoder_sin_pe(TransformerDecoder):

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

        self.embed_positions = None if args.no_token_positional_embeddings else \
            PositionalEmbedding(
            args.max_target_positions + 1,
            embed_tokens.embedding_dim,
            self.padding_idx,
            learned=args.decoder_learned_pos,
        )


@register_model_architecture("transformer_sin_pe", "transformer_sin_pe_wmt_en_de")
def transformer_sin_pe_wmt_en_de(args):
    from fairseq.models.transformer import transformer_wmt_en_de
    transformer_wmt_en_de(args)

@register_model_architecture("transformer_sin_pe", "transformer_sin_pe_wmt_en_de_big")
def transformer_sin_pe_wmt_en_de(args):
    from fairseq.models.transformer import transformer_wmt_en_de_big
    transformer_wmt_en_de_big(args)

@register_model_architecture("transformer_sin_pe", "transformer_sin_pe_wmt_en_fr")
def transformer_sin_pe_wmt_en_fr(args):
    from fairseq.models.transformer import transformer_wmt_en_de
    transformer_wmt_en_de(args)

@register_model_architecture("transformer_sin_pe", "transformer_sin_pe_wmt_en_fr_big")
def transformer_sin_pe_wmt_en_fr(args):
    from fairseq.models.transformer import transformer_vaswani_wmt_en_fr_big
    transformer_vaswani_wmt_en_fr_big(args)


