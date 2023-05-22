# coding: utf-8
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from fairseq.models import (
    register_model, 
    register_model_architecture, 
)
from fairseq.models.transformer import (
    TransformerModel, 
    TransformerEncoder, 
    TransformerDecoder, 
)

from fairseq.models.transformer import base_architecture
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules import LayerDropModuleList

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

from .modules import Posnet

@register_model("transformer_posnet")
class TransformerModel_Posnet(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        args.positional_kernel_dim = getattr(args, "positional_kernel_dim", 128)
        args.positional_kernel_dropout = getattr(args, "positional_kernel_dropout", 0.1)


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)

        parser.add_argument('--positional-kernel-dim', type=int, metavar='N', default=128,
                            help='dimension of positional kernels')
        parser.add_argument('--positional-kernel-dropout', type=float, metavar='D', default=0, 
                            help='dropout rate of positional kernels')

        
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        src_padding_idx, tgt_padding_idx = src_dict.pad(), tgt_dict.pad()
        assert src_padding_idx == tgt_padding_idx, \
            "src_padding_idx don't match tgt_padding_idx"

        hidden_dim = args.encoder_embed_dim
        head_dim = args.positional_kernel_dim
        posnet = Posnet(
            hidden_dim=hidden_dim, 
            head_dim=head_dim, 
            max_positions=512, 
            padding_idx=src_padding_idx, 
            sys_args=args, 
        )
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, posnet)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, posnet)

        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, posnet):
        return TransformerEncoder_Posnet(args, src_dict, embed_tokens, posnet)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, posnet=None):
        return TransformerDecoder_Posnet(
            args,
            tgt_dict,
            embed_tokens,
            posnet,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )        


class TransformerEncoder_Posnet(TransformerEncoder):
    """
    Transformer encoder with concatenated position encoding.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        posnet (~position_encoding.Posnet): concatenated position encoding
    """
    def __init__(self, args, dictionary, embed_tokens, posnet):

        super().__init__(args, dictionary, embed_tokens)
        self.embed_positions = posnet

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.embed_positions is not None:
            if isinstance(self.embed_positions, Posnet):
                x = self.embed_positions(x)
            else:
                x = embed + self.embed_positions(x)

        return x, embed


class TransformerDecoder_Posnet(TransformerDecoder):
    """
    Transformer encoder with concatenated position encoding.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        posnet (~position_encoding.Posnet): concatenated position encoding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    def __init__(self, args, dictionary, embed_tokens, posnet=None, no_encoder_attn=False, output_projection=None):
        super().__init__(
            args, 
            dictionary, 
            embed_tokens, 
            no_encoder_attn=no_encoder_attn, 
            output_projection=output_projection
        )
        self.embed_positions = None if (args.no_token_positional_embeddings or args.no_decoder_pe) else posnet

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        positions = None
        current_step = None
        if self.embed_positions is not None:
            if isinstance(self.embed_positions, Posnet):
                current_step = prev_output_tokens.size(1)
            else:
                positions = self.embed_positions(
                    prev_output_tokens, incremental_state=incremental_state
                )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]  # [b, 1]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # posnet
        if incremental_state is not None:
            if current_step is not None:
                x = self.embed_positions(x, incremental_state=incremental_state, timestep=current_step)  # x [b, 1, d]
        else:
            if self.embed_positions is not None:
                x = self.embed_positions(x)  # [b, n, d]


        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("transformer_posnet", "transformer_posnet_wmt_en_de")
def transformer_posnet_wmt_en_de(args):
    from fairseq.models.transformer import transformer_wmt_en_de
    transformer_wmt_en_de(args)

@register_model_architecture("transformer_posnet", "transformer_posnet_wmt_en_de_big")
def transformer_posnet_wmt_en_de_big(args):
    from fairseq.models.transformer import transformer_wmt_en_de_big
    transformer_wmt_en_de_big(args)

@register_model_architecture("transformer_posnet", "transformer_posnet_wmt_en_fr")
def transformer_posnet_wmt_en_fr(args):
    from fairseq.models.transformer import transformer_wmt_en_de
    transformer_wmt_en_de(args)

@register_model_architecture("transformer_posnet", "transformer_posnet_wmt_en_fr_big")
def transformer_posnet_wmt_en_fr_big(args):
    from fairseq.models.transformer import transformer_vaswani_wmt_en_fr_big
    transformer_vaswani_wmt_en_fr_big(args)


