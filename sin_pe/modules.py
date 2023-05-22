# coding: utf-8
import math
from typing import Any, Dict, Optional

import torch
import torch.onnx.operators
from fairseq import utils
from torch import Tensor, nn
import torch.nn.functional as F

from fairseq.modules import SinusoidalPositionalEmbedding


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int = None,
    learned: bool = False,
):
     
    padding_idx = None
    
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # todo: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.

        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPE(num_embeddings, embedding_dim, padding_idx)

        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinPE(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1 if padding_idx is not None else num_embeddings + 0 + 1,
        )
    
    return m


class LearnedPE(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(input.size(1)))
            else:
                bsz, seqlen = input.size()  # [bsz x seqlen] input 是 tokens，不是 embedding
                positions = torch.arange(seqlen, device=input.device, dtype=input.dtype)

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinPE(nn.Module):
    """
    This implement is based on fairseq.modules.SinusoidalPositionalEmbedding to employ right padding. 
    2022-04-30 11:29:43
    """
    def __init__(self, embedding_dim, padding_idx=None, init_size=1024):
        super().__init__()

        self.embedding_dim = embedding_dim
        # self.padding_idx = padding_idx if padding_idx is not None else 0
        self.padding_idx = None
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        # max_pos = self.padding_idx + 1 + seq_len
        max_pos = 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[pos - 1].expand(bsz, 1, -1)  # b x 1 x d

        return self.weights[:seq_len].expand(bsz, -1, -1).detach()  # b x n x d


