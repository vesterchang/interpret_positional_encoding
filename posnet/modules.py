# coding: utf-8
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Posnet(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        head_dim=64, 
        max_positions=512, 
        padding_idx=1, 
        sys_args=None,
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.padding_idx = padding_idx


        self.dim_weights1 = nn.Parameter(torch.randn(self.hidden_dim, self.head_dim))
        self.dim_weights2 = nn.Parameter(torch.randn(self.head_dim, self.hidden_dim))
        self.pos_weights = nn.Parameter(torch.randn(self.max_positions, self.head_dim, self.head_dim))

        nn.init.xavier_normal_(self.dim_weights1)
        nn.init.xavier_normal_(self.dim_weights2)
        nn.init.xavier_normal_(self.pos_weights)

        self.dropout_p = sys_args.positional_kernel_dropout
        self.pe_dropout = nn.Dropout(p=self.dropout_p, inplace=False)

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
    ):        
        """
        Input is expected to be of size [bsz x seq_len x dim], namely the embedding vector of the sequence
        """
        # input [b, n, d]
        
        output = input.matmul(self.dim_weights1)
        if incremental_state is not None:
            assert timestep is not None, "timestep cannot be None for incremental decoding!"
            output = output[:, -1:].matmul(self.pos_weights[timestep-1])
            output = F.relu_(output)
            output = output.matmul(self.dim_weights2)
            output = self.pe_dropout(output)  # dropout
            output = output + input[:, -1:]  # residual connection

            return output

        n = input.size(1)
        output = output.unsqueeze(2).matmul(self.pos_weights[:n]).squeeze(2)
        output = F.relu_(output)
        output = output.matmul(self.dim_weights2)
        output = self.pe_dropout(output)
        output = output + input  # residual connection

        return output


