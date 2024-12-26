# minimal implementation of Mamba
# the parameter meaning
'''
b : batch_size
l : sequence length
d or d_model : the hidden dim
n or d_state : latent state dim
expand : expansion factor
d_in or d_inner : d * expand
A, B, C, D : state space parameters
Δ or delta : input-dependent step size
dt-rank : rank of Δ
'''

# import the base codebase
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


# define data class, the d_rank is set to be 1 and
@dataclass
class ModelArgs:
    d_model: int # 720
    d_state: int
    expand: int
    d_conv: int
    bias: bool = False
    conv_bias: bool = True
    n_layer: int = 15

    # define preprocess operations
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)



# define the mamba structure
class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
    def forward(self, input_ids):

        x = input_ids
        for layer in self.layers:
            x = layer(x)  # (b, l, d_model)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):

        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)

    def forward(self, x):



        output = self.mixer(x) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block"""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )


        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):

        # get the shape parameter first
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape(b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        # get through the activation layer， silu activation is useful in deeper network
        x = F.silu(x)  # (b, l, d_in)

        # get throught the ssm layer, this layer makes no change on the dimension
        y = x  # (b, l, d_in)


        # make the final projection to get h(t)
        output = self.out_proj(y)  # (b, l, d_model)

        return output





