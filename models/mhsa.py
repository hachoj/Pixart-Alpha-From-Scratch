import torch.nn as nn
from jaxtyping import Float
from sympy.polys.polyconfig import query
from torch import Tensor

from .attention import QKNormedAttention


class MHSA(nn.Module):
    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.attention = QKNormedAttention(num_heads, dim, dim, dim)

    def forward(
        self,
        x: Float[Tensor, "b seq_len embed_dim"],
        gamma: Float[Tensor, "b embed_dim"],
        beta: Float[Tensor, "b embed_dim"],
        alpha: Float[Tensor, "b embed_dim"],
    ) -> Float[Tensor, "b seq_len embed_dim"]:

        residual = x
        x = self.layer_norm(x)
        x = x * (1 + gamma) + beta
        x = self.attention(query=x, key=x, value=x)
        return alpha * x + residual
