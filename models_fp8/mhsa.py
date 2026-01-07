import transformer_engine.pytorch as te
import torch.nn as nn
from jaxtyping import Float
from sympy.polys.polyconfig import query
from torch import Tensor

from .attention import QKNormedAttention


class MHSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.layer_norm = nn.LayerNorm(
            normalized_shape=dim, elementwise_affine=False, bias=False
        )
        self.attention = QKNormedAttention(
            num_heads, dim, dim, dim, attention_type="self"
        )

    def forward(
        self,
        x: Float[Tensor, "b seq_len embed_dim"],
        gamma: Float[Tensor, "b embed_dim"],
        beta: Float[Tensor, "b embed_dim"],
        alpha: Float[Tensor, "b embed_dim"],
    ) -> Float[Tensor, "b seq_len embed_dim"]:

        residual = x
        x = self.layer_norm(x)
        x = x * (1 + gamma[:, None, :]) + beta[:, None, :]
        x = self.attention(query=x, key=x, value=x)
        return alpha[:, None, :] * x + residual
