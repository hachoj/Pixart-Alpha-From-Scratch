import torch
import torch.nn as nn
from einops import repeat
from jaxtyping import Bool, Float
from torch import Tensor

from ..attention import QKNormedAttention
from ..mhsa import MHSA
from ..mlp import MLP


class DiTBlock(nn.Module):
    def __init__(self, dim, text_dim, num_heads, mlp_ratio):
        super().__init__()
        self.attention = MHSA(dim, num_heads)
        self.cross_attention = QKNormedAttention(
            num_heads=num_heads,
            in_kv_dim=text_dim,
            in_query_dim=dim,
            query_dim=dim,
        )
        self.mlp = MLP(dim, mlp_ratio)
        self.adaLNembed = nn.Parameter(torch.zeros((dim * 6,)))

    def forward(
        self,
        x: Float[Tensor, "b seq_len embed_dim"],
        text_tokens: Float[Tensor, "b num_tokens text_embed_dim"],
        sbar: Float[Tensor, "b cond_dim"],
        text_mask: Bool[Tensor, "b num_tokens"],
    ) -> Float[Tensor, "b seq_len embed_dim"]:
        s = x.shape[1]

        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = torch.chunk(
            self.adaLNembed + sbar, 6, dim=-1
        )

        x: Float[Tensor, "b seq_len embed_dim"] = self.attention(
            x, gamma1, beta1, alpha1
        )

        attn_mask: Bool[Tensor, "b seq_len num_tokens"] = repeat(
            text_mask, "b t -> b s t", s=s
        )

        x: Float[Tensor, "b seq_len embed_dim"] = x + self.cross_attention(
            query=x, key=text_tokens, value=text_tokens, mask=attn_mask
        )

        x: Float[Tensor, "b seq_len embed_dim"] = self.mlp(x, gamma2, beta2, alpha2)
        return x
