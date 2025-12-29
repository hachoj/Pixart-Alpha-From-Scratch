import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor


class OutLinear(nn.Linear):
    pass


class QKNormedAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        in_query_dim,
        in_kv_dim,
        query_dim,
    ):
        super().__init__()
        assert (
            query_dim % num_heads == 0
        ), "Query dimension must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = query_dim // num_heads

        self.q_proj = nn.Linear(in_query_dim, query_dim)
        self.k_proj = nn.Linear(in_kv_dim, query_dim)
        self.v_proj = nn.Linear(in_kv_dim, query_dim)
        self.out_proj = OutLinear(query_dim, query_dim)

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

        self.apply(self._init_weights)

    def forward(
        self,
        query: Float[Tensor, "b num_q q_dim"],
        key: Float[Tensor, "b num_kv kv_dim"],
        value: Float[Tensor, "b num_kv kv_dim"],
        mask: Bool[Tensor, "b num_q num_kv"] | None = None,
    ):
        q: Float[Tensor, "b num_q q_dim"] = self.q_proj(query)
        k: Float[Tensor, "b num_kv q_dim"] = self.k_proj(key)
        v: Float[Tensor, "b num_kv q_dim"] = self.v_proj(value)

        q: Float[Tensor, "b num_heads num_q head_dim"] = rearrange(
            q, "b n (h d) -> b h n d", h=self.num_heads
        )
        k: Float[Tensor, "b num_heads num_kv head_dim"] = rearrange(
            k, "b n (h d) -> b h n d", h=self.num_heads
        )
        v: Float[Tensor, "b num_heads num_kv head_dim"] = rearrange(
            v, "b n (h d) -> b h n d", h=self.num_heads
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_output: Float[Tensor, "b num_heads num_q head_dim"] = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        )

        attn_output: Float[Tensor, "b num_q q_dim"] = rearrange(
            attn_output, "b h n d -> b n (h d)"
        )
        out: Float[Tensor, "b num_q q_dim"] = self.out_proj(attn_output)
        return out

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, OutLinear):
            nn.init.zeros_(m.bias)
            nn.init.zeros_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            nn.init.normal_(m.weight, mean=0, std=0.02)
