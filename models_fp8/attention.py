import transformer_engine.pytorch as te
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor


class QKNormedAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        in_query_dim,
        in_kv_dim,
        query_dim,
        attention_type,
    ):
        super().__init__()
        assert (
            query_dim % num_heads == 0
        ), "Query dimension must be divisible by num_heads"
        assert (
            attention_type == "cross" or attention_type == "self"
        ), "Invalid attention type"
        self.num_heads = num_heads
        head_dim = query_dim // num_heads

        self.q_proj = te.Linear(
            in_features=in_query_dim,
            out_features=query_dim,
            bias=True,
            init_method=lambda w: nn.init.normal_(w, mean=0, std=0.02),
            fuse_wgrad_accumulation=True,
        )
        self.k_proj = te.Linear(
            in_features=in_kv_dim,
            out_features=query_dim,
            bias=True,
            init_method=lambda w: nn.init.normal_(w, mean=0, std=0.02),
            fuse_wgrad_accumulation=True,
        )
        self.v_proj = te.Linear(
            in_features=in_kv_dim,
            out_features=query_dim,
            bias=True,
            init_method=lambda w: nn.init.normal_(w, mean=0, std=0.02),
            fuse_wgrad_accumulation=True,
        )
        self.out_proj = te.Linear(
            in_features=query_dim,
            out_features=query_dim,
            bias=True,
            init_method=lambda w: nn.init.zeros_(w),
            fuse_wgrad_accumulation=True,
        )

        self.q_norm = te.RMSNorm(normalized_shape=head_dim)
        self.k_norm = te.RMSNorm(normalized_shape=head_dim)

        self.attention = te.DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=query_dim,
            attention_type=attention_type,
            attn_mask_type="arbitrary",
            qkv_format="bshd",
        )

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

        if mask is not None:
            mask.unsqueeze(1)

        attn_output: Float[Tensor, "b num_heads num_q head_dim"] = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        )

        attn_output: Float[Tensor, "b num_q q_dim"] = rearrange(
            attn_output, "b h n d -> b n (h d)"
        )
        out: Float[Tensor, "b num_q q_dim"] = self.out_proj(attn_output)
        return out
