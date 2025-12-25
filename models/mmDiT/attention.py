import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Bool
from einops import rearrange


class QKNormedAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    q_norm: eqx.nn.RMSNorm
    k_norm: eqx.nn.RMSNorm

    def __init__(
        self,
        num_heads,
        in_query_dim,
        in_kv_dim,
        query_dim,
        kv_dim,
        key,
        zero_out=False,
        dtype=None,
    ):
        assert (
            query_dim % num_heads == 0
        ), "Key dimension must be divisible by num_heads"
        assert kv_dim % num_heads == 0, "Query dimension must be divisible by num_heads"
        self.num_heads = num_heads
        query_head_dim = query_dim // num_heads
        key_head_dim = kv_dim // num_heads

        key1, key2, key3, key4 = jr.split(key, 4)

        self.q_proj = eqx.nn.Linear(in_query_dim, query_dim, key=key1, dtype=dtype)
        self.k_proj = eqx.nn.Linear(in_kv_dim, kv_dim, key=key2, dtype=dtype)
        self.v_proj = eqx.nn.Linear(in_kv_dim, kv_dim, key=key3, dtype=dtype)
        self.out_proj = eqx.nn.Linear(query_dim, query_dim, key=key4, dtype=dtype)

        self.q_norm = eqx.nn.RMSNorm(query_head_dim, dtype=dtype)
        self.k_norm = eqx.nn.RMSNorm(key_head_dim, dtype=dtype)

        if zero_out:
            out_w = jnp.zeros_like(self.out_proj.weight)
            out_b = jnp.zeros_like(self.out_proj.bias)  # pyrefly:ignore
            self.out_proj = eqx.tree_at(
                lambda l: (l.weight, l.bias), self.out_proj, (out_w, out_b)
            )

    def __call__(
        self,
        query: Float[Array, "num_q embed_dim"],
        key: Float[Array, "num_kv embed_dim"],
        value: Float[Array, "num_kv embed_dim"],
        mask: Bool[Array, "num_q num_kv"] | None = None,
    ):
        q = jax.vmap(self.q_proj)(query)
        k = jax.vmap(self.k_proj)(key)
        v = jax.vmap(self.v_proj)(value)

        num_query_tokens, _ = q.shape
        num_kv_tokens, _ = key.shape

        q = q.reshape(num_query_tokens, self.num_heads, -1)
        k = k.reshape(num_kv_tokens, self.num_heads, -1)
        v = v.reshape(num_kv_tokens, self.num_heads, -1)

        q = rearrange(q, "n h d -> (n h) d")
        k = rearrange(k, "n h d -> (n h) d")
        q = jax.vmap(self.q_norm)(q)
        k = jax.vmap(self.k_norm)(k)
        q = rearrange(q, "(n h) d -> n h d", h=self.num_heads)
        k = rearrange(k, "(n h) d -> n h d", h=self.num_heads)

        # mask shape should be broadcastable to [h, num_query, num_key]
        attn_output = jax.nn.dot_product_attention(q, k, v, mask=mask, scale=1.0)

        attn_output = attn_output.reshape(num_query_tokens, -1)
        out = jax.vmap(self.out_proj)(attn_output)
        return out
