import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float
from einops import rearrange


class QKNormedAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    q_norm: eqx.nn.RMSNorm
    k_norm: eqx.nn.RMSNorm

    def __init__(self, num_heads, query_dim, kv_dim, key, dtype=None):
        assert (
            query_dim % num_heads == 0
        ), "Key dimension must be divisible by num_heads"
        assert kv_dim % num_heads == 0, "Query dimension must be divisible by num_heads"
        self.num_heads = num_heads
        query_head_dim = query_dim // num_heads
        key_head_dim = kv_dim // num_heads

        key1, key2, key3, key4 = jr.split(key, 4)

        self.q_proj = eqx.nn.Linear(query_dim, query_dim, key=key1, dtype=dtype)
        self.k_proj = eqx.nn.Linear(kv_dim, kv_dim, key=key2, dtype=dtype)
        self.v_proj = eqx.nn.Linear(kv_dim, kv_dim, key=key3, dtype=dtype)
        self.out_proj = eqx.nn.Linear(query_dim, query_dim, key=key4, dtype=dtype)

        self.q_norm = eqx.nn.RMSNorm(query_head_dim, dtype=dtype)
        self.k_norm = eqx.nn.RMSNorm(key_head_dim, dtype=dtype)

    def __call__(
        self,
        query: Float[Array, "num_q embed_dim"],
        key: Float[Array, "num_kv embed_dim"],
        value: Float[Array, "num_kv embed_dim"],
    ):
        q = jax.vmap(self.q_proj)(query)
        k = jax.vmap(self.k_proj)(key)
        v = jax.vmap(self.v_proj)(value)

        n, _ = q.shape
        q = q.reshape(n, self.num_heads, -1)
        k = k.reshape(n, self.num_heads, -1)
        v = v.reshape(n, self.num_heads, -1)

        q = rearrange(q, "n h d -> (n h) d")
        k = rearrange(k, "n h d -> (n h) d")
        q = jax.vmap(self.q_norm)(q)
        k = jax.vmap(self.k_norm)(k)
        q = rearrange(q, "(n h) d -> n h d", h=self.num_heads)
        k = rearrange(k, "(n h) d -> n h d", h=self.num_heads)

        attn_output = jax.nn.dot_product_attention(q, k, v, scale=1.0)

        attn_output = attn_output.reshape(n, -1)
        out = jax.vmap(self.out_proj)(attn_output)
        return out
