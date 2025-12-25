import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float

from .attention import QKNormedAttention


class MHSA(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    attention: QKNormedAttention

    def __init__(self, dim, num_heads, key: PRNGKeyArray):
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads"
        self.layer_norm = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)
        self.attention = QKNormedAttention(
            num_heads=num_heads,
            in_kv_dim=dim,
            in_query_dim=dim,
            query_dim=dim,
            kv_dim=dim,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "num_patches embed_dim"],
        gamma: Float[Array, "embed_dim"],
        beta: Float[Array, "embed_dim"],
        alpha: Float[Array, "embed_dim"],
    ) -> Float[Array, "num_patches embed_dim"]:

        residual = x
        x = jax.vmap(self.layer_norm)(x)
        x = x * (1 + gamma) + beta
        x = self.attention(query=x, key=x, value=x)
        return alpha * x + residual
