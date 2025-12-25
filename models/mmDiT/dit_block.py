import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float, Bool
from einops import rearrange

from .mhsa import MHSA
from .mlp import MLP
from .attention import QKNormedAttention


class DiTBlock(eqx.Module):
    mlp: MLP
    attention: MHSA
    adaLN: Array
    cross_attention: QKNormedAttention

    def __init__(self, dim, text_dim, num_heads, mlp_ratio, key: PRNGKeyArray):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)

        self.attention = MHSA(dim, num_heads, key=key1)
        self.cross_attention = QKNormedAttention(
            num_heads=num_heads,
            in_kv_dim=text_dim,
            in_query_dim=dim,
            kv_dim=dim,
            query_dim=dim,
            key=key5,
            zero_out=True,
        )
        self.mlp = MLP(dim, mlp_ratio, key=key2)

        self.adaLN = jax.random.normal(key6, dim * 6)

    def __call__(
        self,
        x: Float[Array, "num_patches embed_dim"],
        text_tokens: Float[Array, "num_tokens text_embed_dim"],
        sbar: Float[Array, "cond_dim"],
        text_mask: Bool[Array, "num_tokens"] | None = None,
    ) -> Float[Array, "num_patches embed_dim"]:

        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = jnp.split(
            self.adaLN + sbar, 6, axis=0
        )

        x = self.attention(x, gamma1, beta1, alpha1)

        attn_mask = None
        if text_mask is not None:
            attn_mask = jnp.broadcast_to(
                text_mask[None, :], (x.shape[0], text_mask.shape[0])
            )

        x = x + self.cross_attention(
            query=x, key=text_tokens, value=text_tokens, mask=attn_mask
        )

        x = self.mlp(x, gamma2, beta2, alpha2)
        return x
