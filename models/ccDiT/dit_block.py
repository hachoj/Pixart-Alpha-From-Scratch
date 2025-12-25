import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float
from einops import rearrange

from .mhsa import MHSA
from .mlp import MLP


class DiTBlock(eqx.Module):
    mlp: MLP
    attention: MHSA
    adaLN1: eqx.nn.Linear
    adaLN2: eqx.nn.Linear

    def __init__(self, dim, cond_dim, num_heads, mlp_ratio, key: PRNGKeyArray):
        key1, key2, key3, key4 = jr.split(key, 4)

        self.attention = MHSA(dim, num_heads, key=key1)
        self.mlp = MLP(dim, mlp_ratio, key=key2)

        self.adaLN1 = eqx.nn.Linear(cond_dim, dim, key=key3)
        adaLN2_temp = eqx.nn.Linear(dim, dim * 6, key=key4)
        adaLN_w = jnp.zeros_like(adaLN2_temp.weight)
        adaLN_b = jnp.zeros_like(adaLN2_temp.bias)  # pyrefly:ignore
        self.adaLN2 = eqx.tree_at(
            lambda l: (l.weight, l.bias), adaLN2_temp, (adaLN_w, adaLN_b)
        )

    def __call__(
        self,
        x: Float[Array, "num_patches embed_dim"],
        t: Float[Array, "cond_dim"],
        c: Float[Array, "cond_dim"],
    ) -> Float[Array, "num_patches embed_dim"]:

        cond = t + c

        cond = self.adaLN1(cond)
        cond = jax.nn.silu(cond)
        cond = self.adaLN2(cond)

        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = jnp.split(cond, 6, axis=0)

        x = self.attention(x, gamma1, beta1, alpha1)
        x = self.mlp(x, gamma2, beta2, alpha2)
        return x
