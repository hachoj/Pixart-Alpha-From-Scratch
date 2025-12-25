import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float


class MLP(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, mlp_ratio: int, key: PRNGKeyArray):
        key1, key2 = jr.split(key, 2)
        self.layer_norm = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)

        hidden_dim = dim * mlp_ratio
        self.linear1 = eqx.nn.Linear(dim, hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_dim, dim, key=key2)

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
        x = jax.vmap(self.linear1)(x)
        x = jax.nn.silu(x)
        x = jax.vmap(self.linear2)(x)
        return alpha * x + residual
