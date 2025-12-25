import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Float, Int, Bool
from einops import rearrange

from .dit_block import DiTBlock


class SinusoidalTimeEmbedding(eqx.Module):
    emb: Float[Array, "half_dim"]
    dim: int = eqx.field(static=True)
    half_dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        half_dim = dim // 2
        scale = jnp.log(10000.0) / (half_dim - 1)
        freqs = jnp.exp(jnp.arange(half_dim) * -scale)

        self.dim = dim
        self.half_dim = half_dim
        self.emb = freqs

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "dim"]:
        t = t * 1000
        emb = t * self.emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class SinusoidalPosEmbed(eqx.Module):
    emb: Float[Array, "quarter_dim"]
    dim: int = eqx.field(static=True)
    quarter_dim: int = eqx.field(static=True)
    base_h: int = eqx.field(static=True)
    base_w: int = eqx.field(static=True)

    def __init__(self, dim: int, base_size: int, patch_size: int):
        self.base_h = base_size // patch_size
        self.base_w = base_size // patch_size

        self.dim = dim
        self.quarter_dim = dim // 4

        scale = jnp.log(10000.0) / (self.quarter_dim - 1)
        self.emb = jnp.exp(jnp.arange(self.quarter_dim) * -scale)

    def __call__(self, h: int, w: int) -> Float[Array, "h*w dim"]:
        scale_h = self.base_h / h
        scale_w = self.base_w / w

        grid_y, grid_x = jnp.meshgrid(
            jnp.arange(h) * scale_h, jnp.arange(w) * scale_w, indexing="ij"
        )

        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        emb_y = grid_y[:, None] * self.emb[None, :]
        emb_x = grid_x[:, None] * self.emb[None, :]

        emb_y = jnp.concatenate([jnp.sin(emb_y), jnp.cos(emb_y)], axis=-1)
        emb_x = jnp.concatenate([jnp.sin(emb_x), jnp.cos(emb_x)], axis=-1)

        emb = jnp.concatenate([emb_y, emb_x], axis=-1)

        return emb


class DiT(eqx.Module):
    dit_blocks: list[DiTBlock]
    layer_norm: eqx.nn.LayerNorm
    patchify: eqx.nn.Conv2d
    time_proj: SinusoidalTimeEmbedding
    linear_out: eqx.nn.Linear
    adaLN1: eqx.nn.Linear
    adaLN2: eqx.nn.Linear
    adaLN1_single: eqx.nn.Linear
    adaLN2_single: eqx.nn.Linear
    pos_embed: SinusoidalPosEmbed
    p: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim,
        dim,
        cond_dim,
        text_dim,
        num_heads,
        mlp_ratio,
        num_blocks,
        patch_size,
        base_image_size,
        key: PRNGKeyArray,
    ):
        key1, key3, key4, key5, key6, key7, key8 = jr.split(key, 7)

        self.patchify = eqx.nn.Conv2d(
            in_dim,
            dim,
            kernel_size=[patch_size, patch_size],
            padding=[0, 0],
            stride=[patch_size, patch_size],
            key=key1,
        )

        self.time_proj = SinusoidalTimeEmbedding(cond_dim)

        dit_keys = jr.split(key3, num_blocks)
        self.dit_blocks = [
            DiTBlock(dim, text_dim, num_heads, mlp_ratio, key=dit_keys[i])
            for i in range(num_blocks)
        ]

        self.layer_norm = eqx.nn.LayerNorm(dim, use_bias=False, use_weight=False)

        reshape_dim = in_dim * patch_size**2
        self.linear_out = eqx.nn.Linear(dim, reshape_dim, key=key4)
        self.p = patch_size

        self.pos_embed = SinusoidalPosEmbed(dim, base_image_size, patch_size)

        self.adaLN1 = eqx.nn.Linear(cond_dim, dim, key=key5)
        adaLN2_temp = eqx.nn.Linear(dim, dim * 2, key=key6)
        adaLN_w = jnp.zeros_like(adaLN2_temp.weight)
        adaLN_b = jnp.zeros_like(adaLN2_temp.bias)  # pyrefly:ignore
        self.adaLN2 = eqx.tree_at(
            lambda l: (l.weight, l.bias), adaLN2_temp, (adaLN_w, adaLN_b)
        )

        self.adaLN1_single = eqx.nn.Linear(cond_dim, dim, key=key7)
        self.adaLN2_single = eqx.nn.Linear(dim, dim * 6, key=key8)

    def __call__(
        self,
        x: Float[Array, "in_dim height width"],
        t: Float[Array, ""],
        text_tokens: Float[Array, "text_embed_dim"],
        text_mask: Bool[Array, "num_tokens"] | None = None,
    ) -> Float[Array, "in_dim height width"]:
        _, H, W = x.shape
        p = self.p
        h = H // p
        w = W // p

        time_embed = self.time_proj(t)

        # [in_dim,H,W] -> [C,N]  N:=(H//P)(W//P)
        x = self.patchify(x)
        x = rearrange(x, "c h w -> (h w) c")

        x = x + self.pos_embed(h, w)

        sbar = self.adaLN1_single(time_embed)
        sbar = self.adaLN2_single(sbar)

        for block in self.dit_blocks:
            x = block(x, text_tokens, sbar, text_mask=text_mask)

            # use this below if I need more VRAM capacity
            # x = eqx.filter_checkpoint(block)(x, time_embed, class_embed)

        x = jax.vmap(self.layer_norm)(x)
        cond = time_embed

        cond = self.adaLN1(cond)
        cond = jax.nn.silu(cond)
        cond = self.adaLN2(cond)
        gamma, beta = jnp.split(cond, 2, axis=0)

        x = x * (1 + gamma) + beta

        # [N,C] -> [N,p*p*in_dim]
        x = jax.vmap(self.linear_out)(x)
        x = rearrange(x, "(h w) (p1 p2 c) -> c (h p1) (w p2)", h=h, w=w, p1=p, p2=p)

        return x
