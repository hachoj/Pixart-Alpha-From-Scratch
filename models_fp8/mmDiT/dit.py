import transformer_engine.pytorch as te
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from .dit_block import DiTBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

        half: int = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half) / (half - 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, times: Float[Tensor, "b"]) -> torch.Tensor:
        b: int = times.shape[0]
        device, dtype = times.device, times.dtype

        angles = times[:, None] * self.freqs[None, :]  # pyrefly:ignore
        emb = torch.empty((b, self.dim), device=device, dtype=dtype)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


class SinusoidalPosEmbed(nn.Module):
    # Type hint the buffer to help static analysis
    div_term: Tensor

    def __init__(self, dim: int, base_size: int, patch_size: int) -> None:
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.base_size = base_size
        self.patch_size = patch_size

        # Express the training/reference resolution in *patch-grid* units.
        self.base_grid = base_size // patch_size

        half_dim = dim // 2
        div_term = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, half_dim, 2) / half_dim
        )
        self.register_buffer("div_term", div_term)

    def forward(self, h: int, w: int) -> Float[Tensor, "1 n dim"]:
        # `h`, `w` are patch-grid sizes (number of tokens along height/width).
        grid_h, grid_w = h, w

        scale_h = self.base_grid / grid_h
        scale_w = self.base_grid / grid_w

        device = self.div_term.device
        dtype = self.div_term.dtype

        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype)
            * scale_h,  # pyrefly:ignore
            torch.arange(grid_w, device=device, dtype=dtype)
            * scale_w,  # pyrefly:ignore
            indexing="ij",
        )

        grid_y = grid_y.flatten()[:, None]
        grid_x = grid_x.flatten()[:, None]

        emb_y = grid_y * self.div_term[None, :]  # pyrefly:ignore
        emb_x = grid_x * self.div_term[None, :]  # pyrefly:ignore

        emb = torch.zeros((grid_h * grid_w, self.dim), device=device, dtype=dtype)

        emb[:, 0 : self.dim // 2 : 2] = torch.sin(emb_y)
        emb[:, 1 : self.dim // 2 : 2] = torch.cos(emb_y)

        emb[:, self.dim // 2 :: 2] = torch.sin(emb_x)
        emb[:, self.dim // 2 + 1 :: 2] = torch.cos(emb_x)

        return emb[None, :, :]


class DiT(nn.Module):
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
    ):
        super().__init__()
        self.patchify = nn.Conv2d(
            in_dim,
            dim,
            kernel_size=patch_size,
            padding=0,
            stride=patch_size,
        )

        self.time_proj = SinusoidalTimeEmbedding(cond_dim)

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(dim, text_dim, num_heads, mlp_ratio) for i in range(num_blocks)]
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=dim, elementwise_affine=False, bias=False
        )

        reshape_dim = in_dim * patch_size**2
        self.linear_out = te.Linear(
            in_features=dim,
            out_features=reshape_dim,
            bias=True,
            init_method=lambda w: nn.init.kaiming_normal_(w, mode="fan_in"),
        )
        self.p = patch_size

        self.pos_embed = SinusoidalPosEmbed(dim, base_image_size, patch_size)

        self.adaLN1 = te.Linear(
            in_features=cond_dim,
            out_features=dim,
            bias=True,
            init_method=lambda w: nn.init.kaiming_normal_(w, mode="fan_in"),
        )
        self.act = nn.SiLU()
        self.adaLN2 = te.Linear(
            in_features=dim,
            out_features=dim * 2,
            bias=True,
            init_method=lambda w: nn.init.zeros_(w),
        )

        self.adaLN1_single = te.Linear(
            in_features=cond_dim,
            out_features=dim,
            bias=True,
            init_method=lambda w: nn.init.kaiming_normal_(w, mode="fan_in"),
        )
        self.adaLN2_single = te.Linear(
            in_features=dim,
            out_features=6 * dim,
            bias=True,
            init_method=lambda w: nn.init.zeros_(w),
        )

    def forward(
        self,
        x: Float[Tensor, "b in_dim height width"],
        t: Float[Tensor, "b"],
        text_tokens: Float[Tensor, "b num_tokens text_dim"],
        text_mask: Bool[Tensor, "b num_tokens"],
    ) -> Float[Tensor, "b in_dim height width"]:
        H, W = x.shape[-2:]
        p = self.p
        h = H // p
        w = W // p

        time_embed: Float[Tensor, "b cond_dim"] = self.time_proj(t)

        x: Float[Tensor, "b embed_dim h//p w//p"] = self.patchify(x)
        x: Float[Tensor, "b num_patches embed_dim"] = rearrange(
            x, "b c h w -> b (h w) c"
        )

        x: Float[Tensor, "b num_patches embed_dim"] = x + self.pos_embed(h, w)

        sbar = self.adaLN1_single(time_embed)
        sbar = self.act(sbar)
        sbar = self.adaLN2_single(sbar)

        for block in self.dit_blocks:
            x: Float[Tensor, "b num_patches embed_dim"] = block(
                x, text_tokens, sbar, text_mask
            )

        x: Float[Tensor, "b num_patches embed_dim"] = self.layer_norm(x)
        cond: Float[Tensor, "b cond_dim"] = time_embed

        cond: Float[Tensor, "b ebmed_dim"] = self.adaLN1(cond)
        cond: Float[Tensor, "b ebmed_dim"] = self.act(cond)
        cond: Float[Tensor, "b 2*ebmed_dim"] = self.adaLN2(cond)
        gamma, beta = torch.chunk(cond, 2, dim=-1)

        x: Float[Tensor, "b num_patches embed_dim"] = (
            x * (1 + gamma[:, None, :]) + beta[:, None, :]
        )

        x: Float[Tensor, "b num_patches 3*p*p"] = self.linear_out(x)
        x: Float[Tensor, "b 3 h w"] = rearrange(
            x, "b (h w) (p1 p2 c) ->  b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p
        )

        return x
