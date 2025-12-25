import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
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

        half_dim = dim // 2
        div_term = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(0, half_dim, 2) / half_dim
        )
        self.register_buffer("div_term", div_term)

    def forward(self, h: int, w: int) -> Float[Tensor, "1 n dim"]:
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size

        scale_h = (self.base_size // self.patch_size) / grid_h
        scale_w = (self.base_size // self.patch_size) / grid_w

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


class adaLNOut(nn.Linear):
    pass


class DiT(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        cond_dim,
        num_heads,
        mlp_ratio,
        num_blocks,
        patch_size,
        num_classes,
        base_image_size,
    ):
        self.patchify = nn.Conv2d(
            in_dim,
            dim,
            kernel_size=patch_size,
            padding=0,
            stride=patch_size,
        )

        self.cond_proj = nn.Embedding(num_classes, cond_dim)
        self.time_proj = SinusoidalTimeEmbedding(cond_dim)

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(dim, cond_dim, num_heads, mlp_ratio) for i in range(num_blocks)]
        )

        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)

        reshape_dim = in_dim * patch_size**2
        self.linear_out = nn.Linear(dim, reshape_dim)
        self.p = patch_size

        self.pos_embed = SinusoidalPosEmbed(dim, base_image_size, patch_size)

        self.adaLN1 = nn.Linear(cond_dim, dim)
        self.act = nn.SiLU()
        self.adaLN2 = adaLNOut(dim, dim * 2)

        self.apply(self._init_weights)

    def forward(
        self,
        x: Float[Tensor, "b in_dim height width"],
        t: Float[Tensor, "b"],
        label: Int[Tensor, "b"],
    ) -> Float[Tensor, "b in_dim height width"]:
        H, W = x.shape[:-2]
        p = self.p
        h = H // p
        w = W // p

        time_embed: Float[Tensor, "b cond_dim"] = self.time_proj(t)
        class_embed: Float[Tensor, "b cond_dim"] = self.cond_proj(label)

        x: Float[Tensor, "b embed_dim h//p w//p"] = self.patchify(x)
        x: Float[Tensor, "b num_patches embed_dim"] = rearrange(
            x, "b c h w -> b (h w) c"
        )

        x: Float[Tensor, "b num_patches embed_dim"] = x + self.pos_embed(h, w)

        for block in self.dit_blocks:
            x: Float[Tensor, "b num_patches embed_dim"] = block(
                x, time_embed, class_embed
            )

        x: Float[Tensor, "b num_patches embed_dim"] = self.layer_norm(x)
        cond: Float[Tensor, "b cond_dim"] = time_embed + class_embed

        cond: Float[Tensor, "b ebmed_dim"] = self.adaLN1(cond)
        cond: Float[Tensor, "b ebmed_dim"] = self.act(cond)
        cond: Float[Tensor, "b 2*ebmed_dim"] = self.adaLN2(cond)
        gamma, beta = torch.chunk(cond, 2, dim=-1)

        x: Float[Tensor, "b num_patches embed_dim"] = x * (1 + gamma) + beta

        x: Float[Tensor, "b num_patches 3*p*p"] = self.linear_out(x)
        x: Float[Tensor, "b 3 h w"] = rearrange(
            x, "(h w) (p1 p2 c) -> c (h p1) (w p2)", h=h, w=w, p1=p, p2=p
        )

        return x

    def _init_weights(self, m: nn.Module):
        if isinstance(m, adaLNOut):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
            nn.init.zeros_(m.bias)
