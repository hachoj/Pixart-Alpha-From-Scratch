import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from ..mhsa import MHSA
from ..mlp import MLP


class adaLNOut(nn.Linear):
    pass


class DiTBlock(nn.Module):
    def __init__(self, dim, cond_dim, num_heads, mlp_ratio):
        super().__init__()
        self.attention = MHSA(dim, num_heads)
        self.mlp = MLP(dim, mlp_ratio)

        self.adaLN1 = nn.Linear(cond_dim, dim)
        self.act = nn.SiLU()
        self.adaLN2 = adaLNOut(dim, dim * 6)

    def forward(
        self,
        x: Float[Tensor, "b seq_len embed_dim"],
        t: Float[Tensor, "b cond_dim"],
        c: Float[Tensor, "b cond_dim"],
    ) -> Float[Tensor, "b seq_len embed_dim"]:

        cond: Float[Tensor, "b cond_dim"] = t + c

        cond: Float[Tensor, "b cond_dim"] = self.adaLN1(cond)
        cond: Float[Tensor, "b cond_dim"] = self.act(cond)
        cond: Float[Tensor, "b cond_dim"] = self.adaLN2(cond)

        gamma1, beta1, gamma2, beta2, alpha1, alpha2 = torch.chunk(cond, 6, dim=-1)

        x: Float[Tensor, "b seq_len embed_dim"] = self.attention(
            x, gamma1, beta1, alpha1
        )
        x: Float[Tensor, "b seq_len embed_dim"] = self.mlp(x, gamma2, beta2, alpha2)
        return x

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, adaLNOut):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
            nn.init.zeros_(m.bias)
