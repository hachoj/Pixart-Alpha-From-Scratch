import transformer_engine.pytorch as te
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=dim, elementwise_affine=False, bias=False
        )

        hidden_dim = dim * mlp_ratio
        self.linear1 = te.Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=True,
            init_method=lambda w: nn.init.kaiming_normal_(w, mode="fan_in"),
        )
        self.act = nn.SiLU()
        self.linear2 = te.Linear(
            in_features=hidden_dim,
            out_features=dim,
            bias=True,
            init_method=lambda w: nn.init.zeros_(w),
        )

    def forward(
        self,
        x: Float[Tensor, "b seq_len embed_dim"],
        gamma: Float[Tensor, "b embed_dim"],
        beta: Float[Tensor, "b embed_dim"],
        alpha: Float[Tensor, "b embed_dim"],
    ) -> Float[Tensor, "seq_len embed_dim"]:

        residual = x
        x = self.layer_norm(x)
        x = x * (1 + gamma[:, None, :]) + beta[:, None, :]
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return alpha[:, None, :] * x + residual
