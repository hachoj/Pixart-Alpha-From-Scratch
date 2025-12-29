import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class linearOut(nn.Linear):
    pass


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)

        hidden_dim = dim * mlp_ratio
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.linear2 = linearOut(hidden_dim, dim)

        self.apply(self._init_weights)

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

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, linearOut):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
            nn.init.zeros_(m.bias)
