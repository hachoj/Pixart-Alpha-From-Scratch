from jax.dlpack import from_dlpack as jax_from_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from jaxtyping import Array
from torch import Tensor


def torch_to_jax(x: Tensor) -> Array:
    return jax_from_dlpack(x.contiguous())


def jax_to_torch(x: Array) -> Tensor:
    return torch_from_dlpack(x)
