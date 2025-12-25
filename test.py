import jax.numpy as jnp
from jaxtyping import Array

test_int: int = 10

print(test_int < 100)

test_int_ar: Array = jnp.asarray(test_int)

print(test_int_ar < 100)
