import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return jnp.sin(x) + jnp.cos(x)

xs = jnp.linspace(0.0, 3.14, 10)
ys = jax.vmap(f)(xs)

print(ys)