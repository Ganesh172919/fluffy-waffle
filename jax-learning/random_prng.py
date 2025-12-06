import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)
key, sub = jax.random.split(key)
a = jax.random.normal(sub, (3, 3))

key, sub = jax.random.split(key)
b = jax.random.uniform(sub, (3, 3), minval=0.0, maxval=1.0)

print(a)
print(b)