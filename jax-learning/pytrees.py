import jax
import jax.numpy as jnp

params = {
    "layer1": {"w": jnp.ones((3, 4)), "b": jnp.zeros(4)},
    "layer2": {"w": jnp.ones((4, 2)), "b": jnp.zeros(2)},
}

scaled = jax.tree_map(lambda x: x * 2.0, params)

flat, tree_def = jax.tree_flatten(params)
print(flat)

unflat = jax.tree_unflatten(tree_def, flat)
print(unflat)

def norm(pytree):
    leaves, _ = jax.tree_flatten(pytree)
    return jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves))

print(norm(params))