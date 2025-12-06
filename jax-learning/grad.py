import jax
import jax.numpy as jnp

def loss(w, x, y):
    preds = jnp.dot(x, w)
    return jnp.mean((preds - y) ** 2)

grad_loss = jax.grad(loss)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100,))
w = jnp.array(0.0)
y = 2.0 * x + 1.0

g = grad_loss(w, x, y)

print(g)