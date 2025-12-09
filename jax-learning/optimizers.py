import jax
import jax.numpy as jnp

def sgd_update(params, grads, lr):
    return jax.tree_map(lambda p, g: p - lr * g, params, grads)

def adam_init(params):
    m = jax.tree_map(jnp.zeros_like, params)
    v = jax.tree_map(jnp.zeros_like, params)
    return m, v, 0

def adam_update(params, grads, state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m, v, t = state
    t = t + 1
    m = jax.tree_map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grads)
    v = jax.tree_map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v, grads)
    m_hat = jax.tree_map(lambda mi: mi / (1 - beta1 ** t), m)
    v_hat = jax.tree_map(lambda vi: vi / (1 - beta2 ** t), v)
    params = jax.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, (m, v, t)

def loss(w, x, y):
    pred = x @ w
    return jnp.mean((pred - y) ** 2)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 5))
w_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = x @ w_true + 0.1 * jax.random.normal(key, (100,))

w = jnp.zeros(5)
state = adam_init(w)
grad_fn = jax.grad(loss)

for i in range(200):
    g = grad_fn(w, x, y)
    w, state = adam_update(w, g, state, lr=0.01)

print(w)
print(loss(w, x, y))