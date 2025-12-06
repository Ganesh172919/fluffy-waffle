import jax
import jax.numpy as jnp

def init_params(key, in_dim, hidden_dim, out_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W1 = jax.random.normal(k1, (in_dim, hidden_dim)) * 0.1
    b1 = jnp.zeros((hidden_dim,))
    W2 = jax.random.normal(k2, (hidden_dim, out_dim)) * 0.1
    b2 = jnp.zeros((out_dim,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    y = h @ params["W2"] + params["b2"]
    return y

def loss(params, x, y):
    preds = forward(params, x)
    return jnp.mean((preds - y) ** 2)

grad_loss = jax.grad(loss)

key = jax.random.PRNGKey(0)
params = init_params(key, 4, 16, 1)

x_key = jax.random.PRNGKey(1)
x = jax.random.normal(x_key, (128, 4))
y = jnp.sum(x, axis=1, keepdims=True)

lr = 0.1
for i in range(100):
    g = grad_loss(params, x, y)
    params = {
        "W1": params["W1"] - lr * g["W1"],
        "b1": params["b1"] - lr * g["b1"],
        "W2": params["W2"] - lr * g["W2"],
        "b2": params["b2"] - lr * g["b2"],
    }

print(loss(params, x, y))