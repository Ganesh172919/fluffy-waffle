import jax
import jax.numpy as jnp
from jax import random

def init_conv_params(key, in_channels, out_channels, kernel_size):
    k1, k2 = random.split(key)
    w = random.normal(k1, (out_channels, in_channels, kernel_size, kernel_size)) * 0.1
    b = jnp.zeros(out_channels)
    return w, b

def conv2d(x, w, b):
    y = jax.lax.conv(x, w, (1, 1), "SAME")
    return y + b.reshape(1, -1, 1, 1)

def relu(x):
    return jnp.maximum(0, x)

def max_pool(x, window_size):
    dims = (1, 1, window_size, window_size)
    strides = (1, 1, window_size, window_size)
    return jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, dims, strides, "VALID")

def forward(params, x):
    w1, b1, w2, b2 = params
    h = conv2d(x, w1, b1)
    h = relu(h)
    h = max_pool(h, 2)
    h = conv2d(h, w2, b2)
    h = relu(h)
    h = jnp.mean(h, axis=(2, 3))
    return h

key = random.PRNGKey(0)
k1, k2 = random.split(key)
w1, b1 = init_conv_params(k1, 3, 16, 3)
w2, b2 = init_conv_params(k2, 16, 32, 3)
params = (w1, b1, w2, b2)

x = random.normal(random.PRNGKey(1), (8, 3, 32, 32))
out = forward(params, x)

print(out.shape)