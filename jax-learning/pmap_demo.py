import jax
import jax.numpy as jnp

n_devices = jax.device_count()
xs = jnp.arange(8).reshape((n_devices, -1))

@jax.pmap

def add_one(x):
    return x + 1

ys = add_one(xs)
print(ys)

@jax.pmap

def mean_across_devices(x):
    return jax.lax.pmean(x, axis_name="d")

zs = mean_across_devices(xs, axis_name="d")
print(zs)

@jax.pmap

def scale_sum(x):
    total = jax.lax.psum(x, axis_name="d")
    return total / n_devices

scaled = scale_sum(xs, axis_name="d")
print(scaled)

@jax.pmap

def dot_row(x):
    return jnp.dot(x, x)

dots = dot_row(xs)
print(dots)