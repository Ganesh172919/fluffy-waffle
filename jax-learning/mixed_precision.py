import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
x32 = jax.random.normal(key, (1024, 1024), dtype=jnp.float32)
x16 = x32.astype(jnp.float16)

@jax.jit
def matmul_precise(a, b):
    return jax.lax.dot(a, b, precision=jax.lax.Precision.HIGHEST)

@jax.jit
def matmul_mixed(a, b):
    return jnp.matmul(a, b)

c32 = matmul_precise(x32, x32)
c16 = matmul_mixed(x16, x16).astype(jnp.float32)

print(jnp.mean(jnp.abs(c32 - c16)))

params = {
    "w": jax.random.normal(key, (128, 128), dtype=jnp.float32),
    "b": jnp.zeros((128,), dtype=jnp.float32),
}

@jax.jit
def forward(params, x):
    y = jnp.matmul(x.astype(jnp.bfloat16), params["w"].astype(jnp.bfloat16))
    y = y.astype(jnp.float32) + params["b"]
    return jnp.tanh(y)

x_in = jax.random.normal(key, (64, 128), dtype=jnp.float32)
out = forward(params, x_in)
print(out.dtype)