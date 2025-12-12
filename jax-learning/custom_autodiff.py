import jax
import jax.numpy as jnp

@jax.custom_vjp

def stable_log(x):
    return jnp.log(x)

def stable_log_fwd(x):
    y = jnp.log(x)
    return y, x

def stable_log_bwd(res, g):
    x = res
    return (g / jnp.maximum(x, 1e-5),)

stable_log.defvjp(stable_log_fwd, stable_log_bwd)

x = jnp.array([0.1, 1.0, 10.0])
print(stable_log(x))

@jax.custom_jvp

def square(x):
    return x * x

@square.defjvp

def square_jvp(primals, tangents):
    x, = primals
    t, = tangents
    return square(x), 2 * x * t

value, grad = jax.value_and_grad(stable_log)(jnp.array(2.0))
print(value)
print(grad)

square_grad = jax.grad(square)(jnp.array(3.0))
print(square_grad)

def loss_fn(x):
    return jnp.sum(stable_log(x) + square(x))

x0 = jnp.array([1.0, 2.0, 3.0])
print(jax.grad(loss_fn)(x0))