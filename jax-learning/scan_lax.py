import jax
import jax.numpy as jnp

def step(carry, x):
    total, count = carry
    total = total + x
    count = count + 1
    return (total, count), x ** 2

xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
init_carry = (0.0, 0)
final_carry, ys = jax.lax.scan(step, init_carry, xs)

print(final_carry)
print(ys)

def rnn_step(carry, x):
    h = jnp.tanh(carry + x)
    return h, h

h0 = jnp.zeros(8)
xs_seq = jax.random.normal(jax.random.PRNGKey(0), (10, 8))
final_h, hs = jax.lax.scan(rnn_step, h0, xs_seq)

print(final_h.shape)
print(hs.shape)

def cond_fn(x):
    return jax.lax.cond(x > 0, lambda x: x ** 2, lambda x: -x, x)

print(cond_fn(3.0))
print(cond_fn(-2.0))