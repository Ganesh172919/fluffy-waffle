import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (4, 4))
b = jax.random.normal(key, (4, 4))

c = a @ b
print(c)

w, v = jnp.linalg.eig(a)
print(w)
print(v)

u, s, vh = jnp.linalg.svd(a)
print(u.shape, s.shape, vh.shape)

q, r = jnp.linalg.qr(a)
print(q @ r)

x = jnp.array([1.0, 2.0, 3.0, 4.0])
b_vec = a @ x
x_sol = jnp.linalg.solve(a, b_vec)
print(x_sol)

batched_a = jax.random.normal(key, (8, 4, 4))
batched_x = jax.random.normal(key, (8, 4, 1))
batched_y = jnp.matmul(batched_a, batched_x)
print(batched_y.shape)