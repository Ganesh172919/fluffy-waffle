import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1] ** 2])

x = jnp.array([1.0, 2.0])

jac_fwd = jax.jacfwd(f)(x)
jac_rev = jax.jacrev(f)(x)

print(jac_fwd)
print(jac_rev)

def g(x):
    return jnp.sum(x ** 2)

hess = jax.hessian(g)(x)
print(hess)

def loss(w):
    return jnp.sum(w ** 4)

w = jnp.array([1.0, 2.0, 3.0])

grad_fn = jax.grad(loss)
grad2_fn = jax.grad(lambda w: jnp.sum(grad_fn(w) * w))

print(grad_fn(w))
print(grad2_fn(w))

y, vjp_fn = jax.vjp(f, x)
v = jnp.array([1.0, 1.0])
vjp_result = vjp_fn(v)
print(vjp_result)