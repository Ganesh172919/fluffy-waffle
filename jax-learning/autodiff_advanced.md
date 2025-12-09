Advanced autodiff computes Jacobians, Hessians, and higher-order derivatives.

Jacobian:

- `jax.jacfwd` forward-mode for tall matrices.
- `jax.jacrev` reverse-mode for wide matrices.
- Returns full derivative matrix.

Hessian:

- `jax.hessian` computes second derivatives.
- Equivalent to `jax.jacfwd(jax.grad(f))` or `jax.jacrev(jax.grad(f))`.

Vector-Jacobian and Jacobian-Vector products:

- Efficient for large models.
- Use `jax.vjp` and `jax.jvp` for custom derivatives.

Applications:

- Optimization with Newton methods.
- Sensitivity analysis and uncertainty quantification.