Autodiff computes derivatives using transformation of pure functions.

Reverse-mode `grad`:

- Efficient for scalar-output functions with many parameters.
- Returns gradients with respect to inputs.

Higher-order derivatives:

- Nest `grad` for second derivatives.
- Use `jax.jacfwd` and `jax.jacrev` for Jacobians.

Rules of differentiation:

- Functions must be pure and use JAX operations.
- Control flow must be traceable to build computation graphs.