Linear algebra operations power many models.

Core ops:

- Matrix multiply: `jnp.dot`, `jnp.matmul` for batched multiplication.
- Decompositions: `jnp.linalg.eig`, `jnp.linalg.svd`, `jnp.linalg.qr`.
- Solvers: `jnp.linalg.solve` for linear systems.

Considerations:

- Prefer well-conditioned matrices.
- Regularize ill-conditioned problems.
- Use `jit` for performance on repeated shapes.