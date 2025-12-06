Arrays in JAX use `jax.numpy` and are immutable.

Core concepts:

- Device-backed arrays on CPU or GPU/TPU.
- Operations mirror NumPy but return JAX arrays.
- Broadcasting, indexing, and dtype semantics follow NumPy.

Performance notes:

- Prefer `jit` for hot paths.
- Avoid Python-side loops for large data.
- Use `vmap` for batched application of pure functions.