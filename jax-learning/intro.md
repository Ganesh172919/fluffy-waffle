JAX is a numerical computing library for high-performance machine learning.

Key ideas:

- Pure functions enable transformation-based APIs.
- Autodiff via `grad` computes exact reverse-mode derivatives.
- `jit` compiles Python functions to XLA-optimized kernels.
- `vmap` vectorizes functions without manual loops.
- Functional pseudo-randomness via explicit PRNG keys.

Installation:

- Requires Python and a compatible CPU or GPU.
- Install `jax` for CPU or `jax[cuda]` for GPU.

Use cases:

- Differentiable programming and optimization.
- Accelerated array computation and neural networks.
- Research-grade experimentation with clean functional style.