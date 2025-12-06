`jit` compiles functions for speed and consistent performance.

- Improves throughput by lowering to XLA.
- Shapes and dtypes affect compilation cache keys.

`vmap` vectorizes pure functions across a batch axis.

- Eliminates manual loops and broadcasting trickery.
- Works with `jit` for fused, batched kernels.

Composability:

- Combine `grad`, `jit`, and `vmap` for scalable autodiff.
- Maintain pure functional boundaries for predictable transforms.