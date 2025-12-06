Neural networks in JAX use functional parameters and state.

Parameters:

- Initialize with random keys and shapes.
- Store as PyTrees for structured updates.

Forward pass:

- Pure function mapping params and inputs to outputs.
- Use `jit` and `vmap` for performance and batching.

Training loop:

- Compute loss from predictions and targets.
- Use `grad` to update parameters via optimizers.