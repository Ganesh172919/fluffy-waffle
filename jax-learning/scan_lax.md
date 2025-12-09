Scan and lax primitives enable efficient loops and control flow.

`jax.lax.scan`:

- Efficient sequential iteration with carry state.
- Compiles to optimized loop kernels.
- Returns final carry and stacked outputs.

`jax.lax.cond`:

- Conditional branching based on boolean predicate.
- Both branches must be traceable.

`jax.lax.while_loop` and `jax.lax.fori_loop`:

- While loops with condition functions.
- For loops with fixed iteration counts.

Use cases:

- RNN unrolling and sequence processing.
- Iterative solvers and dynamic algorithms.