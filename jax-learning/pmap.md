`pmap` maps a function over devices for data parallelism.

Basics:

- Splits inputs across devices along a leading axis.
- Executes in parallel with synchronized results.
- Requires array shapes divisible by device count.

Communication:

- `lax.pmean`, `lax.pmax`, `lax.psum` for cross-replica reductions.
- Collectives operate inside pmapped functions.

Usage patterns:

- Replicate parameters, shard data batches.
- Combine with `jit` for compilation on each replica.
- Use `axis_name` to label mapped axes for collectives.