Mixed precision uses lower dtypes to speed up computation.

Dtypes:

- `float16` and `bfloat16` reduce memory and improve throughput.
- Accumulate in `float32` for numerical stability.

Patterns:

- Cast inputs to lower precision while keeping master params in `float32`.
- Use `jax.lax.precision` to control matmul accumulation.

Caveats:

- Watch for overflow and underflow.
- Validate accuracy against full precision baselines.