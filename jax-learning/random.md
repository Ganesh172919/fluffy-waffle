JAX randomness is explicit via PRNG keys.

PRNG keys:

- Initialize with `jax.random.PRNGKey(seed)`.
- Split keys to generate independent streams.

Determinism:

- Reproducibility depends on key management.
- Avoid hidden global state for clarity.

Usage:

- Pass keys through functions.
- Return new keys with values to avoid reuse.