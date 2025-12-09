PyTrees are nested structures of arrays and containers.

Structure:

- Dicts, lists, tuples, and custom classes.
- Leaves are arrays or scalars.
- Used for parameters, states, and nested data.

Operations:

- `jax.tree_map` applies functions element-wise.
- `jax.tree_flatten` and `jax.tree_unflatten` convert to flat lists.
- Custom types registered via `jax.tree_util.register_pytree_node`.

Benefits:

- Uniform handling of complex parameter structures.
- Enables generic optimizer and transformation code.