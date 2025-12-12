Custom autodiff tailors forward and backward passes.

`custom_vjp`:

- Define forward function and custom backward rule.
- Useful for numerically stable or clipped gradients.

`custom_jvp`:

- Define forward pass and directional derivative.
- Useful for functions with known forward-mode rules.

Guidelines:

- Keep functions pure and side-effect free.
- Return residuals needed for backward pass.
- Test correctness against baseline autodiff when possible.