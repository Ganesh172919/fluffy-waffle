Optimizers update parameters using gradients.

SGD:

- Simple gradient descent with learning rate.
- Optional momentum for accelerated convergence.

Adam:

- Adaptive learning rates per parameter.
- Uses first and second moment estimates.
- Requires state tracking across iterations.

Implementation patterns:

- Store optimizer state alongside parameters.
- Apply update rules via pure functions.
- Use functional composition for modularity.