Convolutional neural networks process spatial data with weight sharing.

Convolution:

- `jax.lax.conv` applies 2D or 3D convolutions.
- Parameters include kernel size, stride, padding.
- Outputs have spatial dimensions determined by input and kernel.

Pooling:

- Max and average pooling reduce spatial size.
- Use `jax.lax.reduce_window` for custom pooling.

Architecture patterns:

- Stack conv, activation, and pooling layers.
- Flatten spatial dims before fully connected layers.
- Use batch normalization for training stability.