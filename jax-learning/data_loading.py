import jax
import jax.numpy as jnp
import numpy as np

def create_batches(x, y, batch_size, key):
    n = x.shape[0]
    indices = jax.random.permutation(key, n)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    
    num_batches = n // batch_size
    x_batches = [x_shuffled[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    y_batches = [y_shuffled[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    return x_batches, y_batches

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1000, 10))
y = jax.random.normal(key, (1000, 1))

batch_key = jax.random.PRNGKey(42)
x_batches, y_batches = create_batches(x, y, 32, batch_key)

print(len(x_batches))
print(x_batches[0].shape)

def normalize(x, axis=0):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    std = jnp.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + 1e-8)

x_norm = normalize(x)
print(jnp.mean(x_norm, axis=0))
print(jnp.std(x_norm, axis=0))

def augment_data(x, key):
    noise = jax.random.normal(key, x.shape) * 0.1
    return x + noise

aug_key = jax.random.PRNGKey(99)
x_aug = augment_data(x[:5], aug_key)
print(x_aug.shape)