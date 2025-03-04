# JAX-Flash-Attention2

A flexible and efficient implementation of Flash Attention 2.0 for JAX, supporting multiple backends (GPU/TPU/CPU) and platforms (Triton/Pallas/JAX).

## Installation

```bash
pip install jax-flash-attn2
```

## Basic Usage

```python
import jax
import jax.numpy as jnp
import jax_flash_attn2 as jfa

# Initialize the FlashAttention module with desired configuration
flash_attention = jfa.FlashAttention(
 jfa.AttentionConfig(
  platform=jfa.Platform.TRITON,  # Options: TRITON, PALLAS, JAX
  backend=jfa.Backend.GPU,       # Options: GPU, TPU, CPU
 )
)

# Create sample inputs
batch_size, num_heads, seq_len, head_dim = 2, 4, 512, 64
query = jax.random.normal(jax.random.PRNGKey(0), (batch_size, num_heads * 4, seq_len, head_dim), "f2")
key = jax.random.normal(jax.random.PRNGKey(1), (batch_size, num_heads, seq_len, head_dim), "f2")
value = jax.random.normal(jax.random.PRNGKey(2), (batch_size, num_heads, seq_len, head_dim), "f2")

# Compute attention
output = flash_attention(
 query=query,
 key=key,
 value=value,
 causal=True  # Enable causal masking for decoder-only models
)

# output shape: (batch_size, num_heads, seq_len, head_dim)
```

## Advanced Usage

### With Attention Mask

```python
# Create an attention mask (1 = attend, 0 = mask)
attention_mask = jnp.ones((batch_size, 1, seq_len, seq_len))  # Allow full attention
# For example, mask the first 100 tokens from attending to the last 100 tokens
attention_mask = attention_mask.at[:, :, :100, -100:].set(0)

output = flash_attention(
 query=query,
 key=key,
 value=value,
 attention_mask=attention_mask,
 causal=False  # Using explicit mask instead of causal
)
```

### With Attention Bias

```python
# Create an attention bias
bias = jnp.zeros((batch_size, 1, seq_len, seq_len))
# Add position-dependent bias
for i in range(seq_len):
 for j in range(seq_len):
  bias = bias.at[:, :, i, j].set(1.0 / (1.0 + abs(i - j)))

output = flash_attention(
 query=query,
 key=key,
 value=value,
 bias=bias
)
```

### With Dropout

```python
output = flash_attention(
 query=query,
 key=key,
 value=value,
 dropout_prob=0.1,
 dropout_seed=42,
 causal=True
)
```

## Flax Modules with JFA2

Here's an example of integrating jax-flash-attn2 within a Transformer model implemented in Flax:

```python
import typing as tp
from functools import partial

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp

import jax_flash_attn2 as jfa


class JFAttention2(nn.Module):
 def __init__(
  self,
  hidden_size: int,
  head_dim: int,
  num_attention_heads: int,
  num_key_value_heads: int,
  dtype: jnp.dtype = jnp.float32,
  param_dtype: jnp.dtype = jnp.float32,
  precision: jax.lax.PrecisionLike = None,
  *,
  rngs: nn.Rngs = None,
 ):
  if rngs is None:
   rngs = nn.Rngs(0)
  self.dtype = dtype
  self.param_dtype = param_dtype
  self.precision = precision
  self.rngs = rngs

  self.hidden_size = hidden_size
  self.head_dim = head_dim
  self.num_attention_heads = num_attention_heads
  self.num_key_value_heads = num_key_value_heads

  self.num_key_value_groups = num_attention_heads // num_key_value_heads

  if self.num_key_value_groups == 1:
   assert num_attention_heads == num_key_value_heads

  linear_class = partial(
   nn.Linear,
   dtype=dtype,
   param_dtype=param_dtype,
   use_bias=False,
   kernel_init=jax.nn.initializers.normal(0.02),
   precision=precision,
   rngs=rngs,
  )
  self.q_proj = linear_class(hidden_size, num_attention_heads * self.head_dim)
  self.k_proj = linear_class(hidden_size, num_key_value_heads * self.head_dim)
  self.v_proj = linear_class(hidden_size, num_key_value_heads * self.head_dim)
  self.o_proj = linear_class(num_attention_heads * self.head_dim, hidden_size)

  config = jfa.AttentionConfig(platform=jfa.Platform.TRITON, backend=jfa.Backend.GPU)

  self.jfa2 = jfa.FlashAttention(config)

 def __call__(
  self,
  hidden_states: chex.Array,
  attention_mask: chex.Array,
  causal: bool = True,
 ) -> tp.Tuple[chex.Array, chex.Array]:
  batch_size, sequence_length = hidden_states.shape[:2]
  query_states, key_states, value_states = (
   self.q_proj(hidden_states),
   self.k_proj(hidden_states),
   self.v_proj(hidden_states),
  )
  qshape = (
   batch_size,
   sequence_length,
   self.num_attention_heads,
   self.head_dim,
  )
  kv_shape = (
   batch_size,
   sequence_length,
   self.num_key_value_heads,
   self.head_dim,
  )
  query_states = query_states.reshape(qshape)
  key_states = key_states.reshape(kv_shape)
  value_states = value_states.reshape(kv_shape)
  attn_output = self.jfa2.forward(
   query_states.astype(jnp.bfloat16),
   key_states.astype(jnp.bfloat16),
   value_states.astype(jnp.bfloat16),
   jnp.where(attention_mask, 0, jnp.finfo(query_states).min).astype(jnp.bfloat16),
   causal=causal,
  )
  attn_output = jnp.reshape(attn_output, (batch_size, sequence_length, -1))
  attn_output = self.o_proj(attn_output)
  return attn_output
```

## Platform-Specific Examples

### Using JAX Backend

```python
jax_flash_attn = jfa.FlashAttention(
 jfa.AttentionConfig(
  platform=jfa.Platform.JAX,
  backend=jfa.Backend.CPU,  # Works on any hardware
 )
)

output = jax_flash_attn(query, key, value)
```

### Using Pallas for TPU

```python
tpu_flash_attn = jfa.FlashAttention(
 jfa.AttentionConfig(
  platform=jfa.Platform.PALLAS,
  backend=jfa.Backend.TPU,
 )
)

output = tpu_flash_attn(query, key, value)
```

## Integration with JAX Transformations

```python
@jax.jit
def attention_forward(q, k, v, mask=None):
 return flash_attention(
  query=q,
  key=k,
  value=v,
  attention_mask=mask,
  causal=True
 )

# JIT-compiled function
fast_attention = attention_forward(query, key, value)

# With gradient computation
def loss_fn(q, k, v):
 attn_output = flash_attention(q, k, v, causal=True)
 return jnp.mean(attn_output)

grads = jax.grad(loss_fn)(query, key, value)
```

## Limitations

- Triton platform is only available on NVIDIA GPUs.
- Some platform-backend combinations are not supported (see table above).
- Custom attention masks are not yet supported (use bias instead).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{jax_flash_attn2,
    title = {JAX Flash Attention 2.0},
    year = {2024},
    url = {https://github.com/erfanzar/jax-flash-attn2}
}
```

### refrence citations

```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
@inproceedings{dao2023flashattention2,
  title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Acknowledgments And Refrences

1. All of kernels are copied from [`EasyDeL`](https://github.com/erfanzar/Easydel)
