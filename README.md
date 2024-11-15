# jax-flash-attn2

A flexible and efficient implementation of Flash Attention 2.0 for JAX, supporting multiple backends (GPU/TPU/CPU) and platforms (Triton/Pallas/JAX).

## Features

- 🚀 Multiple backend support: GPU, TPU, and CPU
- 🔧 Multiple platform implementations: Triton, Pallas, and JAX
- ⚡ Efficient caching of attention instances
- 🔄 Support for Grouped Query Attention (GQA) and headdims up to 256.
- 📊 JAX sharding-friendly implementation
- 🎯 Automatic platform selection based on backend
- 🧩 Compatible with existing JAX mesh patterns


## Installation

```bash
pip install jax-flash-attn2
```

## Quick Start

```python
from jax_flash_attn2 import get_cached_flash_attention

# Get a cached attention instance
attention = get_cached_flash_attention(
	backend="gpu", # 'gpu', 'tpu', or 'cpu'
	platform="triton", # 'triton', 'pallas', or 'jax'
	blocksize_q=64, # BLOCK SIZE Q
	blocksize_k=128, # BLOCK SIZE K
	softmax_scale=headdim ** -0.5 # Optional scaling factor
)

# Use with your tensors
outputs = attention(
	query=query_states,
	key=key_states,
	value=value_states,
	bias=attention_bias, # Optional
)
```

## Usage with JAX Sharding

```python
with mesh:
	attention_outputs = get_cached_flash_attention(
		backend="gpu",
		platform="triton",
		blocksize_q=128,
		blocksize_k=128,
		softmax_scale=None,
	)(
		query=with_sharding_constraint(query_states, qps).astype(dtype),
		key=with_sharding_constraint(key_states, kps).astype(dtype),
		value=with_sharding_constraint(value_states, vps).astype(dtype),
		bias=with_sharding_constraint(bias, bps).astype(dtype),
	)
```
## 📊 Benchmarks

- [Triton GPU (MHA/GQA) vs JAX SDPA CUDNN](https://github.com/erfanzar/jax-flash-attn2/tree/main/benchmarks/triton-vs-jax-sdpa-cudnn)
- [Triton GPU (MHA/GQA) vs JAX SDPA](https://github.com/erfanzar/jax-flash-attn2/tree/main/benchmarks/triton-vs-jax-sdpa)
- Pallas GPU MHA (comming soon...)
- Pallas TPU MHA (comming soon...)
- XLA CPU MHA (comming soon...)
## Supported Configurations

### Backends
- `gpu`: CUDA/AMD-capable GPUs
- `tpu`: Google Cloud TPUs
- `cpu`: CPU fallback

### Platforms
- `triton`: Optimized for NVIDIA/AMD GPUs
- `pallas`: Optimized for TPUs and supported on GPUs
- `jax`: Universal fallback, supports all backends

### Valid Backend-Platform Combinations

| Backend          | Supported Platforms |
| ---------------- | ------------------- |
| GPU - AMD/NVIDIA | Triton, JAX         |
| GPU - NVIDIA     | Triton, Pallas, JAX |
| TPU              | Pallas, JAX         |
| CPU              | JAX                 |

## Advanced Configuration

### Custom Block Sizes

```python
attention = get_cached_flash_attention(
    backend="gpu",
    platform="triton",
    blocksize_q=128,    # Customize query block size # Ignored for Triton
    blocksize_k=128,    # Customize key block size Ignored for Triton
    softmax_scale=1.0,  # Custom softmax scaling
)
```

### Environment Variables

- `"GPU_IDX_FLASH_ATTN"` to define GPU INDEX force for computing triton attention
- `"CACHE_TRITON_KERNELS"` whenever to cache triton kernels (`defualt true`)
- `"_JAX_TRITON_DUMP_DIR"` path to save triton kernels
- `"BLOCKSIZE_M_FLASH_ATTN"` block size q seq length for backward
- `"BLOCKSIZE_N_FLASH_ATTN"` block size kv seq length for backward

## Performance Tips
 
1. **Platform Selection**:
   - For NVIDIA GPUs: prefer `triton`
   - For TPUs: prefer `pallas`
   - For CPU or fallback: use `jax`

2. **Caching**: The `get_cached_flash_attention` function automatically caches instances based on parameters. No need to manage caching manually.

## Requirements

- JAX
- einops
- chex
- jax.experimental.pallas (for TPU support)
- triton (for GPU optimized implementation)

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

1. This implementation (MHA) is based on:
- [Flash Attention 2.0 paper](https://arxiv.org/abs/2205.14135)
- JAX ecosystem tools and libraries
- Triton and Pallas optimization frameworks

2. Custom Triton Uses [`JAX-Triton`](https://github.com/jax-ml/jax-triton/)

3. All of kernels are copied from [`EasyDeL`](https://github.com/erfanzar/Easydel)