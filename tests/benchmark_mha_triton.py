import os

import jax
import jaxlib
import triton
from jax import nn
from jax import numpy as jnp
from jax import random as jrnd

from jax_flash_attn2 import get_cached_flash_attention

benchmark_configs = []
for mode in ["bwd", "fwd"]:
	for batch_size in [1, 2, 4]:
		for bias in [True, False]:
			for headdim in [64, 128, 256]:
				for num_heads in [8, 16, 32]:
					for blocksize_q in [32, 64, 128]:
						for blocksize_k in [32, 64, 128]:
							benchmark_configs.append(
								triton.testing.Benchmark(
									x_names=["seqlen"],
									x_vals=[1024, 2048, 4096, 6144, 8192],
									line_arg="provider",
									line_vals=["triton-block-ptr", "triton-ptr-block", "jax"],
									line_names=["Triton-BlockPtr", "Triton-PtrBlock", "Jax"],
									styles=[("green", "-"), ("blue", "-."), ("blue", ":")],
									ylabel="MS",
									plot_name=f"batch_size={batch_size}-bias={bias}-headdim={headdim}-num_heads={num_heads}-blocksize_q={blocksize_q}-blocksize_k={blocksize_k}-mode={mode}",
									args={
										"BATCH": batch_size,
										"H": num_heads,
										"HEAD_DIM": headdim,
										"mode": mode,
										"BIAS": bias,
										"blocksize_k": blocksize_k,
										"blocksize_q": blocksize_q,
									},
								)
							)


@triton.testing.perf_report(benchmark_configs)
def mha_attention_benchmark(
	seqlen,
	H,
	BATCH,
	HEAD_DIM,
	mode,
	BIAS,
	blocksize_k,
	blocksize_q,
	provider,
):
	try:
		q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
		query = jax.nn.initializers.normal(2)(
			q_key, (BATCH, seqlen, H, HEAD_DIM), dtype=jnp.float16
		)
		key = jax.nn.initializers.normal(2)(
			k_key, (BATCH, seqlen, H, HEAD_DIM), dtype=jnp.float16
		)
		value = jax.nn.initializers.normal(2)(
			v_key, (BATCH, seqlen, H, HEAD_DIM), dtype=jnp.float16
		)
		bias = (
			jnp.where(
				jrnd.randint(v_key, (BATCH, 1, seqlen, seqlen), 0, 4) > 2,
				jnp.finfo(jnp.float16).min,
				0,
			)
			if BIAS
			else None
		)
		if mode == "fwd":
			if provider == "triton-block-ptr":
				os.environ["FLASH_ATTN_BLOCK_PTR"] = "1"
				flash_attn = get_cached_flash_attention(
					blocksize_k=blocksize_k,
					blocksize_q=blocksize_q,
				)
				fn = lambda: flash_attn(query, key, value, bias)
			elif provider == "triton-ptr-block":
				os.environ["FLASH_ATTN_BLOCK_PTR"] = "0"
				flash_attn = get_cached_flash_attention(
					blocksize_k=blocksize_k,
					blocksize_q=blocksize_q,
				)
				fn = lambda: flash_attn(query, key, value, bias)
			elif provider == "jax":
				_fn = jax.jit(nn.dot_product_attention)
				fn = lambda: _fn(query, key, value, bias).block_until_ready()
		elif mode == "bwd":
			if provider == "triton-block-ptr":
				os.environ["FLASH_ATTN_BLOCK_PTR"] = "1"
				flash_attn = get_cached_flash_attention(
					blocksize_k=blocksize_k,
					blocksize_q=blocksize_q,
				)
				fn = lambda: jax.grad(lambda *x: flash_attn(*x).sum())(query, key, value, bias)
			elif provider == "triton-ptr-block":
				os.environ["FLASH_ATTN_BLOCK_PTR"] = "0"
				flash_attn = get_cached_flash_attention(
					blocksize_k=blocksize_k,
					blocksize_q=blocksize_q,
				)
				fn = lambda: jax.grad(lambda *x: flash_attn(*x).sum())(query, key, value, bias)
			elif provider == "jax":
				_fn = jax.jit(nn.dot_product_attention)
				fn = lambda: jax.grad(lambda *x: _fn(*x).sum())(
					query, key, value, bias
				).block_until_ready()
		try:
			ms = triton.testing.do_bench(fn)
		except jaxlib.xla_extension.XlaRuntimeError:
			ms = 100.0000
		return ms
	except:  # noqa
		return 300.0000


if __name__ == "__main__":
	mha_attention_benchmark.run(
		print_data=True,
		save_path="jax-flash-attn2/benchmarks/mha/triton",
	)
