import functools
import jax
import jaxlib
import triton
from jax import nn
from jax import numpy as jnp
from jax import random as jrnd

from jax_flash_attn2 import get_cached_flash_attention

benchmark_configs = []
for mode in ["fwd"]:
	for batch_size in [1, 2, 4]:
		for bias in [True, False]:
			for headdim in [64, 128, 256]:
				for num_heads in [8, 16]:
					benchmark_configs.append(
						triton.testing.Benchmark(
							x_names=["seqlen"],
							x_vals=[1024, 2048, 4096, 6144, 8192],
							line_arg="provider",
							line_vals=["triton", "jax"],
							line_names=["Triton", "Jax-cudnn"],
							styles=[("green", "-"), ("blue", ":")],
							ylabel="MS",
							plot_name=f"b={batch_size}-ub={bias}-hdim={headdim}-qh={num_heads*2}-kvh={num_heads}-mode={mode}",
							args={
								"BATCH": batch_size,
								"QH": num_heads * 2,
								"KH": num_heads,
								"HEAD_DIM": headdim,
								"mode": mode,
								"BIAS": bias,
							},
						)
					)


@triton.testing.perf_report(benchmark_configs)
def mha_attention_benchmark(
	seqlen,
	QH,
	KH,
	BATCH,
	HEAD_DIM,
	mode,
	BIAS,
	provider,
):
	try:
		q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
		query = jax.nn.initializers.normal(2)(
			q_key, (BATCH, seqlen, QH, HEAD_DIM), dtype=jnp.float16
		)
		key = jax.nn.initializers.normal(2)(
			k_key, (BATCH, seqlen, KH, HEAD_DIM), dtype=jnp.float16
		)
		value = jax.nn.initializers.normal(2)(
			v_key, (BATCH, seqlen, KH, HEAD_DIM), dtype=jnp.float16
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
			if provider == "triton":
				flash_attn = get_cached_flash_attention()
				fn = lambda: flash_attn(query, key, value, bias)
			elif provider == "jax":
				_fn = jax.jit(
					functools.partial(nn.dot_product_attention, implementation="cudnn")
				)
				fn = lambda: _fn(query, key, value, bias).block_until_ready()
		elif mode == "bwd":
			if provider == "triton":
				flash_attn = get_cached_flash_attention()
				fn = lambda: jax.grad(lambda *x: flash_attn(*x).sum())(query, key, value, bias)
			elif provider == "jax":
				_fn = jax.jit(
					functools.partial(nn.dot_product_attention, implementation="cudnn")
				)
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
		save_path="benchmarks/triton-vs-jax-sdpa-cudnn",
	)
