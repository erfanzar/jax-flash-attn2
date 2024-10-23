import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

import jax
from jax import numpy as jnp
from jax import random as jrnd

from jax_flash_attn2 import create_flash_attention

USE_BIAS = True


def _gqa_attn_refrence(query_states, key_states, value_states, bias):
	b, qs, num_q_heads, d = query_states.shape
	num_kv_heads = value_states.shape[2]
	ks = value_states.shape[1]
	query_states = jnp.reshape(
		query_states,
		(b, qs, num_kv_heads, num_q_heads // num_kv_heads, d),
	)

	query_states = query_states * (d**-0.5)
	attention_weight = jnp.einsum(
		"bskhd,bmkd->bkhsm",
		query_states,
		key_states,
	)

	if bias is not None:
		if bias.shape[1] == num_q_heads:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, num_kv_heads, num_q_heads // num_kv_heads, qs, ks),
			)
		elif bias.shape[1] == num_kv_heads:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, num_kv_heads, 1, qs, ks),
			)
		elif bias.shape[1] == 1:
			attention_weight = jnp.add(
				attention_weight,
				bias.reshape(b, 1, 1, qs, ks),
			)
		else:
			raise NotImplementedError("bias heads wont match!")

	attention_weight = jax.nn.softmax(attention_weight)

	return jnp.einsum("bkhsm,bmkd->bskhd", attention_weight, value_states).reshape(
		b,
		qs,
		num_q_heads,
		d,
	)


def _mha_attn_refrence(query_states, key_states, value_states, bias):
	d = query_states.shape[-1]

	attention_weight = jnp.einsum("bqhd,bkhd->bhqk", query_states * (d**-0.5), key_states)

	if bias is not None:
		attention_weight = jnp.add(attention_weight, bias)
	attention_weight = jax.nn.softmax(attention_weight)

	return jnp.einsum("bhqk,bkhd->bqhd", attention_weight, value_states)


flash_attn = create_flash_attention(
	backend="gpu",
	platform="triton",
	blocksize_q=64,
	blocksize_k=64,
	softmax_scale=None,
)


def test_forward():
	"""Tests the forward pass of the attention mechanism."""
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, QH, KVH, QS, KS, D = 1, 32, 8, 1024, 1024, 128
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KVH, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KVH, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if USE_BIAS
		else None
	)
	print("QKV Allocated")
	co = flash_attn(q, k, v, b)  # passes 256K on 24G GPU 3090
	print(co[-1, -1, -1, :5])
	fo = _gqa_attn_refrence(q, k, v, b)
	print(fo[-1, -1, -1, :5])
	print("Results are Close" if jnp.allclose(co, fo, 0, 0.125) else "Wrong results!")


def test_backward():
	"""Tests the backward pass of the attention mechanism.""" 
	
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	B, QH, KVH, QS, KS, D = 1, 32, 32, 1024, 1024, 128
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KVH, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KVH, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if USE_BIAS
		else None
	)

	try:
		co = jax.grad(lambda *x: flash_attn(*x).sum())(q, k, v, b)
		print("Custom op backward pass gradients:")
		print(co[-1, -1, -1, :5])  # Print last 5 elements of last head of last batch
	except Exception as er:
		print(f"Custom op backward pass failed: {er}")
		co = None

	try:
		fo = jax.grad(lambda *x: _mha_attn_refrence(*x).sum())(q, k, v, b)

		print(fo[-1, -1, -1, :5])
	except Exception as e:
		print(f"Flax backward pass failed : {e}")
		fo = None
		exit()

	if fo is not None and co is not None:
		if jnp.allclose(co, fo, atol=0.125):
			print("Backward pass results are close.")
		else:
			print("Backward pass results differ significantly!")


if __name__ == "__main__":
	test_forward()
	test_backward()
