# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import typing as tp

import jax
import jax.numpy as jnp


def basic_attention_refrence(
	q: jnp.ndarray,
	k: jnp.ndarray,
	v: jnp.ndarray,
	attn_bias: tp.Optional[jnp.ndarray] = None,
	query_padding_mask: tp.Optional[jnp.ndarray] = None,
	key_padding_mask: tp.Optional[jnp.ndarray] = None,
	dropout_prob: float = 0.0,
	dropout_key: tp.Optional[jax.random.PRNGKey] = None,
	window_size: tp.Tuple[int, int] = (-1, -1),
	causal: bool = False,
	softcap: float = 0.0,
):
	if causal:
		window_size = (window_size[0], 0)
	dtype_og = q.dtype
	q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
	QSeq, KSeq = q.shape[1], k.shape[1]
	repeats = q.shape[2] // k.shape[2]
	if repeats > 1:
		k = jnp.repeat(k, repeats=repeats, axis=2)
		v = jnp.repeat(v, repeats=repeats, axis=2)
	d = q.shape[-1]
	q_scaled = q / math.sqrt(d)
	scores = jnp.einsum("bthd,bshd->bhts", q_scaled, k)
	if softcap > 0:
		scores = scores / softcap
		scores = jnp.tanh(scores)
		scores = scores * softcap
	if key_padding_mask is not None:
		key_mask = (~key_padding_mask).reshape(key_padding_mask.shape[0], 1, 1, KSeq)
		scores = jnp.where(key_mask, jnp.finfo(scores.dtype).min, scores)
	if window_size[0] >= 0 or window_size[1] >= 0:
		row_idx = jnp.arange(QSeq).reshape(-1, 1)
		col_idx = jnp.arange(KSeq)
		if key_padding_mask is None:
			sk = KSeq
		else:
			sk = jnp.sum(key_padding_mask, axis=-1).reshape(-1, 1, 1, 1)
		if query_padding_mask is None:
			sq = QSeq
		else:
			sq = jnp.sum(query_padding_mask, axis=-1).reshape(-1, 1, 1, 1)
		if window_size[0] < 0:
			local_mask = col_idx > row_idx + sk - sq + window_size[1]
		else:
			if key_padding_mask is None:
				sk_full = jnp.full_like(col_idx, KSeq)
			else:
				sk_full = sk
			local_mask = jnp.logical_or(
				col_idx > jnp.minimum(row_idx + sk - sq + window_size[1], sk_full),
				col_idx < row_idx + sk - sq - window_size[0],
			)
		scores = jnp.where(local_mask, jnp.finfo(scores.dtype).min, scores)
	if attn_bias is not None:
		scores = scores + attn_bias
	attention = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
	if window_size[0] >= 0 or window_size[1] >= 0:
		all_masked = jnp.all(local_mask, axis=-1, keepdims=True)
		attention = jnp.where(all_masked, 0.0, attention)
	if query_padding_mask is not None:
		query_mask = (~query_padding_mask).reshape(query_padding_mask.shape[0], 1, QSeq, 1)
		attention = jnp.where(query_mask, 0.0, attention)
	dropout_scaling = 1.0 / (1 - dropout_prob)
	if dropout_prob > 0 and dropout_key is not None:
		dropout_mask = jax.random.bernoulli(
			dropout_key, p=1 - dropout_prob, shape=attention.shape
		)
		attention_drop = attention * dropout_mask * dropout_scaling
	else:
		attention_drop = attention
	output = jnp.einsum("bhts,bshd->bthd", attention_drop, v)
	if query_padding_mask is not None:
		query_mask_expanded = (~query_padding_mask).reshape(
			query_padding_mask.shape[0],
			QSeq,
			1,
			1,
		)
		output = jnp.where(query_mask_expanded, 0.0, output)
	return output.astype(dtype_og)
