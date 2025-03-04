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

from .flash_attention import (
	AttentionConfig,
	Backend,
	FlashAttention,
	Platform,
	create_flash_attention,
)
from .flash_attention_jax import jax_flash_attention
from .flash_attention_triton import triton_flash_attention
from .refrence_call import basic_attention_refrence

__all__ = (
	"AttentionConfig",
	"Backend",
	"FlashAttention",
	"Platform",
	"create_flash_attention",
	"triton_flash_attention",
	"jax_flash_attention",
	"basic_attention_refrence",
)

__version__ = "0.0.3"
