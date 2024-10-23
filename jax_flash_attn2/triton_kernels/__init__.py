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

from jax_flash_attn2.triton_kernels.gqa_kernel import triton_flash_gqa_attn_2_gpu
from jax_flash_attn2.triton_kernels.mha_kernel import triton_flash_mha_attn_2_gpu

__all__ = ["triton_flash_gqa_attn_2_gpu", "triton_flash_mha_attn_2_gpu"]