#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def bgmv_shrink(inputs: torch.Tensor,
                lora_a_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                scaling: float = 1.0):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def bgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = True):
    return torch.ops._C_ascend.bgmv_expand(inputs, lora_b_weights,
                                           lora_indices_tensor, output_tensor,
                                           slice_offset, slice_size)


def bgmv_fused(inputs: torch.Tensor,
               lora_a_weights: torch.Tensor,
               lora_b_weights: torch.Tensor,
               output_tensor: torch.Tensor,
               lora_indices_tensor: torch.Tensor,
               scaling: float = 1.0):
    """
    Fused bgmv operation combining shrink and expand phases.
    
    This function performs the following computation in a single kernel:
        buffer = (inputs @ lora_a_weights) * scaling  (shrink phase)
        output_tensor += buffer @ lora_b_weights      (expand phase)
    
    The intermediate buffer stays in L2 cache, avoiding global memory writes/reads.
    
    Args:
        inputs: Input tensor of shape [batch_size, input_hidden_dim]
        lora_a_weights: LoRA A weights of shape [num_loras, input_hidden_dim, max_lora_rank]
        lora_b_weights: LoRA B weights of shape [num_loras, max_lora_rank, output_hidden_dim]
        output_tensor: Output tensor of shape [batch_size, output_hidden_dim] (modified in-place)
        lora_indices_tensor: LoRA indices of shape [batch_size]
        scaling: Scaling factor for the operation
    
    Returns:
        None (output_tensor is modified in-place)
    """
    return torch.ops._C_ascend.bgmv_fused(
        inputs,
        lora_a_weights,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return torch.ops._C_ascend.sgmv_shrink(inputs, lora_a_weights,
                                           lora_indices_tensor, seq_len_tensor,
                                           output_tensor, scaling)


def sgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                b_seq_start_loc: torch.Tensor,
                seq_len_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                batches: int,
                max_seq_length: int,
                token_nums: int,
                add_inputs: bool = False):
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        0,
        output_tensor.size(1),
    )


def sgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      b_seq_start_loc: torch.Tensor,
                      seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      batches: int,
                      max_seq_length: int,
                      token_nums: int,
                      slice_offset: int,
                      slice_size: int,
                      add_inputs: bool = False):
    return torch.ops._C_ascend.sgmv_expand(inputs, lora_b_weights,
                                           lora_indices_tensor, seq_len_tensor,
                                           output_tensor, slice_offset,
                                           slice_size)
