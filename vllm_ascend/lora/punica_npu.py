# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional, Tuple, Union

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    # Maximum number of slices for LoRA operations (QKV has 3 slices)
    MAX_LORA_SLICES = 3

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)
        refresh_all_lora_classes()
        self.lora_config = kwargs.get("lora_config")
        if get_ascend_device_type() == AscendDeviceType._310P or (
                self.lora_config is not None
                and self.lora_config.max_lora_rank >= 128):
            from vllm.lora.ops.torch_ops import (bgmv_expand,
                                                 bgmv_expand_slice,
                                                 bgmv_shrink, sgmv_expand,
                                                 sgmv_expand_slice,
                                                 sgmv_shrink)
        else:
            from vllm_ascend.lora.lora_ops import (bgmv_expand,
                                                   bgmv_expand_slice,
                                                   bgmv_shrink, sgmv_expand,
                                                   sgmv_expand_slice,
                                                   sgmv_shrink)
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink

        # Pre-allocate buffers for LoRA operations to avoid repeated memory
        # allocation. The buffers are allocated based on max_num_batched_tokens
        # and max_lora_rank to accommodate the largest possible batch size.
        self._max_num_batched_tokens = max_num_batched_tokens
        self._init_lora_buffers(device)

    def _init_lora_buffers(self, device: Union[torch.device, str]) -> None:
        """
        Pre-allocate buffers for LoRA operations to avoid repeated memory
        allocation during inference.

        The buffers are allocated based on:
        - max_num_batched_tokens: maximum number of tokens in a batch
        - max_lora_rank: maximum LoRA rank from lora_config

        These buffers will be reused across multiple calls to add_lora_linear
        and add_lora_logits, reducing memory allocation overhead.
        """
        if self.lora_config is None:
            # No LoRA config provided, skip buffer allocation
            self._lora_buffers: Optional[Tuple[torch.Tensor, ...]] = None
            self._logits_buffer: Optional[torch.Tensor] = None
            return

        max_lora_rank = self.lora_config.max_lora_rank

        # Pre-allocate buffers for add_lora_linear (up to MAX_LORA_SLICES for
        # QKV operations). We use float32 for intermediate computations to
        # maintain numerical precision, consistent with the triton op.
        self._lora_buffers = tuple(
            torch.zeros(
                (self._max_num_batched_tokens, max_lora_rank),
                dtype=torch.float32,
                device=device
            )
            for _ in range(self.MAX_LORA_SLICES)
        )

        # Pre-allocate buffer for add_lora_logits
        self._logits_buffer = torch.zeros(
            (self._max_num_batched_tokens, max_lora_rank),
            dtype=torch.float32,
            device=device
        )

    def _get_lora_buffers(
        self,
        batch_size: int,
        lora_rank: int,
        num_slices: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get pre-allocated buffers for LoRA linear operations.

        If pre-allocated buffers are available and sufficient, return sliced
        views of them. Otherwise, fall back to creating new buffers.

        Args:
            batch_size: Current batch size (number of tokens)
            lora_rank: LoRA rank for the current operation
            num_slices: Number of slices needed (e.g., 3 for QKV)
            device: Device to create buffers on if needed

        Returns:
            Tuple of buffer tensors, each with shape (batch_size, lora_rank)
        """
        # Check if we can use pre-allocated buffers
        if (self._lora_buffers is not None
                and num_slices <= self.MAX_LORA_SLICES
                and batch_size <= self._max_num_batched_tokens
                and self.lora_config is not None
                and lora_rank <= self.lora_config.max_lora_rank):
            # Use sliced views of pre-allocated buffers
            buffers = tuple(
                self._lora_buffers[i][:batch_size, :lora_rank]
                for i in range(num_slices)
            )
            # Zero out the buffers before use
            for buf in buffers:
                buf.zero_()
            return buffers

        # Fall back to creating new buffers if pre-allocated ones are
        # insufficient
        return tuple(
            torch.zeros(
                (batch_size, lora_rank),
                dtype=torch.float32,
                device=device
            )
            for _ in range(num_slices)
        )

    def _get_logits_buffer(
        self,
        batch_size: int,
        lora_rank: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get pre-allocated buffer for LoRA logits operations.

        If pre-allocated buffer is available and sufficient, return a sliced
        view of it. Otherwise, fall back to creating a new buffer.

        Args:
            batch_size: Current batch size (number of tokens)
            lora_rank: LoRA rank for the current operation
            device: Device to create buffer on if needed

        Returns:
            Buffer tensor with shape (batch_size, lora_rank)
        """
        # Check if we can use pre-allocated buffer
        if (self._logits_buffer is not None
                and batch_size <= self._max_num_batched_tokens
                and self.lora_config is not None
                and lora_rank <= self.lora_config.max_lora_rank):
            # Use sliced view of pre-allocated buffer
            buffer = self._logits_buffer[:batch_size, :lora_rank]
            # Zero out the buffer before use
            buffer.zero_()
            return buffer

        # Fall back to creating a new buffer
        return torch.zeros(
            (batch_size, lora_rank),
            dtype=torch.float32,
            device=device
        )

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        self.bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        self.bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        self.bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices,
                               y_offset, y_slice_size, add_inputs)

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all`
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = (self._expand_slice_prefill
                                      if self.is_prefill else
                                      self._expand_slice_decode)
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_expand_fused(
        self,
        y: torch.Tensor,
        x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        output_slices: Tuple[int, ...],
        offset_start: int,
        add_inputs: bool,
        num_slices: int
    ):
        """
        Fused expand operation for multiple slices.

        This method optimizes multiple expand operations by:
        1. Pre-selecting the expand function once (prefill vs decode)
        2. Pre-computing all offsets
        3. Processing all slices with minimal Python overhead

        Args:
            y: Output tensor with shape (batch_size, hidden_out)
            x: Tuple of input tensors, each with shape (batch_size, lora_rank)
            lora_b_stacked: Tuple of LoRA B weights
            output_slices: Tuple of slice sizes for each output
            offset_start: Starting offset in the output tensor
            add_inputs: Whether to add to existing values
            num_slices: Number of slices to process
        """
        # Select the expand function once to avoid repeated branch prediction
        expand_slice_fun: Callable = (self._expand_slice_prefill
                                      if self.is_prefill else
                                      self._expand_slice_decode)

        # Pre-compute offsets to avoid repeated additions in the loop
        offsets = [offset_start]
        for i in range(num_slices - 1):
            offsets.append(offsets[i] + output_slices[i])

        # Process all slices with the pre-selected function
        for slice_idx in range(num_slices):
            expand_slice_fun(
                y, x[slice_idx], lora_b_stacked[slice_idx],
                offsets[slice_idx], output_slices[slice_idx], add_inputs
            )

    def _apply_shrink(self, y: torch.Tensor, x: torch.Tensor,
                      w_t_all: torch.Tensor, scale: float):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = (self._shrink_prefill
                                if self.is_prefill else self._shrink_decode)
        shrink_fun(y, x, w_t_all, scale)
        y = y.view_as(y_org)

    def _apply_shrink_fused(
        self,
        y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        scale: float,
        num_slices: int
    ):
        """
        Fused shrink operation for multiple slices with the same LoRA rank.

        This method optimizes multiple shrink operations by:
        1. Pre-selecting the shrink function once (prefill vs decode)
        2. Processing all slices with minimal Python overhead
        3. Reusing the reshaped input tensor

        Args:
            y: Tuple of output tensors, each with shape (batch_size, lora_rank)
            x: Input tensor with shape (batch_size, hidden_in)
            lora_a_stacked: Tuple of LoRA A weights
            scale: Scaling factor
            num_slices: Number of slices to process
        """
        # Select the shrink function once to avoid repeated branch prediction
        shrink_fun: Callable = (self._shrink_prefill
                                if self.is_prefill else self._shrink_decode)

        # Process all slices
        for slice_idx in range(num_slices):
            y_slice = y[slice_idx]
            y_org = y_slice
            y_flat = y_slice.view(-1, y_slice.shape[-1])
            shrink_fun(y_flat, x, lora_a_stacked[slice_idx], scale)
            y_slice = y_flat.view_as(y_org)

    def add_shrink(self, y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   x: torch.Tensor, lora_a_stacked: Tuple[torch.Tensor, ...],
                   scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        num_slices = len(lora_a_stacked)

        # Optimization: single slice case - avoid loop overhead
        if num_slices == 1:
            self._apply_shrink(y[0], x, lora_a_stacked[0], scale)
            return

        # For multiple slices, try to batch process when possible
        # Check if we can use fused shrink (all slices have same rank)
        first_rank = lora_a_stacked[0].size(-2)
        can_fuse = all(w.size(-2) == first_rank for w in lora_a_stacked)

        if can_fuse and num_slices <= self.MAX_LORA_SLICES:
            # Use fused shrink for better performance
            self._apply_shrink_fused(y, x, lora_a_stacked, scale, num_slices)
        else:
            # Fall back to sequential processing
            for slice_idx in range(num_slices):
                self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx],
                                   scale)

    def add_expand(self,
                   y: torch.Tensor,
                   x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   lora_b_stacked: Tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                   output_slices: Tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] +
                    lora_bias_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]):
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        num_slices = len(lora_b_stacked)

        # Apply bias if present
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices,
                             lora_bias_stacked)

        # Optimization: single slice case - avoid loop overhead
        if num_slices == 1:
            self._apply_expand(
                y, x[0], lora_b_stacked[0],
                offset_start, output_slices[0], add_inputs
            )
            y = y.view_as(y_org)
            return

        # Multiple slices: use fused expand for better performance
        self._apply_expand_fused(
            y, x, lora_b_stacked, output_slices,
            offset_start, add_inputs, num_slices
        )
        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only need expand op
        expand_fun: Callable = (self._expand_prefill
                                if self.is_prefill else self._expand_decode)
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: Tuple[torch.Tensor, ...],
                        lora_b_stacked: Tuple[torch.Tensor, ...],
                        scale: float,
                        output_slices: Tuple[int, ...],
                        *,
                        buffer: Optional[Tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
                If None, pre-allocated buffers will be used when available.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is None:
            # Get LoRA rank from lora_b weights
            lora_rank = lora_b_stacked[0].size(-1)
            batch_size = x.size(0)
            num_slices = len(output_slices)
            # Use pre-allocated buffers to avoid repeated memory allocation
            buffer = self._get_lora_buffers(
                batch_size, lora_rank, num_slices, x.device
            )
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y,
                        buffer,
                        lora_b_stacked,
                        None,
                        output_slices,
                        add_inputs=True,
                        **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
                If None, pre-allocated buffer will be used when available.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        lora_rank = lora_b_stacked.size(-1)
        batch_size = x.size(0)

        if buffer is None:
            # Use pre-allocated buffer to avoid repeated memory allocation
            buffer = self._get_logits_buffer(batch_size, lora_rank, x.device)

        indices = self.sampler_indices

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)
