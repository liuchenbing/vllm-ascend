# LoRA性能优化完整分析

## 目录

1. [背景和概述](#背景和概述)
2. [Fused Kernel设计](#fused-kernel设计)
3. [SGLang优化方案分析](#sglang优化方案分析)
4. [详细对比分析](#详细对比分析)
5. [已实施的优化](#已实施的优化)
6. [进一步优化建议](#进一步优化建议)
7. [总结](#总结)

---

## 背景和概述

### 问题描述

当前vllm-ascend的LoRA实现存在性能瓶颈，与SGLang相比有约2倍的性能差距。主要问题包括：

1. **内存带宽开销**：shrink和expand阶段之间需要将buffer写入全局内存，然后再读取
2. **Kernel launch开销**：多次kernel调用增加了启动开销
3. **Batch准备开销**：每次重新准备batch信息，缺少优化
4. **Kernel选择不一致**：decode阶段使用bgmv，而SGLang统一使用sgmv

### 优化目标

1. 减少内存带宽开销（融合算子，利用L2 cache）
2. 统一kernel选择策略（类似SGLang，统一使用sgmv）
3. 优化batch准备方式（预分配和复用）
4. 优化QKV操作（合并shrink，减少kernel调用）

---

## Fused Kernel设计

### 设计思路

当前LoRA实现分为两个独立的kernel：
1. **shrink阶段**: `x @ lora_a -> buffer` (float32)
2. **expand阶段**: `buffer @ lora_b -> y`

这两个阶段之间需要将buffer写入全局内存，然后再读取，造成额外的内存带宽开销。

### bgmv_fused (Decode阶段融合算子)

**输入**:
- `x`: [batch_size, input_hidden_dim] - 输入tensor
- `lora_a_weights`: [num_loras, input_hidden_dim, max_lora_rank] - LoRA A权重
- `lora_b_weights`: [num_loras, max_lora_rank, output_hidden_dim] - LoRA B权重
- `indices`: [batch_size] - LoRA索引
- `scale`: float - 缩放因子

**输出**:
- `y`: [batch_size, output_hidden_dim] - 输出tensor（in-place修改）

**计算流程**:
```
for each token i in batch:
    lora_idx = indices[i]
    if lora_idx < 0:
        continue  # 跳过无LoRA的token
    
    // Shrink阶段 (在L2 cache中完成)
    buffer[0:max_lora_rank] = 0
    for j in range(max_lora_rank):
        acc = 0
        for k in range(input_hidden_dim):
            acc += x[i, k] * lora_a_weights[lora_idx, k, j]
        buffer[j] = acc * scale
    
    // Expand阶段 (直接在L2 cache中使用buffer，不写回全局内存)
    for j in range(output_hidden_dim):
        acc = 0
        for k in range(max_lora_rank):
            acc += buffer[k] * lora_b_weights[lora_idx, k, j]
        y[i, j] += acc  // 累加到输出
```

**关键优化点**:
- buffer存储在L2 cache中，不写回全局内存
- shrink和expand在同一个kernel中完成，减少kernel launch开销
- 减少一次全局内存读写操作

### sgmv_fused (Prefill阶段融合算子)

类似bgmv_fused，但需要处理变长序列：
- 使用`seq_len`和`lora_indices`来组织数据
- 每个序列可能有不同的LoRA适配器

**预期收益**:
- **内存带宽**: 减少约50%的中间buffer读写
- **Kernel Launch**: 减少50%的kernel launch次数
- **整体性能**: 预期提升10-20%的LoRA throughput

---

## SGLang优化方案分析

### 1. QKV LoRA合并Shrink（核心优化）

**SGLang的做法**：
- `qkv_lora_a`在存储层面就已经合并：`(num_loras, 3*rank, input_dim)`
- **一次shrink**得到合并的buffer：`(batch_size, 3*rank)`
- 然后从buffer中切片，分别对Q/K/V做expand

**关键代码示例**：
```python
# 1. 单次shrink（3个slice合并）
lora_a_output = torch.zeros(total_seq_len, weight_intermediate_dim, ...)
torch.ops.npu.sgmv_shrink(x, qkv_lora_a, ..., lora_a_output, 1.0)

# 2. 外部应用scaling（如果需要在外部应用）
scaling = self.batch_info.scalings.gather(...).repeat_interleave(...)
lora_a_output *= scaling

# 3. 分slice expand
for slice_id in range(3):
    torch.ops.npu.sgmv_expand(lora_a_output[:, slice_range], ...)
```

**优势**：
- 减少kernel launch次数：1次shrink vs 3次shrink
- 对于QKV操作，预期约15-20%的性能提升

**vllm-ascend当前做法**：
- `lora_a_stacked`是tuple，3个分开的tensor：`(num_loras, input_dim, rank) * 3`
- 需要对每个slice分别做shrink（3次kernel调用）
- 已实现合并shrink，在运行时通过`torch.cat`合并

### 2. Scaling外部应用

**SGLang的做法**：
- shrink时传入`scale=1.0`（不在kernel内应用scaling）
- shrink后，在外部通过`gather`和`repeat_interleave`应用scaling

**vllm-ascend当前做法**：
- scaling在kernel内部应用（传入`scale`参数）
- 这可能导致kernel复杂度增加，但对于NPU可能影响较小

### 3. Pinned Memory优化

**SGLang的做法**（`ascend_backend.py:prepare_lora_batch`）：
- 使用`pin_memory=True`创建CPU tensor
- 使用`non_blocking=True`进行异步copy

**优势**：
- 减少host-to-device传输的同步开销
- 提高数据传输效率

**vllm-ascend当前状态**：
- 部分场景使用了pinned memory（如speculative decoding）
- LoRA metadata准备可能需要进一步优化

---

## 详细对比分析

### 1. Kernel实现的差异（sgmv vs bgmv）

#### SGLang的做法

**始终使用sgmv（无论prefill还是decode）**：
```python
torch.ops.npu.sgmv_shrink(
    x,                    # [total_seq_len, input_dim]
    weights,              # [num_loras, input_dim, rank]
    self.batch_info.weight_indices,  # [num_segments] - 每个sequence的LoRA ID
    self.batch_info.seg_lens,        # [num_segments] - 每个sequence的长度
    output_tensor,        # [total_seq_len, rank]
    1.0                   # scale (在外部应用)
)
```

**参数说明**：
- `weight_indices`: 形状`[num_segments]`，每个segment（sequence）对应的LoRA ID
- `seg_lens`: 形状`[num_segments]`，每个segment的token数量
- 对于decode阶段，`seg_lens = [1, 1, 1, ...]`（每个sequence长度为1）

#### vllm-ascend的做法（优化前）

**Prefill阶段使用sgmv，Decode阶段使用bgmv**：

**Prefill (sgmv)**:
```python
self.sgmv_shrink(
    x,                    # [total_seq_len, input_dim]
    w_t_all,              # [num_loras, input_dim, rank]
    y,                    # [total_seq_len, rank]
    *self.prefill_metadata,  # 包含多个参数
    scale
)
```

**Decode (bgmv - 优化前)**:
```python
self.bgmv_shrink(
    x,                    # [batch_size, input_dim]
    w_t_all,              # [num_loras, input_dim, rank]
    y,                    # [batch_size, rank]
    self.token_lora_indices,  # [batch_size] - 每个token的LoRA ID
    scale
)
```

#### 关键性能差异

1. **kernel选择**：
   - SGLang: 统一使用sgmv（可能针对prefill和decode都优化）
   - vllm-ascend: prefill用sgmv，decode用bgmv（decode时可能有额外开销）

2. **参数复杂度**：
   - SGLang: 简单的`weight_indices` + `seg_lens`
   - vllm-ascend: sgmv需要更多元数据参数

### 2. Batch准备方式的差异

#### SGLang的batch准备

```python
def prepare_lora_batch(...):
    # 使用pinned memory + non_blocking copy
    weight_indices_tensor = torch.tensor(
        weight_indices, dtype=torch.int32, pin_memory=True, device="cpu"
    )
    # ...
    # 异步copy到device
    batch_info.weight_indices[:bs].copy_(weight_indices_tensor, non_blocking=True)
```

**batch_info结构**：
- `weight_indices`: `[num_segments]` - 每个sequence的LoRA ID
- `seg_lens`: `[num_segments]` - 每个sequence的token数量
- `scalings`: `[max_loras_per_batch]` - 每个LoRA的scaling factor

**关键优化**：
- Pinned memory：使用`pin_memory=True`
- Non-blocking copy：使用`non_blocking=True`
- 序列级别索引：不是token级别
- Batch信息复用：使用`LoRABatchInfo`对象复用

#### vllm-ascend的batch准备（优化前）

**Prefill阶段** (`prefill_metadata`):
- `lora_indices_tensor`: `[num_sequences]` - 每个sequence的LoRA ID
- `seq_len_tensor`: `[num_sequences]` - 每个sequence的长度

**Decode阶段** (`token_lora_indices`):
- `token_lora_indices`: `[batch_size]` - **每个token的LoRA ID**
- 这是token级别的索引，不是sequence级别

#### 关键差异分析

**1. 索引粒度**：
- **SGLang**: Sequence级别（`seg_lens` + `weight_indices`）
  - 对于decode，虽然每个sequence长度为1，但仍然是sequence级别索引
- **vllm-ascend**: Decode阶段是token级别（`token_lora_indices`）
  - 每个token独立索引，即使它们属于同一sequence

**2. 内存传输优化**：
- **SGLang**: 使用pinned memory和non-blocking copy
- **vllm-ascend**: 部分场景使用，LoRA metadata准备可能需要优化

**3. Batch信息复用**：
- **SGLang**: 使用`LoRABatchInfo`对象，复用batch信息
- **vllm-ascend**: 每次可能重新准备，缺少复用机制

---

## 已实施的优化

### ✅ 优化1: Decode阶段统一使用sgmv

**实现内容**：
- 修改`_shrink_decode`、`_expand_decode`、`_expand_slice_decode`使用sgmv
- 添加`_get_decode_metadata`方法准备sequence级别metadata
- 将token级别索引转换为sequence级别索引（类似SGLang）

**关键代码**：
```python
def _get_decode_metadata(self) -> Tuple:
    """准备decode metadata，将token级别索引转换为sequence级别"""
    batch_size = self.token_lora_indices.size(0) if self.token_lora_indices is not None else 0
    
    # Decode阶段：每个sequence长度为1
    seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device=device)
    b_seq_start_loc = torch.cumsum(seq_len_tensor, dim=0)
    lora_indices_tensor = self.token_lora_indices  # sequence级别
    
    return (b_seq_start_loc, seq_len_tensor, lora_indices_tensor, 
            batches, max_seq_length, token_nums)

def _shrink_decode(self, ...):
    # 使用sgmv而不是bgmv
    decode_metadata = self._get_decode_metadata()
    self.sgmv_shrink(x, w_t_all, y, *decode_metadata, scale)
```

**预期收益**：
- 与SGLang保持一致，统一kernel调用路径
- 使用sequence级别索引，减少索引查找开销
- 可能获得更好的kernel优化

### ✅ 优化2: Batch准备方式优化

**实现内容**：
- 预分配decode metadata缓冲区（`_decode_seq_len_buffer`、`_decode_b_seq_start_loc_buffer`）
- 复用缓冲区，避免每次重新创建tensor
- 减少内存分配开销

**关键代码**：
```python
def _init_decode_metadata_buffers(self, device: Union[torch.device, str]) -> None:
    """预分配decode metadata缓冲区"""
    max_batch_size = self._max_num_batched_tokens
    
    # 预分配seq_len buffer（decode阶段全为1）
    self._decode_seq_len_buffer = torch.ones(
        max_batch_size, dtype=torch.int32, device=device
    )
    
    # 预分配b_seq_start_loc buffer: [0, 1, 2, ..., max_batch_size]
    self._decode_b_seq_start_loc_buffer = torch.zeros(
        max_batch_size + 1, dtype=torch.int32, device=device
    )
    self._decode_b_seq_start_loc_buffer[1:] = torch.cumsum(
        self._decode_seq_len_buffer, dim=0
    )

def _get_decode_metadata(self) -> Tuple:
    # 复用预分配的缓冲区（slice视图）
    if batch_size > 0:
        seq_len_tensor = self._decode_seq_len_buffer[:batch_size]
        b_seq_start_loc = self._decode_b_seq_start_loc_buffer[:batch_size + 1]
        # ...
```

**预期收益**：
- 减少内存分配开销
- 通过复用缓冲区，类似SGLang的batch信息复用

### ✅ 优化3: 修复Decode阶段的fused kernel使用不一致

**问题**：
- `add_lora_linear`中decode阶段单slice场景使用`bgmv_fused`
- 与统一sgmv的策略不一致

**修复**：
- 移除decode阶段的`bgmv_fused`路径
- 统一使用sgmv shrink+expand路径

### ✅ 其他已完成的优化

1. **Early Return优化**：
   - `add_lora_embedding`、`add_lora_linear`、`add_lora_logits`都有early return
   - 避免不必要的计算

2. **数据类型转换优化**：
   - 在`add_lora_embedding`中检查`x.dtype != torch.float32`时才转换
   - 避免不必要的类型转换

3. **QKV Merged Shrink**（已在代码中实现）：
   - 对于QKV操作（3个slice，相同rank），合并lora_a并执行一次shrink
   - 然后切片buffer，分别执行expand

---

## 进一步优化建议

### 1. QKV Merged Shrink中的torch.cat开销 ⚠️

**当前实现**：
```python
# 合并lora_a weights
lora_a_merged = torch.cat(lora_a_stacked, dim=-1)  # 创建新tensor，复制数据

# 合并buffer views
merged_buffer_list = [buf.view(-1, rank_per_slice) for buf in buffer]
merged_buffer_flat = torch.cat(merged_buffer_list, dim=-1)  # 再次创建新tensor
```

**问题**：
- `torch.cat`会创建新的tensor并复制所有数据
- 对于QKV（3个slice），需要复制3次数据
- 内存分配和复制开销可能抵消合并shrink的收益

**SGLang的做法**：
- `lora_a`在存储层面就已经是合并格式：`(num_loras, 3*rank, input_dim)`
- 不需要运行时`torch.cat`操作
- 这是**架构级别的优化**

**优化建议**：
- **短期**：如果batch较大，`torch.cat`的开销相对较小，可以接受
- **长期**：考虑在LoRA权重加载时预合并QKV的`lora_a`权重（需要修改权重加载逻辑）

### 2. Prefill阶段的sgmv_fused未使用 ❌

**当前状态**：
- 已实现`sgmv_fused`（fused shrink+expand）
- 但在`add_lora_linear`中没有使用`sgmv_fused`进行优化

**优化建议**：
- 在prefill阶段的单slice场景，可以考虑使用`sgmv_fused`
- 需要检查`sgmv_fused`的实现是否完整和正确

### 3. 进一步优化buffer复用

**当前状态**：
- 已预分配decode metadata缓冲区
- QKV merged shrink中的merged buffer需要临时创建

**优化建议**：
- 考虑为merged buffer预分配缓冲区
- 避免在每次QKV操作时创建临时buffer

### 4. 性能测试建议

**需要验证的点**：

1. **Decode阶段统一sgmv的性能提升**
   - 对比bgmv和sgmv的性能
   - 验证sequence级别索引vs token级别索引的影响

2. **Batch metadata复用 vs 重新创建**
   - 对比预分配缓冲区vs每次创建的性能
   - 验证内存分配开销的影响

3. **QKV merged shrink的开销分析**
   - 对比`torch.cat`开销 vs 多次kernel调用的开销
   - 确定是否需要进一步优化

4. **Fused kernel的使用**
   - 验证`sgmv_fused`在prefill阶段的性能
   - 对比fused vs 分离kernel的性能

### 推荐优先级

**高优先级（已完成）**：
1. ✅ Decode阶段统一使用sgmv
2. ✅ Batch准备方式优化（预分配和复用）
3. ✅ 修复decode阶段的fused kernel使用不一致

**中优先级（后续考虑）**：
1. 评估sgmv_fused在prefill阶段的使用
2. 优化QKV merged shrink的torch.cat开销（如果测试显示是瓶颈）
3. 进一步优化buffer复用

**低优先级（架构级优化）**：
1. 预合并LoRA权重（存储层面）
   - 需要在权重加载时合并QKV的`lora_a`
   - 这是较大的架构改动，需要评估收益

---

## 总结

### 已完成的优化

1. **Decode阶段统一使用sgmv**：与SGLang保持一致，统一kernel调用路径
2. **Batch准备方式优化**：预分配和复用缓冲区，减少内存分配开销
3. **代码一致性修复**：移除decode阶段的`bgmv_fused`路径，保持统一策略

### 预期收益

- **内存带宽**：通过统一sgmv和batch复用，减少不必要的内存操作
- **Kernel Launch**：统一kernel路径，可能减少分支预测开销
- **整体性能**：预期能够接近SGLang的性能水平（测试验证中）

### 下一步

1. **性能测试**：验证当前优化的实际效果
2. **持续优化**：根据测试结果决定是否需要进一步优化
3. **架构级优化**：如果收益明显，考虑存储层面的优化（预合并权重）

### 注意事项

1. **兼容性**：保留原有分离算子的实现，作为fallback
2. **精度验证**：确保所有优化的数值精度与原有实现一致
3. **性能监控**：持续监控不同场景下的性能表现

---

## 附录

### 相关文件

- `vllm_ascend/lora/punica_npu.py` - 主要优化实现
- `vllm_ascend/lora/lora_ops.py` - LoRA算子接口
- `csrc/kernels/bgmv_fused.cpp` - Fused kernel实现（C++）
- `docs/` - 相关设计文档

### 参考

- SGLang LoRA实现：`sglang/python/sglang/srt/lora/backend/ascend_backend.py`
- vLLM LoRA基础：`vllm/lora/punica_wrapper/punica_base.py`