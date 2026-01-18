/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "types.h"

/**
 * BGMVFused: Fused kernel combining bgmv_shrink and bgmv_expand operations.
 * 
 * This kernel performs the following computation in a single pass:
 *   buffer = (x @ lora_a) * scale  (shrink phase)
 *   y += buffer @ lora_b            (expand phase)
 * 
 * The intermediate buffer stays in L2 cache and is never written to global memory,
 * reducing memory bandwidth by ~50% compared to separate shrink/expand kernels.
 */
template <typename scalar_t>
class BGMVFused {
public:
    using X_T = scalar_t;
    using W_A_T = scalar_t;
    using W_B_T = scalar_t;
    using Y_T = scalar_t;
    using Buffer_T = float;  // Intermediate buffer uses float32 for precision

    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t TILE_LENGTH = 11776;  // optimal performance tile length

public:
    __aicore__ inline BGMVFused(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        __gm__ void *x,
        __gm__ void *lora_a_weight,
        __gm__ void *lora_b_weight,
        __gm__ void *indices,
        uint32_t indicesSize,
        __gm__ void *y,
        uint32_t batchSize,
        uint32_t numTokensPerCore,
        uint32_t inputHiddenDim,
        uint32_t maxLoRARank,
        uint32_t outputHiddenDim,
        float scale
    ) {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        outputHiddenDim_ = outputHiddenDim;
        scale_ = scale;
        singleLoRAAWeightLen_ = inputHiddenDim_ * maxLoRARank_;
        singleLoRABWeightLen_ = maxLoRARank_ * outputHiddenDim_;
        incremental_ = inputHiddenDim_ > TILE_LENGTH;

        xGm_.SetGlobalBuffer((__gm__ X_T *)x);
        yGm_.SetGlobalBuffer((__gm__ Y_T *)y);
        loraAGm_.SetGlobalBuffer((__gm__ W_A_T *)lora_a_weight);
        loraBGm_.SetGlobalBuffer((__gm__ W_B_T *)lora_b_weight);
        indicesGm_.SetGlobalBuffer((__gm__ int64_t *)indices, indicesSize);

        // Buffers for shrink phase
        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(X_T));
        pipe_->InitBuffer(inQueueWA_, BUFFER_NUM, TILE_LENGTH * sizeof(W_A_T));
        pipe_->InitBuffer(tmpBufferX_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(tmpBufferWA_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(shrinkBuffer_, maxLoRARank_ * sizeof(Buffer_T));

        // Buffers for expand phase (reuse some buffers from shrink)
        pipe_->InitBuffer(inQueueWB_, BUFFER_NUM, TILE_LENGTH * sizeof(W_B_T));
        pipe_->InitBuffer(tmpBufferWB_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(inQueueY_, BUFFER_NUM, TILE_LENGTH * sizeof(Y_T));
        pipe_->InitBuffer(tmpBufferY_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, TILE_LENGTH * sizeof(Y_T));
    }

    __aicore__ inline void Process() {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIdx = blockIdx * numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        if (endIdx > batchSize_) {
            endIdx = batchSize_;
        }

        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            // Set up LoRA index
            CopyInIndex(idx);
            if (reqLoRAIndex_ < 0) {
                continue;  // Skip tokens without LoRA
            }

            reqLoRAAWeightOffset_ = reqLoRAIndex_ * singleLoRAAWeightLen_;
            reqLoRABWeightOffset_ = reqLoRAIndex_ * singleLoRABWeightLen_;

            // Fused computation: shrink + expand in one pass
            if (incremental_) {
                ProcessImpl<true>(idx);
            } else {
                ProcessImpl<false>(idx);
            }
        }
    }

private:
    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ProcessImpl(const int64_t idx) {
        // Phase 1: Shrink (x @ lora_a * scale) -> buffer (in L2 cache)
        ShrinkPhase<INCREMENTAL_MODE>(idx);

        // Phase 2: Expand (buffer @ lora_b) -> y (directly from L2 cache)
        ExpandPhase(idx);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ShrinkPhase(const int64_t idx) {
        AscendC::LocalTensor<Buffer_T> bufferLocal = shrinkBuffer_.Get<Buffer_T>();

        if constexpr (!INCREMENTAL_MODE) {
            CopyInX(idx, 0, inputHiddenDim_);
            AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            pipe_barrier(PIPE_V);
            inQueueX_.FreeTensor(xLocal);
        }

        // Compute shrink: buffer = (x @ lora_a) * scale
        for (int i = 0; i < maxLoRARank_; i++) {
            float acc(0);
            for (int32_t j = 0; j < inputHiddenDim_ / TILE_LENGTH; j++) {
                if constexpr (INCREMENTAL_MODE) {
                    CopyInX(idx, j);
                }
                CopyInWA(i, j);
                ComputeShrink<INCREMENTAL_MODE>(acc);
            }
            CopyAndComputeShrinkLastIteration<INCREMENTAL_MODE>(idx, i, acc);
            bufferLocal.SetValue(i, acc * scale_);
        }
    }

    __aicore__ inline void ExpandPhase(const int64_t idx) {
        // Use shrink buffer directly from L2 cache (no global memory write/read)
        // This is the key optimization: buffer stays in L2 cache between shrink and expand
        AscendC::LocalTensor<Buffer_T> bufferLocal = shrinkBuffer_.Get<Buffer_T>();

        // Prepare buffer for expand computation (similar to CopyInX in bgmv_expand)
        // Duplicate buffer to match NUM_ELEMENTS_PER_REPEAT for efficient computation
        static constexpr int32_t NUM_BYTES_PER_REPEAT = 256;
        static constexpr int32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);
        static constexpr int32_t MASK_COUNT = NUM_ELEMENTS_PER_REPEAT;
        static constexpr int32_t W_IN_TILE_NUM_ELEMENTS = 8192;
        static constexpr int32_t Y_OUT_TILE_NUM_ELEMENTS = 4096;
        static constexpr int32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE_NUM_ELEMENTS / NUM_ELEMENTS_PER_REPEAT;

        // Duplicate buffer for expand computation
        AscendC::LocalTensor<float> dupBuffer = tmpBufferX_.Get<float>();  // Reuse tmpBufferX
        for (int32_t i = 0; i < NUM_ELEMENTS_PER_REPEAT; i += maxLoRARank_) {
            for (int32_t j = 0; j < maxLoRARank_; j++) {
                float entry = bufferLocal.GetValue(j);
                dupBuffer.SetValue(i + j, entry);
            }
        }

        // Compute expand: y += buffer @ lora_b
        // Process output in tiles
        int32_t numOutputElementsPerInputTile = BLOCK_REDUCE_NUM_REPEATS * (NUM_ELEMENTS_PER_REPEAT / maxLoRARank_);
        int32_t numStreamInPerOutputTile = Y_OUT_TILE_NUM_ELEMENTS / numOutputElementsPerInputTile;
        int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;

        for (int32_t i = 0; i < numStreamOut; i++) {
            CopyInY(idx, i);
            for (int32_t j = 0; j < numStreamInPerOutputTile; j++) {
                CopyInWB(i * numStreamInPerOutputTile + j);
                ComputeExpand(dupBuffer, j * numOutputElementsPerInputTile);
            }
            ScaleAndAddOutput(idx, i);
        }

        // Handle remaining elements
        ComputeExpandLastIteration(dupBuffer, idx, numStreamOut, numOutputElementsPerInputTile, numStreamInPerOutputTile);
    }

    __aicore__ inline void CopyInY(const int64_t idx, int32_t progress, int32_t numElements = 4096) {
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.AllocTensor<Y_T>();
        DataCopy(yInLocal, yGm_[outputHiddenDim_ * idx + progress * numElements], numElements);
        inQueueY_.EnQue(yInLocal);
    }

    __aicore__ inline void CopyInWB(int32_t progress, int32_t numElements = 8192) {
        AscendC::LocalTensor<W_B_T> wLocal = inQueueWB_.AllocTensor<W_B_T>();
        DataCopy(wLocal, loraBGm_[reqLoRABWeightOffset_ + progress * numElements], numElements);
        inQueueWB_.EnQue(wLocal);
    }

    __aicore__ inline void ComputeExpand(
        AscendC::LocalTensor<float> &dupBuffer,
        int32_t progress,
        int32_t blockReduceRepeatCount = 32,
        int32_t pairReduceRepeat16 = 16,
        int32_t pairReduceRepeat32 = 8
    ) {
        static constexpr int32_t MASK_COUNT = 64;
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<W_B_T> wLocal = inQueueWB_.DeQue<W_B_T>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferWB_.Get<float>();

        // Cast weight to float
        AscendC::UnaryRepeatParams castParams = {1, 1, 8, 4};
        Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, MASK_COUNT, blockReduceRepeatCount, castParams);
        pipe_barrier(PIPE_V);
        inQueueWB_.FreeTensor(wLocal);

        // Compute: dupBuffer @ wTmpTensor
        AscendC::BinaryRepeatParams dotProductParams = {1, 1, 1, 8, 0, 8};
        Mul(wTmpTensor, dupBuffer, wTmpTensor, MASK_COUNT, blockReduceRepeatCount, dotProductParams);
        pipe_barrier(PIPE_V);

        // Reduce based on LoRA rank
        AscendC::UnaryRepeatParams reduceSumParams = {1, 1, 1, 8};
        if (maxLoRARank_ == 8) {
            BlockReduceSum(yLocal[progress], wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
        } else if (maxLoRARank_ == 16) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
            PairReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat16, MASK_COUNT,
                          reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
        } else if (maxLoRARank_ == 32) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
            PairReduceSum(wTmpTensor, wTmpTensor, pairReduceRepeat16, MASK_COUNT,
                           reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
            PairReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat32, MASK_COUNT,
                          reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
        } else if (maxLoRARank_ == 64) {
            BlockReduceSum(wTmpTensor, wTmpTensor, blockReduceRepeatCount, MASK_COUNT,
                           reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
            BlockReduceSum(yLocal[progress], wTmpTensor, pairReduceRepeat16, MASK_COUNT,
                          reduceSumParams.dstRepStride, reduceSumParams.srcBlkStride, reduceSumParams.srcRepStride);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ScaleAndAddOutput(const int64_t idx, int32_t progress, int32_t numElements = 4096) {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.DeQue<Y_T>();
        AscendC::LocalTensor<float> yInLocalFP32 = tmpBufferWA_.Get<float>();  // Reuse tmpBufferWA

        Cast(yInLocalFP32, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
        pipe_barrier(PIPE_V);
        inQueueY_.FreeTensor(yInLocal);

        Add(yLocal, yLocal, yInLocalFP32, numElements);
        pipe_barrier(PIPE_V);

        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        pipe_barrier(PIPE_V);

        outQueueY_.EnQue<Y_T>(yOutLocal);

        // Copy out
        yOutLocal = outQueueY_.DeQue<Y_T>();
        DataCopy(yGm_[outputHiddenDim_ * idx + progress * numElements], yOutLocal, numElements);
        outQueueY_.FreeTensor(yOutLocal);
    }

    __aicore__ inline void ComputeExpandLastIteration(
        AscendC::LocalTensor<float> &dupBuffer,
        const int64_t idx,
        int32_t numStreamOut,
        int32_t numOutputElementsPerInputTile,
        int32_t numStreamInPerOutputTile
    ) {
        static constexpr int32_t Y_OUT_TILE_NUM_ELEMENTS = 4096;
        static constexpr int32_t W_IN_TILE_NUM_ELEMENTS = 8192;
        static constexpr int32_t NUM_ELEMENTS_PER_REPEAT = 64;
        static constexpr int32_t NUM_BLOCKS_PER_REPEAT = 8;

        int32_t remainingY = outputHiddenDim_ % Y_OUT_TILE_NUM_ELEMENTS;
        if (remainingY == 0) {
            return;
        }

        int32_t remainingW = remainingY * maxLoRARank_;
        int32_t numCompleteWTileInForLastIteration = remainingW / W_IN_TILE_NUM_ELEMENTS;
        int32_t remainingWForLastRepeat = remainingW % W_IN_TILE_NUM_ELEMENTS;

        CopyInY(idx, numStreamOut, remainingY);

        int32_t outputIdx = 0;
        for (outputIdx = 0; outputIdx < numCompleteWTileInForLastIteration; outputIdx++) {
            CopyInWB(numStreamOut * numStreamInPerOutputTile + outputIdx);
            ComputeExpand(dupBuffer, outputIdx * numOutputElementsPerInputTile);
        }

        if (remainingWForLastRepeat != 0) {
            CopyInWB(numStreamOut * numStreamInPerOutputTile + numCompleteWTileInForLastIteration,
                    remainingWForLastRepeat);
            int32_t lastRepeatCount = remainingWForLastRepeat / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduceRepeat16 = 
                (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
            int32_t pairReduceRepeat32 = (pairReduceRepeat16 + 1) / 2;
            int32_t lastComputeOutputElement = outputIdx * numOutputElementsPerInputTile;
            ComputeExpand(dupBuffer, lastComputeOutputElement, lastRepeatCount, pairReduceRepeat16, pairReduceRepeat32);
        }

        ScaleAndAddOutput(idx, numStreamOut, remainingY);
    }

    // Helper methods similar to bgmv_shrink and bgmv_expand
    __aicore__ inline void CopyInIndex(const int64_t idx) {
        reqLoRAIndex_ = indicesGm_.GetValue(idx);
    }

    __aicore__ inline void CopyInX(const int64_t idx, int32_t colIdx, int32_t numElements = TILE_LENGTH) {
        AscendC::LocalTensor<X_T> xLocal = inQueueX_.AllocTensor<X_T>();
        DataCopy(xLocal, xGm_[inputHiddenDim_ * idx + colIdx * TILE_LENGTH], numElements);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInWA(int32_t rowIdx, int32_t colIdx, int32_t numElements = TILE_LENGTH) {
        AscendC::LocalTensor<W_A_T> wLocal = inQueueWA_.AllocTensor<W_A_T>();
        DataCopy(wLocal, loraAGm_[reqLoRAAWeightOffset_ + rowIdx * inputHiddenDim_ + colIdx * TILE_LENGTH], numElements);
        inQueueWA_.EnQue(wLocal);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ComputeShrink(float &acc, int32_t numElements = TILE_LENGTH) {
        AscendC::LocalTensor<W_A_T> wLocal = inQueueWA_.DeQue<W_A_T>();
        AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferWA_.Get<float>();

        if constexpr (INCREMENTAL_MODE) {
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, numElements);
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            pipe_barrier(PIPE_V);
            inQueueX_.FreeTensor(xLocal);
            inQueueWA_.FreeTensor(wLocal);
        } else {
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            pipe_barrier(PIPE_V);
            inQueueWA_.FreeTensor(wLocal);
        }
        // Dot product of the one tile of X and W
        Mul(wTmpTensor, xTmpTensor, wTmpTensor, numElements);
        pipe_barrier(PIPE_V);
        // Reduce sum generate one number, which is the summation of all the dot product
        ReduceSum<float>(wTmpTensor, wTmpTensor, wTmpTensor, numElements);
        pipe_barrier(PIPE_V);

        acc += wTmpTensor.GetValue(0);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void CopyAndComputeShrinkLastIteration(const int64_t idx, int32_t rowIdx, float &acc) {
        int32_t colIdx = inputHiddenDim_ / TILE_LENGTH;
        int32_t remaining = inputHiddenDim_ % TILE_LENGTH;
        if (remaining == 0) {
            return;
        }
        if constexpr (INCREMENTAL_MODE) {
            CopyInX(idx, colIdx, remaining);
        }
        CopyInWA(rowIdx, colIdx, remaining);
        ComputeShrink<INCREMENTAL_MODE>(acc, remaining);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_, inQueueWA_, inQueueWB_, inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferX_, tmpBufferWA_, tmpBufferWB_, tmpBufferY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> shrinkBuffer_;  // Stays in L2 cache
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_A_T> loraAGm_;
    AscendC::GlobalTensor<W_B_T> loraBGm_;
    AscendC::GlobalTensor<Y_T> yGm_;
    AscendC::GlobalTensor<int64_t> indicesGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t outputHiddenDim_;
    float scale_;
    uint32_t singleLoRAAWeightLen_;
    uint32_t singleLoRABWeightLen_;
    int64_t reqLoRAIndex_;
    uint64_t reqLoRAAWeightOffset_;
    uint64_t reqLoRABWeightOffset_;
    bool incremental_;
};

// Kernel declaration macros (similar to bgmv_shrink/bgmv_expand)
#define BGMV_FUSED_TYPE_DECLARE(TYPE)                                                                                 \
    extern "C" __global__ __aicore__ void bgmv_fused_##TYPE(                                                          \
        __gm__ void* x, __gm__ void* lora_a_weight, __gm__ void* lora_b_weight,                                      \
        __gm__ void* indices, uint32_t indicesSize, __gm__ void* y,                                                   \
        uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim,                                       \
        uint32_t maxLoRARank, uint32_t outputHiddenDim, float scale)                                                  \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        BGMVFused<TYPE> op(&pipe);                                                                                     \
        op.Init(x, lora_a_weight, lora_b_weight, indices, indicesSize, y, batchSize, numTokensPerCore,              \
                inputHiddenDim, maxLoRARank, outputHiddenDim, scale);                                                   \
        op.Process();                                                                                                  \
    }

// Declare all dtype kernels
BGMV_FUSED_TYPE_DECLARE(half)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
    BGMV_FUSED_TYPE_DECLARE(bfloat16_t)
#endif

namespace vllm_ascend {
extern void bgmv_fused_impl(
    AscendType type, void* stream,
    void* x, void* lora_a_weight, void* lora_b_weight,
    void* indices, uint32_t indicesSize, void* y,
    uint32_t batchSize, uint32_t numTokensPerCore,
    uint32_t inputHiddenDim, uint32_t maxLoRARank,
    uint32_t outputHiddenDim, float scale)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == AscendType::FP16) {
        bgmv_fused_half<<<blockDim, nullptr, stream>>>(
            x, lora_a_weight, lora_b_weight, indices, indicesSize, y,
            batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, outputHiddenDim, scale);
    } else if (type == AscendType::BF16) {
        #if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        bgmv_fused_bfloat16_t<<<blockDim, nullptr, stream>>>(
            x, lora_a_weight, lora_b_weight, indices, indicesSize, y,
            batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, outputHiddenDim, scale);
        #endif
    } else {
        return;
    }
}

} // namespace vllm_ascend
