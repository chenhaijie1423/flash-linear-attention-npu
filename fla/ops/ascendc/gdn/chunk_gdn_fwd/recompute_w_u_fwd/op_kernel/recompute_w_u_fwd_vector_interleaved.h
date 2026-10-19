/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd_vector_interleaved.h
 * \brief 910B Membase Interleaved Vector 实现。
 *        将 A5 Regbase 版本的 ring buffer + 交替执行策略移植到 910B：
 *          - 每个 chunk+head 内先做 Vb 再做 KbgExp，使 Cube 端能复用 L1 中的 A 矩阵
 *          - GM ring buffer (RECOMPUTE_W_U_FWD_GM_RING_DEPTH=8) 减少 workspace 占用
 *          - CrossCoreFlagWithReverse 实现深度为 8 的 AIC/AIV 流水重叠
 *        与 Regbase 版本的区别：
 *          - 使用 membase API (Cast/Brcb/Mul/Exp) 替代 MicroAPI (RegTensor/LoadIn/StoreAlign)
 *          - 需要 FP32 中间 buffer，UB 占用略高于 Regbase 版本
 */

#ifndef RECOMPUTE_W_U_FWD_VECTOR_INTERLEAVED_H
#define RECOMPUTE_W_U_FWD_VECTOR_INTERLEAVED_H

#include "recompute_w_u_fwd_struct.h"
#include "recompute_w_u_fwd_common.h"
#include "catlass/arch/cross_core_sync.hpp"

using namespace AscendC;
using GDN::RecomputeWUFwdTilingData;

template <typename kType, typename betaType, int VDim>
class RecomputeWUFwdVectorProcessInterleaved {
public:
    __aicore__ inline RecomputeWUFwdVectorProcessInterleaved(
        GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_, GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void Init(const RecomputeWUFwdTilingData &tiling, AscendC::TPipe *pipe_);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t interleavedMembaseVecRow = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;

    // ring buffer cross-core sync (depth = GM_RING_DEPTH, same as A5 Regbase)
    static constexpr uint32_t GM_RING_DEPTH = GDN::RECOMPUTE_W_U_FWD_GM_RING_DEPTH;

    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<kType> workSpaceTensor;

    // Shared queues for both Vb and KbgExp stages, sized for maxDataDim
    TQue<AscendC::TPosition::VECIN, 1> dataInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECOUT, 1> dataOutQue;

    // FP32 intermediate buffers (membase needs these unlike regbase)
    TBuf<AscendC::TPosition::VECCALC> dataFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;

    __aicore__ inline void ProcessVbAndKbgExpInterleaved();
};

// =================== Implementation ===================

template <typename kType, typename betaType, int VDim>
__aicore__ inline RecomputeWUFwdVectorProcessInterleaved<kType, betaType, VDim>::RecomputeWUFwdVectorProcessInterleaved(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_) {}

template <typename kType, typename betaType, int VDim>
__aicore__ void inline RecomputeWUFwdVectorProcessInterleaved<kType, betaType, VDim>::Init(
    const RecomputeWUFwdTilingData &tiling, AscendC::TPipe *pipe_)
{
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    interleavedMembaseVecRow = tiling.interleavedMembaseVecRow;
}

template <typename kType, typename betaType, int VDim>
__aicore__ void inline RecomputeWUFwdVectorProcessInterleaved<kType, betaType, VDim>::Process()
{
    ProcessVbAndKbgExpInterleaved();
}

// =================== ProcessVbAndKbgExpInterleaved ===================
// 交替模式：每个 chunk+head 内先做 Vb 再做 KbgExp，使 Cube 端能复用 L1 中的 A 矩阵
// workspace 使用 ring buffer (depth=GM_RING_DEPTH) 减少 GM 占用
template <typename kType, typename betaType, int VDim>
__aicore__ void inline RecomputeWUFwdVectorProcessInterleaved<kType, betaType, VDim>::ProcessVbAndKbgExpInterleaved()
{
    uint32_t coreLoops = chunkNum;
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = interleavedMembaseVecRow;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // Vb and KbgExp execute sequentially within the same loop, so reuse data
    // queues sized for the larger K/V dimension.
    uint32_t maxDataDim = K > V ? K : V;
    pipe->InitBuffer(dataInQue, 2, rowNum * maxDataDim * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dataOutQue, 2, rowNum * maxDataDim * sizeof(kType));
    pipe->InitBuffer(dataFp32Buf, rowNum * maxDataDim * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));

    auto tensorDataFp32 = dataFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorGFP32 = gFp32Buf.Get<float32_t>();

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < Hv; h++) {
            bool isLast = (loopIdx + coreNumAic >= coreLoops) && (h == Hv - 1);
            const uint64_t ringTask = static_cast<uint64_t>(coreIdx) * GM_RING_DEPTH + vecTaskIdx % GM_RING_DEPTH;
            ++vecTaskIdx;
            // The first ring window is initially empty. Thereafter every
            // task reuses the slot released by the task GM_RING_DEPTH steps
            // earlier. All AIV subblocks consume the release token so the
            // reverse-depth accounting remains aligned with AIC.
            if ((vecTaskIdx & 0b01) != subBlockIdx) {
                // Both AIV subcores must contribute one credit per stage.
                if ((vecTaskIdx & 0b11) == 0 || isLast) {
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIC_AIV_FLAG_6);
                    if (vecTaskIdx > 4) {
                        CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_RING_SLOT_FREE_FLAG);
                    }
                }
                continue;
            }

            // === 第一部分：Vb = v * beta ===
            const uint64_t vbDstBase = ringTask * chunkSize * V;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorVin = dataInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    dataInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorVin = dataInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorVbOut = dataOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorDataFp32, tensorVin, RoundMode::CAST_NONE, V * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < V) {
                        Mul(tensorDataFp32[perchannelResOffset], tensorDataFp32[perchannelResOffset],
                            tensorBetaBrcbFP32, FP32_PER_REPEAT_64, curRowNum,
                            {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast back
                    Cast(tensorVbOut, tensorDataFp32, RoundMode::CAST_RINT, V * curRowNum);
                    dataInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    dataOutQue.EnQue(tensorVbOut);
                }
                // copyout to ring buffer slot
                {
                    auto tensorVbOut = dataOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[vbDstBase + rowOffset * V], tensorVbOut, V * curRowNum);
                    dataOutQue.FreeTensor(tensorVbOut);
                }
            }

            // === 第二部分：KbgExp = k * beta * exp(g) ===
            uint64_t hk = h / hvPerHk;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                // k 只有 Hk 个 head，需把批次偏移换算成 bIdx*Hk*T
                uint64_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
                uint64_t bIdx = cu_seqlens ? 0 : (loopIdx / coreLoopsInB);
                uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
                auto kSrcOffset = (hk * T + bosK + rowOffset) * K;
                const uint64_t kbgRingBase = static_cast<uint64_t>(coreNumAic) *
                    GM_RING_DEPTH * chunkSize * V;
                const uint64_t kbgDstBase = kbgRingBase + ringTask * chunkSize * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorKin = dataInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    DataCopy(tensorKin, kTensor[kSrcOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(tensorGin, gTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    dataInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                }
                // compute
                {
                    auto tensorKin = dataInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorGin = gInQue.DeQue<betaType>();
                    auto tensorOut = dataOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorGFP32, tensorGin, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        DataCopy(tensorGFP32, tensorGin, rowNum);
                    }
                    Cast(tensorDataFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    Exp(tensorGFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    Mul(tensorBetaFP32, tensorBetaFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDataFp32[perchannelResOffset], tensorDataFp32[perchannelResOffset],
                            tensorBetaBrcbFP32, FP32_PER_REPEAT_64, curRowNum,
                            {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast back
                    Cast(tensorOut, tensorDataFp32, RoundMode::CAST_RINT, K * curRowNum);
                    dataInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    gInQue.FreeTensor(tensorGin);
                    dataOutQue.EnQue(tensorOut);
                }
                // copyout to ring buffer slot
                {
                    auto tensorOut = dataOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kbgDstBase + rowOffset * K], tensorOut, K * curRowNum);
                    dataOutQue.FreeTensor(tensorOut);
                }
            }
            // KbgExp 完成通知 AIC 可以做 W matmul
            if ((vecTaskIdx & 0b11) == 0 || isLast) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIC_AIV_FLAG_6);
                if (vecTaskIdx > 4) {
                    CrossCoreWaitFlag<0x2>(SYNC_AIC_AIV_RING_SLOT_FREE_FLAG);
                }
            }
        }
    }
}

#endif // RECOMPUTE_W_U_FWD_VECTOR_INTERLEAVED_H
