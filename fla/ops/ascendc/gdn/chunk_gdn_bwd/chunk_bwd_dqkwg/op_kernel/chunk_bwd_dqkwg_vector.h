/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_vector.h
 *
 * CV 深融合 (chunk-interleaved A/B/C/D) 的 AIV (Vector) 端。
 *
 * 与 cube 端镜像: 每核任务流 A(c0..cM-1), B(...), C(...), D(...)。
 * 每个 (stage, chunk) task: 两个 AIV sub-block 各 WaitCubeReady 一次,
 * 处理完该 chunk 全部 head (sub-block 内部按 head / row 切分) 后各 SetVectorDone 一次。
 * cube 落后处理: vector 处理 task[i] 时 cube 已在做 task[i+1]。
 *
 * stage 映射 (与 cube 对应):
 *   A_vector = 原 Part1 vector (dw 取负 + dg_last) + 原 Part2 vector (mul1)
 *   B_vector = 原 Part3 vector (ds_temp + dg 部分)
 *   C_vector = 原 Part4 + Part6 vector (dq 最终 + dg)
 *   D_vector = 原 Part5 + Part7 vector (dk 最终 + dg 最终, 含 dg_last 重算)
 *
 * 同步计数平衡 (区别于 cv_merge 死锁版本): 无 preseed, 无 per-head flag, 无 stage drain;
 * 每个 AIV sub-block 每 task 恰好 1 次 WaitCubeReady + 1 次 SetVectorDone。
 */

#ifndef CHUNK_BWD_DQKWG_VECTOR_H
#define CHUNK_BWD_DQKWG_VECTOR_H

#include "chunk_bwd_dqkwg_common.h"
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_bwd_dqkwg_regbase.h"
#endif

using namespace AscendC;

template <typename DataType, typename GType>
class ChunkBwdDqkwgVectorProcess {
public:
    __aicore__ inline ChunkBwdDqkwgVectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_,
                                                 GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices,
                                                 GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace);

    __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling, TPipe *pipe_);
    __aicore__ inline void Process();

private:
    // 4 个 stage 的处理函数 (CV 深融合)
    __aicore__ inline void ProcessAVector(uint32_t loopBase,
                                          uint32_t loopEnd); // 原 Part1 (dw 取负 + dg_last) + 原 Part2 (mul1)
    __aicore__ inline void ProcessBVector(uint32_t loopBase, uint32_t loopEnd); // 原 Part3 (ds_temp + dg 部分)
    __aicore__ inline void ProcessCVector(uint32_t loopBase, uint32_t loopEnd); // 原 Part4 + Part6 (dq 最终)
    __aicore__ inline void ProcessDVector(uint32_t loopBase, uint32_t loopEnd); // 原 Part5 + Part7 (dk 最终 + dg 最终)

    // mul1 一个 row-half 的计算 (= A 的 Part2 per-head 内核, 输出 fp32 到 outFp32[half] )。
    // A (小 case) 调它后 cast+写 GM; B (大 case) 调它两次 (两个 row-half) 后直接乘 ds, 省掉 mul1 GM 往返。
    __aicore__ inline void ComputeMul1HalfFp32(const LocalTensor<float> &outFp32, const LocalTensor<float> &tensorMaskA,
                                               const LocalTensor<float> &tensorBrcbTemp, const LocalTensor<float> &tensorGFp32Right, 
                                               uint64_t gOffset, uint32_t BT_sub_start, uint32_t real_BT, uint32_t actual_chunk_len);

    __aicore__ inline void CopyGateWithPad(const LocalTensor<GType> &dst, const GlobalTensor<GType> &src, uint64_t offset,
                                           uint32_t validLen, uint32_t totalLen);

private:
    // 输入输出指针
    GM_ADDR ptrQ;
    GM_ADDR ptrK;
    GM_ADDR ptrV;
    GM_ADDR ptrG;
    GM_ADDR ptrH;
    GM_ADDR ptrDo;
    GM_ADDR ptrDh;
    GM_ADDR ptrDv;
    GM_ADDR ptrCuSeqLen;
    GM_ADDR ptrChunkIndices;
    GM_ADDR ptrDq;
    GM_ADDR ptrDk;
    GM_ADDR ptrDw;
    GM_ADDR ptrDg;
    GM_ADDR ptrWorkspace;

    // Tiling 参数 (GVA: H 拆为 HV/HK, HV = n_ratio * HK)
    uint64_t B;
    uint64_t HV; // value 侧 head 数 (== 原 H)
    uint64_t HK; // key/query 侧 head 数, HV = n_ratio * HK
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t BT;
    uint64_t numChunks;
    float scale;
    bool isVarLen;
    bool gvaMode;
    uint32_t mul0RowNum = 0;
    uint64_t n_ratio = 1;    // GVA: HV / HK
    uint32_t coreNum = 0; // CV 深融合 blockDim (cube/vector 共用)
    uint32_t coreIdx = 0;
    uint32_t subBlockIdx = 0;

    // Workspace 偏移
    uint64_t wsBtxKSyncSlotsPerHead;
    uint64_t wsDgLastOffset;
    uint64_t wsMm5Offset;
    uint64_t wsDsTempOffset;
    uint64_t wsMm6Offset;
    uint64_t wsMm7Offset;
    int BUFFER_NUM = 1;

    // Pipeline
    TPipe *pipe = nullptr;

    // CV 深融合: raw 信用流水, 无 flag 对象 (helper 用固定 flag id, 与 cube 端共用)。

    // Global Tensors
    GlobalTensor<DataType> gmQ, gmK, gmV, gmDo, gmH, gmDh, gmDv;
    GlobalTensor<DataType> gmDq, gmDk, gmDw;
    GlobalTensor<GType> gmG, gmDg;
    GlobalTensor<float> gmDgLast;
    GlobalTensor<DataType> gmMm5, gmDsTemp, gmMm6, gmMm7;
    GlobalTensor<uint64_t> cuSeqlensTensor, chunkIndicesTensor;

    // Queues (用于流水)
    TQue<TPosition::VECIN, 2> inQue1;
    TQue<TPosition::VECIN, 2> inQue2;
    TQue<TPosition::VECIN, 2> inQue3;
    TQue<TPosition::VECIN, 2> inQue4; // 用于Add0累加
    TQue<TPosition::VECOUT, 2> outQue1;
    TQue<TPosition::VECOUT, 2> outQue2;


    // Calc Buffers (UB 空间)
    TBuf<TPosition::VECCALC> calcBuf1; // 主计算缓冲区 (fp32)
    TBuf<TPosition::VECCALC> calcBuf2; // 辅助计算缓冲区 (fp32)
    TBuf<TPosition::VECCALC> calcBuf3; // Exp 缓冲区
    TBuf<TPosition::VECCALC> maskBuf;  // A_vector mask
    TBuf<TPosition::VECCALC> dgBuf;    // dg 值缓冲区

    // Local Tensors
    LocalTensor<float> tensorMaskA;

    // UB 空间常量
    static constexpr uint32_t UB_BLOCK_SIZE = 32;
    static constexpr uint32_t FP32_ELEMENTS_PER_BLOCK = 8;
    static constexpr uint32_t FP16_ELEMENTS_PER_BLOCK = 16;
};

// ============== 构造函数 ==============
template <typename DataType, typename GType>
__aicore__ inline ChunkBwdDqkwgVectorProcess<DataType, GType>::ChunkBwdDqkwgVectorProcess(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen,
    GM_ADDR chunk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace)
    : ptrQ(q), ptrK(k), ptrV(v), ptrG(g), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen),
      ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrDg(dg), ptrWorkspace(workspace)
{
}

// ============== 初始化 ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling,
                                                                         TPipe *pipe_)
{
    pipe = pipe_;

    scale = tiling.scale;
    B = tiling.B;
    HV = tiling.HV;
    HK = tiling.HK;
    n_ratio = (HK > 0) ? (HV / HK) : 1;
    T = tiling.T;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.BT;
    numChunks = tiling.numChunks;
    coreNum = tiling.aicCoreNum;
    coreIdx = GetBlockIdx() / GetSubBlockNum();
    subBlockIdx = GetSubBlockIdx();
    wsBtxKSyncSlotsPerHead = tiling.wsBtxKSyncSlotsPerHead;
    wsDgLastOffset = tiling.wsDgLastOffset;
    wsMm5Offset = tiling.wsMm5Offset;
    wsDsTempOffset = tiling.wsDsTempOffset;
    wsMm6Offset = tiling.wsMm6Offset;
    wsMm7Offset = tiling.wsMm7Offset;
    isVarLen = (tiling.isVarLen == 1);
    gvaMode = (HV != HK);
    mul0RowNum = tiling.mul0RowNum;

    if (BT == 64) {
        BUFFER_NUM = 2;
    } else {
        BUFFER_NUM = 1;
    }

    gmQ.SetGlobalBuffer((__gm__ DataType *)ptrQ);
    gmK.SetGlobalBuffer((__gm__ DataType *)ptrK);
    gmV.SetGlobalBuffer((__gm__ DataType *)ptrV);
    gmG.SetGlobalBuffer((__gm__ GType *)ptrG);
    gmH.SetGlobalBuffer((__gm__ DataType *)ptrH);
    gmDo.SetGlobalBuffer((__gm__ DataType *)ptrDo);
    gmDh.SetGlobalBuffer((__gm__ DataType *)ptrDh);
    gmDv.SetGlobalBuffer((__gm__ DataType *)ptrDv);

    gmDq.SetGlobalBuffer((__gm__ DataType *)ptrDq);
    gmDk.SetGlobalBuffer((__gm__ DataType *)ptrDk);
    gmDw.SetGlobalBuffer((__gm__ DataType *)ptrDw);
    gmDg.SetGlobalBuffer((__gm__ GType *)ptrDg);

    gmDgLast.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)ptrWorkspace + wsDgLastOffset)); // 中间结果使用float
    gmMm5.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm5Offset));
    gmDsTemp.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsDsTempOffset));
    gmMm6.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm6Offset));
    gmMm7.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm7Offset));

    if (isVarLen) {
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)ptrCuSeqLen);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)ptrChunkIndices);
    }

    pipe->InitBuffer(inQue1, BUFFER_NUM, BT * K * sizeof(float));     // 64K
    pipe->InitBuffer(inQue2, BUFFER_NUM, BT * K * sizeof(float));     // 64K
    pipe->InitBuffer(inQue3, 2, BT * sizeof(float));               // 1K
    pipe->InitBuffer(inQue4, BUFFER_NUM, BT * sizeof(GType));      // 512B
    pipe->InitBuffer(outQue1, BUFFER_NUM, BT * K * sizeof(DataType)); // 32K
    pipe->InitBuffer(outQue2, BUFFER_NUM, BT * sizeof(float));     // 512B

    pipe->InitBuffer(calcBuf1, BT * 8 * sizeof(float));            // 4K
    pipe->InitBuffer(calcBuf2, BT * 8 * sizeof(float));            // 4K
    pipe->InitBuffer(calcBuf3, BT * sizeof(float));                // 512B
    pipe->InitBuffer(maskBuf, 64 * 64 * sizeof(float));               // 16K
    pipe->InitBuffer(dgBuf, BT * 8 * sizeof(float));               // 4K

    tensorMaskA = maskBuf.Get<float>();
    Duplicate(tensorMaskA, 0.0f, 64 * 64);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < 64; i++) {
        Duplicate(tensorMaskA[i * 64], 1.0f, i + 1);
    }
    PipeBarrier<PIPE_V>();
}

// ============== 主处理函数 (CV 深融合: A -> B -> C -> D, 每 stage 内 chunk 级与 cube 流水) ==============
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Process()
{
    // raw 信用流水: 启动时预置 N = min(groupSize, M) 个信用 (M = 本核 chunk 数),
    // 使 cube 可领先 vector 最多 N 个 task (真正并发); N<=M 保证 C/D 读 ds_temp 安全。
    // 两个 AIV sub-block 各预置 N 次 (0x2 汇合 => cube 看到 N 个信用)。
    // stage 之间无 SyncAll, 信用 flag 跨 stage 连续 (pipe->Reset 只重置 UB buffer, 不影响 flag)。
    uint32_t coreLoops = B * numChunks;
    uint32_t M = (coreIdx < coreLoops) ? ((coreLoops - 1 - coreIdx) / coreNum + 1) : 0;
    uint32_t groupSize = DqkwgGroupSizeFromRingDepth((uint32_t)wsBtxKSyncSlotsPerHead);
    uint32_t preseed = M < groupSize ? M : groupSize;
    for (uint32_t i = 0; i < preseed; i++) {
        SetVecCredit();
    }

    // 跨 stage 的 vector->vector GM 数据 (mul1: A->B; dg: B->C->D; dg_last: A->D) 由同一 AIV sub-block
    // 顺序读写。(每个 (c,h) 的 dg/mul1/dg_last 由同一 sub-block 跨 A/B/C/D 处理 => 无需 SyncAll)
    // 固定长度路径 stage 间只 drain MTE3, 保持热路径开销; varlen 下每组 chunk 长度不齐, stage/group
    // 边界 reset pipe 前必须 drain 全 pipeline, 避免 V/MTE2 尾部操作与 pipe->Reset 竞态导致偶发 head 损坏。
    // chunk-group-major: 外层按 G 个 chunk 一组 (与 cube 用同一 DqkwgGroupEnd 算组边界, 握手顺序一致),
    //   组内 A->B->C->D 连着做 -> 这组的输入张量只从 HBM 读一次、在 L2 内被 4 个 stage 复用 (砍 MTE2 重复搬运)。
    //   stage 间仍 PipeBarrier<MTE3>+Reset (drain mul1/dg/dg_last 跨 stage GM 可见); 组末额外 Reset 供下组 A 干净开始。
    //   信用 flag 跨组连续 (preseed 一次); ds_temp 安全靠每组 >= N (DqkwgGroupEnd 的尾巴合并保证)。
    uint32_t loopBase = coreIdx;
    while (loopBase < coreLoops) {
        uint32_t loopEnd = DqkwgGroupEnd(loopBase, coreLoops, coreNum, (uint32_t)wsBtxKSyncSlotsPerHead);
        ProcessAVector(loopBase, loopEnd);
        ProcessBVector(loopBase, loopEnd);
        ProcessCVector(loopBase, loopEnd);
        ProcessDVector(loopBase, loopEnd);
        loopBase = loopEnd;
    }
}

template <typename DataType, typename GType>
__aicore__ inline void
ChunkBwdDqkwgVectorProcess<DataType, GType>::CopyGateWithPad(const LocalTensor<GType> &dst, const GlobalTensor<GType> &src,
                                                             uint64_t offset, uint32_t validLen, uint32_t totalLen)
{
    if (validLen == totalLen) {
        DataCopy(dst, src[offset], totalLen);
        return;
    }
    Duplicate(dst, static_cast<GType>(0), totalLen);
    TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    if (validLen == 0) {
        return;
    }

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLen * sizeof(GType)), 0, 0, 0};
    constexpr uint32_t elemsPerBlock = UB_BLOCK_SIZE / sizeof(GType);
    uint8_t rightPadding = static_cast<uint8_t>((elemsPerBlock - (validLen % elemsPerBlock)) % elemsPerBlock);
    DataCopyPadExtParams<GType> padParams{true, 0, rightPadding, 0};
    DataCopyPad(dst, src[offset], copyParams, padParams);
}

// mul1 一个 row-half 的计算: 输出 fp32 因果掩码 exp 到 outFp32 (该 half 的 [real_BT, BT], 行优先)。
// 内核与 ProcessAVector Part2 完全一致 (同 buffer: inQue3 g / calcBuf1 brcb / calcBuf2,3 g±)。
// 调用方需先 InitBuffer 这些 buffer 并构建 tensorMaskA。outFp32 必须指向该 half 的行起点。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ComputeMul1HalfFp32(
    const LocalTensor<float> &outFp32, const LocalTensor<float> &tensorMaskA, 
    const LocalTensor<float> &tensorBrcbTemp, const LocalTensor<float> &tensorGFp32Right, 
    uint64_t gOffset, uint32_t BT_sub_start, uint32_t real_BT, uint32_t actual_chunk_len)
{
    if (real_BT == 0) {
        return;
    }
    {
        auto tensorGIn = inQue3.AllocTensor<GType>();
        if constexpr (std::is_same<GType, float>::value) {
            CopyGateWithPad(tensorGIn, gmG, gOffset, actual_chunk_len, BT);
        } else {
            CopyGateWithPad(tensorGIn[BT], gmG, gOffset, actual_chunk_len, BT);
        }
        inQue3.EnQue(tensorGIn);
    }
    auto tensorGIn = inQue3.DeQue<GType>();
    auto tensorGFp32Left = tensorGIn.template ReinterpretCast<float>();
    if constexpr (!std::is_same<GType, float>::value) {
        Cast(tensorGFp32Left, tensorGIn[BT], RoundMode::CAST_NONE, BT);
    }
    PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (BT == 64) {
        if (BT_sub_start == 0) {
            Mul1Half<64, 0>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                            (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        } else {
            Mul1Half<64, 32>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                             (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        }
    } else {
        if (BT_sub_start == 0) {
            Mul1Half<128, 0>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                             (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        } else {
            Mul1Half<128, 64>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                              (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        }
    }
#else
    Muls(tensorGFp32Right, tensorGFp32Left, static_cast<float>(-1), BT);
    Brcb(tensorBrcbTemp, tensorGFp32Left[BT_sub_start], CEIL_DIV(real_BT, 8), {1, 8});
    PipeBarrier<PIPE_V>();
    if (BT == 64) {
        AscendC::Add(outFp32, tensorGFp32Right, tensorBrcbTemp, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 8, 0, 1});
        PipeBarrier<PIPE_V>();
        Mins(outFp32, outFp32, static_cast<float>(0.0), real_BT * BT);
        PipeBarrier<PIPE_V>();
        Exp(outFp32, outFp32, real_BT * BT);
        PipeBarrier<PIPE_V>();
    } else {
        AscendC::Copy(outFp32, tensorGFp32Right, CAL_NUM_FLOAT, real_BT, {1, 1, 16, 0});
        PipeBarrier<PIPE_V>();
        AscendC::Copy(outFp32[CAL_NUM_FLOAT], tensorGFp32Right[CAL_NUM_FLOAT], CAL_NUM_FLOAT, real_BT, {1, 1, 16, 0});
        PipeBarrier<PIPE_V>();
        AscendC::Add(outFp32, outFp32, tensorBrcbTemp, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        AscendC::Add(outFp32[CAL_NUM_FLOAT], outFp32[CAL_NUM_FLOAT], tensorBrcbTemp, CAL_NUM_FLOAT, real_BT,
                     {1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        Mins(outFp32, outFp32, static_cast<float>(0.0), real_BT * BT);
        PipeBarrier<PIPE_V>();
        Exp(outFp32, outFp32, real_BT * BT);
        PipeBarrier<PIPE_V>();
    }
    if (BT == 64) {
        Mul(outFp32, outFp32, tensorMaskA[BT_sub_start * 64], 32 * 64);
        PipeBarrier<PIPE_V>();
    } else {
        BinaryRepeatParams binaryRepeatParams{1, 1, 1, 16, 16, 8};
        UnaryRepeatParams unaryRepeatParams{1, 1, 16, 8};
        if (BT_sub_start == 0) {
            Mul(outFp32, outFp32, tensorMaskA, 64, 64, binaryRepeatParams);
            PipeBarrier<PIPE_V>();
            Muls(outFp32[64], outFp32[64], static_cast<float>(0), 64, 64, unaryRepeatParams);
        } else {
            Mul(outFp32[64], outFp32[64], tensorMaskA, 64, 64, binaryRepeatParams);
        }
        PipeBarrier<PIPE_V>();
    }
    AscendC::Muls(outFp32, outFp32, static_cast<float>(scale), real_BT * BT);
    PipeBarrier<PIPE_V>();
#endif
    inQue3.FreeTensor(tensorGIn);
}

// ============== A_vector: Part1 (dw 取负 + dg_last) ==============
// 每 chunk: WaitCubeReady 一次 -> Part1 (head-split) -> SetVectorDone 一次。
// 依赖: Part1 读 dw(由 A_cube 产出); mul1 改由 B_vector 内联计算, 不再预写 GM。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessAVector(uint32_t loopBase, uint32_t loopEnd)
{
    const uint32_t hDhSize = mul0RowNum * V; // 16 * 256 or 32 * 128 = 4k
    const uint32_t dwSize = BT * K;

    // ----- Part1 buffers (h/dh, dw) -----
    uint32_t bos = 0;
    uint32_t eos = 0;
    for (uint32_t loopIdx = loopBase; loopIdx < loopEnd; loopIdx += coreNum) {
        uint32_t bIdx = loopIdx / numChunks;
        uint32_t chunkIdx = loopIdx % numChunks;
        GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, B, HV, T, BT, loopIdx, bos, eos, isVarLen);
        uint32_t real_BT = eos - bos;
        uint32_t actual_dwSize = real_BT * K;
        WaitCubeReady();

        // ---------- Part1: dg_last = sum(h*dh), dw = -dw (head-split) ----------
        {
            for (uint32_t h = 0; h < HV; h++) {
                if ((h & 1) != subBlockIdx) {
                    continue;
                }
                uint64_t hOffset = ((bIdx * HV + h) * numChunks + chunkIdx) * K * V;
                uint64_t dwOffset = (h * T + bos) * K; // 最终输出 ptrDw 仍全局寻址
                uint64_t dgLastOffset = DqkwgScalarRingElemOffset(coreIdx, loopBase, loopIdx, coreNum, h, HV,
                                                                  (uint32_t)wsBtxKSyncSlotsPerHead);

                // ===== dg_last = sum(h * dh) =====
                // CV 融合优化: 在 cube-bound 的 stage A 算 (被 cube 的 2H 个 matmul 藏住), 写 wsDgLast;
                //   vector-bound 的 D_vector 改为读 wsDgLast, 省掉 D 的 h/dh 读 + K*V 归约 (给瓶颈减负)。
                auto tensorSumFp32 = inQue1.AllocTensor<float>();
                for (uint32_t row = 0; row < K; row += mul0RowNum) {
                    {
                        auto tensorHFp32 = inQue2.AllocTensor<float>();
                        // 拷入后半部分，给后面Cast预留空间
                        auto tensorHIn = tensorHFp32[hDhSize].template ReinterpretCast<DataType>();
                        DataCopy(tensorHIn, gmH[hOffset + row * V], hDhSize);
                        DataCopy(tensorHIn[hDhSize], gmDh[hOffset + row * V], hDhSize);
                        inQue2.EnQue(tensorHFp32);
                    }
                    {
                        auto tensorHFp32 = inQue2.DeQue<float>();
                        auto tensorDhFp32 = tensorHFp32[hDhSize];
                        auto tensorHIn = tensorDhFp32.template ReinterpretCast<DataType>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                        if (row == 0) {
                            DgLastMulAccum<DataType, false>(
                                (__ubuf__ float *)tensorSumFp32.GetPhyAddr(), (__ubuf__ DataType *)tensorHIn.GetPhyAddr(),
                                (__ubuf__ DataType *)tensorHIn[hDhSize].GetPhyAddr(), hDhSize);
                        } else {
                            DgLastMulAccum<DataType, true>(
                                (__ubuf__ float *)tensorSumFp32.GetPhyAddr(), (__ubuf__ DataType *)tensorHIn.GetPhyAddr(),
                                (__ubuf__ DataType *)tensorHIn[hDhSize].GetPhyAddr(), hDhSize);
                        }
#else
                        Cast(tensorHFp32, tensorHIn, RoundMode::CAST_NONE, 2 * hDhSize);
                        PipeBarrier<PIPE_V>();
                        if (row == 0) {
                            Mul(tensorSumFp32, tensorHFp32, tensorDhFp32, hDhSize);
                        } else {
                            Mul(tensorHFp32, tensorHFp32, tensorDhFp32, hDhSize);
                            PipeBarrier<PIPE_V>();
                            Add(tensorSumFp32, tensorSumFp32, tensorHFp32, hDhSize);
                        }
                        PipeBarrier<PIPE_V>();
#endif
                        inQue2.FreeTensor(tensorHFp32);
                    }
                }
                {
                    auto tensorDgLastOut = outQue2.AllocTensor<float>();
                    ReduceSumCustom(tensorDgLastOut, tensorSumFp32, hDhSize);
                    PipeBarrier<PIPE_V>();
                    inQue1.FreeTensor(tensorSumFp32);
                    outQue2.EnQue(tensorDgLastOut);
                }
                {
                    auto tensorDgLastOut = outQue2.DeQue<float>();
                    DataCopyPad(gmDgLast[dgLastOffset], tensorDgLastOut, {1, sizeof(float), 0, 0});
                    outQue2.FreeTensor(tensorDgLastOut);
                }

                // ===== dw = -dw, then vector-repair row-0 first block =====
                {
                    auto tensorDwIn = inQue1.AllocTensor<DataType>();
                    DataCopy(tensorDwIn[dwSize], gmDw[dwOffset], actual_dwSize);
                    inQue1.EnQue(tensorDwIn);
                }
                {
                    auto tensorDwIn = inQue1.DeQue<DataType>();
                    auto tensorDwOut = outQue1.AllocTensor<DataType>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                    DwNegate<DataType>((__ubuf__ DataType *)tensorDwOut.GetPhyAddr(),
                                       (__ubuf__ DataType *)tensorDwIn[dwSize].GetPhyAddr(),
                                       actual_dwSize);
#else
                    auto tensorDwInFp32 = tensorDwIn.template ReinterpretCast<float>();
                    Cast(tensorDwInFp32, tensorDwIn[dwSize], RoundMode::CAST_NONE, actual_dwSize);
                    PipeBarrier<PIPE_V>();
                    Muls(tensorDwInFp32, tensorDwInFp32, -1.0f, actual_dwSize);
                    PipeBarrier<PIPE_V>();
                    Cast(tensorDwOut, tensorDwInFp32, RoundMode::CAST_RINT, actual_dwSize);
                    PipeBarrier<PIPE_V>();
#endif
                    inQue1.FreeTensor(tensorDwIn);
                    outQue1.EnQue(tensorDwOut);
                }
                {
                    auto tensorDwOut = outQue1.DeQue<DataType>();
                    DataCopy(gmDw[dwOffset], tensorDwOut, actual_dwSize);
                    outQue1.FreeTensor(tensorDwOut);
                }
            }
        }
        SetVecCredit();
    }
}

// ============== B_vector: 原 Part3 (ds_temp + dg 部分) ==============
// 每 chunk: WaitCubeReady 一次 -> head-split 处理 (读 ds(B_cube), 内联算 mul1, 读 mm5(A_cube)) -> SetVectorDone 一次。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessBVector(uint32_t loopBase, uint32_t loopEnd)
{
    auto tensorBrcbTemp = calcBuf1.Get<float>();
    auto tensorGFp32Right = calcBuf3.Get<float>();

    uint32_t bos = 0;
    uint32_t eos = 0;
    for (uint32_t loopIdx = loopBase; loopIdx < loopEnd; loopIdx += coreNum) {
        GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, B, HV, T, BT, loopIdx, bos, eos, isVarLen);
        uint32_t real_BT = eos - bos;
        uint32_t actual_dsSize = real_BT * BT;
        WaitCubeReady();

        for (uint32_t h = 0; h < HV; h++) {
            if ((h & 1) != subBlockIdx) {
                continue;
            }
            uint64_t gOffset = (h * T + bos);
            uint64_t dsOffset = DqkwgBtbRingElemOffset(coreIdx, loopBase, loopIdx, coreNum, h, HV, BT,
                                                       (uint32_t)wsBtxKSyncSlotsPerHead);
            uint64_t mm5Offset = DqkwgBtxKRingElemOffset(coreIdx, loopBase, loopIdx, coreNum, h, HV, BT, K,
                                                         (uint32_t)wsBtxKSyncSlotsPerHead);
            uint64_t dgOffset = gOffset;

            {
                auto tensorDsIn = inQue1.AllocTensor<DataType>();
                DataCopy(tensorDsIn[BT * BT], gmDsTemp[dsOffset], actual_dsSize);
                inQue1.EnQue(tensorDsIn);
            }
            {
                auto tensorDsInFp16 = inQue1.DeQue<DataType>();
                auto tensorDsInFp32 = tensorDsInFp16.template ReinterpretCast<float>();
                auto tensorDsTempOut = outQue1.AllocTensor<DataType>();
                auto tensorDgOut = outQue2.AllocTensor<float>();

                Cast(tensorDsInFp32, tensorDsInFp16[BT * BT], RoundMode::CAST_NONE, actual_dsSize);

                // mul1 从 g 内联算两个 row-half, 写入 inQue2 buffer (fp32)。
                auto tensorMul1InFp32 = inQue2.AllocTensor<float>();
                uint32_t vec0 = real_BT >= (BT / 2) ? (BT / 2) : real_BT;
                uint32_t rb1 = (real_BT > vec0) ? (real_BT - vec0) : 0;
                ComputeMul1HalfFp32(tensorMul1InFp32, tensorMaskA, tensorBrcbTemp, tensorGFp32Right, gOffset, 0, vec0, real_BT);
                if (rb1 > 0) {
                    ComputeMul1HalfFp32(tensorMul1InFp32[vec0 * BT], tensorMaskA, tensorBrcbTemp, tensorGFp32Right, gOffset, vec0, rb1, real_BT);
                }
                PipeBarrier<PIPE_V>();
                // b_ds_temp = b_ds * mul1 (已应用掩码)
                Mul(tensorDsInFp32, tensorDsInFp32, tensorMul1InFp32, actual_dsSize);
                inQue2.FreeTensor(tensorMul1InFp32);

                // 搬入 mm5, 复用 mul1 空间
                auto tensorMm5InFp16Tmp = inQue2.AllocTensor<DataType>();
                DataCopy(tensorMm5InFp16Tmp[BT * BT], gmMm5[mm5Offset], actual_dsSize);
                inQue2.EnQue(tensorMm5InFp16Tmp);
                auto tensorMm5InFp16 = inQue2.DeQue<DataType>();
                auto tensorMm5InFp32 = tensorMm5InFp16.template ReinterpretCast<float>();
                Cast(tensorMm5InFp32, tensorMm5InFp16[BT * BT], RoundMode::CAST_NONE, actual_dsSize);
                PipeBarrier<PIPE_V>();

                Mul(tensorMm5InFp32, tensorDsInFp32, tensorMm5InFp32, actual_dsSize);       // b_ds2 = b_ds_temp * mm5
                Cast(tensorDsTempOut, tensorDsInFp32, RoundMode::CAST_RINT, actual_dsSize); // ds_temp -> fp16

                Duplicate(tensorDgOut, static_cast<float>(0.0), BT);
                PipeBarrier<PIPE_V>();

                // 行求和 -> [BT] (+Add0.C)
                uint64_t wholeReduceSumCnt = CeilDiv(real_BT, FP32_PER_REPEAT);
                uint32_t remainCnt = real_BT % FP32_PER_REPEAT;
                if (remainCnt > 0) {
                    uint32_t DuplicateOffset = wholeReduceSumCnt * FP32_PER_REPEAT - FP32_PER_REPEAT;
                    uint64_t mask[1] = {0xffffffffffffffff};
                    mask[0] <<= remainCnt;
                    for (uint32_t row = 0; row < real_BT; row++) {
                        Duplicate(tensorMm5InFp32[row * BT + DuplicateOffset], 0.0f, mask, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t i = 0; i < real_BT; i++) {
                    WholeReduceSum(tensorDsInFp32[i * 8], tensorMm5InFp32[i * BT], FP32_PER_REPEAT, wholeReduceSumCnt,
                                   1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut, tensorDsInFp32, wholeReduceSumCnt, real_BT, 1, 1, 1);

                // 列求和 -> [BT] (-Add0.D)
                PipeBarrier<PIPE_V>();
                uint32_t remain_row = real_BT;
                uint32_t CalcCnt = 0;
                uint32_t Offset = 0;
                while (remain_row > 1) {
                    CalcCnt = (remain_row / 2) * BT;
                    remain_row = CeilDiv(remain_row, 2);
                    Offset = remain_row * BT;
                    Add(tensorMm5InFp32, tensorMm5InFp32, tensorMm5InFp32[Offset], CalcCnt);
                    PipeBarrier<PIPE_V>();
                }
                Sub(tensorDgOut, tensorDgOut, tensorMm5InFp32, BT);
                PipeBarrier<PIPE_V>();
                if constexpr (!std::is_same<GType, float>::value) {
                    Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, BT);
                }

                inQue1.FreeTensor(tensorDsInFp16);
                inQue2.FreeTensor(tensorMm5InFp16);
                outQue1.EnQue(tensorDsTempOut);
                outQue2.EnQue(tensorDgOut);
            }
            {
                auto tensorDsTempOut = outQue1.DeQue<DataType>();
                auto tensorDgOut = outQue2.DeQue<GType>();
                DataCopy(gmDsTemp[dsOffset], tensorDsTempOut, actual_dsSize);
                DataCopyPad(gmDg[dgOffset], tensorDgOut, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
                outQue1.FreeTensor(tensorDsTempOut);
                outQue2.FreeTensor(tensorDgOut);
            }
        }
        SetVecCredit();
    }
}

// ============== C_vector: 原 Part4 + Part6 (dq 最终 + dg) ==============
// C_cube 在一次 SetCubeReady 前已产出 dq_inner(ptrDq) 与 mm6(wsMm6); C_vector 一次 WaitCubeReady 后
// 同时读取二者 (无 per-head phase 握手), 计算 dq_state/dg 并 dq += mm6。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessCVector(uint32_t loopBase, uint32_t loopEnd)
{
    auto tensorShareTmpFp32 = calcBuf1.Get<float>();
    uint32_t bos = 0;
    uint32_t eos = 0;
    for (uint32_t loopIdx = loopBase; loopIdx < loopEnd; loopIdx += coreNum) {
        GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, B, HV, T, BT, loopIdx, bos, eos, isVarLen);
        uint32_t real_BT = eos - bos;
        uint32_t actual_dqSize = real_BT * K;
        WaitCubeReady();

        for (uint32_t h = 0; h < HV; h++) {
            uint32_t hk_idx = h / n_ratio;
            uint32_t ownerHead = gvaMode ? hk_idx : h;
            if ((ownerHead & 1) != subBlockIdx) {
                continue;
            }
            uint32_t bIdx = loopIdx / numChunks;
            uint64_t bos_hk = bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(HV - HK) * T;
            uint64_t qkOffset = (hk_idx * T + bos_hk) * K;    // q (HK)
            uint64_t dqOutOffset = (hk_idx * T + bos_hk) * K; // dq (HK)
            uint64_t gOffset = (h * T + bos);

            // CopyIn: dq_inner, q, g, dg
            {
                auto tensorDqIn = inQue1.AllocTensor<DataType>();
                auto tensorQIn = inQue2.AllocTensor<DataType>();
                auto tensorGIn = inQue3.AllocTensor<GType>();
                auto tensorDgIn = inQue4.AllocTensor<GType>();
                DataCopy(tensorDqIn[actual_dqSize], gmDq[dqOutOffset], actual_dqSize);
                DataCopy(tensorQIn[actual_dqSize], gmQ[qkOffset], actual_dqSize);
                if constexpr (std::is_same<GType, float>::value) {
                    CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                    CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
                } else {
                    CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                    CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
                }
                inQue1.EnQue(tensorDqIn);
                inQue2.EnQue(tensorQIn);
                inQue3.EnQue(tensorGIn);
                inQue4.EnQue(tensorDgIn);
            }
            {
                auto tensorDqInFp16 = inQue1.DeQue<DataType>();
                auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
                auto tensorQInFp16 = inQue2.DeQue<DataType>();
                auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();
                auto tensorGIn = inQue3.DeQue<GType>();
                auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
                auto tensorDgIn = inQue4.DeQue<GType>();
                auto tensorDgAdd = tensorDgIn.template ReinterpretCast<float>();
                auto tensorDgOut = outQue2.AllocTensor<float>();

                Cast(tensorDqInFp32, tensorDqInFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
                Cast(tensorQInFp32, tensorQInFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
                if constexpr (!std::is_same<GType, float>::value) {
                    Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, BT);
                    Cast(tensorDgAdd, tensorDgIn[BT], RoundMode::CAST_NONE, BT);
                }
                PipeBarrier<PIPE_V>();

                // dq_state = dq_inner * exp(g)[:,None] * scale
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                DqState((__ubuf__ float *)tensorDqInFp32.GetPhyAddr(), (__ubuf__ float *)tensorDqInFp32.GetPhyAddr(), 
                        (__ubuf__ float *)tensorGFp32.GetPhyAddr(), (__ubuf__ float *)tensorShareTmpFp32.GetPhyAddr(), real_BT, scale);
#else
                Exp(tensorGFp32, tensorGFp32, BT);
                PipeBarrier<PIPE_V>();
                Muls(tensorGFp32, tensorGFp32, scale, BT);
                PipeBarrier<PIPE_V>();
                Brcb(tensorShareTmpFp32, tensorGFp32, CEIL_DIV(real_BT, 8), {1, 8});
                PipeBarrier<PIPE_V>();
                AscendC::Mul(tensorDqInFp32, tensorDqInFp32, tensorShareTmpFp32, CAL_NUM_FLOAT, real_BT,
                             {1, 1, 0, 16, 16, 1});
                AscendC::Mul(tensorDqInFp32[CAL_NUM_FLOAT], tensorDqInFp32[CAL_NUM_FLOAT], tensorShareTmpFp32,
                             CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
                PipeBarrier<PIPE_V>();
#endif

                // dg_C = row_sum(dq_state * q)
                Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, actual_dqSize);
                PipeBarrier<PIPE_V>();
                Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, real_BT, {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut, tensorQInFp32, FP32_PER_REPEAT, real_BT, 1, 1, 16);
                PipeBarrier<PIPE_V>();
                Add(tensorDgOut, tensorDgAdd, tensorDgOut, real_BT);
                PipeBarrier<PIPE_V>();
                if constexpr (!std::is_same<GType, float>::value) {
                    Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, BT);
                }

                inQue2.FreeTensor(tensorQInFp16);
                inQue3.FreeTensor(tensorGIn);
                inQue4.FreeTensor(tensorDgIn);
                outQue2.EnQue(tensorDgOut);

                // dg 写回
                {
                    auto tensorDgOutDeq = outQue2.DeQue<GType>();
                    DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = 1;
                    dataCopyParams.blockLen = real_BT * sizeof(GType);
                    dataCopyParams.srcStride = 0;
                    dataCopyParams.dstStride = 0;
                    DataCopyPad(gmDg[gOffset], tensorDgOutDeq, dataCopyParams);
                    outQue2.FreeTensor(tensorDgOutDeq);
                }

                // dq += mm6 (从 wsMm6 环形区读取, dq_state 仍在 UB)
                {
                    auto tensorMm6In = inQue2.AllocTensor<DataType>();
                    uint64_t mm6RingOffset =
                        DqkwgShortBtxKRingElemOffset(coreIdx, loopIdx, coreNum, h, HV, BT, K,
                                                     DqkwgShortRingDepthFromGroup((uint32_t)wsBtxKSyncSlotsPerHead));
                    DataCopy(tensorMm6In[actual_dqSize], gmMm6[mm6RingOffset], actual_dqSize); // mm6 compact ring
                    inQue2.EnQue(tensorMm6In);
                }
                {
                    auto tensorMm6InFp16 = inQue2.DeQue<DataType>();
                    auto tensorMm6Fp32 = tensorMm6InFp16.template ReinterpretCast<float>();
                    Cast(tensorMm6Fp32, tensorMm6InFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
                    PipeBarrier<PIPE_V>();
                    Add(tensorDqInFp32, tensorDqInFp32, tensorMm6Fp32, actual_dqSize);
                    PipeBarrier<PIPE_V>();
                    inQue2.FreeTensor(tensorMm6InFp16);
                    auto tensorDqOut = outQue1.AllocTensor<DataType>();
                    Cast(tensorDqOut, tensorDqInFp32, RoundMode::CAST_RINT, actual_dqSize);
                    inQue1.FreeTensor(tensorDqInFp16);
                    outQue1.EnQue(tensorDqOut);
                }
                {
                    auto tensorDqOut = outQue1.DeQue<DataType>();
                    DataCopy(gmDq[dqOutOffset], tensorDqOut, actual_dqSize);
                    outQue1.FreeTensor(tensorDqOut);
                }
            }
        }
        SetVecCredit();
    }
}

// ============== D_vector: 原 Part5 + Part7 (dk 最终 + dg 最终) ==============
// D_cube 在一次 SetCubeReady 前已产出 dk_inner(ptrDk) 与 mm7(wsMm7); D_vector 一次 WaitCubeReady 后
// 读取二者, 完成 dk_state / dg 最终 / dk += mm7。dg_last 本地重算 (与原 Part5 一致)。
template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessDVector(uint32_t loopBase, uint32_t loopEnd)
{
    auto tensorDgFinal = dgBuf.Get<float>();
    auto tensorGLastFp32Tmp = calcBuf1.Get<float>();
    auto tensorDgLastFp32Tmp = calcBuf2.Get<float>();

    uint32_t bos = 0;
    uint32_t eos = 0;
    for (uint32_t loopIdx = loopBase; loopIdx < loopEnd; loopIdx += coreNum) {
        GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, B, HV, T, BT, loopIdx, bos, eos, isVarLen);
        uint32_t real_BT = eos - bos;
        uint32_t real_BT_aligned = (real_BT + 15) / 16 * 16;
        uint32_t actual_dkSize = real_BT * K;
        uint32_t bIdx = loopIdx / numChunks;
        WaitCubeReady();

        for (uint32_t h = 0; h < HV; h++) {
            uint32_t hk_idx = h / n_ratio;
            uint32_t ownerHead = gvaMode ? hk_idx : h;
            if ((ownerHead & 1) != subBlockIdx) {
                continue;
            }
            uint64_t bos_hk = bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(HV - HK) * T;
            uint64_t kOffset = (hk_idx * T + bos_hk) * K;     // k (HK)
            uint64_t dkOutOffset = (hk_idx * T + bos_hk) * K; // dk (HK)
            uint64_t gOffset = (h * T + bos);

            // CV 融合优化: 读 A_vector 算好的 dg_last = sum(h*dh) (替代本地重算, 省 D 的 h/dh 读 + K*V 归约)。
            // 跨 stage 可见性由 Process() 中 A->B->C->D 的 PipeBarrier<PIPE_ALL> 保证; A(c,h)/D(c,h) 同 sub-block。
            // 输出格式与原重算一致: tensorGLastFp32Tmp[16..23] = 8 份 dg_last。
            {
                uint64_t dgLastOffset = DqkwgScalarRingElemOffset(coreIdx, loopBase, loopIdx, coreNum, h, HV,
                                                                  (uint32_t)wsBtxKSyncSlotsPerHead);
                {
                    DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
                    DataCopyPadExtParams<float> padParams{true, 0, 7, 0};
                    DataCopyPad(tensorDgLastFp32Tmp[8], gmDgLast[dgLastOffset], copyParams, padParams);
                }
                TEventID eDg = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(eDg);
                WaitFlag<HardEvent::MTE2_V>(eDg);
                Brcb(tensorGLastFp32Tmp[16], tensorDgLastFp32Tmp[8], 1, {1, 8}); // 广播到 [16..23] 8 份
                PipeBarrier<PIPE_V>();
            }

            // CopyIn: dk_inner, k, g, dg
            {
                auto tensorDkIn = inQue1.AllocTensor<DataType>();
                auto tensorKIn = inQue2.AllocTensor<DataType>();
                auto tensorGIn = inQue3.AllocTensor<GType>();
                auto tensorDgIn = inQue4.AllocTensor<GType>();
                DataCopy(tensorDkIn[actual_dkSize], gmDk[dkOutOffset], actual_dkSize);
                DataCopy(tensorKIn[actual_dkSize], gmK[kOffset], actual_dkSize);
                if constexpr (std::is_same<GType, float>::value) {
                    CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                    CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
                } else {
                    CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                    CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
                }
                inQue1.EnQue(tensorDkIn);
                inQue2.EnQue(tensorKIn);
                inQue3.EnQue(tensorGIn);
                inQue4.EnQue(tensorDgIn);
            }
            {
                auto tensorDkIn = inQue1.DeQue<DataType>();
                auto tensorDkFp32 = tensorDkIn.template ReinterpretCast<float>();
                auto tensorKIn = inQue2.DeQue<DataType>();
                auto tensorKFp32 = tensorKIn.template ReinterpretCast<float>();
                auto tensorGIn = inQue3.DeQue<GType>();
                auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
                auto tensorDgIn = inQue4.DeQue<GType>();
                auto tensorDgTmp = tensorDgIn.template ReinterpretCast<float>();
                auto tensorDgOut = outQue2.AllocTensor<GType>();

                Cast(tensorDkFp32, tensorDkIn[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
                Cast(tensorKFp32, tensorKIn[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
                if constexpr (!std::is_same<GType, float>::value) {
                    Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
                    Cast(tensorDgTmp, tensorDgIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
                }
                PipeBarrier<PIPE_V>();

                // MUL2: dk_state = dk_inner * exp(-g + g_last)[:,None]
                uint32_t last_line_no = (real_BT - 1) / 8 * 8;
                uint32_t last_line_idx = real_BT - 1 - last_line_no;
                Brcb(tensorDgFinal, tensorGFp32[last_line_no], 1, {1, 8}); // [8,8] 第 last_line_idx 行 = gLast
                PipeBarrier<PIPE_V>();
                Muls(tensorGFp32, tensorGFp32, -1.0f, real_BT_aligned);
                DataCopy(tensorGLastFp32Tmp, tensorDgFinal[last_line_idx * 8], 8);
                PipeBarrier<PIPE_V>();
                AscendC::Add(tensorGFp32, tensorGFp32, tensorDgFinal[last_line_idx * 8], CAL_NUM_FLOAT,
                             BT / CAL_NUM_FLOAT, {1, 1, 0, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                Exp(tensorGFp32, tensorGFp32, real_BT_aligned);
                PipeBarrier<PIPE_V>();

                Brcb(tensorDgFinal, tensorGFp32, CEIL_DIV(real_BT, 8), {1, 8});
                PipeBarrier<PIPE_V>();
                AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
                AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                             real_BT, {1, 1, 0, 16, 16, 1});
                PipeBarrier<PIPE_V>();

                Mul(tensorKFp32, tensorKFp32, tensorDkFp32, actual_dkSize); // mul8 = dk_state * k
                PipeBarrier<PIPE_V>();

                // Add0.B = row_sum(dk_state * k)
                Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, real_BT, {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorGFp32, tensorKFp32, FP32_PER_REPEAT, real_BT, 1, 1, 16);
                PipeBarrier<PIPE_V>();

                // Sum0: [real_BT] -> [1]
                if (real_BT > FP32_PER_REPEAT) {
                    Add(tensorGFp32[BT], tensorGFp32, tensorGFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT);
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(tensorDgFinal, tensorGFp32[BT], FP32_PER_REPEAT, 1, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                } else {
                    WholeReduceSum(tensorDgFinal, tensorGFp32, real_BT, 1, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                }
                Brcb(tensorDgLastFp32Tmp, tensorDgFinal, 1, {1, 8});
                PipeBarrier<PIPE_V>();
                Exp(tensorGLastFp32Tmp, tensorGLastFp32Tmp, 8);
                PipeBarrier<PIPE_V>();
                Mul(tensorDgLastFp32Tmp[16], tensorGLastFp32Tmp[16], tensorGLastFp32Tmp, 8);
                PipeBarrier<PIPE_V>();
                Add(tensorDgLastFp32Tmp[16], tensorDgLastFp32Tmp[16], tensorDgLastFp32Tmp, 8); // add4 = dg_last_term

                Sub(tensorGFp32, tensorDgTmp, tensorGFp32, BT); // Add.0 最终结果 (dg_B+dg_C+dg_D)
                PipeBarrier<PIPE_V>();
                Brcb(tensorDgFinal, tensorDgLastFp32Tmp[16], 1, {1, 8});
                PipeBarrier<PIPE_V>();
                uint64_t offset = (real_BT - 1) / 8 * 8;
                uint64_t mask[1] = {0};
                mask[0] = 1ULL << (real_BT - 1 - offset); // 仅最后一个位置加 dg_last_term
                Add(tensorGFp32[offset], tensorGFp32[offset], tensorDgFinal, mask, 1, {1, 1, 1, 8, 8, 8});
                PipeBarrier<PIPE_V>();
                if constexpr (std::is_same<GType, float>::value) {
                    DataCopy(tensorDgOut, tensorGFp32, BT);
                } else {
                    Cast(tensorDgOut, tensorGFp32, RoundMode::CAST_RINT, BT);
                }
                PipeBarrier<PIPE_V>();

                inQue2.FreeTensor(tensorKIn);
                inQue3.FreeTensor(tensorGIn);
                inQue4.FreeTensor(tensorDgIn);
                outQue2.EnQue(tensorDgOut);
                {
                    auto tensorDgOutDeq = outQue2.DeQue<GType>();
                    DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = 1;
                    dataCopyParams.blockLen = real_BT * sizeof(GType);
                    dataCopyParams.srcStride = 0;
                    dataCopyParams.dstStride = 0;
                    DataCopyPad(gmDg[gOffset], tensorDgOutDeq, dataCopyParams);
                    outQue2.FreeTensor(tensorDgOutDeq);
                }

                // dk += mm7 (从 wsMm7 读取, dk_state 仍在 UB)
                {
                    auto tensorMm7In = inQue2.AllocTensor<DataType>();
                    uint64_t mm7RingOffset =
                        DqkwgBtxKRingElemOffset(coreIdx, loopBase, loopIdx, coreNum, h, HV, BT, K,
                                                (uint32_t)wsBtxKSyncSlotsPerHead); // mm7 用 group 环 (与 cube 一致)
                    DataCopy(tensorMm7In[actual_dkSize], gmMm7[mm7RingOffset], actual_dkSize);
                    inQue2.EnQue(tensorMm7In);
                }
                {
                    auto tensorMm7In = inQue2.DeQue<DataType>();
                    auto tensorMm7Fp32 = tensorMm7In.template ReinterpretCast<float>();
                    Cast(tensorMm7Fp32, tensorMm7In[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
                    PipeBarrier<PIPE_V>();
                    Add(tensorDkFp32, tensorDkFp32, tensorMm7Fp32, actual_dkSize);
                    PipeBarrier<PIPE_V>();
                    inQue2.FreeTensor(tensorMm7In);
                    auto tensorDkOut = outQue1.AllocTensor<DataType>();
                    Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, actual_dkSize);
                    inQue1.FreeTensor(tensorDkIn);
                    outQue1.EnQue(tensorDkOut);
                }
                {
                    auto tensorDkOut = outQue1.DeQue<DataType>();
                    DataCopy(gmDk[dkOutOffset], tensorDkOut, actual_dkSize);
                    outQue1.FreeTensor(tensorDkOut);
                }
            }
        }
        SetVecCredit();
    }
}

#endif // CHUNK_BWD_DQKWG_VECTOR_H
