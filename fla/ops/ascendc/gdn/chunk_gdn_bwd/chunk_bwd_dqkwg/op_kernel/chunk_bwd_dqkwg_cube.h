/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_cube.h
 */

#ifndef CHUNK_BWD_DQKWG_CUBE_H
#define CHUNK_BWD_DQKWG_CUBE_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#include "chunk_bwd_dqkwg_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;
#else
#define CATLASS_ARCH 2201
#include "chunk_bwd_dqkwg_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#endif
using namespace tla;
using namespace Catlass;
using namespace AscendC;

namespace Catlass::Gemm::Kernel {

/**
 * TileGemmDirect: Single-shot tile-level GEMM replacing BlockMmadTla.
 * Directly performs GM→L1→L0→Mmad→FixPipe→GM without tiling loops or pingpong buffers.
 * Optimal when all matrix dimensions fit in a single L0 tile (M≤128, N≤128, K≤128).
 *
 * Compared to BlockMmadTla<MmadPingpong>:
 *  - No multi-stage L1/L0 buffers (1 stage vs 2)
 *  - No nested tiling loops (kL1×mL0×kL0×nL0 all = 1 for our sizes)
 *  - Minimal event overhead: 4 Set + 4 Wait per GEMM vs ~20+ in Block
 *  - Half the L1/L0 memory usage (no pingpong double-buffering)
 */
template <class ArchTag_, class ElementC_, class TileCopy_>
struct TileGemmDirect {
    using ArchTag = ArchTag_;
    using TileCopy = TileCopy_;
    using ElementA = typename TileCopy::ElementA;
    using ElementB = typename TileCopy::ElementB;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmadType = Tile::TileMmadTla<ArchTag, ElementA, LayoutTagL1A>;

    // Single L0 tile dimensions (max supported by AtlasA2 Cube)
    static constexpr uint32_t TILE_M = 128;
    static constexpr uint32_t TILE_N = 128;
    static constexpr uint32_t TILE_K = 128;

    // Single-stage buffer sizes (no pingpong)
    static constexpr uint32_t L1A_TILE_SIZE = TILE_M * TILE_K * sizeof(ElementA);

    // Static L1 layouts for tile-sized buffers
    static constexpr auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<TILE_M>{}, tla::Int<TILE_K>{});
    static constexpr auto L1B_LAYOUT = tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<TILE_K>{}, tla::Int<TILE_N>{});

    /// Construct: allocate single-stage buffers, initialize events
    CATLASS_DEVICE
    TileGemmDirect(Arch::Resource<ArchTag> &resource)
    {
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(true);

            // Allocate single-stage buffers from resource pools
            l1ABuf = resource.l1Buf.template GetBufferByByte<ElementA>(0);
            l1BBuf = resource.l1Buf.template GetBufferByByte<ElementB>(L1A_TILE_SIZE);
            l0ABuf = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0BBuf = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0CBuf = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

            // Initialize events: buffers are initially free
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0); // L1 free for MTE2 write
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);    // L0A/L0B free for MTE1 write
        }
    }

    /// Destructor: drain pending events
    CATLASS_DEVICE
    ~TileGemmDirect()
    {
        if ASCEND_IS_AIC {
            AscendC::SetMMLayoutTransform(false);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        }
    }

    /// Perform a single-tile GEMM: C = A @ B
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, GemmCoord const &actualShape)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<TensorC>;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<TensorC>;
#endif

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;
        TileMmadType tileMmad;

        uint32_t m = actualShape.m();
        uint32_t k = actualShape.k();
        uint32_t n = actualShape.n();

        // Avoid gemv mode on AtlasA2
        uint32_t mActual = m;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            if (mActual == 1)
                mActual = 16;
        }

        // Create L1 tensor views (static tile-size layout)
        auto tensorL1A = tla::MakeTensor(l1ABuf, L1A_LAYOUT, Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1BBuf, L1B_LAYOUT, Arch::PositionL1{});

        // ---- Step 1: GM → L1 (MTE2) ----
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0); // Wait L1 free
        copyGmToL1A(tensorL1A, tensorA);
        copyGmToL1B(tensorL1B, tensorB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0); // Signal L1 data ready

        // ---- Step 2: L1 → L0A/L0B (MTE1) ----
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0); // Wait L1 data ready
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);    // Wait L0A/L0B free

        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(mActual, k);
        auto tensorL0A = tla::MakeTensor(l0ABuf, layoutL0A, Arch::PositionL0A{});
        auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, k));
        copyL1ToL0A(tensorL0A, tensorTileL1A);

        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto tensorL0B = tla::MakeTensor(l0BBuf, layoutL0B, Arch::PositionL0B{});
        auto tensorTileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        copyL1ToL0B(tensorL0B, tensorTileL1B);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0); // Signal L1 free
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);    // Signal L0 data ready

        // ---- Step 3: Mmad (Cube) ----
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);

        auto layoutL0C = tla::MakeLayoutL0C(mActual, n);
        auto tensorL0C = tla::MakeTensor(l0CBuf, layoutL0C, Arch::PositionL0C{});

        tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, n, k, true, 0b11);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0); // Signal L0A/L0B free

        // ---- Step 4: L0C → GM (FixPipe, auto-triggered by unitFlag=0b11) ----
        copyL0CToGm(tensorC, tensorL0C, 0b11);
    }

private:
    AscendC::LocalTensor<ElementA> l1ABuf;
    AscendC::LocalTensor<ElementB> l1BBuf;
    AscendC::LocalTensor<ElementA> l0ABuf;
    AscendC::LocalTensor<ElementB> l0BBuf;
    AscendC::LocalTensor<ElementAccumulator> l0CBuf;
};

/**
 * ChunkBwdDqkwg Cube Kernel 模板类 (CV 深融合, chunk-interleaved A/B/C/D)
 *
 * 4 个 stage 的 cube GEMM:
 * - A_cube: Part1 dw = dv @ h^T  和  Part2 mm5 = q @ k^T
 * - B_cube: Part3 ds = do @ v^T
 * - C_cube: Part4 dq_inner = do @ h^T  和  Part6 mm6 = ds_temp @ k
 * - D_cube: Part5 dk_inner = v @ dh   和  Part7 mm7 = ds_temp^T @ q
 *
 * 模板参数 (沿用原 7-part 命名, 实际按 A/B/C/D 调度):
 * - BlockMmadPart1_: dv @ h^T -> dw
 * - BlockMmadPart2_: q @ k^T -> mm5
 * - BlockMmadPart3_: do @ v^T -> ds
 * - BlockMmadPart4_: do @ h^T -> dq
 * - BlockMmadPart5_: v @ dh -> dk
 * - BlockMmadPart6_: ds @ k -> dq+=
 * - BlockMmadPart7_: ds^T @ q -> dk+=
 */
template <class BlockMmadPart1_, // dv @ h^T -> dw
          class BlockMmadPart2_, // q @ k^T -> mm5
          class BlockMmadPart3_, // do @ v^T -> ds
          class BlockMmadPart4_, // do @ h^T -> dq
          class BlockMmadPart5_, // v @ dh -> dk
          class BlockMmadPart6_, // ds @ k -> dq
          class BlockMmadPart7_  // ds^T @ q -> dk
          >
class ChunkBwdDqkwgTla {
public:
    using BlockMmadPart1 = BlockMmadPart1_;
    using BlockMmadPart2 = BlockMmadPart2_;
    using BlockMmadPart3 = BlockMmadPart3_;
    using BlockMmadPart4 = BlockMmadPart4_;
    using BlockMmadPart5 = BlockMmadPart5_;
    using BlockMmadPart6 = BlockMmadPart6_;
    using BlockMmadPart7 = BlockMmadPart7_;

    using ArchTag = typename BlockMmadPart1::ArchTag;

    // 数据类型定义 (基于 Part 2: q @ k^T)
    using ElementA = typename BlockMmadPart2::ElementA;
    using ElementB = typename BlockMmadPart2::ElementB;
    using ElementC = typename BlockMmadPart2::ElementC;

    // Layout 定义
    using LayoutRowMajor = layout::RowMajor;
    using LayoutColMajor = layout::ColumnMajor;

    // CV 深融合: raw 信用流水, 无 flag 对象 (helper 用固定 flag id)。详见 chunk_bwd_dqkwg_common.h。

    /// Parameters structure
    struct Params {
        // 输入指针
        GM_ADDR ptrQ;  // [B, HK, T, K]
        GM_ADDR ptrK;  // [B, HK, T, K]
        GM_ADDR ptrV;  // [B, HV, T, V]
        GM_ADDR ptrH;  // [B, HV, num_chunks, K, V]
        GM_ADDR ptrDo; // [B, HV, T, V]
        GM_ADDR ptrDh; // [B, HV, num_chunks, K, V]
        GM_ADDR ptrDv; // [B, HV, T, V]
        // varlen
        GM_ADDR ptrCuSeqLens;
        GM_ADDR ptrChunkIndices;

        // 输出指针
        GM_ADDR ptrDq; // [B, HK, T, K]
        GM_ADDR ptrDk; // [B, HK, T, K]
        GM_ADDR ptrDw; // [B, HV, T, K]

        // Workspace 指针
        GM_ADDR ptrWorkspace;

        // Workspace 偏移
        uint64_t wsMm3Offset;
        uint64_t wsMm4Offset;
        uint64_t wsMm6Offset;
        uint64_t wsMm5Offset;
        uint64_t wsMm7Offset;
        uint64_t wsDsTempOffset;

        // 形状参数 (GVA: H 拆为 HV/HK, HV = n_ratio * HK)
        uint64_t B;              // = CONST_B;
        uint64_t HV;             // value 侧 head 数 (v/g/h/do/dh/dv/dw/dg), == 原 H
        uint64_t HK;             // key/query 侧 head 数 (q/k), HV = n_ratio * HK
        uint64_t T;              // = CONST_T;
        uint64_t K;              // = CONST_K;
        uint64_t V;              // = CONST_V;
        uint64_t BT;             // = CONST_BT;
        uint64_t numChunks;      // = CONST_NUM_CHUNKS;
        uint64_t n_ratio;        // HV / HK
        uint64_t aicCoreNum = 0; // CV 深融合 blockDim (cube/vector 共用), 由 host tiling 给出
        bool isVarLen = false;

        CATLASS_DEVICE
        Params(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen,
               GM_ADDR chunk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw,GM_ADDR workspace, uint64_t B, uint64_t HV, uint64_t HK,
               uint64_t T, uint64_t K, uint64_t V, uint64_t BT, uint64_t numChunks, uint64_t n_ratio, bool isVarLen,
               uint64_t wsMm3, uint64_t wsMm4, uint64_t wsMm6, uint64_t wsMm5, uint64_t wsMm7, uint64_t wsDsTemp)
            : ptrQ(q), ptrK(k), ptrV(v), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLens(cu_seqlen),
              ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrWorkspace(workspace),
              wsMm3Offset(wsMm3), wsMm4Offset(wsMm4), wsMm6Offset(wsMm6), wsMm5Offset(wsMm5),
              wsMm7Offset(wsMm7), wsDsTempOffset(wsDsTemp), B(B), HV(HV), HK(HK), T(T), K(K), V(V), BT(BT), 
              numChunks(numChunks), n_ratio(n_ratio), isVarLen(isVarLen)
        {
        }
    };

    // Global Tensors
    GlobalTensor<ElementA> gmQ;
    GlobalTensor<ElementA> gmK;
    GlobalTensor<ElementA> gmV;
    GlobalTensor<ElementA> gmH;
    GlobalTensor<ElementA> gmDo;
    GlobalTensor<ElementA> gmDh;
    GlobalTensor<ElementA> gmDv;
    GlobalTensor<ElementC> gmDq;
    GlobalTensor<ElementC> gmDk;
    GlobalTensor<ElementC> gmDw;
    GlobalTensor<ElementC> gmMm3;
    GlobalTensor<ElementC> gmMm4;
    GlobalTensor<ElementC> gmMm6;
    GlobalTensor<ElementC> gmMm5;
    GlobalTensor<ElementC> gmMm7;
    GlobalTensor<ElementC> gmDs;
    GlobalTensor<ElementA> gmDsTemp;
    GlobalTensor<uint64_t> cuSeqlensTensor;
    GlobalTensor<uint64_t> chunkIndicesTensor;

    CATLASS_DEVICE
    ChunkBwdDqkwgTla()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    /**
     * AIC (Cube) 执行入口 (CV 深融合, chunk-interleaved A/B/C/D, raw 信用流水)
     *
     * 每核任务流: A(c0..cM-1), B(c0..cM-1), C(c0..cM-1), D(c0..cM-1) (L = 4*M), 跨 stage 连续。
     * 每个 (stage, chunk) task: 先 WaitCredit (节流, cube 领先 vector <=N 个 task),
     * chunk 内全部 head 数据 ready 后 SetCubeReady 一次。vector 预置 N=min(groupSize, M) 个信用。
     * N<=M 保证 C_cube/D_cube 读 B_vector 产出的 ds_temp 时其已就绪。
     */
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Arch::Resource<ArchTag> resource;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = params.aicCoreNum; // CV 深融合 blockDim (cube/vector 共用)
        uint32_t coreLoops = params.B * params.numChunks;
        uint32_t K = static_cast<uint32_t>(params.K);
        uint32_t V = static_cast<uint32_t>(params.V);
        const bool gvaMode = params.HV != params.HK;

        gmQ.SetGlobalBuffer((__gm__ ElementA *)params.ptrQ);
        gmK.SetGlobalBuffer((__gm__ ElementA *)params.ptrK);
        gmV.SetGlobalBuffer((__gm__ ElementA *)params.ptrV);
        gmH.SetGlobalBuffer((__gm__ ElementA *)params.ptrH);
        gmDo.SetGlobalBuffer((__gm__ ElementA *)params.ptrDo);
        gmDh.SetGlobalBuffer((__gm__ ElementA *)params.ptrDh);
        gmDv.SetGlobalBuffer((__gm__ ElementA *)params.ptrDv);
        gmDq.SetGlobalBuffer((__gm__ ElementC *)params.ptrDq);
        gmDk.SetGlobalBuffer((__gm__ ElementC *)params.ptrDk);
        gmDw.SetGlobalBuffer((__gm__ ElementC *)params.ptrDw);
        gmMm3.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm3Offset));
        gmMm4.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm4Offset));
        gmMm6.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm6Offset));
        gmMm5.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm5Offset));
        gmMm7.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsMm7Offset));
        gmDs.SetGlobalBuffer((__gm__ ElementC *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsDsTempOffset));
        gmDsTemp.SetGlobalBuffer((__gm__ ElementA *)((__gm__ uint8_t *)params.ptrWorkspace + params.wsDsTempOffset));
        if (params.isVarLen) {
            cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)params.ptrCuSeqLens);
            chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)params.ptrChunkIndices);
        }

        // Layout 创建
        auto layoutBTxK = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.K);
        auto layoutKxBT = LayoutColMajor::MakeLayout<ElementA>(params.K, params.BT);
        auto layoutBTxV = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.V);
        auto layoutVxBT = LayoutColMajor::MakeLayout<ElementA>(params.V, params.BT);
        auto layoutBTxBT = LayoutRowMajor::MakeLayout<ElementA>(params.BT, params.BT);
        auto layoutBTxBT_T = LayoutColMajor::MakeLayout<ElementA>(params.BT, params.BT);
        auto layoutVxK = LayoutColMajor::MakeLayout<ElementA>(params.V, params.K);

        uint32_t bos = 0;
        uint32_t eos = 0;

        // raw 信用流水: cube 每 task 先 WaitCredit (节流到领先 vector <=N 个 task), 算完 SetCubeReady。
        // vector 预置 N=min(groupSize, M) 个信用 (见 vector.h)。N<=M 保证 C/D 读 ds_temp 安全。
        // cube WaitCredit 总次数 == L (==4M); 与 vector 端预置 N + SetCredit L 对应, 末尾余 N 信用 (无害)。
        // 现按 chunk-group-major 组织: 外层按 G 个 chunk 一组, 组内 A->B->C->D 连着做 (组的输入在 L2 内复用)。

        // ========== A_cube: dw = dv @ h^T, 然后 mm5 = q @ k^T ==========
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, params.B, params.HV, params.T, params.BT,
                            loopIdx, bos, eos, params.isVarLen);
            uint32_t actual_chunk_len = eos - bos;
            uint32_t bIdx = loopIdx / params.numChunks;
            uint32_t chunkIdx = loopIdx % params.numChunks;
            uint64_t bos_hk =
                        bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(params.HV - params.HK) * params.T;
            
            GemmCoord actualBlockShape1{actual_chunk_len, K, V};                    // [BT, V] @ [V, K]
            GemmCoord actualBlockShape2{actual_chunk_len, actual_chunk_len, K};     // [BT, K] @ [K, BT]
            GemmCoord actualBlockShape3{actual_chunk_len, actual_chunk_len, V};     // [BT, V] @ [V, BT]
            GemmCoord actualBlockShape4{actual_chunk_len, K, V};                    // [BT, V] @ [V, K]
            GemmCoord actualBlockShape6{actual_chunk_len, K, actual_chunk_len};     // [BT, BT] @ [BT, K]
            GemmCoord actualBlockShape5{actual_chunk_len, K, V};                    // [BT, V] @ [V, K]
            GemmCoord actualBlockShape7{actual_chunk_len, K, actual_chunk_len};     // [BT, BT] @ [BT, K]
            
            // --- Part1: dw = dv @ h^T ---
            for (uint32_t h = 0; h < params.HV; h++) {
                uint64_t dvOffset = (h * params.T + bos) * params.V;
                uint64_t hOffset = ((bIdx * params.HV + h) * params.numChunks + chunkIdx) * params.K * params.V;
                uint64_t dwOffset = (h * params.T + bos) * params.K;

                auto tensorDv = tla::MakeTensor(gmDv[dvOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[hOffset], MakeLayoutFromTag(layoutVxK), Arch::PositionGM{}); // h^T
                auto tensorDw = tla::MakeTensor(gmDw[dwOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                auto tensorBlockDv = GetTile(tensorDv, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape1.m(), actualBlockShape1.k()));
                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape1.k(), actualBlockShape1.n()));
                auto tensorBlockDw = GetTile(tensorDw, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape1.m(), actualBlockShape1.n()));
                BlockMmadPart1 blockMmadPart1(resource);
                blockMmadPart1(tensorBlockDv, tensorBlockH, tensorBlockDw, actualBlockShape1);
            }
            SetCubeReady();
            // --- Part2: mm3 = q @ k^T -> wsMm3 (B_vector 在后续 stage 消费) ---
            for (uint32_t h = 0; h < params.HV; h++) {
                // GVA: q/k 为 HK 头; hv_idx=h -> hk_idx=h/n_ratio, 并把 HV-based bos 修正为 HK-based
                uint32_t hk_idx = h / params.n_ratio;
                uint64_t qkOffset = (hk_idx * params.T + bos_hk) * params.K;
                uint64_t mm3Offset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);

                auto tensorQ = tla::MakeTensor(gmQ[qkOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK[qkOffset], MakeLayoutFromTag(layoutKxBT), Arch::PositionGM{}); // k^T
                auto tensorMm3 = tla::MakeTensor(gmMm3[mm3Offset], MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});

                auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape2.m(), actualBlockShape2.k()));
                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape2.k(), actualBlockShape2.n()));
                auto tensorBlockMm3 = GetTile(tensorMm3, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape2.m(), actualBlockShape2.n()));

                BlockMmadPart2 blockMmadPart2(resource);
                blockMmadPart2(tensorBlockQ, tensorBlockK, tensorBlockMm3, actualBlockShape2);
            }
            // --- Part3: ds = do @ v^T -> wsDsTemp ---
            for (uint32_t h = 0; h < params.HV; h++) {
                uint64_t dvOffset = (h * params.T + bos) * params.V;
                uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);

                auto tensorDo = tla::MakeTensor(gmDo[dvOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                auto tensorV = tla::MakeTensor(gmV[dvOffset], MakeLayoutFromTag(layoutVxBT), Arch::PositionGM{}); // v^T
                auto tensorDs = tla::MakeTensor(gmDs[dsOffset], MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});

                auto tensorBlockDo = GetTile(tensorDo, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape3.m(), actualBlockShape3.k()));
                auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape3.k(), actualBlockShape3.n()));
                auto tensorBlockDs = GetTile(tensorDs, tla::MakeCoord(0, 0),
                                            tla::MakeShape(actualBlockShape3.m(), actualBlockShape3.n()));
                BlockMmadPart3 blockMmadPart3(resource);
                blockMmadPart3(tensorBlockDo, tensorBlockV, tensorBlockDs, actualBlockShape3);
            }
            SetCubeReady();
            for (uint32_t h = 0; h < params.HV; h++) {
                // --- Part4: dq_inner = do @ h^T ---
                {
                    uint64_t doOffset = (h * params.T + bos) * params.V;
                    uint64_t hOffset = ((bIdx * params.HV + h) * params.numChunks + chunkIdx) * params.K * params.V;
                    uint64_t mm4Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorDo = tla::MakeTensor(gmDo[doOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[hOffset], MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                    auto tensorMm4 = tla::MakeTensor(gmMm4[mm4Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDo = GetTile(tensorDo, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape4.m(), actualBlockShape4.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape4.k(), actualBlockShape4.n()));
                    auto tensorBlockMm4 = GetTile(tensorMm4, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape4.m(), actualBlockShape4.n()));

                    BlockMmadPart4 blockMmadPart4(resource);
                    blockMmadPart4(tensorBlockDo, tensorBlockH, tensorBlockMm4, actualBlockShape4);
                }
            }
            WaitCredit();
            for (uint32_t h = 0; h < params.HV; h++) {
                // --- Part6: mm6 = ds_temp @ k -> wsMm6 ---
                {
                    // GVA: k 为 HK 头
                    uint32_t hk_idx = h / params.n_ratio;
                    uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);
                    uint64_t kOffset = (hk_idx * params.T + bos_hk) * params.K;
                    uint64_t mm6Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorDsTemp = tla::MakeTensor(gmDsTemp[dsOffset], MakeLayoutFromTag(layoutBTxBT), Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[kOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                    auto tensorMm6 = tla::MakeTensor(gmMm6[mm6Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDsTemp = GetTile(tensorDsTemp, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape6.m(), actualBlockShape6.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape6.k(), actualBlockShape6.n()));
                    auto tensorBlockMm6 = GetTile(tensorMm6, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape6.m(), actualBlockShape6.n()));

                    BlockMmadPart6 blockMmadPart6(resource);
                    blockMmadPart6(tensorBlockDsTemp, tensorBlockK, tensorBlockMm6, actualBlockShape6);
                }
            }
            SetCubeReady();
            for (uint32_t h = 0; h < params.HV; h++) {
                // --- Part5: dk_inner = v @ dh ---
                {
                    uint64_t vOffset = (h * params.T + bos) * params.V;
                    uint64_t dhOffset = ((bIdx * params.HV + h) * params.numChunks + chunkIdx) * params.K * params.V;
                    uint64_t mm5Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorV = tla::MakeTensor(gmV[vOffset], MakeLayoutFromTag(layoutBTxV), Arch::PositionGM{});
                    auto tensorDh = tla::MakeTensor(gmDh[dhOffset], MakeLayoutFromTag(layoutVxK), Arch::PositionGM{});
                    auto tensorMm5 = tla::MakeTensor(gmMm5[mm5Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                    
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape5.m(), actualBlockShape5.k()));
                    auto tensorBlockDh = GetTile(tensorDh, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape5.k(), actualBlockShape5.n()));
                    auto tensorBlockMm5 = GetTile(tensorMm5, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape5.m(), actualBlockShape5.n()));

                    BlockMmadPart5 blockMmadPart5(resource);
                    blockMmadPart5(tensorBlockV, tensorBlockDh, tensorBlockMm5, actualBlockShape5);
                }
                // --- Part7: mm7 = ds_temp^T @ q -> wsMm7 (复用 wsMm5) ---
                {
                    // GVA: q 为 HK 头
                    uint32_t hk_idx = h / params.n_ratio;
                    uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, params.HV, params.BT);
                    uint64_t qOffset = (hk_idx * params.T + bos_hk) * params.K;
                    uint64_t mm7Offset = DqkwgBtxKElemOffset(coreIdx, h, params.HV, params.BT, params.K);

                    auto tensorDsTemp =tla::MakeTensor(gmDsTemp[dsOffset], MakeLayoutFromTag(layoutBTxBT_T), Arch::PositionGM{});
                    auto tensorQ = tla::MakeTensor(gmQ[qOffset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});
                    auto tensorMm7 = tla::MakeTensor(gmMm7[mm7Offset], MakeLayoutFromTag(layoutBTxK), Arch::PositionGM{});

                    auto tensorBlockDsTemp = GetTile(tensorDsTemp, tla::MakeCoord(0, 0),
                                                        tla::MakeShape(actualBlockShape7.m(), actualBlockShape7.k()));
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0),
                                                tla::MakeShape(actualBlockShape7.k(), actualBlockShape7.n()));
                    auto tensorBlockMm7 = GetTile(tensorMm7, tla::MakeCoord(0, 0),
                                                    tla::MakeShape(actualBlockShape7.m(), actualBlockShape7.n()));

                    BlockMmadPart7 blockMmadPart7(resource);
                    blockMmadPart7(tensorBlockDsTemp, tensorBlockQ, tensorBlockMm7, actualBlockShape7);
                }
            }
            SetCubeReady();
        } // while group (chunk-group-major)
    }
};

} // namespace Catlass::Gemm::Kernel

/**
 * Cube Process 类 (AIC 端入口)
 */
template <typename DataType, typename GType>
class ChunkBwdDqkwgCubeProcess {
public:
    __aicore__ inline ChunkBwdDqkwgCubeProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR do_, GM_ADDR dh,
                                               GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices, GM_ADDR dq,
                                               GM_ADDR dk, GM_ADDR dw,GM_ADDR workspace)
        : ptrQ(q), ptrK(k), ptrV(v), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen),
          ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrWorkspace(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling);
    __aicore__ inline void Process();

private:
    // 输入输出指针
    GM_ADDR ptrQ;
    GM_ADDR ptrK;
    GM_ADDR ptrV;
    GM_ADDR ptrH;
    GM_ADDR ptrDo;
    GM_ADDR ptrDh;
    GM_ADDR ptrDv;
    GM_ADDR ptrCuSeqLen;
    GM_ADDR ptrChunkIndices;
    GM_ADDR ptrDq;
    GM_ADDR ptrDk;
    GM_ADDR ptrDw;
    GM_ADDR ptrWorkspace;

    // Tiling 参数
    uint64_t B;
    uint64_t HV;
    uint64_t HK;
    uint64_t n_ratio = 1;
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t BT;
    uint64_t numChunks;
    uint64_t aicCoreNum; // CV 深融合 blockDim (cube/vector 共用)
    bool isVarLen;

    // Workspace 偏移
    uint64_t wsMm3Offset;
    uint64_t wsMm4Offset;
    uint64_t wsMm6Offset;
    uint64_t wsMm5Offset;
    uint64_t wsMm7Offset;
    uint64_t wsDsTempOffset;
};

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgCubeProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling)
{
    B = tiling.B;
    HV = tiling.HV;
    HK = tiling.HK;
    n_ratio = (HK > 0) ? (HV / HK) : 1;
    T = tiling.T;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.BT;
    numChunks = tiling.numChunks;
    aicCoreNum = tiling.aicCoreNum;
    isVarLen = (tiling.isVarLen == 1);
    wsMm3Offset = tiling.wsMm3Offset;
    wsMm4Offset = tiling.wsMm4Offset;
    wsMm6Offset = tiling.wsMm6Offset;
    wsMm5Offset = tiling.wsMm5Offset;
    wsMm7Offset = tiling.wsMm7Offset;
    wsDsTempOffset = tiling.wsDsTempOffset;
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgCubeProcess<DataType, GType>::Process()
{
    using namespace Catlass;
    using namespace Catlass::Gemm;

    // Layout 类型定义
    using LayoutRowMajor = layout::RowMajor;
    using LayoutColMajor = layout::ColumnMajor;

    // 架构和策略定义
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using ArchTag = Arch::Ascend950;
#else
    using ArchTag = Arch::AtlasA2;
#endif
    using DispatchPolicy = Gemm::MmadPingpong<ArchTag, true>;
    using L1TileShape = tla::Shape<_128, _128, _256>;
    using L0TileShape = tla::Shape<_128, _128, _128>;

    // Part 1: dv @ h^T -> dw  [BT, V] @ [V, K] -> [BT, K]
    using TileCopyPart1 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart1 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart1>;

    // Part 2: q @ k^T -> mm5  [BT, K] @ [K, BT] -> [BT, BT]
    using TileCopyPart2 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart2 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart2>;

    // Part 3: do @ v^T -> ds  [BT, V] @ [V, BT] -> [BT, BT]
    using TileCopyPart3 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart3 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart3>;

    // Part 4: do @ h^T -> dq  [BT, V] @ [V, K] -> [BT, K]
    using TileCopyPart4 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart4 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart4>;

    // Part 5: v @ dh -> dk  [BT, V] @ [V, K] -> [BT, K]
    using TileCopyPart5 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutColMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart5 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart5>;

    // Part 6: ds @ k -> dq+=  [BT, BT] @ [BT, K] -> [BT, K]
    using TileCopyPart6 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType, LayoutRowMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart6 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart6>;

    // Part 7: ds^T @ q -> dk+=  [BT, BT] @ [BT, K] -> [BT, K]
    using TileCopyPart7 = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutColMajor, DataType, LayoutRowMajor,
                                                        DataType, LayoutRowMajor>;
    using BlockMmadPart7 = Kernel::TileGemmDirect<ArchTag, DataType, TileCopyPart7>;

    // Kernel 实例
    using MatmulKernel = Kernel::ChunkBwdDqkwgTla<BlockMmadPart1, BlockMmadPart2, BlockMmadPart3, BlockMmadPart4,
                                                  BlockMmadPart5, BlockMmadPart6, BlockMmadPart7>;

    // V=256 makes Part1/3/4/5 use a 256-wide reduction dimension. TileGemmDirect is a single
    // 128-wide tile, so keep the fast direct path for V<=128 and use the tiled CATLASS path only
    // for the V=256 corner case.
    using TileCopyPart1Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart1Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart1Tiled>;
    using TileCopyPart2Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart2Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart2Tiled>;
    using TileCopyPart3Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart3Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart3Tiled>;
    using TileCopyPart4Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart4Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart4Tiled>;
    using TileCopyPart5Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutColMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart5Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart5Tiled>;
    using TileCopyPart6Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutRowMajor, DataType,
                                                             LayoutRowMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart6Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart6Tiled>;
    using TileCopyPart7Tiled = Gemm::Tile::PackedTileCopyTla<ArchTag, DataType, LayoutColMajor, DataType,
                                                             LayoutRowMajor, DataType, LayoutRowMajor>;
    using BlockMmadPart7Tiled = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, DataType, DataType,
                                                          DataType, void, TileCopyPart7Tiled>;
    using MatmulKernelTiled =
        Kernel::ChunkBwdDqkwgTla<BlockMmadPart1Tiled, BlockMmadPart2Tiled, BlockMmadPart3Tiled, BlockMmadPart4Tiled,
                                 BlockMmadPart5Tiled, BlockMmadPart6Tiled, BlockMmadPart7Tiled>;

    if (V > 128) {
        MatmulKernelTiled kernel;
        typename MatmulKernelTiled::Params params(ptrQ, ptrK, ptrV, ptrH, ptrDo, ptrDh, ptrDv, ptrCuSeqLen,
                                                  ptrChunkIndices, ptrDq, ptrDk, ptrDw, ptrWorkspace, B, HV, HK, T, K, V, BT,
                                                  numChunks, n_ratio, isVarLen, wsMm3Offset, wsMm4Offset, wsMm6Offset,
                                                  wsMm5Offset, wsMm7Offset, wsDsTempOffset);
        params.aicCoreNum = aicCoreNum;
        kernel(params);
    } else {
        MatmulKernel kernel;
        typename MatmulKernel::Params params(ptrQ, ptrK, ptrV, ptrH, ptrDo, ptrDh, ptrDv, ptrCuSeqLen, ptrChunkIndices,
                                             ptrDq, ptrDk, ptrDw, ptrWorkspace, B, HV, HK, T, K, V, BT, numChunks, n_ratio,
                                             isVarLen, wsMm3Offset, wsMm4Offset, wsMm6Offset, wsMm5Offset, 
                                             wsMm7Offset, wsDsTempOffset);
        params.aicCoreNum = aicCoreNum;
        kernel(params);
    }
}

#endif // CHUNK_BWD_DQKWG_CUBE_H
