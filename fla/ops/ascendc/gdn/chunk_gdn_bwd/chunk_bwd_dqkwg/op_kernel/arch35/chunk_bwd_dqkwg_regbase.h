/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_regbase.h
 * \brief A5 (dav-3510) RegBase __simd_vf__ 热链骨架 for chunk_bwd_dqkwg vector 端。
 */

#ifndef CHUNK_BWD_DQKWG_ARCH35_REGBASE_H
#define CHUNK_BWD_DQKWG_ARCH35_REGBASE_H

#include "../chunk_bwd_dqkwg_common.h"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

// 一个 fp32 寄存器可容纳的元素数 (dav-3510 = 64)
constexpr uint16_t V_LENGTH_FP32 = VECTOR_REG_WIDTH / sizeof(float);
// 一个 half 寄存器可容纳的元素数 (dav-3510 = 128; bfloat16_t 同为 2 字节, 共用此常量)
constexpr uint16_t V_LENGTH_HALF = VECTOR_REG_WIDTH / sizeof(half);
// QKV 16-bit 类型 (half / bfloat16_t) 由 caller 按 DataType 作为 HalfT 模板实参传入;
// CastHalf2Float<HalfT> / CastFloat2Half<HalfT> 已模板化, ctHalf2Fp32*/ctFp322Half* trait
// 配置 (ZERO/SAT/ZEROING/CAST_NONE 等) 对 half 与 bfloat16_t 通用 (对照 kda kKdaRegbaseBf16ToFp32)。

// ============================================================================
// P0-1: Mul1Half
//   out[i][j] = scale * mask[i][j] * exp(min(0, g[BT_SUB_START+i] - g[j]))
//
// 调用方需先完成 g 的 GType->fp32 cast (保留 MemBase, 1 条指令), 将 fp32 g 指针
// 传入; Muls(-1)/Brcb/strided-Add/Mins/Exp/Mul(mask)/Muls(scale) 全部下沉到本 VF。
//
// mask 布局 (与 vector.h gBuf 64x64 下三角一致):
//   BT==64 : out[i][0..63], mask row = BT_SUB_START + i  (causal col j <= BT_SUB_START+i)
//   BT==128: 分前后两个 64 元素半段
//            BT_SUB_START==0  : cols[0..63]  *= maskA[i]; cols[64..127] = 0
//            BT_SUB_START==64 : cols[0..63]  保留 (mask=1); cols[64..127] *= maskA[i]
//
// BT_SUB_START 作为模板参数 (编译期常量), BT==128 的分流用 if constexpr 实现,
// 不破坏 Hardware Loop / #pragma unroll 约束 (循环体内无 runtime 分支)。
//
// 注意: gFp32 buffer 需 >= BT_SIZE + V_LENGTH_FP32 元素以避免末行 broadcast-load
// 越界 (calcBuf2 分配 >=128 对 BT==64 安全; BT==128 需 caller 确认 padding)。
// ============================================================================
template <uint16_t BT_SIZE, uint16_t BT_SUB_START>
static __simd_vf__ inline void Mul1Half(
    __ubuf__ float *outFp32,       // [realBt, BT_SIZE] row-major 输出 (fp32)
    __ubuf__ float *gFp32,         // [BT_SIZE] gate 已升精度 (caller cast 后)
    __ubuf__ float *maskAddr,      // [64, 64] 下三角 mask, 行 r = maskAddr + r*64
    uint16_t realBt,               // 有效输出行数
    float scale)
{
    RegTensor<float> regGLeft, regSum, regExp, regMask, regOut, scaleReg;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    Duplicate(scaleReg, scale);

    // 一次性载入 g[0..BT-1] 并取负 (-g 在所有行共享)
    if constexpr (BT_SIZE == 64) {
        RegTensor<float> regG, regNegG;
        LoadIn<float, false>(regG, gFp32);
        Muls(regNegG, regG, -1.0f, maskFull);

        for (uint16_t i = 0; i < realBt; ++i) {
            // gLeft = g[BT_SUB_START + i], 广播到全 lane (DIST_BRC_B32)
            LoadIn<float, true>(regGLeft, gFp32 + BT_SUB_START + i);
            Add(regSum, regNegG, regGLeft, maskFull);     // gLeft - g
            Mins(regSum, regSum, 0.0f, maskFull);
            Exp(regExp, regSum, maskFull);
            // mask row = BT_SUB_START + i
            LoadIn<float, false>(regMask, maskAddr + (BT_SUB_START + i) * 64);
            Mul(regOut, regExp, regMask, maskFull);
            Mul(regOut, regOut, scaleReg, maskFull);
            StoreAlign(outFp32 + i * BT_SIZE, regOut, maskFull);
        }
    } else {
        // BT_SIZE == 128: g 分两个 64 元素半段
        RegTensor<float> regG0, regG1, regNegG0, regNegG1;
        RegTensor<float> regSum1, regExp1, regOut1;
        LoadIn<float, false>(regG0, gFp32);
        LoadIn<float, false>(regG1, gFp32 + 64);
        Muls(regNegG0, regG0, -1.0f, maskFull);
        Muls(regNegG1, regG1, -1.0f, maskFull);

        for (uint16_t i = 0; i < realBt; ++i) {
            LoadIn<float, true>(regGLeft, gFp32 + BT_SUB_START + i);
            // 前半 cols[0..63]
            Add(regSum, regNegG0, regGLeft, maskFull);
            Mins(regSum, regSum, 0.0f, maskFull);
            Exp(regExp, regSum, maskFull);
            // 后半 cols[64..127]
            Add(regSum1, regNegG1, regGLeft, maskFull);
            Mins(regSum1, regSum1, 0.0f, maskFull);
            Exp(regExp1, regSum1, maskFull);
            // mask (row = i): BT_SUB_START==0 -> 前半 mask / 后半置 0;
            //                 BT_SUB_START==64 -> 前半保留 / 后半 mask
            LoadIn<float, false>(regMask, maskAddr + i * 64);
            if constexpr (BT_SUB_START == 0) {
                Mul(regOut, regExp, regMask, maskFull);
                Mul(regOut, regOut, scaleReg, maskFull);
                Duplicate(regOut1, 0.0f);       // 后半置 0 (scale*0=0)
            } else {
                Mul(regOut, regExp, scaleReg, maskFull);  // 前半保留, 仅 scale
                Mul(regOut1, regExp1, regMask, maskFull);
                Mul(regOut1, regOut1, scaleReg, maskFull);
            }
            StoreAlign(outFp32 + i * BT_SIZE, regOut, maskFull);
            StoreAlign(outFp32 + i * BT_SIZE + 64, regOut1, maskFull);
        }
    }
}

// ============================================================================
// P1-4: DqState  (dq_state = dq_inner * exp(g) * scale, factor 按行广播到 K)
//   MemBase: ProcessCVector L979-988 (Exp -> Muls(scale) -> Brcb -> Mul strided x2)
//   与 DkStateMUL2 同构 (factor = scale*exp(g[i]) 而非 exp(gLast-g[i])), 阶段 2 一致。
// ============================================================================
static __simd_vf__ inline void DqState(
    __ubuf__ float *dqFp32,        // [realBt, kDim] in/out (fp32)
    __ubuf__ float *gFp32,         // [realBt] gate (fp32, 原值只读)
    __ubuf__ float *factorScratch, // [>= realBt + V_LENGTH_FP32] caller 分配
    uint16_t realBt,
    uint16_t kDim,
    float scale)
{
    RegTensor<float> regG, regFactor, regScale, regDq, regOut;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
    Duplicate(regScale, scale);

    // 阶段 1: factor[i] = scale * exp(g[i]) -> factorScratch
    for (uint16_t blk = 0; blk < realBt; blk += V_LENGTH_FP32) {
        uint32_t remain = realBt - blk;
        MaskReg maskG = UpdateMask<float>(remain);
        LoadIn<float, false>(regG, gFp32 + blk);
        Exp(regFactor, regG, maskG);
        Mul(regFactor, regFactor, regScale, maskG);
        StoreAlign(factorScratch + blk, regFactor, maskG);
    }
    // 阶段 2: dq[i][:] *= factor[i]  (与 DkStateMUL2 阶段 2 一致)
    for (uint16_t i = 0; i < realBt; ++i) {
        LoadIn<float, true>(regFactor, factorScratch + i);   // 广播 factor[i]
        for (uint16_t k = 0; k < kDim; k += V_LENGTH_FP32) {
            uint32_t kRemain = kDim - k;
            MaskReg maskK = UpdateMask<float>(kRemain);
            LoadIn<float, false>(regDq, dqFp32 + i * kDim + k);
            Mul(regOut, regDq, regFactor, maskK);
            StoreAlign(dqFp32 + i * kDim + k, regOut, maskK);
        }
    }
}

// ============================================================================
// DwNegate  (dw = -dw, half in-place)
//   (Cast half->fp32 -> Muls(-1) -> Cast fp32->half, 3 PipeBarrier)
//   极短极热 (每 head 每 chunk 一次)。half 寄存器 128 元素, CastHalf2Float 拆两个 64 float reg。
// ============================================================================
template <typename HalfT>
static __simd_vf__ inline void DwNegate(__ubuf__ HalfT *dwOutBuf, __ubuf__ HalfT *dwInBuf, uint32_t elemCount)
{
    // elemCount must be multiple of V_LENGTH_HALF
    uint16_t colLoopTimes = static_cast<uint16_t>(elemCount / V_LENGTH_HALF);
    RegTensor<HalfT> regH;
    RegTensor<float> regF0, regF1;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskHalfAll = CreateMask<HalfT, MaskPattern::ALL>();
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(regH, dwInBuf, V_LENGTH_HALF);
        CastHalf2Float<HalfT>(regF0, regF1, regH, maskHalfAll);
        Muls(regF0, regF0, -1.0f, maskAll);
        Muls(regF1, regF1, -1.0f, maskAll);
        Cast<HalfT, float, ctFp322HalfOne>(regH, regF1, maskAll);
        Cast<HalfT, float, ctFp322HalfZero>(regH, regF0, maskAll);
        StoreAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(dwOutBuf, regH, V_LENGTH_HALF, maskHalfAll);
    }
}

// ============================================================================
// P3-6: DgLastMulAccum  (sum = h * dh 或 sum += h * dh, 预归约段)
//   MemBase: ProcessAVector dg_last L510-534 (Cast h/dh -> Mul -> Add 累加, 跨 K 行 tile)
//   模板参数 needAdd:
//     - true  : 累加模式, sum = sum + h*dh (读已有 sum 并相加, 用于第 2+ 个 K-row tile)
//     - false : 覆盖模式, sum = h*dh       (首轮 tile 初始化, 等价 MemBase row==0 的 Mul 初始化)
//   caller 跨 K-row tile 多次调用: 首 tile 用 needAdd=false, 后续 tile 用 needAdd=true。
//   归约 (Add 折半 + WholeReduceSum) 仍走 MemBase。
// ============================================================================
template <typename HalfT, bool needAdd>
static __simd_vf__ inline void DgLastMulAccum(
    __ubuf__ float *sumFp32,       // [elemCount] in/out running sum (fp32); needAdd=false 时被覆盖, needAdd=true 时读加写
    __ubuf__ HalfT *hHalf,         // [elemCount] h tile (half / bfloat16_t)
    __ubuf__ HalfT *dhHalf,        // [elemCount] dh tile (half / bfloat16_t)
    uint32_t elemCount)            // elemCount must be multiple of V_LENGTH_HALF
{
    uint16_t colLoopTimes = static_cast<uint16_t>(elemCount / V_LENGTH_HALF);
    RegTensor<HalfT> regHH, regDhH;
    RegTensor<float> regHF0, regHF1, regDhF0, regDhF1, regProd0, regProd1;
    RegTensor<float> regS0, regS1;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t j = 0; j < colLoopTimes; j++) {
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(regHH, hHalf, V_LENGTH_HALF);
        LoadAlign<HalfT, PostLiteral::POST_MODE_UPDATE>(regDhH, dhHalf, V_LENGTH_HALF);
        CastHalf2Float<HalfT>(regHF0, regHF1, regHH, maskAll);
        CastHalf2Float<HalfT>(regDhF0, regDhF1, regDhH, maskAll);
        Mul(regProd0, regHF0, regDhF0, maskAll);
        Mul(regProd1, regHF1, regDhF1, maskAll);
        if constexpr (needAdd) {
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regS0, sumFp32, V_LENGTH_FP32);
            LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regS1, sumFp32, V_LENGTH_FP32);
            Add(regProd0, regS0, regProd0, maskAll);
            Add(regProd1, regS1, regProd1, maskAll);
        }
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(sumFp32, regProd0, V_LENGTH_FP32, maskAll);
        StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(sumFp32, regProd1, V_LENGTH_FP32, maskAll);
    }
}

// ============================================================================
// P3-8: BDsTempMulCast  (ds_temp = ds*mul1; ds2 = ds_temp*mm5; out ds_temp(half) + ds2(fp32))
//   MemBase: ProcessB L783-817 (Cast ds/mul1/mm5 -> Mul ds*mul1 -> Mul ds_temp*mm5 -> Cast ds_temp)
//   mul1 由 caller 预 cast 到 fp32 (大 case 已 fp32, 小 case caller cast)。归约仍走 MemBase。
// ============================================================================
template <typename HalfT>
static __simd_vf__ inline void BDsTempMulCast(
    __ubuf__ HalfT *dsTempHalfOut, // [elemCount] ds_temp output (half / bfloat16_t)
    __ubuf__ float *ds2Fp32Out,    // [elemCount] ds2 = ds_temp*mm5 output (fp32, 供 B 端归约)
    __ubuf__ HalfT *dsHalf,        // [elemCount] ds (half / bfloat16_t)
    __ubuf__ float *mul1Fp32,      // [elemCount] mul1 (fp32, caller 预 cast)
    __ubuf__ HalfT *mm5Half,       // [elemCount] mm5 (half / bfloat16_t)
    uint16_t elemCount)
{
    RegTensor<HalfT> regDsH, regMm5H, regTempH;
    RegTensor<float> regDsF0, regDsF1, regM10, regM11, regTemp0, regTemp1;
    RegTensor<float> regMm5F0, regMm5F1, regDs20, regDs21;
    uint32_t halfRemaining = elemCount;
    uint32_t floatRemaining = elemCount;
    for (uint16_t blk = 0; blk < elemCount; blk += V_LENGTH_HALF) {
        MaskReg maskH = UpdateMask<HalfT>(halfRemaining);
        MaskReg maskF0 = UpdateMask<float>(floatRemaining);
        MaskReg maskF1 = UpdateMask<float>(floatRemaining);
        // ds_temp = ds * mul1
        LoadIn<HalfT, false>(regDsH, dsHalf + blk);
        CastHalf2Float<HalfT>(regDsF0, regDsF1, regDsH, maskH);
        LoadIn<float, false>(regM10, mul1Fp32 + blk);
        LoadIn<float, false>(regM11, mul1Fp32 + blk + V_LENGTH_FP32);
        Mul(regTemp0, regDsF0, regM10, maskF0);
        Mul(regTemp1, regDsF1, regM11, maskF1);
        // ds2 = ds_temp * mm5
        LoadIn<HalfT, false>(regMm5H, mm5Half + blk);
        CastHalf2Float<HalfT>(regMm5F0, regMm5F1, regMm5H, maskH);
        Mul(regDs20, regTemp0, regMm5F0, maskF0);
        Mul(regDs21, regTemp1, regMm5F1, maskF1);
        // 输出: ds_temp -> half, ds2 -> fp32
        Cast<HalfT, float, ctFp322HalfOne>(regTempH, regTemp1, maskF1);
        Cast<HalfT, float, ctFp322HalfZero>(regTempH, regTemp0, maskF0);
        StoreAlign(dsTempHalfOut + blk, regTempH, maskH);
        StoreAlign(ds2Fp32Out + blk, regDs20, maskF0);
        StoreAlign(ds2Fp32Out + blk + V_LENGTH_FP32, regDs21, maskF1);
    }
}

#endif // CHUNK_BWD_DQKWG_ARCH35_REGBASE_H
