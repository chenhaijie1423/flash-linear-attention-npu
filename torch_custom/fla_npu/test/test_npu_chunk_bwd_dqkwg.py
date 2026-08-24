# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
# from ct import single
import torch
import torch.nn.functional as F
from typing import Tuple
# import custom_ops
from fla_npu.ops import ascendc as ascendc_ops

import os

# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
# 如果不存在就创建
os.makedirs(data_path, exist_ok=True)
data_path_in = os.path.join(data_path, "in")
os.makedirs(data_path_in, exist_ok=True)
data_path_out = os.path.join(data_path, "out")
os.makedirs(data_path_out, exist_ok=True)

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

def pause():
    print("pause")
    input()

from typing import Optional
import pickle
import math
import sys

import time

# ---------------------------------------------------------------------------
# 耗时统计工具（仅诊断用，不影响计算结果）
# 通过 chunk_bwd_dqkwg_cpu(..., verbose_timing=True) 或环境变量
# FLA_NPU_PROFILE=1 启用，会按 stage 汇总耗时并打印明细。
# ---------------------------------------------------------------------------
class _StageTimer:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self._starts = {}

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str):
        t0 = self._starts.pop(stage, None)
        if t0 is None:
            return
        dt = time.perf_counter() - t0
        self.totals[stage] = self.totals.get(stage, 0.0) + dt
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def summary(self, total_time: float, chunks: int) -> str:
        if not self.totals:
            return ""
        rows = sorted(self.totals.items(), key=lambda x: -x[1])
        sum_stages = sum(self.totals.values())
        overhead = max(total_time - sum_stages, 0.0)
        lines = []
        lines.append(f"[chunk_bwd_dqkwg_cpu] chunks={chunks} total={total_time*1000:.3f}ms "
                     f"sum_stages={sum_stages*1000:.3f}ms overhead={overhead*1000:.3f}ms "
                     f"({overhead/total_time*100:.1f}%)")
        lines.append(f"[chunk_bwd_dqkwg_cpu]   {'stage':<24}{'total_ms':>12}{'calls':>10}{'avg_us':>12}{'pct':>8}")
        for stage, tot in rows:
            cnt = self.counts[stage]
            avg_us = (tot / cnt) * 1e6 if cnt else 0.0
            pct = (tot / total_time * 100) if total_time > 0 else 0.0
            lines.append(f"[chunk_bwd_dqkwg_cpu]   {stage:<24}{tot*1000:>12.3f}{cnt:>10d}{avg_us:>12.2f}{pct:>7.1f}%")
        return "\n".join(lines)


class _NullTimer:
    def start(self, stage: str): pass
    def stop(self, stage: str): pass
    def summary(self, total_time: float, chunks: int) -> str: return ""

def chunk_bwd_dqkwg_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    benchmark = False,
    verbose_timing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU Equivalent of chunk_bwd_kernel_dqkwg.

    优化说明：将原按 HV 头展开的 Python 循环改为对 HV 维做批量 (bmm) 计算。
    CPU 上 bmm 对每个 batch 调用与单次 mm 相同的 gemm，因此每个头的矩阵乘
    与原循环逐位等价；所有 dtype 往返 (.to(datatype).to(calc_type)) 和运算顺序
    均原样保留，维持与 kernel 一致的精度模拟。

    verbose_timing=True 或环境变量 FLA_NPU_PROFILE=1 时，按 stage 打印耗时明细，
    便于定位瓶颈（仅诊断用，关闭时零开销）。
    """
    if os.environ.get("FLA_NPU_PROFILE", "") == "1":
        verbose_timing = True
    timer = _StageTimer() if verbose_timing else _NullTimer()
    t_total_start = time.perf_counter()
    chunk_count = 0

    calc_type = torch.float64 if benchmark else torch.float32
    B, T, HK, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    if HK <= 0 or HV <= 0 or HV % HK != 0:
        raise ValueError(f"GVA requires HV divisible by HK, got HV={HV}, HK={HK}")
    n_ratio = HV // HK  # HV = n_ratio * HK
    datatype = q.dtype
    gtype = g.dtype
    if benchmark:
        datatype = torch.float64
        gtype = torch.float64

    # Keep per-value-head contributions first, then reduce them into key heads.
    timer.start("alloc")
    dq_hv = torch.zeros((B, T, HV, K), dtype=datatype)
    dk_hv = torch.zeros((B, T, HV, K), dtype=datatype)
    dg = torch.zeros_like(g)
    dw = torch.zeros((B, T, HV, K), dtype=datatype)
    timer.stop("alloc")

    # 缓存因果 mask（按 actual_chunk_len），避免每个 chunk 重复构造
    # 同时缓存 calc_type 版本（用于融合掩码乘法），避免 where(scalar) 提升类型
    mask_cache = {}

    def get_causal_mask(L: int, device):
        m = mask_cache.get(L)
        if m is None or m.device != device:
            idx = torch.arange(L, device=device)
            m = idx[:, None] >= idx[None, :]
            mask_cache[L] = m
        return m

    def get_causal_mask_f(L: int, device):
        """返回 [1, L, L] 的 calc_type 浮点 mask，供 ds *= mask_f 原地掩码用。"""
        key = (L, str(calc_type))
        mf = mask_cache.get(key)
        if mf is None or mf.device != device:
            mf = get_causal_mask(L, device).to(calc_type)[None]
            mask_cache[key] = mf
        return mf

    # 模拟 kernel 中间结果的 dtype 往返；datatype == calc_type 时短路（避免无谓分配）
    def cast_round(t):
        if datatype == calc_type:
            return t
        return t.to(datatype).to(calc_type)

    # 将 [HK, L, K] 按 n_ratio 复制到 [HV, L, K]；n_ratio==1 时短路
    def expand_heads(t):
        if n_ratio == 1:
            return t
        return t.repeat_interleave(n_ratio, dim=0)

    def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
        nonlocal chunk_count
        seq_len = t_end - t_start
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for i_t in range(num_chunks):
            chunk_start_token_idx = t_start + i_t * chunk_size
            chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
            L = chunk_end_token_idx - chunk_start_token_idx
            if L <= 0:
                continue
            s = chunk_start_token_idx
            e = chunk_end_token_idx
            chunk_count += 1

            # ---- 取当前 chunk 全部头的数据，统一以 head 作为 batch 维 ----
            # q/k: [L, HK, K] -> [HK, L, K] -> 复制 n_ratio 份 -> [HV, L, K]
            #   head h_idx 对应 hk_idx = h_idx // n_ratio（与原循环一致）
            timer.start("data_prep")
            q_h = expand_heads(q[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type))
            k_h = expand_heads(k[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type))
            # v/do: [L, HV, V] -> [HV, L, V]
            v_h = v[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type)
            do_h = do[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type)
            # h/dh: [HV, K, V]（保留原始 datatype，供 dg_last_accum 使用）
            h_prev = h[b_idx, i_t + chunk_start_idx, :, :, :]       # [HV, K, V]
            dh_curr = dh[b_idx, i_t + chunk_start_idx, :, :, :]     # [HV, K, V]
            h_prev_t = h_prev.transpose(-1, -2).to(calc_type)       # [HV, V, K]
            dh_curr_t = dh_curr.transpose(-1, -2).to(calc_type)     # [HV, V, K]
            timer.stop("data_prep")

            # -----------------------------------------------------------
            # 1. State Contributions (Inter-chunk)
            # -----------------------------------------------------------
            # b_dq += dot(b_do, b_h); b_dk += dot(b_v, b_dh)
            timer.start("bmm_state")
            dq_from_state = cast_round(torch.bmm(do_h, h_prev_t))   # [HV, L, K]
            dk_from_state = cast_round(torch.bmm(v_h, dh_curr_t))   # [HV, L, K]
            timer.stop("bmm_state")

            # b_dw += dot(b_dv, b_h) (kernel 存 -b_dw)
            timer.start("bmm_dw")
            dv_h = dv[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type)
            dw_c = cast_round(torch.bmm(dv_h, h_prev_t))       # [HV, L, K]
            dw[b_idx, s:e, :, :] = (-dw_c).permute(1, 0, 2)
            timer.stop("bmm_dw")

            timer.start("mask")
            mask_f = get_causal_mask_f(L, q.device)
            timer.stop("mask")

            # -----------------------------------------------------------
            # 2. Gating / Decay Logic Preparation
            # -----------------------------------------------------------
            timer.start("decay_scale")
            g_h = g[b_idx, s:e, :].permute(1, 0)                # [HV, L] (gtype)
            g_last = g[b_idx, min(s + chunk_size, t_end) - 1, :]  # [HV]

            exp_gc = torch.exp(g_h)                              # [HV, L]
            exp_neg_gc_glast = torch.exp(-g_h + g_last[:, None]) # [HV, L]

            dq_from_state = dq_from_state * (exp_gc[:, :, None] * scale)
            dk_from_state = dk_from_state * exp_neg_gc_glast[:, :, None]
            timer.stop("decay_scale")

            # -----------------------------------------------------------
            # 3. Intra-chunk Attention
            # -----------------------------------------------------------
            timer.start("bmm_intra_ds")
            ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
            timer.stop("bmm_intra_ds")
            timer.start("decay_apply")
            decay_mat = torch.exp(torch.min(g_h[:, :, None] - g_h[:, None, :], torch.tensor(0)))  # [HV, L, L]
            # 融合：避免 where(scalar) 类型提升与多次临时分配
            ds = ds * decay_mat
            ds = ds * mask_f                  # [1, L, L] 广播掩码
            ds = ds * scale
            timer.stop("decay_apply")

            timer.start("bmm_qk")
            qk_t = cast_round(torch.bmm(q_h, k_h.transpose(1, 2)))   # [HV, L, L]

            ds2 = ds * qk_t
            dg_c = ds2.sum(dim=-1)
            dg_c = dg_c - ds2.sum(dim=-2)
            if datatype == gtype:
                dg_c = dg_c.to(gtype)                               # 等价 .to(datatype).to(gtype)
            else:
                dg_c = dg_c.to(datatype).to(gtype)                  # [HV, L]
            timer.stop("bmm_qk")

            # b_dg += sum(b_dq * b_q) ; b_dg -= sum(b_k * b_dk)
            # 保留显式 product+sum（与原参考一致，避免 einsum 改变归约顺序）
            timer.start("dg_state")
            dg_c = cast_round(dg_c)
            dg_c += (dq_from_state * q_h).sum(dim=-1)             # [HV, L]
            dg_c = cast_round(dg_c)
            # k_h * dk_from_state 在 dg_c 和 dg_last_accum 中各用一次，提取公共子表达式
            k_dk_prod = k_h * dk_from_state                       # [HV, L, K]
            dg_c = dg_c - k_dk_prod.sum(dim=-1)                   # [HV, L]

            # b_dg_last += sum(h * dh) * exp(g_last) + sum(dk * k)
            # 注意 h_prev/dh_curr 保留原始 datatype（与原实现一致）
            dg_last_accum = (h_prev * dh_curr).sum(dim=(-1, -2)) * torch.exp(g_last)  # [HV]
            dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))             # [HV]
            timer.stop("dg_state")

            # b_ds2 = b_ds * (q @ k.T)
            timer.start("dg_intra_accum")
            # 仅块最后一个有效 token 累加 dg_last
            dg_c[:, L - 1] = dg_c[:, L - 1] + dg_last_accum
            dg[b_idx, s:e, :] = dg_c.to(gtype).permute(1, 0)                   # [L, HV]
            timer.stop("dg_intra_accum")

            # -----------------------------------------------------------
            # 4. Final Accumulation for dq, dk
            # -----------------------------------------------------------
            timer.start("bmm_dqdk_intra")
            dq_intra = cast_round(torch.bmm(ds, k_h))               # [HV, L, K]
            dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_h))# [HV, L, K]
            timer.stop("bmm_dqdk_intra")

            timer.start("accumulate")
            dq_total = dq_from_state + dq_intra
            dk_total = dk_from_state + dk_intra
            timer.stop("accumulate")

            timer.start("write_back")
            if datatype == calc_type:
                dq_hv[b_idx, s:e, :, :] = dq_total.permute(1, 0, 2)
                dk_hv[b_idx, s:e, :, :] = dk_total.permute(1, 0, 2)
            else:
                dq_hv[b_idx, s:e, :, :] = dq_total.to(datatype).permute(1, 0, 2)
                dk_hv[b_idx, s:e, :, :] = dk_total.to(datatype).permute(1, 0, 2)
            timer.stop("write_back")

    # ------------------------------------------------------------------
    # 批量 dense 路径：把同一 b 下所有满 chunk（长度 == chunk_size）合并成
    # 一次大 bmm，消除 Python chunk 循环与算子派发开销。
    # 仅当 cu_seqlens is None 且存在满 chunk 时启用；不足 chunk_size 的尾部
    # 仍走逐 chunk 的 process_sequence（保证 ragged/varlen 正确性）。
    # chunk 之间无数据依赖（h/dh 是只读输入，dq_hv 等只写不跨 chunk 读），
    # 故改变 chunk 处理顺序不改变结果；单个 chunk 内运算顺序保持不变。
    # ------------------------------------------------------------------
    def process_dense_batched(b_idx: int, n_full: int, t_offset: int, h_offset: int):
        nonlocal chunk_count
        N = n_full
        if N <= 0:
            return
        C = chunk_size
        H = HV
        s0 = t_offset
        chunk_count += N
        timer.start("batched_total")

        # ---- 数据准备：把 [N*C, ...] 重排为 [N*HV, C, ...] ----
        timer.start("batched_data_prep")
        # q/k: [N*C, HK, K] -> [N, C, HK, K] -> [N, HK, C, K] -> [N*HV, C, K]
        q_b = q[b_idx, s0:s0 + N * C, :, :].view(N, C, HK, K).permute(0, 2, 1, 3)
        if n_ratio > 1:
            q_b = q_b.repeat_interleave(n_ratio, dim=1)           # [N, HV, C, K]
        q_b = q_b.reshape(N * H, C, K).to(calc_type)
        k_b = k[b_idx, s0:s0 + N * C, :, :].view(N, C, HK, K).permute(0, 2, 1, 3)
        if n_ratio > 1:
            k_b = k_b.repeat_interleave(n_ratio, dim=1)
        k_b = k_b.reshape(N * H, C, K).to(calc_type)
        # v/do: [N*C, HV, V] -> [N, HV, C, V] -> [N*HV, C, V]
        v_b = v[b_idx, s0:s0 + N * C, :, :].view(N, C, H, V).permute(0, 2, 1, 3).reshape(N * H, C, V).to(calc_type)
        do_b = do[b_idx, s0:s0 + N * C, :, :].view(N, C, H, V).permute(0, 2, 1, 3).reshape(N * H, C, V).to(calc_type)
        # h/dh: [N, HV, K, V]（原始 datatype 保留，供 dg_last_accum）
        h_b = h[b_idx, h_offset:h_offset + N, :, :, :].reshape(N * H, K, V)
        dh_b = dh[b_idx, h_offset:h_offset + N, :, :, :].reshape(N * H, K, V)
        h_b_t = h_b.transpose(-1, -2).to(calc_type)               # [N*H, V, K]
        dh_b_t = dh_b.transpose(-1, -2).to(calc_type)            # [N*H, V, K]
        timer.stop("batched_data_prep")

        # ---- 1. State contributions ----
        timer.start("batched_bmm_state")
        dq_state = cast_round(torch.bmm(do_b, h_b_t))             # [N*H, C, K]
        dk_state = cast_round(torch.bmm(v_b, dh_b_t))            # [N*H, C, K]
        timer.stop("batched_bmm_state")

        timer.start("batched_bmm_dw")
        dv_b = dv[b_idx, s0:s0 + N * C, :, :].view(N, C, H, V).permute(0, 2, 1, 3).reshape(N * H, C, V).to(calc_type)
        dw_c = cast_round(torch.bmm(dv_b, h_b_t))            # [N*H, C, K]
        # [N*H, C, K] -> [N, H, C, K] -> [N, C, H, K] -> [N*C, H, K]
        dw_neg = (-dw_c).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        dw[b_idx, s0:s0 + N * C, :, :] = dw_neg
        timer.stop("batched_bmm_dw")

        timer.start("batched_mask")
        mask_f = get_causal_mask_f(C, q.device)                  # [1, C, C]
        timer.stop("batched_mask")

        # ---- 2. Decay scale ----
        timer.start("batched_decay_scale")
        # g: [N*C, HV] -> [N, HV, C] -> [N*HV, C]
        g_b = g[b_idx, s0:s0 + N * C, :].view(N, C, H).permute(0, 2, 1).reshape(N * H, C)  # gtype
        # 每个 chunk 的最后有效 token = chunk 内第 C-1 个位置（满 chunk 下成立）
        g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)                          # [N*H]
        exp_gc = torch.exp(g_b)                                                             # [N*H, C]
        exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])                               # [N*H, C]
        dq_state = dq_state * (exp_gc[:, :, None] * scale)
        dk_state = dk_state * exp_neg_gc_glast[:, :, None]
        timer.stop("batched_decay_scale")

        # ---- 3. Intra-chunk attention ----
        timer.start("batched_bmm_intra_ds")
        ds = cast_round(torch.bmm(do_b, v_b.transpose(1, 2)))                              # [N*H, C, C]
        timer.stop("batched_bmm_intra_ds")
        timer.start("batched_decay_apply")
        decay_mat = torch.exp(torch.min(g_b[:, :, None] - g_b[:, None, :], torch.tensor(0)))                           # [N*H, C, C]
        ds = ds * decay_mat
        ds = ds * mask_f
        ds = ds * scale
        timer.stop("batched_decay_apply")

        timer.start("batched_bmm_qk")
        qk_t = cast_round(torch.bmm(q_b, k_b.transpose(1, 2)))                             # [N*H, C, C]
        ds2 = ds * qk_t
        dg_c = ds2.sum(dim=-1)
        dg_c = dg_c - ds2.sum(dim=-2)
        timer.stop("batched_bmm_qk")

        # ---- dg_state（显式 product+sum，与原参考一致）----
        timer.start("batched_dg_state")
        dg_c = cast_round(dg_c)
        dg_c += (dq_state * q_b).sum(dim=-1)                                              # [N*H, C]
        dg_c = cast_round(dg_c)
        # k_b * dk_state 在 dg_c 和 dg_last_accum 中各用一次，提取公共子表达式
        k_dk_prod = k_b * dk_state                                                       # [N*H, C, K]
        dg_c = dg_c - k_dk_prod.sum(dim=-1)
        # h_prev*dh_curr 保留原始 datatype（与原实现一致）
        dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)                # [N*H]
        dg_last_accum = dg_last_accum + k_dk_prod.sum(dim=(-1, -2))                      # [N*H]
        timer.stop("batched_dg_state")

        # 每个 chunk 第 C-1 个位置累加 dg_last
        timer.start("batched_dg_intra_accum")
        dg_c = dg_c.reshape(N, H, C)
        dg_c[:, :, C - 1] = dg_c[:, :, C - 1] + dg_last_accum.reshape(N, H)
        # [N, H, C] -> [N, C, H] -> [N*C, H]
        dg[b_idx, s0:s0 + N * C, :] = dg_c.to(gtype).permute(0, 2, 1).reshape(N * C, H)
        timer.stop("batched_dg_intra_accum")

        # ---- 4. dq/dk intra ----
        timer.start("batched_bmm_dqdk_intra")
        dq_intra = cast_round(torch.bmm(ds, k_b))                                          # [N*H, C, K]
        dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))                          # [N*H, C, K]
        timer.stop("batched_bmm_dqdk_intra")

        timer.start("batched_accumulate")
        dq_total = dq_state + dq_intra
        dk_total = dk_state + dk_intra
        timer.stop("batched_accumulate")

        # ---- 写回 dq_hv/dk_hv ----
        timer.start("batched_write_back")
        # [N*H, C, K] -> [N, H, C, K] -> [N, C, H, K] -> [N*C, H, K]
        if datatype == calc_type:
            dq_hv[b_idx, s0:s0 + N * C, :, :] = dq_total.reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
            dk_hv[b_idx, s0:s0 + N * C, :, :] = dk_total.reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        else:
            dq_hv[b_idx, s0:s0 + N * C, :, :] = dq_total.to(datatype).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
            dk_hv[b_idx, s0:s0 + N * C, :, :] = dk_total.to(datatype).reshape(N, H, C, K).permute(0, 2, 1, 3).reshape(N * C, H, K)
        timer.stop("batched_write_back")

        timer.stop("batched_total")

    # Main Loop
    mode = "varlen" if cu_seqlens is not None else "dense"
    if verbose_timing:
        print(f"[chunk_bwd_dqkwg_cpu] start: B={B} T={T} HK={HK} HV={HV} K={K} V={V} "
              f"n_ratio={n_ratio} chunk_size={chunk_size} mode={mode} benchmark={benchmark}")
    timer.start("main_loop")
    if cu_seqlens is None:
        # Fixed length padding assumed or B*T
        C = chunk_size
        for b in range(B):
            n_full = T // C
            if n_full > 0:
                process_dense_batched(b, n_full, 0, 0)
            # ragged 尾部：走逐 chunk 路径
            rem_start = n_full * C
            if rem_start < T:
                process_sequence(b, rem_start, T, b, n_full)
    else:
        # Variable length
        chunk_location = torch.zeros(cu_seqlens.shape[0], dtype=torch.int64) #每个seq的chunk起始位置
        #chunk_location tensor([0, 64, 96, 128]) 代表：[0,63] [64,95] [96,127]

        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
            seq_length = end - start
            if i == 0:
                chunk_start_token_idx = 0
            else:
                chunk_start_token_idx = chunk_location[i]
            chunk_end_token_idx = chunk_start_token_idx + (seq_length + chunk_size - 1) // chunk_size
            chunk_location[i + 1] = chunk_end_token_idx

            # 在 Varlen 模式下，q/k/v 通常已经是 (Total_T, ...) 或者是 (1, Total_T, ...)
            # 但这里输入还是 (B, T, ...)，我们需要确认输入格式。
            # 通常 Triton varlen kernel 的输入 q 是 (Total_T, H, K)。
            # 如果输入是 packed (1, Total_T, ...)，b_idx 永远是 0。
            # 如果输入是 padded (B, T, ...)，则需要根据 cu_seqlens 切分。
            # 假设输入已根据 varlen 展平 (Batch=1) 或保持 Padded 格式。
            # 鉴于 Triton 代码 `i_b = i_bh // H`，如果 IS_VARLEN，逻辑略有不同。
            # 为保证通用性，这里假设输入是 Padded (B, T) 且 cu_seqlens 描述有效区域，
            # 或者 B=1 的 Packed 模式。
            if B == 1:
                process_sequence(0, start, end, i, chunk_location[i])
            else:
                # 如果是 Padded Batch 且提供了 cu_seqlens，这通常不常见，
                # 但如果发生，通常 cu_seqlens[i] 是第 i 个样本的长度。
                # 简化起见，我们假设 input 是 packed flat tensor 如果 cu_seqlens 存在。
                pass
    timer.stop("main_loop")

    timer.start("reduce")
    dq = dq_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
    dk = dk_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
    timer.stop("reduce")

    t_total = time.perf_counter() - t_total_start
    summary = timer.summary(t_total, chunk_count)
    if summary:
        print(summary)

    return dq, dk, dw, dg

def compare_data(golden_out, npu_out, threshold):
    a = golden_out.cpu().reshape(1, golden_out.numel())[0]
    b = npu_out.cpu().reshape(1, npu_out.numel())[0]
    diff = torch.abs(a - b) / torch.max(torch.abs(a), torch.tensor(1))

    
    # 处理 inf 情况：同号 inf 视为相等算通过，inf 与正常值（或异号 inf）算失败
    a_inf = torch.isinf(a)
    b_inf = torch.isinf(b)
    both_inf_same_sign = a_inf & b_inf & (torch.sign(a) == torch.sign(b))
    inf_mismatch = a_inf != b_inf
    diff = torch.where(both_inf_same_sign, torch.zeros_like(diff), diff)
    diff = torch.where(inf_mismatch, torch.full_like(diff, float('inf')), diff)

    # 处理 nan 情况：同为 nan 视为相等算通过，nan 与非 nan（正常值或 inf）算失败
    a_nan = torch.isnan(a)
    b_nan = torch.isnan(b)
    both_nan = a_nan & b_nan
    nan_mismatch = a_nan != b_nan
    diff = torch.where(both_nan, torch.zeros_like(diff), diff)
    diff = torch.where(nan_mismatch, torch.full_like(diff, float('inf')), diff)

    max_diff, max_index = torch.max(diff, dim = 0)
    c = max_diff < threshold
    print("阈值为：", threshold)
    print("最大误差为：", max_diff.item(), "，位于索引：", max_index.item())
    print("两个值：", a[max_index.item()], b[max_index.item()])
    if c==False:
        print("test case fail")
    else:
        print("test case ok")

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def prepare_chunk_indices_torch(
    cu_seqlens: torch.LongTensor,
    chunkSize: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunkSize).tolist()])
    # print("cu_seqlens is ", cu_seqlens)
    # print("indices is ", indices)

    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_chunk_indices(
    cu_seqlens: list[int],
    chunk_size: int
) -> list[int]: 
    """
    基于 cu_seqlens (list[int]) 生成 chunk 索引。
    
    注意：原 PyTorch 版本返回的是 shape [N, 2] 的 Tensor。
    为了保持纯 Python 兼容性，这里返回 list[tuple[start_seq_idx, chunk_idx_in_seq]]。
    如果算子需要扁平化的 list[int] (如 [s0, c0, s1, c1, ...])，请在调用前展开。
    
    逻辑复刻原代码：
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对
    """
    indices = []
    
    # 遍历每个序列段
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        length = end - start
        
        if length <= 0:
            continue
            
        # 计算该序列需要多少个 chunk
        # 等价于 cdiv(length, chunk_size)
        num_chunks = (length + chunk_size - 1) // chunk_size
        
        for chunk_id in range(num_chunks):
            # 原逻辑: indices.eq(0).cumsum(0) - 1 对应的是序列索引 i
            # 原逻辑: indices 对应的是 chunk_id
            indices.append((i))
            indices.append((chunk_id))
            
    return indices

# def chunk_bwd_dqkwg_cpu(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     do: torch.Tensor,
#     h: torch.Tensor,
#     dh: torch.Tensor,
#     w: torch.Tensor,
#     g: torch.Tensor,
#     dv: torch.Tensor,
#     scale: float,
#     cu_seqlens: torch.LongTensor,
#     chunk_size: int = 64,
#     benchmark = False
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     CPU Equivalent of chunk_bwd_kernel_dqkwg.

#     优化说明：将原按 HV 头展开的 Python 循环改为对 HV 维做批量 (bmm) 计算。
#     CPU 上 bmm 对每个 batch 调用与单次 mm 相同的 gemm，因此每个头的矩阵乘
#     与原循环逐位等价；所有 dtype 往返 (.to(datatype).to(calc_type)) 和运算顺序
#     均原样保留，维持与 kernel 一致的精度模拟。
#     """
#     calc_type = torch.float64 if benchmark else torch.float32
#     B, T, HK, K = q.shape
#     HV = v.shape[2]
#     V = v.shape[-1]
#     if HK <= 0 or HV <= 0 or HV % HK != 0:
#         raise ValueError(f"GVA requires HV divisible by HK, got HV={HV}, HK={HK}")
#     n_ratio = HV // HK  # HV = n_ratio * HK
#     datatype = q.dtype
#     gtype = g.dtype
#     if benchmark:
#         datatype = torch.float64
#         gtype = torch.float64

#     # Keep per-value-head contributions first, then reduce them into key heads.
#     dq_hv = torch.zeros((B, T, HV, K), dtype=datatype)
#     dk_hv = torch.zeros((B, T, HV, K), dtype=datatype)
#     dg = torch.zeros_like(g) if g is not None else None
#     dw = torch.zeros((B, T, HV, K), dtype=datatype)

#     # 缓存因果 mask（按 actual_chunk_len），避免每个 chunk 重复构造
#     mask_cache = {}

#     def get_causal_mask(L: int, device):
#         m = mask_cache.get(L)
#         if m is None or m.device != device:
#             idx = torch.arange(L, device=device)
#             m = idx[:, None] >= idx[None, :]
#             mask_cache[L] = m
#         return m

#     # 模拟 kernel 中间结果的 dtype 往返（datatype == calc_type 时为 no-op）
#     def cast_round(t):
#         return t.to(datatype).to(calc_type)

#     def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
#         seq_len = t_end - t_start
#         num_chunks = (seq_len + chunk_size - 1) // chunk_size

#         for i_t in range(num_chunks):
#             chunk_start_token_idx = t_start + i_t * chunk_size
#             chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
#             L = chunk_end_token_idx - chunk_start_token_idx
#             if L <= 0:
#                 continue
#             s = chunk_start_token_idx
#             e = chunk_end_token_idx

#             # ---- 取当前 chunk 全部头的数据，统一以 head 作为 batch 维 ----
#             # q/k: [L, HK, K] -> [HK, L, K] -> 复制 n_ratio 份 -> [HV, L, K]
#             #   head h_idx 对应 hk_idx = h_idx // n_ratio（与原循环一致）
#             q_h = q[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type).repeat_interleave(n_ratio, dim=0)
#             k_h = k[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type).repeat_interleave(n_ratio, dim=0)
#             # v/do: [L, HV, V] -> [HV, L, V]
#             v_h = v[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type)
#             do_h = do[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type)
#             # h/dh: [HV, K, V]（保留原始 datatype，供 dg_last_accum 使用）
#             h_prev = h[b_idx, i_t + chunk_start_idx, :, :, :]       # [HV, K, V]
#             dh_curr = dh[b_idx, i_t + chunk_start_idx, :, :, :]     # [HV, K, V]
#             h_prev_t = h_prev.transpose(-1, -2).to(calc_type)       # [HV, V, K]
#             dh_curr_t = dh_curr.transpose(-1, -2).to(calc_type)     # [HV, V, K]

#             # -----------------------------------------------------------
#             # 1. State Contributions (Inter-chunk)
#             # -----------------------------------------------------------
#             # b_dq += dot(b_do, b_h); b_dk += dot(b_v, b_dh)
#             dq_from_state = cast_round(torch.bmm(do_h, h_prev_t))   # [HV, L, K]
#             dk_from_state = cast_round(torch.bmm(v_h, dh_curr_t))   # [HV, L, K]
#             # if USE_DW: b_dw += dot(b_dv, b_h) (kernel 存 -b_dw)
#             if dv is not None:
#                 dv_h = dv[b_idx, s:e, :, :].permute(1, 0, 2).to(calc_type)
#                 dw_c = cast_round(torch.bmm(dv_h, h_prev_t))       # [HV, L, K]
#                 dw[b_idx, s:e, :, :] = (-dw_c).permute(1, 0, 2)

#             mask = get_causal_mask(L, q.device)

#             # -----------------------------------------------------------
#             # 2. Gating / Decay Logic Preparation
#             # -----------------------------------------------------------
#             if g is not None:
#                 g_h = g[b_idx, s:e, :].permute(1, 0)                # [HV, L] (gtype)
#                 g_last = g[b_idx, min(s + chunk_size, t_end) - 1, :]  # [HV]

#                 exp_gc = torch.exp(g_h)                              # [HV, L]
#                 exp_neg_gc_glast = torch.exp(-g_h + g_last[:, None]) # [HV, L]

#                 dq_from_state = dq_from_state * exp_gc[:, :, None] * scale
#                 dk_from_state = dk_from_state * exp_neg_gc_glast[:, :, None]

#                 # b_dg += sum(b_dq * b_q) ; b_dg -= sum(b_k * b_dk)
#                 dg_c = (dq_from_state * q_h).sum(dim=-1)             # [HV, L]
#                 dg_c = cast_round(dg_c)
#                 dg_c = dg_c - (k_h * dk_from_state).sum(dim=-1)      # [HV, L]
#                 dg_c = cast_round(dg_c)

#                 # b_dg_last += sum(h * dh) * exp(g_last) + sum(dk * k)
#                 dg_last_accum = (h_prev * dh_curr).sum(dim=(-1, -2)) * torch.exp(g_last)  # [HV]
#                 dg_last_accum = dg_last_accum + (dk_from_state * k_h).sum(dim=(-1, -2))   # [HV]

#                 # -----------------------------------------------------------
#                 # 3. Intra-chunk Attention
#                 # -----------------------------------------------------------
#                 ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
#                 decay_mat = torch.exp(g_h[:, :, None] - g_h[:, None, :])  # [HV, L, L]
#                 ds = torch.where(mask[None], ds * decay_mat, 0.0) * scale

#                 # b_ds2 = b_ds * (q @ k.T)
#                 qk_t = cast_round(torch.bmm(q_h, k_h.transpose(1, 2)))   # [HV, L, L]
#                 ds2 = ds * qk_t
#                 dg_c = dg_c + ds2.sum(dim=-1)
#                 dg_c = dg_c - ds2.sum(dim=-2)
#                 dg_c = dg_c.to(datatype).to(gtype)                       # [HV, L]

#                 # 仅块最后一个有效 token 累加 dg_last
#                 dg_c[:, L - 1] = dg_c[:, L - 1] + dg_last_accum.to(gtype)
#                 dg[b_idx, s:e, :] = dg_c.permute(1, 0)                   # [L, HV]

#                 # -----------------------------------------------------------
#                 # 4. Final Accumulation for dq, dk
#                 # -----------------------------------------------------------
#                 dq_intra = cast_round(torch.bmm(ds, k_h))               # [HV, L, K]
#                 dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_h))# [HV, L, K]

#                 dq_total = dq_from_state + dq_intra
#                 dk_total = dk_from_state + dk_intra
#             else:
#                 # No decay：保留与原实现一致的 scale 顺序
#                 dk_from_state = dk_from_state * scale
#                 dq_from_state = dq_from_state * scale

#                 ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
#                 ds = torch.where(mask[None], ds, 0.0)

#                 dq_intra = cast_round(torch.bmm(ds, k_h))               # [HV, L, K]
#                 dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_h))# [HV, L, K]
#                 dk_intra = dk_intra * scale
#                 dq_total = (dq_from_state + dq_intra) * scale
#                 dk_total = dk_from_state + dk_intra

#             dq_hv[b_idx, s:e, :, :] = dq_total.to(datatype).permute(1, 0, 2)
#             dk_hv[b_idx, s:e, :, :] = dk_total.to(datatype).permute(1, 0, 2)

#     # Main Loop
#     if cu_seqlens is None:
#         # Fixed length padding assumed or B*T
#         for b in range(B):
#             process_sequence(b, 0, T, b, 0)
#     else:
#         # Variable length
#         chunk_location = torch.zeros(cu_seqlens.shape[0], dtype=torch.int64) #每个seq的chunk起始位置
#         #chunk_location tensor([0, 64, 96, 128]) 代表：[0,63] [64,95] [96,127]

#         for i in range(len(cu_seqlens) - 1):
#             start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
#             seq_length = end - start
#             if i == 0:
#                 chunk_start_token_idx = 0
#             else:
#                 chunk_start_token_idx = chunk_location[i]
#             chunk_end_token_idx = chunk_start_token_idx + (seq_length + chunk_size - 1) // chunk_size
#             chunk_location[i + 1] = chunk_end_token_idx

#             # 在 Varlen 模式下，q/k/v 通常已经是 (Total_T, ...) 或者是 (1, Total_T, ...)
#             # 但这里输入还是 (B, T, ...)，我们需要确认输入格式。
#             # 通常 Triton varlen kernel 的输入 q 是 (Total_T, H, K)。
#             # 如果输入是 packed (1, Total_T, ...)，b_idx 永远是 0。
#             # 如果输入是 padded (B, T, ...)，则需要根据 cu_seqlens 切分。
#             # 假设输入已根据 varlen 展平 (Batch=1) 或保持 Padded 格式。
#             # 鉴于 Triton 代码 `i_b = i_bh // H`，如果 IS_VARLEN，逻辑略有不同。
#             # 为保证通用性，这里假设输入是 Padded (B, T) 且 cu_seqlens 描述有效区域，
#             # 或者 B=1 的 Packed 模式。
#             if B == 1:
#                 process_sequence(0, start, end, i, chunk_location[i])
#             else:
#                 # 如果是 Padded Batch 且提供了 cu_seqlens，这通常不常见，
#                 # 但如果发生，通常 cu_seqlens[i] 是第 i 个样本的长度。
#                 # 简化起见，我们假设 input 是 packed flat tensor 如果 cu_seqlens 存在。
#                 pass 

#     dq = dq_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
#     dk = dk_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)

#     return dq, dk, dw, dg

# def chunk_bwd_dqkwg_cpu(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     do: torch.Tensor,
#     h: torch.Tensor,
#     dh: torch.Tensor,
#     w: torch.Tensor,
#     g: torch.Tensor,
#     dv: torch.Tensor,
#     scale: float,
#     cu_seqlens: torch.LongTensor,
#     chunk_size: int = 64,
#     benchmark = False
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     CPU Equivalent of chunk_bwd_kernel_dqkwg.
#     """
#     if benchmark:
#         calc_type = torch.float64
#     else:
#         calc_type = torch.float32
#     q.to(calc_type)
#     k.to(calc_type)
#     v.to(calc_type)
#     do.to(calc_type)
#     h.to(calc_type)
#     dh.to(calc_type)
    
#     g.to(calc_type)
#     dv.to(calc_type)
#     B, T, HK, K = q.shape
#     HV = v.shape[2]
#     V = v.shape[-1]
#     if HK <= 0 or HV <= 0 or HV % HK != 0:
#         raise ValueError(f"GVA requires HV divisible by HK, got HV={HV}, HK={HK}")
#     n_ratio = HV // HK  # HV = n_ratio * HK
#     datatype = q.dtype
#     gtype = g.dtype
#     if benchmark:
#         datatype = torch.float64
#         gtype = torch.float64
#     g_gamma = None
    
#     # Keep per-value-head contributions first, then reduce them into key heads.
#     dq_hv = torch.zeros((B, T, HV, K), dtype=datatype)
#     dk_hv = torch.zeros((B, T, HV, K), dtype=datatype)
#     dg = torch.zeros_like(g) if g is not None else None
#     dw = torch.zeros((B, T, HV, K), dtype=datatype)
#     w = torch.zeros((B, T, HV, K), dtype=datatype)

    
#     # 辅助函数：处理单个序列的逻辑
#     def process_sequence(b_idx, t_start, t_end, seq_idx_in_batch, chunk_start_idx):
#         # 计算该序列有多少个块
#         seq_len = t_end - t_start
#         num_chunks = (seq_len + chunk_size - 1) // chunk_size
#         # print("H(head)", H, "num_chunks", num_chunks, "b_idx", b_idx, "t_start", t_start, "t_end", t_end, "seq_idx_in_batch", seq_idx_in_batch, "chunk_start_idx", chunk_start_idx)

        
#         for h_idx in range(HV):
#             # h_idx is hv_idx; compute hk_idx for q/k access
#             hk_idx = h_idx // n_ratio
#             # 获取当前头的 gamma (如果 USE_G_GAMMA)
#             gamma_val = None
#             if g_gamma is not None:
#                 gamma_val = g_gamma[h_idx].item()

#             for i_t in range(num_chunks):
#                 # 块的绝对起始位置
#                 chunk_start_token_idx = t_start + i_t * chunk_size
#                 chunk_end_token_idx = min(t_start + (i_t + 1) * chunk_size, t_end)
#                 actual_chunk_len = chunk_end_token_idx - chunk_start_token_idx

#                 # 当前块在 h/dh 中的索引 (NT 维度)
#                 # Triton 代码逻辑: i_tg = i_b * NT + i_t (定长) 或 i_t (变长且 chunk_indices 处理)
#                 # 这里我们假设 h 形状为 (B, H, NT, K, V) 或者兼容的扁平结构。
#                 # 为简化，假设标准 FLA 布局 (B, H, NT, K, V)
#                 # 注意：Triton 中 h 是指向第 i_t 个块的*起始*状态 (即上一个块的输出)
                
#                 # 切片当前块的数据
                
#                 q_c = q[b_idx, chunk_start_token_idx:chunk_end_token_idx, hk_idx, :]  # [BT, K]
#                 k_c = k[b_idx, chunk_start_token_idx:chunk_end_token_idx, hk_idx, :]  # [BT, K]

#                 v_c = v[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :]  # [BT, V]
#                 do_c = do[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] # [BT, V]

#                 # 获取状态 (h_prev) 和 状态梯度 (dh_curr)
#                 # h[..., i_t, ...] 存储的是第 i_t 块之前的状态 (即第 i_t-1 块的输出)
#                 # print("\th ", h.shape, f"h[{b_idx}, {i_t} + {chunk_start_idx}, {h_idx}, :, :]")
#                 h_prev = h[b_idx, i_t + chunk_start_idx, h_idx, :, :]  # [K, V]  ## 不对齐的情况??
#                 dh_curr = dh[b_idx, i_t + chunk_start_idx, h_idx, :, :] # [K, V]

#                 # -----------------------------------------------------------
#                 # 1. State Contributions (Inter-chunk)
#                 # -----------------------------------------------------------
#                 # Triton: b_dq += dot(b_do, b_h) -> do @ h_prev.T
#                 # h_prev 是 [K, V], do_c 是 [BT, V] -> [BT, K]
#                 dq_from_state = do_c.to(calc_type) @ h_prev.transpose(-1, -2).to(calc_type)

#                 dq_from_state = dq_from_state.to(datatype).to(calc_type)

#                 # Triton: b_dk += dot(b_v, b_dh) -> v @ dh_curr.T
#                 # dh_curr 是 [K, V], v_c 是 [BT, V] -> [BT, K]
#                 dk_from_state = v_c.to(calc_type) @ dh_curr.transpose(-1, -2).to(calc_type)
#                 dk_from_state = dk_from_state.to(datatype).to(calc_type)
#                 # Triton: if USE_DW -> b_dw += dot(b_dv, b_h)
#                 if w is not None and dv is not None:
#                     dv_c = dv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] # [BT, V]
#                     # dw_c: [BT, K]
#                     dw_c_val = dv_c.to(calc_type) @ h_prev.transpose(-1, -2).to(calc_type)
#                     dw_c_val = dw_c_val.to(datatype).to(calc_type)
#                     # Triton stores -b_dw
#                     dw[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = -dw_c_val

#                 # -----------------------------------------------------------
#                 # 2. Gating / Decay Logic Preparation
#                 # -----------------------------------------------------------
#                 # 构建 g_c (decay values)
#                 if g is not None:
#                     g_c = g[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] # [BT]

#                     g_last = g[b_idx, min(chunk_start_token_idx + chunk_size, t_end) - 1, h_idx]
                    
#                     # Triton: b_dg_last += sum(h * dh)
#                     dg_last_accum = (h_prev * dh_curr).sum()

#                     dg_last_accum = dg_last_accum * torch.exp(g_last)
#                     # Apply decay to state contributions

#                     dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
#                     dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]

#                     # Accumulate gradients into dg (from state terms)
#                     # b_dg += sum(b_dq * b_q)
#                     dg_c = (dq_from_state * q_c).sum(dim=-1)
#                     # print("ADD0.A", dg_c.to(datatype).to(calc_type))
#                     dg_c = dg_c.to(datatype).to(calc_type)         #ADD0.A

#                     # b_dg -= sum(b_k * b_dk)
#                     dg_c -= (k_c * dk_from_state).sum(dim=-1)           #ADD0.B
#                     # print("k_c",k_c)
#                     # print("dk_from_state",dk_from_state)
#                     # print("k_c * dk_from_state",( k_c * dk_from_state)[0])
#                     # print("Add0.B", -(k_c * dk_from_state).sum(dim=-1))
#                     dg_c = dg_c.to(datatype).to(calc_type)

#                     # b_dg_last += sum(b_dk * b_k)
#                     # print(f"dg_last_accum {dg_last_accum} += (dk_from_state * k_c).sum() {(dk_from_state * k_c).sum()}")
#                     # if h_idx == 0 and i_t == 31:
#                     #     print("     sum0 result", (dk_from_state * k_c).sum())
#                     dg_last_accum += (dk_from_state * k_c).sum()
#                     # print("dg_last_accum += (dk_from_state * k_c).sum()", dg_last_accum)
#                     # pause()
                    

#                 elif g_gamma is not None:
#                     # Scalar decay
#                     # b_g = b_gamma * (arange + 1)
#                     # b_g_last = b_gamma * actual_chunk_len
#                     # 这里模拟 Triton 里的相对 decay 逻辑
#                     arange = torch.arange(actual_chunk_len, device=q.device, dtype=q.dtype)
#                     g_c = gamma_val * (arange + 1)
#                     g_last = gamma_val * actual_chunk_len
                    
#                     dq_from_state = dq_from_state * torch.exp(g_c)[:, None] * scale
#                     dk_from_state = dk_from_state * torch.exp(-g_c + g_last)[:, None]
#                     # USE_G_GAMMA 模式下不需要计算 dg
#                 else:
#                     # No decay
#                     # Triton: b_dk *= scale (else block)
#                     dk_from_state = dk_from_state * scale
#                     dq_from_state = dq_from_state * scale

#                 # -----------------------------------------------------------
#                 # 3. Intra-chunk Attention
#                 # -----------------------------------------------------------
#                 ds = do_c.to(calc_type) @ v_c.transpose(-1, -2).to(calc_type) # [BT, BT]
#                 ds = ds.to(datatype).to(calc_type)

                
#                 # Causal Mask
#                 i_indices = torch.arange(actual_chunk_len, device=q.device)[:, None]
#                 j_indices = torch.arange(actual_chunk_len, device=q.device)[None, :]
#                 mask = i_indices >= j_indices
                
#                 if g is not None:
#                     # Decay: exp(g[i] - g[j])

#                     decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
#                     # if h_idx == 0 and i_t == 7:
#                     #     print("g_c[:, None] - g_c[None, :]", g_c[:, None] - g_c[None, :])
#                     #     print("decay_mat = torch.exp(g_c[:, None] - g_c[None, :])", decay_mat)

#                     ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale
#                     # print("decay_mat",decay_mat)
#                     # print("ds",ds)

                    
#                     # DG Calculation Part 2 (Intra-chunk)
#                     # b_ds2 = b_ds * (q @ k.T)
#                     qk_t = q_c.to(calc_type) @ k_c.transpose(-1, -2).to(calc_type)
#                     qk_t = qk_t.to(datatype).to(calc_type)


#                     ds2 = ds * qk_t

#                     # print("ADD0.C : +ds2.sum(dim=1)", ds2.sum(dim=1))
#                     # print("ADD0.D : -ds2.sum(dim=0)", ds2.sum(dim=0))
#                     dg_c += ds2.sum(dim=1)
#                     dg_c -= ds2.sum(dim=0)

#                     # dg_c = dg_c_C.to(torch.float16) + dg_c_D.to(torch.float16) + dg_c_A.to(torch.float16) + dg_c_B.to(torch.float16)
#                     dg_c = dg_c.to(datatype).to(gtype)

#                     # print("dg_c after", dg_c.shape)
#                     # pause()
                    
#                     # Finalize dg: revcumsum-like logic
#                     # Triton: b_dg = where(o_t < T-1, b_dg, b_dg + b_dg_last)
#                     # 只有块的最后一个有效 token 加上 dg_last_accum
#                     # 注意：Triton 内核中的 revcumsum 通常在单独内核或最后处理，
#                     # 但这里代码片段显示的是直接加上。
#                     # 实际上 dg 在时间轴上是累积的梯度。
#                     # 根据 Triton 代码: b_dg = ... + (idx == last ? b_dg_last : 0)
#                     # 这里的 dg_c 仅仅是该位置的梯度 contribution。
#                     # 为了完全匹配 Triton 的输出，我们需要把 dg_last_accum 加到块的最后。
#                     if actual_chunk_len > 0:
#                         dg_c[actual_chunk_len - 1] += dg_last_accum.to(gtype)  ## 实际上是is_last_mask

#                     #     print(f"dg_c[{actual_chunk_len - 1}] += {dg_last_accum}")
#                     # print("dg_c", dg_c)
#                     dg[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx] = dg_c
#                     # print("dg_c",dg_c)

#                 elif g_gamma is not None:

#                     decay_mat = torch.exp(g_c[:, None] - g_c[None, :])
                    
#                     ds = torch.where(mask, ds * decay_mat, torch.zeros_like(ds)) * scale

#                 else:
#                     ds = torch.where(mask, ds, torch.zeros_like(ds))
#                     # 在 else 分支，triton 代码: b_dq *= scale (最后)
#                     # 但前面 state part 已经 scale 了。
#                     # ds 计算时不乘 scale，最后 dq 乘 scale。
#                     # 为了统一，这里先不乘 scale，下面加完后再处理，或者这里乘了下面不再乘。
#                     # Triton 代码: b_dk += dot(trans(b_ds), b_q) * scale
#                     # b_dq += dot(b_ds, b_k); b_dq *= scale
#                     pass # logic handled below

#                 # -----------------------------------------------------------
#                 # 4. Final Accumulation for dq, dk
#                 # -----------------------------------------------------------
#                 # dq += ds @ k

#                 dq_intra = ds.to(calc_type) @ k_c.to(calc_type)
#                 # if h_idx == 0 and i_t == 7:
#                 #     print("ds.to(torch.float32)",ds.to(torch.float32))
#                 #     print("k_c.to(torch.float32)",k_c.to(torch.float32))
#                 dq_intra = dq_intra.to(datatype).to(calc_type)
#                 # dk += ds.T @ q
#                 dk_intra = ds.transpose(-1, -2).to(calc_type) @ q_c.to(calc_type)
#                 dk_intra = dk_intra.to(datatype).to(calc_type)

                
#                 if g is None and g_gamma is None:
#                     # Special scaling for "No Decay" mode based on Triton code
#                     dk_intra = dk_intra * scale
#                     dq_total = (dq_from_state + dq_intra) * scale # Triton: b_dq *= scale at end
#                     dk_total = dk_from_state + dk_intra
#                 else:
#                     dq_total = dq_from_state + dq_intra
#                     # if h_idx == 0 and i_t == 7:
#                     #     print("dq_from_state",dq_from_state[-1])
#                     #     print("dq_intra",dq_intra[-1])
#                     #     print("dq_total",dq_total.shape,dq_total[-1])
#                     dk_total = dk_from_state + dk_intra

#                     # print(h_idx,i_t,"dk_from_state",dk_from_state)
#                     # print(h_idx,i_t,"dk_intra",dk_intra)
#                     # print(h_idx,i_t,"dk_total",dk_total)


#                 dq_hv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dq_total.to(datatype)
#                 dk_hv[b_idx, chunk_start_token_idx:chunk_end_token_idx, h_idx, :] = dk_total.to(datatype)      

#     # Main Loop
#     if cu_seqlens is None:
#         # Fixed length padding assumed or B*T
#         for b in range(B):
#             process_sequence(b, 0, T, b, 0)
#     else:
#         # Variable length
#         chunk_location = torch.zeros(cu_seqlens.shape[0], dtype=torch.int64) #每个seq的chunk起始位置
#         #chunk_location tensor([0, 64, 96, 128]) 代表：[0,63] [64,95] [96,127]

#         for i in range(len(cu_seqlens) - 1):
#             start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
#             seq_length = end - start
#             # print("seq_length", seq_length)
#             if i == 0:
#                 chunk_start_token_idx = 0
#             else:
#                 chunk_start_token_idx = chunk_location[i]
#             # print("chunk_start_token_idx before", chunk_start_token_idx)
#             chunk_end_token_idx = chunk_start_token_idx + (seq_length + chunk_size - 1) // chunk_size
#             # print("chunk_end_token_idx after", chunk_end_token_idx)
#             chunk_location[i + 1] = chunk_end_token_idx

#             # 在 Varlen 模式下，q/k/v 通常已经是 (Total_T, ...) 或者是 (1, Total_T, ...)
#             # 但这里输入还是 (B, T, ...)，我们需要确认输入格式。
#             # 通常 Triton varlen kernel 的输入 q 是 (Total_T, H, K)。
#             # 如果输入是 packed (1, Total_T, ...)，b_idx 永远是 0。
#             # 如果输入是 padded (B, T, ...)，则需要根据 cu_seqlens 切分。
#             # 假设输入已根据 varlen 展平 (Batch=1) 或保持 Padded 格式。
#             # 鉴于 Triton 代码 `i_b = i_bh // H`，如果 IS_VARLEN，逻辑略有不同。
#             # 为保证通用性，这里假设输入是 Padded (B, T) 且 cu_seqlens 描述有效区域，
#             # 或者 B=1 的 Packed 模式。
#             if B == 1:
#                 print(f"start {start}, end {end}")
#                 # if (i == 0):
#                 #     continue
#                 process_sequence(0, start, end, i, chunk_location[i])
#             else:
#                 # 如果是 Padded Batch 且提供了 cu_seqlens，这通常不常见，
#                 # 但如果发生，通常 cu_seqlens[i] 是第 i 个样本的长度。
#                 # 简化起见，我们假设 input 是 packed flat tensor 如果 cu_seqlens 存在。
#                 pass 

#     dq = dq_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)
#     dk = dk_hv.view(B, T, HK, n_ratio, K).sum(dim=3).to(datatype)

#     return dq, dk, dw, dg

# -------------------------------------------------------------------------
# 使用示例 / 验证
# -------------------------------------------------------------------------
if __name__ == "__main__":
    RANDOM_DATA = True
    torch.manual_seed(1)
    case_number = 21
    if len(sys.argv) > 1:
        regen = sys.argv[1]
        if regen == "random":
            print("[test.py] regenerate all random data!")
            RANDOM_DATA=True

    # 简单的形状参数
    K, V = 128, 128
    calc_type = torch.float16
    isVarLen = False
    chunk_size = 128
    cases = [   #B,H,T,chunk_size,dtype,Gtype,scale,cu_seqlens
        [64,8,8,1024,64,torch.float16,torch.float16,0.088,None],
        [32,16,16,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None],
        [16,32,32,4096,64,torch.float16,torch.float16,0.0442,None],
        [8,32,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [128,4,4,1024,64,torch.float16,torch.float16,0.088,None],
        [64,4,4,4096,128,torch.bfloat16,torch.bfloat16,0.0625,None],
        [32,16,16,8192,64,torch.float16,torch.float16,0.0442,None],
        [16,32,32,16384,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [64,8,8,2048,128,torch.float16,torch.float16,0.0625,None],
        [32,16,16,4096,128,torch.bfloat16,torch.bfloat16,0.0442,None],
        [16,32,32,8192,128,torch.float16,torch.float16,0.03125,None],
        [8,32,32,8192,128,torch.bfloat16,torch.bfloat16,0.0221,None],  #C12
        [1,4,4,1024,64,torch.float16,torch.float16,0.088,None],
        [48,8,8,2048,64,torch.bfloat16,torch.bfloat16,0.0625,None],
        [24,16,16,4096,64,torch.float16,torch.float16,0.0442,None],
        [12,32,32,8192,64,torch.bfloat16,torch.bfloat16,0.03125,None],
        [1,16,16,32768,64,torch.float16,torch.float32,0.0625,torch.tensor([0,16,20000,30000,32768])],      # V1
        [1,8,8,65536,64,torch.bfloat16,torch.bfloat16,0.0625,torch.tensor([0,16,20000,65536])],
        [1,32,32,65536,64,torch.float16,torch.float32,0.0442,torch.tensor([0,16,20000,50000,65536])],
        [1,32,32,262144,64,torch.bfloat16,torch.bfloat16,0.03125,torch.tensor([0,16,20000,50000,65536,210000,262144])],
        # [1,1,1,32768,128,torch.float16,torch.float32,0.088,torch.tensor([0,1])],
        [8,8,8,4096,64,torch.float16,torch.float16,0.088,None],  #21 [0,16,128] [0,16,135,512]
        [1,32,32,16384,64,torch.bfloat16,torch.float32,0.088,None],  #21 [0,16,128]
    ]
    device_id = int(os.environ.get("TEST_DEVICE_ID", 3))
    

    dtype = torch.float16
    Gtype = torch.float16
    B, HK, HV = 4, 8, 8
    T = 1024
    scale = 0.088
    if isVarLen:
        cu_seqlens = torch.cumsum(torch.tensor([0, 3, 64, 63, 66, 260]), dim=0)
    else:
        cu_seqlens = None
    if case_number != -1:
        single_case = cases[case_number-1]  #case_01 => cases[0]
        dtype = single_case[5]
        Gtype = single_case[6]
        B, HK, HV = single_case[0], single_case[1], single_case[2]
        chunk_size = single_case[4]
        cu_seqlens = single_case[8]
        cu_seqlens_torch = torch.tensor(cu_seqlens) if cu_seqlens is not None else None

        if single_case[8] is None:
            isVarLen = False
        else:
            isVarLen = True
        # isVarLen == single_case[7] != None
        T = single_case[3]
        scale = single_case[7]

    if isVarLen:
        B = 1  ##变长只支持B=1
        T = cu_seqlens_torch[-1]
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        num_chunks = len(chunk_indices) // 2
        print("chunk_indices",chunk_indices)
    else:
        chunk_indices = None
        num_chunks = (T + chunk_size - 1) // chunk_size
    
    if RANDOM_DATA:
        # q = torch.randn(B,T,HK,K, dtype=dtype) # std≈5e-6#torch.randn([B, T, H, K], dtype=dtype)
        # k = torch.randn(B,T,HK,K, dtype=dtype) # torch.randn([B, T, H, K], dtype=dtype)
        # v = torch.randn(B,T,HV,V, dtype=dtype) # torch.randn([B, T, H, V], dtype=dtype)

        # # g = torch.randn(B,T,H, dtype=dtype) * 5e-2   # torch.randn([B, T, H], dtype=Gtype)
        # g = -torch.sort(torch.rand(B*T*HV), descending=False)[0].reshape((B,T,HV)).to(Gtype)    #G必须递减且为负数
        # # print("g",g)
        # do = torch.randn(B,T,HV,V, dtype=dtype) # torch.randn([B, T, H, V], dtype=dtype)

        # dv = torch.randn(B,T,HV,V, dtype=dtype) # torch.randn([B, T, H, V], dtype=dtype)
        # w = torch.randn(B,T,HV,K, dtype=dtype) # torch.randn([B, T, H, K], dtype=dtype)

        # h = torch.randn(B, num_chunks, HV, K, V, dtype=dtype)  # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        # dh = torch.randn(B, num_chunks, HV, K, V, dtype=dtype) # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        q = torch.randn(B,T,HK,K, dtype=dtype) * 5e-2 # std≈5e-6#torch.randn([B, T, H, K], dtype=dtype)
        k = torch.randn(B,T,HK,K, dtype=dtype) * 5e-2 # torch.randn([B, T, H, K], dtype=dtype)
        v = torch.randn(B,T,HV,V, dtype=dtype) * 5e-2 # torch.randn([B, T, H, V], dtype=dtype)

        # g = torch.randn(B,T,H, dtype=dtype) * 5e-2   # torch.randn([B, T, H], dtype=Gtype)
        g = -torch.sort(torch.rand(B*T*HV), descending=False)[0].reshape((B,T,HV)).to(Gtype)    #G必须递减且为负数
        # print("g",g)
        do = torch.randn(B,T,HV,V, dtype=dtype) * 5e-2 # torch.randn([B, T, H, V], dtype=dtype)

        dv = torch.randn(B,T,HV,V, dtype=dtype) * 5e-1 # torch.randn([B, T, H, V], dtype=dtype)
        w = torch.randn(B,T,HV,K, dtype=dtype) * 5e-2 # torch.randn([B, T, H, K], dtype=dtype)

        h = torch.randn(B, num_chunks, HV, K, V, dtype=dtype) * 5e-2  # torch.randn([B, num_chunks, H, K, V], dtype=dtype)
        dh = torch.randn(B, num_chunks, HV, K, V, dtype=dtype) * 5e-2 # torch.randn([B, num_chunks, H, K, V], dtype=dtype)

    q = q.to(dtype).to(calc_type)
    k = k.to(dtype).to(calc_type)
    v = v.to(dtype).to(calc_type)
    h = h.to(dtype).to(calc_type)
    g = g.to(Gtype).to(calc_type)
    do = do.to(dtype).to(calc_type)
    dh = dh.to(dtype).to(calc_type)
    dv = dv.to(dtype).to(calc_type)
    w = w.to(dtype).to(calc_type)
    print("entering chunk_bwd_dqkwg")
    print(f"q: {q.shape} {dtype} => {q.dtype}")
    print(f"k: {k.shape} {dtype} => {k.dtype}")
    print(f"v: {v.shape} {dtype} => {v.dtype}")
    print(f"w: {w.shape} {dtype} => {w.dtype}")
    print(f"g: {g.shape} {Gtype} => {g.dtype}")
    print(f"h: {h.shape} {dtype} => {h.dtype}")
    print(f"dv: {dv.shape} {dtype} => {dv.dtype}")
    print(f"do: {do.shape} {dtype} => {do.dtype}")
    print(f"dh: {dh.shape} {dtype} => {dh.dtype}")
    if cu_seqlens == None:
        print("cu_seqlens is None")
    else:
        print(f"cu_seqlens: {cu_seqlens_torch.shape} {cu_seqlens_torch.dtype} {cu_seqlens_torch}")
        # print(f"chunk_indices: {chunk_indices.shape} {chunk_indices.dtype} {chunk_indices}")
    print(f"scale: {scale}")
    print(f"chunk_size: {chunk_size}")


    print("==============start NPU=============")
    torch.npu.set_device(device_id)
    print("dtype")
    q_npu = torch.transpose(q, 1, 2).to(dtype).npu()
    print("q_npu", q_npu.shape, q_npu.dtype)
    k_npu = torch.transpose(k, 1, 2).to(dtype).npu()
    print("k_npu", k_npu.shape, k_npu.dtype)
    v_npu = torch.transpose(v, 1, 2).to(dtype).npu()
    print("v_npu", v_npu.shape, v_npu.dtype)
    w_npu = torch.transpose(w, 1, 2).to(dtype).npu()
    print("w_npu", w_npu.shape, w_npu.dtype)
    g_npu = torch.transpose(g, 1, 2).to(Gtype).npu()
    print("g_npu", g_npu.shape, g_npu.dtype)
    h_npu = torch.transpose(h, 1, 2).to(dtype).npu()
    print("h_npu", h_npu.shape, h_npu.dtype)
    dv_npu = torch.transpose(dv, 1, 2).to(dtype).npu()
    print("dv_npu", dv_npu.shape, dv_npu.dtype)
    do_npu = torch.transpose(do, 1, 2).to(dtype).npu()
    print("do_npu", do_npu.shape, do_npu.dtype)
    dh_npu = torch.transpose(dh, 1, 2).to(dtype).npu()
    print("dh_npu", dh_npu.shape, dh_npu.dtype)
    # q_npu = q.permute(0, 2, 1, 3).to(dtype).npu()
    # print("q_npu", q_npu.shape, q_npu.dtype)
    # k_npu = k.permute(0, 2, 1, 3).to(dtype).npu()
    # print("k_npu", k_npu.shape, k_npu.dtype)
    # v_npu = v.permute(0, 2, 1, 3).to(dtype).npu()
    # print("v_npu", v_npu.shape, v_npu.dtype)
    # w_npu = w.permute(0, 2, 1, 3).to(dtype).npu()
    # print("w_npu", w_npu.shape, w_npu.dtype)
    # g_npu = g.permute(0, 2, 1).to(Gtype).npu()
    # print("g_npu", g_npu.shape, g_npu.dtype)
    # h_npu = h.permute(0, 2, 1, 3, 4).to(dtype).npu()
    # print("h_npu", h_npu.shape, h_npu.dtype)
    # dv_npu = dv.permute(0, 2, 1, 3).to(dtype).npu()
    # print("dv_npu", dv_npu.shape, dv_npu.dtype)
    # do_npu = do.permute(0, 2, 1, 3).to(dtype).npu()
    # print("do_npu", do_npu.shape, do_npu.dtype)
    # dh_npu = dh.permute(0, 2, 1, 3, 4).to(dtype).npu()
    # print("dh_npu", dh_npu.shape, dh_npu.dtype)
    # cu_seqlens_npu = cu_seqlens if cu_seqlens is not None else None
    chunk_indices_npu = chunk_indices if cu_seqlens is not None else None
    print("chunk_indices_npu")
    down_tri = q_npu

    print("qqqqqqqq")
    dq_npu, dk_npu, dw_npu, dg_npu = ascendc_ops.npu_chunk_bwd_dqkwg(
        q_npu, k_npu, v_npu, g_npu, h_npu, do_npu, dh_npu, dv_npu, chunk_size, cu_seqlens=cu_seqlens, w=None, g_gamma=None, chunk_indices=chunk_indices_npu, scale=scale, use_exp2=None, transpose_state_layout=None
    )
    print("custom_ops.npu_chunk_bwd_dqkwg done")
    dq_npu = dq_npu.cpu()
    dk_npu = dk_npu.cpu()
    dw_npu = dw_npu.cpu()
    dg_npu = dg_npu.cpu()

    # print("Output shapes:", dq.shape, dk.shape, dg.shape, dw.shape)
    print("dq_npu", dq_npu.shape, dq_npu.dtype)
    print("dk_npu", dk_npu.shape, dk_npu.dtype)
    print("dw_npu", dw_npu.shape, dw_npu.dtype)
    print("dg_npu", dg_npu.shape, dg_npu.dtype)

    # print("dq_npu[0][0][-1]", dq_npu[0][0][-1])

    print("=====start cpu=========")


    dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
        q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens_torch, chunk_size
    )
    # dq = dq.to(dtype)
    # dk = dk.to(dtype)
    # dw = dw.to(dtype)
    # dg = dg.to(Gtype)
    dq = torch.transpose(dq, 1, 2).cpu()
    dk = torch.transpose(dk, 1, 2).cpu()
    dw = torch.transpose(dw, 1, 2).cpu()
    dg = torch.transpose(dg, 1, 2).cpu()
    # print("dq[0][0][-1]", dq[0][0][-1])
    # print("dk", dk)
    # print("dw", dw)
    # print("dg", dg)

    print("dq", dq.cpu().shape, dq.cpu().dtype)
    print("dk", dk.cpu().shape, dk.cpu().dtype)
    print("dw", dw.cpu().shape, dw.cpu().dtype)
    print("dg", dg.cpu().shape, dg.cpu().dtype)

    type_dict = {torch.float16:"float16", torch.float32:"float32",torch.bfloat16:"bf16"}
    # single(dq_npu,dq,calc_count=100000,dtype=type_dict[dtype])
    # single(dk_npu,dk,calc_count=100000,dtype=type_dict[dtype])
    # single(dw_npu,dw,calc_count=100000,dtype=type_dict[dtype])
    # single(dg_npu,dg,calc_count=100000,dtype=type_dict[Gtype])

    compare_data(dq, dq_npu, threshold=2**-8 + 2**-14)
    compare_data(dk, dk_npu, threshold=2**-8 + 2**-14)
    compare_data(dw, dw_npu, threshold=2**-8 + 2**-14)
    compare_data(dg, dg_npu, threshold=2**-8 + 2**-14)
    # print("dq = ", dq[1][4][128][6])
    # print("dq_npu = ", dq_npu[1][4][128][6])
    # print("dq = ", dq[2][4][832][18])
    # print("dq_npu = ", dq_npu[2][4][832][18])

    print("All done!")
