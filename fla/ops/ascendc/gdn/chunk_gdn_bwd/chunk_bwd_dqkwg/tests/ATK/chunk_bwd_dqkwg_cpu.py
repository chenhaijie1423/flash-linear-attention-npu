import torch
import torch_npu
import os
import torch
from typing import Tuple
from typing import Optional
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

    # 容器型 stage：内部还嵌套了子 stage，汇总时不应计入 sum_stages，
    # 否则会与子 stage 重复累加（表现为 sum_stages > total）。
    _CONTAINER_STAGES = {"main_loop", "batched_total"}

    def summary(self, total_time: float, chunks: int) -> str:
        if not self.totals:
            return ""
        rows = sorted(self.totals.items(), key=lambda x: -x[1])
        leaf_totals = {k: v for k, v in self.totals.items() if k not in self._CONTAINER_STAGES}
        sum_stages = sum(leaf_totals.values())
        overhead = max(total_time - sum_stages, 0.0)
        lines = []
        lines.append(f"[chunk_bwd_dqkwg_cpu] chunks={chunks} total={total_time*1000:.3f}ms "
                     f"sum_leaf_stages={sum_stages*1000:.3f}ms overhead={overhead*1000:.3f}ms "
                     f"({overhead/total_time*100:.1f}%)")
        lines.append(f"[chunk_bwd_dqkwg_cpu]   {'stage':<24}{'total_ms':>12}{'calls':>10}{'avg_us':>12}{'pct':>8}")
        for stage, tot in rows:
            cnt = self.counts[stage]
            avg_us = (tot / cnt) * 1e6 if cnt else 0.0
            pct = (tot / total_time * 100) if total_time > 0 else 0.0
            tag = " *" if stage in self._CONTAINER_STAGES else ""
            lines.append(f"[chunk_bwd_dqkwg_cpu]   {stage:<24}{tot*1000:>12.3f}{cnt:>10d}{avg_us:>12.2f}{pct:>7.1f}%{tag}")
        return "\n".join(lines)


class _NullTimer:
    def start(self, stage: str): pass
    def stop(self, stage: str): pass
    def summary(self, total_time: float, chunks: int) -> str: return ""


def pause():
    print("pause")
    input()

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
    dg = torch.zeros_like(g) if g is not None else None
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
            # if USE_DW: b_dw += dot(b_dv, b_h) (kernel 存 -b_dw)
            if dv is not None:
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
            if g is not None:
                timer.start("decay_scale")
                g_h = g[b_idx, s:e, :].permute(1, 0)                # [HV, L] (gtype)
                g_last = g[b_idx, min(s + chunk_size, t_end) - 1, :]  # [HV]

                exp_gc = torch.exp(g_h)                              # [HV, L]
                exp_neg_gc_glast = torch.exp(-g_h + g_last[:, None]) # [HV, L]

                dq_from_state = dq_from_state * exp_gc[:, :, None] * scale
                dk_from_state = dk_from_state * exp_neg_gc_glast[:, :, None]
                timer.stop("decay_scale")

                # b_dg += sum(b_dq * b_q) ; b_dg -= sum(b_k * b_dk)
                # 保留显式 product+sum（与原参考一致，避免 einsum 改变归约顺序）
                timer.start("dg_state")
                dg_c = (dq_from_state * q_h).sum(dim=-1)             # [HV, L]
                dg_c = cast_round(dg_c)
                dg_c = dg_c - (k_h * dk_from_state).sum(dim=-1)      # [HV, L]
                dg_c = cast_round(dg_c)

                # b_dg_last += sum(h * dh) * exp(g_last) + sum(dk * k)
                # 注意 h_prev/dh_curr 保留原始 datatype（与原实现一致）
                dg_last_accum = (h_prev * dh_curr).sum(dim=(-1, -2)) * torch.exp(g_last)  # [HV]
                dg_last_accum = dg_last_accum + (dk_from_state * k_h).sum(dim=(-1, -2))   # [HV]
                timer.stop("dg_state")

                # -----------------------------------------------------------
                # 3. Intra-chunk Attention
                # -----------------------------------------------------------
                timer.start("bmm_intra_ds")
                ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
                timer.stop("bmm_intra_ds")
                timer.start("decay_apply")
                decay_mat = torch.exp(g_h[:, :, None] - g_h[:, None, :])  # [HV, L, L]
                # 融合：避免 where(scalar) 类型提升与多次临时分配
                ds = ds * decay_mat
                ds = ds * mask_f                  # [1, L, L] 广播掩码
                ds = ds * scale
                timer.stop("decay_apply")

                # b_ds2 = b_ds * (q @ k.T)
                timer.start("bmm_qk")
                qk_t = cast_round(torch.bmm(q_h, k_h.transpose(1, 2)))   # [HV, L, L]
                timer.stop("bmm_qk")
                timer.start("dg_intra_accum")
                ds2 = ds * qk_t
                dg_c = dg_c + ds2.sum(dim=-1)
                dg_c = dg_c - ds2.sum(dim=-2)
                if datatype == gtype:
                    dg_c = dg_c.to(gtype)                               # 等价 .to(datatype).to(gtype)
                else:
                    dg_c = dg_c.to(datatype).to(gtype)                  # [HV, L]

                # 仅块最后一个有效 token 累加 dg_last
                dg_c[:, L - 1] = dg_c[:, L - 1] + dg_last_accum.to(gtype)
                dg[b_idx, s:e, :] = dg_c.permute(1, 0)                   # [L, HV]
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
            else:
                # No decay：保留与原实现一致的 scale 顺序
                timer.start("decay_scale")
                dk_from_state = dk_from_state * scale
                dq_from_state = dq_from_state * scale
                timer.stop("decay_scale")

                timer.start("bmm_intra_ds")
                ds = cast_round(torch.bmm(do_h, v_h.transpose(1, 2)))  # [HV, L, L]
                timer.stop("bmm_intra_ds")
                timer.start("decay_apply")
                ds = ds * mask_f
                timer.stop("decay_apply")

                timer.start("bmm_dqdk_intra")
                dq_intra = cast_round(torch.bmm(ds, k_h))               # [HV, L, K]
                dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_h))# [HV, L, K]
                dk_intra = dk_intra * scale
                dq_total = (dq_from_state + dq_intra) * scale
                dk_total = dk_from_state + dk_intra
                timer.stop("bmm_dqdk_intra")

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
        if dv is not None:
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

        if g is not None:
            # ---- 2. Decay scale ----
            timer.start("batched_decay_scale")
            # g: [N*C, HV] -> [N, HV, C] -> [N*HV, C]
            g_b = g[b_idx, s0:s0 + N * C, :].view(N, C, H).permute(0, 2, 1).reshape(N * H, C)  # gtype
            # 每个 chunk 的最后有效 token = chunk 内第 C-1 个位置（满 chunk 下成立）
            g_last = g_b.reshape(N, H, C)[:, :, C - 1].reshape(N * H)                          # [N*H]
            exp_gc = torch.exp(g_b)                                                             # [N*H, C]
            exp_neg_gc_glast = torch.exp(-g_b + g_last[:, None])                               # [N*H, C]
            dq_state = dq_state * exp_gc[:, :, None] * scale
            dk_state = dk_state * exp_neg_gc_glast[:, :, None]
            timer.stop("batched_decay_scale")

            # ---- dg_state（显式 product+sum，与原参考一致）----
            timer.start("batched_dg_state")
            dg_c = (dq_state * q_b).sum(dim=-1)                                              # [N*H, C]
            dg_c = cast_round(dg_c)
            dg_c = dg_c - (k_b * dk_state).sum(dim=-1)
            dg_c = cast_round(dg_c)
            # h_prev*dh_curr 保留原始 datatype（与原实现一致）
            dg_last_accum = (h_b * dh_b).sum(dim=(-1, -2)) * torch.exp(g_last)                # [N*H]
            dg_last_accum = dg_last_accum + (dk_state * k_b).sum(dim=(-1, -2))                # [N*H]
            timer.stop("batched_dg_state")

            # ---- 3. Intra-chunk attention ----
            timer.start("batched_bmm_intra_ds")
            ds = cast_round(torch.bmm(do_b, v_b.transpose(1, 2)))                              # [N*H, C, C]
            timer.stop("batched_bmm_intra_ds")
            timer.start("batched_decay_apply")
            decay_mat = torch.exp(g_b[:, :, None] - g_b[:, None, :])                           # [N*H, C, C]
            ds = ds * decay_mat
            ds = ds * mask_f
            ds = ds * scale
            timer.stop("batched_decay_apply")

            timer.start("batched_bmm_qk")
            qk_t = cast_round(torch.bmm(q_b, k_b.transpose(1, 2)))                             # [N*H, C, C]
            timer.stop("batched_bmm_qk")
            timer.start("batched_dg_intra_accum")
            ds2 = ds * qk_t
            dg_c = dg_c + ds2.sum(dim=-1)
            dg_c = dg_c - ds2.sum(dim=-2)
            if datatype == gtype:
                dg_c = dg_c.to(gtype)
            else:
                dg_c = dg_c.to(datatype).to(gtype)
            # 每个 chunk 第 C-1 个位置累加 dg_last
            dg_c = dg_c.reshape(N, H, C)
            dg_c[:, :, C - 1] = dg_c[:, :, C - 1] + dg_last_accum.to(gtype).reshape(N, H)
            # [N, H, C] -> [N, C, H] -> [N*C, H]
            dg[b_idx, s0:s0 + N * C, :] = dg_c.permute(0, 2, 1).reshape(N * C, H)
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
        else:
            timer.start("batched_decay_scale")
            dk_state = dk_state * scale
            dq_state = dq_state * scale
            timer.stop("batched_decay_scale")

            timer.start("batched_bmm_intra_ds")
            ds = cast_round(torch.bmm(do_b, v_b.transpose(1, 2)))                              # [N*H, C, C]
            timer.stop("batched_bmm_intra_ds")
            timer.start("batched_decay_apply")
            ds = ds * mask_f
            timer.stop("batched_decay_apply")

            timer.start("batched_bmm_dqdk_intra")
            dq_intra = cast_round(torch.bmm(ds, k_b))                                          # [N*H, C, K]
            dk_intra = cast_round(torch.bmm(ds.transpose(1, 2), q_b))                          # [N*H, C, K]
            dk_intra = dk_intra * scale
            dq_total = (dq_state + dq_intra) * scale
            dk_total = dk_state + dk_intra
            timer.stop("batched_bmm_dqdk_intra")

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
