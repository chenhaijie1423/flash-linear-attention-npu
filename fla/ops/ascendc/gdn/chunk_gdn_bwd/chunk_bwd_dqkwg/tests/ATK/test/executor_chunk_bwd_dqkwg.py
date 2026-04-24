# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import torch
import numpy as np
import math
import sys
import os
from typing import Optional, Tuple

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chunk_bwd_dqkwg_cpu import chunk_bwd_dqkwg_cpu


def generate_tensor(shape, data_type, data_max):
    tensor = torch.rand(shape) * (data_max * 2) - data_max
    return tensor.to(data_type)


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def cdiv(a: torch.LongTensor, b: int):
    return (a + b - 1) // b


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


def cumsum_cu_seqlens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.nn.functional.pad(
        torch.cumsum(cu_seqlens, dim=0),
        (1, 0),
        value=0
    )


def chunk_bwd_dqkwg_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    dv: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()
    dv_t = dv.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous()
    w_t = w.transpose(1, 2).contiguous() if w is not None else None
    h_t = h.transpose(1, 2).contiguous()
    dh_t = dh.transpose(1, 2).contiguous()

    cu_seqlens_t = cu_seqlens if cu_seqlens is not None else None

    dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
        q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t, scale, torch.tensor(cu_seqlens_t), chunk_size
    )

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dw = dw.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    return dq, dk, dw, dg


@register("executor_chunk_bwd_dqkwg")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        do = input_data.kwargs["do"]
        h = input_data.kwargs["h"]
        dh = input_data.kwargs["dh"]
        w = input_data.kwargs["w"]
        g = input_data.kwargs["g"]
        dv = input_data.kwargs["dv"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]
        dq, dk, dw_out, dg = chunk_bwd_dqkwg_torch(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size
        )
        # print("in __call__", device, q.dtype, k.dtype, v.dtype, do.dtype, h.dtype, dh.dtype, g.dtype, dv.dtype, scale, cu_seqlens, chunk_indices, chunk_size)
        # print("dq = ", dq, "dk = ", dk, "dw_out = ", dw_out, "dg = ", dg)
        if self.qkv_type == "bf16":
            dq = dq.to(torch.bfloat16)
            dk = dk.to(torch.bfloat16)
            dw_out = dw_out.to(torch.bfloat16) if dw_out is not None else None
        if self.qkv_type == "fp16":
            dq = dq.to(torch.float16)
            dk = dk.to(torch.float16)
            dw_out = dw_out.to(torch.float16) if dw_out is not None else None

        is_mix = input_data.kwargs.get("is_mix", True)
        if not is_mix:
            if self.qkv_type == "bf16":
                dg = dg.to(torch.bfloat16)
            if self.qkv_type == "fp16":
                dg = dg.to(torch.float16)

        return dq, dk, dw_out, dg

    def init_by_input_data(self, input_data: InputDataset):
        B, H, T, K = input_data.kwargs["q"].shape
        V = input_data.kwargs["v"].shape[3]
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        do = input_data.kwargs["do"]
        h = input_data.kwargs["h"]
        dh = input_data.kwargs["dh"]
        w = input_data.kwargs["w"]
        g = input_data.kwargs["g"]
        dv = input_data.kwargs["dv"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        is_fix = input_data.kwargs["is_fix"]
        self.qkv_type = input_data.kwargs["qkv_type"]

        is_fix = False
        # print("is_fix = ", is_fix)
        # print("chunk_size = ", chunk_size)

        qkv_type = input_data.kwargs["q"].dtype
        # print("qkv_type = ", qkv_type)
        g_type = input_data.kwargs["g"].dtype
        is_mix = input_data.kwargs["is_mix"]
        if not is_mix:
            g_type = qkv_type

        if not is_fix:
            cu_seqlens = cumsum_cu_seqlens(cu_seqlens).tolist()
            T = cu_seqlens[-1]
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            num_chunks = len(chunk_indices) // 2
            q = generate_tensor((B, H, T, K), torch.bfloat16, 5)
            k = generate_tensor((B, H, T, K), torch.bfloat16, 5)
            v = generate_tensor((B, H, T, V), torch.bfloat16, 5)
            do = generate_tensor((B, H, T, V), torch.bfloat16, 5)
            dv = generate_tensor((B, H, T, V), torch.bfloat16, 5)
            w = generate_tensor((B, H, T, K), torch.bfloat16, 5)
            g = -torch.sort(torch.rand(B * H * T) * 10, descending=False)[0].reshape((B, H, T)).to(torch.bfloat16)
            h = generate_tensor((B, H, num_chunks, K, V), torch.bfloat16, 5)
            dh = generate_tensor((B, H, num_chunks, K, V), torch.bfloat16, 5)
        else:
            cu_seqlens = None
            chunk_indices = None
            num_chunks = (T + chunk_size - 1) // chunk_size

        # print("cu_seqlens = ", cu_seqlens)
        # print("chunk_indices = ", chunk_indices)

        q = q.to(qkv_type)
        k = k.to(qkv_type)
        v = v.to(qkv_type)
        do = do.to(qkv_type)
        dv = dv.to(qkv_type)
        w = w.to(qkv_type)
        h = h.to(qkv_type)
        dh = dh.to(qkv_type)
        g = g.to(g_type)

        g_gamma = torch.zeros(1, 1, dtype=torch.float32)

        if self.device == "pyaclnn":
            q = q.npu()
            k = k.npu()
            v = v.npu()
            do = do.npu()
            dv = dv.npu()
            w = w.npu()
            g = g.npu()
            h = h.npu()
            dh = dh.npu()
            # g_gamma = g_gamma.npu()
            # if cu_seqlens is not None:
            #     cu_seqlens = cu_seqlens.npu()
            # if chunk_indices is not None:
            #     chunk_indices = chunk_indices.npu()
            print("q", q.shape, q.dtype)
            print("k", k.shape, k.dtype)
            print("v", v.shape, v.dtype)
            print("do", do.shape, do.dtype)
            print("dv", dv.shape, dv.dtype)
            # print("w", w.shape, w.dtype)
            print("g", g.shape, g.dtype)
            print("h", h.shape, h.dtype)
            print("dh", dh.shape, dh.dtype)
            print("cu_seqlens", cu_seqlens)
            print("chunk_indices", chunk_indices)



        input_data.kwargs['q'] = q
        input_data.kwargs['k'] = k
        input_data.kwargs['v'] = v
        input_data.kwargs['do'] = do
        input_data.kwargs['h'] = h
        input_data.kwargs['dh'] = dh
        input_data.kwargs['w'] = None
        input_data.kwargs['g'] = g
        input_data.kwargs['dv'] = dv
        input_data.kwargs['cu_seqlens'] = cu_seqlens
        input_data.kwargs['chunk_indices'] = chunk_indices
        input_data.kwargs['scale'] = scale
        input_data.kwargs['chunk_size'] = chunk_size
        input_data.kwargs['use_exp2'] = False
        input_data.kwargs['transpose_state_layout'] = False
        input_data.kwargs["g_gamma"] = None
        input_data.kwargs.pop("is_mix", None)
        input_data.kwargs.pop("is_fix", None)
        input_data.kwargs.pop("qkv_type", None)
