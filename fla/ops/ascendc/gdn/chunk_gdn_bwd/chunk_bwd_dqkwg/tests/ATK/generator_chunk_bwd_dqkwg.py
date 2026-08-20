# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


Q_INDEX = 0
K_INDEX = 1
V_INDEX = 2
G_INDEX = 3
H_INDEX = 4
DO_INDEX = 5
DH_INDEX = 6
DV_INDEX = 7
W_INDEX = 10
G_GAMMA_INDEX = 11
CHUNK_SIZE_INDEX = 13
IS_MIX_INDEX = 14
IS_FIX_INDEX = 15
USE_EXP2_INDEX = 16
TRANSPOSE_STATE_LAYOUT_INDEX = 17
QKV_TYPE_INDEX = 18


@GENERATOR_REGISTRY.register("generator_chunk_bwd_dqkwg")
class ChunkBwdDqkwgGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        qkv_type = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[K_INDEX].dtype = qkv_type
        case_config.inputs[V_INDEX].dtype = qkv_type
        case_config.inputs[DO_INDEX].dtype = qkv_type
        case_config.inputs[DH_INDEX].dtype = qkv_type
        case_config.inputs[DV_INDEX].dtype = qkv_type
        case_config.inputs[W_INDEX].dtype = qkv_type
        case_config.inputs[H_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_type

        is_mix = case_config.inputs[IS_MIX_INDEX].range_values
        if not is_mix:
            case_config.inputs[G_INDEX].dtype = qkv_type

        is_fix = case_config.inputs[IS_FIX_INDEX].range_values
        H = random.randint(1, 32)
        if not is_fix:
            B = 1
            T = random.randint(1, 409600)
        else:
            B = random.randint(1, 100)
            T = random.randint(1, 4096)

        K = 128
        V = random.choice((128, 256))
        chunk_size = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        if isinstance(chunk_size, list):
            chunk_size = chunk_size[0]

        # HV = n * HK, pick n from {1, 2, 3, 4, 5, 6, 7, 8} for dimension split testing
        gva_mode = random.choice((0, 1))
        n_ratio = 1 if gva_mode == 0 else random.choice((2, 3, 4, 5, 6, 7, 8))
        HK = H
        HV = HK * n_ratio

        num_chunks = (T + chunk_size - 1) // chunk_size

        # q, k: [B, HK, T, K]; v, dox, dv: [B, HV, T, V]
        # g: [B, HV, T]; h, dh: [B, HV, num_chunks, K, V]
        # outputs: dq, dk, dw: [B, HV, T, K]; dg: [B, HV, T]
        case_config.inputs[Q_INDEX].shape = [B, HK, T, K]
        case_config.inputs[K_INDEX].shape = [B, HK, T, K]
        case_config.inputs[V_INDEX].shape = [B, HV, T, V]
        case_config.inputs[G_INDEX].shape = [B, HV, T]
        case_config.inputs[H_INDEX].shape = [B, HV, num_chunks, K, V]
        case_config.inputs[DO_INDEX].shape = [B, HV, T, V]
        case_config.inputs[DH_INDEX].shape = [B, HV, num_chunks, K, V]
        case_config.inputs[DV_INDEX].shape = [B, HV, T, V]
        case_config.inputs[W_INDEX].shape = [B, HV, T, K]

        return case_config

@GENERATOR_REGISTRY.register("generator_chunk_bwd_dqkwg_network")
class ChunkBwdDqkwgNetworkGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        qkv_type = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[K_INDEX].dtype = qkv_type
        case_config.inputs[V_INDEX].dtype = qkv_type
        case_config.inputs[DO_INDEX].dtype = qkv_type
        case_config.inputs[DH_INDEX].dtype = qkv_type
        case_config.inputs[DV_INDEX].dtype = qkv_type
        case_config.inputs[W_INDEX].dtype = qkv_type
        case_config.inputs[H_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_type

        is_mix = case_config.inputs[IS_MIX_INDEX].range_values
        if not is_mix:
            case_config.inputs[G_INDEX].dtype = qkv_type

        # 网络用例定长场景
        case_config.inputs[IS_FIX_INDEX].range_values = True
        is_fix = True
        H = random.choice((4, 8))
        B = random.choice((1, 8))
        T = 32768 if B == 1 else 4096

        K = 128
        V = random.choice((128, 256))
        chunk_size = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        if isinstance(chunk_size, list):
            chunk_size = chunk_size[0]

        # HV = n * HK, pick n from {1, 2, 3} for dimension split testing
        gva_mode = random.choice((0, 1))
        n_ratio = 1 if gva_mode == 0 else random.choice((2, 3))
        HK = H
        HV = HK * n_ratio

        num_chunks = (T + chunk_size - 1) // chunk_size

        # q, k: [B, HK, T, K]; v, dox, dv: [B, HV, T, V]
        # g: [B, HV, T]; h, dh: [B, HV, num_chunks, K, V]
        # outputs: dq, dk, dw: [B, HV, T, K]; dg: [B, HV, T]
        case_config.inputs[Q_INDEX].shape = [B, HK, T, K]
        case_config.inputs[K_INDEX].shape = [B, HK, T, K]
        case_config.inputs[V_INDEX].shape = [B, HV, T, V]
        case_config.inputs[G_INDEX].shape = [B, HV, T]
        case_config.inputs[H_INDEX].shape = [B, HV, num_chunks, K, V]
        case_config.inputs[DO_INDEX].shape = [B, HV, T, V]
        case_config.inputs[DH_INDEX].shape = [B, HV, num_chunks, K, V]
        case_config.inputs[DV_INDEX].shape = [B, HV, T, V]
        case_config.inputs[W_INDEX].shape = [B, HV, T, K]

        return case_config