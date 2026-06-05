#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    set -u
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if command -v npu-smi >/dev/null 2>&1; then
    if [[ -n "${NPU_SELECTED_DEVICE:-}" ]]; then
        if ! summary="$(bash ci/detect_npu.sh --summary 2>&1)"; then
            echo "[CI][WARN] Container npu-smi did not report devices; using host-selected NPU ${NPU_SELECTED_DEVICE}."
        else
            echo "$summary"
        fi
    else
        eval "$(bash ci/detect_npu.sh --env)"
        bash ci/detect_npu.sh --summary
    fi
fi

ci_mode="${CI_MODE:-quick}"
ci_soc="${CI_SOC:-${NPU_SOC:-ascend910b}}"
ci_ops="${CI_OPS:-}"
ci_jobs="${CI_JOBS:-$(nproc)}"
ci_cpack_jobs="${CI_CPACK_JOBS:-$ci_jobs}"

if [[ "$ci_soc" == "unknown" ]]; then
    ci_soc="ascend910b"
fi

export CMAKE_BUILD_PARALLEL_LEVEL="$ci_cpack_jobs"
export MAKEFLAGS="${MAKEFLAGS:+$MAKEFLAGS }-j${ci_cpack_jobs}"

bash ci/prepare_ci_cache.sh

ops_arg=()
if [[ -n "$ci_ops" ]]; then
    ops_arg=(--ops="$ci_ops")
fi

echo "[CI] mode=$ci_mode soc=$ci_soc ops=${ci_ops:-<all>} jobs=$ci_jobs cpack_jobs=$ci_cpack_jobs"

case "$ci_mode" in
    quick)
        bash build.sh --pkg --soc="$ci_soc" --vendor_name=fla_npu "${ops_arg[@]}" -j"$ci_jobs"
        ;;
    full)
        device="${NPU_SELECTED_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
        extra=()
        if [[ -n "$ci_ops" ]]; then
            extra=(--mode single --op "$ci_ops")
        fi
        bash gdn-verify.sh --device "$device" "${extra[@]}"
        ;;
    *)
        echo "[CI][ERROR] Unsupported CI_MODE: $ci_mode" >&2
        exit 2
        ;;
esac

if [[ "${CI_BUILD_TORCH_CUSTOM:-false}" == "true" ]]; then
    (cd torch_custom/fla_npu && bash build.sh)
fi

if [[ "${CI_RUN_TORCH_TESTS:-false}" == "true" ]]; then
    test_device="${NPU_SELECTED_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
    test_args=(--device "$test_device")
    if [[ -n "${CI_TEST_OP:-}" ]]; then
        test_args+=(--op "$CI_TEST_OP")
    fi
    (cd torch_custom/fla_npu/test && bash test.sh "${test_args[@]}")
fi

if [[ "${CI_RUN_EXAMPLE_ST:-false}" == "true" ]]; then
    python3 examples/flash_gated_delta_rule.py
fi
