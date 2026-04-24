code_path="/data/huangjunzhe/GDN/flash-linear-attention-npu"
custom_path="/data/huangjunzhe/GDN/custom"
ascend_path="/data/huangjunzhe/Ascend.GDN/cann-9.0.0"
conda activate hjz39

source ${ascend_path}/set_env.sh

if [ "$1" = "compile" ]; then
    cd ${code_path}
    bash build.sh --pkg --ops=chunk_bwd_dqkwg
    bash build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=${custom_path}
fi

source ${custom_path}/vendors/custom_transformer/bin/set_env.bash
cp ${custom_path}/vendors/custom_transformer/op_api/lib/libcust_opapi.so ${ascend_path}/arm64-linux/lib64/libopapi_transformer.so
cd ${code_path}/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/ATK
atk node --backend PYACLNN --devices 4 node --backend CPU task -c all_aclnn_chunk_bwd_dqkwg.json --task accuracy   -p executor_chunk_bwd_dqkwg.py -wl [0]
