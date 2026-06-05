# flash-linear-attention-npu

## 🔥Latest News

- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

## ⚡️快速上手

### ​CANN 开发环境部署

首先需安装 CANN 开发包，提供 NPU 算子运行所需的底层驱动与工具链。
推荐使用是社区版8.5.2，总共要下2个run包，这里以A3机器为例（即需要下载A3-ops、toolkit）
下载地址为
[https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2)
需要找到与你当前机器对应的包

```
#设置需要安装的路径
export INSTALL_PATH=/usr/local/Ascend

./Ascend-cann-toolkit*run --install-path=$INSTALL_PATH --full  --quiet
./Ascend-cann-A3*run --install-path=$INSTALL_PATH --install --quiet
source $INSTALL_PATH/ascend-toolkit/set_env.sh
```

### 编译自定义算子包

编译GDN算子run包并安装

```
# 编译命令，注意 --soc=${soc_version} 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu

# 安装 run 包（custom 包名：fla-npu-<vendor>_linux-<arch>.run）
./build_out/fla-npu-*.run
```

### ​torch_custom 框架编译构建

下载并安装对应python和torch版本的最新发行版[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)


编译torch适配whl包并安装
```sh
cd torch_custom/fla_npu
bash build.sh  # 一键编译安装脚本，先调用torchnpugen自动接入算子，再运行setup编whl包，最后安装whl包
```

### 测试单算子

```sh
# 运行测试
cd torch_custom/fla_npu/test
bash test.sh --device 0                 # 全量测试
bash test.sh --device 0 --op causal_conv1d  # 单算子测试
```


### 算子调用方式参考

使用torch.ops.npu.npu_{算子名称}()调用对应算子，具体可参考torch_custom/fla_npu/test下面的对应算子测试脚本

例如：

```python
import torch
import torch_npu
import fla_npu

torch.ops.npu.npu_chunk_bwd_dv_local(...)
```

### 接入实践

环境准备：除本仓根目录 `requirements.txt` 外，Triton 算子路径还依赖 `triton`、[triton-ascend](https://gitcode.com/Ascend/triton-ascend) 和 `pybind11`。可以使用以下命令安装：

```sh
pip install -r requirements.txt
pip install triton
pip install triton-ascend
pip install pybind11
```

一键运行GDN模块，组装了所有GDN相关算子，包括前向和反向，包括AscendC和Triton算子
```sh
python examples/flash_gated_delta_rule.py
```

## CI 环境

本仓提供基于 GitHub self-hosted runner 的 NPU CI 配置，当前规划部署在 `192.168.9.221` 的 `/workspace` 目录下，后续迁移时只需要迁移 runner 与 Docker 镜像/脚本。

为节省 NPU 资源，CI 不会在 PR 新建、重开或提交新 commit 时自动执行。PR 合入门禁会要求当前 head commit 存在 `NPU CI / manual` 成功状态；如果 PR 更新了 commit，旧 commit 上的 CI 成功状态不会继续生效，需要维护者重新手动触发。

### Docker CI 镜像

CI 镜像定义在 `ci/Dockerfile`，默认基于 CANN 8.5.0 910B Ubuntu 22.04 镜像构建，并预装工程基础依赖、`triton` 和 `triton-ascend`：

```sh
docker build -t fla-npu-ci:8.5.0-910b -f ci/Dockerfile .
```

如果需要额外安装 `tests/requirements.txt` 中的重型测试依赖，可以启用构建参数：

```sh
docker build --build-arg INSTALL_TEST_REQUIREMENTS=true -t fla-npu-ci:8.5.0-910b -f ci/Dockerfile .
```

### NPU 自动探测

`ci/detect_npu.sh` 会扫描宿主机 `npu-smi info`，优先选择健康且空闲的 NPU；如果没有完全空闲的健康卡，会退化选择空闲卡，并在日志中打印所选卡的健康状态、空闲状态和推断出的 `--soc`。

```sh
bash ci/detect_npu.sh --summary
bash ci/run_ci_container.sh
```

常用可配置环境变量如下：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CI_IMAGE` | `fla-npu-ci:8.5.0-910b` | CI Docker 镜像名 |
| `CI_MODE` | `quick` | `quick` 执行当前 SOC 编译；`full` 调用 `gdn-verify.sh` |
| `CI_OPS` | 空 | 指定逗号分隔的算子列表；为空时按脚本默认范围执行 |
| `CI_REBUILD_IMAGE` | `false` | 为 `true` 时运行前重新构建镜像 |
| `CI_REQUIRE_HEALTHY_NPU` | `false` | 为 `true` 时所选 NPU 非 `OK` 会直接失败 |
| `CI_BUILD_TORCH_CUSTOM` | `false` | 为 `true` 时额外编译 `torch_custom/fla_npu` |
| `CI_RUN_TORCH_TESTS` | `false` | 为 `true` 时额外执行 `torch_custom/fla_npu/test/test.sh` |
| `CI_RUN_EXAMPLE_ST` | `false` | 为 `true` 时额外执行 `examples/flash_gated_delta_rule.py` |
| `CI_CACHE_ROOT` | `/workspace/flash-linear-attention-npu-ci/cache` | self-hosted runner 上的持久 CI 缓存根目录 |
| `CI_THIRD_PARTY_CACHE` | `$CI_CACHE_ROOT/third_party` | 挂载到容器 `/workspace/repo/third_party` 的三方依赖缓存目录 |
| `CI_CPACK_JOBS` | `$CI_JOBS` | 传给 `CMAKE_BUILD_PARALLEL_LEVEL`/`MAKEFLAGS`，用于加速 cpack 内部 preinstall 构建 |
| `CI_FORCE_CLEAN_CACHE` | `false` | 为 `true` 时强制清理 `build/`、`build_out/`、`output/` 和三方依赖缓存 |
| `CI_SEED_THIRD_PARTY` | `true` | 为 `true` 时从 CI 镜像的 `/opt/fla-ci/third_party` 预烘种子填充三方缓存 |

CI 会对 `build.sh`、`CMakeLists.txt`、`cmake/`、`ci/`、`scripts/package/`、`scripts/util/`、`scripts/ci/`、依赖清单等构建输入计算签名。若新增三方引用、修改三方版本、调整 CMake/打包/CI 编译流程，签名变化后会自动删除 `build/`、`build_out/`、`output/` 并清空三方缓存，随后重新走完整编译。

### 触发 NPU CI

NPU CI 支持两种手动触发方式。

方式一：在 GitHub `Actions` 页面点击 `NPU CI`，再点击 `Run workflow`，填写：

- `pr_number`: 需要验证的 PR 编号
- `ci_mode`: `quick` 或 `full`
- `ops`: 可选，逗号分隔的算子列表

方式二：在 PR 评论区发送命令。只有维护账号可以触发：

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
```

如果当前 PR head commit 已经有成功的 `NPU CI / manual` 状态，重复点击按钮或重复评论不会再次启动 NPU CI。

### 注册 self-hosted runner

仓库管理员需要先在 GitHub 仓库设置里生成 self-hosted runner 注册 token，然后在目标服务器执行：

```sh
bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <registration-token>
```

runner 默认标签为 `linux,arm64,npu,flash-linear-attention-npu`，与 `.github/workflows/ci.yml` 中的 `runs-on` 保持一致。

## 🔍目录结构
关键目录如下：
```
├── cmake                              # 项目工程编译目录
├── common                             # 项目公共头文件和公共源码
├── fla                                # 算子库核心包
│   └── ops
│       ├── ascendc                    # AscendC 算子实现
│       │   ├── common                 # 公共模块（GroupedMatMul 等）
│       │   └── gdn                    # GDN 算子
│       │       ├── chunk_gdn_fwd      # 前向传播算子
│       │       │   ├── chunk_fwd_o
│       │       │   ├── chunk_gated_delta_rule_fwd_h
│       │       │   └── recompute_wu_fwd
│       │       ├── chunk_gdn_bwd      # 反向传播算子
│       │       │   ├── chunk_bwd_dqkwg
│       │       │   ├── chunk_bwd_dv_local
│       │       │   ├── chunk_gated_delta_rule_bwd_dhu
│       │       │   ├── prepare_wy_repr_bwd_da
│       │       │   └── prepare_wy_repr_bwd_full
│       │       ├── gdn_preprocess     # 预处理算子
│       │       │   └── causal_conv1d
│       │       └── recurrent_gdn      # 推理算子
│       │           └── recurrent_gated_delta_rule
│       └── triton                     # Triton 算子实现
├── torch_custom                       # 自定义PyTorch算子适配
├── examples                           # 端到端算子开发和调用示例
│   └── flash_gated_delta_rule.py      # 完整GDN接入调用示例
├── scripts                            # 脚本目录，包含算子构建相关配置文件
├── tests                              # 测试工程目录
├── gdn-verify.sh                      # GDN 一键验证脚本
├── CMakeLists.txt
├── README.md
├── build.sh                           # 项目工程编译脚本
├── install_deps.sh                    # 安装依赖包脚本
├── QUICKSTART.md                      # 快速入门文档
├── CONTRIBUTING.md                    # 贡献指南
├── SECURITY.md                        # 安全声明
├── LICENSE                            # 许可证
└── requirements.txt                   # 本项目需要的第三方依赖包
```

## 📝相关信息

- [安全声明](SECURITY.md)
- [许可证](LICENSE)

## 🙏致谢

本项目的部分实现参考了 [ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，感谢华为 CANN 社区及相关开发团队的开源贡献。
