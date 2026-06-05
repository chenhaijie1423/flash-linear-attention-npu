FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops

ARG INSTALL_TEST_REQUIREMENTS=false
ARG PREWARM_THIRD_PARTY=true

ENV DEBIAN_FRONTEND=noninteractive \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      cmake \
      curl \
      g++ \
      gcc \
      git \
      git-lfs \
      make \
      ninja-build \
      openssh-client \
      patch \
      pkg-config \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-wheel \
      valgrind && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/fla-ci/requirements.txt
COPY tests/requirements.txt /tmp/fla-ci/tests-requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/fla-ci/requirements.txt && \
    python3 -m pip install triton triton-ascend && \
    python3 -m pip install --target=/tmp/pybind11-target pybind11 && \
    cp -a /tmp/pybind11-target/. /usr/local/lib/python3.10/dist-packages/ && \
    rm -rf /tmp/pybind11-target && \
    python3 -c "import pybind11; print(pybind11.__file__)" && \
    if [[ "${INSTALL_TEST_REQUIREMENTS}" == "true" ]]; then \
      grep -v -E '^[[:space:]]*(#|$|valgrind[[:space:]]*$)' /tmp/fla-ci/tests-requirements.txt \
        | python3 -m pip install -r /dev/stdin; \
    fi

RUN if [[ "${PREWARM_THIRD_PARTY}" == "true" ]]; then \
      mkdir -p /opt/fla-ci/third_party/pkg; \
      download() { \
        local url="$1"; \
        local dest="$2"; \
        if [[ -s "$dest" ]]; then return 0; fi; \
        echo "[CI][prewarm] downloading $url"; \
        if ! curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"; then \
          echo "[CI][prewarm][WARN] failed to download $url"; \
          rm -f "$dest"; \
        fi; \
      }; \
      clone_at() { \
        local url="$1"; \
        local rev="$2"; \
        local dest="$3"; \
        if [[ -d "$dest/.git" ]]; then return 0; fi; \
        echo "[CI][prewarm] cloning $url"; \
        if git clone "$url" "$dest" && git -C "$dest" checkout "$rev"; then \
          true; \
        else \
          echo "[CI][prewarm][WARN] failed to clone $url"; \
          rm -rf "$dest"; \
        fi; \
      }; \
      download "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz" \
        /opt/fla-ci/third_party/pkg/eigen-5.0.0.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz" \
        /opt/fla-ci/third_party/pkg/abseil-cpp-20230802.1.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz" \
        /opt/fla-ci/third_party/pkg/protobuf-25.1.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz" \
        /opt/fla-ci/third_party/pkg/makeself-release-2.5.0-patch1.tar.gz; \
      download "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip" \
        /opt/fla-ci/third_party/pkg/include.zip; \
      clone_at "https://gitcode.com/cann/opbase.git" \
        c8d83f3e57a63a7375e89a2d6937452c0ae2e522 \
        /opt/fla-ci/third_party/opbase; \
      clone_at "https://gitcode.com/cann/catlass.git" \
        41bf90da655bba3c66d0acd7e00abe33960ecfd6 \
        /opt/fla-ci/third_party/catlass; \
    fi

WORKDIR /workspace/repo
