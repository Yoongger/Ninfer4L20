# syntax=docker/dockerfile:1
# ============================================================
# ninfer-l20 : reproducible Ubuntu-based build image.
#
# Same base as the upstream project's own Dockerfile
# (nvidia/cuda on Ubuntu 24.04), so the container build and the
# bare-Ubuntu build (scripts/setup.sh + scripts/build.sh) are
# equivalent environments.
#
# Build:
#   docker build -f docker/Dockerfile -t ninfer-l20 .
#
# Run (the model artifact is mounted in; pick the profile with env):
#   docker run --gpus all -it --rm \
#     -p 8090:8090 \
#     -v "$PWD/models:/ninfer/models:ro" \
#     -e NINFER_MODEL=/ninfer/models/qwen3_8_27b.ninfer \
#     ninfer-l20 8090
# ============================================================

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
# Exactly the dependency set the upstream project declares
# (its Dockerfile) plus git/python3 which the port tooling needs.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        curl \
        ca-certificates \
        git \
        python3 \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ninfer
COPY . /ninfer

# Clone the upstream tree, apply the port, build. Everything is
# manifest-driven; no pinned SHAs.
RUN bash scripts/build.sh --test

# Runtime stage: CUDA runtime + the shared libraries the engine links.
FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        python3 \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libswscale7 \
        libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ninfer
COPY --from=build /ninfer/src/build /ninfer/build
COPY scripts/start.sh /ninfer/start.sh
COPY port /ninfer/port

ENTRYPOINT ["/bin/bash", "/ninfer/start.sh"]
CMD ["8090"]
