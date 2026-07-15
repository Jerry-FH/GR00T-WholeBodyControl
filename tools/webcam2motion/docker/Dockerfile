# webcam2motion: webcam/video -> SMPL -> ZMQ pose stream for GEAR-SONIC
# Blackwell (sm_120) needs CUDA >= 12.8 and torch cu128+ wheels.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv-managed python 3.10 venv (repo convention)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV VENV=/opt/venv
RUN uv venv --python 3.10 $VENV
ENV PATH="$VENV/bin:$PATH" VIRTUAL_ENV=$VENV

# torch cu128 (Blackwell-capable); long timeout — the cu128 wheels are >3 GB
ENV UV_HTTP_TIMEOUT=1200
RUN uv pip install --python $VENV/bin/python \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# GVHMR (pinned) — DPVO/renderer extras deliberately skipped (static camera,
# headless inference only; avoids custom CUDA builds on sm_120)
ARG GVHMR_REF=main
RUN git clone --depth 1 --branch $GVHMR_REF https://github.com/zju3dv/GVHMR /opt/GVHMR
RUN uv pip install --python $VENV/bin/python \
    numpy==1.26.4 scipy opencv-python-headless pyzmq smplx joblib \
    hydra-core omegaconf einops timm ultralytics onnxruntime-gpu \
    imageio[ffmpeg] tqdm pytorch-lightning
# GVHMR as an importable package (setup.py present in repo)
RUN cd /opt/GVHMR && uv pip install --python $VENV/bin/python -e . --no-deps || true

# pytorch3d CPU-only (GVHMR imports it at module level; demo inference path
# never calls CUDA kernels — knn is unused). FORCE_CUDA=0 avoids sm_120 builds.
# MAX_JOBS kept low: 20 parallel g++ jobs OOM-killed the host desktop session.
RUN uv pip install --python $VENV/bin/python ninja && \
    CUDA_HOME= FORCE_CUDA=0 MAX_JOBS=4 \
    uv pip install --python $VENV/bin/python --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# remaining GVHMR runtime deps (import-tested against tools/demo/demo.py)
RUN uv pip install --python $VENV/bin/python \
    colorlog hydra_colorlog hydra_zen rich termcolor ffmpeg-python scikit-image \
    "av==13.0.0" trimesh cython_bbox lapx matplotlib tensorboardX wis3d pycolmap yacs

# hand landmarks for palm-orientation (wrist roll) + future Dex3 finger mapping
# (mediapipe >=0.10.2x dropped the legacy solutions API -> Tasks API + .task model;
#  needs GLES/EGL system libs even for CPU inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2 libegl1 && rm -rf /var/lib/apt/lists/*
RUN uv pip install --python $VENV/bin/python mediapipe && \
    mkdir -p /opt/models && curl -fsSL -o /opt/models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# WiLoR GPU hand backend (--hand-backend wilor): MANO regression via wilor-mini.
# Weights land in /opt/GVHMR/inputs/checkpoints/wilor (host bind mount) on first
# run; the official MANO_RIGHT.pkl is pre-placed there from the user's
# registered MANO download, so no mirrored copy is fetched.
# --no-deps is ESSENTIAL: wilor-mini's pins drag torch down to 2.5.0+cu124,
# which has no sm_120 (Blackwell) kernels and bricks CUDA for everything.
# Its real runtime deps are listed explicitly (chumpy builds need pip).
RUN uv pip install --python $VENV/bin/python pip && \
    uv pip install --python $VENV/bin/python --no-build-isolation chumpy && \
    uv pip install --python $VENV/bin/python einops timm && \
    uv pip install --python $VENV/bin/python --no-deps \
    "git+https://github.com/warmshao/WiLoR-mini"

# streaming code + gear_sonic utils come from the bind-mounted repo at runtime
ENV PYTHONPATH=/workspace/gr00t-wbc:/opt/GVHMR
WORKDIR /workspace/gr00t-wbc/tools/webcam2motion
CMD ["bash"]
