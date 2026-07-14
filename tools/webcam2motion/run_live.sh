#!/bin/bash
# One-liner: start the live webcam->SMPL->ZMQ streamer container (preview on).
# Prereqs: MuJoCo sim (run_sim_loop.py) + deploy (--input-type zmq) already up.
# Usage: ./run_live.sh [--camera N | --video PATH_IN_CONTAINER] [extra streamer args...]
#   extra args e.g.: --latency-mode predict | hold | smooth (default)
# Preview window (host): .venv_sim/bin/python tools/webcam2motion/preview_viewer.py
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"

SRC_ARGS=("--camera" "0")
DEV_ARGS=(--device /dev/video0)
if [[ "$1" == "--video" ]]; then
    SRC_ARGS=("--video" "$2"); DEV_ARGS=(); shift 2
elif [[ "$1" == "--camera" ]]; then
    SRC_ARGS=("--camera" "$2"); DEV_ARGS=(--device "/dev/video$2"); shift 2
fi

echo ">>> 預覽視窗（另開 host 終端）: .venv_sim/bin/python tools/webcam2motion/preview_viewer.py"
docker rm -f w2m-stream >/dev/null 2>&1 || true
exec docker run -it --rm --name w2m-stream \
    --network host --ipc host --gpus all "${DEV_ARGS[@]}" \
    -v "$REPO_ROOT:/workspace/gr00t-wbc:rw" \
    -v "$REPO_ROOT/tools/webcam2motion/checkpoints:/opt/GVHMR/inputs/checkpoints:ro" \
    -w /opt/GVHMR webcam2motion \
    python -u /workspace/gr00t-wbc/tools/webcam2motion/stream_webcam_zmq.py \
        "${SRC_ARGS[@]}" --preview "$@"
