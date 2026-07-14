#!/bin/bash
# Launch the webcam2motion container (publisher peer of g1-deploy-dev).
# Usage: ./run.sh [--build] [CMD...]
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../../.. && pwd)"
IMAGE=webcam2motion
NAME=webcam2motion

if [[ "$1" == "--build" ]]; then
    shift
    docker build -t $IMAGE .
fi

DEVICES=""
for v in /dev/video*; do
    [ -e "$v" ] && DEVICES="$DEVICES --device $v"
done

SMPL_MOUNT=""
if [ -d "$HOME/smpl_models" ]; then
    SMPL_MOUNT="-v $HOME/smpl_models:/models/smpl:ro"
fi

# checkpoints persisted on host so image rebuilds don't re-download
mkdir -p "$REPO_ROOT/tools/webcam2motion/checkpoints"

docker run -it --rm --name $NAME \
    --network host --ipc host --gpus all \
    $DEVICES \
    -v "$REPO_ROOT:/workspace/gr00t-wbc:rw" \
    -v "$REPO_ROOT/tools/webcam2motion/checkpoints:/opt/GVHMR/inputs/checkpoints:rw" \
    $SMPL_MOUNT \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    $IMAGE "${@:-bash}"
