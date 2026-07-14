"""M1: convert GVHMR demo output (hmr4d_results.pt) to a replayable pkl.

Inside the webcam2motion container:
  cd /opt/GVHMR && python tools/demo/demo.py --video /path/video.mp4 -s
  python /workspace/gr00t-wbc/tools/webcam2motion/estimators/gvhmr_offline.py \
      outputs/demo/<name>/hmr4d_results.pt /tmp/my_motion.pkl

Then on the host:
  .venv_sim/bin/python tools/webcam2motion/replay_smpl_zmq.py \
      --pkl /tmp/my_motion.pkl --src-fps 30

Uses smpl_params_global (world y-up, gravity-aligned) — NOT incam — so the
y-up->z-up conversion in smpl_adapter holds. betas are ignored downstream.
"""

import sys

import joblib
import numpy as np
import torch


def main(results_pt: str, out_pkl: str):
    results = torch.load(results_pt, map_location="cpu", weights_only=False)
    params = results["smpl_params_global"]
    body_pose = params["body_pose"].numpy().reshape(-1, 63).astype(np.float32)
    global_orient = params["global_orient"].numpy().reshape(-1, 3).astype(np.float32)
    T = body_pose.shape[0]
    out = {
        "pose_aa": np.concatenate(
            [global_orient, body_pose, np.zeros((T, 6), np.float32)], axis=1),  # (T,72)
        "transl": params.get("transl", torch.zeros(T, 3)).numpy().astype(np.float32),
        "fps": 30.0,  # GVHMR demo assumes 30 fps input; pass --src-fps 30 to the replayer
    }
    joblib.dump(out, out_pkl)
    print(f"saved {T} frames -> {out_pkl}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
