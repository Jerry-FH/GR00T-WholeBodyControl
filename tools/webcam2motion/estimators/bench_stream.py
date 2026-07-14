"""M2 bench: drive GVHMRStreamEstimator with video frames as a fake camera.

Measures per-frame latency/fps and compares the causal sliding-window output
against the offline full-video result (M1 hmr4d_results.pt) on body_pose.

Run inside the container (CWD=/opt/GVHMR, GPU):
  python /workspace/gr00t-wbc/tools/webcam2motion/estimators/bench_stream.py \
      --video docs/example_video/tennis.mp4 \
      --offline /workspace/gr00t-wbc/tools/webcam2motion/outputs/tennis/hmr4d_results.pt \
      --frames 200
"""

import argparse
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/gr00t-wbc/tools/webcam2motion")
from estimators.gvhmr_runner import GVHMRStreamEstimator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--offline", default=None, help="hmr4d_results.pt for accuracy diff")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--yolo-period", type=int, default=1)
    ap.add_argument("--no-flip", action="store_true", help="disable ViTPose flip test")
    ap.add_argument("--no-postproc", action="store_true", help="skip pp transl + IK refine")
    ap.add_argument("--autocast", action="store_true", help="fp16 autocast on the head")
    ap.add_argument("--fp16", action="store_true", help="fp16 ViT front-end (kp2d+feat+yolo)")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()

    est = GVHMRStreamEstimator(window=args.window, yolo_period=args.yolo_period,
                               flip_test=not args.no_flip,
                               postproc=not args.no_postproc, autocast=args.autocast,
                               fp16=args.fp16, verbose_timing=args.timing)
    cap = cv2.VideoCapture(args.video)

    results, times = [], []
    i = 0
    while i < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.monotonic()
        out = est.estimate(frame, t0)
        times.append(time.monotonic() - t0)
        results.append(out)
        i += 1
    cap.release()

    times = np.array(times)
    n_valid = sum(r is not None for r in results)
    print(f"\n[bench] frames={len(times)} valid={n_valid} "
          f"(first valid at #{next((k for k, r in enumerate(results) if r), -1)})")
    # steady-state = after warmup (first 20 processed frames)
    ss = times[20:] if len(times) > 40 else times
    print(f"[bench] latency p50={1e3*np.percentile(ss,50):.0f}ms "
          f"p95={1e3*np.percentile(ss,95):.0f}ms -> {1/np.percentile(ss,50):.1f} fps")

    # subject lock-on check: bbox center should move continuously (a bystander
    # steal shows up as a large single-frame jump)
    centers = np.array([r["bbx_xys"][:2] for r in results if r is not None])
    if len(centers) > 2:
        jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        print(f"[bench] bbox center jump: mean={jumps.mean():.1f}px "
              f"max={jumps.max():.1f}px (>100px = subject switch)")

    if args.offline:
        off = torch.load(args.offline, map_location="cpu", weights_only=False)
        off_bp = off["smpl_params_global"]["body_pose"].numpy().reshape(-1, 63)
        diffs = [np.abs(r["body_pose"] - off_bp[k]).mean()
                 for k, r in enumerate(results) if r is not None and k < len(off_bp)]
        # skip warmup frames in the accuracy stat
        d = np.array(diffs[10:]) if len(diffs) > 20 else np.array(diffs)
        print(f"[bench] causal-vs-offline body_pose MAE={np.degrees(d.mean()):.2f} deg "
              f"p95={np.degrees(np.percentile(d,95)):.2f} deg over {len(d)} frames")


if __name__ == "__main__":
    main()
