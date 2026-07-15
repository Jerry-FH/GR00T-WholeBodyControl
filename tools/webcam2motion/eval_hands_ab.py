"""A/B the hand backends (mediapipe vs wilor) offline on a video.

Runs the SAME GVHMR streaming estimator twice over the same frames and
compares the hand chain only: per-side detection rate, hand-stage latency,
and wrist-roll jitter (frame-to-frame delta while detected — the value that
actually drives the G1 wrist).

Run inside the webcam2motion container:
  cd /opt/GVHMR && python -u \
    /workspace/gr00t-wbc/tools/webcam2motion/eval_hands_ab.py \
    --video docs/example_video/tennis.mp4 --frames 300
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_backend(backend: str, frames: list) -> dict:
    from estimators.gvhmr_runner import GVHMRStreamEstimator
    est = GVHMRStreamEstimator(device="cuda", window=32, flip_test=False,
                               postproc=False, fp16=True, hands=True,
                               hand_backend=backend, yolo_period=2)
    # time the in-pipeline track() call itself — do NOT call it a second
    # time (mediapipe VIDEO mode keeps temporal state; extra calls skew it)
    hand_ms = []
    inner_track = est.hand_tracker.track

    def timed_track(*a, **kw):
        t0 = time.perf_counter()
        out = inner_track(*a, **kw)
        hand_ms.append(1e3 * (time.perf_counter() - t0))
        return out

    est.hand_tracker.track = timed_track

    hits = {"left": 0, "right": 0}
    rolls = {"left": [], "right": []}
    total = 0
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        r = est.estimate(frame, i / 30.0)
        if r is None:
            continue
        total += 1
        hands = r.get("hands") or {}
        for side in ("left", "right"):
            if side in hands:
                hits[side] += 1
                rolls[side].append(float(hands[side]["wrist_angles"][0]))
    wall = time.perf_counter() - t0

    def jitter(v):
        v = np.asarray(v)
        return float(np.abs(np.diff(v)).mean()) if len(v) > 3 else float("nan")

    return {
        "backend": backend,
        "est_frames": total,
        "det_rate": {s: hits[s] / max(total, 1) for s in ("left", "right")},
        "hand_ms": float(np.median(hand_ms)) if hand_ms else float("nan"),
        "roll_jitter_rad": {s: jitter(rolls[s]) for s in ("left", "right")},
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="docs/example_video/tennis.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--backends", nargs="+", default=["mediapipe", "wilor"])
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"[ab] {len(frames)} frames from {args.video}")

    results = [run_backend(b, frames) for b in args.backends]
    print(f"\n{'backend':10s} {'det L':>7s} {'det R':>7s} {'hand ms':>8s} "
          f"{'jit L':>7s} {'jit R':>7s} {'wall s':>7s}")
    for r in results:
        print(f"{r['backend']:10s} {r['det_rate']['left']:7.1%} "
              f"{r['det_rate']['right']:7.1%} {r['hand_ms']:8.1f} "
              f"{r['roll_jitter_rad']['left']:7.3f} "
              f"{r['roll_jitter_rad']['right']:7.3f} {r['wall_s']:7.1f}")


if __name__ == "__main__":
    main()
