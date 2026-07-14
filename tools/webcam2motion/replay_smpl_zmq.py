"""Replay SMPL motion over ZMQ into the SONIC deploy binary (M0 validation).

Modes:
  --synthetic          sinusoidal arm-raise generated in SMPL space (no data needed)
  --pkl PATH [--motion NAME]   SMPL pkl (e.g. sample_data/smpl_filtered/)

Usage (from repo root, .venv_sim):
  python tools/webcam2motion/replay_smpl_zmq.py --synthetic
  python tools/webcam2motion/replay_smpl_zmq.py --pkl data/sample_data/smpl_filtered/xxx.pkl

Deploy side: bash deploy.sh --input-type zmq ... then ']' + drop + ENTER.
"""

import argparse
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publisher import SlidingWindowPosePublisher, run_paced  # noqa: E402
from smpl_adapter import smpl_to_stream_frames  # noqa: E402

FPS = 50

# body_pose 21-joint indices (SMPL joints 1..21): shoulders and elbows
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 15, 16, 17, 18


def synthetic_arm_raise(seconds: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Gentle alternating arm raise + elbow bend, 50 Hz, y-up SMPL world."""
    T = int(seconds * FPS)
    t = np.arange(T) / FPS
    body_pose = np.zeros((T, 21, 3), dtype=np.float32)

    # 0.25 Hz sinusoid, ramped in over the first 2 s
    ramp = np.clip(t / 2.0, 0, 1)
    a = 0.6 * ramp * np.sin(2 * np.pi * 0.25 * t)  # rad

    # shoulder rotation about z (raise/lower from T-pose), mirrored L/R
    body_pose[:, L_SHOULDER, 2] = -np.abs(a)
    body_pose[:, R_SHOULDER, 2] = np.abs(a)
    # elbow flexion about y, alternating with the shoulders
    body_pose[:, L_ELBOW, 1] = -0.5 * ramp * (1 + np.sin(2 * np.pi * 0.25 * t + np.pi / 2)) / 2
    body_pose[:, R_ELBOW, 1] = 0.5 * ramp * (1 + np.sin(2 * np.pi * 0.25 * t + np.pi / 2)) / 2

    global_orient = np.zeros((T, 3), dtype=np.float32)
    return body_pose.reshape(T, 63), global_orient


def synthetic_wrist_heading(seconds: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Wrist roll/pitch oscillation (0.3 Hz) + slow body-heading yaw (0.05 Hz,
    ±0.5 rad) — verifies the v3 wrist joints and body_quat facing are tracked.
    Arms held slightly out so the wrists move freely."""
    L_WRIST, R_WRIST = 19, 20  # body_pose 21-joint indices (SMPL joints 20/21)
    T = int(seconds * FPS)
    t = np.arange(T) / FPS
    ramp = np.clip(t / 2.0, 0, 1)
    body_pose = np.zeros((T, 21, 3), dtype=np.float32)

    # hold arms a bit forward/down so wrist motion is unobstructed
    body_pose[:, L_SHOULDER, 2] = -0.5 * ramp
    body_pose[:, R_SHOULDER, 2] = 0.5 * ramp
    body_pose[:, L_ELBOW, 1] = -0.6 * ramp
    body_pose[:, R_ELBOW, 1] = 0.6 * ramp

    w = 0.5 * ramp * np.sin(2 * np.pi * 0.3 * t)
    body_pose[:, L_WRIST, 0] = w          # -> G1 wrist roll
    body_pose[:, R_WRIST, 0] = -w
    body_pose[:, L_WRIST, 1] = 0.4 * ramp * np.sin(2 * np.pi * 0.3 * t + np.pi / 2)
    body_pose[:, R_WRIST, 1] = body_pose[:, L_WRIST, 1]  # -> G1 wrist pitch (sign-mapped)

    global_orient = np.zeros((T, 3), dtype=np.float32)
    global_orient[:, 1] = 0.5 * ramp * np.sin(2 * np.pi * 0.05 * t)  # yaw about y (y-up)
    return body_pose.reshape(T, 63), global_orient


def load_pkl(path: str, motion: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Load one motion from an SMPL pkl; handles (T,72) / (T,66) smpl_pose or
    separate global_orient+body_pose keys."""
    data = joblib.load(path)
    if isinstance(data, dict) and motion is None and all(isinstance(v, dict) for v in data.values()):
        motion = next(iter(data))
        print(f"[replay] motions in pkl: {list(data.keys())[:10]}{'...' if len(data) > 10 else ''}")
    entry = data[motion] if motion is not None and isinstance(data, dict) else data
    print(f"[replay] using motion: {motion}, keys: {list(entry.keys())}")

    if "pose_aa" in entry:  # sample_data/smpl_filtered schema: global(3)+body(69)
        pose = np.asarray(entry["pose_aa"], dtype=np.float32).reshape(len(entry["pose_aa"]), -1)
        return pose[:, 3:66], pose[:, :3]
    if "smpl_pose" in entry:
        pose = np.asarray(entry["smpl_pose"], dtype=np.float32)
        pose = pose.reshape(pose.shape[0], -1)
        if pose.shape[1] >= 66:  # global_orient(3) + body(63+)
            return pose[:, 3:66], pose[:, :3]
        if pose.shape[1] == 63:
            go = np.asarray(entry.get("global_orient", np.zeros((pose.shape[0], 3))),
                            dtype=np.float32)
            return pose, go
    if "body_pose" in entry:
        return (np.asarray(entry["body_pose"], dtype=np.float32).reshape(-1, 69)[:, :63],
                np.asarray(entry["global_orient"], dtype=np.float32))
    raise ValueError(f"Unrecognized pkl schema: keys={list(entry.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-motion", choices=["arm_raise", "wrist_heading"],
                    default="arm_raise")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--pkl", type=str)
    ap.add_argument("--motion", type=str, default=None)
    ap.add_argument("--src-fps", type=float, default=None,
                    help="fps of the pkl motion; resampled to 50 Hz if given")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        gen = {"arm_raise": synthetic_arm_raise,
               "wrist_heading": synthetic_wrist_heading}[args.synthetic_motion]
        body_pose, global_orient = gen(args.seconds)
    elif args.pkl:
        body_pose, global_orient = load_pkl(args.pkl, args.motion)
        if args.src_fps and abs(args.src_fps - FPS) > 1e-3:
            T = body_pose.shape[0]
            grid = np.arange(0, T - 1, args.src_fps / FPS)
            i0 = np.floor(grid).astype(int)
            w = (grid - i0)[:, None]
            body_pose = (1 - w) * body_pose[i0] + w * body_pose[i0 + 1]
            global_orient = (1 - w) * global_orient[i0] + w * global_orient[i0 + 1]
            print(f"[replay] resampled {T} frames @{args.src_fps} -> {len(grid)} @{FPS}")
    else:
        ap.error("choose --synthetic or --pkl PATH")

    print(f"[replay] converting {body_pose.shape[0]} frames to stream format...")
    frames = smpl_to_stream_frames(body_pose, global_orient)
    print(f"[replay] streaming {len(frames)} frames @{FPS} Hz on :{args.port} "
          f"(topic 'pose', protocol v3){' [loop]' if args.loop else ''}")

    pub = SlidingWindowPosePublisher(port=args.port)
    try:
        run_paced(pub, frames, fps=FPS, loop=args.loop)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        print(f"[replay] done, {pub.sent} messages sent")


if __name__ == "__main__":
    main()
