"""M2: live webcam -> SMPL -> ZMQ pose stream (runs inside the webcam2motion container).

Threads:
  CaptureThread   — cv2 grab, keep latest frame (video files: paced + looped)
  EstimatorThread — GVHMR sliding window per frame; OneEuro/quat filters;
                    draws the preview overlay (bbox + COCO17 skeleton + latency
                    HUD) and publishes JPEG on tcp://*:5559 topic "preview"
  main            — 50 Hz publisher: interpolates between the last two filtered
                    estimates and pushes protocol-v3 windows on :5556

Latency modes (--latency-mode):
  smooth  (default) interpolate between estimates -> +1 est-interval latency
  hold    always hold the latest estimate (7-12 fps steps, lowest latency)
  predict mild constant-velocity extrapolation beyond the latest estimate

Confidence gating: estimator returns None -> stop publishing (robot holds
pose); on recovery frame_index jumps -> clean catch-up reset downstream.

Usage:
  python stream_webcam_zmq.py --camera 0 --preview
  python stream_webcam_zmq.py --video docs/example_video/tennis.mp4 --preview
Host-side viewer: .venv_sim/bin/python tools/webcam2motion/preview_viewer.py
"""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import OneEuro, QuatLowpass  # noqa: E402
from publisher import POLICY_FPS, SlidingWindowPosePublisher  # noqa: E402
from smpl_adapter import smpl_to_stream_frames  # noqa: E402

from gear_sonic.trl.utils.torch_transform import (  # noqa: E402
    angle_axis_to_quaternion,
    quaternion_to_angle_axis,
)
import torch  # noqa: E402

COCO17_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


class CaptureThread(threading.Thread):
    """Grab frames as fast as the camera allows; keep only the latest.

    `source` is a camera index, or a video path (paced at its native fps and
    looped — a fake camera for end-to-end tests without a person on camera).
    """

    def __init__(self, source, width: int = 1280, height: int = 720):
        super().__init__(daemon=True)
        self.is_video = isinstance(source, str)
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.is_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30) if self.is_video else 0.0
        self.latest: tuple[np.ndarray, float] | None = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        next_t = time.monotonic()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_video:  # loop the fake camera
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    continue
                time.sleep(0.05)
                continue
            with self.lock:
                self.latest = (frame, time.monotonic())
            if self.is_video:  # pace at native fps
                next_t += self.period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False
        self.cap.release()


class EstimatorThread(threading.Thread):
    """Run the estimator back-to-back on the newest frame; keep the last two
    filtered estimates for the 50 Hz publisher; render/publish the preview."""

    def __init__(self, est, cap: CaptureThread, args):
        super().__init__(daemon=True)
        self.est = est
        self.cap = cap
        self.args = args
        self.lock = threading.Lock()
        self.prev: tuple | None = None   # (t_cap, body_pose, global_orient)
        self.curr: tuple | None = None
        self.hands_latest: dict = {}     # smoothed wrist angles for override
        self.lost = True
        self.est_count = 0
        self.publish_fps = 0.0  # set by main for the HUD
        self.running = True
        self._pose_filter = OneEuro(min_cutoff=args.min_cutoff, beta=args.beta)
        self._quat_filter = QuatLowpass(alpha=0.4)
        self._hand_filters = {"left": OneEuro(min_cutoff=0.8, beta=0.05),
                              "right": OneEuro(min_cutoff=0.8, beta=0.05)}
        self._rh56_filters = {"left": OneEuro(min_cutoff=1.5, beta=0.1),
                              "right": OneEuro(min_cutoff=1.5, beta=0.1)}
        self._preview_sock = None
        self._last_stage_ms = {}

    def _preview(self, frame, result, t_cap):
        if not self.args.preview:
            return
        if self._preview_sock is None:
            ctx = zmq.Context.instance()
            self._preview_sock = ctx.socket(zmq.PUB)
            self._preview_sock.setsockopt(zmq.SNDHWM, 2)
            self._preview_sock.bind(f"tcp://*:{self.args.preview_port}")
        img = frame.copy()
        if result is not None:
            x, y, s = result["bbx_xys"]
            cv2.rectangle(img, (int(x - s / 2), int(y - s / 2)),
                          (int(x + s / 2), int(y + s / 2)), (0, 200, 255), 2)
            kp = result["kp2d"]
            for a, b in COCO17_EDGES:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])),
                             (int(kp[b, 0]), int(kp[b, 1])), (0, 255, 0), 2)
            for j in range(17):
                if kp[j, 2] > 0.3:
                    cv2.circle(img, (int(kp[j, 0]), int(kp[j, 1])), 3, (0, 0, 255), -1)
            from estimators.hand_tracker import draw_hand_panel
            for side, h in (result.get("hands") or {}).items():
                lm = h["landmarks"]
                for p in lm:
                    cv2.circle(img, (int(p[0]), int(p[1])), 2, (255, 0, 255), -1)
                # palm direction arrow: wrist -> middle_mcp
                cv2.arrowedLine(img, (int(lm[0, 0]), int(lm[0, 1])),
                                (int(lm[9, 0]), int(lm[9, 1])), (255, 0, 255), 2)
                roll_deg = float(np.degrees(h["wrist_angles"][0]))
                cv2.putText(img, f"{side[0].upper()} roll {roll_deg:+.0f}",
                            (int(lm[0, 0]) + 8, int(lm[0, 1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
                # simulated-hand inset: canonical palm view + RH56 command bars
                panel_x = 8 if side == "left" else img.shape[1] - 158
                draw_hand_panel(img, h, side, (panel_x, img.shape[0] - 204))
        sm = self._last_stage_ms
        age_ms = 1e3 * (time.monotonic() - t_cap)
        hands_hud = ""
        if result is not None and result.get("hands"):
            hands_hud = "  hands: " + ",".join(sorted(result["hands"].keys()))
        hud = [
            ("PERSON" if result is not None else "NO PERSON")
            + f"  est_age={age_ms:.0f}ms  mode={self.args.latency_mode}" + hands_hud,
            (f"yolo={sm.get('yolo', 0):.0f} kp2d+hands={sm.get('kp2d+hands', 0):.0f} "
             f"feat={sm.get('feat', 0):.0f} head={sm.get('head', 0):.0f} "
             f"total={sm.get('total', 0):.0f}ms") if sm else "warming up...",
            f"est={self.est_rate:.1f}fps  publish={self.publish_fps:.1f}fps",
        ]
        for i, line in enumerate(hud):
            cv2.putText(img, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4)
            cv2.putText(img, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._preview_sock.send(b"preview" + jpg.tobytes())

    est_rate = 0.0

    def run(self):
        last_t = 0.0
        rate_t0, rate_n = time.monotonic(), 0
        while self.running:
            got = self.cap.get()
            if got is None or got[1] <= last_t:
                time.sleep(0.002)
                continue
            frame, t_cap = got
            last_t = t_cap
            result = self.est.estimate(frame, t_cap)
            if result is not None:
                self._last_stage_ms = result.get("stage_ms", {})
                bp = self._pose_filter(result["body_pose"], t_cap)
                q = self._quat_filter(
                    angle_axis_to_quaternion(
                        torch.tensor(result["global_orient"][None], dtype=torch.float32)
                    )[0].numpy())
                go = quaternion_to_angle_axis(
                    torch.tensor(q[None], dtype=torch.float32))[0].numpy()
                hands = {}
                for side, h in (result.get("hands") or {}).items():
                    hands[side] = {
                        "wrist_angles": self._hand_filters[side](h["wrist_angles"], t_cap),
                        "rh56": self._rh56_filters[side](h["rh56"], t_cap)
                        if h.get("rh56") is not None else None,
                    }
                with self.lock:
                    self.prev = self.curr
                    self.curr = (t_cap, bp.astype(np.float32), go.astype(np.float32))
                    self.hands_latest = hands
                    self.lost = False
                rate_n += 1
            else:
                if not self.lost:
                    print("[stream] person lost — pausing publish")
                self._pose_filter.reset()
                self._quat_filter.reset()
                with self.lock:
                    self.prev = self.curr = None
                    self.lost = True
            now = time.monotonic()
            if now - rate_t0 >= 2.0:
                self.est_rate = rate_n / (now - rate_t0)
                rate_t0, rate_n = now, 0
            self._preview(frame, result, t_cap)

    def snapshot(self):
        with self.lock:
            return self.prev, self.curr, self.lost, self.hands_latest


def slerp_aa(aa0: np.ndarray, aa1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate/extrapolate two global_orient axis-angles via quaternion nlerp."""
    q = angle_axis_to_quaternion(torch.tensor(np.stack([aa0, aa1]), dtype=torch.float32))
    q0, q1 = q[0].numpy(), q[1].numpy()
    if np.dot(q0, q1) < 0:
        q1 = -q1
    qi = (1 - alpha) * q0 + alpha * q1
    qi /= np.linalg.norm(qi) + 1e-12
    return quaternion_to_angle_axis(torch.tensor(qi[None], dtype=torch.float32))[0].numpy()


def make_estimator(name: str, args):
    if name == "gvhmr":
        from estimators.gvhmr_runner import GVHMRStreamEstimator
        # RTX 5080 Laptop, tennis.mp4: see README for the bench table.
        return GVHMRStreamEstimator(device="cuda", window=32, flip_test=False,
                                    postproc=False, fp16=not args.no_fp16,
                                    hands=not args.no_hands,
                                    yolo_period=args.yolo_period)
    raise SystemExit(f"unknown estimator: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="gvhmr")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, default=None,
                    help="video file as a fake camera (paced + looped)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--preview", action="store_true",
                    help="publish overlay JPEG on --preview-port (view on host "
                         "with preview_viewer.py)")
    ap.add_argument("--preview-port", type=int, default=5559)
    ap.add_argument("--latency-mode", choices=["smooth", "hold", "predict"],
                    default="smooth")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-hands", action="store_true",
                    help="disable MediaPipe palm-orientation (wrist roll) tracking")
    ap.add_argument("--yolo-period", type=int, default=2)
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    est = make_estimator(args.estimator, args)
    cap = CaptureThread(args.video if args.video else args.camera)
    cap.start()
    worker = EstimatorThread(est, cap, args)
    worker.start()
    pub = SlidingWindowPosePublisher(port=args.port)
    wrist_blender = None
    if not args.no_hands:
        from estimators.hand_tracker import WristBlender
        wrist_blender = WristBlender()

    pub_count = 0
    was_lost = True
    last_report = time.monotonic()
    period = 1.0 / POLICY_FPS
    next_tick = time.monotonic()
    est_interval = 0.15  # adaptive EMA of the estimate spacing

    print(f"[stream] running (latency-mode={args.latency_mode}"
          f"{', preview on :%d' % args.preview_port if args.preview else ''}) "
          "— Ctrl-C to stop")
    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.002, next_tick - now))
                continue
            next_tick += period
            if next_tick < now - 0.5:  # fell far behind; resync
                next_tick = now + period

            prev, curr, lost, hands = worker.snapshot()
            if lost or curr is None:
                if not was_lost:
                    pub.skip(POLICY_FPS)  # frame_index jump -> clean catch-up
                    was_lost = True
                continue
            was_lost = False

            t1, bp1, go1 = curr
            if prev is not None and curr[0] > prev[0]:
                t0, bp0, go0 = prev
                est_interval = 0.9 * est_interval + 0.1 * (t1 - t0)
                if args.latency_mode == "smooth":
                    target = now - est_interval
                    alpha = float(np.clip((target - t0) / max(t1 - t0, 1e-3), 0.0, 1.0))
                elif args.latency_mode == "predict":
                    alpha = float(np.clip((now - t0) / max(t1 - t0, 1e-3), 0.0, 1.5))
                else:  # hold
                    alpha = 1.0
                bp = (1 - alpha) * bp0 + alpha * bp1
                go = slerp_aa(go0, go1, alpha)
            else:
                bp, go = bp1, go1

            frame_dict = smpl_to_stream_frames(bp[None], go[None])[0]
            if wrist_blender is not None:
                frame_dict["joint_pos"] = wrist_blender.apply(hands, frame_dict["joint_pos"])
                for side, h in wrist_blender.held_hands().items():  # holds through dropouts
                    if h.get("rh56") is not None:
                        frame_dict[f"{side}_hand_rh56"] = h["rh56"]  # Inspire RH56DFQ, 0..1
            pub.push(frame_dict)
            pub_count += 1

            if now - last_report >= 5.0:
                worker.publish_fps = pub_count / (now - last_report)
                print(f"[stream] est fps={worker.est_rate:.1f} "
                      f"publish fps={worker.publish_fps:.1f} "
                      f"est_interval={1e3*est_interval:.0f}ms")
                pub_count = 0
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        worker.running = False
        cap.stop()
        pub.close()


if __name__ == "__main__":
    main()
