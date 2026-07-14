"""50 Hz sliding-window pose publisher for the SONIC ZMQ streaming interface.

Replicates the proven chunking of gear_sonic/scripts/pico_manager_thread_server.py:
each message is a window of the last `window` frames (stride-1, monotonically
increasing int64 frame_index), published on topic "pose" as a protocol-v3
packed message via gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message
(HEADER_SIZE=1280 — do NOT use the stale 1024-byte packer from
pose_estimation_server_onboard_test.py).

Minimal v3 field set accepted by the C++ decoder
(zmq_endpoint_interface.hpp): smpl_pose, smpl_joints, joint_pos, joint_vel,
body_quat_w, frame_index.
"""

from collections import deque
import time

import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

POLICY_FPS = 50


class SlidingWindowPosePublisher:
    """Buffers per-frame pose dicts and publishes overlapping v3 windows."""

    def __init__(self, port: int = 5556, topic: str = "pose", window: int = 5):
        self.topic = topic
        self.window = window
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 3)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        time.sleep(0.2)  # let SUBs connect before the first frames

        self._buf = {
            "smpl_pose": deque(maxlen=window),
            "smpl_joints": deque(maxlen=window),
            "body_quat_w": deque(maxlen=window),
            "joint_pos": deque(maxlen=window),
        }
        self._frame_index = deque(maxlen=window)
        # epoch-based start: stays monotonic across publisher restarts, so a
        # deploy session that already saw a previous stream always observes a
        # forward jump (clean catch-up) instead of a backward index. Modulo
        # keeps it well inside int32 (the C++ merger casts frame_index to int).
        self._step = int(time.time() % 10**7) * POLICY_FPS
        self._sent = 0

    def push(self, frame: dict) -> bool:
        """Append one frame and publish the window once it is full.

        frame keys: smpl_pose (21,3) f32, smpl_joints (24,3) f32,
        body_quat_w (4,) f32 wxyz, joint_pos (29,) f32 (wrists only).
        Optional: left_hand_joints / right_hand_joints (7,) f32 (Dex3) —
        forwarded per-message (latest value) like the PICO teleop server does.
        Returns True if a message was sent.
        """
        for key, buf in self._buf.items():
            buf.append(np.asarray(frame[key], dtype=np.float32))
        self._frame_index.append(self._step)
        self._step += 1

        if len(self._frame_index) < self.window:
            return False

        n = len(self._frame_index)
        data = {
            "smpl_pose": np.stack(self._buf["smpl_pose"], axis=0),
            "smpl_joints": np.stack(self._buf["smpl_joints"], axis=0),
            "body_quat_w": np.stack(self._buf["body_quat_w"], axis=0),
            "joint_pos": np.stack(self._buf["joint_pos"], axis=0),
            "joint_vel": np.zeros((n, 29), dtype=np.float32),
            "frame_index": np.array(self._frame_index, dtype=np.int64),
        }
        for hand in ("left_hand_joints", "right_hand_joints",     # Dex3 (7,)
                     "left_hand_rh56", "right_hand_rh56"):        # Inspire RH56DFQ (6,)
            if frame.get(hand) is not None:
                data[hand] = np.asarray(frame[hand], dtype=np.float32).reshape(-1)
        self.socket.send(pack_pose_message(data, topic=self.topic))
        self._sent += 1
        return True

    def skip(self, frames: int = 1):
        """Advance frame_index without publishing (pause semantics: the
        receiver sees a gap on resume and does a clean catch-up reset)."""
        self._step += frames
        for buf in self._buf.values():
            buf.clear()
        self._frame_index.clear()

    @property
    def sent(self) -> int:
        return self._sent

    def close(self):
        self.socket.close(0)
        self.context.term()


def run_paced(publisher: SlidingWindowPosePublisher, frames, fps: int = POLICY_FPS,
              loop: bool = False, verbose_every: float = 5.0):
    """Publish an iterable of frame dicts at a fixed wall-clock rate."""
    period = 1.0 / fps
    next_deadline = time.monotonic()
    last_report = time.monotonic()
    count = 0
    while True:
        for frame in frames:
            publisher.push(frame)
            count += 1
            next_deadline += period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_deadline = time.monotonic()  # fell behind; don't burst
            now = time.monotonic()
            if now - last_report >= verbose_every:
                print(f"[publisher] frames={count} sent={publisher.sent} "
                      f"({count / (now - last_report + 1e-9):.0f} eff fps window)")
                last_report = now
                count = 0
        if not loop:
            break
