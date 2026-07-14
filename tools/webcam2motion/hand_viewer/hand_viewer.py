"""MuJoCo viewer for the Inspire RH56DFQ hands, driven by the rh56 stream.

Stage 1 of the hand-in-sim plan: a standalone window showing both DFQ hands
(real linkage geometry from unitree_ros URDFs) posed kinematically from the
`left/right_hand_rh56 (6,)` fields on the :5556 pose stream. No dynamics —
per-frame qpos writes; underactuated linkages are expanded with the URDF
mimic ratios (thumb intermediate = 1.6x pitch, distal = 2.4x pitch,
finger intermediate = 1x proximal), so no equality constraints are needed.

Run on the host (needs GUI):
  .venv_sim/bin/python tools/webcam2motion/hand_viewer/hand_viewer.py          # live
  .venv_sim/bin/python tools/webcam2motion/hand_viewer/hand_viewer.py --demo  # sinusoid
"""

import argparse
import json
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import zmq

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HEADER_SIZE = 1280

# RH56DFQ command order (matches estimators/hand_tracker.RH56_ORDER)
RH56_ORDER = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

# per-hand joint layout in the URDF (S = side prefix L/R). driven joint and
# its mimic followers (name, multiplier) per RH56 axis.
def _axes(S: str) -> dict:
    return {
        "little": (f"{S}_pinky_proximal_joint", [(f"{S}_pinky_intermediate_joint", 1.0)]),
        "ring": (f"{S}_ring_proximal_joint", [(f"{S}_ring_intermediate_joint", 1.0)]),
        "middle": (f"{S}_middle_proximal_joint", [(f"{S}_middle_intermediate_joint", 1.0)]),
        "index": (f"{S}_index_proximal_joint", [(f"{S}_index_intermediate_joint", 1.0)]),
        "thumb_bend": (f"{S}_thumb_proximal_pitch_joint",
                       [(f"{S}_thumb_intermediate_joint", 1.6),
                        (f"{S}_thumb_distal_joint", 2.4)]),
        "thumb_rot": (f"{S}_thumb_proximal_yaw_joint", []),
    }


class HandModel:
    def __init__(self, spec_parent: mujoco.MjSpec, side: str, x_offset: float):
        S = "L" if side == "left" else "R"
        urdf = os.path.join(ASSETS, f"DFQ_{side}_hand.urdf")
        sub = mujoco.MjSpec.from_file(urdf)
        frame = spec_parent.worldbody.add_frame(pos=[x_offset, 0, 0.15],
                                                euler=[np.pi / 2, 0, 0])
        frame.attach_body(sub.worldbody.first_body(), f"{side}_", "")
        self.side, self.S = side, S
        self._qadr: dict[str, list[tuple[int, float]]] = {}

    def resolve(self, model: mujoco.MjModel):
        """Map each RH56 axis to (qpos address, joint range, multiplier)."""
        for axis, (driven, followers) in _axes(self.S).items():
            entries = []
            for name, mult in [(driven, 1.0)] + followers:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f"{self.side}_{name}")
                if jid < 0:
                    continue
                lo, hi = model.jnt_range[jid]
                entries.append((model.jnt_qposadr[jid], lo, hi, mult))
            self._qadr[axis] = entries

    def write(self, data: mujoco.MjData, cmd: np.ndarray):
        """cmd: (6,) 0..1 in RH56_ORDER -> qpos. Driven joint sweeps its URDF
        range; mimic followers = multiplier x driven angle (their own range
        clips them, matching the real four-bar linkage travel)."""
        for i, axis in enumerate(RH56_ORDER):
            v = float(np.clip(cmd[i], 0, 1))
            entries = self._qadr[axis]
            if not entries:
                continue
            adr0, lo0, hi0, _ = entries[0]
            q0 = lo0 + v * (hi0 - lo0)
            data.qpos[adr0] = q0
            for adr, lo, hi, mult in entries[1:]:
                data.qpos[adr] = np.clip(q0 * mult, lo, hi)


class Rh56Subscriber(threading.Thread):
    """Latest left/right rh56 command from the :5556 pose stream."""

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"pose")
        self.sock.setsockopt(zmq.RCVTIMEO, 500)
        self.sock.connect(f"tcp://{host}:{port}")
        self.latest = {"left": None, "right": None}
        self.running = True

    def run(self):
        while self.running:
            try:
                msg = self.sock.recv()
            except zmq.error.Again:
                continue
            try:
                hdr = json.loads(msg[4:4 + HEADER_SIZE].rstrip(b"\x00"))
                off = 4 + HEADER_SIZE
                for f in hdr["fields"]:
                    n = int(np.prod(f["shape"]))
                    nbytes = n * (8 if f["dtype"] in ("f64", "i64") else 4)
                    if f["name"] in ("left_hand_rh56", "right_hand_rh56"):
                        arr = np.frombuffer(msg, dtype="<f4", count=n, offset=off)
                        self.latest[f["name"].split("_")[0]] = arr.copy()
                    off += nbytes
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--demo", action="store_true", help="sinusoidal open/close, no stream")
    args = ap.parse_args()

    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, -0.4, 0.8], dir=[0, 0.4, -0.8])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.5, 0.5, 0.1],
                            rgba=[0.15, 0.15, 0.18, 1])
    hands = [HandModel(spec, "left", -0.12), HandModel(spec, "right", 0.12)]
    model = spec.compile()
    for h in hands:
        h.resolve(model)
    data = mujoco.MjData(model)

    sub = None
    if not args.demo:
        sub = Rh56Subscriber(args.host, args.port)
        sub.start()
        print(f"[hand_viewer] listening for rh56 on tcp://{args.host}:{args.port}")

    cmd = {"left": np.zeros(6), "right": np.zeros(6)}
    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            if args.demo:
                t = time.time() - t0
                wave = 0.5 * (1 + np.sin(2 * np.pi * 0.2 * t))
                for side in cmd:
                    cmd[side][:] = [wave, wave, wave, wave, wave,
                                    0.5 * (1 + np.sin(2 * np.pi * 0.1 * t))]
            elif sub is not None:
                for side in cmd:
                    if sub.latest[side] is not None:
                        cmd[side] = sub.latest[side]
            for h in hands:
                h.write(data, cmd[h.side])
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
