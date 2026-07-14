"""M0 end-to-end test: stream SMPL over ZMQ into the deploy binary + MuJoCo sim.

Reuses the sonic_bench harness (headless sim server + stdin key injection into
the g1-deploy-dev container). Sequence:

  sim_server -> deploy (--input-type zmq, low_latency) -> ']' start control
  -> drop robot -> settle -> start replayer (synthetic or pkl) -> ENTER
  -> expect "ZMQ STREAMING MODE: ENABLED" + "Protocol version 3 established"
  -> watch fall_count / pelvis height for the streaming window.

Run from repo root with .venv_sim python:
  .venv_sim/bin/python tools/webcam2motion/test_m0.py [--pkl PATH] [--duration 30]
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "sonic_bench"))

from common import (  # noqa: E402
    CONTAINER,
    DDS_INTERFACE,
    DEPLOY_BINARY,
    DEPLOY_WS_CONTAINER,
    MOTION_DATA_REL,
    PLANNER_REL,
    VARIANTS,
    docker_exec_cmd,
    kill_deploy,
)
from deploy_client import DeployProc  # noqa: E402
from record_debug_stream import StreamRecorder  # noqa: E402
from run_benchmark import SimCtrl, start_sim_server, stop_sim_server, wait_sim_ready  # noqa: E402

RE_STREAM_ENABLED = re.compile(r"ZMQ STREAMING MODE: ENABLED")
RE_PROTOCOL = re.compile(r"Protocol version (\d+) established")
RE_STREAM_ERROR = re.compile(
    r"Missing required field|Unsupported protocol|Invalid incoming|Header too|Version \d+ missing"
)

LOGS = REPO_ROOT / "gear_sonic_deploy" / "logs" / "webcam2motion_m0"


class ZmqDeployProc(DeployProc):
    """DeployProc variant: --input-type zmq (keyboard keys still work on stdin)."""

    def __init__(self, variant: str, logs_dir_container: str, stdout_log: Path,
                 zmq_port: int = 5556):
        super().__init__(variant, logs_dir_container, stdout_log)
        v = VARIANTS[variant]
        inner = (
            f"exec {DEPLOY_BINARY} {DDS_INTERFACE} {v['decoder']} {MOTION_DATA_REL} "
            f"--obs-config {v['obs_config']} "
            f"--encoder-file {v['encoder']} "
            f"--planner-file {PLANNER_REL} "
            f"--input-type zmq --zmq-host localhost --zmq-port {zmq_port} --zmq-topic pose "
            f"--output-type zmq --zmq-out-port 5557 --zmq-out-topic g1_debug "
            f"--disable-crc-check "
            f"--logs-dir {logs_dir_container}"
        )
        self.cmd = docker_exec_cmd(inner)
        self.stream_events: list[str] = []

    def _parse_line(self, line: str) -> None:
        if RE_STREAM_ENABLED.search(line):
            self.events.put(("stream_enabled", None))
            return
        m = RE_PROTOCOL.search(line)
        if m:
            self.events.put(("protocol_established", int(m.group(1))))
            return
        if RE_STREAM_ERROR.search(line):
            self.stream_events.append(line.strip())
            self.events.put(("stream_error", line.strip()))
            return
        super()._parse_line(line)


def analyze_wrist_heading(npz_path: Path) -> bool:
    """For the wrist_heading synthetic motion: measured G1 wrist joints
    (MuJoCo idx L=19..21, R=26..28) must oscillate at ~0.3 Hz, and the base
    yaw (from base_quat_measured, wxyz) must sweep ~±0.5 rad at ~0.05 Hz."""
    import numpy as np
    data = np.load(npz_path)
    q_meas = np.asarray(data["body_q_measured"], dtype=np.float64)
    base_q = np.asarray(data["base_quat_measured"], dtype=np.float64)  # wxyz

    wrists = q_meas[:, [19, 20, 21, 26, 27, 28]]
    stds = wrists.std(axis=0)
    j = wrists[:, int(np.argmax(stds))]
    freqs = np.fft.rfftfreq(len(j), 1 / 50.0)
    peak = float(freqs[np.argmax(np.abs(np.fft.rfft(j - j.mean())))])

    w, x, y, z = base_q.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    yaw_range = float(yaw.max() - yaw.min())

    print(f"[m0] wrist max-std={stds.max():.3f} rad, dominant freq={peak:.2f} Hz "
          f"(expect 0.3); base yaw sweep={np.degrees(yaw_range):.1f} deg (expect ~57)")
    wrist_ok = stds.max() > 0.1 and abs(peak - 0.3) < 0.1
    heading_ok = yaw_range > 0.5  # at least ~29 deg of commanded ~57
    print(f"[m0] wrist_tracking={'ok' if wrist_ok else 'POOR'} "
          f"heading_tracking={'ok' if heading_ok else 'POOR'}")
    return wrist_ok and heading_ok


def analyze_tracking(npz_path: Path, expect_freq: float | None = None) -> bool:
    """The streamed SMPL reference must visibly move the robot's arms.

    Note: in encoder/SMPL mode `body_q_target` only mirrors the streamed
    `joint_pos` field (wrists-only, ~zero) — the motion intent goes through the
    SMPL encoder — so assert on `body_q_measured` instead. Optionally check the
    dominant oscillation frequency (synthetic arm-raise runs at 0.25 Hz).
    """
    import numpy as np
    try:
        data = np.load(npz_path)
        measured = np.asarray(data["body_q_measured"], dtype=np.float64)
    except Exception as e:
        print(f"[m0] tracking analysis unavailable ({e})")
        return False
    arms = measured[:, 15:29]  # MuJoCo order arm joints
    stds = arms.std(axis=0)
    moving = float(stds.max())
    msg = f"[m0] measured arm max-std={moving:.3f} rad ({len(arms)} samples)"
    ok = moving > 0.05
    if expect_freq is not None and ok:
        j = arms[:, int(np.argmax(stds))]
        freqs = np.fft.rfftfreq(len(j), 1 / 50.0)
        peak = float(freqs[np.argmax(np.abs(np.fft.rfft(j - j.mean())))])
        msg += f", dominant freq={peak:.3f} Hz (expected {expect_freq})"
        ok = abs(peak - expect_freq) < 0.1
    print(msg)
    return ok


def ensure_container() -> None:
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"container {CONTAINER} not found — run gear_sonic_deploy/docker/run-ros2-dev.sh once")
    if r.stdout.strip() != "true":
        print(f"[m0] starting container {CONTAINER}...")
        subprocess.run(["docker", "start", CONTAINER], check=True, capture_output=True)
        time.sleep(2)


def assert_dds_clear(seconds: float = 1.5) -> None:
    """Abort early if ANOTHER simulator is already publishing rt/lowstate on
    DDS domain 0 (e.g. a leftover run_sim_loop.py / run_sim_rh56.py window).
    Two sims interleave their states into the deploy's observations and the
    robot falls constantly — a maddening failure mode that looks like a bad
    model. Cost us an evening: 2026-07-14."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    ChannelFactoryInitialize(0, "lo")
    n = [0]
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda msg: n.__setitem__(0, n[0] + 1), 10)
    time.sleep(seconds)
    sub.Close()
    if n[0] > 0:
        raise SystemExit(
            f"[dds] rt/lowstate already flowing on domain 0 ({n[0]} msgs in "
            f"{seconds:.0f}s) — close any other running simulator "
            "(run_sim_loop.py / run_sim_rh56.py) before running tests")


def kill_deploy_and_wait(timeout: float = 15.0) -> None:
    """kill_deploy + wait until the g1_debug port is actually released —
    a half-dead leftover binary aborts the next run with 'Address already in
    use' on :5557."""
    from common import port_free
    kill_deploy()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_free(5557):
            return
        subprocess.run(["docker", "exec", CONTAINER, "pkill", "-9", "-f",
                        "g1_deploy_onnx_ref"], capture_output=True)
        time.sleep(1)
    raise RuntimeError("port 5557 still busy after kill_deploy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, default=None, help="SMPL pkl to replay (default: synthetic)")
    ap.add_argument("--motion", type=str, default=None)
    ap.add_argument("--src-fps", type=float, default=None)
    ap.add_argument("--variant", default="low_latency", choices=["low_latency", "release"])
    ap.add_argument("--duration", type=float, default=30.0, help="streaming observation window (s)")
    ap.add_argument("--synthetic-motion", choices=["arm_raise", "wrist_heading"],
                    default="arm_raise")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    ensure_container()
    kill_deploy_and_wait()
    assert_dds_clear()

    sim_proc = start_sim_server(LOGS / "sim_stdout.log")
    ctrl = None
    dp = None
    streamer = None
    ok = False
    try:
        ctrl = SimCtrl()
        wait_sim_ready(ctrl)
        print("[m0] sim ready")

        dp = ZmqDeployProc(args.variant,
                           f"{DEPLOY_WS_CONTAINER}/logs/webcam2motion_m0",
                           LOGS / "deploy_stdout.log")
        dp.start()
        dp.wait_event("init_done", timeout=720)
        print(f"[m0] deploy init done ({len(dp.motions)} motions preloaded)")

        dp.send_key("]")
        dp.wait_event("control_started", timeout=15)
        print("[m0] control started")
        ctrl.call("drop")
        time.sleep(4)
        state = ctrl.call("state")
        print(f"[m0] after drop: height={state.get('height'):.3f} falls={state.get('fall_count')}")

        # start the streamer first so data is flowing when streaming mode engages
        streamer_cmd = [sys.executable, str(REPO_ROOT / "tools/webcam2motion/replay_smpl_zmq.py"),
                        "--loop"]
        if args.pkl:
            streamer_cmd += ["--pkl", args.pkl]
            if args.motion:
                streamer_cmd += ["--motion", args.motion]
            if args.src_fps:
                streamer_cmd += ["--src-fps", str(args.src_fps)]
        else:
            streamer_cmd += ["--synthetic", "--synthetic-motion", args.synthetic_motion,
                             "--seconds", "30" if args.synthetic_motion == "wrist_heading" else "20"]
        streamer = subprocess.Popen(streamer_cmd, cwd=REPO_ROOT,
                                    stdout=open(LOGS / "streamer_stdout.log", "wb"),
                                    stderr=subprocess.STDOUT)
        time.sleep(2)

        dp.send_key("\n")
        dp.wait_event("stream_enabled", timeout=10)
        print("[m0] ZMQ STREAMING MODE: ENABLED")
        proto = dp.wait_event("protocol_established", timeout=10)
        print(f"[m0] protocol version {proto} established")
        if proto != 3:
            raise RuntimeError(f"expected protocol 3, got {proto}")

        recorder = StreamRecorder(LOGS / "stream.npz")
        recorder.start()
        t0 = time.monotonic()
        falls0 = ctrl.call("state").get("fall_count", 0)
        while time.monotonic() - t0 < args.duration:
            time.sleep(2)
            s = ctrl.call("state")
            print(f"[m0] t={time.monotonic()-t0:5.1f}s height={s.get('height'):.3f} "
                  f"falls={s.get('fall_count')}")
            if streamer.poll() is not None:
                raise RuntimeError("streamer exited early — check streamer_stdout.log")
        falls = ctrl.call("state").get("fall_count", 0) - falls0
        recorder.stop()

        if dp.stream_events:
            print("[m0] STREAM ERRORS:")
            for e in dp.stream_events[:10]:
                print("   ", e)
        if not args.pkl and args.synthetic_motion == "wrist_heading":
            tracked = analyze_wrist_heading(LOGS / "stream.npz")
        else:
            tracked = analyze_tracking(LOGS / "stream.npz",
                                       expect_freq=None if args.pkl else 0.25)
        ok = falls == 0 and not dp.stream_events and tracked
        print(f"[m0] RESULT: {'PASS' if ok else 'FAIL'} (falls={falls}, "
              f"stream_errors={len(dp.stream_events)}, arm_tracking={'ok' if tracked else 'FLAT/POOR'})")
    finally:
        if streamer and streamer.poll() is None:
            streamer.terminate()
        if dp:
            dp.stop()
        stop_sim_server(sim_proc, ctrl)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
