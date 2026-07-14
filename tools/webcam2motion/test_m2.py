"""M2 end-to-end test: LIVE streaming pipeline (container) -> deploy -> MuJoCo.

Same harness as test_m0, but the publisher is stream_webcam_zmq.py running in
the webcam2motion container with a video file as fake camera (--video) or a
real webcam (--camera N).

Run from repo root with .venv_sim python:
  .venv_sim/bin/python tools/webcam2motion/test_m2.py                 # tennis fake-cam
  .venv_sim/bin/python tools/webcam2motion/test_m2.py --camera 0      # real webcam
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "sonic_bench"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "webcam2motion"))

from record_debug_stream import StreamRecorder  # noqa: E402
from run_benchmark import SimCtrl, start_sim_server, stop_sim_server, wait_sim_ready  # noqa: E402

from test_m0 import ZmqDeployProc, analyze_tracking, ensure_container  # noqa: E402
from test_m0 import assert_dds_clear, kill_deploy_and_wait  # noqa: E402
from common import DEPLOY_WS_CONTAINER  # noqa: E402

LOGS = REPO_ROOT / "gear_sonic_deploy" / "logs" / "webcam2motion_m2"


def start_streamer(args, log_name: str = "streamer_stdout.log") -> subprocess.Popen:
    src = ["--camera", str(args.camera)] if args.camera is not None else \
          ["--video", args.video]
    cmd = [
        "docker", "run", "--rm", "--name", "w2m-stream", "--network", "host",
        "--ipc", "host", "--gpus", "all",
    ]
    if args.camera is not None:
        cmd += ["--device", f"/dev/video{args.camera}"]
    cmd += [
        "-v", f"{REPO_ROOT}:/workspace/gr00t-wbc:rw",
        "-v", f"{REPO_ROOT}/tools/webcam2motion/checkpoints:/opt/GVHMR/inputs/checkpoints:ro",
        "-w", "/opt/GVHMR", "webcam2motion",
        "python", "-u", "/workspace/gr00t-wbc/tools/webcam2motion/stream_webcam_zmq.py",
    ] + src
    if args.preview:
        cmd += ["--preview"]
    return subprocess.Popen(cmd, stdout=open(LOGS / log_name, "wb"),
                            stderr=subprocess.STDOUT)


def wait_publishing(streamer: subprocess.Popen, log_name: str, timeout: float = 180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        log_path = LOGS / log_name
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        for line in reversed(log.splitlines()):
            if "publish fps=" in line:
                if float(line.rsplit("publish fps=", 1)[1].split()[0]) > 1.0:
                    return
                break
        if streamer.poll() is not None:
            raise RuntimeError(f"streamer died — see {log_name}")
        time.sleep(2)
    raise RuntimeError("streamer never started publishing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="docs/example_video/tennis.mp4",
                    help="container-side path used as fake camera")
    ap.add_argument("--camera", type=int, default=None, help="real webcam index")
    ap.add_argument("--variant", default="low_latency")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--restart-streamer", action="store_true",
                    help="kill + relaunch the streamer halfway (reproduces the "
                         "user re-running run_live.sh against a live deploy)")
    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    ensure_container()
    kill_deploy_and_wait()
    assert_dds_clear()
    subprocess.run(["docker", "rm", "-f", "w2m-stream"], capture_output=True)

    sim_proc = start_sim_server(LOGS / "sim_stdout.log")
    ctrl = dp = streamer = None
    ok = False
    try:
        ctrl = SimCtrl()
        wait_sim_ready(ctrl)
        print("[m2] sim ready")

        dp = ZmqDeployProc(args.variant,
                           f"{DEPLOY_WS_CONTAINER}/logs/webcam2motion_m2",
                           LOGS / "deploy_stdout.log")
        dp.start()
        dp.wait_event("init_done", timeout=720)
        dp.send_key("]")
        dp.wait_event("control_started", timeout=15)
        ctrl.call("drop")
        time.sleep(4)
        print("[m2] robot dropped; starting live streamer (model load ~30s)...")

        streamer = start_streamer(args)
        wait_publishing(streamer, "streamer_stdout.log")

        dp.send_key("\n")
        dp.wait_event("stream_enabled", timeout=10)
        proto = dp.wait_event("protocol_established", timeout=30)
        print(f"[m2] streaming enabled, protocol v{proto}")

        falls0 = ctrl.call("state").get("fall_count", 0)
        phases = ["phase1", "phase2"] if args.restart_streamer else ["phase1"]
        phase_ok = []
        for phase in phases:
            if phase == "phase2":
                print("[m2] === restarting streamer against the live deploy ===")
                subprocess.run(["docker", "rm", "-f", "w2m-stream"], capture_output=True)
                streamer.wait(timeout=30)
                time.sleep(2)
                streamer = start_streamer(args, "streamer2_stdout.log")
                wait_publishing(streamer, "streamer2_stdout.log")
            recorder = StreamRecorder(LOGS / f"stream_{phase}.npz")
            recorder.start()
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.duration:
                time.sleep(2)
                s = ctrl.call("state")
                print(f"[m2] {phase} t={time.monotonic()-t0:5.1f}s "
                      f"height={s.get('height'):.3f} falls={s.get('fall_count')}")
                if streamer.poll() is not None:
                    raise RuntimeError("streamer exited early")
            recorder.stop()
            phase_ok.append(analyze_tracking(LOGS / f"stream_{phase}.npz"))
        falls = ctrl.call("state").get("fall_count", 0) - falls0

        ok = falls == 0 and not dp.stream_events and all(phase_ok)
        print(f"[m2] RESULT: {'PASS' if ok else 'FAIL'} (falls={falls}, "
              f"stream_errors={len(dp.stream_events)}, "
              f"arm_tracking={['ok' if p else 'poor' for p in phase_ok]})")
        tail = (LOGS / "streamer_stdout.log").read_text(errors="replace").splitlines()
        for line in tail[-4:]:
            print(f"[m2] streamer: {line}")
    finally:
        subprocess.run(["docker", "rm", "-f", "w2m-stream"], capture_output=True)
        if dp:
            dp.stop()
        stop_sim_server(sim_proc, ctrl)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
