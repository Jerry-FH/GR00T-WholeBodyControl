"""Orchestrate the SONIC release-vs-low_latency MuJoCo sim2sim benchmark.

For each variant x motion x trial: fresh headless sim + fresh deploy process,
scripted key injection, ZMQ/CSV capture, teardown. See tools/sonic_bench/README.md.

Run from the repo root with the sim venv:
    .venv_sim/bin/python tools/sonic_bench/run_benchmark.py \
        --variants release,low_latency --motions all --trials 3
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zmq

from common import (
    DEPLOY_DIR,
    DEPLOY_WS_CONTAINER,
    FREQ_TEST_BINARY,
    LAND_HEIGHT_MAX,
    LAND_HEIGHT_MIN,
    RE_FREQ_TEST,
    RESULTS_REL,
    SIM_CTRL_ADDR,
    SIM_CTRL_PORT,
    VARIANTS,
    deploy_running,
    docker_exec_cmd,
    kill_deploy,
    port_free,
)
from deploy_client import DeployProc
from record_debug_stream import StreamRecorder

REPO_ROOT = DEPLOY_DIR.parent
SIM_SERVER = Path(__file__).resolve().parent / "sim_server.py"

INIT_TIMEOUT_S = 90.0
INIT_TIMEOUT_WARMUP_S = 720.0  # first run per variant may rebuild TRT engines
NAV_KEY_INTERVAL_S = 0.3
LANDING_SETTLE_S = 4.0
POLICY_STABILIZE_S = 3.0


class SimCtrl:
    """REQ client for sim_server.py."""

    def __init__(self, addr: str = SIM_CTRL_ADDR):
        self.ctx = zmq.Context()
        self.addr = addr
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock is not None:
            self.sock.close(0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 5000)
        self.sock.setsockopt(zmq.SNDTIMEO, 5000)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, cmd: str) -> dict:
        try:
            self.sock.send_string(json.dumps({"cmd": cmd}))
            return json.loads(self.sock.recv())
        except zmq.error.ZMQError as e:
            self._connect()  # REQ socket is stuck after a timeout; rebuild
            raise TimeoutError(f"sim ctrl {cmd!r} failed: {e}") from e

    def close(self):
        self.sock.close(0)
        self.ctx.term()


def start_sim_server(sim_log: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(SIM_SERVER), "--port", str(SIM_CTRL_PORT)],
        stdout=open(sim_log, "wb"),
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    return proc


def wait_sim_ready(ctrl: SimCtrl, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if ctrl.call("ping").get("ok"):
                return
        except TimeoutError:
            pass
        time.sleep(0.5)
    raise TimeoutError("sim_server did not become ready")


def stop_sim_server(proc: subprocess.Popen, ctrl: SimCtrl | None) -> None:
    if ctrl is not None:
        try:
            ctrl.call("quit")
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def read_played_motions(trial_dir: Path) -> list[str]:
    """Names logged to motion_name.csv by the state logger (post-hoc check)."""
    f = trial_dir / "motion_name.csv"
    if not f.exists():
        return []
    names = []
    for line in f.read_text(errors="replace").splitlines()[1:]:
        name = line.strip().split(",")[-1].strip('"')
        if name and name not in names:
            names.append(name)
    return names


def run_trial(
    variant: str,
    motion_name: str,
    trial_dir_host: Path,
    trial_dir_container: str,
    init_timeout: float,
    policy_precision: int,
) -> dict:
    """One full trial. Returns trial_meta dict (also written to trial_meta.json)."""
    trial_dir_host.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "variant": variant,
        "motion": motion_name,
        "status": "unknown",
        "fall_count": 0,
        "trt_rebuilt": False,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    sim_proc = None
    ctrl = None
    recorder = None
    deploy = None
    try:
        # 1. preflight
        kill_deploy()
        t0 = time.monotonic()
        while deploy_running() and time.monotonic() - t0 < 10:
            time.sleep(0.5)
        for port in (5557, SIM_CTRL_PORT):
            if not port_free(port):
                raise RuntimeError(f"port {port} still in use during preflight")

        # 2. sim
        sim_proc = start_sim_server(trial_dir_host / "sim_stdout.log")
        ctrl = SimCtrl()
        wait_sim_ready(ctrl)

        # 3. recorder before deploy (SUB before PUB bind is fine)
        recorder = StreamRecorder(trial_dir_host / "stream.npz")
        recorder.start()

        # 4. deploy
        deploy = DeployProc(
            variant,
            trial_dir_container,
            trial_dir_host / "deploy_stdout.log",
            policy_precision=policy_precision,
        )
        deploy.start()
        deploy.wait_event("init_done", timeout=init_timeout)
        meta["motions_loaded"] = len(deploy.motions)
        motion_names = [m[0] for m in deploy.motions]
        if motion_name not in motion_names:
            raise RuntimeError(f"motion {motion_name!r} not in loaded set: {motion_names}")
        motion_idx = motion_names.index(motion_name)
        timesteps = deploy.motions[motion_idx][1]
        motion_timeout = timesteps / 50.0 + 30.0

        # 5. start policy while suspended
        deploy.send_key("]")
        deploy.wait_event("control_started", timeout=10)
        time.sleep(POLICY_STABILIZE_S)

        # 6. drop to ground
        ctrl.call("drop")
        time.sleep(LANDING_SETTLE_S)
        st = ctrl.call("state")
        meta["landing_height"] = st.get("height")
        if not (st.get("ok") and st.get("fall_count") == 0
                and LAND_HEIGHT_MIN < st.get("height", 0) < LAND_HEIGHT_MAX):
            meta["status"] = "landing_failed"
            meta["fall_count"] = st.get("fall_count", -1)
            return meta

        # 7. navigate to target motion
        for _ in range(motion_idx):
            deploy.send_key("n")
            time.sleep(NAV_KEY_INTERVAL_S)

        # 8. play and monitor
        deploy.send_key("t")
        deadline = time.monotonic() + motion_timeout
        status = "timeout"
        while time.monotonic() < deadline:
            try:
                payload = deploy.wait_event("motion_completed", timeout=0.5)
                completed_idx, completed_name = payload
                meta["completed_motion"] = completed_name
                status = "success" if completed_name == motion_name else "wrong_motion"
                break
            except TimeoutError:
                pass
            except RuntimeError as e:  # deploy died
                meta["error"] = str(e)
                status = "deploy_died"
                break
            st = ctrl.call("state")
            if st.get("fall_count", 0) > 0:
                status = "fell"
                meta["fall_count"] = st["fall_count"]
                break
        meta["status"] = status
        return meta

    except (TimeoutError, RuntimeError) as e:
        meta["status"] = meta.get("status", "error") if meta.get("status") != "unknown" else "error"
        meta["error"] = str(e)
        return meta
    finally:
        # 9. teardown, always
        if deploy is not None:
            try:
                deploy.stop()
            except Exception:
                kill_deploy()
            meta["trt_rebuilt"] = deploy.trt_rebuilt
            meta["timing_samples"] = len(deploy.timing_samples)
            (trial_dir_host / "timing_samples.json").write_text(
                json.dumps(deploy.timing_samples)
            )
        if recorder is not None:
            try:
                recorder.stop()
                meta["stream_messages"] = recorder.n_messages
            except Exception:
                pass
        if sim_proc is not None:
            stop_sim_server(sim_proc, ctrl)
        if ctrl is not None:
            ctrl.close()
        # 10. post-hoc motion check
        played = read_played_motions(trial_dir_host)
        meta["played_motions"] = played
        if meta.get("status") == "success" and motion_name not in played:
            meta["status"] = "wrong_motion"
        meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
        (trial_dir_host / "trial_meta.json").write_text(json.dumps(meta, indent=2))


def run_freq_test(variant: str, out_file: Path, iters: int = 2000) -> float | None:
    """Isolated ONNX inference latency (CPU EP — see README caveat)."""
    cmd = docker_exec_cmd(
        f"{FREQ_TEST_BINARY} {VARIANTS[variant]['decoder']} {iters} random"
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        out_file.write_text("timeout")
        return None
    out_file.write_text(r.stdout + r.stderr)
    m = RE_FREQ_TEST.search(r.stdout)
    return float(m.group(1)) if m else None


def discover_motions() -> list[str]:
    """Motion names = subdirectories of reference/example/ (matches loader)."""
    d = DEPLOY_DIR / "reference" / "example"
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default="release,low_latency")
    ap.add_argument("--motions", default="all", help="'all' or comma-separated names")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--results-root", default=RESULTS_REL,
                    help="relative to gear_sonic_deploy/ (must stay under the bind mount)")
    ap.add_argument("--policy-precision", type=int, default=32, choices=[16, 32])
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--skip-freqtest", action="store_true")
    ap.add_argument("--retry-setup-failures", type=int, default=2,
                    help="retries for landing_failed/wrong_motion (setup issues, "
                         "not properties of the motion under test)")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant {v!r} (known: {list(VARIANTS)})")

    all_motions = discover_motions()
    if args.motions == "all":
        motions = all_motions
    else:
        motions = [m.strip() for m in args.motions.split(",") if m.strip()]
        unknown = [m for m in motions if m not in all_motions]
        if unknown:
            sys.exit(f"unknown motions {unknown}; available: {all_motions}")

    results_host = DEPLOY_DIR / args.results_root
    results_container = f"{DEPLOY_WS_CONTAINER}/{args.results_root}"
    results_host.mkdir(parents=True, exist_ok=True)

    print(f"[bench] variants={variants} motions={len(motions)} trials={args.trials}")
    print(f"[bench] results -> {results_host}")

    summary = []
    for variant in variants:
        vdir = results_host / variant
        vdir.mkdir(parents=True, exist_ok=True)

        if not args.skip_freqtest:
            print(f"[bench] {variant}: freq_test ...")
            avg = run_freq_test(variant, vdir / "freq_test.txt")
            print(f"[bench] {variant}: freq_test avg = {avg} us (CPU EP)")

        if not args.skip_warmup:
            print(f"[bench] {variant}: warm-up trial (up to 12 min if TRT rebuilds) ...")
            wdir = vdir / "_warmup"
            meta = run_trial(
                variant, motions[0], wdir,
                f"{results_container}/{variant}/_warmup",
                INIT_TIMEOUT_WARMUP_S, args.policy_precision,
            )
            print(f"[bench] {variant}: warm-up -> {meta['status']}"
                  f" (trt_rebuilt={meta.get('trt_rebuilt')})")

        for motion in motions:
            for k in range(1, args.trials + 1):
                label = f"{variant}/{motion}/trial{k}"
                tdir = vdir / motion / f"trial{k}"
                attempts = 0
                while True:
                    attempts += 1
                    print(f"[bench] run {label} (attempt {attempts}) ...", flush=True)
                    meta = run_trial(
                        variant, motion, tdir,
                        f"{results_container}/{variant}/{motion}/trial{k}",
                        INIT_TIMEOUT_S, args.policy_precision,
                    )
                    print(f"[bench] {label}: {meta['status']}"
                          + (f" ({meta.get('error')})" if meta.get("error") else ""),
                          flush=True)
                    if (meta["status"] in ("wrong_motion", "landing_failed")
                            and attempts <= args.retry_setup_failures):
                        continue
                    break
                summary.append(
                    {"trial": label, "status": meta["status"], "attempts": attempts}
                )

    kill_deploy()
    (results_host / "run_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if s["status"] == "success")
    print(f"[bench] done: {n_ok}/{len(summary)} successful trials")
    print(f"[bench] analyze with: .venv_sim/bin/python tools/sonic_bench/analyze.py "
          f"--results-root {results_host}")


if __name__ == "__main__":
    main()
