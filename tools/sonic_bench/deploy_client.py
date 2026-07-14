"""Manage the g1_deploy_onnx_ref process inside the g1-deploy-dev container.

The binary's SimpleKeyboard handler reads single chars from stdin with a
non-blocking read() (termios errors are ignored), so a plain pipe through
`docker exec -i` is enough for key injection — no pty needed.

A reader thread tees stdout to a log file and turns known markers into events
consumable via wait_event().
"""

import queue
import subprocess
import threading
import time
from pathlib import Path

from common import (
    DEPLOY_BINARY,
    DEPLOY_WS_CONTAINER,
    DDS_INTERFACE,
    MOTION_DATA_REL,
    PLANNER_REL,
    RE_CLEAN_EXIT,
    RE_CONTROL_STARTED,
    RE_INIT_DONE,
    RE_LOOP_TIMING,
    RE_MOTION_COMPLETED,
    RE_MOTION_LOADED,
    RE_TRT_CONVERTED,
    VARIANTS,
    docker_exec_cmd,
    kill_deploy,
)


class DeployProc:
    def __init__(
        self,
        variant: str,
        logs_dir_container: str,
        stdout_log: Path,
        policy_precision: int = 32,
        zmq_out_port: int = 5557,
    ):
        v = VARIANTS[variant]
        inner = (
            f"exec {DEPLOY_BINARY} {DDS_INTERFACE} {v['decoder']} {MOTION_DATA_REL} "
            f"--obs-config {v['obs_config']} "
            f"--encoder-file {v['encoder']} "
            f"--planner-file {PLANNER_REL} "
            f"--input-type keyboard --output-type zmq "
            f"--zmq-out-port {zmq_out_port} --zmq-out-topic g1_debug "
            f"--disable-crc-check --enable-csv-logs "
            f"--logs-dir {logs_dir_container} "
            f"--policy-precision {policy_precision}"
        )
        self.cmd = docker_exec_cmd(inner)
        self.stdout_log = stdout_log
        self.events: queue.Queue = queue.Queue()
        self.motions: list[tuple[str, int]] = []  # (name, timesteps) in load order
        self.timing_samples: list[dict] = []
        self.trt_rebuilt = False
        self.proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        with open(self.stdout_log, "wb") as log:
            for raw in self.proc.stdout:
                log.write(raw)
                log.flush()
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
                self._parse_line(line)
        self.events.put(("exited", self.proc.poll()))

    def _parse_line(self, line: str) -> None:
        m = RE_MOTION_LOADED.search(line)
        if m:
            self.motions.append((m.group(1), int(m.group(2))))
            self.events.put(("motion_loaded", m.group(1)))
            return
        if RE_INIT_DONE.search(line):
            self.events.put(("init_done", None))
            return
        if RE_CONTROL_STARTED.search(line):
            self.events.put(("control_started", None))
            return
        m = RE_MOTION_COMPLETED.search(line)
        if m:
            self.events.put(("motion_completed", (int(m.group(1)), m.group(2))))
            return
        if RE_TRT_CONVERTED.search(line):
            self.trt_rebuilt = True
            self.events.put(("trt_converted", None))
            return
        m = RE_LOOP_TIMING.search(line)
        if m:
            self.timing_samples.append(
                {
                    "lowstate_age_ms": float(m.group(1)),
                    "obs_us": int(m.group(2)),
                    "policy_us": int(m.group(3)),
                    "obs2motor_us": int(m.group(4)),
                    "post_us": int(m.group(5)),
                }
            )
            return
        if RE_CLEAN_EXIT.search(line):
            self.events.put(("clean_exit", None))

    # -- interaction --------------------------------------------------------
    def send_key(self, ch: str) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(ch.encode())
        self.proc.stdin.flush()

    def wait_event(self, name: str, timeout: float):
        """Block until an event `name` arrives; returns its payload.

        Raises TimeoutError on timeout and RuntimeError if the process exits first.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {name!r} after {timeout}s")
            try:
                ev, payload = self.events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if ev == name:
                return payload
            if ev == "exited":
                raise RuntimeError(f"deploy exited (code {payload}) while waiting for {name!r}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, graceful_timeout: float = 15.0) -> None:
        """Graceful 'o' exit, then pkill fallback. Always reaps the local client."""
        if self.proc is None:
            return
        if self.alive():
            try:
                self.send_key("o")
            except Exception:
                pass
            deadline = time.monotonic() + graceful_timeout
            while self.alive() and time.monotonic() < deadline:
                time.sleep(0.2)
        if self.alive():
            kill_deploy()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Standalone smoke test: start release deploy, start control, clean exit.
    # Requires sim_server running (Init Done needs LowState from the sim).
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "release"
    logs_dir = f"{DEPLOY_WS_CONTAINER}/logs/sonic_bench/_client_test"
    dp = DeployProc(variant, logs_dir, Path("/tmp/deploy_client_test.log"))
    kill_deploy()
    dp.start()
    try:
        dp.wait_event("init_done", timeout=720)
        print(f"init done; motions loaded: {len(dp.motions)}")
        dp.send_key("]")
        dp.wait_event("control_started", timeout=10)
        print("control started")
        time.sleep(3)
    finally:
        dp.stop()
        print(f"stopped; timing samples: {len(dp.timing_samples)}")
