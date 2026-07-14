"""Shared constants and helpers for the SONIC release-vs-low_latency benchmark harness.

All paths are host-side unless suffixed _CONTAINER. The g1-deploy-dev container
bind-mounts gear_sonic_deploy/ at /workspace/g1_deploy with host networking, so
files written in-container appear on the host and ZMQ ports are shared.
"""

import re
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "gear_sonic_deploy"

CONTAINER = "g1-deploy-dev"
DEPLOY_WS_CONTAINER = "/workspace/g1_deploy"

DEPLOY_BINARY = "./target/release/g1_deploy_onnx_ref"
FREQ_TEST_BINARY = "./target/release/freq_test"
DDS_INTERFACE = "lo"  # deploy.sh sim -> TARGET=lo; matches sim wbc yaml (DOMAIN_ID 0, lo)

ZMQ_DEBUG_ADDR = "tcp://localhost:5557"
ZMQ_DEBUG_TOPIC = "g1_debug"
ZMQ_CONFIG_TOPIC = "robot_config"
SIM_CTRL_PORT = 5560
SIM_CTRL_ADDR = f"tcp://127.0.0.1:{SIM_CTRL_PORT}"

MOTION_DATA_REL = "reference/example/"
RESULTS_REL = "logs/sonic_bench"  # under gear_sonic_deploy (bind-mounted)

VARIANTS = {
    "release": {
        "decoder": "policy/release/model_decoder.onnx",
        "encoder": "policy/release/model_encoder.onnx",
        "obs_config": "policy/release/observation_config.yaml",
    },
    "low_latency": {
        "decoder": "policy/low_latency/model_decoder.onnx",
        "encoder": "policy/low_latency/model_encoder.onnx",
        "obs_config": "policy/low_latency/observation_config.yaml",
    },
}
PLANNER_REL = "planner/target_vel/V2/planner_sonic.onnx"

# --- deploy binary stdout markers (verified against g1_deploy_onnx_ref.cpp) ---
RE_MOTION_LOADED = re.compile(r"✓ Loaded (\S+) \((\d+) timesteps\)")
RE_INIT_DONE = re.compile(r"^Init Done")
RE_CONTROL_STARTED = re.compile(r"transitioning to CONTROL state")
RE_MOTION_COMPLETED = re.compile(r"Motion index: (\d+) : (\S+) completed\.")
RE_TRT_CONVERTED = re.compile(r"Successfully converted ONNX to TRT")
RE_CLEAN_EXIT = re.compile(r"Program exiting normally")
# Loop timing - LowState age: Xms, ..., Obs: Xus, Policy: Xus, Obs 2 Motor Command: Xus, Post processing: Xus
RE_LOOP_TIMING = re.compile(
    r"Loop timing - LowState age: ([\d.eE+-]+)ms.*?"
    r"Obs: (\d+)us, Policy: (\d+)us, Obs 2 Motor Command: (\d+)us, Post processing: (\d+)us"
)
RE_FREQ_TEST = re.compile(r"Average time per inference:\s*([\d.]+)")

# Sim-side fall warning (base_sim.py check_fall)
RE_SIM_FALL = re.compile(r"Warning: Robot has fallen")

# --- G1 29-DOF joint groups, MuJoCo order (legs 2x6, waist 3, arms 2x7) ---
JOINT_GROUPS = {
    "legs": list(range(0, 12)),
    "waist": list(range(12, 15)),
    "arms": list(range(15, 29)),
}
NUM_BODY_JOINTS = 29

CONTROL_DT = 0.02  # 50 Hz control loop
OVERRUN_DT_MS = 22.0  # 50 Hz + 10% margin

# Landing sanity window after elastic-band release (pelvis z, meters).
# Spawn height 0.793 (g1_29dof_with_hand.xml); fallen threshold 0.2 (base_sim.py).
LAND_HEIGHT_MIN = 0.55
LAND_HEIGHT_MAX = 0.95


def kill_deploy(timeout: float = 10.0) -> None:
    """Kill any g1_deploy_onnx_ref left inside the container.

    Killing the local `docker exec` client orphans the in-container process,
    so teardown must always pkill inside the container.
    """
    subprocess.run(
        ["docker", "exec", CONTAINER, "pkill", "-f", "g1_deploy_onnx_ref"],
        capture_output=True,
        timeout=timeout,
    )


def deploy_running() -> bool:
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "pgrep", "-f", "g1_deploy_onnx_ref"],
        capture_output=True,
        timeout=10,
    )
    return r.returncode == 0


def port_free(port: int) -> bool:
    """True if nothing is listening on the TCP port (host network shared with container)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def docker_exec_cmd(inner: str) -> list[str]:
    """Build a docker exec command that sets up the deploy env and runs `inner`.

    setup_env.sh is what deploy.sh sources before `just run`; it exports the
    LD_LIBRARY_PATH (TensorRT/CUDA/onnxruntime) and FastRTPS profile the binary
    needs at runtime.
    """
    script = (
        f"cd {DEPLOY_WS_CONTAINER} && "
        "source scripts/setup_env.sh >/dev/null 2>&1; "
        f"{inner}"
    )
    return ["docker", "exec", "-i", CONTAINER, "bash", "-c", script]
