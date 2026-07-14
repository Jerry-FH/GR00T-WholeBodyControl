"""Headless MuJoCo sim for the SONIC benchmark, controllable over ZMQ REP.

Replicates the wiring of gear_sonic/scripts/run_sim_loop.py but runs the
simulator in a background thread (BaseSimulator.start_as_thread) and serves a
small JSON command channel in the main thread:

    {"cmd": "ping"}     -> {"ok": true}
    {"cmd": "drop"}     -> disable the elastic band (same as pressing '9')
    {"cmd": "band_on"}  -> re-enable the elastic band
    {"cmd": "state"}    -> {"ok": true, "height": <pelvis z>, "fall_count": N,
                            "cmd_received": bool, "sim_time": t}
    {"cmd": "reset"}    -> mj_resetData
    {"cmd": "quit"}     -> close sim and exit

Run with the repo's .venv_sim python from the repo root (or anywhere; the repo
root is put on sys.path explicitly).
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import zmq  # noqa: E402

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.utils.mujoco_sim.simulator_factory import (  # noqa: E402
    SimulatorFactory,
    init_channel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument(
        "--onscreen", action="store_true", help="show the MuJoCo viewer (debugging)"
    )
    args = parser.parse_args()

    config = SimLoopConfig(interface="sim", enable_onscreen=args.onscreen)
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name

    init_channel(config=wbc_config)
    sim = SimulatorFactory.create_simulator(
        config=wbc_config,
        env_name=config.env_name,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=False,
    )

    env = sim.sim_env

    # Count falls without changing behavior: check_fall prints a warning and
    # auto-resets whenever pelvis z < 0.2 (base_sim.py). Wrap it on the instance.
    fall_state = {"count": 0}
    original_check_fall = env.check_fall

    def counting_check_fall():
        if env.mj_data.qpos[2] < 0.2:
            fall_state["count"] += 1
        return original_check_fall()

    env.check_fall = counting_check_fall

    sim.start_as_thread()
    print(f"[sim_server] simulator thread started (onscreen={args.onscreen})", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[sim_server] control REP listening on tcp://127.0.0.1:{args.port}", flush=True)

    running = True
    try:
        while running:
            if not sock.poll(timeout=500):
                if sim.sim_thread is not None and not sim.sim_thread.is_alive():
                    print("[sim_server] sim thread died, exiting", flush=True)
                    break
                continue
            req = json.loads(sock.recv())
            cmd = req.get("cmd")
            reply = {"ok": True}
            try:
                if cmd == "ping":
                    pass
                elif cmd == "drop":
                    # Gentle drop: walk the band equilibrium (point + [0,0,length])
                    # down to ~stance height before releasing, so the landing
                    # impact never dips the pelvis below the 0.2 m fall threshold.
                    for _ in range(9):
                        env.elastic_band.length -= 0.02
                        time.sleep(0.08)
                    time.sleep(0.5)
                    env.elastic_band.enable = False
                    env.elastic_band.length = 0.0
                    print("[sim_server] elastic band lowered and released (drop)", flush=True)
                elif cmd == "band_on":
                    env.elastic_band.enable = True
                elif cmd == "state":
                    reply.update(
                        height=float(env.mj_data.qpos[2]),
                        fall_count=fall_state["count"],
                        cmd_received=bool(
                            env.unitree_bridge is not None
                            and env.unitree_bridge.low_cmd_received
                        ),
                        band_enabled=bool(env.elastic_band.enable),
                        sim_time=float(env.mj_data.time),
                    )
                elif cmd == "reset":
                    sim.reset()
                    fall_state["count"] = 0
                elif cmd == "quit":
                    running = False
                else:
                    reply = {"ok": False, "error": f"unknown cmd: {cmd}"}
            except Exception as e:  # keep REP cycle intact on any error
                reply = {"ok": False, "error": repr(e)}
            sock.send_string(json.dumps(reply))
    finally:
        print("[sim_server] closing simulator", flush=True)
        sim.close()
        sock.close(0)
        ctx.term()
        # sim thread's finally calls close() too; give it a moment then exit hard
        # (DDS threads are non-daemon and would otherwise keep the process alive)
        time.sleep(1.0)
        sys.stdout.flush()
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
