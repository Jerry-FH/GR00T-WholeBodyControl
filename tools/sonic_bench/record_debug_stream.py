"""Record the deploy binary's g1_debug ZMQ stream to stream.npz.

Wire format (zmq_output_handler.hpp): single-part message = topic prefix bytes
("g1_debug" / "robot_config") followed directly by a msgpack map. 50 Hz, PUB
with HWM=10 — the recorder keeps per-message work minimal and saves at stop.

Usable as a thread (StreamRecorder) inside run_benchmark, or standalone:
    python record_debug_stream.py --out /tmp/stream.npz --seconds 10
"""

import json
import threading
from pathlib import Path

import msgpack
import numpy as np
import zmq

from common import ZMQ_CONFIG_TOPIC, ZMQ_DEBUG_ADDR, ZMQ_DEBUG_TOPIC

# Fixed-size numeric fields worth keeping for analysis
KEEP_KEYS = [
    "index",
    "ros_timestamp",
    "base_quat",
    "base_ang_vel",
    "body_q",
    "body_dq",
    "last_action",
    "base_trans_target",
    "base_quat_target",
    "body_q_target",
    "base_quat_measured",
    "body_q_measured",
]

TOPIC_DEBUG = ZMQ_DEBUG_TOPIC.encode()
TOPIC_CONFIG = ZMQ_CONFIG_TOPIC.encode()


class StreamRecorder:
    def __init__(self, out_npz: Path, addr: str = ZMQ_DEBUG_ADDR):
        self.out_npz = Path(out_npz)
        self.addr = addr
        self.columns: dict[str, list] = {k: [] for k in KEEP_KEYS}
        self.robot_config: dict | None = None
        self.n_messages = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._save()

    def _run(self) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(self.addr)
        sock.setsockopt(zmq.SUBSCRIBE, TOPIC_DEBUG)
        sock.setsockopt(zmq.SUBSCRIBE, TOPIC_CONFIG)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if not poller.poll(timeout=200):
                    continue
                msg = sock.recv()
                if msg.startswith(TOPIC_CONFIG):
                    if self.robot_config is None:
                        try:
                            self.robot_config = msgpack.unpackb(
                                msg[len(TOPIC_CONFIG):], raw=False
                            )
                        except Exception:
                            pass
                    continue
                if not msg.startswith(TOPIC_DEBUG):
                    continue
                try:
                    data = msgpack.unpackb(msg[len(TOPIC_DEBUG):], raw=False)
                except Exception:
                    continue
                self.n_messages += 1
                for k in KEEP_KEYS:
                    v = data.get(k)
                    if v is not None:
                        self.columns[k].append(v)
        finally:
            sock.close(0)
            ctx.term()

    def _save(self) -> None:
        arrays = {}
        for k, rows in self.columns.items():
            if rows:
                try:
                    arrays[k] = np.asarray(rows, dtype=np.float64)
                except Exception:
                    pass
        self.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out_npz, **arrays)
        if self.robot_config is not None:
            cfg_path = self.out_npz.with_name("robot_config.json")
            cfg_path.write_text(json.dumps(self.robot_config, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/stream.npz"))
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    rec = StreamRecorder(args.out)
    rec.start()
    time.sleep(args.seconds)
    rec.stop()
    print(f"recorded {rec.n_messages} messages -> {args.out}")
