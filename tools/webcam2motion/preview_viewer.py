"""Host-side preview window for the live streamer.

The container publishes overlay JPEGs (bbox + skeleton + latency HUD) on
tcp://*:5559 topic "preview"; this just shows the latest one. Run on the host
(needs GUI OpenCV, e.g. .venv_sim):

  .venv_sim/bin/python tools/webcam2motion/preview_viewer.py
  q / ESC to quit.
"""

import argparse

import cv2
import numpy as np
import zmq

TOPIC = b"preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5559)
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
    sock.setsockopt(zmq.CONFLATE, 1)  # always show the latest frame
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(f"[viewer] waiting for preview on tcp://{args.host}:{args.port} ... (q to quit)")

    try:
        while True:
            try:
                msg = sock.recv()
            except zmq.error.Again:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            img = cv2.imdecode(np.frombuffer(msg[len(TOPIC):], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imshow("webcam2motion preview", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
