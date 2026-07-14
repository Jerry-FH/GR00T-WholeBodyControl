"""M1: GVHMR inference without rendering (renders need SMPL pkl + chumpy; skip).

Mirrors /opt/GVHMR/tools/demo/demo.py __main__ up to torch.save(pred), then
stops — no render_incam/render_global. Run inside the webcam2motion container
with CWD=/opt/GVHMR (relative ckpt paths):

  docker run --rm --gpus all \
    -v <repo>:/workspace/gr00t-wbc:rw \
    -v <repo>/tools/webcam2motion/checkpoints:/opt/GVHMR/inputs/checkpoints:ro \
    -w /opt/GVHMR webcam2motion \
    python /workspace/gr00t-wbc/tools/webcam2motion/estimators/m1_infer.py \
      --video docs/example_video/tennis.mp4 -s \
      --output_root /workspace/gr00t-wbc/tools/webcam2motion/outputs
"""

import sys
from pathlib import Path

sys.path.insert(0, "/opt/GVHMR/tools/demo")

import hydra  # noqa: E402
import torch  # noqa: E402

from demo import load_data_dict, parse_args_to_cfg, run_preprocess  # noqa: E402
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: E402
from hmr4d.utils.net_utils import detach_to_cpu  # noqa: E402
from hmr4d.utils.pylogger import Log  # noqa: E402


def main():
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")

    run_preprocess(cfg)
    data = load_data_dict(cfg)

    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s "
                 f"for data-length={data['length'] / 30:.1f}s")
        torch.save(pred, paths.hmr4d_results)
    print(f"[m1_infer] results: {paths.hmr4d_results}")


if __name__ == "__main__":
    main()
