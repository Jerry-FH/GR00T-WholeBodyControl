"""M2: GVHMR sliding-window causal estimator for live streaming.

Runs inside the webcam2motion container with CWD=/opt/GVHMR (relative ckpts).
Per frame:
  YOLO person box (every `yolo_period` frames, else reuse)
    -> get_batch(np frame, bbx) -> 256x256 crop tensor (shared by both nets)
    -> ViTPose kp2d (17,3)  +  HMR2 ViT feature (1024,)   [cached in ring buffers]
    -> GVHMR motion head over the last `window` frames (RoPE, static_cam)
    -> last frame of smpl_params_global -> {"body_pose": (63,), "global_orient": (3,)}

Returns None when no confident person detection (caller pauses publishing).
"""

import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, "/opt/GVHMR/tools/demo")

CONF_THR_DET = 0.5      # YOLO person confidence
CONF_THR_KP = 0.35      # mean ViTPose confidence over core joints
CORE_JOINTS = [5, 6, 11, 12]  # COCO17 shoulders + hips


class GVHMRStreamEstimator:
    def __init__(self, device: str = "cuda", window: int = 96, min_window: int = 16,
                 yolo_period: int = 1, flip_test: bool = True, postproc: bool = True,
                 autocast: bool = False, fp16: bool = False, hands: bool = False,
                 verbose_timing: bool = False):
        assert device == "cuda"
        self.window = window
        self.min_window = min_window
        self.yolo_period = yolo_period
        self.verbose_timing = verbose_timing
        self._flip_test = flip_test
        self._postproc = postproc
        self._autocast = autocast
        self._fp16 = fp16  # fp16 autocast for the ViT front-end (kp2d + feature)

        self.hand_tracker = None
        # VIDEO-mode tracking is cheap after the first detection and needs a
        # continuous frame stream to stay locked — run every estimate.
        self.hand_period = 1
        self._last_hands: dict = {}
        if hands:
            from estimators.hand_tracker import HandTracker
            self.hand_tracker = HandTracker()

        import hydra
        from demo import parse_args_to_cfg  # noqa: F401 (registers hydra store)
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy, normalize_kp2d
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
        from hmr4d.utils.preproc import Extractor, VitPoseExtractor
        from hmr4d.utils.preproc.vitfeat_extractor import get_batch
        from ultralytics import YOLO

        self._keypoints_from_heatmaps = keypoints_from_heatmaps

        self._estimate_K = estimate_K
        self._get_bbx_xys_from_xyxy = get_bbx_xys_from_xyxy
        self._normalize_kp2d = normalize_kp2d
        self._compute_cam_angvel = compute_cam_angvel
        self._get_batch = get_batch

        # hydra cfg via the demo's own parser (guaranteed-compatible overrides);
        # the --video arg only anchors output paths, never read in streaming.
        argv_bak = sys.argv
        sys.argv = ["gvhmr_runner", "--video", "docs/example_video/tennis.mp4", "-s",
                    "--output_root", "/tmp/gvhmr_live"]
        cfg = parse_args_to_cfg()
        sys.argv = argv_bak

        print("[gvhmr] loading models...")
        self.model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        self.model.load_pretrained_model(cfg.ckpt_path)
        self.model = self.model.eval().cuda()
        self.yolo = YOLO("inputs/checkpoints/yolo/yolov8x.pt")
        self.vitpose = VitPoseExtractor(tqdm_leave=False)
        self.vitpose.flip_test = self._flip_test  # False halves kp2d latency
        self.extractor = Extractor(tqdm_leave=False)
        print("[gvhmr] models ready")

        self._buf_kp2d = deque(maxlen=window)
        self._buf_bbx = deque(maxlen=window)
        self._buf_feat = deque(maxlen=window)
        self._K = None          # (3,3) estimated from full-frame size
        self._last_bbx_xys = None
        self._last_xyxy = None  # subject lock-on state
        self._frame_i = 0
        self._miss_streak = 0

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """IoU of one box `a` (4,) against boxes `b` (N,4), xyxy."""
        x0 = np.maximum(a[0], b[:, 0]); y0 = np.maximum(a[1], b[:, 1])
        x1 = np.minimum(a[2], b[:, 2]); y1 = np.minimum(a[3], b[:, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / (area_a + area_b - inter + 1e-9)

    def _select_subject(self, xyxy_all: np.ndarray, conf_all: np.ndarray) -> np.ndarray | None:
        """Stick to the tracked subject: prefer the detection overlapping the
        previous box (bystanders walking through don't steal the crop); fall
        back to highest confidence only when there is no locked subject."""
        if self._last_xyxy is not None:
            ious = self._iou(self._last_xyxy, xyxy_all)
            k = int(ious.argmax())
            if ious[k] >= 0.2:
                # EMA smooths crop jitter without lagging real movement much
                box = 0.7 * self._last_xyxy + 0.3 * xyxy_all[k]
                self._last_xyxy = box
                return box
            return None  # subject not among detections (occluded/left frame)
        box = xyxy_all[int(conf_all.argmax())]
        self._last_xyxy = box
        return box

    def reset(self):
        self._buf_kp2d.clear()
        self._buf_bbx.clear()
        self._buf_feat.clear()
        self._last_bbx_xys = None
        self._last_xyxy = None
        self._miss_streak = 0

    @torch.no_grad()
    def _extract_kp2d(self, imgs: torch.Tensor, bbx_xys: torch.Tensor) -> torch.Tensor:
        """VitPoseExtractor.extract single-batch path with optional fp16 forward
        (the stock extract() has no autocast hook; postprocess stays fp32)."""
        imgs_batch = imgs[:, :, :, 32:224].cuda()
        with torch.autocast("cuda", enabled=self._fp16, dtype=torch.float16):
            heatmap = self.vitpose.pose(imgs_batch)
            if self._flip_test:
                from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
                heatmap_f = flip_heatmap_coco17(self.vitpose.pose(imgs_batch.flip(3)))
                heatmap = (heatmap + heatmap_f) * 0.5
        heatmap = heatmap.float().cpu().numpy()
        center = bbx_xys[:, :2].numpy()
        scale = (torch.cat((bbx_xys[:, [2]] * 24 / 32, bbx_xys[:, [2]]), dim=1) / 200).numpy()
        preds, maxvals = self._keypoints_from_heatmaps(
            heatmaps=heatmap, center=center, scale=scale, use_udp=True)
        return torch.from_numpy(np.concatenate((preds, maxvals), axis=-1))

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray, t: float) -> dict | None:
        t0 = time.monotonic()
        H, W = frame_bgr.shape[:2]
        if self._K is None:
            self._K = self._estimate_K(W, H)
        frame_rgb = frame_bgr[..., ::-1].copy()

        # --- person detection (throttled) ---
        run_yolo = (self._frame_i % self.yolo_period == 0) or self._last_bbx_xys is None
        self._frame_i += 1
        if run_yolo:
            det = self.yolo(frame_rgb, classes=[0], conf=CONF_THR_DET, verbose=False)[0]
            box = None
            if len(det.boxes) > 0:
                box = self._select_subject(det.boxes.xyxy.cpu().numpy(),
                                           det.boxes.conf.cpu().numpy())
            if box is None:
                self._miss_streak += 1
                if self._miss_streak >= 5:
                    self.reset()
                return None
            xyxy = torch.from_numpy(np.asarray(box, dtype=np.float32))[None]  # (1,4)
            self._last_bbx_xys = self._get_bbx_xys_from_xyxy(xyxy, base_enlarge=1.2).float()
            self._miss_streak = 0
        bbx_xys = self._last_bbx_xys  # (1,3)
        t1 = time.monotonic()

        # --- shared 256x256 crop -> kp2d + ViT feature (fp16-capable) ---
        imgs, _ = self._get_batch(frame_rgb[None], bbx_xys, img_ds=1.0, path_type="np")
        kp2d = self._extract_kp2d(imgs, bbx_xys)  # (1,17,3) full-image px coords
        if float(kp2d[0, CORE_JOINTS, 2].mean()) < CONF_THR_KP:
            self._miss_streak += 1
            if self._miss_streak >= 5:
                self.reset()
            return None
        self._miss_streak = 0
        if self.hand_tracker is not None and self._frame_i % self.hand_period == 0:
            self._last_hands = self.hand_tracker.track(frame_rgb, kp2d[0].numpy())
        hands = self._last_hands
        t2 = time.monotonic()
        with torch.autocast("cuda", enabled=self._fp16, dtype=torch.float16):
            feat = self.extractor.extractor({"img": imgs.cuda()}).float().cpu()  # (1,1024)
        t3 = time.monotonic()

        self._buf_kp2d.append(kp2d[0])
        self._buf_bbx.append(bbx_xys[0])
        self._buf_feat.append(feat[0])
        L = len(self._buf_kp2d)
        if L < self.min_window:
            return None

        # --- GVHMR motion head over the window (predict() minus fixed
        #     postproc=True: pp transl + process_ik are skippable — we zero
        #     transl anyway and take raw decoder body_pose) ---
        kp2d_w = torch.stack(list(self._buf_kp2d))
        bbx_w = torch.stack(list(self._buf_bbx))
        batch = {
            "length": torch.tensor(L)[None],
            "obs": self._normalize_kp2d(kp2d_w, bbx_w)[None],
            "bbx_xys": bbx_w[None],
            "K_fullimg": self._K[None, None].repeat(1, L, 1, 1),
            "cam_angvel": self._compute_cam_angvel(torch.eye(3).repeat(L, 1, 1))[None],
            "f_imgseq": torch.stack(list(self._buf_feat))[None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.autocast("cuda", enabled=self._autocast, dtype=torch.float16):
            outputs = self.model.pipeline.forward(
                batch, train=False, postproc=self._postproc, static_cam=True)
        params = {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()}
        t4 = time.monotonic()

        if self.verbose_timing:
            print(f"[gvhmr] yolo={1e3*(t1-t0):.0f}ms crop+kp={1e3*(t2-t1):.0f}ms "
                  f"feat={1e3*(t3-t2):.0f}ms head(L={L})={1e3*(t4-t3):.0f}ms "
                  f"total={1e3*(t4-t0):.0f}ms")

        return {
            "body_pose": params["body_pose"][-1].float().cpu().numpy().reshape(63),
            "global_orient": params["global_orient"][-1].float().cpu().numpy().reshape(3),
            "confidence": float(kp2d[0, :, 2].mean()),
            "hands": hands,  # {side: {landmarks (21,3) px, wrist_angles (3,)}}
            # debug/preview extras
            "kp2d": kp2d[0].numpy(),                   # (17,3) full-image px + conf
            "bbx_xys": bbx_xys[0].numpy(),             # (3,) center xy + size
            "stage_ms": {"yolo": 1e3 * (t1 - t0), "kp2d+hands": 1e3 * (t2 - t1),
                         "feat": 1e3 * (t3 - t2), "head": 1e3 * (t4 - t3),
                         "total": 1e3 * (t4 - t0)},
        }
