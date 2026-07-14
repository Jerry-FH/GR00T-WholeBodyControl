"""WiLoR hand backend: GPU MANO regression instead of MediaPipe's CPU tracker.

Why: MediaPipe HandLandmarker runs on CPU (~30 ms/frame via XNNPACK) and
competes with the sim + deploy processes, while the GPU still has headroom
next to GVHMR. WiLoR (wilor-mini package) regresses full MANO pose from a
224px hand crop on GPU (~25-40 ms fp16) and is markedly more robust to
motion blur / partial occlusion than landmark tracking.

Drop-in for HandTracker: same track(frame_rgb, kp2d) contract, and the
output landmarks use the SAME 21-point ordering — wilor-mini's MANO wrapper
reorders to OpenPose hand convention (mano_to_openpose map), which is
identical to MediaPipe's [wrist, thumb1-4, index1-4, middle1-4, ring1-4,
pinky1-4]. So rh56_from_landmarks / _wrist_angles / draw_hand_panel work
unchanged.

We feed our own ViTPose-derived wrist ROIs via predict_with_bboxes(), which
skips wilor-mini's built-in YOLO hand detector entirely (one less model, and
our ROIs already track the subject lock-on). The official MANO_RIGHT.pkl the
user registered for is pre-placed in WILOR_DIR/pretrained_models/ so the
pipeline's HuggingFace auto-download of a mirrored copy never triggers.
"""

import numpy as np

from estimators.hand_tracker import (
    HandTracker,
    L_ELBOW,
    L_WRIST,
    R_ELBOW,
    R_WRIST,
    hand_roi,
    rh56_from_landmarks,
)

# host: tools/webcam2motion/checkpoints/wilor (bind-mounted, survives rebuilds)
WILOR_DIR = "/opt/GVHMR/inputs/checkpoints/wilor"


class WilorHandTracker:
    def __init__(self, device: str = "cuda", roi_scale: float = 1.6,
                 model_dir: str = WILOR_DIR, rescale_factor: float = 1.2):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=torch.device(device), dtype=torch.float16, verbose=False,
            wilor_pretrained_dir=model_dir)
        self.roi_scale = roi_scale
        # our ROI is already ~1.6x the forearm; wilor's default 2.5 assumes
        # tight YOLO boxes and would blow the crop up too far
        self.rescale_factor = rescale_factor

    def track(self, frame_rgb: np.ndarray, kp2d: np.ndarray) -> dict:
        """Returns {side: {"landmarks": (21,3) px, "wrist_angles": (3,),
        "rh56": (6,)}} — same contract as HandTracker.track."""
        boxes, sides = [], []
        for side in ("left", "right"):
            roi = hand_roi(kp2d, side, frame_rgb.shape, self.roi_scale)
            if roi is not None:
                boxes.append(roi)
                sides.append(side)
        if not boxes:
            return {}
        preds = self.pipe.predict_with_bboxes(
            frame_rgb, np.asarray(boxes, dtype=np.float32),
            [1 if s == "right" else 0 for s in sides],
            rescale_factor=self.rescale_factor)
        out = {}
        for side, p in zip(sides, preds):  # output order follows input order
            w = p.get("wilor_preds") or {}
            if "pred_keypoints_2d" not in w:
                continue
            kp2 = np.asarray(w["pred_keypoints_2d"], dtype=np.float64).reshape(21, 2)
            kp3 = np.asarray(w["pred_keypoints_3d"], dtype=np.float64).reshape(21, 3)
            # mediapipe-style landmarks: x,y in full-image px + relative depth
            # scaled to px (metric mm -> px via the 2D/3D palm-size ratio; both
            # conventions have z more negative toward the camera)
            kp3 -= kp3[0]
            scale_2d = np.linalg.norm(kp2[9] - kp2[0])
            scale_3d = np.linalg.norm(kp3[9, :2]) + 1e-9
            lm = np.concatenate([kp2, kp3[:, 2:3] * (scale_2d / scale_3d)], axis=1)
            e, wr = (L_ELBOW, L_WRIST) if side == "left" else (R_ELBOW, R_WRIST)
            angles = HandTracker._wrist_angles(lm, kp2d[wr, :2] - kp2d[e, :2], side)
            out[side] = {"landmarks": lm, "wrist_angles": angles,
                         "rh56": rh56_from_landmarks(lm, side)}
        return out
