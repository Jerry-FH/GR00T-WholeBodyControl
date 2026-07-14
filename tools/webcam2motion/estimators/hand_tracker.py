"""Palm-orientation tracking from MediaPipe hand landmarks.

Why: COCO17 has only a wrist *point* — forearm pronation (palm up/down/in/out)
is unobservable to ViTPose, and monocular body-crop GVHMR leaves SMPL wrist
rotation near neutral (palms stuck facing inward). This module crops a hand
ROI around the ViTPose wrist keypoint, runs MediaPipe Hands (21 landmarks with
relative depth), and derives G1 wrist angles geometrically:

  forearm frame  (elbow->wrist image direction, roll-free convention)
  hand frame     (wrist->middle_mcp = x, palm normal = z)
  R_wrist = R_forearm^-1 @ R_hand  ->  XYZ euler -> G1 roll/pitch/yaw

Landmarks are also returned raw — they are the future Dex3 finger input.
Per-hand MediaPipe instances keep temporal tracking working on ROI streams.
"""

import numpy as np

# COCO17 indices
L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 7, 8, 9, 10
# MediaPipe hand landmark indices
MP_WRIST, MP_INDEX_MCP, MP_MIDDLE_MCP, MP_PINKY_MCP = 0, 5, 9, 17
# MediaPipe finger chains (wrist -> tip), used for both curl and drawing
MP_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "little": [0, 17, 18, 19, 20],
}

# Inspire RH56DFQ actuator order (SDK register order: little..thumb_rot).
# Values are normalized 0.0 = fully open .. 1.0 = fully closed/opposed;
# the future hand driver maps these onto the SDK's 0-1000 range + per-unit
# calibration.
RH56_ORDER = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

# G1 IsaacLab wrist joint order in joint_pos: [L_roll, R_roll, L_pitch, R_pitch, L_yaw, R_yaw]
G1_L_ROLL, G1_R_ROLL, G1_L_PITCH, G1_R_PITCH, G1_L_YAW, G1_R_YAW = 23, 24, 25, 26, 27, 28


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _chain_curl(lm: np.ndarray, chain: list, lo: float, hi: float) -> float:
    """Total bend along a finger chain (sum of inter-segment angles, rad),
    normalized to 0..1 by an empirical [open, closed] range."""
    total = 0.0
    for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
        v1 = _normalize(lm[b] - lm[a])
        v2 = _normalize(lm[c] - lm[b])
        total += float(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
    return float(np.clip((total - lo) / (hi - lo), 0.0, 1.0))


def rh56_from_landmarks(lm: np.ndarray, side: str) -> np.ndarray:
    """MediaPipe 21 landmarks -> RH56DFQ 6-DOF command, RH56_ORDER, 0..1.

    - four fingers + thumb bend: angle-chain curl
    - thumb rotation (opposition): thumb proximal direction swung across the
      palm, measured against the index->pinky knuckle axis in the palm plane
    """
    out = np.zeros(6, dtype=np.float32)
    # empirical open/closed angle-sum ranges (rad)
    for i, name in enumerate(("little", "ring", "middle", "index")):
        out[i] = _chain_curl(lm, MP_CHAINS[name], lo=0.6, hi=3.2)
    out[4] = _chain_curl(lm, MP_CHAINS["thumb"], lo=0.5, hi=2.0)

    # thumb rotation: angle between thumb metacarpal and the knuckle axis,
    # projected on the palm plane
    v1 = lm[MP_INDEX_MCP] - lm[MP_WRIST]
    v2 = lm[MP_PINKY_MCP] - lm[MP_WRIST]
    n = _normalize(np.cross(v1, v2))
    if side == "left":
        n = -n
    knuckle_axis = _normalize(lm[MP_INDEX_MCP] - lm[MP_PINKY_MCP])
    thumb_dir = lm[2] - lm[1]  # thumb metacarpal
    thumb_in_palm = _normalize(thumb_dir - np.dot(thumb_dir, n) * n)
    ang = float(np.arccos(np.clip(np.dot(thumb_in_palm, knuckle_axis), -1, 1)))
    # ~0.3 rad = thumb alongside palm (open), ~1.4 rad = full opposition
    out[5] = float(np.clip((ang - 0.3) / (1.4 - 0.3), 0.0, 1.0))
    return out


MODEL_PATH = "/opt/models/hand_landmarker.task"


class HandTracker:
    def __init__(self, roi_scale: float = 1.6, min_conf: float = 0.3,
                 model_path: str = MODEL_PATH):
        # mediapipe >=0.10.2x removed the legacy solutions API -> Tasks API.
        # VIDEO mode is essential: IMAGE mode re-runs full palm detection every
        # frame (flickery); VIDEO detects once then TRACKS landmarks across
        # frames — per-side instances see a consistent ROI stream, so tracking
        # holds. min_conf 0.3: the presence gate at 0.5 dropped valid frames.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = dict(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._hands = {
            side: vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(**opts))
            for side in ("left", "right")
        }
        self._ts_ms = {"left": 0, "right": 0}  # VIDEO mode needs monotonic ts
        self.roi_scale = roi_scale

    def _roi(self, kp2d: np.ndarray, side: str, frame_shape) -> tuple | None:
        """Square hand ROI: centered beyond the wrist along the forearm.

        The 2D forearm foreshortens when the arm points at the camera — a
        torso-proportional floor keeps the ROI from collapsing then."""
        e, w = (L_ELBOW, L_WRIST) if side == "left" else (R_ELBOW, R_WRIST)
        if kp2d[w, 2] < 0.3 or kp2d[e, 2] < 0.3:
            return None
        elbow, wrist = kp2d[e, :2], kp2d[w, :2]
        forearm = wrist - elbow
        flen = np.linalg.norm(forearm)
        if flen < 5:
            return None
        # torso scale: mid-shoulder to mid-hip (COCO17: 5,6 shoulders; 11,12 hips)
        torso = 0.0
        if kp2d[[5, 6, 11, 12], 2].min() > 0.3:
            torso = float(np.linalg.norm(
                (kp2d[5, :2] + kp2d[6, :2]) / 2 - (kp2d[11, :2] + kp2d[12, :2]) / 2))
        center = wrist + 0.35 * forearm  # hand extends past the wrist
        half = max(self.roi_scale * 0.5 * flen, 0.28 * torso, 28.0)
        H, W = frame_shape[:2]
        x0, y0 = int(max(center[0] - half, 0)), int(max(center[1] - half, 0))
        x1, y1 = int(min(center[0] + half, W)), int(min(center[1] + half, H))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _wrist_angles(lm: np.ndarray, elbow_to_wrist_2d: np.ndarray, side: str) -> np.ndarray:
        """(roll, pitch, yaw) of the hand relative to a roll-free forearm frame.

        lm: (21,3) landmarks in image px (z = mediapipe relative depth, px-scaled).
        Camera frame: x right, y down, z toward camera (mediapipe z is negative
        toward camera; we use it directly as a relative axis).
        """
        # forearm frame: x = elbow->wrist (in-plane), roll-free about camera z
        fx = _normalize(np.array([elbow_to_wrist_2d[0], elbow_to_wrist_2d[1], 0.0]))
        fz = _normalize(np.cross(fx, [0.0, 0.0, 1.0]))  # in-plane perpendicular
        fy = np.cross(fz, fx)
        R_f = np.stack([fx, fy, fz], axis=1)

        # hand frame: x = wrist->middle_mcp, z = palm normal
        hx = _normalize(lm[MP_MIDDLE_MCP] - lm[MP_WRIST])
        v1 = lm[MP_INDEX_MCP] - lm[MP_WRIST]
        v2 = lm[MP_PINKY_MCP] - lm[MP_WRIST]
        n = _normalize(np.cross(v1, v2))
        if side == "left":  # keep n = out of the palm for both hands
            n = -n
        hz = _normalize(n - np.dot(n, hx) * hx)
        hy = np.cross(hz, hx)
        R_h = np.stack([hx, hy, hz], axis=1)

        R = R_f.T @ R_h
        # intrinsic XYZ euler (matches the smpl_adapter wrist mapping convention)
        pitch = float(np.arcsin(np.clip(R[0, 2], -1, 1)))
        roll = float(np.arctan2(-R[1, 2], R[2, 2]))
        yaw = float(np.arctan2(-R[0, 1], R[0, 0]))
        return np.array([roll, pitch, yaw])

    def track(self, frame_rgb: np.ndarray, kp2d: np.ndarray) -> dict:
        """Returns {side: {"landmarks": (21,3) px, "wrist_angles": (3,)}} for
        each confidently detected hand."""
        out = {}
        for side in ("left", "right"):
            roi = self._roi(kp2d, side, frame_rgb.shape)
            if roi is None:
                continue
            x0, y0, x1, y1 = roi
            crop = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])
            h, w = crop.shape[:2]
            if max(h, w) > 256:  # mediapipe cost scales with input size
                import cv2
                s = 224.0 / max(h, w)
                crop = np.ascontiguousarray(
                    cv2.resize(crop, (max(int(w * s), 8), max(int(h * s), 8))))
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop)
            self._ts_ms[side] += 33  # VIDEO mode: monotonic per-instance clock
            res = self._hands[side].detect_for_video(mp_img, self._ts_ms[side])
            if not res.hand_landmarks:
                continue
            lm = np.array([[p.x * w + x0, p.y * h + y0, p.z * w]
                           for p in res.hand_landmarks[0]])
            e, wr = (L_ELBOW, L_WRIST) if side == "left" else (R_ELBOW, R_WRIST)
            angles = self._wrist_angles(lm, kp2d[wr, :2] - kp2d[e, :2], side)
            out[side] = {"landmarks": lm, "wrist_angles": angles,
                         "rh56": rh56_from_landmarks(lm, side)}
        return out


def wrist_override(hands: dict, joint_pos: np.ndarray,
                   blends: dict | None = None) -> np.ndarray:
    """Overwrite the G1 wrist ROLL (pronation — the axis COCO17 can't see)
    with the hand-landmark estimate; keep SMPL-derived pitch/yaw. Signs follow
    the smpl_adapter convention (right side roll negated).

    blends: per-side 0..1 fade weight (see WristBlender) — hard switching
    between the SMPL and hand-landmark values on detection flicker destabilizes
    the policy."""
    jp = joint_pos.copy()
    blends = blends or {}
    if "left" in hands:
        b = float(blends.get("left", 1.0))
        roll = float(np.clip(hands["left"]["wrist_angles"][0], -1.9, 1.9))
        jp[G1_L_ROLL] = (1 - b) * jp[G1_L_ROLL] + b * roll
    if "right" in hands:
        b = float(blends.get("right", 1.0))
        roll = float(np.clip(-hands["right"]["wrist_angles"][0], -1.9, 1.9))
        jp[G1_R_ROLL] = (1 - b) * jp[G1_R_ROLL] + b * roll
    return jp


RH56_LABELS = ["L", "R", "M", "I", "Tb", "Tr"]  # matches RH56_ORDER


def draw_hand_panel(img: np.ndarray, hand: dict, side: str, org: tuple,
                    size: int = 150) -> None:
    """Simulated-hand inset for the preview window: the landmark skeleton
    re-projected into a canonical palm-facing view (independent of the camera
    angle) plus the 6 RH56DFQ command bars."""
    import cv2
    x0, y0 = org
    h_panel = size + 46
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + h_panel), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + h_panel), (200, 200, 200), 1)
    cv2.putText(img, f"{side.upper()} (RH56)", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    lm = hand["landmarks"]
    # canonical palm-on view: hand frame = (x: wrist->middle_mcp, z: palm normal)
    hx = _normalize(lm[MP_MIDDLE_MCP] - lm[MP_WRIST])
    v1 = lm[MP_INDEX_MCP] - lm[MP_WRIST]
    v2 = lm[MP_PINKY_MCP] - lm[MP_WRIST]
    n = _normalize(np.cross(v1, v2))
    if side == "left":
        n = -n
    hz = _normalize(n - np.dot(n, hx) * hx)
    hy = np.cross(hz, hx)
    R = np.stack([hx, hy, hz], axis=1)
    pts = (lm - lm[MP_WRIST]) @ R  # rows: (along-hand, across-hand, out-of-palm)
    scale = (size * 0.8) / (np.abs(pts[:, :2]).max() + 1e-6)
    # panel coords: along-hand -> up, across-hand -> right
    px = (x0 + size // 2 + pts[:, 1] * scale * 0.5).astype(int)
    py = (y0 + 18 + size - 24 - pts[:, 0] * scale * 0.85).astype(int)
    for chain in MP_CHAINS.values():
        for a, b in zip(chain[:-1], chain[1:]):
            cv2.line(img, (px[a], py[a]), (px[b], py[b]), (0, 255, 255), 1)
    for j in range(21):
        cv2.circle(img, (px[j], py[j]), 2, (0, 128, 255), -1)

    # RH56 command bars
    rh56 = hand.get("rh56")
    if rh56 is not None:
        bw = (size - 14) // 6
        for i, (v, lbl) in enumerate(zip(rh56, RH56_LABELS)):
            bx = x0 + 7 + i * bw
            by1 = y0 + h_panel - 6
            bh = int(24 * float(v))
            cv2.rectangle(img, (bx, by1 - 24), (bx + bw - 3, by1), (80, 80, 80), 1)
            cv2.rectangle(img, (bx, by1 - bh), (bx + bw - 3, by1), (0, 200, 0), -1)
            cv2.putText(img, lbl, (bx, by1 - 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (255, 255, 255), 1)


class WristBlender:
    """Fade the hand-landmark override in on detection, HOLD it through
    detection dropouts, and clamp the per-tick wrist delta.

    Hold-not-fade: the SMPL wrist value is the *wrong* answer (near-neutral,
    palms inward), so falling back to it on every detection flicker looks like
    glitching. Instead the last hand-derived value is held for `hold_s`, then
    slowly faded out only on a genuine long absence."""

    def __init__(self, fade_in_s: float = 0.4, hold_s: float = 3.0,
                 fade_out_s: float = 1.5, max_delta: float = 0.06,
                 fps: float = 50.0):
        self._blend = {"left": 0.0, "right": 0.0}
        self._absent_ticks = {"left": 0, "right": 0}
        self._last: dict = {}  # last seen hand data per side (held during dropout)
        self._up = 1.0 / (fade_in_s * fps)
        self._down = 1.0 / (fade_out_s * fps)
        self._hold_ticks = int(hold_s * fps)
        self.max_delta = max_delta  # rad per 50Hz tick = 3 rad/s
        self._prev_wrists = None

    def apply(self, hands: dict, joint_pos: np.ndarray) -> np.ndarray:
        for side in ("left", "right"):
            if side in hands:
                self._last[side] = hands[side]
                self._absent_ticks[side] = 0
                self._blend[side] = min(1.0, self._blend[side] + self._up)
            else:
                self._absent_ticks[side] += 1
                if self._absent_ticks[side] > self._hold_ticks:  # hold, then fade
                    self._blend[side] = max(0.0, self._blend[side] - self._down)
        active = {s: v for s, v in self._last.items() if self._blend[s] > 0.0}
        jp = wrist_override(active, joint_pos, self._blend)
        wr = jp[23:29]
        if self._prev_wrists is not None:
            wr = self._prev_wrists + np.clip(wr - self._prev_wrists,
                                             -self.max_delta, self.max_delta)
        self._prev_wrists = wr
        jp[23:29] = wr
        return jp

    def held_hands(self) -> dict:
        """Sides whose (possibly held) hand data is still active — use this
        for the rh56 stream so finger commands also hold through dropouts."""
        return {s: v for s, v in self._last.items() if self._blend[s] > 0.3}
