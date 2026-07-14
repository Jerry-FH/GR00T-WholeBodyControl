"""Pose estimator plugins: frame(s) -> SMPL params.

Each estimator implements:
    estimate(frame_bgr: np.ndarray, t: float) -> dict | None
returning {"body_pose": (63,), "global_orient": (3,), "confidence": float}
in the SMPL y-up world convention (GVHMR 'global' params), or None when no
person is confidently detected (caller pauses publishing -> robot holds pose).
"""
