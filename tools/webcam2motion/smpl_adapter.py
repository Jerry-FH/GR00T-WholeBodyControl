"""SMPL params → SONIC v3 stream frames.

Conversion chain mirrors process_smpl_joints in
gear_sonic/scripts/pico_manager_thread_server.py:450 (which is single-frame
only due to a hardcoded global_orient_6d reshape) — reimplemented batched here
with the same repo primitives:
  global_orient --smpl_root_ytoz_up--> z-up
  compute_human_joints FK (canonical rest skeleton, betas ignored)
  remove_smpl_base_rot -> body_quat (wxyz)
  root-local smpl_joints
plus the G1 wrist mapping from the same file (~lines 1418-1476), vectorized.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUMAN_JOINTS_INFO = os.path.join(REPO_ROOT, "gear_sonic/data/human/human_joints_info.pkl")

# body_pose 21-joint indices (SMPL joints 1..21, zero-indexed minus pelvis)
SMPL_L_ELBOW, SMPL_R_ELBOW, SMPL_L_WRIST, SMPL_R_WRIST = 17, 18, 19, 20
# G1 IsaacLab-order wrist joint indices
G1_L_ROLL, G1_R_ROLL, G1_L_PITCH, G1_R_PITCH, G1_L_YAW, G1_R_YAW = 23, 24, 25, 26, 27, 28


def process_smpl_joints_batch(body_pose: torch.Tensor, global_orient: torch.Tensor) -> dict:
    """Batched equivalent of pico_manager_thread_server.process_smpl_joints
    (minus the unused global_orient_6d). body_pose (T,63+), global_orient (T,3)."""
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
        human_joints_info_path=HUMAN_JOINTS_INFO,
    )  # (T, 24, 3)

    global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)
    quat_inv_rep = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(quat_inv_rep, joints)

    return {
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
    }


def wrist_joint_pos(body_pose: np.ndarray) -> np.ndarray:
    """Map SMPL elbow/wrist rotations to G1 wrist joints.

    body_pose: (T, 21, 3) axis-angle. Returns (T, 29) with only wrist
    indices [23..28] populated (protocol v3 requirement).
    """
    T = body_pose.shape[0]
    joint_pos = np.zeros((T, 29), dtype=np.float32)

    def _safe_aa(aa):
        # decompose_rotation_aa divides by the rotation angle; nudge exact zeros
        aa = np.asarray(aa, dtype=np.float64).copy()
        zero = np.linalg.norm(aa, axis=-1) < 1e-8
        aa[zero] = [1e-8, 0.0, 0.0]
        return aa

    y_axis = np.array([0, 1, 0])
    _, l_swing = decompose_rotation_aa(_safe_aa(body_pose[:, SMPL_L_ELBOW]), y_axis)
    _, r_swing = decompose_rotation_aa(_safe_aa(body_pose[:, SMPL_R_ELBOW]), y_axis)

    l_swing_euler = R.from_quat(l_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
    r_swing_euler = R.from_quat(r_swing[:, [1, 2, 3, 0]]).as_euler("XYZ")
    l_wrist_euler = R.from_rotvec(body_pose[:, SMPL_L_WRIST]).as_euler("XYZ")
    r_wrist_euler = R.from_rotvec(body_pose[:, SMPL_R_WRIST]).as_euler("XYZ")

    # elbow roll/yaw folded into the wrist, wrist pitch straight from SMPL
    joint_pos[:, G1_L_ROLL] = l_swing_euler[:, 0] + l_wrist_euler[:, 0]
    joint_pos[:, G1_L_PITCH] = l_wrist_euler[:, 1]
    joint_pos[:, G1_L_YAW] = l_swing_euler[:, 2] + l_wrist_euler[:, 2]
    joint_pos[:, G1_R_ROLL] = -(r_swing_euler[:, 0] + r_wrist_euler[:, 0])
    joint_pos[:, G1_R_PITCH] = -r_wrist_euler[:, 1]
    joint_pos[:, G1_R_YAW] = r_swing_euler[:, 2] + r_wrist_euler[:, 2]

    # euler composition can wrap past the G1 wrist limits on extreme poses
    roll_cols = [G1_L_ROLL, G1_R_ROLL]
    py_cols = [G1_L_PITCH, G1_R_PITCH, G1_L_YAW, G1_R_YAW]
    joint_pos[:, roll_cols] = np.clip(joint_pos[:, roll_cols], -1.9, 1.9)
    joint_pos[:, py_cols] = np.clip(joint_pos[:, py_cols], -1.6, 1.6)
    return joint_pos


def smpl_to_stream_frames(body_pose: np.ndarray, global_orient: np.ndarray,
                          device: str = "cpu") -> list[dict]:
    """Convert an SMPL sequence into a list of per-frame stream dicts.

    body_pose: (T, 63) or (T, 69) axis-angle (first 63 used)
    global_orient: (T, 3) axis-angle in SMPL y-up world (e.g. GVHMR global)
    """
    body_pose = np.asarray(body_pose, dtype=np.float32).reshape(len(body_pose), -1)
    global_orient = np.asarray(global_orient, dtype=np.float32)
    T = body_pose.shape[0]

    with torch.no_grad():
        out = process_smpl_joints_batch(
            body_pose=torch.from_numpy(body_pose).to(device),
            global_orient=torch.from_numpy(global_orient).to(device),
        )
    smpl_pose = body_pose[:, :63].reshape(T, 21, 3)
    smpl_joints = out["smpl_joints_local"].cpu().numpy().astype(np.float32)
    body_quat = out["global_orient_quat"].cpu().numpy().astype(np.float32)
    joint_pos = wrist_joint_pos(smpl_pose)

    return [
        {
            "smpl_pose": smpl_pose[t],
            "smpl_joints": smpl_joints[t],
            "body_quat_w": body_quat[t],
            "joint_pos": joint_pos[t],
        }
        for t in range(T)
    ]
