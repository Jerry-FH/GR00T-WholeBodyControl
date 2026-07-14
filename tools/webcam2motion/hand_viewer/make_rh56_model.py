"""Generate the SONIC sim2sim model with Inspire RH56DFQ hands (stage 2).

Surgery on gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml:
  1. delete the Dex3 finger subtrees + their 14 motors
  2. attach the DFQ URDFs' hand_base_link at the wrist_yaw links with an
     IDENTITY frame. The DFQ URDFs already carry the full mount transform
     (L: xyz 0.0415 0 0, rpy 0 0 pi/2; R: rpy pi 0 -pi/2) as the base
     link's own pos/quat, and MjSpec attach preserves it — adding the same
     transform again on the outer frame doubles it and yields fingers
     90 deg outboard + right palm flipped 180 deg (verified against the
     Dex3 scene: finger dir must be ~+x, curl dir ~+y in the wrist frame).
  3. joints get left_hand_/right_hand_ prefixes (BaseSimulator classifies hand
     joints by that substring; none may contain body substrings like 'wrist')
  4. regenerate ALL actuators, one <motor> per non-free joint in joint-id
     order — BaseSimulator's torque scatter assumes actuator_id == joint_id-1
  5. emit scene_43dof_rh56.xml + the MOTOR_EFFORT_LIMIT_LIST for the yaml
     (body joints keep their old limits, mapped by name; fingers get 2.45 —
     value is moot since NUM_HAND_MOTORS=0 keeps their torque at zero)

Run from repo root:  .venv_sim/bin/python tools/webcam2motion/hand_viewer/make_rh56_model.py
"""

import os

import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(REPO, "gear_sonic/data/robot_model/model_data/g1")
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

DEX3_ROOTS = [f"{s}_hand_{f}_0_link" for s in ("left", "right")
              for f in ("thumb", "middle", "index")]


def main():
    src = os.path.join(MODEL_DIR, "g1_29dof_with_hand.xml")
    old = mujoco.MjModel.from_xml_path(os.path.join(MODEL_DIR, "scene_43dof.xml"))
    # per-joint effort limits of the CURRENT model, by joint name (actuator k <-> joint k+1)
    old_limits = {}
    for k in range(old.nu):
        jname = mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, k + 1)
        old_limits[jname] = float(old.actuator_forcerange[k][1]) or 2.45

    spec = mujoco.MjSpec.from_file(src)

    for name in DEX3_ROOTS:  # 1. drop Dex3 finger subtrees
        body = spec.body(name)
        spec.delete(body)
    for act in list(spec.actuators):  # and every actuator (regenerated below)
        spec.delete(act)

    for side in ("left", "right"):  # 2-3. attach DFQ hands
        sub = mujoco.MjSpec.from_file(os.path.join(ASSETS, f"DFQ_{side}_hand.urdf"))
        # fingers are kinematically driven (NUM_HAND_MOTORS=0, qpos writes) —
        # disable their contacts: at the correct mount the thumb mesh
        # interpenetrates the wrist link (-6 mm at rest), and the resulting
        # 1 kHz contact forces on the arm tip destabilize the policy
        # (21 falls/35 s in test_m2). Damping stops torqueless finger flail.
        for body in sub.bodies:
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0
        for joint in sub.joints:
            joint.damping = [0.05, 0.0, 0.0]  # MjsJoint stores per-dof (3,)
        wrist = spec.body(f"{side}_wrist_yaw_link")
        frame = wrist.add_frame()  # identity — URDF base link carries the mount
        base = sub.body(f"{side[0].upper()}_hand_base_link")  # skip URDF wrist dummy
        frame.attach_body(base, f"{side}_hand_", "")

    model = spec.compile()  # 4. regenerate actuators in joint-id order
    joint_order = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(model.njnt)
                   if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    limits = []
    for jname in joint_order:
        act = spec.add_actuator()
        act.name = jname.removesuffix("_joint")
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        limits.append(old_limits.get(jname, 2.45))

    model = spec.compile()
    assert model.nu == model.njnt - 1, (model.nu, model.njnt)
    for k in range(model.nu):  # verify the scatter invariant
        assert model.actuator_trnid[k][0] == k + 1

    out_robot = os.path.join(MODEL_DIR, "g1_29dof_with_rh56.xml")
    with open(out_robot, "w") as f:
        f.write(spec.to_xml())

    scene = open(os.path.join(MODEL_DIR, "scene_43dof.xml")).read()
    scene = scene.replace("g1_29dof_with_hand.xml", "g1_29dof_with_rh56.xml")
    out_scene = os.path.join(MODEL_DIR, "scene_53dof_rh56.xml")
    with open(out_scene, "w") as f:
        f.write(scene)

    # sanity: scene compiles, classifier counts match
    m2 = mujoco.MjModel.from_xml_path(out_scene)
    body = hands_l = hands_r = 0
    for j in range(m2.njnt):
        n = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if any(p in n for p in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]):
            body += 1
        elif "left_hand" in n:
            hands_l += 1
        elif "right_hand" in n:
            hands_r += 1
    print(f"scene OK: njnt={m2.njnt} nu={m2.nu} | body={body} left_hand={hands_l} "
          f"right_hand={hands_r}")
    print(f"robot: {out_robot}\nscene: {out_scene}")
    print("\nMOTOR_EFFORT_LIMIT_LIST ({} entries):".format(len(limits)))
    print("[" + ", ".join(f"{v:g}" for v in limits) + "]")


if __name__ == "__main__":
    main()
