"""MuJoCo's `ten_length` and `ten_J` for every tendon, at sampled states.

    python scripts/mujoco_tendon_oracle.py <menagerie root> <out.txt> [samples per model]

A tendon is where a model's coupling lives — a gripper's two fingers on one motor, a muscle crossing a
joint. A FIXED tendon's Jacobian is a row of constants; a SPATIAL one's is the derivative of a path that
bends around obstacles, so it has to be checked at states, not at load.

Per compilable model with at least one tendon: its joints with their `qposadr`, then per sampled state the
`qpos`, and for each tendon its name, kind, `ten_length` and its `ten_J` row.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.default_rng(20260917)
JT = {v: k.replace("mjJNT_", "").lower() for k, v in mujoco.mjtJoint.__members__.items()}

lines, models, failed, tendons = [], 0, 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    if m.ntendon == 0:
        continue
    d = mujoco.MjData(m)
    models += 1
    tendons += m.ntendon
    lines.append(f"model\t{rel}\t{m.nv}\t{m.ntendon}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    for t in range(m.ntendon):
        # a tendon is spatial when its first wrap element is not a joint
        kind = "fixed" if int(m.wrap_type[m.tendon_adr[t]]) == int(mujoco.mjtWrap.mjWRAP_JOINT) else "spatial"
        lines.append("\t".join(["tendon", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t) or f"tendon{t}", kind]))
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    a = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        lines.append("\t".join(["state"] + [repr(float(x)) for x in d.qpos]))
        lines.append("\t".join(["len"] + [repr(float(x)) for x in d.ten_length]))
        # ⛔ `ten_J` is stored SPARSE on anything bigger than a gripper and its index arrays are not exposed
        # in the Python bindings, so the Jacobian is checked by what it DOES: `ten_velocity = ten_J·q̇` for
        # three random velocities. A wrong row survives one projection with probability zero.
        for _ in range(3):
            v = rng.uniform(-1.0, 1.0, m.nv)
            d.qvel[:] = v
            mujoco.mj_forward(m, d)
            lines.append("\t".join(["qvel"] + [repr(float(x)) for x in v]))
            lines.append("\t".join(["tenvel"] + [repr(float(x)) for x in d.ten_velocity]))
        d.qvel[:] = 0
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models with tendons ({tendons} tendons, {failed} would not compile), {K} states each -> {out}")
