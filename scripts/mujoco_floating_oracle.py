"""MuJoCo's mass matrix and accelerations for models with a FREE base, in MuJoCo's own basis.

    python scripts/mujoco_floating_oracle.py <menagerie root> <out.txt> [samples per model]

A free joint is six coordinates here and six there, but not the SAME six: MuJoCo's are the body-frame origin's
world velocity and the body-frame angular velocity, about the subtree centre of mass; this port's are three
world slides and three Euler hinges. Every comparison of `qacc` on such a model needs the change of basis,
and the thing to check it against is the MASS MATRIX, which MuJoCo will hand over in full.

Per compilable model whose first joint is free and which is small enough to print: the joints with their
`qposadr`, each body's name, and per sampled state the `qpos`, `qvel`, the dense `M`, `subtree_com` of the
root, `qfrc_bias`, `qacc_smooth`, `qacc` and `dof_armature`.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAXNV = int(os.environ.get("MAXNV", "40"))
rng = np.random.default_rng(20260917)
JT = {v: k.replace("mjJNT_", "").lower() for k, v in mujoco.mjtJoint.__members__.items()}

lines, models, failed = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    if m.njnt == 0 or int(m.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or m.nv > MAXNV:
        continue
    d = mujoco.MjData(m)
    models += 1
    lines.append(f"model\t{rel}\t{m.nq}\t{m.nv}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    lines.append("\t".join(["armature"] + [repr(float(x)) for x in m.dof_armature]))
    # the constraint rows regularise against these, and MuJoCo computes them at ITS qpos0 in ITS basis
    lines.append("\t".join(["invweight0"] + [repr(float(x)) for x in m.dof_invweight0]))
    lines.append("\t".join(["qpos0"] + [repr(float(x)) for x in m.qpos0]))
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            for j in range(m.njnt):
                t = int(m.jnt_type[j])
                a = int(m.jnt_qposadr[j])
                if t in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
                elif t == int(mujoco.mjtJoint.mjJNT_FREE):
                    d.qpos[a:a + 3] = rng.uniform(-0.3, 0.3, 3) + m.qpos0[a:a + 3]
                    v = rng.normal(size=4)
                    d.qpos[a + 3:a + 7] = v / np.linalg.norm(v)
            # ⛔ the last state MOVING: the basis change is exact at rest whatever the orientation, and the
            # velocity-dependent half only shows up when something is turning
            if k > 1:
                d.qvel[:] = rng.uniform(-0.4, 0.4, m.nv)
        mujoco.mj_forward(m, d)
        # ⛔ `d.M` is the sparse factor storage; the dense matrix comes from mj_fullM, whose Python
        # signature is (model, dst, sparse_source)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        lines.append("\t".join(["state", str(k)]))
        for tag, v in (("qpos", d.qpos), ("qvel", d.qvel), ("bias", d.qfrc_bias), ("smooth", d.qacc_smooth), ("qacc", d.qacc),
                       ("subtreecom", d.subtree_com[0]), ("xquat", d.xquat[1]), ("xpos", d.xpos[1]), ("xipos", d.xipos[1])):
            lines.append("\t".join([tag] + [repr(float(x)) for x in np.asarray(v).reshape(-1)]))
        for r in range(m.nv):
            lines.append("\t".join(["M"] + [repr(float(x)) for x in M[r]]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} free-base models (nv <= {MAXNV}; {failed} would not compile), {K} states each -> {out}")
