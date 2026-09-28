"""MuJoCo's own arrays, by name — the oracle for `examples/menagerie_native.rs`.

    python scripts/mujoco_native_oracle.py <menagerie root> <out.txt> [states per model] [substring]

Per compilable model, `model <rel>`, then one `m <name> <values>` line per compiled `mjModel` field the
native pipeline reads, and per state (`qpos0` at rest, then random states that move every joint and every
dof's velocity, every control — a little past each end of its range, so the clamp is exercised — and every
activation) `qpos`, `qvel`, `ctrl` and `act` lines followed by one `d <name> <values>` line per `mjData`
array after `mj_forward`. Integers are written as integers, floats as `float.hex` (exact).
"""
import glob
import os
import sys

import mujoco
import numpy as np

root = os.path.abspath(sys.argv[1])  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice
out = sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sub = sys.argv[4] if len(sys.argv) > 4 else ""
rng = np.random.default_rng(20260927)

MODEL = ["body_parentid", "body_rootid", "body_weldid", "body_jntnum", "body_jntadr", "body_dofnum", "body_dofadr", "body_simple",
         "body_mass", "body_subtreemass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
         "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "jnt_pos", "jnt_axis",
         "dof_bodyid", "dof_jntid", "dof_parentid", "dof_simplenum", "dof_M0", "dof_armature", "dof_damping",
         "M_rownnz", "M_rowadr", "M_colind", "qpos0", "body_invweight0", "dof_invweight0",
         "tendon_invweight0"]
DATA = ["xpos", "xquat", "xmat", "xipos", "ximat", "xanchor", "xaxis", "subtree_com", "cinert", "cdof", "crb",
        "M", "qLD", "qLDiagInv", "cvel", "cdof_dot", "qfrc_bias",
        "qfrc_spring", "qfrc_damper", "qfrc_gravcomp", "qfrc_passive", "ten_length", "ten_velocity",
        "actuator_length", "actuator_velocity",
        "actuator_force", "act_dot", "qfrc_actuator", "qfrc_smooth", "qacc_smooth",
        "efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss", "efc_diagA", "efc_R", "efc_D", "efc_KBIP",
        "efc_vel", "efc_aref", "efc_b", "efc_force", "efc_state", "qfrc_constraint", "qacc"]


def fmt(a):
    a = np.asarray(a).ravel()
    if a.dtype.kind in "iub":
        return " ".join(str(int(x)) for x in a)
    return " ".join(float(x).hex() for x in a)


lines, models = [], 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    if sub not in rel:
        continue
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    models += 1
    d = mujoco.MjData(m)
    lines.append(f"model\t{rel}")
    for name in MODEL:
        lines.append(f"m\t{name}\t{fmt(getattr(m, name))}")
    lines.append(f"m\tgravity\t{fmt(m.opt.gravity)}")
    for s in range(k):
        q = m.qpos0.copy()
        v = np.zeros(m.nv)
        if s > 0:
            for j in range(m.njnt):
                a, t = m.jnt_qposadr[j], m.jnt_type[j]
                if t == mujoco.mjtJoint.mjJNT_FREE:
                    q[a:a + 3] = rng.uniform(-1, 1, 3)
                    q[a + 3:a + 7] = rng.normal(size=4)
                elif t == mujoco.mjtJoint.mjJNT_BALL:
                    q[a:a + 4] = rng.normal(size=4)
                else:
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else ((-np.pi, np.pi) if t == mujoco.mjtJoint.mjJNT_HINGE else (-0.5, 0.5))
                    q[a] = rng.uniform(lo, hi)
            v = rng.uniform(-1, 1, m.nv)
        u = np.zeros(m.nu)
        a = np.zeros(m.na)
        if s > 0:
            for i in range(m.nu):
                lo, hi = m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else (-1.0, 1.0)
                # a little past each end, so the clamp is exercised
                u[i] = rng.uniform(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
            a = rng.uniform(0, 1, m.na)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.qvel[:] = v
        d.ctrl[:] = u
        d.act[:] = a
        mujoco.mj_forward(m, d)
        lines.append("qpos\t" + fmt(q))
        lines.append("qvel\t" + fmt(v))
        lines.append("ctrl\t" + fmt(u))
        lines.append("act\t" + fmt(a))
        for name in DATA:
            lines.append(f"d\t{name}\t{fmt(getattr(d, name))}")
        # efc_J, dense whichever way MuJoCo stores it
        J = np.zeros((d.nefc, m.nv))
        if d.nefc:
            if mujoco.mj_isSparse(m):
                mujoco.mju_sparse2dense(J, d.efc_J, d.efc_J_rownnz, d.efc_J_rowadr, d.efc_J_colind)
            else:
                J[:] = d.efc_J[:d.nefc * m.nv].reshape(d.nefc, m.nv)
        lines.append(f"d\tefc_J\t{fmt(J)}")
        # the dual matrix, where a noslip post-pass (or PGS) makes MuJoCo assemble it, dense models only
        if m.opt.noslip_iterations > 0 and not mujoco.mj_isSparse(m):
            lines.append(f"d\tefc_AR\t{fmt(d.efc_AR[:d.nefc * d.nefc])}")
        lines.append(f"d\tnisland\t{d.nisland}")
        lines.append(f"d\tsolver_niter\t{fmt(d.solver_niter[:max(d.nisland, 1)])}")
open(out, "w").write("\n".join(lines) + "\n")
print(models, "models,", k, "states each ->", out)
