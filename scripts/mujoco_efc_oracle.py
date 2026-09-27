"""MuJoCo's whole constraint set — every row, not only the contacts.

    python scripts/mujoco_efc_oracle.py <menagerie root> <out.txt> [samples per model] [max rows]

Per sampled state with at least one constraint row: `ne`/`nf` (MuJoCo decides a row's LAW from its position,
so the two counts are part of the data), every row's type, `efc_J`, `efc_aref`, `efc_D`, `efc_R` and
`efc_frictionloss`, each contact's cone parameters, the dense mass matrix, `qacc_smooth`, `qacc`, and the
options that decide which `qacc` MuJoCo reports (noslip, `iterations`, `tolerance`), and `qacc_conv`, the
same state with the main solve told to converge. Enough
to test a solver against MuJoCo's answer using MuJoCo's own rows, which separates the solver from the
assembly that builds them. Tab separated.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
root = os.path.abspath(root)  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAXROWS = int(sys.argv[4]) if len(sys.argv) > 4 else 160
rng = np.random.default_rng(20260918)
CT = {int(v): k.replace("mjCNSTR_", "").lower() for k, v in mujoco.mjtConstraint.__members__.items()}

lines, models, kept = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    # dense Jacobians, so a row is a row and nothing has to be un-sparsified on the way out
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    d = mujoco.MjData(m)
    models += 1
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    a = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
            d.qvel[:] = rng.uniform(-0.3, 0.3, m.nv)
        mujoco.mj_forward(m, d)
        nefc = int(d.nefc)
        if nefc == 0 or nefc > MAXROWS:
            continue
        kept += 1
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        lines.append(f"case\t{rel}\t{k}\t{m.nv}\t{nefc}\t{int(d.ne)}\t{int(d.nf)}\t{int(m.opt.cone)}")
        lines.append("\t".join(["type"] + [CT[int(t)] for t in d.efc_type[:nefc]]))
        # ⭐ the solver options that shape `qacc`: a model that asks for noslip reports the acceleration AFTER
        # that pass, which is not the optimum of the rows above, and one that caps `iterations` or sets a
        # loose `tolerance` reports wherever the main solve stopped
        lines.append(f"opt\t{int(m.opt.noslip_iterations)}\t{float(m.opt.noslip_tolerance)!r}\t{float(m.stat.meaninertia)!r}\t{int(m.opt.iterations)}\t{float(m.opt.tolerance)!r}")
        for j in range(m.njnt):
            lines.append("\t".join(["joint", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}",
                                    {int(v): k.replace("mjJNT_", "").lower() for k, v in mujoco.mjtJoint.__members__.items()}[int(m.jnt_type[j])],
                                    str(int(m.jnt_qposadr[j]))]))
        for tag, v in (("qpos", d.qpos), ("qvel", d.qvel), ("aref", d.efc_aref[:nefc]), ("D", d.efc_D[:nefc]), ("R", d.efc_R[:nefc]),
                       ("floss", d.efc_frictionloss[:nefc]), ("qacc_smooth", d.qacc_smooth), ("qacc", d.qacc)):
            lines.append("\t".join([tag] + [repr(float(x)) for x in v]))
        for r in range(nefc):
            lines.append("\t".join(["J"] + [repr(float(x)) for x in d.efc_J[r * m.nv:(r + 1) * m.nv]]))
        for r in range(m.nv):
            lines.append("\t".join(["M"] + [repr(float(x)) for x in M[r]]))
        for ci in range(int(d.ncon)):
            c = d.contact[ci]
            # ⛔ a contact inside the gap band is in `d.contact` and has NO rows; `efc_address` is how the
            # two lists are lined up, and guessing by index silently shifts every contact after the first one
            if int(c.efc_address) < 0:
                continue
            lines.append("\t".join(["contact", str(int(c.efc_address)), str(int(c.dim)), repr(float(c.mu))] + [repr(float(x)) for x in c.friction]))
        # ⭐ and the same state with MuJoCo's main solve told to CONVERGE (noslip, if asked for, still runs after
        # it): the optimum of these rows is unique, so this is the answer a converged solver owes, while `qacc`
        # above is wherever MuJoCo stopped under the model's own `iterations` and `tolerance`
        opts = (m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations)
        m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations = 1000, 0.0, 50
        mujoco.mj_forward(m, d)
        lines.append("\t".join(["qacc_conv"] + [repr(float(x)) for x in d.qacc]))
        m.opt.iterations, m.opt.tolerance, m.opt.ls_iterations = opts
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models, {kept} states with 1..{MAXROWS} constraint rows -> {out}")
