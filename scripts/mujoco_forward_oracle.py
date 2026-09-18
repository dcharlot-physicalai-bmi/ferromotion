"""MuJoCo's `qacc` at rest — the oracle for a whole forward step, from an MJCF file to an acceleration.

    python scripts/mujoco_forward_oracle.py <menagerie root> <out.txt> [samples per model]

Per compilable model and per sampled state: the joints with their `qposadr`, `qpos`, `qvel`, MuJoCo's
`qfrc_bias`, `qacc_smooth` and `qacc`, and the gates a port has to respect to be comparing the same thing —
how many constraint rows are contacts rather than limits or equalities, whether any actuator or passive force
is acting, the cone, and `impratio`. Tab separated, one record per line.
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

lines, models, failed = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    d = mujoco.MjData(m)
    models += 1
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    lines.append(f"model\t{rel}\t{m.nq}\t{m.nv}\t{int(m.opt.cone)}\t{float(m.opt.impratio)!r}\t{float(m.opt.gravity[0])!r}\t{float(m.opt.gravity[1])!r}\t{float(m.opt.gravity[2])!r}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", name(mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            # the same sampling the contact oracle uses: hinges and slides jiggled, free joints left alone
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    a = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
            # ⛔ a state at rest cannot tell a velocity-dependent term from a missing one: damping and
            # joint friction are both zero at qvel = 0, so sample the last state MOVING
            if k > 1:
                d.qvel[:] = rng.uniform(-0.4, 0.4, m.nv)
        mujoco.mj_forward(m, d)
        # which constraint rows are contacts: anything else needs machinery this port does not claim
        CT = {int(v): k.replace("mjCNSTR_", "").lower() for k, v in mujoco.mjtConstraint.__members__.items()}
        hist = {}
        for t in d.efc_type[: d.nefc]:
            hist[CT[int(t)]] = hist.get(CT[int(t)], 0) + 1
        ncontact = sum(v for k, v in hist.items() if k.startswith("contact"))
        dims = sorted({int(c.dim) for c in d.contact[: d.ncon]})
        lines.append("\t".join([
            "state", str(k), str(int(d.nefc)), str(ncontact), str(int(d.ncon)),
            repr(float(np.abs(d.qfrc_actuator).max(initial=0.0))),
            repr(float(np.abs(d.qfrc_passive).max(initial=0.0))),
            ",".join(str(x) for x in dims) or "-",
            ";".join(f"{k}={v}" for k, v in sorted(hist.items())) or "-",
        ]))
        for tag, v in (("qpos", d.qpos), ("qvel", d.qvel), ("qfrc_bias", d.qfrc_bias), ("qfrc_passive", d.qfrc_passive), ("qacc_smooth", d.qacc_smooth), ("qacc", d.qacc)):
            lines.append("\t".join([tag] + [repr(float(x)) for x in v]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models ({failed} would not compile), {K} states each -> {out}")
