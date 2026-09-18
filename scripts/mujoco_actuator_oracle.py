"""MuJoCo's actuators — the resolved parameters and the force they produce.

    python scripts/mujoco_actuator_oracle.py <menagerie root> <out.txt> [samples per model]

Per compilable model: every actuator's COMPILED parameters (the shortcut tags are sugar — MuJoCo resolves
them into `gaintype`/`biastype`/`gainprm`/`biasprm` before anything runs, and `dampratio` becomes a damping
only after the mass matrix is known), then per sampled state the control, `actuator_length`,
`actuator_velocity`, `actuator_force` and `qfrc_actuator`. Enough to say WHICH link of the chain differs
rather than only that the answer does. Tab separated.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.default_rng(20260918)
TRN = {v: k.replace("mjTRN_", "").lower() for k, v in mujoco.mjtTrn.__members__.items()}
GAIN = {v: k.replace("mjGAIN_", "").lower() for k, v in mujoco.mjtGain.__members__.items()}
BIAS = {v: k.replace("mjBIAS_", "").lower() for k, v in mujoco.mjtBias.__members__.items()}
DYN = {v: k.replace("mjDYN_", "").lower() for k, v in mujoco.mjtDyn.__members__.items()}
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
    lines.append(f"model\t{rel}\t{m.nq}\t{m.nv}\t{m.nu}\t{m.na}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", name(mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j])), str(int(m.jnt_dofadr[j]))]))
    for u in range(m.nu):
        trn = TRN[int(m.actuator_trntype[u])]
        tgt = int(m.actuator_trnid[u, 0])
        tname = name(mujoco.mjtObj.mjOBJ_JOINT, tgt) if trn in ("joint", "jointinparent") else str(tgt)
        lines.append("\t".join([
            "actuator", name(mujoco.mjtObj.mjOBJ_ACTUATOR, u) or f"act{u}", trn, tname or "?",
            GAIN[int(m.actuator_gaintype[u])], BIAS[int(m.actuator_biastype[u])], DYN[int(m.actuator_dyntype[u])],
            repr(float(m.actuator_gear[u, 0])),
            *[repr(float(x)) for x in m.actuator_gainprm[u, :3]],
            *[repr(float(x)) for x in m.actuator_biasprm[u, :3]],
            str(int(m.actuator_ctrllimited[u])), repr(float(m.actuator_ctrlrange[u, 0])), repr(float(m.actuator_ctrlrange[u, 1])),
            str(int(m.actuator_forcelimited[u])), repr(float(m.actuator_forcerange[u, 0])), repr(float(m.actuator_forcerange[u, 1])),
            repr(float(m.actuator_acc0[u])),
        ]))
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    a = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
            d.qvel[:] = rng.uniform(-0.3, 0.3, m.nv)
            # controls outside their range too, so the clamp is exercised rather than assumed
            d.ctrl[:] = rng.uniform(-2.0, 2.0, m.nu)
        mujoco.mj_forward(m, d)
        lines.append("state\t" + str(k))
        for tag, v in (("qpos", d.qpos), ("qvel", d.qvel), ("ctrl", d.ctrl), ("act_length", d.actuator_length),
                       ("act_velocity", d.actuator_velocity), ("act_force", d.actuator_force), ("qfrc_actuator", d.qfrc_actuator)):
            lines.append("\t".join([tag] + [repr(float(x)) for x in v]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models ({failed} would not compile), {K} states each -> {out}")
