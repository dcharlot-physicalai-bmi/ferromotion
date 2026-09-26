"""MuJoCo's own trajectory under `mj_step`, for every compilable Menagerie model.

    python scripts/mujoco_step_oracle.py <menagerie root> <out.txt> [steps]

Each model starts at `qpos0` with its hinges and slides jiggled (the same sampling the other oracles use),
every dof moving at a random velocity in [-0.5, 0.5], and `ctrl = 0`. The solver is then told to CONVERGE
(`iterations = 1000`, `ls_iterations = 50`, `tolerance = 0`) so that each of MuJoCo's steps is the
optimum of the rows it builds, which is what `MjcfTree::step_mujoco` takes. A model that caps its solver
(every `*_mjx.xml`) is therefore stepped here as it would be uncapped, and says so on its `model` line.

Tab separated: `model rel nq nv capped`, one `joint name type qposadr` per joint, then `start qpos...`,
`startv qvel...`, and `step k qpos...` / `stepv k qvel...` after steps 1, 5 and the last.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
rng = np.random.default_rng(20260926)
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
    capped = int(m.opt.iterations < 100 or m.opt.ls_iterations < 50)
    m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = 1000, 50, 0.0
    models += 1
    lines.append(f"model\t{rel}\t{m.nq}\t{m.nv}\t{capped}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    mujoco.mj_resetData(m, d)
    for j in range(m.njnt):
        if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            a = int(m.jnt_qposadr[j])
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
            d.qpos[a] = rng.uniform(lo, hi)
    d.qvel[:] = rng.uniform(-0.5, 0.5, m.nv)
    lines.append("\t".join(["start"] + [repr(float(x)) for x in d.qpos]))
    lines.append("\t".join(["startv"] + [repr(float(x)) for x in d.qvel]))
    for k in range(1, N + 1):
        mujoco.mj_step(m, d)
        if k in (1, 5, N):
            lines.append("\t".join(["step", str(k)] + [repr(float(x)) for x in d.qpos]))
            lines.append("\t".join(["stepv", str(k)] + [repr(float(x)) for x in d.qvel]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models ({failed} would not compile), {N} steps each -> {out}")
