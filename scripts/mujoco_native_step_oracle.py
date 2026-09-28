"""MuJoCo's own trajectory, every step, every bit — the oracle for `examples/menagerie_native_step.rs`.

    python scripts/mujoco_native_step_oracle.py <menagerie root> <out.txt> [steps] [substring]

Each compilable model starts at `qpos0` with its hinges and slides jiggled, every dof moving at a random
velocity and every control drawn in its range (the same draws as `mujoco_step_oracle.py`, plus controls),
then runs `mj_step` with the model's OWN solver settings. Written per model: `model rel`, `start` (qpos,
qvel, act, ctrl lines), then after every step `s k` followed by `qpos`, `qvel`, `act`, `qacc_warmstart`
lines. Every float is `float.hex`.
"""
import glob
import os
import sys

import mujoco
import numpy as np

root = os.path.abspath(sys.argv[1])
out = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
sub = sys.argv[4] if len(sys.argv) > 4 else ""
rng = np.random.default_rng(20260928)
h = lambda xs: " ".join(float(x).hex() for x in np.asarray(xs).ravel())
lines = []
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    if sub not in rel:
        continue
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for j in range(m.njnt):
        if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            a = int(m.jnt_qposadr[j])
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
            d.qpos[a] = rng.uniform(lo, hi)
    d.qvel[:] = rng.uniform(-0.5, 0.5, m.nv)
    for i in range(m.nu):
        lo, hi = m.actuator_ctrlrange[i] if m.actuator_ctrllimited[i] else (-1.0, 1.0)
        d.ctrl[i] = rng.uniform(lo, hi)
    d.act[:] = rng.uniform(0.0, 1.0, m.na)
    lines.append(f"model\t{rel}")
    lines.append("qpos\t" + h(d.qpos))
    lines.append("qvel\t" + h(d.qvel))
    lines.append("act\t" + h(d.act))
    lines.append("ctrl\t" + h(d.ctrl))
    for k in range(1, N + 1):
        mujoco.mj_step(m, d)
        lines.append(f"s\t{k}")
        lines.append("qpos\t" + h(d.qpos))
        lines.append("qvel\t" + h(d.qvel))
        lines.append("act\t" + h(d.act))
        lines.append("warm\t" + h(d.qacc_warmstart))
open(out, "w").write("\n".join(lines) + "\n")
print(len([l for l in lines if l.startswith("model")]), "models,", N, "steps ->", out)
