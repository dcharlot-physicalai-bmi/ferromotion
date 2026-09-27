"""MuJoCo's forward kinematics at random states — the oracle for `examples/menagerie_kinematics.rs`.

    python scripts/mujoco_kinematics_oracle.py <menagerie root> <out.txt> [states per model]

Per compilable model: `model <rel>`, then per state a `qpos` line and one `geom <i> <xpos x3> <xmat x9>` line per
geom, every number as `repr` (exact). State 0 is `qpos0`; the rest move every joint: a hinge or slide to a
uniform point of its range (±π or ±0.5 m when unlimited), a ball or free joint to a random unnormalised
quaternion (MuJoCo normalises it), a free joint's position to a random point in a 2 m cube.
"""
import glob
import os
import sys

import mujoco
import numpy as np

root = os.path.abspath(sys.argv[1])  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice
out = sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rng = np.random.default_rng(20260927)
lines, models = [], 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    models += 1
    d = mujoco.MjData(m)
    lines.append(f"model\t{rel}")
    for s in range(k):
        q = m.qpos0.copy()
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
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        lines.append("qpos\t" + "\t".join(repr(float(x)) for x in q))
        for g in range(m.ngeom):
            lines.append("geom\t%d\t%s" % (g, "\t".join(repr(float(x)) for x in list(d.geom_xpos[g]) + list(d.geom_xmat[g]))))
open(out, "w").write("\n".join(lines) + "\n")
print(models, "models,", k, "states each ->", out)
