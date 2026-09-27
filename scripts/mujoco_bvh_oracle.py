"""MuJoCo's collision driver, decision by decision — the oracle for `examples/menagerie_midphase.rs`.

    python scripts/mujoco_bvh_oracle.py <menagerie root> <out.txt> [states per model]

Per compilable model, `model <rel>`, then what the compiler built for the midphase:

    body <b> <bvhadr> <bvhnum> <body_margin>          one line per body
    node <b> <k> <child0> <child1> <nodeid> <depth> <aabb x6>   one line per node of a body's BVH
    gaabb <g> <aabb x6>                                  every geom_aabb

then per state (`qpos0`, then random states that move every joint, as `mujoco_kinematics_oracle.py`
draws them) a `qpos` line, `active <node ids>` — every body-BVH node `mj_collideTree` marked as passing
(`bvh_active`, with `vis.global.bvactive` on), world-relative to `bvh_aabb` — and one
`contact <g1> <g2> <dist>` per contact in MuJoCo's own order. Every float is `float.hex` (exact).
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
h = lambda xs: " ".join(float(x).hex() for x in xs)
lines, models = [], 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    models += 1
    m.vis.global_.bvactive = 1
    d = mujoco.MjData(m)
    lines.append(f"model\t{rel}")
    for b in range(m.nbody):
        lines.append(f"body\t{b}\t{m.body_bvhadr[b]}\t{m.body_bvhnum[b]}\t{float(m.body_margin[b]).hex()}")
        for n in range(m.body_bvhnum[b]):
            a = m.body_bvhadr[b] + n
            lines.append(f"node\t{b}\t{n}\t{m.bvh_child[a][0]}\t{m.bvh_child[a][1]}\t{m.bvh_nodeid[a]}\t{m.bvh_depth[a]}\t{h(m.bvh_aabb[a])}")
    for g in range(m.ngeom):
        lines.append(f"gaabb\t{g}\t{h(m.geom_aabb[g])}")
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
        mujoco.mj_collision(m, d)
        lines.append("qpos\t" + "\t".join(repr(float(x)) for x in q))
        body_nodes = [m.body_bvhadr[b] + n for b in range(m.nbody) if m.body_bvhadr[b] >= 0 for n in range(m.body_bvhnum[b])]
        lines.append("active\t" + " ".join(str(a) for a in body_nodes if d.bvh_active[a]))
        for c in d.contact[:d.ncon]:
            lines.append(f"contact\t{c.geom1}\t{c.geom2}\t{float(c.dist).hex()}")
open(out, "w").write("\n".join(lines) + "\n")
print(models, "models,", k, "states each ->", out)
