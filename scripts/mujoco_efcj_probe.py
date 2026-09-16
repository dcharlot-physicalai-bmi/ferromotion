"""MuJoCo's constraint Jacobian and body_invweight0 on a loaded model — the oracle for the glue that turns a
contact into constraint rows (`tree_jacobian.rs`).

    python scripts/mujoco_efcj_probe.py <model.xml> [qpos ...]

Prints, at qpos0 unless a qpos is given: body_invweight0 per body, then per contact its geoms, bodies,
condim, position and frame, and the `efc_J` rows MuJoCo built for it.
"""
import sys

import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path(sys.argv[1])
d = mujoco.MjData(m)
if len(sys.argv) > 2:
    d.qpos[:] = [float(x) for x in sys.argv[2:]]
mujoco.mj_forward(m, d)
name = lambda t, i: mujoco.mj_id2name(m, t, i) or f"{t.name}{i}"

print(f"model {sys.argv[1]} nv={m.nv} nbody={m.nbody} ncon={d.ncon} nefc={d.nefc} cone={int(m.opt.cone)}")
for b in range(m.nbody):
    print(f"  invweight0 body {b} {name(mujoco.mjtObj.mjOBJ_BODY, b)}: tran={float(m.body_invweight0[b][0])!r} rot={float(m.body_invweight0[b][1])!r}")
J = np.asarray(d.efc_J).reshape(d.nefc, m.nv) if d.nefc else np.zeros((0, m.nv))
row = 0
for i in range(d.ncon):
    c = d.contact[i]
    g1, g2 = int(c.geom[0]), int(c.geom[1])
    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
    nrow = 1 if c.dim == 1 else (2 * (c.dim - 1) if m.opt.cone == 0 else c.dim)
    print(f"  contact {i}: g1={name(mujoco.mjtObj.mjOBJ_GEOM,g1)} g2={name(mujoco.mjtObj.mjOBJ_GEOM,g2)} "
          f"b1={b1} b2={b2} dim={c.dim} rows={nrow} dist={c.dist!r} friction={[repr(float(x)) for x in c.friction]}")
    print(f"    pos={[repr(float(x)) for x in c.pos]}")
    print(f"    frame={[repr(float(x)) for x in c.frame]}")
    for r in range(nrow):
        print(f"    J[{r}]={[repr(float(x)) for x in J[row + r]]}")
    row += nrow
print(f"  qpos={[repr(float(x)) for x in d.qpos]}")
