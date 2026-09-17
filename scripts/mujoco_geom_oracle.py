"""MuJoCo's geoms at qpos0 — the oracle for what an MJCF loader must produce before anything can collide.

    python scripts/mujoco_geom_oracle.py <menagerie root> <out.txt>

Per compilable model: the joints with their `qposadr` (so a port can map `qpos` to its own coordinates),
`qpos0`, and every geom's name, type, body, resolved `geom_size`, contact parameters, and its world pose at
`qpos0`. Tab separated, one record per line.
"""

import glob
import os
import sys

import mujoco

GT = {v: k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}
JT = {v: k.replace("mjJNT_", "").lower() for k, v in mujoco.mjtJoint.__members__.items()}

root, out = sys.argv[1], sys.argv[2]
lines, models, failed = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    models += 1
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    lines.append(f"model\t{rel}\t{m.njnt}\t{m.ngeom}\t{m.nq}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", name(mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    lines.append("\t".join(["qpos0"] + [repr(float(x)) for x in d.qpos]))
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        lines.append(
            "\t".join(
                ["geom", name(mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", GT[int(m.geom_type[g])],
                 name(mujoco.mjtObj.mjOBJ_BODY, b) or ("world" if b == 0 else f"body{b}")]
                + [repr(float(x)) for x in m.geom_size[g]]
                + [str(int(m.geom_contype[g])), str(int(m.geom_conaffinity[g])), str(int(m.geom_condim[g])),
                   repr(float(m.geom_margin[g])), repr(float(m.geom_gap[g])), str(int(m.geom_priority[g])),
                   repr(float(m.geom_solmix[g]))]
                + [repr(float(x)) for x in m.geom_friction[g]]
                + [repr(float(x)) for x in m.geom_solref[g]]
                + [repr(float(x)) for x in m.geom_solimp[g]]
                + [repr(float(x)) for x in d.geom_xpos[g]]
                + [repr(float(x)) for x in d.geom_xmat[g]]
            )
        )
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models ({failed} would not compile) -> {out}")
