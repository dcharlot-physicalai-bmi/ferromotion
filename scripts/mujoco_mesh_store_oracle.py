"""What MuJoCo's mesh compiler STORES, bit for bit: `mesh_pos` and `mesh_quat` (the centre of mass and principal
frame `mjCMesh::Process` moved the mesh by) for every mesh, and `mesh_vert` (the float vertices every collider
reads) for every collision mesh — the oracle for `examples/menagerie_mesh_store.rs`.

    python scripts/mujoco_mesh_store_oracle.py <menagerie root> <out.txt>

`frame <model dir> <mesh> <pos x3> <quat x4>` in C99 hex; `vert <model dir> <mesh> <n> <3n float bit patterns>`.
A mesh is dumped once per model directory (the first model that uses it). Run it with the working directory
somewhere disposable: MuJoCo writes MUJOCO_LOG.TXT there on warnings.
"""
import glob
import os
import sys

import mujoco
import numpy as np

# ⛔ absolute: MuJoCo resolves an <include> inside an included file against a RELATIVE model path twice over
# (`ms_human_700` fails to load from `menagerie/...` and silently drops out of the oracle)
root, out = os.path.abspath(sys.argv[1]), sys.argv[2]
lines, seen = [], set()
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    d = rel.split("/")[0]
    collides = set()
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g]):
            collides.add(int(m.geom_dataid[g]))
    for i in range(m.nmesh):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
        if (d, name) in seen:
            continue
        seen.add((d, name))
        vals = [float(x).hex() for x in m.mesh_pos[i]] + [float(x).hex() for x in m.mesh_quat[i]]
        lines.append("frame\t%s\t%s\t%s" % (d, name, " ".join(vals)))
        if i in collides:
            a, n = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
            bits = np.ascontiguousarray(m.mesh_vert[a:a + n], dtype=np.float32).view(np.uint32).ravel()
            lines.append("vert\t%s\t%s\t%d\t%s" % (d, name, n, " ".join("%08x" % b for b in bits)))
open(out, "w").write("\n".join(lines) + "\n")
print(sum(l.startswith("frame") for l in lines), "meshes,", sum(l.startswith("vert") for l in lines), "with vertices")
