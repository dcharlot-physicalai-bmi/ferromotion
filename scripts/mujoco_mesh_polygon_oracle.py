"""MuJoCo's own convex-hull POLYGONS for every collision mesh — the structure `multicontact` clips.

    python scripts/mujoco_mesh_polygon_oracle.py <menagerie root> <out.txt> [substring]

MuJoCo 3.x exposes the whole thing on the model: `mesh_polynum`, `mesh_polyadr`, `mesh_polynormal`,
`mesh_polyvert` with `mesh_polyvertadr`/`mesh_polyvertnum`, and the vertex → polygon map
`mesh_polymap`/`mesh_polymapadr`/`mesh_polymapnum`. A port's hull can therefore be checked face by face
rather than by vertex count, which is what decides whether a face-face contact produces four witnesses or one.

Per mesh: its name, vertex count, polygon count, and one line per polygon with its normal and its vertex
indices in order. Meshes are keyed by name, and a name is dumped once (the first model that uses it).
"""

import glob
import os
import sys

import mujoco

root, out = sys.argv[1], sys.argv[2]
sub = sys.argv[3] if len(sys.argv) > 3 else None

lines, seen, models, failed = [], set(), 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    if sub and sub not in rel:
        continue
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    models += 1
    for i in range(m.nmesh):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
        key = (rel.split("/")[0], name)
        if key in seen:
            continue
        seen.add(key)
        npoly = int(m.mesh_polynum[i])
        lines.append(f"mesh\t{rel}\t{name}\t{int(m.mesh_vertnum[i])}\t{npoly}")
        adr = int(m.mesh_polyadr[i])
        for p in range(adr, adr + npoly):
            va, vn = int(m.mesh_polyvertadr[p]), int(m.mesh_polyvertnum[p])
            n = m.mesh_polynormal[p]
            verts = [str(int(m.mesh_polyvert[va + k])) for k in range(vn)]
            lines.append("\t".join(["poly", repr(float(n[0])), repr(float(n[1])), repr(float(n[2]))] + verts))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models ({failed} would not compile), {len(seen)} meshes -> {out}")
