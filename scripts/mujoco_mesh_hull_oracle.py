"""MuJoCo's own convex-hull data per collision mesh — hull vertex count, polygon count, and the polygons
themselves — so a port can be compared against qhull's answer rather than against a contact downstream of it.

    python scripts/mujoco_mesh_hull_oracle.py <menagerie root> <out.txt>

One line per mesh: `mesh <model> <name> <nvert> <nhullvert> <npolygon>`, then `poly <n> <normal xyz>` lines.
"""
import mujoco, os, sys, glob, collections

root, out = sys.argv[1], sys.argv[2]
lines, seen, models = [], set(), 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    models += 1
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        i = int(m.geom_dataid[g])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
        key = (rel.split("/")[0], name)
        if key in seen:
            continue
        seen.add(key)
        nvert = int(m.mesh_vertnum[i])
        adr = int(m.mesh_graphadr[i])
        nhull = int(m.mesh_graph[adr]) if adr >= 0 else -1
        npoly = int(m.mesh_polynum[i])
        polyadr = int(m.mesh_polyadr[i])
        lines.append(f"mesh\t{key[0]}\t{name}\t{nvert}\t{nhull}\t{npoly}")
        for p in range(npoly):
            n = m.mesh_polynormal[polyadr + p]
            lines.append("poly\t%d\t%r\t%r\t%r" % (int(m.mesh_polyvertnum[polyadr + p]), *[float(x) for x in n]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
nm = sum(1 for l in lines if l.startswith("mesh"))
print(f"{models} models, {nm} distinct collision meshes -> {out}")
