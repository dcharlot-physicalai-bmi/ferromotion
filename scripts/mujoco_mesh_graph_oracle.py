"""MuJoCo's `mesh_graph` — qhull's hull in qhull's own vertex, neighbour and facet order — per collision mesh,
the oracle for `examples/menagerie_qhull.rs`.

    python scripts/mujoco_mesh_graph_oracle.py <menagerie root> <out.txt>

One `graph <model dir> <mesh name> <ints>` line per distinct collision mesh (`-1` when MuJoCo stored no graph).
Run it with the working directory somewhere disposable: MuJoCo writes MUJOCO_LOG.TXT there on warnings.
"""
import mujoco, os, sys, glob
root, out = sys.argv[1], sys.argv[2]
root = os.path.abspath(root)  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice
lines, seen = [], set()
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        i = int(m.geom_dataid[g])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
        key = (rel.split("/")[0], name)
        if key in seen:
            continue
        seen.add(key)
        adr = int(m.mesh_graphadr[i])
        if adr < 0:
            lines.append(f"graph\t{key[0]}\t{name}\t-1"); continue
        nv, nf = int(m.mesh_graph[adr]), int(m.mesh_graph[adr + 1])
        g_ = m.mesh_graph[adr: adr + 2 + 3 * nv + 6 * nf]
        lines.append("graph\t%s\t%s\t%s" % (key[0], name, " ".join(str(int(x)) for x in g_)))
open(out, "w").write("\n".join(lines) + "\n")
print(len(lines), "graphs")
