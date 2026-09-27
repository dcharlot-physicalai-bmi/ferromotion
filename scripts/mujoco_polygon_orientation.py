"""Are MuJoCo's own `mesh_polynormal`s outward? — a check with no port in it.

    python scripts/mujoco_polygon_orientation.py <menagerie root>

A polygon normal is outward exactly when every hull corner is on or behind the polygon's own plane.
Nothing about that needs a second implementation: MuJoCo's `mesh_polyvert` gives the hull's corners and
`mesh_polynormal` gives the normal, so the question is answered inside MuJoCo's own data.

⛔ The threshold is 1e-3 OF THE PART, not machine epsilon. MuJoCo stores mesh vertices as `float`, and its
own polygon planes miss its own corners by up to ~2.4e-5 of the part's size; at 1e-9 this test reports
thousands of "inward" normals that are only storage precision. A genuinely reversed normal puts a corner a
whole polytope width outside, which is nowhere near either number.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root = sys.argv[1]
root = os.path.abspath(root)  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice
tot = bad = 0
worst = (0.0, "")
seen = set()
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    d0 = os.path.basename(os.path.dirname(f))
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    for i in range(m.nmesh):
        npoly = int(m.mesh_polynum[i])
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
        if npoly == 0 or (d0, name) in seen:
            continue
        seen.add((d0, name))
        va, nv = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
        V = m.mesh_vert[va:va + nv].astype(np.float64)
        extent = max(np.abs(V).max(), 1e-9)
        pa = int(m.mesh_polyadr[i])
        corners = set()
        for p in range(pa, pa + npoly):
            a, n = int(m.mesh_polyvertadr[p]), int(m.mesh_polyvertnum[p])
            corners.update(int(x) for x in m.mesh_polyvert[a:a + n])
        H = V[sorted(corners)]
        for p in range(pa, pa + npoly):
            a = int(m.mesh_polyvertadr[p])
            nrm = m.mesh_polynormal[p].astype(np.float64)
            slack = (H @ nrm - nrm @ V[int(m.mesh_polyvert[a])]).max() / extent
            tot += 1
            if slack > 1e-3:
                bad += 1
                if slack > worst[0]:
                    worst = (slack, f"{d0}/{name} poly{p - pa}")
print(f"{tot} polygons over {len(seen)} meshes")
print(f"  {bad} put one of MuJoCo's OWN hull corners more than 1e-3 of the part outside the polygon's own plane")
print(f"  worst {worst[0]:.3e} on {worst[1]}")
