"""MuJoCo's contact SET on Menagerie at sampled states — the oracle for a collision-pipeline port.
Per compilable model and sample: every mjContact with geom ids/types, dist, pos, frame, dim, includemargin,
friction, solref, solimp, exclude, and the normal force; plus the geom census and geom poses."""
import json, os, sys, glob, math, collections
import numpy as np, mujoco

root, out_dir = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rng = np.random.default_rng(20260914)
os.makedirs(out_dir, exist_ok=True)
GT = {v: k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}

def sample_qpos(m):
    q = np.array(m.qpos0, dtype=float)
    for j in range(m.njnt):
        t = int(m.jnt_type[j]); a = int(m.jnt_qposadr[j])
        if t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]; q[a] = rng.uniform(lo, hi)
            else:
                q[a] = rng.uniform(-math.pi, math.pi) if t == mujoco.mjtJoint.mjJNT_HINGE else rng.uniform(-0.5, 0.5)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            v = rng.normal(size=4); v /= np.linalg.norm(v); q[a:a+4] = v
        elif t == mujoco.mjtJoint.mjJNT_FREE:
            q[a:a+3] = rng.uniform(-0.3, 0.3, size=3) + m.qpos0[a:a+3]
            v = rng.normal(size=4); v /= np.linalg.norm(v); q[a+3:a+7] = v
    return q

pair_census = collections.Counter(); summary = {"compiled": [], "failed": {}}
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception as e:
        summary["failed"][rel] = str(e).splitlines()[0][:160]; continue
    d = mujoco.MjData(m)
    geoms = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", "type": GT[int(m.geom_type[g])],
              "body": (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ("world" if m.geom_bodyid[g] == 0 else f"body{int(m.geom_bodyid[g])}")),
              "size": m.geom_size[g].tolist(), "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g]),
              "condim": int(m.geom_condim[g]), "margin": float(m.geom_margin[g]), "gap": float(m.geom_gap[g]),
              "priority": int(m.geom_priority[g]), "solmix": float(m.geom_solmix[g]),
              "friction": m.geom_friction[g].tolist(), "solref": m.geom_solref[g].tolist(), "solimp": m.geom_solimp[g].tolist(),
              "rbound": float(m.geom_rbound[g]), "adhesion": float(m.geom_adhesion[g]), "aabb": m.geom_aabb[g].tolist(),
              "mesh": (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None)}
             for g in range(m.ngeom)]
    samples = []
    for _ in range(K):
        d.qpos[:] = sample_qpos(m); d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        cons = []
        for i in range(d.ncon):
            c = d.contact[i]
            fn = 0.0
            if c.efc_address >= 0:
                nrow = 1 if c.dim == 1 else (2 * (c.dim - 1) if m.opt.cone == 0 else 1)
                fn = float(sum(d.efc_force[c.efc_address:c.efc_address + nrow]))
            cons.append({"g1": int(c.geom[0]), "g2": int(c.geom[1]), "dist": float(c.dist), "pos": c.pos.tolist(), "frame": c.frame.tolist(),
                         "dim": int(c.dim), "includemargin": float(c.includemargin), "friction": c.friction.tolist(),
                         "solref": c.solref.tolist(), "solimp": c.solimp.tolist(), "exclude": int(c.exclude), "fn": fn})
            pair_census[tuple(sorted((geoms[c.geom[0]]["type"], geoms[c.geom[1]]["type"])))] += 1
        samples.append({"qpos": d.qpos.tolist(), "geom_xpos": d.geom_xpos.tolist(), "geom_xmat": d.geom_xmat.tolist(), "contacts": cons})
    meshes = {(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"): {"pos": m.mesh_pos[i].tolist(), "quat": m.mesh_quat[i].tolist(), "nvert": int(m.mesh_vertnum[i])} for i in range(m.nmesh)}
    rec = {"file": rel, "cone": int(m.opt.cone), "meshes": meshes,
           "filterparent": not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT),
           "nativeccd": not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD),
           "multiccd": not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD),
           "ccd_tolerance": float(m.opt.ccd_tolerance), "ccd_iterations": int(m.opt.ccd_iterations),
           "o_margin": float(m.opt.o_margin), "override": bool(m.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE),
           "npair": int(m.npair), "nexclude": int(m.nexclude), "geoms": geoms, "samples": samples}
    with open(os.path.join(out_dir, rel.replace("/", "__") + ".json"), "w") as fh:
        json.dump(rec, fh)
    summary["compiled"].append(rel)
summary["pair_census"] = {"+".join(k): v for k, v in pair_census.most_common()}
with open(os.path.join(out_dir, "_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"compiled {len(summary['compiled'])} / failed {len(summary['failed'])}")
print("contact pair types:", summary["pair_census"])
