"""Dump MuJoCo's own kinematics for every Menagerie model as the oracle a Rust loader is measured against.

For each compilable XML: the body tree (name, parent), the joints (name, type, body, qpos address, range) and,
for K sampled configurations, every body's world pose from `mj_kinematics`. One JSON per model.
"""
import json, os, sys, glob, math
import numpy as np
import mujoco

root = sys.argv[1]
out_dir = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rng = np.random.default_rng(20260914)
os.makedirs(out_dir, exist_ok=True)

JT = {mujoco.mjtJoint.mjJNT_FREE: "free", mujoco.mjtJoint.mjJNT_BALL: "ball",
      mujoco.mjtJoint.mjJNT_SLIDE: "slide", mujoco.mjtJoint.mjJNT_HINGE: "hinge"}

def sample_qpos(m):
    q = np.array(m.qpos0, dtype=float)
    for j in range(m.njnt):
        t = m.jnt_type[j]; a = m.jnt_qposadr[j]
        if t == mujoco.mjtJoint.mjJNT_HINGE or t == mujoco.mjtJoint.mjJNT_SLIDE:
            if m.jnt_limited[j]:
                lo, hi = m.jnt_range[j]
                q[a] = rng.uniform(lo, hi)
            else:
                q[a] = rng.uniform(-math.pi, math.pi) if t == mujoco.mjtJoint.mjJNT_HINGE else rng.uniform(-0.5, 0.5)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            v = rng.normal(size=4); v /= np.linalg.norm(v); q[a:a+4] = v
        elif t == mujoco.mjtJoint.mjJNT_FREE:
            q[a:a+3] = rng.uniform(-1.0, 1.0, size=3)
            v = rng.normal(size=4); v /= np.linalg.norm(v); q[a+3:a+7] = v
    return q

files = sorted(glob.glob(os.path.join(root, "*", "*.xml")))
summary = {"compiled": [], "failed": {}}
for f in files:
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception as e:  # noqa: BLE001 — the reason is the datum
        summary["failed"][rel] = str(e).splitlines()[0][:200]
        continue
    # the parity question is the rigid-body dynamics: no contacts, no equality constraints, no passive
    # springs/gravity-compensation, no actuators — gravity and the joint inertias only
    # (damping stays ON: it is a joint term both sides model; springs, frictionloss, gravity compensation,
    # contacts and actuators are not part of the rigid-body question)
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                           | mujoco.mjtDisableBit.mjDSBL_SPRING | mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    m.body_gravcomp[:] = 0.0
    d = mujoco.MjData(m)
    bodies = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}",
               "parent": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[b])) or "world",
               "mass": float(m.body_mass[b]),
               "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(),
               "ipos": m.body_ipos[b].tolist(), "iquat": m.body_iquat[b].tolist(), "inertia": m.body_inertia[b].tolist()}
              for b in range(1, m.nbody)]
    joints = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}",
               "type": JT[int(m.jnt_type[j])],
               "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])),
               "qposadr": int(m.jnt_qposadr[j]), "dofadr": int(m.jnt_dofadr[j]),
               "axis": m.jnt_axis[j].tolist(), "pos": m.jnt_pos[j].tolist(),
               "limited": bool(m.jnt_limited[j]), "range": m.jnt_range[j].tolist(),
               "ref": float(m.qpos0[m.jnt_qposadr[j]]) if int(m.jnt_type[j]) in (2, 3) else None}
              for j in range(m.njnt)]
    sites = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or f"site{s}",
              "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[s])) or "world"}
             for s in range(m.nsite)]
    samples = []
    for _ in range(K):
        q = sample_qpos(m)
        d.qpos[:] = q
        d.qvel[:] = rng.uniform(-1.0, 1.0, size=m.nv)
        d.ctrl[:] = 0
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
        samples.append({"qpos": q.tolist(), "qvel": d.qvel.tolist(),
                        "xpos": d.xpos[1:].tolist(), "xquat": d.xquat[1:].tolist(),
                        "site_xpos": d.site_xpos.tolist(),
                        "qfrc_bias": d.qfrc_bias.tolist(), "qacc": d.qacc.tolist(), "M": M.reshape(-1).tolist()})
    rec = {"file": rel, "nq": int(m.nq), "nv": int(m.nv), "nbody": int(m.nbody) - 1, "njnt": int(m.njnt),
           "compiler_angle": None, "bodies": bodies, "joints": joints, "sites": sites, "samples": samples}
    op = os.path.join(out_dir, rel.replace("/", "__") + ".json")
    with open(op, "w") as fh:
        json.dump(rec, fh)
    # flat text twin: one token stream a dependency-free Rust example can read with split_whitespace
    with open(op[:-5] + ".txt", "w") as fh:
        fh.write(f"model {rel} {m.nq} {m.nbody-1} {m.njnt} {m.nsite}\n")
        for b in bodies:
            fh.write(f"body {b['name']} {b['parent']} {b['mass']!r}\n")
        for j in joints:
            fh.write(f"joint {j['name']} {j['type']} {j['body']} {j['qposadr']} {j['dofadr']}\n")
        for b in bodies:
            fh.write(f"binertia {b['name']} " + " ".join(repr(x) for x in [b['mass']] + b['ipos'] + b['iquat'] + b['inertia']) + "\n")
        fh.write(f"nv {m.nv}\n")
        fh.write("gravity " + " ".join(repr(float(x)) for x in m.opt.gravity) + "\n")
        for s in sites:
            fh.write(f"site {s['name']} {s['body']}\n")
        for smp in samples:
            fh.write("sample " + " ".join(repr(x) for x in smp['qpos']) + "\n")
            for b, p, qq in zip(bodies, smp['xpos'], smp['xquat']):
                fh.write(f"xpos {b['name']} " + " ".join(repr(x) for x in p + qq) + "\n")
            for s, p in zip(sites, smp['site_xpos']):
                fh.write(f"sxpos {s['name']} " + " ".join(repr(x) for x in p) + "\n")
            fh.write("qvel " + " ".join(repr(x) for x in smp['qvel']) + "\n")
            fh.write("bias " + " ".join(repr(x) for x in smp['qfrc_bias']) + "\n")
            fh.write("qacc " + " ".join(repr(x) for x in smp['qacc']) + "\n")
            fh.write("M " + " ".join(repr(x) for x in smp['M']) + "\n")
    summary["compiled"].append(rel)
with open(os.path.join(out_dir, "_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(f"compiled {len(summary['compiled'])} / failed {len(summary['failed'])} of {len(files)} xml files")
for k, v in list(summary["failed"].items())[:15]:
    print("  FAIL", k, "->", v)
