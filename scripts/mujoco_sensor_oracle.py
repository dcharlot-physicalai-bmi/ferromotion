"""MuJoCo's `sensordata`, for every compilable Menagerie model that declares sensors.

    python scripts/mujoco_sensor_oracle.py <menagerie root> <out.txt> [samples per model]

Per model: one `sensor` line per sensor — name, type, dim, adr, datatype, the object and reference it reads
(type and name), cutoff — and the joints with their `qposadr`. Per sampled state (the forward oracle's
sampling: hinges and slides jiggled, the last state moving, the solver told to converge): `qpos`, `qvel`,
`qacc` and `sensordata` after `mj_forward`, which runs `mj_sensorPos`, `mj_sensorVel` and `mj_sensorAcc` —
`qacc` so an acceleration-stage sensor can be judged apart from the solve it reads.
Tab separated.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.default_rng(20260927)
JT = {v: k.replace("mjJNT_", "").lower() for k, v in mujoco.mjtJoint.__members__.items()}
ST = {int(v): k.replace("mjSENS_", "").lower() for k, v in mujoco.mjtSensor.__members__.items()}
OT = {int(v): k.replace("mjOBJ_", "").lower() for k, v in mujoco.mjtObj.__members__.items()}
DT = {int(v): k.replace("mjDATATYPE_", "").lower() for k, v in mujoco.mjtDataType.__members__.items()}

lines, models, failed = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    if m.nsensor == 0:
        continue
    d = mujoco.MjData(m)
    m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = 1000, 50, 0.0
    models += 1
    lines.append(f"model\t{rel}\t{m.nq}\t{m.nv}\t{m.nsensor}\t{m.nsensordata}")
    for j in range(m.njnt):
        lines.append("\t".join(["joint", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    for i in range(m.nsensor):
        ot, oid, rt, rid = int(m.sensor_objtype[i]), int(m.sensor_objid[i]), int(m.sensor_reftype[i]), int(m.sensor_refid[i])
        oname = (mujoco.mj_id2name(m, ot, oid) or "") if oid >= 0 and ot in OT else ""
        rname = (mujoco.mj_id2name(m, rt, rid) or "") if rid >= 0 and rt in OT else ""
        lines.append("\t".join(["sensor", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or "", ST[int(m.sensor_type[i])],
                                str(int(m.sensor_dim[i])), str(int(m.sensor_adr[i])), DT[int(m.sensor_datatype[i])],
                                OT.get(ot, str(ot)), oname, OT.get(rt, str(rt)) if rid >= 0 else "-", rname, repr(float(m.sensor_cutoff[i]))]))
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    a = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
            if k > 1:
                d.qvel[:] = rng.uniform(-0.4, 0.4, m.nv)
        mujoco.mj_forward(m, d)
        lines.append(f"state\t{k}")
        for tag, v in (("qpos", d.qpos), ("qvel", d.qvel), ("qacc", d.qacc), ("sensordata", d.sensordata)):
            lines.append("\t".join([tag] + [repr(float(x)) for x in v]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models with sensors ({failed} would not compile), {K} states each -> {out}")
