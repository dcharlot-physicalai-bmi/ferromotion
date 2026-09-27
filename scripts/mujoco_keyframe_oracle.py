"""MuJoCo's compiled keyframes, for every compilable Menagerie model that declares one.

    python scripts/mujoco_keyframe_oracle.py <menagerie root> <out.txt>

Per model: `model rel nkey`, then per key `key name time`, `qpos ...`, `qvel ...`, `act ...`, `ctrl ...` —
what `mjCKey::Compile` produced, the unstated parts filled in. Tab separated.
"""

import glob
import os
import sys

import mujoco

root, out = sys.argv[1], sys.argv[2]
lines, models = [], 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    if m.nkey == 0:
        continue
    models += 1
    lines.append(f"model\t{rel}\t{m.nkey}")
    for k in range(m.nkey):
        lines.append("\t".join(["key", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "", repr(float(m.key_time[k]))]))
        for tag, v in (("qpos", m.key_qpos[k]), ("qvel", m.key_qvel[k]), ("act", m.key_act[k]), ("ctrl", m.key_ctrl[k])):
            lines.append("\t".join([tag] + [repr(float(x)) for x in v]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models with keyframes -> {out}")
