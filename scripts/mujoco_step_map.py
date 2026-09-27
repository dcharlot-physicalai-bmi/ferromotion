"""MuJoCo's step MAP along the port's own trajectory: is `mj_step` from each of the port's states the port's
next state?

    MENAGERIE_STEP_TRACE=1 cargo run --release --example menagerie_step -- <menagerie root> <step oracle txt> > ours.txt
    python scripts/mujoco_step_map.py <menagerie root> <step oracle txt> ours.txt

⛔⛔ Comparing two TRAJECTORIES cannot tell a wrong step from a sensitive system. MuJoCo's own step is not
continuous in the state everywhere: from a state 1e-16 away from MuJoCo's own, `rethink_robotics_sawyer`'s
next state lands 1.4e-7 away (qvel 3.1e-5), and `robotiq_2f85`'s 9e-4 away — in MuJoCo, stepping from the
port's state. So after the first such state the trajectories part however right the port is. This asks the
question that does not compound: at every state the port visits, does MuJoCo, handed that state, step to the
port's next one? Each step starts from `mj_resetData` (no warmstart), with the solver told to converge as
`mujoco_step_oracle.py` tells it, the port's `time`, the oracle's control (zeroed after a reset, as
`mj_resetData` zeroes it) and the port's activations.
"""

import os
import sys

import mujoco
import numpy as np

root, oracle, trace = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]

# the starts and controls the oracle used
cases, cur = {}, None
for line in open(oracle):
    t = line.rstrip("\n").split("\t")
    if t[0] == "model":
        cur = cases.setdefault(t[1], {})
    elif t[0] in ("start", "startv", "ctrl", "startact") and cur is not None:
        cur[t[0]] = np.array([float(x) for x in t[1:] if x != ""])

# the port's trajectory: (qpos, qvel, act, time, reset) after each step
ours = {}
for line in open(trace):
    t = line.rstrip("\n").split("\t")
    if t[0] not in ("trace", "tracev", "traceact"):
        continue
    s = ours.setdefault(t[1], {}).setdefault(int(t[2]), {})
    if t[0] == "trace":
        s["qpos"] = np.array([float(x) for x in t[5:] if x != ""])
    elif t[0] == "tracev":
        s["qvel"] = np.array([float(x) for x in t[3:] if x != ""])
    else:
        s["time"], s["reset"] = float(t[3]), t[4] == "true"
        s["act"] = np.array([float(x) for x in t[5:] if x != ""])

rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)), initial=0.0))
W = mujoco.mjtWarning
bad = (W.mjWARN_BADQPOS, W.mjWARN_BADQVEL, W.mjWARN_BADQACC)
steps, within8, within4, worst, notes, both_reset, one_reset = 0, 0, 0, (0.0, ""), [], 0, 0
per_model = []
for name, traj in sorted(ours.items()):
    c = cases.get(name)
    if c is None:
        continue
    m = mujoco.MjModel.from_xml_path(os.path.join(root, name))
    m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = 1000, 50, 0.0
    d = mujoco.MjData(m)
    ctrl = c.get("ctrl", np.zeros(m.nu))
    prev = {"qpos": c["start"], "qvel": c["startv"], "act": c.get("startact", np.zeros(m.na)), "time": 0.0, "reset": False}
    model_worst = 0.0
    for k in sorted(traj):
        s = traj[k]
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.time = prev["qpos"], prev["qvel"], prev["time"]
        if m.na:
            d.act[:] = prev["act"]
        d.ctrl[:] = ctrl
        for w in bad:
            d.warning[w].number = 0
        mujoco.mj_step(m, d)
        mj_reset = any(d.warning[w].number for w in bad)
        if s["reset"] or mj_reset:
            if s["reset"] and mj_reset:
                both_reset += 1
            else:
                one_reset += 1
                notes.append(f"{name} step {k}: {'only the port' if s['reset'] else 'only MuJoCo'} reset")
            # ⛔ `mj_resetData` zeroes the control for the rest of the run
            ctrl = np.zeros(m.nu)
            prev = s
            continue
        e = max(rel(s["qpos"], d.qpos), rel(s["qvel"], d.qvel), rel(s["act"], d.act) if m.na else 0.0)
        steps += 1
        within8 += e < 1e-8
        within4 += e < 1e-4
        model_worst = max(model_worst, e)
        if e > worst[0]:
            worst = (e, f"{name} step {k}")
        if e >= 1e-8 and len(notes) < 30:
            notes.append(f"{name} step {k}: off {e:.2e}")
        prev = s
    per_model.append((model_worst, name))
print(f"MuJoCo 3.13.0 stepped from each of the port's states: {within8} of {steps} steps within 1e-8, {within4} within 1e-4 (worst {worst[0]:.2e} on {worst[1]}), over {len(per_model)} models")
print(f"  resets: {both_reset} steps where both reset, {one_reset} where only one did")
for n in notes:
    print("  " + n)
