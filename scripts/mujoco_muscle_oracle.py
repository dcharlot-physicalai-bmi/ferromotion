"""MuJoCo's own muscle curves, sampled — `mju_muscleGain`, `mju_muscleBias`, `mju_muscleDynamics`.

    python scripts/mujoco_muscle_oracle.py <menagerie root> <out.txt>

A muscle is three piecewise curves and a stiff activation filter, and every one of them has branches a port
can get right on the middle of the range and wrong at an end. The grid here is chosen to land in EVERY
branch: the normalised length below `lmin`, in each of the four quadratic segments of the active
force-length curve, and above `lmax`; the normalised velocity at and past both ends of the force-velocity
curve; the passive curve on both sides of its linear knee; and the activation filter with and without
smoothing, with `ctrl` above and below `act`.

The parameter sets are MuJoCo's defaults plus the ones `ms_human_700` actually ships, and the last block is
every compiled muscle in that model against its own `lengthrange` and `acc0` — so the port is checked on the
curves AND on the numbers the compiler put in front of them.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
root = os.path.abspath(root)  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice
lines = []

# prm = (range[0], range[1], force, scale, lmin, lmax, vmax, fpmax, fvmax)
PRM = [
    [0.75, 1.05, -1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],   # MuJoCo's defaults: force < 0, so `scale/acc0`
    [0.75, 1.05, 625.82, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],  # ms_human_700's addbrev_r
    [0.6, 1.4, 1000.0, 150.0, 0.4, 1.8, 2.0, 0.9, 1.4],    # a set with every knee moved
    [1.0, 1.0, 500.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2],     # range[1] == range[0]: the MINVAL guard
]
LR = [[0.1, 0.3], [0.086661, 0.189083], [0.5, 0.5]]
ACC0 = [0.8993203068802182, 1.0, 1e-16]

for prm in PRM:
    for lr in LR:
        for acc0 in ACC0:
            # normalised length from below lmin to above lmax, and the raw length that produces it
            L0 = (lr[1] - lr[0]) / max(1e-15, prm[1] - prm[0])
            for Ln in np.linspace(prm[4] - 0.3, prm[5] + 0.3, 37):
                length = lr[0] + (Ln - prm[0]) * L0
                for V in np.linspace(-1.6, prm[8] + 0.4, 23):
                    vel = V * L0 * prm[6]
                    g = mujoco.mju_muscleGain(length, vel, np.array(lr), acc0, np.array(prm))
                    b = mujoco.mju_muscleBias(length, np.array(lr), acc0, np.array(prm))
                    lines.append("\t".join(["curve"] + [repr(float(x)) for x in (length, vel, *lr, acc0, *prm, g, b)]))

for dynprm in ([0.01, 0.04, 0.0], [0.01, 0.04, 0.1], [0.05, 0.02, 0.5], [0.0, 0.0, 0.0]):
    for ctrl in np.linspace(-0.4, 1.4, 19):
        for act in np.linspace(-0.4, 1.4, 19):
            a = mujoco.mju_muscleDynamics(ctrl, act, np.array(dynprm))
            lines.append("\t".join(["dyn"] + [repr(float(x)) for x in (ctrl, act, *dynprm, a)]))

# every compiled muscle in every model that has one: the parameters MuJoCo's COMPILER produced
models = 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        continue
    if not any(int(t) == int(mujoco.mjtGain.mjGAIN_MUSCLE) for t in m.actuator_gaintype):
        continue
    models += 1
    # ⭐ the transmission at `qpos0` alongside the muscle parameters. `acc0` is `‖M⁻¹·moment‖` there, so a
    # port whose `acc0` is wrong has either the formula wrong or the MOMENT wrong, and only these two
    # numbers together tell which. On `ms_human_700` they say it is the moment: every muscle whose `acc0`
    # differs is on a spatial tendon whose PATH already disagrees.
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    mom = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(mom, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}"
        lines.append("\t".join(["compiled", rel, name, str(int(m.actuator_gaintype[a])), str(int(m.actuator_biastype[a])), str(int(m.actuator_dyntype[a]))]
                               + [repr(float(x)) for x in m.actuator_gainprm[a][:9]]
                               + [repr(float(x)) for x in m.actuator_biasprm[a][:9]]
                               + [repr(float(x)) for x in m.actuator_dynprm[a][:3]]
                               + [repr(float(x)) for x in m.actuator_lengthrange[a]]
                               + [repr(float(m.actuator_acc0[a]))]
                               + [repr(float(d.actuator_length[a])), repr(float(np.linalg.norm(mom[a])))]))

with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{len(lines)} rows from {models} models with muscles -> {out}")
