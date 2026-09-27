"""MuJoCo's OWN cost function, evaluated at a port's answer.

    DUMP_QACC=/tmp/ours.txt cargo run --release --example menagerie_efc -- <efc oracle txt>
    python scripts/mujoco_cost_at.py <menagerie root> <efc oracle txt> /tmp/ours.txt

⛔⛔ "our cost is no worse than MuJoCo's" is only as good as OUR cost. The rows can be MuJoCo's to the
last bit and the verdict still be circular, because the map from a row's residual to a cost is the port's
own — and a friction or contact row's LINEAR TAILS, the part that only bites once a row is driven past
its bound, are exactly where two implementations diverge. This asks `mj_constraintUpdate` instead: the
state is rebuilt from the oracle, `jar = efc_J·a − efc_aref` is formed at the port's acceleration with
MuJoCo's own (possibly sparse) Jacobian, and MuJoCo scores it. The quadratic half is added here from
`mj_fullM`, which is the one piece `mj_constraintUpdate` does not carry.
"""

import os
import sys

import mujoco
import numpy as np

root, oracle, ours_path = sys.argv[1], sys.argv[2], sys.argv[3]
root = os.path.abspath(root)  # ⛔ MuJoCo resolves a nested <include> against a RELATIVE path twice

# the port's answers, keyed by model and state
ours = {}
for line in open(ours_path):
    t = line.rstrip("\n").split("\t")
    if t[0] == "ours":
        ours[(t[1], int(t[2]))] = np.array([float(x) for x in t[3:]])

# the states, from the same oracle the port read
cases, cur = [], None
for line in open(oracle):
    t = line.rstrip("\n").split("\t")
    if t[0] == "case":
        cur = {"rel": t[1], "state": int(t[2])}
        cases.append(cur)
    elif t[0] in ("qpos", "qvel", "qacc") and cur is not None:
        cur[t[0]] = np.array([float(x) for x in t[1:]])

models, worst = {}, []
for c in cases:
    key = (c["rel"], c["state"])
    if key not in ours:
        continue
    if c["rel"] not in models:
        m = mujoco.MjModel.from_xml_path(f"{root}/{c['rel']}")
        models[c["rel"]] = (m, mujoco.MjData(m))
    m, d = models[c["rel"]]
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:] = c["qpos"], c["qvel"]
    mujoco.mj_forward(m, d)
    if d.nefc == 0:
        continue
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    a_smooth = d.qacc_smooth.copy()

    def cost(a):
        jar = np.zeros(d.nefc)
        mujoco.mj_mulJacVec(m, d, jar, a)
        jar -= d.efc_aref[: d.nefc]
        out = np.zeros(1)
        mujoco.mj_constraintUpdate(m, d, jar, out, 0)
        da = a - a_smooth
        return float(out[0] + 0.5 * da @ M @ da)

    theirs, mine = cost(c["qacc"]), cost(ours[key])
    err = np.abs(ours[key] - c["qacc"]).max() / max(1.0, np.abs(c["qacc"]).max())
    if err > 1e-6:
        worst.append((theirs - mine, theirs, mine, err, c["rel"], c["state"]))

worst.sort(key=lambda r: -abs(r[0]))
print(f"{len(worst)} states where the two answers differ by more than 1e-6, scored by MUJOCO'S cost:")
for gap, theirs, mine, err, rel, st in worst:
    verdict = "ours is LOWER" if gap > 0 else ("equal" if gap == 0 else "⛔ ours is HIGHER — ours is wrong")
    print(f"  {rel} [{st}]: qacc off {err:.2e}; mujoco {theirs!r} vs ours {mine!r}; gap {gap:+.6e} — {verdict}")
higher = [w for w in worst if w[0] < 0]
print(f"{len(higher)} of {len(worst)} cost MORE under MuJoCo's own cost function")
