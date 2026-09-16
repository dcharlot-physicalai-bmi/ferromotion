"""MuJoCo's friction cones — the elliptic cone and condim 4/6 — as the pinned oracle for `mujoco_contact.rs`.

    python scripts/mujoco_cone_probe.py     (needs the `mujoco` wheel; numbers in the tests are 3.13.0's)

For each case it prints the per-row constraint data MuJoCo derives (`efc_R`, `efc_D`, `efc_aref`, the
contact's regularized `mu`), the solved `efc_force`, and `qacc`. A row's state is the zone the cone
projection put it in: 0 satisfied, 1 quadratic, 2 linear-neg, 3 linear-pos, 4 cone.

⛔ The `efc_state` left in `mjData` by `mj_forward` is STALE — it is whatever the Newton solver's last
internal evaluation set, not the state at the `qacc` it returned. A sliding elliptic contact is reported as
`cone` while its own `efc_force` is exactly the quadratic branch's. So this probe re-runs
`mj_constraintUpdate` at `jar = J·qacc − a_ref` and prints THAT state, which is the one a port must match;
the solver's leftover is printed alongside it as `stale_state` and must not be pinned.
"""

import mujoco
import numpy as np

STATE = {0: "satisfied", 1: "quadratic", 2: "linearneg", 3: "linearpos", 4: "cone"}


def run(label, xml, qvel=None, nstep=0):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if qvel is not None:
        d.qvel[:] = qvel
    for _ in range(nstep):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    cone = "elliptic" if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal"
    J = np.asarray(d.efc_J).reshape(d.nefc, m.nv) if d.nefc else np.zeros((0, m.nv))
    qacc = np.array(d.qacc)
    stale = [int(x) for x in d.efc_state]
    jar = J @ qacc - d.efc_aref
    cost = np.zeros(1)
    if d.nefc:
        mujoco.mj_constraintUpdate(m, d, jar, cost, 0)
    print(f"=== {label}: cone={cone} impratio={m.opt.impratio!r} ncon={d.ncon} nefc={d.nefc}")
    for i in range(d.ncon):
        c = d.contact[i]
        print(f"  contact {i}: dim={c.dim} dist={c.dist!r} mu={c.mu!r} friction={[repr(float(x)) for x in c.friction]}")
        print(f"    pos={[repr(float(x)) for x in c.pos]} normal={[repr(float(x)) for x in c.frame[:3]]}")
    for i in range(d.nefc):
        print(
            f"  efc[{i}] type={int(d.efc_type[i])} id={int(d.efc_id[i])} R={float(d.efc_R[i])!r} D={float(d.efc_D[i])!r} "
            f"aref={float(d.efc_aref[i])!r} pos={float(d.efc_pos[i])!r} margin={float(d.efc_margin[i])!r} "
            f"force={float(d.efc_force[i])!r} state={STATE.get(int(d.efc_state[i]), int(d.efc_state[i]))}"
            f" stale_state={STATE.get(stale[i], stale[i])} jar={float(jar[i])!r}"
            f" J={[repr(float(x)) for x in np.asarray(d.efc_J).reshape(d.nefc, m.nv)[i]]}"
        )
    print(f"  cost={float(cost[0])!r}")
    print(f"  qacc={[repr(float(x)) for x in qacc]}")
    print(f"  qacc_smooth={[repr(float(x)) for x in d.qacc_smooth]}")
    print(f"  qvel={[repr(float(x)) for x in d.qvel]}")
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    for r in range(m.nv):
        print(f"  M[{r}]={[repr(float(x)) for x in M[r]]}")


def scene(body, option="", floor_friction="1 0.005 0.0001"):
    return f"""<mujoco>
<option timestep="0.002" jacobian="dense" {option}/>
<worldbody>
  <geom name="floor" type="plane" size="5 5 0.1" friction="{floor_friction}"/>
  {body}
</worldbody>
</mujoco>"""


def ball(condim, pos, vel="0 0 0 0 0 0", friction="1 0.005 0.0001", extra=""):
    return f"""<body name="b" pos="{pos}"><freejoint/>
      <geom name="s" type="sphere" size="0.1" condim="{condim}" friction="{friction}" {extra}/>
    </body>"""


# --- the elliptic cone, one contact, condim 3 --------------------------------------------------------
run("elliptic condim3 at rest", scene(ball(3, "0 0 0.095"), option='cone="elliptic"'))
run("elliptic condim3 sliding", scene(ball(3, "0 0 0.095"), option='cone="elliptic"'), qvel=[0.4, 0.2, 0.0, 0.0, 0.0, 0.0])
run(
    "elliptic condim3 impratio 3",
    scene(ball(3, "0 0 0.095"), option='cone="elliptic" impratio="3"'),
)
run(
    "pyramidal condim3 impratio 3 (same model, other cone)",
    scene(ball(3, "0 0 0.095"), option='cone="pyramidal" impratio="3"'),
)

# --- condim 4 (torsional friction) and condim 6 (torsional + rolling) --------------------------------
SPIN = [0.4, 0.2, 0.0, 0.3, -0.1, 0.9]
run("elliptic condim4 sliding and spinning", scene(ball(4, "0 0 0.095"), option='cone="elliptic"'), qvel=SPIN)
run("elliptic condim6 sliding and spinning", scene(ball(6, "0 0 0.095"), option='cone="elliptic"'), qvel=SPIN)
run("pyramidal condim4 sliding and spinning", scene(ball(4, "0 0 0.095"), option='cone="pyramidal"'), qvel=SPIN)
run("pyramidal condim6 sliding and spinning", scene(ball(6, "0 0 0.095"), option='cone="pyramidal"'), qvel=SPIN)
run("elliptic condim3 slow slide (inside the cone)", scene(ball(3, "0 0 0.095"), option='cone="elliptic"'), qvel=[0.002, 0.0, 0.0, 0.0, 0.0, 0.0])

# --- anisotropic friction, which is what makes the elliptic R's differ per row -----------------------
run(
    "elliptic condim6 anisotropic friction",
    scene(ball(6, "0 0 0.095", friction="0.8 0.01 0.002"), option='cone="elliptic"', floor_friction="0.5 0.004 0.0005"),
    qvel=SPIN,
)

# --- a box on the floor: four contacts, so the cone acts on a real manifold --------------------------
run(
    "elliptic box on floor, four contacts",
    scene('<body name="b" pos="0 0 0.0995"><freejoint/><geom name="bx" type="box" size="0.2 0.3 0.1" condim="3"/></body>', option='cone="elliptic"'),
    qvel=[0.3, 0.1, 0.0, 0.0, 0.0, 0.2],
)

# --- in motion: a few steps of a sliding, spinning ball, where the cone's zones actually differ ------
run(
    "elliptic condim6 after 25 steps of sliding and spin",
    scene('<body name="b" pos="0 0 0.0995"><freejoint/><geom name="s" type="sphere" size="0.1" condim="6"/></body>', option='cone="elliptic"'),
    qvel=SPIN,
    nstep=25,
)
run(
    "pyramidal condim6 after 25 steps of sliding and spin",
    scene('<body name="b" pos="0 0 0.0995"><freejoint/><geom name="s" type="sphere" size="0.1" condim="6"/></body>', option='cone="pyramidal"'),
    qvel=SPIN,
    nstep=25,
)
