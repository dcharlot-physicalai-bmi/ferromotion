"""MuJoCo's `qacc` at rest — the oracle for a whole forward step, from an MJCF file to an acceleration.

    python scripts/mujoco_forward_oracle.py <menagerie root> <out.txt> [samples per model]

Per compilable model and per sampled state: the joints with their `qposadr`, `qpos`, `qvel`, MuJoCo's
`qfrc_bias`, `qacc_smooth` and `qacc`, and the gates a port has to respect to be comparing the same thing —
how many constraint rows are contacts rather than limits or equalities, whether any actuator or passive force
is acting, the cone, and `impratio`. Every contact is dumped with its geom ids, `dist`, `pos` and the whole
3x3 `frame`, so a port can be checked on the tangent pair and not only the normal. Tab separated, one record
per line.
"""

import glob
import os
import sys

import mujoco
import numpy as np

root, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.default_rng(20260917)
JT = {v: k.replace("mjJNT_", "").lower() for k, v in mujoco.mjtJoint.__members__.items()}

lines, models, failed = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, "*", "*.xml"))):
    rel = os.path.relpath(f, root)
    try:
        m = mujoco.MjModel.from_xml_path(f)
    except Exception:
        failed += 1
        continue
    d = mujoco.MjData(m)
    models += 1
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    # ⛔⛔ the SOLVER SETTINGS are part of the problem statement, not a tuning knob. 29 of Menagerie's
    # 210 compilable models cap `iterations` (every `*_mjx.xml` at 1 or 5), so MuJoCo's `qacc` on
    # them is ONE Newton iterate from a zero warm start, not the optimum of the rows it built. A
    # port that solves to convergence is then compared against a number MuJoCo never claimed was
    # converged, and the gap reads as a physics error. Recorded so the sweep can tell them apart.
    lines.append(f"model\t{rel}\t{m.nq}\t{m.nv}\t{int(m.opt.cone)}\t{float(m.opt.impratio)!r}\t{float(m.opt.gravity[0])!r}\t{float(m.opt.gravity[1])!r}\t{float(m.opt.gravity[2])!r}\t{int(m.opt.iterations)}\t{int(m.opt.ls_iterations)}\t{int(m.opt.solver)}\t{float(m.opt.tolerance)!r}")
    # ⭐ `body_invweight0` with the body, because every contact row's regularisation is built from the two
    # bodies' translational entries. It is compiled ONCE at `qpos0`, so a port that recomputes it at the
    # current state is exact on the reference pose and drifts everywhere else — a gate that only samples
    # `qpos0` cannot see that, and `efc_D` is the only place it shows.
    for b in range(m.nbody):
        # ⛔ an UNNAMED body is written with an empty name and left uncompared. Filling in `body{id}`
        # pairs it with whatever the port happens to call its own unnamed bodies, which is a guess: the
        # two numberings agree until they do not, and then the sweep compares two different bodies.
        # ⭐ and the inertia MuJoCo actually STORES: the mass, the three principal values and the frame
        # they are diagonal in. Not the tensor the file states — the compiler diagonalises with `mjuu_eig3`
        # and keeps the eigendecomposition, so this is what the dynamics read.
        lines.append("\t".join(["body", name(mujoco.mjtObj.mjOBJ_BODY, b) or "",
                                 repr(float(m.body_invweight0[b][0])), repr(float(m.body_invweight0[b][1])),
                                 repr(float(m.body_mass[b])),
                                 *[repr(float(x)) for x in m.body_inertia[b]],
                                 *[repr(float(x)) for x in m.body_iquat[b]],
                                 *[repr(float(x)) for x in m.body_ipos[b]],
                                 # ⛔ whether the body's WELD has no dofs — the exact condition under which
                                 # MuJoCo leaves `body_invweight0` at zero, so a port that keeps no entry
                                 # for it agrees. Recorded so the gate's denominator can be RECONCILED
                                 # against the corpus instead of asserted.
                                 # ⛔⛔ "welded to the world" is NOT the same test: a MOCAP body is its own
                                 # weld (`body_weldid[b] == b`), has no dofs, and MuJoCo gives it a ZERO
                                 # inverse weight anyway. `franka_emika_panda/mjx_single_cube` has one, it
                                 # carries a colliding geom, and it is the single body the weld-to-world
                                 # test could not account for.
                                 str(int(m.body_dofnum[m.body_weldid[b]] == 0))]))
    for j in range(m.njnt):
        lines.append("\t".join(["joint", name(mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}", JT[int(m.jnt_type[j])], str(int(m.jnt_qposadr[j]))]))
    for k in range(K):
        mujoco.mj_resetData(m, d)
        if k:
            # the same sampling the contact oracle uses: hinges and slides jiggled, free joints left alone
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    a = int(m.jnt_qposadr[j])
                    lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-0.5, 0.5)
                    d.qpos[a] = rng.uniform(lo, hi)
            # ⛔ a state at rest cannot tell a velocity-dependent term from a missing one: damping and
            # joint friction are both zero at qvel = 0, so sample the last state MOVING
            if k > 1:
                d.qvel[:] = rng.uniform(-0.4, 0.4, m.nv)
        mujoco.mj_forward(m, d)
        # which constraint rows are contacts: anything else needs machinery this port does not claim
        CT = {int(v): k.replace("mjCNSTR_", "").lower() for k, v in mujoco.mjtConstraint.__members__.items()}
        hist = {}
        for t in d.efc_type[: d.nefc]:
            hist[CT[int(t)]] = hist.get(CT[int(t)], 0) + 1
        ncontact = sum(v for k, v in hist.items() if k.startswith("contact"))
        dims = sorted({int(c.dim) for c in d.contact[: d.ncon]})
        lines.append("\t".join([
            "state", str(k), str(int(d.nefc)), str(ncontact), str(int(d.ncon)),
            repr(float(np.abs(d.qfrc_actuator).max(initial=0.0))),
            repr(float(np.abs(d.qfrc_passive).max(initial=0.0))),
            ",".join(str(x) for x in dims) or "-",
            ";".join(f"{k}={v}" for k, v in sorted(hist.items())) or "-",
        ]))
        # ⛔ the body accelerations in the WORLD frame: the only comparison that survives a different dof
        # basis. A free joint is six coordinates here and six there, but not the SAME six.
        mujoco.mj_rnePostConstraint(m, d)
        acc = []
        for b in range(m.nbody):
            res = np.zeros(6)
            mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            acc.extend(res)
        lines.append("\t".join(["bodyacc"] + [repr(float(x)) for x in acc]))
        # ⭐ `qfrc_passive` is a SUM, and MuJoCo keeps its parts: a port that misses it by 3e-4 is missing
        # one of these four, and the total cannot say which.
        for tag, v in (("qpos", d.qpos), ("qvel", d.qvel), ("qfrc_bias", d.qfrc_bias), ("qfrc_passive", d.qfrc_passive),
                       ("qfrc_spring", d.qfrc_spring), ("qfrc_damper", d.qfrc_damper), ("qfrc_gravcomp", d.qfrc_gravcomp), ("qfrc_fluid", d.qfrc_fluid),
                       ("qacc_smooth", d.qacc_smooth), ("qacc", d.qacc)):
            lines.append("\t".join([tag] + [repr(float(x)) for x in v]))
        # ⭐ the SAME problem solved to convergence: what the rows MuJoCo built actually imply. On an
        # `*_mjx.xml` it is 18-74% away from `qacc`. Restored afterwards so nothing downstream sees the
        # raised limits.
        #
        # ⛔⛔ `tolerance` too, not only `iterations`. This used to keep the model's `tolerance` (1e-8 by
        # default), claiming the result was "`qacc` to the last bit" on every uncapped model. It is not:
        # MuJoCo's default tolerance ends the main solve with `qacc` still 2e-5 to 6e-5 from the optimum on
        # `i2rt_yam`, `toddlerbot_2xc` and `hello_robot_stretch(_3)` (measured by `examples/menagerie_efc`
        # against the same state at tolerance 0), and a port that solves to 1e-14 is charged for it.
        it0, ls0, tol0 = int(m.opt.iterations), int(m.opt.ls_iterations), float(m.opt.tolerance)
        qpos_k, qvel_k = d.qpos.copy(), d.qvel.copy()
        m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = max(it0, 500), max(ls0, 50), 0.0
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = qpos_k, qvel_k
        mujoco.mj_forward(m, d)
        lines.append("\t".join(["qacc_converged"] + [repr(float(x)) for x in d.qacc]))
        m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance = it0, ls0, tol0
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = qpos_k, qvel_k
        mujoco.mj_forward(m, d)
        # ⛔ the contact FRAME, not only its normal. Under a pyramidal cone the friction rows are
        # J_n ± μ_k·J_tk, so rotating the tangent pair about the normal rotates the pyramid and changes
        # the feasible force set: two ports can agree on every normal and still solve different problems.
        for ci in range(int(d.ncon)):
            c = d.contact[ci]
            # ⛔ a contact inside the gap band is in `d.contact` and has NO rows: `efc_address` is -1, and
            # the rows of every LATER contact are still where `efc_address` says, not where counting says
            a = int(c.efc_address)
            n = 0 if a < 0 else (1 if int(c.dim) == 1 else (2 * (int(c.dim) - 1) if int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL) else int(c.dim)))
            lines.append("\t".join(["contact", str(int(c.geom[0])), str(int(c.geom[1])), repr(float(c.dist))]
                                   + [repr(float(x)) for x in c.pos] + [repr(float(x)) for x in c.frame]
                                   + [str(int(c.dim)), repr(float(c.includemargin)), repr(float(c.mu))]
                                   + [repr(float(x)) for x in c.friction] + [str(a), str(n)]
                                   + [repr(float(x)) for x in d.efc_D[a:a + n]] + [repr(float(x)) for x in d.efc_aref[a:a + n]]))
with open(out, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"{models} models ({failed} would not compile), {K} states each -> {out}")
