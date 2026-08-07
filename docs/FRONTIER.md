# Ferromotion — the SOTA Frontier Map

> **⚑ CURRENT STATUS (v0.32.0, 2026-07-25 — supersedes everything below).** The detailed
> sections that follow are the historical v0.21-era sweep; **many items marked 🔲 missing in
> them are in fact shipped.** A full 2026-07-25 re-review (capability-map over all crates +
> global landscape: Pinocchio/Parry/Rapier/Dojo/Newton/Genesis/cuRobo/OMPL/GTSAM/Isaac) found
> ferromotion is the closest thing to a Rust-native Pinocchio + control + estimation + planning +
> differentiable-physics stack, wasm-clean. Scale: **15 crates, ~860 test fns, v0.32.0 on
> crates.io.**
>
> **Shipped (confirmed in source), not to rebuild:** FK/Jac/RNEA/ABA/floating-base + analytic
> dynamics derivatives + generic-scalar `gendyn` + **CRBA**; constraint/contact (Delassus, PGS,
> ADMM, IPC/LCP/IPM/hydroelastic, C3); collision **GJK/EPA/CCD/DiffCol/SDF/ESDF/C-space-SDF** +
> **BVH broadphase**; URDF **and MJCF and SDFormat**; the ~64-controller stack (iLQR/FDDP/boxDDP/
> ALGAMES, MPPI/CEM/iCEM/DIAL, tube/SRBD/centroidal MPC, CBF/HJ, CCM, WBC/OSC/HQP, ZMP/DCM/ALIP/
> SLIP) + **batched/data-parallel rollouts**; estimation EKF/UKF/InEKF/MSCKF/iSAM/pose-graph/GNC +
> **particle filter + moving-horizon estimation**; planning RRT*/BIT*/CHOMP/GPMP2/IRIS/GCS/
> Hybrid-A*/Dubins/Reeds-Shepp + **PRM***; deformables MPM/cloth/rod/tactile + **volumetric
> tetrahedral Neo-Hookean FEM** + **DEM granular**; the full verified fluid suite (MAC/LBM-2D&3D+GPU/
> SPH/VOF/spectral/Euler/FVM/unstructured-NS/DMD/harness); geometric vision PnP/essential/homography/
> BA/ORB/AprilTag/SGM/ICP/KISS-ICP/TEASER; learn AD (reverse+forward+hyperdual)/PINN/DeLaN/HNN/
> Neural-ODE/SINDy/calib + **smoothed (randomized) differentiable contact**; **sensor rendering**
> (depth-camera/lidar/segmentation by SDF sphere-tracing); **3-D mesh** (area/volume/convex hull).
>
> **Genuinely remaining (short):** SIMPLE/PISO unstructured incompressible momentum (fluids have
> incompressible NS via MAC + spectral + Taylor-Hood already); interactively-authenticated MCP
> caveats aside — nothing else is a *capability* gap. Residual polish: WebGPU throughput legs
> (sensor render, batched rollouts), tetrahedral-mesh generation, non-frozen-active-set contact for
> the constraint solver proper (the smoothing estimator exists in `learn`), MPR narrowphase.

---

*What a complete, state-of-the-art Rust library for the motion of physical AI needs — and where we stand.*

This document is the canonical roadmap. It was produced by sweeping the full research
landscape across six domains (optimal control, legged locomotion, manipulation, motion
planning, estimation & simulation, learning & domain-specific motion), cross-checked
against the live library inventory, and de-duplicated. Every item is a *genuinely
distinct* capability with a canonical citation and a note on how it verifies in pure
Rust (`nalgebra` / `faer` / `clarabel`) against an analytic answer or a physical
invariant — the library's standing discipline.

**Scope of "complete":** cover the model-based motion stack end to end — represent a
body, simulate it (rigid, contact, deformable, fluid), estimate its state from sensors,
plan through the world, control it under constraints, and bridge to learned policies —
each layer inverted toward the energy-first / on-device / verifiable posture of the
Institute.

Legend: ✅ shipped · 🟡 partial / adjacent exists · 🔲 missing (ranked).

---

## 0. What we already have (as of v0.21.0 + unreleased)

- **Dynamics/sim:** ABA forward dynamics, analytic dynamics derivatives, LGVI, modal
  analysis, closed-loop constraints; contact IPM, IPC, DCOL, planar/robot contact;
  MLS-MPM (differentiable, forward-mode material grad), FEM cloth, discrete elastic
  rods, Cosserat, tensegrity, 2-D Navier–Stokes fluid, photometric tactile sim.
- **Optimal control / trajopt / MPC:** iLQR, ProxDDP, DIRCON, TOPP, SCvx, CTR, C3,
  trajectory bundles, CEM, MPPI; MPC, SRBD-MPC, centroidal MPC, TinyMPC, ReLU-QP,
  ALGAMES, es-MPC, **Distributed Potential iLQR**; LQR, H∞, sliding-mode, Koopman,
  Hamilton–Jacobi reachability, CBF; QP/HQP; ruckig jerk-limited OTG.
- **Locomotion / WBC:** ZMP, DCM, capture point, SLIP, WBC, OSC, HQP, pink/placo IK.
- **Manipulation:** force-closure / Ferrari–Canny Q1, dexterous retargeting, tactile
  servo, admittance, visual servoing, RMPflow.
- **Planning / geometry:** RRT (basic), analytic SDF + collision spheres, **composite
  C-space SDF**, FOCI (Gaussian-splat collision), **Perceptive MIQP footsteps**.
- **Estimation:** InEKF, IMU preintegration, batch + fixed-lag factor-graph smoothers,
  UKF/EKF, complementary filter, momentum observer, sysid.
- **Learning:** flow-matching policy, real-time chunking, DMP; **a reverse-mode autodiff
  tape + differentiable soft-body adjoint**.
- **Domains:** quadrotor, Fossen marine, swarm/consensus, CDPR cable robots.

---

## 1. Top cross-cutting priorities

These surfaced as highest-leverage across *multiple* sweeps, or unblock large families.

| # | Capability | Why it tops the list | Verify against |
|---|-----------|----------------------|----------------|
| 1 | **FDDP / box-FDDP** (Crocoddyl) | The globalization that makes whole-body legged MPC actually converge from bad warm-starts; appeared in 3 sweeps | LTI ⇒ exact LQR; dynamics gaps → 0 |
| 2 | **GJK / EPA + CCD** narrowphase | Geometric bedrock every engine/planner calls; we have no general convex distance/penetration or swept collision | closed-form sphere/box/capsule distance |
| 3 | **IRIS → GCS** convex free-space | Enabling primitive for seed-free global trajopt & our own C-space work; clarabel-native SDP/conic | region contains no obstacle; single-region GCS = QP optimum |
| 4 | **RRT-Connect / RRT\* / BIT\*** | We only have basic RRT — no optimal/complete global planner for cluttered/narrow-passage | cost → analytic geodesic optimum |
| 5 | **ALIP + H-LIP** | Underactuated/point-foot dynamic walking templates — a regime none of our current templates cover | closed-form S2S LTI; deadbeat eigenvalues |
| 6 | **Screw-theory + manipulability toolkit** | The SE(3) twist/wrench/adjoint algebra + Yoshikawa measures the whole manipulation stack is written in | exp∘log = id; PoE FK = DH FK; analytic 2R ellipsoid |
| 7 | **iSAM2 + MSCKF/VIO** | Incremental smoothing + our *zero* exteroceptive estimation — the real-time SLAM/odometry backend | incremental = batch re-solve (in-repo oracle) |
| 8 | **Differentiable contact stepper** (Dojo-style) | Ties our contact energies + derivatives into contact-implicit trajopt & sysid | energy/momentum conservation; grad vs FD |

---

## 2. Optimal control / MPC / trajopt

- 🔲 **FDDP — feasibility-driven DDP** (Mastalli et al., ICRA 2020) — multiple-shooting,
  gap-tolerant globalization; the community default for legged MPC.
- 🔲 **ALTRO / AL-iLQR + projected-Newton polish** (Howell et al., IROS 2019) —
  constrained trajopt with high-accuracy constraint satisfaction.
- 🔲 **Box-DDP** (Tassa et al., ICRA 2014) — control-limited DDP; cheap actuator limits.
- 🔲 **Real-Time Iteration NMPC** (Diehl et al., 2005) — preparation/feedback split; the
  scheme (acados) that runs NMPC at kHz.
- 🔲 **Tube MPC** (Mayne et al., 2005/2011) — robust invariant-tube constraint tightening.
- 🔲 **MPCC** — model-predictive contouring (Liniger et al., 2015) — online time-optimal
  path following (racing, agile flight); distinct from our offline TOPP.
- 🔲 **iCEM** (Pinneri et al., CoRL 2020) — colored-noise, elite-memory CEM; deployable
  sampling-MPC.
- 🔲 **Variational contact-implicit trajopt** (Manchester et al., IJRR 2019) — through-
  contact, no mode schedule, symplectic (conservation invariants to test).
- 🔲 **Chance-constrained / stochastic MPC** — probabilistic constraint tightening;
  Monte-Carlo violation-rate oracle.
- 🔲 **Differentiable MPC** (Amos et al., NeurIPS 2018) — MPC as a learnable layer; KKT
  sensitivities vs FD. *(flagged by two sweeps)*
- 🔲 **Covariance steering** (Chen–Georgiou–Pavon) — drive terminal *covariance* to a
  target; analytic Riccati oracle.
- 🔲 **DeePC** (Coulson et al., 2019) — data-driven predictive control; exact equivalence
  to model MPC on LTI (fundamental lemma) — a superb invariant test.
- 🔲 **Risk-sensitive iLQG / iLEQG** (Farshidian–Buchli, 2015) — exponential-cost DDP
  robustness knob.
- 🔲 **GuSTO** (Bonalli et al., ICRA 2019) — SCP with free-final-time + continuous-time
  guarantees; lower marginal value given SCvx.

## 3. Legged locomotion & whole-body control

- 🔲 **ALIP** (Gong–Grizzle 2020) & **H-LIP** (Xiong–Ames, T-RO 2022) — angular-momentum
  templates for underactuated walking.
- 🔲 **HZD + virtual constraints** (Westervelt/Ames/Grizzle) — provable periodic-orbit
  gaits; Poincaré return-map spectral test.
- 🔲 **TOWR** (Winkler et al., RA-L 2018) — phase-based single-NLP gait/foothold discovery
  over terrain.
- 🔲 **MPC↔WBC bridge — Whole-Body Impulse Control** (Kim et al., MIT Cheetah 3) — the glue
  that runs our SRBD-MPC + WBC together on hardware. *High-leverage integration gap.*
- 🔲 **Step-timing adaptation** (Khadiv et al., T-RO 2020) — adapt *when* and *where* to step.
- 🔲 **N-step capturability** (Koolen et al., IJRR 2012) — multi-step capture regions;
  we have only 1-step capture point.
- 🔲 **VHIP + ICI** (Caron et al., T-RO 2019) — variable-height balancing template.
- 🔲 **Raibert heuristic** (1986) — the canonical hopping/running stepping law on SLIP.
- 🔲 **Central Pattern Generators** (Ijspeert 2008) — coupled-oscillator gaits & transitions.
- 🔲 **Learned-locomotion (RL) interface + RMA** (Kumar et al., RSS 2021) — obs/action/
  adaptation hooks feeding the WBC.
- 🔲 **Fall detection / protective stepping / get-up** — the deployability safety layer.

## 4. Manipulation, grasping, dexterity, tactile

> Status audited 2026-08-05 against the source tree. Several items marked open here had in fact shipped, so the
> checkboxes below name the module that closed them. An unaudited roadmap overstates what is left and understates
> what is done, and both directions cost work.

- ✅ **Screw-theory / Lie-group kinematics toolkit** (PoE, twists, wrenches, adjoint) — twists and adjoints in
  `control/visual_servo.rs`, `control/placo.rs`; wrench algebra in `core/grasp_spatial.rs`.
- ✅ **Manipulability ellipsoids & measures** (Yoshikawa 1985) — `core/manipulability.rs`.
- ✅ **Antipodal / analytic grasp sampling** (GPD, Dex-Net) — `core/grasp.rs`, exercised by `wasm/grasp_lab.rs`.
- ✅ **Grasp matrix + internal-force decomposition** — `core/grasp_spatial.rs` (`grasp_matrix`, `grasp_split`,
  `wrench_rank`), cross-validated against the hand-object contact loop to `1.5e11x` separation.
- ✅ **Task-oriented + probabilistic grasp quality** — `force_closure_soft_spatial` alongside the hard
  `force_closure_q1_spatial`; the smoothed metric falls when a contact is lifted, which was measured rather than
  assumed (the opposite was asserted first and was wrong).
- ✅ **Tactile slip detection + slip-aware force regulation** (Dong et al., 2019) — `tactile/shear.rs`
  (Cattaneo–Mindlin partial slip; `SlipSignal::incipient` warns at 35% of capacity) plus `control/slip.rs`.
- ✅ **Hierarchical (task-priority) whole-body QP** — `control/hierarchy.rs`, with the two-rate interface and delay
  margin now driven by a real clock in `policy/clock.rs`.
- 🔲 **Soft-finger contact / friction limit surface** (Xydas–Kao 1999) — torsion-coupled contact realism. Not built:
  `shear.rs` covers tangential partial slip but not the torsional-coupling limit surface. What exists is the
  **decoupled polyhedral** form in `grasp_spatial.rs`: independent bounds `|f_t| ≤ μ·f_n` and `|m_n| ≤ μ_t·f_n` as
  separate generators, which is what makes the wrench set a polytope and is exactly the coupling Xydas–Kao supplies.
  A correction landed here on 2026-08-06: the torsional generator was scaled as though the normal force were `1`,
  while the cone edges are normalised to unit **total** force and so deliver `f_n = 1/√(1+μ²)` (matched to `2.2e-16`
  over `μ ∈ [0,2]`). The mixed convention overstated torsion by `√(1+μ²)` — `1.80×` at `μ=1.5`. Cost, measured:
  `Q1` is exactly flat in `μ_t` until torsion **binds** (from `μ_t ≈ 0.28` at `μ=0.3`, `≈ 0.70` at `μ=1.0`, never up
  to `4.0` at `μ=1.5`), so the largest premise error sits where it costs nothing; above the threshold the old form
  overstated `Q1` by **+10.2%** at `μ=0.5, μ_t=1.0`. Hard contacts are bit-identical, so no published hard-grasp
  number moves. The first probe sampled only `μ_t ∈ {0.1, 0.3}` — all inside the flat region — and reported "no
  change" in twelve cases; the finding required sweeping past the binding threshold.
- 🔲 **Dexterous grasp synthesis via differentiable force closure** (DexGraspNet) — invent whole-hand grasps. We can
  score and split a grasp; we cannot yet synthesise one by descent.
- 🔲 **Caging / energy-bounded caging** — topological grasp guarantee without force closure.
- 🔲 **In-hand manipulation / finger-gaiting / regrasp** — manifold-switching dexterity. `core/hand_object.rs` gives
  the contact loop a gait would sit on; the gait itself is absent.
- 🔲 **Tactile localization / SLAM** (GPIS + factor graph) — pose/shape from touch.
- 🔲 **Stable-pushing / non-prehensile planner** (Lynch–Mason) — on our pusher-slider model.
- 🔲 **TAMP** (PDDLStream / LGP) — symbolic↔geometric multi-step planning. *(harder to verify)*
- 🔲 **Deformable-object manipulation** (cloth/rope shape servoing).

## 5. Motion planning, collision & geometry

- 🔲 **GJK / EPA / MPR + CCD** — convex narrowphase + swept collision (anti-tunneling).
- 🔲 **RRT-Connect / RRT\* / Informed-RRT\* / BIT\*** — the real global planner family.
- 🔲 **IRIS / IRIS-NP** (Deits–Tedrake) — convex free-space region inflation (clarabel SDP).
- 🔲 **GCS — Graphs of Convex Sets** (Marcucci et al., Science Robotics 2023) — seed-free
  global trajopt around obstacles. *A genuine moat (no pure-Rust impl exists).*
- 🔲 **CHOMP / STOMP** (Ratliff 2009 / Kalakrishnan 2011) — SDF-consuming trajectory
  optimizers; nearly free on our SDF.
- 🔲 **C-IRIS + SOS certified collision-free regions** — *proven* safe regions (auditable).
- 🔲 **Incremental ESDF mapping** (Voxblox/FIESTA/nvblox) — distance fields *from sensors*.
- 🔲 **GPMP2** — Gaussian-process motion planning as inference.
- 🔲 **Reeds-Shepp / Dubins + Hybrid A\* + state lattices** — nonholonomic/kinodynamic.
- 🔲 **Safe Flight Corridors** — task-space convex corridor + QP (agile flight).
- 🔲 **PRM / PRM\*** — multi-query roadmaps.
- 🔲 **VAMP-style SIMD collision validation** — kHz sampling planners on CPU.
- 🔲 **Homotopy / H-signature planning** — distinct-route reasoning (tethers, deconfliction).
- 🔲 **Time-Elastic-Band** — online trajectory-level reactive replanning.

## 6. State estimation & SLAM

- 🔲 **iSAM2 — incremental smoothing (Bayes tree)** — constant-time real-time backend;
  verify against our in-repo batch oracle.
- 🔲 **MSCKF / VIO** — visual-inertial odometry; we have *zero* exteroceptive fusion.
- 🔲 **Robust kernels + GNC** (Yang et al., 2020) — outlier-robust graph optimization; drops
  onto `sparse.rs`. *Fast high-value win.*
- 🔲 **Point-cloud / LiDAR odometry** — point-to-plane ICP / GICP / NDT (SE(3) registration).
- 🔲 **Moving-Horizon Estimation** — constrained receding-horizon estimator (reuses QP).
- 🔲 **Particle / Rao-Blackwellized filter** — non-Gaussian/global localization.

## 6b. Fresh sweep, 2026-08-05

What the current literature says that changes what we should build. Recorded with the claim we tested, because a
citation is not a result.

- **DiffMJX / Contacts-from-Distance** (arXiv 2506.14186, ICLR 2026). Tolerance-driven adaptive integration cuts
  penalty-contact gradient error by orders of magnitude. **Tested and reproduced** in `core/adaptive_contact.rs`:
  1.59e5x over four decades of tolerance, and the tolerance route reaches the closed-form saltation Jacobian to three
  digits. This overturned our own attribution, not theirs. ⛔ Do NOT quote "~199 steps against 8000 fixed" as a saving:
  199 is ONE rollout's accepted steps and the Jacobian needs eight, so the honest cost is ~14,805 force evaluations
  against the fixed route's 8,000 — **1.85x MORE work**, buying the right answer rather than a cheaper one. Their CFD straight-through trick for
  *pre-contact* gradients is still unbuilt and is the obvious next piece.
- ✅ **Certified contact-rich manipulation via smoothing-error reachable tubes** (arXiv 2602.09368, Li & Chou, RSS
  2026). Plan on smoothed dynamics, bound the smoothing error, certify under the *original* nonsmooth dynamics. Built
  as `control/smoothing_tube.rs` on top of `control/zonotope.rs`, measured in
  `examples/smoothing_tube_certificate.rs`. What it found:
  - ⛔ **BOTH VERDICTS WITHDRAWN (2026-08-06).** This section recorded that a gap of `3.2e-1` at `k = 1e4` refutes a
    ceiling constraint while `9.5e-3` at `k = 1e6` would certify it at margin `2.6e-2`. Neither holds. The decisive
    evidence is `escaping_sample`: **the tube does not contain the penalty trajectory its own gap was measured from**,
    outside by `6.09 mm` and `4.87 m/s` one millisecond after impact — `2.48x` its own half-width. A tube that misses
    the real trajectory cannot certify, and cannot refute either (the k=1e4 "violation" is fabricated: a 460 mm ceiling
    reported breached by 1 mm when the true apex is 398 mm with 62 mm of slack).
    The cause is structural. The measured "gap" is the two models differenced at ONE instant, and what it measures is a
    **time offset of one contact duration** — `gap_dv = 3.157e-1 m/s` against `g*tau = 3.139e-1 m/s`, agreeing to
    `0.57%`. A time reparametrisation is not an additive state disturbance and **no per-step `W` can be one**: matching
    the endpoint needs a small box, bounding the path needs a large one, and Minkowski sums cannot cancel. Representing
    it needs a saltation-style timing term, which is not built.
    What survives and is tested: the tube algebra, the exact support test, the evidence asymmetry, derived soundness,
    the empty-tube refusal, the region preconditions, and `escaping_sample` itself. **A tube is not a certificate until
    it has passed containment against a trajectory the true system actually takes.**
  - **That is the same regime where a fixed-step gradient is unusable.** At `k = 1e6` fixed-step autodiff reports
    `dv/dh = -209.76` against a true `+3.65`: wrong sign, 57x magnitude. So the only stiffness at which the tube is
    tight enough to certify is a stiffness at which you must have the exact or tolerance-driven Jacobian. That is the
    bridge, measured rather than argued.
  - **Certifying with the wrong Jacobian inflates the tube 2.38x** at `k = 1e6` (pessimistic here, not dangerous — a
    certificate you could have had and did not get).
  - 🔲 **THE TOP OPEN ITEM: a sound gap bound. Soundness is currently *relocated*, not eliminated.** Every sampled
    verdict is `Undecided { GapOnlySampled }` by design — sampling gives a *lower* bound on the gap, so it can refute
    and must never certify. But `GapBound::from_lipschitz` is the only producer of `Proved` evidence and it **verifies
    nothing**: it stamps `Proved` on whatever half-widths it is handed, so passing sampled numbers through it certifies.
    The accompanying example and the shipped `certificate_lab` do exactly that on purpose, labelled as assumed, to show
    what a proof *would* buy. The type moves the unsupported step to one named, auditable call site; it does not
    discharge it. An adversarial audit named this precisely and it is the correct framing.

    What is needed is a computable, provable upper bound on
    `sup_{x in X} ||Phi_penalty(x,T) - Phi_rigid(x,T)||`, over a set of entry states, for a stiff one-sided
    spring-damper against a rigid impact at the *measured* restitution. Until that exists, **every certificate in this
    workspace that rests on a smoothing gap is conditional**, and the conditional is worth stating in any paper.
    Target: the bound must exceed the sampled `9.5e-3` at `k = 1e6` (a proved bound below a sampled measurement is a
    proof of a bug, not a tighter result) while staying under the `2.6e-2` constraint margin — and it must additionally
    pass `escaping_sample`, which the current fixture does not — or the certificate is
    sound and vacuous.

  - Two defects the same audit found in the tube itself, both fixed: `certify()` read a trusted `pub bool` and never
    consulted the recorded evidence, so a hand-built report certified regardless of its gaps; and
    `propagate_tube(x0, &[])` certified every constraint because the evidence loop never ran. Soundness is now derived
    from a per-step evidence record, and an empty tube is unsound by construction.
  - Two traps encoded as tests: a sampled bound can never return `Certified` however wide the margin, and
    `nominal_activity` exposes a **vacuous** certificate. The first version of the example certified a ceiling over a
    60 ms horizon when the mass needed 285 ms to reach its apex; the verdict was `Certified` and the constraint was
    unreachable.
- **Newton 1.0** (GTC 2026, Linux Foundation; NVIDIA + Google DeepMind + Disney Research). MuJoCo-Warp and Kamino
  solvers, a Vertex Block Descent deformable solver, SDF collision, hydroelastic contact, OpenUSD throughout. Our
  ingest lines up: `cloth/vbd.rs`, `core/hydroelastic.rs`, `core/esdf.rs`, `core/usda.rs`. Treat as the reference
  solver list, not a target.
- **MuJoCoUni** (arXiv 2605.24922, Tsinghua). Persistent batched runtime primitives: per-environment model copies,
  sparse reset, reset-lifecycle domain randomisation, batched sensor evaluation without advancing dynamics. This is
  the shape of the batched-engine gap we recorded and deferred (single-copy model buffers mean all N worlds are the
  same world).
- **GPU-parallel linearization error bounds** (arXiv 2607.01203) and **certified deformable MPC** (arXiv 2606.14188)
  — both are error-bound-carrying control, the same discipline as our certificates.
- **Action-chunking latency**: RTC, REMAC/masked action chunking, and VLA-Perf. `policy/clock.rs` covers the
  execution side; corrective adjustment under asynchronous mismatch is unbuilt.

## 7. Differentiable simulation & physics

- 🔲 **Differentiable contact-implicit stepper** (Dojo) — the flagship dynamics primitive.
- 🔲 **Analytic LCP/contact gradients** (Belbute-Peres 2018; ADD 2020) — differentiable
  contact node for sysid.
- 🔲 **XPBD / Position-Based Dynamics** — unified fast compliant constraint solver; very
  WASM-friendly. *Fast win.*
- 🔲 **Tetrahedral co-rotational FEM** — true volumetric soft solids (we have shell/particle).
- 🔲 **Batched / data-parallel rigid-body dynamics** (Brax/MJX) — parallel rollouts for RL;
  bit-identical-lane determinism test.
- 🔲 **DEM — granular / sand** (Cundall–Strack) — locomotion on granular media.
- 🔲 **Fluid-structure two-way coupling** — buoyancy/drag reaction (with our marine control).
- 🔲 **Full MLS-MPM reverse-AD adjoint** — thread the new AD tape through `MpmSim`
  (needs nalgebra generic-scalar plumbing); the forward-mode material grad already exists.

## 8. Learning for control & domain-specific motion

- 🔲 **Control Contraction Metrics** (Manchester–Slotine 2017) — certified nonlinear tracking.
- 🔲 **GP dynamics + PILCO** — data-efficient model-based RL; uncertainty-aware MPC. *(AD-heavy)*
- 🔲 **Adaptive control — L1 / MRAC** — online model-mismatch robustness.
- 🔲 **Residual policy learning** — learn a correction on top of our classical controllers.
- 🔲 **DAgger + behavior-cloning utilities** — imitation data for the flow/RTC policies.
- 🔲 **Neural ODE / continuous learned dynamics** — gated on a reverse-AD decision *(the new
  autodiff tape is the enabler)*.
- 🔲 **Geometric SE(3) tracking + differential-flatness min-snap** (Lee 2010 / Mellinger 2011)
  — the aerial reference (SE(3) coupling + the trajectory *generator*).
- 🔲 **ORCA / reciprocal velocity obstacles** — decentralized reactive multi-robot avoidance.
- 🔲 **Clohessy–Wiltshire rendezvous + reaction-wheel momentum management** — spacecraft
  *translation* + actuator allocation (we have attitude only); closed-form Φ(t) oracle.
- 🔲 **Piecewise-constant-curvature + tendon-driven continuum kinematics** — control-oriented
  soft-arm kinematics; cross-checks against our Cosserat rod in the constant-strain limit.

---

## Suggested sequencing

1. **Foundational, no-solver, exact-oracle quick wins:** screw/manipulability toolkit,
   GJK/EPA+CCD, RRT-Connect/RRT\*, XPBD, GNC robust kernels, ALIP/Raibert/CPG, CHOMP.
2. **clarabel-native convex-geometry line:** IRIS → GCS → Safe Flight Corridors →
   C-IRIS (the certified-region moat), reusing the composite C-space SDF.
3. **Whole-body/legged deployment tier:** FDDP/box-DDP → WBIC bridge → step-timing → TOWR.
4. **Perception backbone:** iSAM2 → GNC → ICP/GICP → MSCKF VIO → MHE.
5. **Differentiable-physics & learning:** differentiable contact stepper → analytic LCP
   gradients → full MLS-MPM adjoint (via the AD tape) → Differentiable MPC → CCM / L1 /
   residual / DAgger.
6. **Domains:** SE(3)+min-snap, ORCA, Clohessy–Wiltshire, PCC continuum, DEM, FSI.

Every item above is verifiable in pure Rust against a closed-form solution or a physical
invariant — which is the bar the library holds itself to, and the reason a user can trust
that what they are driving is the real thing.
