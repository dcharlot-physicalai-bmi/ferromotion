# Ferromotion — the SOTA Frontier Map

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

- 🔲 **Screw-theory / Lie-group kinematics toolkit** (PoE, twists, wrenches, adjoint) —
  the algebra everything else is written in. *(quick win, exact tests)*
- 🔲 **Manipulability ellipsoids & measures** (Yoshikawa 1985) — singularity/posture signal.
- 🔲 **Antipodal / analytic grasp sampling** (GPD, Dex-Net) — *generate* grasp candidates
  (we can only score with Q1).
- 🔲 **Soft-finger contact / friction limit surface** (Xydas–Kao 1999) — torsion-coupled
  contact realism.
- 🔲 **Task-oriented + probabilistic grasp quality** (TWS, robust-ε) — beyond origin-ball Q1.
- 🔲 **Dexterous grasp synthesis via differentiable force closure** (DexGraspNet) — invent
  whole-hand grasps.
- 🔲 **Caging / energy-bounded caging** — topological grasp guarantee without force closure.
- 🔲 **In-hand manipulation / finger-gaiting / regrasp** — manifold-switching dexterity.
- 🔲 **Grasp matrix + internal-force decomposition** — bimanual/multi-contact wrench split.
- 🔲 **Hierarchical (task-priority) whole-body QP** — prioritized redundant multi-task control.
- 🔲 **Tactile slip detection + slip-aware force regulation** (Dong et al., 2019).
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
