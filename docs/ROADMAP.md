# Ferromotion — Roadmap, v0.30.0 cycle (2026-07-22)

_Fresh audit: full workspace inventory (185 modules / 12 crates / 754 tests) + two web sweeps
(classical robotics stacks; differentiable physics & physics-informed learning, both mid-2026).
Supersedes the v0.23-era plan below, whose Tier-1 list is now essentially shipped (iCEM, DIAL-MPC,
CCM, differentiable MPC, BIT*, C3, CTR, FOCI, IPM contact, hydroelastic, GJK/EPA/DCOL, InEKF/MSCKF,
GCS/IRIS, ALGAMES/DP-iLQR, …)._

## The strategic read (both sweeps corroborate)

- **The Rust model-based core is an open lane.** Mid-2026 web sweep verdict: no Rust-native
  equivalent of Pinocchio exists — no crate with Featherstone dynamics + analytical derivatives +
  constrained/contact dynamics + an optimal-control stack. Ferromotion is the closest thing; the
  gaps below are what closes the claim.
- **Differentiability found its real job: calibration, not policy learning.** The industry's fast
  paths (MJWarp, Genesis rigid) dropped differentiability; gradient-based real-to-sim/sys-ID is the
  validated 2026 use (1–2 orders over black-box). MuJoCo 3.5 shipped a SysID toolbox; actuator/sensor
  fidelity (delays, electrical dynamics) is the sim-to-real agenda.
- **Our unclaimed seams, named by the sweeps**: portable/deterministic/memory-lean differentiable
  stack; maintained LNN/HNN/VIN model classes (ferromotion-learn IS this); a maintained
  differentiable-NLS + Lie-group layer (theseus frozen since 2024-09); permissive answers to the
  monetized choke points (Ruckig Pro waypoints). Sampling-MPC — where the field's momentum actually
  is — we already cover (MPPI/CEM/iCEM/DIAL/tube/Tiny/SRBD).
- **We do not out-FPS Newton on NVIDIA silicon.** GPU scale is not this library's axis; the
  browser/wasm + determinism + structure axis is.

## Ranked gaps (corroborated across sweeps + inventory)

### Tier 1 — build next (each unlocks the tier below it)
1. **Generic-scalar core dynamics** — make RNEA/ABA/kinematics generic over the scalar (nalgebra
   `RealField`), so `learn::dual` forward-mode AD flows *through the dynamics* natively (the Rust
   answer to Pinocchio's template/CasADi pipeline, capability #1/#3 of the classical list).
   Oracle: matches `dyn_derivatives` analytical results + finite differences.
2. **`calib` — turnkey gradient real-to-sim calibration** — fit {inertial params, joint friction
   (Coulomb/viscous/Stribeck), actuator (SEA), payload} to recorded trajectories by gradient
   through a differentiable rollout; identifiability report; pseudo-inertia consistency projection
   (already in `sysid`); optional neural residual (`learn::neural_ode`). The single most validated
   2026 use of differentiability, and pure assembly of in-house parts.

### Tier 2 — high-value, independent
3. **Waypoint OTG (Ruckig-Pro flank)** — jerk-limited trajectories through intermediate waypoints +
   directional limits, permissive-licensed; our `ruckig` crate is the thinnest in the workspace.
4. **Constraint-API modernization (the Pinocchio 4.0 bar)** — mimic joints, Delassus operator,
   point/anchor/limit/friction constraint models with ADMM/PGS inside the dynamics layer.
5. **MJCF ingestion** alongside URDF (labs + models interop; SDF later).
6. **Differentiable-NLS-as-a-layer** on `sparse` (theseus-class niche: implicit differentiation of
   the Gauss–Newton fixed point).
7. **Actuator realism pack** — command/sensor delays, DC-motor electrical dynamics + cogging
   (MuJoCo 3.5–3.7 direction; feeds `calib`).
8. **Native Rerun logging (optional feature)** — Rust-native observability; the cheapest
   ecosystem-grade move (capability #10).

### Tier 3 — frontier / hygiene
9. **Error-controlled contact integration (CENIC-class)** — convex time-stepping with accuracy
   guarantees; the new trustworthy-contact bar.
10. **Depth pass on thin crates** — cloth(3)/rod(3)/ruckig(4)/tactile(7)/fluid(8) tests vs scope;
    volumetric Neo-Hookean FEM remains from the old Tier-2.

## Positioning sentence
Ferromotion does not race Newton's FPS; it owns the portable, deterministic, wasm-clean seam:
structured mechanics (LNN/HNN/VIN) + certifiable control + gradient calibration, one pure-Rust
source, browser to workstation.

---

# HISTORICAL — v0.23-era plan (superseded, kept for the record)

# Ferromotion — Roadmap & Frontier Plan

_A full review, fruition/integration audit, fresh SOTA sweep (6 parallel scouts against the
current library), and a sequenced ingest-and-implement plan. Supersedes the earlier
`docs/FRONTIER.md`, which predates this cycle's ~19 new modules and is now partly stale._

Generated at v0.23.0 + 3 unreleased (Tube MPC, Dubins, GICP). Workspace: 11 crates, **122
modules, 453 tests, ~31k LOC, all green**.

---

## 1. Fruition & integration audit

**Health: clean.**

- **No orphaned public API.** The umbrella `ferromotion` crate re-exports all 9 sub-crates;
  every public module is re-exported. The only non-exported `mod`s are `control::qp` (a
  `pub(crate)` QP/box-QP backend, 8 consumers) and it is correctly internal.
- **Test coverage.** Only `qp` (internal backend) and `core::costs` (the `Cost` trait, 6
  consumers) lack *direct* tests — both are exercised through consumers. Every other module is
  FD/oracle/invariant-verified.
- **Clippy.** 53 workspace warnings, **all pre-existing style debt** (9 `too_many_arguments`
  accepted-style, a few loop-index in older modules, ~10 "add `Default`" on the wasm `*Lab`
  structs). **Zero in any of this cycle's 19 new modules.** → _Action: one cleanup pass
  (add `Default` for the labs, `#[allow]` the accepted `too_many_arguments`)._
- **Release state.** 3 modules unreleased on v0.23.0 → cut **v0.24.0**.
- **Deferred/refinement backlog** (all documented in-code, none silent): EPA penetration
  recovery, iSAM2 Bayes tree, IRIS log-det volume SDP, GCS full convex relaxation, Reeds-Shepp
  (48 words), MSCKF IMU-propagation predict step.
- **Surfacing.** 16 interactive labs/textbook chapters cover a curated slice of 122 modules
  (intentional). Candidates for new chapters flagged in §5.

---

## 2. Cross-cutting infrastructure (build FIRST — each unlocks ≥3 downstream items)

These are the shared dependencies the six sweeps repeatedly bottomed out on. Building them first
turns many Tier-1/2 items from "hard" into "assembly".

| Infra module | Unlocks | Oracle |
|---|---|---|
| **`core::spatial`** ✅ — kd-tree + voxel-hash spatial index | KISS-ICP/NDT, ESDF mapping, GPMP2, VAMP-planning, large-cloud ICP/GICP | NN query == brute-force; range-query completeness |
| **`core::lmi`** ✅ — Lyapunov/LMI certificates (pure `nalgebra`, wasm-clean) | CCM (LTI reduction), LQR-trees/funnels ROA | Hurwitz A ⇒ P≻0 with AᵀP+PA=−I; unstable ⇒ none. _(Note: clarabel's PSD cone needs BLAS → breaks wasm, so the general polynomial-SOS / inequality-LMI feasibility path is a follow-up needing a pure-Rust PSD interior point; the quadratic Lyapunov certificate covers the LTI reductions.)_ |
| **`control::diff_qp`** ✅ — implicit differentiation through the equality QP (OptNet) (IFT on the KKT fixed point) | Differentiable-MPC-as-layer, learned-cost/dynamics MPC | IFT gradient == finite-diff; LQR has closed-form `∂U*/∂θ` via DARE sensitivity |
| **`core::simd_collision`** — `portable_simd` / wasm `simd128` batched FK + collision | VAMP-style μs-planning force-multiplier across every sampling planner | **lane == scalar bit-exact** (also a determinism-doctrine win) |

_The reverse-mode AD tape (`mpm::adjoint`) already exists and unlocks the structured-dynamics
learning items (§4) with no new infra._

---

## 3. Highest-signal ingest queue (corroborated across ≥2 sweeps, or clean exact oracle)

Ranked. ★ = surfaced by **two independent** sweeps (strong signal).

1. **iCEM** ★-adjacent — colored-noise + elite-reuse sampling MPC (Pinneri, CoRL 2020). _Trivial_
   upgrade to `cem`/`mppi`; oracle: PSD slope of sampled actions ≈ −β; LQG mean → analytic optimum.
2. **DIAL-MPC** ★ (opt-control + loco-manip) — diffusion-annealed **full-order torque-level**
   sampling MPC (Xue 2024, ICRA'25). Training-free legged control where MPPI diverges. Oracle:
   anneal→0 ⇒ LQR optimum; monotone cost across the schedule.
3. **Geometric SE(3) tracker** (Lee 2010) — the coupled nonlinear aerial *feedback* controller
   our flatness-only `quadrotor` and attitude-only `es_mpc` lack. _Low_; Lyapunov-decrease + hover-
   equilibrium oracle.
4. **iSAM2 Bayes tree** (Kaess 2012) — closes the gap `isam.rs` explicitly scopes out (fluid
   relinearization + incremental reordering). Oracle: **bit-exact vs batch `sparse` re-solve**.
5. **Dojo maximal-coordinate NCP contact** (Howell 2022) — general body-to-body contact + a
   variational integrator; our `robot_contact` is minimal-coordinate LCP-vs-floor. Oracle: energy/
   momentum drift ≈ 0 (Poinsot); IFT gradient == complex-step FD. Needs `clarabel`/`faer`.
6. **Control Contraction Metrics** ★ (opt-control + learning; Manchester–Slotine 2017) — certified
   exponential tracking; the tracking-analog of our CBFs. Needs **`core::sos`**. Oracle: LMI eigen
   ≤ 0; LTI ⇒ LQR `P`; two trajectories contract at rate ≥ λ.
7. **Differentiable-MPC-as-layer** ★ (opt-control + learning; Amos 2018) — makes the whole control
   crate trainable modules. Needs **`control::diff_solve`**. Oracle: IFT == FD; LQR closed form.
8. **BIT\*/AIT\*/EIT\*** (Gammell 2020/22) — informed anytime-optimal search; our only sampling
   planner is RRT*. Oracle: cost → analytic geodesic; informed-set ellipsoid membership.
9. **Geometric Fabrics** (Ratliff/Van Wyk 2020) — the energized successor to `rmpflow` with
   provable stability + speed-invariance. Oracle: path-invariance under time reparam (which
   RMPflow *fails* — a clean differentiator test).
10. **Differentiable dexterous grasp synthesis** (Diff-Force-Closure 2021 / SpringGrasp 2024) — we
    can *score* grasps (`grasp`) but not *generate* them. Oracle: FC energy → 0 & `G` spans ℝ⁶;
    synthesized grasp passes our exact Q1.

---

## 4. Full ranked gap list by domain (the six sweeps, deduped)

### Optimal control / MPC / trajopt
- **Tier 1:** iCEM · DIAL-MPC · CCM (→sos) · Differentiable-MPC (→diff_solve).
- **Tier 2:** RTI-SQP condensing (acados-style) · ALTRO conic AL + projected-Newton · GuSTO (free-
  final-time SCP) · DeePC (data-enabled, model-free — distinct from Koopman) · LQR-trees / funnel
  libraries (→sos).
- **Tier 3:** risk-sensitive iLEQG · Stein-variational MPC (multimodal) · MPCC contouring · TD-MPC2-
  style learned terminal value.

### Motion planning / collision / geometry
- **Tier 1:** VAMP SIMD planning (→simd_collision) · BIT*/AIT*/EIT* · C-IRIS certified (→sos) ·
  GCS* implicit best-first (upgrades our BFS+SOCP GCS) · ESDF incremental mapping (→spatial).
- **Tier 2:** GPMP2 GP-inference planning (→spatial, →isam2) · constrained-manifold sampling
  (IMACS) · safe-flight-corridors · configuration-space distance fields (RDF) · swept-volume SDF.
- **Tier 3:** contact-implicit planning-through-contact (MPCC/complementarity) · homotopy/h-signature.

### Estimation / SLAM / differentiable sim
- **Tier 1:** iSAM2 Bayes tree · Dojo maximal-coord NCP contact.
- **Tier 2 (estimation):** continuous-time GP/STEAM (WNOJ) · Gaussian Belief Propagation · Schur-
  complement sliding-window marginalization · KISS-ICP/NDT LiDAR odometry (→spatial).
- **Tier 2 (diff-sim):** tetrahedral Neo-Hookean **volumetric** FEM · SAP convex frictional contact ·
  Vertex Block Descent integrator · Affine Body Dynamics (reuses `ipc`) · Position-Based Fluids (3-D
  Lagrangian; pairs with `xpbd`).
- **Tier 3:** observability-constrained / FEJ EKF consistency.

### Domains: aerial / space / marine / soft / multi-robot
- **Tier 1:** Geometric SE(3) tracker · ORCA reciprocal avoidance · Clohessy–Wiltshire rendezvous ·
  QUEST/TRIAD attitude determination.
- **Tier 2:** reaction-wheel/CMG momentum management + desaturation · tendon-driven Cosserat (actuated
  3-D shape) · algebraic-connectivity maintenance · auction/Hungarian task assignment · thrust-
  allocation & dynamic positioning (marine) · Voronoi coverage (Lloyd).
- **Tier 3:** perception-aware control (PAMPC) · tailsitter/VTOL transition.

### Locomotion + whole-body / manipulation
- **Tier 1:** Geometric Fabrics · differentiable dexterous grasp synthesis.
- **Tier 2:** whole-body loco-manipulation MPC (object dynamics in the OCP) · perceptive-locomotion
  terrain-SDF NMPC · DIAL-MPC (also above) · humanoid whole-body retargeting (foot-contact/skate) ·
  multi-contact fall mitigation & recovery.
- **Tier 3:** caging / energy-bounded caging · task-oriented (task-wrench-space) grasp quality ·
  grasp/dexterous foundation-model ONNX interface (verifiability degrades — interop, flag honestly).

### Learning-for-control (pure-Rust, inference/structure; no PyTorch training)
- **Tier 1:** Differentiable-MPC-as-layer (also above) · CCM (also above).
- **Tier 2:** structured Lagrangian/Hamiltonian NN dynamics (uses the AD tape; energy-conservation
  oracle) · L1/MRAC adaptive control (an empty category today) · residual-policy composition
  operator (identity oracle) · PILCO/PETS uncertainty-propagating model rollout (feeds cem/mppi).
- **Tier 3:** FAST DCT action tokenization (π0-FAST) · VQ-BeT residual-VQ tokenization · equivariant
  SE(3) policy head (exact-equivariance oracle) · TD-MPC2 latent-imagination wrapper.

### Deferred refinements (finish what exists)
EPA horizon-expansion · IRIS log-det SDP · GCS convex relaxation · Reeds-Shepp (48 words, verify each
via the Dubins integrate-to-goal oracle) · MSCKF IMU-propagation predict · FEJ consistency for MSCKF.

---

## 5. Sequenced release milestones

**v0.24.0 — cut now + quick wins.** Ship the 3 unreleased (Tube MPC, Dubins, GICP) plus the
low-effort exact-oracle openers: **iCEM · Geometric SE(3) tracker · ORCA · QUEST/TRIAD · auction/
Hungarian**. (Also: the clippy cleanup pass.) New textbook chapter candidate: **Dubins** (car
threading a course) or **ORCA** (agents crossing).

**v0.25.0 — infrastructure + its unlocks.** `core::spatial` · `core::sos` · `control::diff_solve`,
then the items they gate: **CCM · Differentiable-MPC-layer · iSAM2 Bayes tree · ESDF mapping ·
KISS-ICP/NDT**. This is the highest-leverage release (four force-multipliers + five capabilities).

**v0.26.0 — differentiable-sim wave.** **Dojo maximal-coord NCP · SAP convex contact · Vertex Block
Descent · tetrahedral Neo-Hookean FEM · Position-Based Fluids · Affine Body Dynamics.** Chapter
candidate: Dojo/contact ("simulate a stack that can't lie").

**v0.27.0 — planning frontier.** **BIT*/AIT*/EIT* · C-IRIS · GCS* · GPMP2 · safe-flight-corridors ·
Geometric Fabrics** (+ `simd_collision` → VAMP). Chapter candidate: Geometric Fabrics vs RMPflow.

**v0.28.0 — loco-manip + learning frontier.** **DIAL-MPC · whole-body loco-manip MPC · perceptive-
locomotion NMPC · humanoid retargeting · fall mitigation · differentiable grasp synthesis ·
Lagrangian/Hamiltonian NN · L1/MRAC · residual policy · PILCO/PETS.**

**Rolling:** RTI-SQP · ALTRO-conic · GuSTO · DeePC · continuous-time GP/STEAM · GBP · Schur
marginalization · Clohessy–Wiltshire · momentum management · connectivity maintenance · Voronoi
coverage · thrust-allocation/DP · FAST/VQ-BeT/equivariant/TD-MPC2 · Reeds-Shepp + deferred
refinements — folded into whichever milestone their prerequisites land in.

---

## 6. Method discipline (unchanged, applied to every item above)

Inventory-first (`ls`, not grep, before claiming a gap) → verify-first (an analytic solution,
physical invariant, or finite-difference check as the test **before** any page) → pure `nalgebra`/
`faer`/`clarabel`, wasm-clean → honest scope notes in-code for anything simplified or deferred →
crates.io release only on explicit go-ahead (irreversible).
