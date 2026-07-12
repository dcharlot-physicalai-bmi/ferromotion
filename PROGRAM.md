# The Rust Physical-AI Port Program

Goal: **ingest every PyRoki-class ecosystem into a unified Rust stack — native + universal WASM.**
Compiled 2026-07-11 from a four-track research sweep (physics/sim · kinematics/motion-opt ·
diff-opt & Rust substrate · learning stacks & existing Rust). Every project below was checked
against its actual repo/license.

## Scope (2026-07-12): also porting the control corpus
Beyond kinematics/dynamics/optimization, ferromotion now ingests **every control scheme/method/technique of
physical AI** into `ferromotion-control` (pure Rust, native + WASM). Same 3-tier rule: control is almost all
algorithmic → PORT-shaped; learned policies run as exported weights (interop). Full map + priorities in
**`CONTROL.md`**. First batch shipped + verified in closed loop (forward-dynamics sim): **PID**,
**computed-torque (inverse-dynamics) control**, **Cartesian impedance**; plus **`forward_dynamics`** in
ferromotion-core (simulator). Also validated the whole toolkit on a **real external robot** — loaded NormaCore's
**ElRobot** (7-DoF, Feetech ST3215) straight from its URDF: FK, IK (residual 3e-11), RNEA gravity + mass
matrix, all from the unmodified file (`crates/ferromotion-core/examples/elrobot.rs`).
Next control: LQR → linear MPC (clarabel) → OSC/WBC → iLQR/DDP → CBF-QP → MPPI (first heavy GPU kernel).

## Latest status (2026-07-12)
- **Dynamics — Pinocchio port** (`dynamics.rs`): Recursive Newton-Euler `inverse_dynamics(q,q̇,q̈,g)`, with
  `gravity_vector` and `mass_matrix` as special cases; inertia-aware URDF loading (`from_urdf_full`, folds fixed
  links via composite inertia). Verified: pendulum gravity torque = m·g·d exactly, `M` symmetric + positive-definite,
  `τ = M·q̈` consistency. **23 core tests green; WASM-clean.**
- **GPU fleet path** (`gpu/fk.html`): batched forward kinematics as a **WebGPU compute shader** (the universal GPU
  target — native via wgpu, and in-browser). Verified on this Mac's Metal through real Chrome (`examples/shoot_gpu.js`):
  **200k configs, max error 9.3e-7** vs the CPU reference. Honest perf note: for a kernel this light the batch
  upload/dispatch/readback overhead dominates (GPU 160 ms vs CPU 8.5 ms) — the GPU wins only with heavier
  per-element work (full 3-D FK+Jacobian, IK sweeps) or huge fleets, or when results feed a GPU render pipeline.
  Correctness of the fleet-compute path is proven.
- **GPU fleet path — the heavy kernel wins** (`gpu/mppi.html`): batched **MPPI rollouts** (K parallel rollouts of a
  2-link manipulator through the horizon) as a WebGPU compute shader. Verified on Metal via Chrome
  (`examples/shoot_gpu_mppi.js`): **16,384 rollouts × 64 steps, GPU 2.4 ms vs CPU 62.7 ms → 26× faster**, matching the
  CPU reference to 7e-7. (Timed in steady state — a receding-horizon loop dispatches every step; the first launch's
  ~70 ms shader compile is one-time.) Confirms the earlier prediction: light kernels (FK) lose to the CPU JIT, heavy
  batched-rollout kernels (MPPI) win big — the real GPU-MPPI fleet path.

## The one strategic finding that shapes everything
"Port everything" needs one honest refinement: **not everything is port-shaped.** Three tiers:

1. **PORT (pure-Rust, reaches the browser)** — weights-free classical optimization. This is the real
   prize and has **no Rust equivalent today**: differentiable/batched kinematics, trajectory + collision-aware
   optimization, motion retargeting. JAX/CUDA fundamentally *cannot* run in a browser; a pure-Rust core can.
   This is our moat.
2. **REIMPLEMENT A SUBSET** — large C++ toolboxes where we only need a slice (Pinocchio's FK/Jacobian/RNEA
   + derivatives; KDL's solvers — also LGPL, so clean-room only; Drake's IK-as-NLP *pattern*).
3. **INTEROP ONLY (do NOT port)** — the differentiable-GPU physics frontier (Newton, MJX, MuJoCo-Warp, Genesis,
   Warp) and learned VLAs (GR00T, openpi). Their value is CUDA data-parallelism or learned weights; a CPU/WASM
   rewrite throws away the whole point. Consume via FFI/ONNX; for the browser, reimplement *specific algorithms*
   on WebGPU rather than the framework.

## Port catalog (kinematics / motion optimization)
| Project | Org | License | Difficulty | Strategy | WASM |
|---|---|---|---|---|---|
| **PyRoki** | Berkeley AUTOLab | MIT | 4/5 | **Full port — flagship differentiable-IK toolkit (`ferromotion`)** | ✅ best strategic |
| **Ruckig** (Community) | pantor/Berscheid | MIT | 3/5 | Full port — self-contained, no deps | ✅ best clean target |
| **frax** | Stanford ASL (Morton, Pavone) | MIT (code) | 4/5 | Full port — differentiable RBD, cheaper than Pinocchio | ✅ |
| **Pink** | S. Caron | Apache-2.0 | 3/5 | Full port atop RBD-subset + `clarabel` QP | ✅ |
| **mink** | K. Zakka | Apache-2.0 | 3/5 | Port IK layer; WASM-MuJoCo for physics | ✅ |
| **dex-retargeting** | dexsuite/Qin | MIT | 3/5 | Full port; swap Pinocchio→our FK, nlopt→Rust NLLS | ✅ |
| **Orocos KDL** | KU Leuven | **LGPL-2.1** | 2–3/5 | Reimplement (copyleft — don't vendor); extend `k` | ✅ |
| **TRAC-IK** | TRACLabs | BSD-3 | 2/5 | Adopt `optik` (already a Rust reimpl) | ⚠ NLopt backend |
| **placo** | Rhoban | MIT | 4/5 | Reimplement task-space QP core; scope tightly | ✅ core |
| **Pinocchio** | INRIA/Carpentier | BSD-2 | 5/5 | Reimplement needed subset only | ✅ subset |
| **robosuite** | ARISE (Stanford/UT) | MIT | 4/5 | Framework layer only, atop a MuJoCo binding | partial |
| **cuRobo / V2** | NVIDIA | Apache-2.0 (v0.8+; v1 was NC) | 5/5 | Don't port — WebGPU reimpl of algorithms, or reference | ❌ CUDA |
| **Drake** | TRI/MIT | BSD-3 | 5/5 | Don't port — replicate the IK-as-NLP *pattern*; oracle | ❌ |

## Interop-only (physics & learned models)
Newton (Apache-2.0, LF-governed, Warp+MJWarp) · MuJoCo/MJX/MuJoCo-Warp (Apache-2.0) · Genesis (Apache-2.0) ·
Brax (Apache-2.0 — *analytic pipelines* are a 3/5 port POC) · Warp (Apache-2.0) · Isaac Sim/Lab · SAPIEN ·
PyBullet (redundant — `rapier` covers it) · GR00T N1.7 (weights = NVIDIA Open Model Lic.) · openpi (Apache-2.0
weights — friendliest for a Rust inference wrapper).

## The Rust substrate (what we build ON, not rewrite)
- **Kinematics baseline:** `k` (FK + Jacobian-IK, already wasm-tested) + `urdf-rs` (parse URDF *from memory* in WASM).
- **Solver core (the jaxls-equivalent):** **`factrs`** (CMU — GN + LM, Lie groups SO/SE(2,3) optimized in the Lie
  algebra, forward-mode autodiff via `num-dual`) on **`faer`** (sparse Cholesky/LU/QR; disable `rayon` for WASM).
  Nothing in Rust matches jaxls's *batched/vectorized factor* + *differentiate-through-solve* + GPU — that layer
  is our net-new work.
- **Linear algebra:** `nalgebra` (pure Rust, `no_std`, wasm out of the box).
- **QP backend:** `clarabel` (Dimforge/Stanford, pure-Rust interior-point — powers Pink/mink/placo ports in WASM).
- **Autodiff:** forward-mode is production-ready (`num-dual`) and enough for LM/IK. **Reverse-mode is the genuine
  greenfield gap** (no JAX-grade `grad`/`vmap`/`jit`; Enzyme is nightly + non-WASM). Mitigation: analytical
  derivatives (Pinocchio-style) sidestep it for kinematics.
- **Collision/physics:** `rapier` + `parry` (Dimforge, first-class WASM, optional cross-platform determinism).
- **GPU + differentiable (the "Warp" of Rust):** **`CubeCL` + `Burn`** (kernels across CUDA/ROCm/Metal/**WebGPU**,
  reverse-mode autodiff) — the only stack that reaches native *and* browser GPU. `wgpu` underneath; `candle` for
  inference-only. First differentiable-GPU milestone worth attempting: a native XPBD (`avian`-style) or Brax
  analytic pipeline.
- **Viz:** `rerun` (native + in-browser wasm viewer) — log robot state identically in both targets.
- **Native-only (out of scope for browser):** `dora-rs`, `ros2-rust`/`r2r`, `openrr` app tiers.

## Suggested build order
1. **Foundation:** SE(3)/SO(3) Lie groups + analytical Jacobians/RNEA + derivatives on top of `k`/`urdf-rs`
   (shared substrate for frax, Pink, placo, PyRoki). — *`ferromotion-core` M0 landed this for revolute/prismatic chains.*
2. **Quick win:** port **Ruckig** Community (self-contained, no deps) — a clean, high-value WASM target.
3. **The moat:** build the **PyRoki-style composable variable/cost differentiable optimizer** on `factrs`/`faer` —
   the piece with no browser-native competitor.
4. **Adjacent ports:** Pink/mink/dex-retargeting on the shared core + `clarabel`.
5. **Physics POC (optional):** CubeCL+Burn native XPBD to test the differentiable-GPU path to WASM.

## Status — `ferromotion` (the PyRoki port), M0–M3 shipped this pass
### Ruckig — 2nd ecosystem crate shipped (`ferromotion-ruckig`)
Jerk-limited, time-optimal trajectory generation — a port of Ruckig's **MIT Community** S-curve
(rest-to-rest, per-DoF) + **multi-DoF time synchronization** (Ruckig's signature). Profile built as
constant-jerk segments; peak velocity found by monotone bisection, so it hits the target exactly and
respects v/a/jerk limits by construction. Pure `f64`/std → trivially WASM-clean, zero deps. 4 tests
green (cruise, short-move-no-cruise, reversed direction, synchronized multi-DoF); compiles to `wasm32`.
Not ported (Ruckig **Pro**, closed): arbitrary non-rest states, intermediate waypoints, tracking.

### M3 — trajectory, collision, hard constraints
- **Trajectory optimization** (`TrajectoryProblem`): jointly optimize `q₀…q_{T-1}` with per-timestep costs +
  velocity smoothness. The coupling is **block-tridiagonal**, solved with a **block-Thomas** LM step (linear in
  T — the sparse structure jaxls exploits) in pure `nalgebra` (no external sparse solver; WASM-clean). General
  (branched/loop) sparsity via `faer`/`factrs` remains the documented extension.
- **Collision costs** (`SphereCollisionCost`, `PlaneCollisionCost`): the **sphere model (as cuRobo uses)** —
  smooth signed distance + analytic gradient, so trajectory optimization converges cleanly. Jacobian verified vs
  finite-difference; a trajectory that started 3 cm inside an obstacle is routed clear. `parry` general-shape
  distance is the next extension.
- **Augmented-Lagrangian hard constraints** (`solve_al`, `PlaneConstraint`): outer AL loop lifts penalties +
  updates multipliers until every `c(q) ≤ 0` holds to tolerance — verified enforcing a keep-above-plane
  constraint the objective actively fights (constraint binds; violation < 1e-3).
- **12 tests green**; ferromotion-core + ferromotion-wasm still compile to `wasm32`; Node runtime still green.

## (earlier) Status — M0 + M1 + M2
- `ferromotion-core`: SE(3) FK, **multi-frame FK + reference-point Jacobian** (target any point on the body), analytic
  geometric Jacobian (verified vs finite-difference), a composable **`Cost` trait** (PoseCost, PointCost,
  VectorCost, JointLimitCost, PostureCost — the PyRoki-shaped decoupled design), a generic Levenberg–Marquardt
  `solve` over stacked costs, and a `solve_ik` wrapper.
- **URDF loader** (`from_urdf_str`): parses from a *string* (browser-safe), walks the tree base→tip, folds
  fixed joints, handles revolute/continuous/prismatic + limits.
- **Motion retargeting** (`Retargeter`): warm-started per-frame solves with temporal smoothness + joint limits;
  position (`PointCost`) and translation-invariant vector (`VectorCost`) matching — the "show it, don't program
  it" primitive, with no Rust-native equivalent.
- **8 tests green** incl. loading a real 6-DoF arm, IK on it, and retargeting a 40-step trajectory (<2 mm keypoint reproduction).
- `ferromotion-wasm`: `wasm-bindgen` `Chain` API — build joints *or* `Chain.from_urdf(...)`, then `fk` / `solve_ik` /
  `retarget_step` (a live teleop loop) from JS.
- **Compiles to `wasm32`; `wasm-pack --target web` → loadable 53 KB ES module.** Verified in a **Node runtime
  end-to-end** (`examples/smoke.js`): planar-3R IK 3.8e-13; 6-DoF arm from URDF, IK 2.2e-11; retarget stream
  reproduces the tool point to 0.12 mm. The universal variant runs.

### Browser demo — the universal variant, made visible
`ferromotion-wasm` now exposes the full toolkit: `fk`, `solve_ik`, `retarget_step`, `from_urdf`, and **`plan_reach`**
(plan a smooth trajectory to a goal, optionally routing around a sphere obstacle). `examples/build_demo.js`
assembles a **fully self-contained page** (`demo/index.html`, 583 KB — wasm-bindgen glue inlined + wasm
base64-embedded, no fetch/server): a 4-joint arm you drive with **live IK (Follow)** and an **obstacle-avoiding
trajectory (Reach → Plan & play)**. Verified in real headless Chrome (`examples/shoot_demo.js`): no page errors,
arm renders and both modes run on-device. Not yet wired into production (`physicalai-bmi.org`) — that deploy is
the next outward-facing step.

### Ecosystem ports + depth (this pass)
- **Pink / mink — differential IK as a QP** (`solve_diffik`): their shared method (task Jacobians → QP each step,
  box velocity + joint-position limits, integrate) on the ferromotion kinematics, with **`clarabel`** (pure-Rust
  interior-point) as the QP backend. Converges + respects limits; **`clarabel` compiles to `wasm32`** (pulls its
  `web-time` shim) so this is universal.
- **dex-retargeting — vector optimizer** (`VectorRetargeter`, `VectorTask`): translation-invariant keypoint-vector
  matching for cross-morphology retargeting, composed on the shared core via `VectorCost`. Self-retargeting test
  reproduces target vectors exactly.
- **Capsule collision** (`CapsuleCollisionCost`): segment-to-segment closest points + barycentric Jacobian —
  robot links as *swept capsules* (a sphere is a degenerate capsule), covering link-vs-obstacle and link-vs-link
  self-collision with smooth analytic gradients. FD-verified. (Chose analytic capsules over `parry` for smooth
  gradients + no nalgebra-version skew; `parry` convex/mesh remains a pluggable backend.)
- **General factor-graph solver** (`solve_factor_graph`, `SparseFactor`): arbitrary topologies beyond
  chain/trajectory — coupled multi-robot, **loop closures**, kinematic trees. Verified on a two-arm loop closure
  (arm1→target + tips-coincide) — converges in ~14 iters to ~1e-8.

### Depth #2 — `faer` sparse backend (wired)
`solve_normal_equations` now assembles the normal equations as **sparse triplets** (pre-summed, lower triangle)
and factors them with **`faer`'s sparse Cholesky** (`try_new_from_triplets` → `sp_cholesky(Side::Lower)` →
`Solve::solve_in_place`); non-PD → raise λ. Cost scales with fill, not `n³`. **`faer` compiles to `wasm32`**
(default-features off, rayon disabled) — the sparse backend stays universal.

### More ecosystem ports
- **TRAC-IK → `solve_ik_robust`**: random-restart IK (deterministic LCG seeds within joint limits, first
  in-limits success wins) — TRAC-IK's escape-local-minima method, in pure Rust. (`optik` ports TRAC-IK but on an
  NLopt C backend that can't reach WASM, so we reimplement the method to stay universal.)
- **Orocos KDL → `resolved_rate`** (clean-room; KDL is LGPL): damped-least-squares resolved-rate velocity IK
  `q̇ = Jᵀ(JJᵀ+λ²I)⁻¹·twist` + nullspace projection `(I−J⁺J)·q̇₀` for redundancy resolution. Verified the
  realized twist matches and the secondary task lives in the nullspace.
- **placo → posture task in the QP**: `DiffIkOptions.posture` adds placo's weighted posture/regularization task to
  the `clarabel` task-space QP (position tasks + posture + hard joint/velocity limits). Verified the primary reach
  holds while redundant DoF resolve toward the rest posture.

**Status: 25 tests green (21 core + 4 ruckig); whole stack (incl. `clarabel` + `faer`) compiles to `wasm32`.**

**Next:** deploy the browser demo to the site; more ports (Ruckig arbitrary-state online, Pinocchio RNEA/dynamics
once URDF carries inertias); GPU batched solves via `CubeCL`/`wgpu`.

## Carry-forward caveats
Rust-autodiff gap is the recurring risk (use analytical derivatives). KDL is LGPL (reimplement, don't vendor).
cuRobo v1 code is non-commercial (only v0.8.0+ Apache-2.0). Ruckig Pro features are closed (port Community only).
`frax`/`cuRoboV2` arXiv IDs are very recent — re-confirm before formal citation. ManiSkill assets are CC-BY-NC.
