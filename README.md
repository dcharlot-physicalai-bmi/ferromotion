# Ferromotion

[![CI](https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/workflows/ci.yml/badge.svg)](https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ferromotion.svg)](https://crates.io/crates/ferromotion)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-core)](https://docs.rs/ferromotion-core)
[![license](https://img.shields.io/crates/l/ferromotion-core.svg)](#license)

**A Rust library for the kinematics, dynamics, and control of physical AI — native and in the browser.**

Ferromotion is a pure-Rust ecosystem for embodied control: forward/inverse kinematics, rigid-body
dynamics, actuator modelling and identification, trajectory optimization, motion retargeting, and a
broad library of controllers — all
compiling to native *and* `wasm32`, so the same solver runs on a workstation, an edge robot, and a
browser tab with zero install. It grew out of the [PyRoki](https://github.com/chungmin99/pyroki)-class
ecosystem (JAX/Python, server-side) and expanded well past it into a full control stack.

Sibling to [Ferric](https://physicalai-bmi.org) (the Institute's pure-Rust compute fabric). From the
[Institute for Physical AI](https://physicalai-bmi.org).

## Crates
| Crate | Description |
|---|---|
| [`ferromotion`](https://crates.io/crates/ferromotion) | Umbrella — re-exports everything below. |
| [`ferromotion-core`](https://crates.io/crates/ferromotion-core) | FK, analytic + multi-frame Jacobians, **RNEA** inverse dynamics + **ABA** O(n) forward dynamics + **floating-base** dynamics + **analytical dynamics derivatives** (exact ∂/∂q,∂/∂q̇,∂/∂τ), mass matrix, IK (LM + robust), trajectory optimization, sparse factor-graph solve, collision costs, motion retargeting, augmented-Lagrangian, **differentiable contact** (interior-point LCP + articulated frictional contact + contacts-from-distance gradients), **IPC** (intersection-free log-barrier contact), **grasp force-closure** (differentiable Ferrari-Canny Q1), **signed distance fields** (analytic SDF collision scene + robot-sphere clearance), **RRT\*** (asymptotically-optimal sampling-based planning), **Lie-group variational integrator** (structure-preserving SO(3) rigid-body), **modal reduced-order deformables** (generalized-eigenproblem subspace), **inertial parameter identification** (RNEA regressor + pseudo-inertia consistency), **actuator modelling and identification** (reflected rotor inertia / armature, viscous damping, Coulomb friction — read from URDF, MJCF and SDF, applied inside RNEA; fitted from motion or from current *together with the torque constant*, as an exact per-joint linear regression; plus `actuator_plausibility` to spot an impossible declared limit from the model alone and `confounding` to screen an excitation before running it), **closed-loop / parallel mechanisms** (loop-closure KKT dynamics), **exact hybrid trajectory gradients** (`hybrid_jacobian`: event detection + saltation composition, refusing on grazing / hidden state / probe dependence), URDF · MJCF · SDFormat · **OpenUSD (`UsdPhysics`)** loading and writing. |
| [`ferromotion-control`](https://crates.io/crates/ferromotion-control) | PID · computed-torque · Cartesian impedance · **series-elastic actuator / transmission model** · **Hill muscle model** (biological actuation, morphological computation) · LQR · linear MPC · OSC · WBC · **HQP** (strict-priority hierarchical QP) · placo · iLQR/DDP · **constrained DDP (PROXDDP)** · **direct collocation (DIRCON)** · MPPI · CEM · CBF-QP · **HJ reachability** (backward reachable tubes) · sliding-mode · **SRBD MPC** · capture-point/ZMP · **SLIP + Raibert hopper** (running) · **DCM walking + footstep planning** · **TOPP** (time-optimal path parameterization) · **quadrotor differential flatness + minimum-snap** · **Fossen 6-DOF marine-craft dynamics + LOS guidance** (surface/underwater vehicles) · centroidal MPC · Kalman/EKF/UKF · **invariant EKF (InEKF)** on SE₂(3) · **Koopman/EDMD** (data-driven linearization) · **IMU preintegration** (Forster, on-manifold, + leg-odometry bias correction) · **RMPflow** (Riemannian motion policies — reactive multi-task motion) · **IBVS** (image-based visual servoing) · **DMPs** (learning-from-demonstration primitives) · **multi-robot consensus + formation control** (graph Laplacian, algebraic connectivity) · complementary filter · momentum observer. |
| [`ferromotion-fluid`](https://crates.io/crates/ferromotion-fluid) | 2D incompressible **Navier–Stokes** (MAC projection) for fluid–robot interaction — verified against the Ghia Re=100 lid-driven-cavity benchmark. |
| [`ferromotion-mpm`](https://crates.io/crates/ferromotion-mpm) | Differentiable 2D **Material Point Method** (MLS-MPM, neo-Hookean) for soft/elastic/granular material — analytic material-stiffness gradients. |
| [`ferromotion-tactile`](https://crates.io/crates/ferromotion-tactile) | Differentiable **optical-tactile** (GelSight/DIGIT) sensor simulation — gel deformation → photometric image — plus **tactile servoing** (contact features → sensor motion), closing the sense→act loop. |
| [`ferromotion-models`](https://crates.io/crates/ferromotion-models) | Robot arms built from their **published Denavit–Hartenberg tables** (Universal Robots UR3–UR20, Franka Panda/FR3, KUKA LBR iiwa, PUMA 560, Stanford arm, planar teaching arms, and more), each citing its primary source, stating its DH convention, and verified at a known pose plus the workspace-wide Jacobian/Hessian check. No URDF or mesh files vendored. |
| [`ferromotion-rod`](https://crates.io/crates/ferromotion-rod) | Differentiable **Discrete Elastic Rods** (stretch + bending) for cables, tendons, and continuum robots — validated vs Euler-Bernoulli. |
| [`ferromotion-cloth`](https://crates.io/crates/ferromotion-cloth) | Differentiable **FEM thin-shell cloth** (StVK membrane + bending) — exact forces + analytic material-stiffness gradients — plus **Vertex Block Descent** (VBD/AVBD): stable at one sweep per step where explicit Euler goes NaN (measured at `dt = 1/30 s`, `k = 1e6`), converging to implicit Euler with more sweeps. One sweep buys boundedness, not accuracy: it overstretches a 1.2 m chain by +60%, falling to +0.5% at 64 sweeps. |
| [`ferromotion-ruckig`](https://crates.io/crates/ferromotion-ruckig) | Jerk-limited online trajectory generation. |
| [`ferromotion-policy`](https://crates.io/crates/ferromotion-policy) | On-device runner for exported learned (RL/VLA) policies — MLP inference + **flow-matching action sampler** (ODE integration of a learned velocity field). |
| [`ferromotion-bench`](https://crates.io/crates/ferromotion-bench) | Dependency-free measurement harness: median + median-absolute-deviation + 10th/90th percentiles and an explicit "do not quote" verdict when a run is too noisy to cite. No `criterion` dependency, because it does not build for `wasm32` — a performance claim should be checkable where the code runs. |
| [`ferromotion-wasm`](https://crates.io/crates/ferromotion-wasm) | WebAssembly bindings — build a chain or load a URDF, then FK / IK / retargeting / motion planning in the browser. |

## Quickstart
```toml
[dependencies]
ferromotion-core = "0.1"
nalgebra = "0.35"
```
```rust
use ferromotion_core::{from_urdf_str, solve_ik, IkOptions};
use nalgebra::{Isometry3, Translation3};

// Load a robot from URDF text (works natively and in the browser).
let robot = from_urdf_str(urdf, "base_link", "tool").unwrap();

// Solve inverse kinematics to a target pose.
let target = Isometry3::from_parts(Translation3::new(0.4, 0.1, 0.3), Default::default());
let seed = vec![0.0; robot.dof()];
let res = solve_ik(&robot, &target, &seed, &IkOptions::default());
println!("q = {:?}  converged = {}  residual = {:.2e}", res.q, res.converged, res.error);
```

## Highlights
- **Kinematics & dynamics** on SE(3): analytic geometric Jacobians (verified vs finite differences),
  Recursive Newton-Euler inverse/forward dynamics, joint-space mass matrix, gravity compensation.
- **Optimization**: composable-cost IK, block-tridiagonal trajectory optimization, sparse factor-graph
  solve (`faer`), collision-aware planning, augmented-Lagrangian hard constraints, motion retargeting
  (position / vector / DexPilot).
- **Control corpus**: from PID to MPC to whole-body QP to nonlinear optimal control (iLQR/DDP), sampling
  MPC (MPPI/CEM), safety filters (CBF-QP), legged balance (capture point, ZMP preview, centroidal &
  single-rigid-body MPC), and estimation (Kalman/EKF/UKF, momentum observer, complementary filter).
- **GPU fleet path**: batched MPPI rollouts as a WebGPU compute kernel — **~26× over CPU** at 16k
  rollouts, matching the CPU reference to 1e-6 (`gpu/mppi.html`).
- **Universal**: every library crate compiles to `wasm32`; `ferromotion-wasm` ships a browser API.

### Correct gradients through contact

Every GPU-differentiable simulator softens contact so it can be differentiated, and the published
position on the consequence is a trade-off: stiff settings give a faithful robot and unusable
gradients, soft settings give usable gradients and a sim-to-real gap. Measured on a contact whose
Jacobian is known in closed form, it is not a trade-off — the penalty derivative **diverges**.

| what was measured | result |
|---|---|
| fixed-step autodiff gradient vs stiffness | grows as `sqrt(k)`; exponent **0.5127** against an exact `1/2` |
| sign of the dominant entry | **wrong at 5 of 7** realistic stiffnesses |
| does *uniform* timestep refinement fix it | **no** — at a constant resolved step count the error stays within `0.93x`–`1.24x` |
| does a *tolerance-driven* stepper fix it | **yes** — `199` steps reach the exact answer at `3.9e-3`; `8000` fixed steps get the sign wrong |
| whose error is it, integrator or contact model | **integrator**, by four orders: `1.6e2` against `9.4e-3` at `k = 1e6` |
| the exact route vs finite differences | agrees to **8e-10** |
| descent on a shooting problem through one impact | exact **4/4** starts converge; penalty **0/4**, and from 3 of 4 it cannot take a single downhill step |
| same question, quadruped with 4 frictional feet | rigid law wins **both** axes at once — the trade-off is a false dilemma |

The missing piece is the **saltation matrix**: perturbing the state also moves the *time* of impact,
a term that scales as `1/|impact speed|` (3.92 at 4 m/s, 1569.6 at 0.01 m/s) and that no smooth
contact model contains. It never shows up in the trajectory, only in its derivative.

`hybrid_jacobian` makes that derivative a callable, and refuses rather than guesses:

```rust
use ferromotion_core::{hybrid_jacobian, HybridGradientOptions};

// Detects events, brackets each to one timestep, composes flow Jacobians with saltation matrices.
// Errors on a grazing event, on a flow that carries hidden state (a warm-started solver breaks the
// chain rule across a split), and on two events sharing a timestep.
let lin = hybrid_jacobian(&system, &x0, 0.0, horizon, HybridGradientOptions::default())?;
println!("rho = {:.4}, worst gain = {:.4}", lin.spectral_radius(), lin.worst_gain());
```

There are therefore **two** routes to a correct contact gradient, and a correction worth stating plainly. We first
attributed the divergence to the penalty model and wrote that its derivative "has no limit to converge to". That was
wrong. Splitting the error into the integrator's share and the contact model's share shows the model's share
*shrinking* as roughly `1/k` toward the exact saltation Jacobian, while the integrator's share grows. So the same
penalty physics under an error tolerance instead of a fixed step lands on the exact answer:

```rust
use ferromotion_core::{decompose_gradient_error, AdaptiveOptions, GRAVITY};

let split = decompose_gradient_error(
    GRAVITY, 1e6, 200.0, 1e-3, [0.5, 0.0], 0.4, 1e-6,
    AdaptiveOptions::with_tolerance(1e-11),
).unwrap();
// split.discretisation dominates split.model by ~1e4. The contact model was never the problem.
```

Two measurement traps are pinned by tests in `adaptive_contact.rs`, because both produced wrong conclusions here
first: uniform-step finite differences on this system report the same entry as `5.80`, `-78.3`, `694.2`, `-143.2` as
the probe shrinks by decades, and a central difference's `O(h^2)` term exceeds the model error it is being used to
measure. The reference is checked non-circularly against the closed-form saltation Jacobian, which shares none of its
machinery.

Reproduce it: `cargo run --release -p ferromotion-core --example contact_gradient_audit`
(and `contact_gradient_descent`, `quadruped_contact_gradient`, `adaptive_gradient_decomposition`).

### Certifying a policy built on a smoothed contact

Getting the gradient right is half the question. The other half is whether a policy optimised through a smoothed
contact survives the nonsmooth dynamics it will actually meet. `smoothing_tube` treats the model gap as a set-valued
disturbance, propagates it as a reachable tube, and checks constraints against the tube rather than the nominal path.

| measured on one mass, one impact, one ceiling | result |
|---|---|
| smoothing gap at `k = 1e4` vs `k = 1e6` | `3.2e-1` → `9.5e-3` |
| the ceiling constraint at those two stiffnesses | **refuted** at step 208 → would **certify** at margin `2.6e-2` |
| certifying with the fixed-step Jacobian instead | tube **2.38x** wider |

So stiffness decides whether the certificate exists at all, and that is the same regime where a fixed-step gradient
is unusable: at `k = 1e6` it reports `dv/dh = -209.76` against a true `+3.65`. The only contact stiff enough to
certify is one you cannot differentiate with a fixed step.

The blocking piece is the gap bound, not the tube algebra, and the API says so rather than papering over it. A gap
measured by sampling is a **lower** bound on its own supremum, so it can refute a constraint and can never certify
one:

```rust
// Sampled evidence, an enormous margin, and still not a certificate.
let sampled = GapBound::from_samples(&residuals)?;
// -> Undecided { reason: GapOnlySampled }
```

`nominal_activity` reports how close the nominal trajectory came to the constraint, because a certificate over a
horizon where the constraint cannot be reached is perfectly true and worth nothing. The first version of this
example certified a ceiling over a 60 ms horizon when the mass needed 285 ms to reach its apex.

Reproduce it: `cargo run --release -p ferromotion-control --example smoothing_tube_certificate`.

### Measured, not asserted

`ferromotion-bench` reports what a claim needs to survive a hardware change — the **scaling
exponent**, not a timing on one machine:

| algorithm | claimed | measured (asymptotic) |
|---|---|---|
| ABA forward dynamics, in DOF | 1.0 | **0.994** |
| CRBA mass matrix, in DOF | 2.0 | **1.833** |
| contact PGS, in contact count | 1.0 | **0.808** |

ABA runs 192 DOF in 26.6 µs (37.5 kHz) and 48 DOF in 6.6 µs (151 kHz) on one core.
A full-range fit reads CRBA 30% low, because fixed per-call cost dominates the small end — the
harness reports both columns so the difference is visible.

Run it: `cargo run --release -p ferromotion-bench --example dynamics_suite`

## In the browser
`ferromotion-wasm` builds with `wasm-pack` to a self-contained module. See `demo/` for a page that
drives a robot arm with live IK and an obstacle-avoiding planned trajectory — computed entirely
on-device, no server. Validated end-to-end on a real open-source robot (NormaCore's 3D-printed
7-DoF ElRobot) straight from its URDF (`crates/ferromotion-core/examples/elrobot.rs`).

## License
Dual-licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
