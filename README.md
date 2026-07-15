# Ferromotion

[![CI](https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/workflows/ci.yml/badge.svg)](https://github.com/dcharlot-physicalai-bmi/ferromotion/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ferromotion.svg)](https://crates.io/crates/ferromotion)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-core)](https://docs.rs/ferromotion-core)
[![license](https://img.shields.io/crates/l/ferromotion-core.svg)](#license)

**A Rust library for the kinematics, dynamics, and control of physical AI — native and in the browser.**

Ferromotion is a pure-Rust ecosystem for embodied control: forward/inverse kinematics, rigid-body
dynamics, trajectory optimization, motion retargeting, and a broad library of controllers — all
compiling to native *and* `wasm32`, so the same solver runs on a workstation, an edge robot, and a
browser tab with zero install. It grew out of the [PyRoki](https://github.com/chungmin99/pyroki)-class
ecosystem (JAX/Python, server-side) and expanded well past it into a full control stack.

Sibling to [Ferric](https://physicalai-bmi.org) (the Institute's pure-Rust compute fabric). From the
[Institute for Physical AI](https://physicalai-bmi.org).

## Crates
| Crate | Description |
|---|---|
| [`ferromotion`](https://crates.io/crates/ferromotion) | Umbrella — re-exports everything below. |
| [`ferromotion-core`](https://crates.io/crates/ferromotion-core) | FK, analytic + multi-frame Jacobians, **RNEA** inverse dynamics + **ABA** O(n) forward dynamics + **floating-base** dynamics + **analytical dynamics derivatives** (exact ∂/∂q,∂/∂q̇,∂/∂τ), mass matrix, IK (LM + robust), trajectory optimization, sparse factor-graph solve, collision costs, motion retargeting, augmented-Lagrangian, **differentiable contact** (interior-point LCP + articulated frictional contact + contacts-from-distance gradients), **IPC** (intersection-free log-barrier contact), **grasp force-closure** (differentiable Ferrari-Canny Q1), **signed distance fields** (analytic SDF collision scene + robot-sphere clearance), **RRT\*** (asymptotically-optimal sampling-based planning), **Lie-group variational integrator** (structure-preserving SO(3) rigid-body), **modal reduced-order deformables** (generalized-eigenproblem subspace), **inertial parameter identification** (RNEA regressor + pseudo-inertia consistency), **closed-loop / parallel mechanisms** (loop-closure KKT dynamics), URDF loading. |
| [`ferromotion-control`](https://crates.io/crates/ferromotion-control) | PID · computed-torque · Cartesian impedance · **series-elastic actuator / transmission model** · LQR · linear MPC · OSC · WBC · **HQP** (strict-priority hierarchical QP) · placo · iLQR/DDP · **constrained DDP (PROXDDP)** · **direct collocation (DIRCON)** · MPPI · CEM · CBF-QP · **HJ reachability** (backward reachable tubes) · sliding-mode · **SRBD MPC** · capture-point/ZMP · **SLIP + Raibert hopper** (running) · **DCM walking + footstep planning** · **TOPP** (time-optimal path parameterization) · **quadrotor differential flatness + minimum-snap** · centroidal MPC · Kalman/EKF/UKF · **invariant EKF (InEKF)** on SE₂(3) · **Koopman/EDMD** (data-driven linearization) · **IMU preintegration** (Forster, on-manifold, + leg-odometry bias correction) · **RMPflow** (Riemannian motion policies — reactive multi-task motion) · **IBVS** (image-based visual servoing) · **DMPs** (learning-from-demonstration primitives) · complementary filter · momentum observer. |
| [`ferromotion-fluid`](https://crates.io/crates/ferromotion-fluid) | 2D incompressible **Navier–Stokes** (MAC projection) for fluid–robot interaction — verified against the Ghia Re=100 lid-driven-cavity benchmark. |
| [`ferromotion-mpm`](https://crates.io/crates/ferromotion-mpm) | Differentiable 2D **Material Point Method** (MLS-MPM, neo-Hookean) for soft/elastic/granular material — analytic material-stiffness gradients. |
| [`ferromotion-tactile`](https://crates.io/crates/ferromotion-tactile) | Differentiable **optical-tactile** (GelSight/DIGIT) sensor simulation — gel deformation → photometric image. |
| [`ferromotion-rod`](https://crates.io/crates/ferromotion-rod) | Differentiable **Discrete Elastic Rods** (stretch + bending) for cables, tendons, and continuum robots — validated vs Euler-Bernoulli. |
| [`ferromotion-cloth`](https://crates.io/crates/ferromotion-cloth) | Differentiable **FEM thin-shell cloth** (StVK membrane + bending) — exact forces + analytic material-stiffness gradients. |
| [`ferromotion-ruckig`](https://crates.io/crates/ferromotion-ruckig) | Jerk-limited online trajectory generation. |
| [`ferromotion-policy`](https://crates.io/crates/ferromotion-policy) | On-device runner for exported learned (RL/VLA) policies — MLP inference + **flow-matching action sampler** (ODE integration of a learned velocity field). |
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

## In the browser
`ferromotion-wasm` builds with `wasm-pack` to a self-contained module. See `demo/` for a page that
drives a robot arm with live IK and an obstacle-avoiding planned trajectory — computed entirely
on-device, no server. Validated end-to-end on a real open-source robot (NormaCore's 3D-printed
7-DoF ElRobot) straight from its URDF (`crates/ferromotion-core/examples/elrobot.rs`).

## License
Dual-licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
