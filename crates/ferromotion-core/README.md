# ferromotion-core

[![crates.io](https://img.shields.io/crates/v/ferromotion-core.svg)](https://crates.io/crates/ferromotion-core)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-core)](https://docs.rs/ferromotion-core)

Kinematics, dynamics, and optimization for physical AI, in pure Rust (native + `wasm32`).

- **Kinematics:** SE(3) forward kinematics, analytic geometric + multi-frame Jacobians, URDF loading
  (from a string — browser-friendly), robust IK (Levenberg–Marquardt + random-restart).
- **Dynamics:** Recursive Newton-Euler inverse dynamics, forward dynamics, joint-space mass matrix,
  gravity compensation, and a symplectic free-rigid-body integrator.
- **Optimization:** composable costs, block-tridiagonal trajectory optimization, sparse factor-graph
  solve, augmented-Lagrangian hard constraints, motion retargeting (position / vector / DexPilot).
- **Contact:** convex contact-implicit dynamics, SOC Coulomb friction, interior-point differentiable
  contact (Dojo-style, smooth through stick↔slip), DCOL differentiable collision, a planar
  rigid-body-with-friction simulator, and **articulated-multibody contact** — differentiable
  frictional floor contact on a full robot chain, with the control gradient `∂q̇⁺/∂τ`.

```rust
use ferromotion_core::{from_urdf_str, solve_ik, IkOptions};
use nalgebra::{Isometry3, Translation3};

let robot = from_urdf_str(urdf, "base_link", "tool").unwrap();
let target = Isometry3::from_parts(Translation3::new(0.4, 0.1, 0.3), Default::default());
let res = solve_ik(&robot, &target, &vec![0.0; robot.dof()], &IkOptions::default());
```

Part of [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion). Dual-licensed MIT OR Apache-2.0.
