# ferromotion

[![crates.io](https://img.shields.io/crates/v/ferromotion.svg)](https://crates.io/crates/ferromotion)

The umbrella crate for **Ferromotion** — a Rust library for the kinematics, dynamics, and control of
physical AI, native and in the browser (WebAssembly). Re-exports the components:

```rust
use ferromotion::{core, control, ruckig, policy};
```

- [`core`](https://crates.io/crates/ferromotion-core) — FK, Jacobians, RNEA dynamics, IK, trajectory
  optimization, contact-implicit dynamics, collision, motion retargeting, URDF.
- [`control`](https://crates.io/crates/ferromotion-control) — PID, MPC, OSC, WBC, iLQR/DDP, MPPI, CBF,
  SRBD/centroidal MPC, ALGAMES, Kalman/EKF/UKF, and more.
- [`ruckig`](https://crates.io/crates/ferromotion-ruckig) — jerk-limited online trajectory generation.
- [`policy`](https://crates.io/crates/ferromotion-policy) — on-device runner for exported learned policies.

Depend on the individual crates directly for a leaner build. See the
[repository](https://github.com/dcharlot-physicalai-bmi/ferromotion) for the full picture.

From the [Institute for Physical AI](https://physicalai-bmi.org). Dual-licensed MIT OR Apache-2.0.
