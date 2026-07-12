# ferromotion-control

[![crates.io](https://img.shields.io/crates/v/ferromotion-control.svg)](https://crates.io/crates/ferromotion-control)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-control)](https://docs.rs/ferromotion-control)

A control corpus for physical AI, in pure Rust (native + `wasm32`), built on
[`ferromotion-core`](https://crates.io/crates/ferromotion-core) dynamics.

- **Feedback / optimal:** PID, LQR, computed-torque, Cartesian impedance, admittance + hybrid force/position.
- **Predictive:** linear MPC, TinyMPC (embedded ADMM), iLQR/DDP, MPPI, CEM, SRBD & centroidal MPC.
- **Whole-body / task-space:** operational-space control (OSC), whole-body QP (WBC), placo-style QP-IK.
- **Safety / robust:** CBF-QP safety filter, sliding-mode control.
- **Legged:** capture point, ZMP preview.
- **Multi-agent:** ALGAMES game-theoretic (generalized-Nash) trajectory optimization.
- **Estimation:** Kalman / EKF / UKF, complementary filter, generalized-momentum observer.
- **Solvers:** ReLUQP (unrolled-ADMM QP), TrajectoryBundles (gradient-free), plus a `clarabel` QP backend.

Every controller is verified in closed loop (against `ferromotion-core::forward_dynamics`).

Part of [Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion). Dual-licensed MIT OR Apache-2.0.
