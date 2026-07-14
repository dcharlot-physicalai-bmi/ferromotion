//! # Ferromotion
//!
//! A Rust library for the **kinematics, dynamics, and control of physical AI** — native and in the
//! browser (WebAssembly). This umbrella crate re-exports the components; depend on the individual
//! `ferromotion-*` crates directly if you want a leaner build.
//!
//! - [`core`] — forward kinematics, analytic Jacobians, RNEA dynamics, IK, trajectory optimization,
//!   collision, motion retargeting, URDF loading.
//! - [`control`] — PID, computed-torque, impedance, LQR, MPC, OSC, WBC, iLQR/DDP, MPPI, CBF-QP,
//!   sliding-mode, SRBD MPC, ZMP/capture-point, Kalman/EKF/UKF, and more.
//! - [`ruckig`] — jerk-limited online trajectory generation.
//! - [`policy`] — on-device runner for exported learned (RL/VLA) policies.
//! - [`fluid`] — 2D incompressible Navier–Stokes (MAC projection) for fluid–robot interaction.
//! - [`mpm`] — differentiable 2D Material Point Method for soft/elastic/granular material.
//! - [`cloth`] — differentiable FEM thin-shell cloth (StVK membrane + bending).
//! - [`tactile`] — differentiable optical-tactile (GelSight/DIGIT) sensor simulation.
//!
//! From the [Institute for Physical AI](https://physicalai-bmi.org). Sibling to `ferric` (the
//! pure-Rust compute fabric).

pub use ferromotion_control as control;
pub use ferromotion_core as core;
pub use ferromotion_fluid as fluid;
pub use ferromotion_cloth as cloth;
pub use ferromotion_mpm as mpm;
pub use ferromotion_policy as policy;
pub use ferromotion_ruckig as ruckig;
pub use ferromotion_tactile as tactile;
