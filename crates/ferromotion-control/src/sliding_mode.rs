//! Joint-space sliding-mode control (SMC) — robust to model uncertainty and matched disturbances.
//! Define the sliding surface `s = ė + λ·e` (with `e = q − q_des`, `ė = q̇ − q̇_des`). The control
//! blends a model-based equivalent term that drives the reference acceleration
//! `q̈_r = q̈_des − λ·ė` with a robust term that forces `s → 0`:
//! `τ = M(q)·q̈_r + bias(q,q̇) − M(q)·(kd·s + η·sat(s/φ))`,
//! where `bias = C(q,q̇)q̇ + G(q)` (RNEA with `q̈ = 0`) and `sat(·)` is the boundary-layer
//! saturation that replaces the discontinuous `sign(·)` to suppress chattering. Outside the layer
//! the `η·sign(s)` term dominates any bounded matched disturbance (the robustness guarantee); inside
//! it, `kd + η/φ` give smooth exponential convergence of `s`, hence of the tracking error.

use nalgebra::{DVector, Vector3};
use ferromotion_core::{inverse_dynamics, mass_matrix, LinkInertia, Robot};

/// Boundary-layer saturation: `sat(x) = clamp(x, −1, 1)`. Continuous replacement for `sign(x)`.
#[inline]
pub fn sat(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

/// Joint-space sliding-mode controller.
#[derive(Clone, Debug)]
pub struct SlidingMode {
    /// Sliding-surface slope `λ`: sets the closed-loop error time-constant `1/λ` once on the surface.
    pub lambda: f64,
    /// Reaching gain `η`: switching magnitude; must exceed the (unknown) disturbance in `M⁻¹` units.
    pub eta: f64,
    /// Boundary-layer thickness `φ`: width of the linear region of `sat(s/φ)` (bigger = less chatter,
    /// larger steady-state error). Must be `> 0`.
    pub boundary: f64,
    /// Continuous linear feedback on the sliding variable inside the boundary layer.
    pub kd: f64,
}

impl SlidingMode {
    pub fn new(lambda: f64, eta: f64, boundary: f64, kd: f64) -> Self {
        Self { lambda, eta, boundary, kd }
    }

    /// One control step: torques tracking `(q_des, q̇_des)` (regulation when `q̇_des = 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn torque(
        &self,
        robot: &Robot,
        inertia: &[LinkInertia],
        q: &[f64],
        qd: &[f64],
        q_des: &[f64],
        qd_des: &[f64],
        gravity: Vector3<f64>,
    ) -> Vec<f64> {
        let n = robot.dof();
        // Tracking error and sliding surface s = ė + λ·e.
        let e: Vec<f64> = (0..n).map(|i| q[i] - q_des[i]).collect();
        let edot: Vec<f64> = (0..n).map(|i| qd[i] - qd_des[i]).collect();
        let s: Vec<f64> = (0..n).map(|i| edot[i] + self.lambda * e[i]).collect();

        // Reference acceleration for the equivalent term: q̈_r = q̈_des − λ·ė (q̈_des = 0 here).
        let qddr: Vec<f64> = (0..n).map(|i| -self.lambda * edot[i]).collect();

        // Model-based equivalent term M·q̈_r + bias = RNEA(q, q̇, q̈_r, g).
        let m = mass_matrix(robot, inertia, q);
        let bias = inverse_dynamics(robot, inertia, q, qd, &vec![0.0; n], gravity);
        let mqddr = &m * DVector::from_row_slice(&qddr);

        // Robust term mapped through M: M·(kd·s + η·sat(s/φ)), subtracted to drive s → 0.
        let phi = if self.boundary > 0.0 { self.boundary } else { 1e-9 };
        let robust: Vec<f64> = (0..n).map(|i| self.kd * s[i] + self.eta * sat(s[i] / phi)).collect();
        let mrobust = &m * DVector::from_row_slice(&robust);

        (0..n).map(|i| mqddr[i] + bias[i] - mrobust[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use ferromotion_core::{forward_dynamics, from_urdf_full};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    #[test]
    fn sat_is_a_clamped_ramp() {
        assert_eq!(sat(0.3), 0.3);
        assert_eq!(sat(5.0), 1.0);
        assert_eq!(sat(-5.0), -1.0);
        assert_eq!(sat(0.0), 0.0);
    }

    /// Nominal (no disturbance): the surface controller regulates to the setpoint.
    #[test]
    fn regulates_to_setpoint_no_disturbance() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let ctrl = SlidingMode::new(8.0, 40.0, 0.05, 10.0);
        let q_des = [0.5, -0.8];
        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 1e-3);
        for _ in 0..6000 {
            let tau = ctrl.torque(&robot, &inertia, &q, &qd, &q_des, &[0.0, 0.0], g);
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let err = ((q[0] - q_des[0]).powi(2) + (q[1] - q_des[1]).powi(2)).sqrt();
        assert!(err < 5e-3, "joint error {err}, q = {q:?}");
    }

    /// The point of SMC: a constant matched disturbance is injected into the PLANT (added to the
    /// torque before forward dynamics) that the controller does NOT know about. The arm must still
    /// converge to the setpoint within a small boundary-layer error.
    #[test]
    fn rejects_unmodeled_disturbance() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let ctrl = SlidingMode::new(8.0, 60.0, 0.05, 10.0);
        let q_des = [0.5, 0.8];
        // Constant unmodeled disturbance torque (N·m) applied at both joints.
        let disturbance = [1.2, -0.9];
        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 1e-3);
        let mut max_speed: f64 = 0.0;
        for _ in 0..8000 {
            let mut tau = ctrl.torque(&robot, &inertia, &q, &qd, &q_des, &[0.0, 0.0], g);
            for i in 0..2 {
                tau[i] += disturbance[i]; // plant sees the disturbance; controller does not
            }
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
                assert!(q[i].is_finite() && qd[i].is_finite(), "NaN/blowup: q={q:?}, qd={qd:?}");
                max_speed = max_speed.max(qd[i].abs());
            }
        }
        let err = ((q[0] - q_des[0]).powi(2) + (q[1] - q_des[1]).powi(2)).sqrt();
        // Robustness: converges despite a disturbance it never modeled, to within the boundary layer.
        assert!(err < 2e-2, "disturbed joint error {err}, q = {q:?}");
        // No chatter blow-up: velocities stayed bounded and settled.
        assert!(max_speed < 50.0, "unbounded velocity, chatter: {max_speed}");
        assert!(qd[0].abs() < 1e-2 && qd[1].abs() < 1e-2, "did not settle: qd = {qd:?}");
    }

    /// A larger boundary layer trades tracking accuracy for less switching activity: with a
    /// disturbance present, the steady-state error grows with φ (classic SMC boundary-layer bias).
    #[test]
    fn boundary_layer_trades_accuracy_for_smoothness() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let q_des = [0.4, 0.6];
        let disturbance = [1.0, -0.8];
        let run = |phi: f64| {
            // η/φ kept modest so both runs stay in the stable, non-chattering Euler regime.
            let ctrl = SlidingMode::new(8.0, 30.0, phi, 2.0);
            let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 1e-3);
            for _ in 0..8000 {
                let mut tau = ctrl.torque(&robot, &inertia, &q, &qd, &q_des, &[0.0, 0.0], g);
                for i in 0..2 {
                    tau[i] += disturbance[i];
                }
                let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
                for i in 0..2 {
                    qd[i] += qdd[i] * dt;
                    q[i] += qd[i] * dt;
                }
            }
            ((q[0] - q_des[0]).powi(2) + (q[1] - q_des[1]).powi(2)).sqrt()
        };
        let err_thin = run(0.05);
        let err_thick = run(0.4);
        assert!(err_thin.is_finite() && err_thick.is_finite());
        assert!(err_thick > err_thin, "thick φ error {err_thick} not > thin φ error {err_thin}");
    }
}
