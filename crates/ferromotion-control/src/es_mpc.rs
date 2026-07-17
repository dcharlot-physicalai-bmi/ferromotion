//! **Error-state control on a matrix Lie group** (Teng, Ghaffari et al., IROS 2022) — the control-side
//! twin of the invariant EKF: formulate attitude regulation and tracking in the Lie *algebra* so the
//! problem is singularity-free and consistent with `SO(3)`, unlike an Euler-angle LQR that degrades near
//! gimbal lock.
//!
//! The rigid body obeys `Ṙ = R [ω]ₓ`, `J ω̇ = −ω × Jω + τ`. Measure the attitude error to the reference
//! as `ξ = log(R_dᵀ R) ∈ so(3)` and stack the error state `x = [ξ; δω]`. Near the reference the error
//! dynamics linearize to a **double integrator on the algebra**,
//!
//! ```text
//!   ξ̇ = δω,   δω̇ = J⁻¹ τ    ⇒   A = [[0, I],[0, 0]],  B = [[0],[J⁻¹]],
//! ```
//!
//! whose infinite-horizon LQR gain (the converged error-state MPC) is solved once by [`crate::dlqr`].
//! The control is `τ = τ_ff − K x`, with a feedforward `τ_ff = ω_d × J ω_d + J ω̇_d` for tracking. The
//! linearization is state-independent — the whole point — so one gain works over the entire group.
//! Reuses `exp_so3`/`right_jacobian`. Pure `nalgebra` → WASM-clean.

use crate::{dlqr, exp_so3};
use nalgebra::{DMatrix, Matrix3, Vector3};

/// Log map of `SO(3)` → `so(3)` (rotation vector), the minimal-angle (geodesic) error.
pub fn log_so3(r: &Matrix3<f64>) -> Vector3<f64> {
    let c = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let t = c.acos();
    let v = Vector3::new(r[(2, 1)] - r[(1, 2)], r[(0, 2)] - r[(2, 0)], r[(1, 0)] - r[(0, 1)]);
    if t < 1e-9 {
        v * 0.5
    } else {
        v * (t / (2.0 * t.sin()))
    }
}

/// Error-state attitude controller for a rigid body with diagonal inertia `J`.
#[derive(Clone, Debug)]
pub struct EsAttitude {
    /// Diagonal inertia.
    pub j: Vector3<f64>,
    /// LQR gain `K` (3×6), mapping `[ξ; δω]` to a corrective torque.
    pub k: DMatrix<f64>,
}

impl EsAttitude {
    /// Build the error-state LQR from the tangent-space double-integrator, weighting attitude error
    /// `q_att`, rate error `q_rate`, and control effort `r_ctrl`. `dt` is the discretization step.
    pub fn new(j: Vector3<f64>, dt: f64, q_att: f64, q_rate: f64, r_ctrl: f64) -> Self {
        let jinv = Matrix3::from_diagonal(&Vector3::new(1.0 / j.x, 1.0 / j.y, 1.0 / j.z));
        // Continuous A = [[0,I],[0,0]], B = [[0],[J⁻¹]]. A is nilpotent (A²=0) ⇒ exact discretization
        // A_d = I + A·dt,  B_d = (I·dt + A·dt²/2)·B.
        let mut ad = DMatrix::<f64>::identity(6, 6);
        for i in 0..3 {
            ad[(i, 3 + i)] = dt;
        }
        let mut bd = DMatrix::<f64>::zeros(6, 3);
        for i in 0..3 {
            for k in 0..3 {
                bd[(i, k)] = 0.5 * dt * dt * jinv[(i, k)]; // top block: A·dt²/2·B
                bd[(3 + i, k)] = dt * jinv[(i, k)]; // bottom block: I·dt·B
            }
        }
        let mut q = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            q[(i, i)] = q_att;
            q[(3 + i, 3 + i)] = q_rate;
        }
        let r = DMatrix::<f64>::identity(3, 3) * r_ctrl;
        let k = dlqr(&ad, &bd, &q, &r);
        EsAttitude { j, k }
    }

    /// The control torque to track `(R_d, ω_d, ω̇_d)` from the current `(R, ω)`.
    /// `τ = ω_d × J ω_d + J ω̇_d − K [ξ; δω]`, with `ξ = log(R_dᵀ R)` and `δω = ω − Rᵀ R_d ω_d`.
    pub fn control(&self, r: &Matrix3<f64>, omega: Vector3<f64>, r_d: &Matrix3<f64>, omega_d: Vector3<f64>, omega_d_dot: Vector3<f64>) -> Vector3<f64> {
        let xi = log_so3(&(r_d.transpose() * r));
        let dω = omega - r.transpose() * r_d * omega_d;
        let jd = Matrix3::from_diagonal(&self.j);
        let ff = omega_d.cross(&(jd * omega_d)) + jd * omega_d_dot;
        let x = nalgebra::DVector::from_row_slice(&[xi.x, xi.y, xi.z, dω.x, dω.y, dω.z]);
        let fb = &self.k * x;
        ff - Vector3::new(fb[0], fb[1], fb[2])
    }

    /// One nonlinear rigid-body step: `J ω̇ = −ω × Jω + τ`, `R ← R exp([ω]ₓ dt)` (semi-implicit).
    pub fn step(&self, r: &Matrix3<f64>, omega: Vector3<f64>, tau: Vector3<f64>, dt: f64) -> (Matrix3<f64>, Vector3<f64>) {
        let jd = Matrix3::from_diagonal(&self.j);
        let jinv = Matrix3::from_diagonal(&Vector3::new(1.0 / self.j.x, 1.0 / self.j.y, 1.0 / self.j.z));
        let wdot = jinv * (-omega.cross(&(jd * omega)) + tau);
        let w2 = omega + wdot * dt;
        let r2 = r * exp_so3(w2 * dt);
        (r2, w2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::right_jacobian;

    fn ctrl() -> EsAttitude {
        EsAttitude::new(Vector3::new(0.05, 0.08, 0.10), 0.01, 40.0, 8.0, 0.5)
    }

    #[test]
    fn zero_error_gives_zero_regulation_torque() {
        // At the reference with zero rate, the feedback vanishes (feedforward is zero for ω_d=0).
        let c = ctrl();
        let r = exp_so3(Vector3::new(0.3, -0.2, 0.1));
        let tau = c.control(&r, Vector3::zeros(), &r, Vector3::zeros(), Vector3::zeros());
        assert!(tau.norm() < 1e-12, "no error ⇒ no torque, got {tau:?}");
    }

    #[test]
    fn the_error_state_linearization_is_xidot_equals_omega() {
        // The tangent model claims ξ̇ = δω. Verify d/dt log(R_dᵀ R) = J_r⁻¹(ξ)·ω → ω near the
        // reference, by finite-differencing the true log-error under a small body-rate.
        let r_d: Matrix3<f64> = Matrix3::identity();
        let (omega, dt) = (Vector3::new(0.4, -0.3, 0.6), 1e-6);
        for base in [Vector3::zeros(), Vector3::new(0.05, -0.02, 0.03)] {
            let r0 = exp_so3(base);
            let r1 = r0 * exp_so3(omega * dt);
            let xidot = (log_so3(&(r_d.transpose() * r1)) - log_so3(&(r_d.transpose() * r0))) / dt;
            // near identity ξ̇ ≈ ω; the exact map is J_r⁻¹(ξ)ω, so check that too.
            let exact = right_jacobian(log_so3(&r0)).try_inverse().unwrap() * omega;
            assert!((xidot - exact).norm() < 1e-3, "log-error rate {xidot:?} vs J_r⁻¹ω {exact:?}");
        }
    }

    #[test]
    fn regulates_from_a_large_attitude_error_without_singularity() {
        // THE CHAPTER. Start 2.5 rad off (well past π/2, gimbal-lock territory for Euler angles) about
        // a tilted axis; the error-state controller drives it to the reference on the geodesic.
        let c = ctrl();
        let axis = Vector3::new(1.0, -0.8, 0.6).normalize();
        let (mut r, mut omega) = (exp_so3(axis * 2.5), Vector3::zeros());
        let r_d = Matrix3::identity();
        let err0 = log_so3(&r).norm();
        let mut prev = err0;
        for step in 0..4000 {
            let tau = c.control(&r, omega, &r_d, Vector3::zeros(), Vector3::zeros());
            let (r2, w2) = c.step(&r, omega, tau, 0.01);
            r = r2;
            omega = w2;
            let e = log_so3(&r).norm();
            // no blow-up / singularity at any point
            assert!(e.is_finite() && e <= err0 + 0.2, "error grew/diverged at step {step}: {e}");
            prev = e;
        }
        assert!(prev < 1e-2, "attitude did not converge: final error {prev} rad (started {err0})");
        assert!(omega.norm() < 1e-2, "angular velocity did not settle: {}", omega.norm());
    }

    #[test]
    fn tracks_a_spinning_reference() {
        // Feedforward + error feedback: follow a reference spinning at constant ω_d from a wrong start.
        let c = ctrl();
        let omega_d = Vector3::new(0.0, 0.0, 1.5); // spin about z
        let (mut r, mut omega) = (exp_so3(Vector3::new(0.4, 0.3, 0.0)), Vector3::zeros());
        let mut r_d = Matrix3::identity();
        let dt = 0.005;
        let mut err_late: f64 = 1e9;
        for k in 0..4000 {
            let tau = c.control(&r, omega, &r_d, omega_d, Vector3::zeros());
            let (r2, w2) = c.step(&r, omega, tau, dt);
            r = r2;
            omega = w2;
            r_d *= exp_so3(omega_d * dt); // advance the reference
            if k > 3000 {
                err_late = err_late.min(log_so3(&(r_d.transpose() * r)).norm());
            }
        }
        assert!(err_late < 5e-2, "did not lock onto the spinning reference: {err_late}");
    }

    #[test]
    fn the_error_is_the_short_way_around() {
        // log gives the minimal-angle (geodesic) error, so a 179° error resolves the short way — the
        // rotation-vector norm never exceeds π.
        for ang in [0.5, 3.0, 3.1] {
            let r = exp_so3(Vector3::new(0.0, 0.0, ang));
            let e = log_so3(&r).norm();
            let wrapped = ang.min(std::f64::consts::TAU - ang);
            assert!((e - wrapped).abs() < 1e-6 && e <= std::f64::consts::PI + 1e-9, "geodesic error wrong at {ang}: {e}");
        }
    }
}
