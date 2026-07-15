//! **Invariant EKF (InEKF)** — Barrau & Bonnabel's invariant observer, the estimator behind modern
//! legged/visual-inertial state estimation (e.g. Hartley et al.'s contact-aided InEKF).
//!
//! The state lives on the matrix Lie group `SE₂(3)` — attitude, velocity, position together:
//!
//! ```text
//!   X = [ R  v  p ]        (R, v, p) · (R', v', p') = (R R',  R v' + v,  R p' + p)
//!       [ 0  1  0 ]
//!       [ 0  0  1 ]
//! ```
//!
//! IMU dead-reckoning on this group is **group-affine**, and Barrau & Bonnabel's theorem then says the
//! right-invariant error `η = X̂ X⁻¹` obeys an *exactly* **log-linear** equation `ξ̇ = A ξ` in which
//!
//! ```text
//!   A = [ 0      0  0 ]      ξ = [θ; ν; ρ]   (attitude, velocity, position error)
//!       [ ĝ×     0  0 ]
//!       [ 0      I  0 ]
//! ```
//!
//! **`A` depends only on gravity — not on the state estimate.** That is the whole point: a standard
//! EKF's Jacobian depends on `R̂` and the measured acceleration, so its linearization (and consistency)
//! degrade with estimate error, while the InEKF's does not. Pure `nalgebra` → WASM-clean.

use crate::{exp_so3, right_jacobian};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

/// 9×9 error-state matrix, ordered `[θ; ν; ρ]`.
pub type Matrix9 = SMatrix<f64, 9, 9>;
/// 9-vector error state `[θ; ν; ρ]`.
pub type Vector9 = SVector<f64, 9>;

fn hat(w: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

fn log_so3(r: &Matrix3<f64>) -> Vector3<f64> {
    let c = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let t = c.acos();
    let v = Vector3::new(r[(2, 1)] - r[(1, 2)], r[(0, 2)] - r[(2, 0)], r[(1, 0)] - r[(0, 1)]);
    if t < 1e-9 { v * 0.5 } else { v * (t / (2.0 * t.sin())) }
}

/// Left Jacobian of `SO(3)` (`J_l(θ) = J_r(−θ)`).
fn left_jacobian(w: Vector3<f64>) -> Matrix3<f64> {
    right_jacobian(-w)
}

/// An element of `SE₂(3)`: attitude, velocity, position.
#[derive(Clone, Copy, Debug)]
pub struct Se23 {
    pub r: Matrix3<f64>,
    pub v: Vector3<f64>,
    pub p: Vector3<f64>,
}

impl Se23 {
    pub fn identity() -> Self {
        Self { r: Matrix3::identity(), v: Vector3::zeros(), p: Vector3::zeros() }
    }

    /// Group composition `self · other`.
    pub fn compose(&self, o: &Se23) -> Se23 {
        Se23 { r: self.r * o.r, v: self.r * o.v + self.v, p: self.r * o.p + self.p }
    }

    pub fn inverse(&self) -> Se23 {
        let rt = self.r.transpose();
        Se23 { r: rt, v: -rt * self.v, p: -rt * self.p }
    }

    /// Group exponential of `ξ = [θ; ν; ρ]`.
    pub fn exp(xi: &Vector9) -> Se23 {
        let th = Vector3::new(xi[0], xi[1], xi[2]);
        let nu = Vector3::new(xi[3], xi[4], xi[5]);
        let rho = Vector3::new(xi[6], xi[7], xi[8]);
        let jl = left_jacobian(th);
        Se23 { r: exp_so3(th), v: jl * nu, p: jl * rho }
    }

    /// Group logarithm.
    pub fn log(&self) -> Vector9 {
        let th = log_so3(&self.r);
        let jli = left_jacobian(th).try_inverse().unwrap_or_else(Matrix3::identity);
        let nu = jli * self.v;
        let rho = jli * self.p;
        Vector9::from_column_slice(&[th.x, th.y, th.z, nu.x, nu.y, nu.z, rho.x, rho.y, rho.z])
    }
}

/// The right-invariant error-dynamics matrix `A`. **It depends only on gravity** — the signature takes
/// no state, which is precisely Barrau & Bonnabel's log-linear property.
pub fn riekf_a_matrix(gravity: Vector3<f64>) -> Matrix9 {
    let mut a = Matrix9::zeros();
    a.fixed_view_mut::<3, 3>(3, 0).copy_from(&hat(gravity)); // ν̇ = g× θ
    a.fixed_view_mut::<3, 3>(6, 3).copy_from(&Matrix3::identity()); // ρ̇ = ν
    a
}

/// For contrast: a *standard* EKF's attitude/velocity Jacobian block, which **does** depend on the
/// estimate (`R̂`) and the measured acceleration — the thing the InEKF avoids.
pub fn standard_ekf_f(r_hat: &Matrix3<f64>, accel: Vector3<f64>) -> Matrix9 {
    let mut f = Matrix9::zeros();
    f.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-r_hat * hat(accel))); // depends on R̂ and a
    f.fixed_view_mut::<3, 3>(6, 3).copy_from(&Matrix3::identity());
    f
}

/// A right-invariant EKF over `SE₂(3)` driven by IMU measurements.
#[derive(Clone, Debug)]
pub struct InEkf {
    pub x: Se23,
    pub cov: Matrix9,
    pub gravity: Vector3<f64>,
}

impl InEkf {
    pub fn new(x: Se23, cov: Matrix9, gravity: Vector3<f64>) -> Self {
        Self { x, cov, gravity }
    }

    /// IMU dead-reckoning of the group state (group-affine dynamics).
    pub fn propagate_state(x: &Se23, omega: Vector3<f64>, accel: Vector3<f64>, gravity: Vector3<f64>, dt: f64) -> Se23 {
        let a_world = x.r * accel + gravity;
        Se23 {
            r: x.r * exp_so3(omega * dt),
            v: x.v + a_world * dt,
            p: x.p + x.v * dt + 0.5 * a_world * dt * dt,
        }
    }

    /// Propagate state and covariance. `Φ = exp(A·dt)` with the *state-independent* `A`.
    pub fn propagate(&mut self, omega: Vector3<f64>, accel: Vector3<f64>, dt: f64, q: &Matrix9) {
        self.x = Self::propagate_state(&self.x, omega, accel, self.gravity, dt);
        let phi = (riekf_a_matrix(self.gravity) * dt).exp();
        self.cov = phi * self.cov * phi.transpose() + q * dt;
    }

    /// Update from a direct position measurement `y ≈ p`. For the right-invariant error
    /// `ξ = [θ; ν; ρ]`, `y − p̂ ≈ [p̂×, 0, −I]·ξ`, so the correction is applied on the group as
    /// `X̂ ← Exp(−δξ)·X̂`.
    pub fn update_position(&mut self, y: Vector3<f64>, noise: &Matrix3<f64>) {
        let mut h = SMatrix::<f64, 3, 9>::zeros();
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat(self.x.p));
        h.fixed_view_mut::<3, 3>(0, 6).copy_from(&(-Matrix3::identity()));
        let s = h * self.cov * h.transpose() + noise;
        let k = self.cov * h.transpose() * s.try_inverse().expect("innovation covariance invertible");
        let r = y - self.x.p; // innovation
        let dxi: Vector9 = k * r;
        self.x = Se23::exp(&(-dxi)).compose(&self.x); // right-invariant correction
        self.cov = (Matrix9::identity() - k * h) * self.cov;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, -9.81)
    }

    #[test]
    fn se23_group_operations_are_consistent() {
        let xi = Vector9::from_column_slice(&[0.3, -0.2, 0.15, 1.0, -0.5, 0.2, 0.4, 0.7, -0.3]);
        let x = Se23::exp(&xi);
        // exp/log round-trip.
        assert!((x.log() - xi).norm() < 1e-9, "log(exp(ξ)) ≠ ξ");
        // X · X⁻¹ = identity.
        let i = x.compose(&x.inverse());
        assert!((i.r - Matrix3::identity()).norm() < 1e-12 && i.v.norm() < 1e-12 && i.p.norm() < 1e-12);
        // Associativity.
        let y = Se23::exp(&(xi * 0.4));
        let z = Se23::exp(&(xi * -0.7));
        let l = x.compose(&y).compose(&z);
        let r = x.compose(&y.compose(&z));
        assert!((l.r - r.r).norm() < 1e-12 && (l.v - r.v).norm() < 1e-12 && (l.p - r.p).norm() < 1e-12);
    }

    #[test]
    fn right_invariant_error_is_exactly_log_linear() {
        // Barrau & Bonnabel's theorem: with group-affine IMU dynamics, the right-invariant error
        // satisfies ξ(t) = exp(A·t)·ξ(0) exactly — with A independent of the trajectory.
        let mut truth = Se23 { r: exp_so3(Vector3::new(0.2, -0.1, 0.4)), v: Vector3::new(0.3, 0.1, -0.2), p: Vector3::new(1.0, -2.0, 0.5) };
        // The estimate differs by a right-invariant perturbation: X̂ = Exp(ξ₀)·X.
        let xi0 = Vector9::from_column_slice(&[0.02, -0.03, 0.01, 0.05, 0.02, -0.04, 0.1, -0.05, 0.03]);
        let mut est = Se23::exp(&xi0).compose(&truth);

        let a = riekf_a_matrix(g());
        let (dt, steps) = (1e-4, 5000); // t = 0.5 s
        for k in 0..steps {
            // Both propagate with the *same* IMU measurements.
            let t = k as f64 * dt;
            let omega = Vector3::new(0.5 * (2.0 * t).sin(), 0.3, -0.4 * (1.5 * t).cos());
            let accel = Vector3::new(1.0 * (t).cos(), -0.7, 2.0 * (0.8 * t).sin());
            truth = InEkf::propagate_state(&truth, omega, accel, g(), dt);
            est = InEkf::propagate_state(&est, omega, accel, g(), dt);
        }
        let t_end = steps as f64 * dt;
        let xi_actual = est.compose(&truth.inverse()).log();
        let xi_pred = (a * t_end).exp() * xi0;
        assert!((xi_actual - xi_pred).norm() < 1e-6, "log-linear property violated: {xi_actual:?} vs {xi_pred:?}");

        // Corollary of A's structure: the right-invariant *attitude* error is constant.
        assert!((xi_actual.fixed_rows::<3>(0) - xi0.fixed_rows::<3>(0)).norm() < 1e-8, "attitude error should not drift");
    }

    #[test]
    fn a_matrix_is_state_independent_unlike_a_standard_ekf() {
        // The InEKF's A is the same everywhere …
        let a1 = riekf_a_matrix(g());
        let a2 = riekf_a_matrix(g());
        assert!((a1 - a2).norm() < 1e-15);

        // … while a standard EKF's Jacobian changes with the estimate and the measurement.
        let (r1, r2) = (Matrix3::identity(), exp_so3(Vector3::new(0.0, 0.0, 1.2)));
        let accel = Vector3::new(0.5, -0.3, 9.0);
        let f1 = standard_ekf_f(&r1, accel);
        let f2 = standard_ekf_f(&r2, accel);
        assert!((f1 - f2).norm() > 1e-3, "standard EKF Jacobian should depend on the estimate");
    }

    #[test]
    fn filter_converges_from_a_wrong_initial_estimate() {
        let truth = Se23 { r: Matrix3::identity(), v: Vector3::new(0.2, 0.0, 0.0), p: Vector3::new(0.0, 0.0, 1.0) };
        // Start the filter off with a sizeable error.
        let xi0 = Vector9::from_column_slice(&[0.0, 0.0, 0.05, 0.1, -0.1, 0.05, 0.3, 0.2, -0.2]);
        let mut f = InEkf::new(Se23::exp(&xi0).compose(&truth), Matrix9::identity() * 0.5, g());

        let q = Matrix9::identity() * 1e-6;
        let noise = Matrix3::identity() * 1e-4;
        let (dt, steps) = (1e-3, 3000);
        let mut t_state = truth;
        let err0 = (f.x.p - t_state.p).norm();
        for k in 0..steps {
            // Hover: the IMU reads the specific force that cancels gravity.
            let accel = t_state.r.transpose() * (-g());
            let omega = Vector3::zeros();
            t_state = InEkf::propagate_state(&t_state, omega, accel, g(), dt);
            f.propagate(omega, accel, dt, &q);
            if k % 10 == 0 {
                f.update_position(t_state.p, &noise); // noiseless position fixes
            }
        }
        let err = (f.x.p - t_state.p).norm();
        assert!(err < 0.02 && err < 0.2 * err0, "filter did not converge: {err} (started at {err0})");
        assert!((f.x.v - t_state.v).norm() < 0.05, "velocity did not converge");
    }
}
