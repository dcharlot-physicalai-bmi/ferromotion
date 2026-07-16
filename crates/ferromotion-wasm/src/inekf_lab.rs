//! **Estimator lab** — the rig behind the textbook chapter on why some state estimators stay honest.
//!
//! A robot dead-reckons and its guess drifts from the truth. What a filter needs is a model of how that
//! *error* grows, so it can keep an honest sense of its own uncertainty. The trouble is that the error
//! dynamics are nonlinear, so a standard EKF linearizes them around its current — wrong — estimate, and
//! the model degrades exactly when the estimate is worst. Barrau & Bonnabel's **invariant EKF** picks a
//! different error, `η = X̂ X⁻¹` on the group `SE₂(3)`, whose evolution `ξ̇ = A ξ` has a matrix `A` that
//! depends only on gravity — **not on the estimate at all**. Its linear error model is therefore *exact*,
//! for any error, however large.
//!
//! The rig propagates a true state and a deliberately-wrong estimate through the same IMU stream (all
//! via the real [`ferromotion_control`] `Se23`/`riekf_a_matrix`/`standard_ekf_f`), and compares the true
//! error against two linear predictions — the InEKF's estimate-independent one, and a standard EKF's
//! estimate-dependent one — so the reader can crank the error up and watch one stay exact while the
//! other drifts.

use ferromotion_control::{riekf_a_matrix, standard_ekf_f, InEkf, Matrix9, Se23, Vector9};
use ferromotion_control::exp_so3;
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

fn gravity() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -9.81)
}

/// Closed-form `exp(M·dt)` for the nilpotent error-transition matrices here (`M³ = 0`), so no matrix
/// exponential is needed: `exp(M·dt) = I + M·dt + ½ M² dt²`.
fn nilpotent_exp(m: &Matrix9, dt: f64) -> Matrix9 {
    Matrix9::identity() + m * dt + (m * m) * (0.5 * dt * dt)
}

#[wasm_bindgen]
pub struct InekfLab {
    // recorded traces
    t: Vec<f64>,
    inekf_resid: Vec<f64>,
    ekf_resid: Vec<f64>,
    true_xy: Vec<f64>,
    est_xy: Vec<f64>,
    err_norm: Vec<f64>,
}

/// The fixed direction of the initial error `ξ₀` (attitude, velocity, position), scaled by the reader.
const XI0_DIR: [f64; 9] = [0.10, -0.14, 0.20, 0.30, 0.16, -0.22, 0.55, -0.35, 0.22];

#[wasm_bindgen]
impl InekfLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> InekfLab {
        InekfLab { t: vec![], inekf_resid: vec![], ekf_resid: vec![], true_xy: vec![], est_xy: vec![], err_norm: vec![] }
    }

    /// Propagate truth and a wrong estimate (initial error `scale·ξ₀`) through a shared IMU stream,
    /// recording the true right-invariant error against the InEKF and EKF linear predictions.
    pub fn simulate(&mut self, scale: f64) {
        let g = gravity();
        let xi0 = Vector9::from_column_slice(&XI0_DIR) * scale;

        let mut truth = Se23 {
            r: exp_so3(Vector3::new(0.15, -0.1, 0.3)),
            v: Vector3::new(0.4, 0.15, 0.0),
            p: Vector3::new(0.0, 0.0, 1.0),
        };
        let mut est = Se23::exp(&xi0).compose(&truth); // X̂ = Exp(ξ₀)·X (right-invariant perturbation)

        let a = riekf_a_matrix(g);
        let mut phi_ekf = Matrix9::identity(); // accumulated standard-EKF transition (estimate-dependent)

        let (dt, steps, every) = (2e-3, 1000, 10);
        self.t.clear();
        self.inekf_resid.clear();
        self.ekf_resid.clear();
        self.true_xy.clear();
        self.est_xy.clear();
        self.err_norm.clear();

        for k in 0..=steps {
            let time = k as f64 * dt;
            // IMU stream (shared by truth and estimate).
            let omega = Vector3::new(0.6 * (1.5 * time).sin(), 0.2, 0.9 * (0.8 * time).cos());
            let accel = Vector3::new(0.8 * time.cos(), 0.5, 2.0 * (0.7 * time).sin());

            if k % every == 0 {
                let xi_actual = est.compose(&truth.inverse()).log();
                let xi_inekf = nilpotent_exp(&a, time) * xi0; // exact, estimate-independent
                let xi_ekf = phi_ekf * xi0; // standard EKF: linearized around the (wrong) estimate
                self.t.push(time);
                self.inekf_resid.push((xi_actual - xi_inekf).norm());
                self.ekf_resid.push((xi_actual - xi_ekf).norm());
                self.err_norm.push(xi_actual.norm());
                self.true_xy.push(truth.p.x);
                self.true_xy.push(truth.p.y);
                self.est_xy.push(est.p.x);
                self.est_xy.push(est.p.y);
            }

            if k == steps {
                break;
            }
            // Standard-EKF transition uses the ESTIMATE's attitude and the measured accel.
            let f = standard_ekf_f(&est.r, accel);
            phi_ekf = nilpotent_exp(&f, dt) * phi_ekf;
            truth = InEkf::propagate_state(&truth, omega, accel, g, dt);
            est = InEkf::propagate_state(&est, omega, accel, g, dt);
        }
    }

    pub fn t(&self) -> Vec<f64> { self.t.clone() }
    pub fn inekf_residual(&self) -> Vec<f64> { self.inekf_resid.clone() }
    pub fn ekf_residual(&self) -> Vec<f64> { self.ekf_resid.clone() }
    pub fn err_norm(&self) -> Vec<f64> { self.err_norm.clone() }
    pub fn true_xy(&self) -> Vec<f64> { self.true_xy.clone() }
    pub fn est_xy(&self) -> Vec<f64> { self.est_xy.clone() }

    /// Peak InEKF prediction error over the run (should stay ~0 for any scale — the theorem).
    pub fn inekf_peak(&self) -> f64 {
        self.inekf_resid.iter().cloned().fold(0.0, f64::max)
    }
    /// Peak standard-EKF prediction error (grows with the estimate error).
    pub fn ekf_peak(&self) -> f64 {
        self.ekf_resid.iter().cloned().fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn the_invariant_error_model_is_exact_at_any_error_size() {
        // THE CHAPTER. The InEKF's linear prediction matches the true nonlinear error to machine
        // precision — and it stays exact as the initial error is scaled from small to enormous.
        for scale in [0.2, 1.0, 3.0, 8.0] {
            let mut lab = InekfLab::new();
            lab.simulate(scale);
            assert!(
                lab.inekf_peak() < 1e-4,
                "InEKF prediction should be exact at scale {scale}: peak {}",
                lab.inekf_peak()
            );
        }
    }

    #[test]
    fn the_standard_ekf_model_drifts_and_worsens_with_error() {
        // Linearizing around the (wrong) estimate leaves a growing residual, and it grows with the
        // estimate error — the InEKF's whole reason for being.
        let mut small = InekfLab::new();
        small.simulate(1.0);
        let mut big = InekfLab::new();
        big.simulate(6.0);
        // The EKF model is far worse than the InEKF model …
        assert!(big.ekf_peak() > 100.0 * big.inekf_peak(), "EKF model should be far worse: {} vs {}", big.ekf_peak(), big.inekf_peak());
        // … and it degrades as the error grows (measured relative to the error size).
        let small_rel = small.ekf_peak() / small.err_norm.iter().cloned().fold(0.0, f64::max);
        let big_rel = big.ekf_peak() / big.err_norm.iter().cloned().fold(0.0, f64::max);
        assert!(big_rel > small_rel, "EKF relative error should worsen with the estimate error: {big_rel} vs {small_rel}");
    }

    #[test]
    fn the_invariant_attitude_error_is_conserved() {
        // A corollary of A's structure: the right-invariant attitude error does not drift, so the
        // reported error norm should never fall below the (constant) attitude-error magnitude.
        let mut lab = InekfLab::new();
        lab.simulate(1.0);
        let att0 = Vector3::new(XI0_DIR[0], XI0_DIR[1], XI0_DIR[2]).norm();
        assert!(lab.err_norm.iter().all(|&e| e > att0 - 1e-6), "error norm dipped below the conserved attitude error");
    }

    #[test]
    fn a_is_state_independent_but_f_is_not() {
        // The mechanism, directly: A is the same everywhere; F changes with the estimate's attitude.
        let a1 = riekf_a_matrix(gravity());
        let a2 = riekf_a_matrix(gravity());
        assert!((a1 - a2).norm() < 1e-15, "A must not depend on anything but gravity");
        let accel = Vector3::new(0.5, -0.3, 9.0);
        let f1 = standard_ekf_f(&Matrix3::identity(), accel);
        let f2 = standard_ekf_f(&exp_so3(Vector3::new(0.0, 0.0, 1.2)), accel);
        assert!((f1 - f2).norm() > 1e-3, "F must depend on the estimate's attitude");
    }
}
