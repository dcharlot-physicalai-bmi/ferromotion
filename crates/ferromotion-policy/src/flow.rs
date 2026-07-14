//! **Flow-matching / rectified-flow policy sampling** — the fast alternative to diffusion policies
//! (Lipman et al.; FlowPolicy, AAAI 2025). A flow policy is a learned **velocity field** `v(a, t | obs)`
//! over action space; an action is drawn by starting from a sample `a₀` and integrating the ODE
//! `da/dt = v(a, t | obs)` from `t = 0` to `t = 1`. Unlike diffusion's many stochastic denoising
//! steps, a (rectified) flow is near-straight, so a handful of ODE steps — or even one — suffice,
//! which is what makes it viable for real-time control.
//!
//! This is the on-device *runner*: it integrates a provided velocity field (a closure, or the crate's
//! [`crate::Mlp`] with `[a; t; obs]` as input). Euler and Heun (RK2) integrators are provided. Pure
//! Rust → WASM-clean; training stays out of scope, as with the rest of `ferromotion-policy`.

use crate::Mlp;

/// ODE integrator for the flow.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Integrator {
    Euler,
    /// Heun / explicit trapezoid (RK2) — 2nd order, far more accurate per step.
    Heun,
}

/// Integrate `da/dt = v(a, t)` from `t=0` (`a = a0`) to `t=1` in `steps` steps.
pub fn sample_field(v: &dyn Fn(&[f64], f64) -> Vec<f64>, a0: &[f64], steps: usize, method: Integrator) -> Vec<f64> {
    let h = 1.0 / steps as f64;
    let mut a = a0.to_vec();
    for k in 0..steps {
        let t = k as f64 * h;
        let k1 = v(&a, t);
        match method {
            Integrator::Euler => {
                for i in 0..a.len() {
                    a[i] += h * k1[i];
                }
            }
            Integrator::Heun => {
                let a_pred: Vec<f64> = (0..a.len()).map(|i| a[i] + h * k1[i]).collect();
                let k2 = v(&a_pred, t + h);
                for i in 0..a.len() {
                    a[i] += 0.5 * h * (k1[i] + k2[i]);
                }
            }
        }
    }
    a
}

/// Sample an action from an MLP flow policy conditioned on `obs`, integrating from noise `a0`.
/// The network maps `[a (action-dim); t (1); obs]` → the velocity `v` (action-dim).
pub fn sample_mlp(mlp: &Mlp, obs: &[f64], a0: &[f64], steps: usize, method: Integrator) -> Vec<f64> {
    let field = |a: &[f64], t: f64| -> Vec<f64> {
        let mut input = Vec::with_capacity(a.len() + 1 + obs.len());
        input.extend_from_slice(a);
        input.push(t);
        input.extend_from_slice(obs);
        mlp.forward(&input)
    };
    sample_field(&field, a0, steps, method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, Layer, Mlp};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn constant_field_translates_exactly() {
        // da/dt = c ⇒ a(1) = a0 + c, for any integrator / step count.
        let c = vec![0.3, -0.7];
        let v = |_a: &[f64], _t: f64| c.clone();
        let a1 = sample_field(&v, &[1.0, 2.0], 4, Integrator::Euler);
        assert!((a1[0] - 1.3).abs() < 1e-12 && (a1[1] - 1.3).abs() < 1e-12, "{a1:?}");
    }

    #[test]
    fn linear_decay_converges_to_the_analytic_flow() {
        // da/dt = −k·a ⇒ a(1) = a0·e^{−k}. Euler → truth as steps↑; Heun is far more accurate.
        let k = 1.5;
        let v = |a: &[f64], _t: f64| a.iter().map(|x| -k * x).collect();
        let a0 = [2.0];
        let truth = 2.0 * (-k).exp();

        let euler_coarse = (sample_field(&v, &a0, 4, Integrator::Euler)[0] - truth).abs();
        let euler_fine = (sample_field(&v, &a0, 256, Integrator::Euler)[0] - truth).abs();
        let heun_coarse = (sample_field(&v, &a0, 4, Integrator::Heun)[0] - truth).abs();

        assert!(euler_fine < euler_coarse, "Euler should improve with more steps");
        assert!(euler_fine < 1e-2, "fine Euler off analytic flow: {euler_fine}");
        assert!(heun_coarse < euler_coarse, "Heun should beat Euler at equal steps: {heun_coarse} vs {euler_coarse}");
    }

    #[test]
    fn mlp_velocity_field_integrates_end_to_end() {
        // A linear MLP realizing v = −k·a (ignoring t and obs): input [a(1); t(1); obs(1)] → v(1).
        let k = 1.0;
        let w = DMatrix::from_row_slice(1, 3, &[-k, 0.0, 0.0]); // pick out the action component only
        let b = DVector::from_row_slice(&[0.0]);
        let mlp = Mlp::new(vec![Layer { w, b, act: Activation::Identity }]);
        let a1 = sample_mlp(&mlp, &[0.42], &[3.0], 200, Integrator::Heun);
        let truth = 3.0 * (-k).exp();
        assert!((a1[0] - truth).abs() < 1e-3, "MLP flow sample {} vs analytic {truth}", a1[0]);
    }
}
