//! **Koopman lab** — the rig behind the textbook chapter on making a nonlinear system linear.
//!
//! A linear model is a wonderful thing to have — it predicts forever, and the whole toolbox of linear
//! control applies. Most systems are not linear, so we usually give that up. Koopman's idea is that you
//! often do not have to: a nonlinear system, viewed through the right *observables* of its state, moves
//! *linearly*. Lift the state `x` through a dictionary `ψ(x)`, and for a rich-enough dictionary the
//! dynamics become `ψ(x_{k+1}) = A ψ(x_k)` — an exact linear operator. The nonlinearity was never in the
//! system; it was in looking at only `x` and not, say, `x²`.
//!
//! The rig uses Brunton's slow-manifold system, which has an *exact* three-dimensional Koopman invariant
//! subspace, and fits the operator two ways with the real [`ferromotion_control::edmd`]: once with the
//! lifting observable and once without. With it the linear prediction is exact; without it, it drifts.

use ferromotion_control::{edmd, Koopman};
use nalgebra::{DMatrix, DVector};
use wasm_bindgen::prelude::*;

// Brunton's system: ẋ1 = μ x1, ẋ2 = λ(x2 − x1²). Stable, with a parabolic slow manifold.
const MU: f64 = -0.05;
const LAM: f64 = -1.0;
const DT: f64 = 0.06;

/// Exact closed-form flow of the system over one step `DT`.
fn exact_step(x: [f64; 2]) -> [f64; 2] {
    let x1 = x[0] * (MU * DT).exp();
    let b = LAM * x[0] * x[0] / (LAM - 2.0 * MU);
    let c = x[1] - b;
    let x2 = c * (LAM * DT).exp() + b * (2.0 * MU * DT).exp();
    [x1, x2]
}

fn lift3(x: [f64; 2]) -> DVector<f64> {
    DVector::from_row_slice(&[x[0], x[1], x[0] * x[0]]) // the Koopman dictionary: includes x1²
}
fn lift2(x: [f64; 2]) -> DVector<f64> {
    DVector::from_row_slice(&[x[0], x[1]]) // the naive dictionary: raw state only
}

#[wasm_bindgen]
pub struct KoopmanLab {
    koop: Koopman, // 3-D lifted operator (with x1²)
    naive: DMatrix<f64>, // 2-D operator (raw state only)
}

#[wasm_bindgen]
impl KoopmanLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> KoopmanLab {
        // Snapshot data from assorted initial conditions (deterministic — no RNG).
        let seeds = [[1.4, -1.0], [-0.9, 1.3], [0.5, -0.8], [1.5, 0.6], [-1.2, -0.5], [0.8, 1.1], [-0.4, 0.9]];
        let (mut px3, mut py3, mut px2, mut py2) = (vec![], vec![], vec![], vec![]);
        for s in seeds {
            let mut x = s;
            for _ in 0..24 {
                let xn = exact_step(x);
                px3.push(lift3(x));
                py3.push(lift3(xn));
                px2.push(lift2(x));
                py2.push(lift2(xn));
                x = xn;
            }
        }
        KoopmanLab { koop: Koopman { a: edmd(&px3, &py3), b: None }, naive: edmd(&px2, &py2) }
    }

    /// How far the fitted lifted operator is from the analytic Koopman operator `exp(K·DT)` — the
    /// system has an exact finite invariant subspace, so this should be machine zero.
    pub fn operator_error(&self) -> f64 {
        let k = DMatrix::from_row_slice(3, 3, &[MU, 0.0, 0.0, 0.0, LAM, -LAM, 0.0, 0.0, 2.0 * MU]);
        let a_true = (k * DT).exp();
        (&self.koop.a - &a_true).norm()
    }

    /// The true nonlinear trajectory from `(x1, x2)`, as interleaved `[x1,x2,…]`.
    pub fn true_traj(&self, x1: f64, x2: f64, steps: usize) -> Vec<f64> {
        let mut x = [x1, x2];
        let mut out = vec![x[0], x[1]];
        for _ in 0..steps {
            x = exact_step(x);
            out.push(x[0]);
            out.push(x[1]);
        }
        out
    }

    /// The Koopman (lifted-linear) prediction — roll the linear operator on `ψ = [x1,x2,x1²]`.
    pub fn koopman_traj(&self, x1: f64, x2: f64, steps: usize) -> Vec<f64> {
        let mut psi = lift3([x1, x2]);
        let mut out = vec![psi[0], psi[1]];
        for _ in 0..steps {
            psi = self.koop.predict(&psi);
            out.push(psi[0]);
            out.push(psi[1]);
        }
        out
    }

    /// The naive linear prediction — the best linear model of the raw state `[x1,x2]`, no lifting.
    pub fn naive_traj(&self, x1: f64, x2: f64, steps: usize) -> Vec<f64> {
        let mut p = lift2([x1, x2]);
        let mut out = vec![p[0], p[1]];
        for _ in 0..steps {
            p = &self.naive * p;
            out.push(p[0]);
            out.push(p[1]);
        }
        out
    }

    /// Peak prediction error of a rolled-out trajectory vs the truth (Euclidean, over the run).
    fn peak_err(&self, pred: &[f64], truth: &[f64]) -> f64 {
        let n = pred.len().min(truth.len()) / 2;
        let mut worst = 0.0f64;
        for i in 0..n {
            let e = ((pred[2 * i] - truth[2 * i]).powi(2) + (pred[2 * i + 1] - truth[2 * i + 1]).powi(2)).sqrt();
            worst = worst.max(e);
        }
        worst
    }

    pub fn koopman_peak_error(&self, x1: f64, x2: f64, steps: usize) -> f64 {
        self.peak_err(&self.koopman_traj(x1, x2, steps), &self.true_traj(x1, x2, steps))
    }
    pub fn naive_peak_error(&self, x1: f64, x2: f64, steps: usize) -> f64 {
        self.peak_err(&self.naive_traj(x1, x2, steps), &self.true_traj(x1, x2, steps))
    }

    /// The slow manifold `x2 = b·x1²` (`b = λ/(λ−2μ)`), sampled for drawing.
    pub fn manifold(&self, x1_lo: f64, x1_hi: f64, n: usize) -> Vec<f64> {
        let b = LAM / (LAM - 2.0 * MU);
        (0..n)
            .flat_map(|i| {
                let x1 = x1_lo + (x1_hi - x1_lo) * i as f64 / (n - 1) as f64;
                [x1, b * x1 * x1]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edmd_recovers_the_exact_koopman_operator() {
        let lab = KoopmanLab::new();
        assert!(lab.operator_error() < 1e-9, "operator not recovered exactly: {}", lab.operator_error());
    }

    #[test]
    fn the_lifted_model_predicts_exactly_while_the_naive_one_drifts() {
        // THE CHAPTER. From a fresh start (not in the training set), the Koopman model tracks the true
        // nonlinear trajectory to machine precision; the naive linear model, missing the x1² observable,
        // cannot represent the curvature and drifts away.
        let lab = KoopmanLab::new();
        let (x1, x2, steps) = (1.3, -0.9, 60);
        let kerr = lab.koopman_peak_error(x1, x2, steps);
        let nerr = lab.naive_peak_error(x1, x2, steps);
        assert!(kerr < 1e-7, "Koopman prediction should be exact: peak error {kerr}");
        assert!(nerr > 50.0 * kerr.max(1e-9), "naive linear model should drift far more: {nerr} vs {kerr}");
        assert!(nerr > 0.05, "the naive model should have a visible error to contrast: {nerr}");
    }

    #[test]
    fn the_x1_squared_observable_is_what_makes_it_exact() {
        // Sanity that the contrast is really about the dictionary: the naive model gets the (linear) x1
        // channel right but fails on the (nonlinear) x2 channel.
        let lab = KoopmanLab::new();
        let (x1, x2, steps) = (1.2, 0.8, 50);
        let naive = lab.naive_traj(x1, x2, steps);
        let truth = lab.true_traj(x1, x2, steps);
        let n = naive.len() / 2;
        let (mut worst_x1, mut worst_x2) = (0.0f64, 0.0f64);
        for i in 0..n {
            worst_x1 = worst_x1.max((naive[2 * i] - truth[2 * i]).abs());
            worst_x2 = worst_x2.max((naive[2 * i + 1] - truth[2 * i + 1]).abs());
        }
        assert!(worst_x1 < 1e-6, "the x1 channel is linear — naive should nail it: {worst_x1}");
        assert!(worst_x2 > 0.05, "the x2 channel is where the missing x1² bites: {worst_x2}");
    }
}
