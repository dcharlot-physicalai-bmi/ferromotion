//! **Koopman operator / EDMD** — data-driven linearization of nonlinear dynamics. The Koopman
//! operator advances *observables* of the state linearly; if we lift the state through a dictionary
//! `ψ(x)`, a rich-enough dictionary makes the dynamics (approximately) linear in the lifted space,
//! `ψ(x_{k+1}) ≈ A ψ(x_k)` — after which the whole linear toolbox (prediction, [`crate::dlqr`]) applies
//! to a nonlinear system. **Extended DMD** fits `A` (and, with inputs, `B`) by least squares over
//! data. This is the data-driven complement to our model-based controllers.
//!
//! Verified to machine precision on Brunton's canonical example, which has an *exact* finite Koopman
//! invariant subspace, so EDMD recovers the true linear operator exactly. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// Stack a set of lifted column vectors into an `N × m` matrix.
fn stack(cols: &[DVector<f64>]) -> DMatrix<f64> {
    let (nn, m) = (cols[0].len(), cols.len());
    let mut a = DMatrix::zeros(nn, m);
    for (j, c) in cols.iter().enumerate() {
        a.set_column(j, c);
    }
    a
}

/// **EDMD**: fit the Koopman operator `A` (`N×N`) from lifted snapshot pairs so that
/// `ψ(x_{k+1}) ≈ A ψ(x_k)`. `A = Ψ_y Ψ_xᵀ (Ψ_x Ψ_xᵀ)⁺`.
pub fn edmd(psi_x: &[DVector<f64>], psi_y: &[DVector<f64>]) -> DMatrix<f64> {
    let (px, py) = (stack(psi_x), stack(psi_y));
    let g = &px * px.transpose(); // N×N
    &py * px.transpose() * g.pseudo_inverse(1e-12).expect("EDMD Gram pseudo-inverse")
}

/// **EDMD with control**: fit `(A, B)` so that `ψ(x_{k+1}) ≈ A ψ(x_k) + B u_k`, by regressing the
/// lifted next-state on the stacked `[ψ(x_k); u_k]`.
pub fn edmdc(psi_x: &[DVector<f64>], u: &[DVector<f64>], psi_y: &[DVector<f64>]) -> (DMatrix<f64>, DMatrix<f64>) {
    let (nn, nu, m) = (psi_x[0].len(), u[0].len(), psi_x.len());
    let mut z = DMatrix::zeros(nn + nu, m); // stacked regressors [ψ; u]
    for j in 0..m {
        z.view_mut((0, j), (nn, 1)).copy_from(&psi_x[j]);
        z.view_mut((nn, j), (nu, 1)).copy_from(&u[j]);
    }
    let py = stack(psi_y);
    let g = &z * z.transpose();
    let k = &py * z.transpose() * g.pseudo_inverse(1e-12).expect("EDMDc pseudo-inverse"); // N×(N+nu)
    (k.view((0, 0), (nn, nn)).into_owned(), k.view((0, nn), (nn, nu)).into_owned())
}

/// A fitted Koopman model: the lifted linear operator (and optional input matrix).
#[derive(Clone, Debug)]
pub struct Koopman {
    pub a: DMatrix<f64>,
    pub b: Option<DMatrix<f64>>,
}

impl Koopman {
    /// Advance the lifted state one step.
    pub fn predict(&self, psi: &DVector<f64>) -> DVector<f64> {
        &self.a * psi
    }

    /// Advance the lifted state one step under input `u`.
    pub fn predict_u(&self, psi: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        &self.a * psi + self.b.as_ref().expect("no input matrix") * u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Brunton's slow-manifold system: ẋ1 = μx1, ẋ2 = λ(x2 − x1²). With observables [x1, x2, x1²] the
    // dynamics are EXACTLY linear: d/dt ψ = K ψ, K = [[μ,0,0],[0,λ,−λ],[0,0,2μ]].
    const MU: f64 = -0.05;
    const LAM: f64 = -1.0;

    fn lift(x: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(&[x[0], x[1], x[0] * x[0]])
    }

    // Exact closed-form flow of the system over time `dt`.
    fn exact_step(x: &[f64], dt: f64) -> [f64; 2] {
        let x1 = x[0] * (MU * dt).exp();
        let b = LAM * x[0] * x[0] / (LAM - 2.0 * MU); // particular coefficient
        let c = x[1] - b;
        let x2 = c * (LAM * dt).exp() + b * (2.0 * MU * dt).exp();
        [x1, x2]
    }

    #[test]
    fn edmd_recovers_the_exact_koopman_operator() {
        let dt = 0.05;
        // Snapshot data from assorted initial conditions.
        let seeds = [[1.0, 0.5], [-0.7, 1.2], [0.3, -0.9], [1.5, 0.1], [-1.1, -0.4], [0.6, 0.8], [-0.2, 1.0]];
        let (mut px, mut py) = (Vec::new(), Vec::new());
        for s in seeds {
            let mut x = s;
            for _ in 0..20 {
                let xn = exact_step(&x, dt);
                px.push(lift(&x));
                py.push(lift(&xn));
                x = xn;
            }
        }
        let a = edmd(&px, &py);

        // Analytic discrete operator exp(K·dt).
        let k = DMatrix::from_row_slice(3, 3, &[MU, 0.0, 0.0, 0.0, LAM, -LAM, 0.0, 0.0, 2.0 * MU]);
        let a_true = (k * dt).exp();
        assert!((&a - &a_true).norm() < 1e-9, "EDMD operator off exact Koopman: {}", (&a - &a_true).norm());
    }

    #[test]
    fn koopman_prediction_matches_the_true_trajectory() {
        let dt = 0.05;
        let seeds = [[1.0, 0.5], [-0.7, 1.2], [0.3, -0.9], [1.5, 0.1], [-1.1, -0.4]];
        let (mut px, mut py) = (Vec::new(), Vec::new());
        for s in seeds {
            let mut x = s;
            for _ in 0..25 {
                let xn = exact_step(&x, dt);
                px.push(lift(&x));
                py.push(lift(&xn));
                x = xn;
            }
        }
        let model = Koopman { a: edmd(&px, &py), b: None };

        // Roll the lifted linear model forward from a fresh start and compare to the true flow.
        let mut x = [0.9, -0.6];
        let mut psi = lift(&x);
        for _ in 0..40 {
            psi = model.predict(&psi);
            x = exact_step(&x, dt);
            assert!((psi[0] - x[0]).abs() < 1e-8 && (psi[1] - x[1]).abs() < 1e-8, "prediction drift: {psi:?} vs {x:?}");
        }
    }

    #[test]
    fn edmdc_fits_a_controlled_lifted_model() {
        // A controlled variant: ẋ1 = μx1 + u, ẋ2 = λ(x2 − x1²). Fit (A,B) and check one-step prediction.
        let dt = 0.02;
        let mut rng: u64 = 0x1234_5678;
        let mut rand = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f64) / (1u64 << 31) as f64 - 1.0
        };
        let ctrl_step = |x: &[f64], u: f64| -> [f64; 2] {
            // Semi-implicit-ish explicit Euler on the controlled system.
            let x1 = x[0] + dt * (MU * x[0] + u);
            let x2 = x[1] + dt * (LAM * (x[1] - x[0] * x[0]));
            [x1, x2]
        };
        let (mut px, mut us, mut py) = (Vec::new(), Vec::new(), Vec::new());
        let mut x = [0.5, 0.2];
        for _ in 0..400 {
            let u = rand();
            let xn = ctrl_step(&x, u);
            px.push(lift(&x));
            us.push(DVector::from_row_slice(&[u]));
            py.push(lift(&xn));
            x = xn;
        }
        let (a, b) = edmdc(&px, &us, &py);
        let model = Koopman { a, b: Some(b) };
        // Predict on a held-out step.
        let x0 = [0.3, -0.4];
        let u0 = 0.25;
        let pred = model.predict_u(&lift(&x0), &DVector::from_row_slice(&[u0]));
        let truth = ctrl_step(&x0, u0);
        assert!((pred[0] - truth[0]).abs() < 1e-6 && (pred[1] - truth[1]).abs() < 1e-6, "EDMDc prediction off: {pred:?} vs {truth:?}");
    }
}
