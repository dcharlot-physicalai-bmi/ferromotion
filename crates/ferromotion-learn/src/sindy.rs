//! **SINDy — Sparse Identification of Nonlinear Dynamics** — discover the *equation itself* from data. Given
//! measured states `x(t)` and their derivatives `ẋ(t)`, build a library `Θ(x)` of candidate terms (constants,
//! monomials, …) and solve `ẋ = Θ(x) Ξ` for a **sparse** coefficient matrix `Ξ`: the few nonzero entries name
//! the terms actually present in the dynamics. This is a different flavor of physics-informed learning — not a
//! network at all, just linear algebra with a sparsity prior — and it recovers an *interpretable* model you
//! can read as an equation, rather than a black box.
//!
//! Sparsity comes from **sequentially thresholded least squares (STLSQ)**: fit by least squares, zero every
//! coefficient below a threshold `λ`, refit on the survivors, repeat. `λ` is the knob between an overfit dense
//! model (too small — keeps spurious terms) and an oversparse one (too large — drops real terms). Pure
//! `nalgebra`, wasm-clean. Verified on the unforced **Duffing oscillator** `ẋ = y`, `ẏ = −x − 0.3x³ − 0.1y`:
//! from a degree-3 library of ten candidate terms, SINDy recovers exactly those coefficients and zeros the
//! rest.

use nalgebra::{DMatrix, DVector};

/// A discovered model: the library term names and the sparse coefficient matrix `Ξ` (one column per state
/// dimension).
pub struct Sindy {
    pub names: Vec<String>,
    /// `coeffs[(term, dim)]` — coefficient of library `term` in the equation for state component `dim`.
    pub coeffs: DMatrix<f64>,
}

/// All monomial exponent tuples for `n_vars` variables up to total degree `degree` (including the constant).
pub fn monomial_exponents(n_vars: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    // recursive enumeration of exponent vectors with sum ≤ degree
    fn rec(pos: usize, n: usize, left: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if pos == n {
            out.push(cur.clone());
            return;
        }
        for e in 0..=left {
            cur.push(e);
            rec(pos + 1, n, left - e, cur, out);
            cur.pop();
        }
    }
    rec(0, n_vars, degree, &mut Vec::new(), &mut out);
    // sort by total degree then lexicographically for readable, stable ordering
    out.sort_by_key(|v| (v.iter().sum::<usize>(), v.clone()));
    out
}

fn term_name(exps: &[usize]) -> String {
    let vars = ["x", "y", "z", "w"];
    let parts: Vec<String> = exps
        .iter()
        .enumerate()
        .filter(|(_, e)| **e > 0)
        .map(|(i, &e)| if e == 1 { vars[i].to_string() } else { format!("{}^{}", vars[i], e) })
        .collect();
    if parts.is_empty() { "1".to_string() } else { parts.join(" ") }
}

fn eval_term(exps: &[usize], state: &[f64]) -> f64 {
    exps.iter().zip(state.iter()).map(|(&e, &v)| v.powi(e as i32)).product()
}

/// Least-squares solve `A ξ = b` via SVD (robust to rank deficiency).
fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    ferromotion_core::finite_svd(a, true, true).and_then(|s| s.solve(b, 1e-12).ok()).unwrap_or_else(|| DVector::zeros(a.ncols()))
}

/// Sequentially thresholded least squares: sparse solution to `Θ ξ = d`.
fn stlsq(theta: &DMatrix<f64>, d: &DVector<f64>, lambda: f64, iters: usize) -> DVector<f64> {
    let n = theta.ncols();
    let mut xi = lstsq(theta, d);
    for _ in 0..iters {
        let active: Vec<usize> = (0..n).filter(|&j| xi[j].abs() >= lambda).collect();
        // zero out the small ones
        for j in 0..n {
            if xi[j].abs() < lambda {
                xi[j] = 0.0;
            }
        }
        if active.is_empty() {
            break;
        }
        // refit on the active columns only
        let sub = DMatrix::from_columns(&active.iter().map(|&j| theta.column(j).into_owned()).collect::<Vec<_>>());
        let sol = lstsq(&sub, d);
        for (k, &j) in active.iter().enumerate() {
            xi[j] = sol[k];
        }
    }
    xi
}

impl Sindy {
    /// Fit a sparse dynamics model. `states[i]` is the state at sample `i`; `derivs[i]` its time derivative.
    /// `degree` bounds the polynomial library; `lambda` is the sparsity threshold.
    pub fn fit(states: &[Vec<f64>], derivs: &[Vec<f64>], degree: usize, lambda: f64) -> Sindy {
        let n_vars = states[0].len();
        let exps = monomial_exponents(n_vars, degree);
        let n_feat = exps.len();
        let n_samp = states.len();

        // library matrix Θ (n_samp × n_feat)
        let mut theta = DMatrix::zeros(n_samp, n_feat);
        for (i, s) in states.iter().enumerate() {
            for (j, e) in exps.iter().enumerate() {
                theta[(i, j)] = eval_term(e, s);
            }
        }

        // solve for each state-derivative dimension
        let mut coeffs = DMatrix::zeros(n_feat, n_vars);
        for dim in 0..n_vars {
            let d = DVector::from_iterator(n_samp, derivs.iter().map(|dv| dv[dim]));
            let xi = stlsq(&theta, &d, lambda, 12);
            for j in 0..n_feat {
                coeffs[(j, dim)] = xi[j];
            }
        }

        Sindy { names: exps.iter().map(|e| term_name(e)).collect(), coeffs }
    }

    /// Number of nonzero terms across all equations.
    pub fn n_active(&self) -> usize {
        self.coeffs.iter().filter(|&&c| c != 0.0).count()
    }

    /// Human-readable equation for state dimension `dim`, e.g. `"ẋ = 1.00 y"`.
    pub fn equation(&self, dim: usize) -> String {
        let lhs = ["ẋ", "ẏ", "ż", "ẇ"][dim.min(3)];
        let terms: Vec<String> = (0..self.names.len())
            .filter(|&j| self.coeffs[(j, dim)] != 0.0)
            .map(|j| {
                let c = self.coeffs[(j, dim)];
                if self.names[j] == "1" { format!("{c:.3}") } else { format!("{c:.3} {}", self.names[j]) }
            })
            .collect();
        format!("{lhs} = {}", if terms.is_empty() { "0".to_string() } else { terms.join(" + ") })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unforced damped Duffing oscillator: ẋ = y, ẏ = −x − 0.3 x³ − 0.1 y.
    fn duffing(x: f64, y: f64) -> (f64, f64) {
        (y, -x - 0.3 * x * x * x - 0.1 * y)
    }

    fn trajectory(x0: f64, y0: f64, n: usize, dt: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let (mut x, mut y) = (x0, y0);
        let (mut states, mut derivs) = (vec![], vec![]);
        for _ in 0..n {
            let (dx, dy) = duffing(x, y);
            states.push(vec![x, y]);
            derivs.push(vec![dx, dy]); // exact derivatives (what generated the trajectory)
            // RK4 advance
            let (k1x, k1y) = duffing(x, y);
            let (k2x, k2y) = duffing(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
            let (k3x, k3y) = duffing(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
            let (k4x, k4y) = duffing(x + dt * k3x, y + dt * k3y);
            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        }
        (states, derivs)
    }

    #[test]
    fn sindy_discovers_the_duffing_oscillator() {
        // THE HEADLINE. From a degree-3 library of 10 candidate terms, SINDy recovers exactly ẋ = y and
        // ẏ = −x − 0.3x³ − 0.1y, with every spurious coefficient zero.
        let (s1, d1) = trajectory(1.5, 0.0, 1200, 0.01);
        let (s2, d2) = trajectory(0.4, 1.2, 1200, 0.01); // a second orbit enriches the data
        let states: Vec<Vec<f64>> = s1.into_iter().chain(s2).collect();
        let derivs: Vec<Vec<f64>> = d1.into_iter().chain(d2).collect();

        let sindy = Sindy::fit(&states, &derivs, 3, 0.05);
        let idx = |name: &str| sindy.names.iter().position(|n| n == name).unwrap();

        // ẋ = y exactly
        assert!((sindy.coeffs[(idx("y"), 0)] - 1.0).abs() < 1e-2, "ẋ should be y: {}", sindy.equation(0));
        // ẏ = −x − 0.3 x³ − 0.1 y
        assert!((sindy.coeffs[(idx("x"), 1)] - (-1.0)).abs() < 2e-2, "x term: {}", sindy.equation(1));
        assert!((sindy.coeffs[(idx("y"), 1)] - (-0.1)).abs() < 2e-2, "y term: {}", sindy.equation(1));
        assert!((sindy.coeffs[(idx("x^3"), 1)] - (-0.3)).abs() < 2e-2, "x³ term: {}", sindy.equation(1));
        // and it is sparse: exactly 4 active terms total (1 + 3)
        assert_eq!(sindy.n_active(), 4, "should recover exactly 4 terms, got {}: {} ; {}", sindy.n_active(), sindy.equation(0), sindy.equation(1));
    }

    #[test]
    fn too_large_a_threshold_oversparsifies() {
        // The sparsity knob: a huge λ kills real terms too (here the small −0.1y and −0.3x³ vanish).
        let (states, derivs) = trajectory(1.5, 0.0, 1500, 0.01);
        let sindy = Sindy::fit(&states, &derivs, 3, 0.5);
        assert!(sindy.n_active() < 4, "an over-large threshold should drop real terms");
    }

    #[test]
    fn monomial_library_has_the_expected_terms() {
        let e = monomial_exponents(2, 3);
        assert_eq!(e.len(), 10, "2 vars, degree 3 → 10 terms");
        let names: Vec<String> = e.iter().map(|x| term_name(x)).collect();
        assert!(names.contains(&"1".to_string()) && names.contains(&"x^3".to_string()) && names.contains(&"x y".to_string()));
    }
}
