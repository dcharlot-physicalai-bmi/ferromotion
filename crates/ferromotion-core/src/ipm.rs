//! Interior-point contact solver — Dojo's core mechanism. The contact LCP `0 ≤ λ ⟂ (Aλ+b) ≥ 0`
//! (A symmetric PSD) is solved on the **central path** `λ∘(Aλ+b) = κ·1`, `λ > 0`, by damped Newton
//! with fraction-to-boundary steps. Shrinking `κ → 0` recovers the hard LCP; keeping `κ > 0` gives a
//! *smoothed* solution whose gradient is well-defined everywhere — including at the active/inactive
//! boundary where the active-set QP is non-differentiable. Gradients come from the implicit function
//! theorem on the KKT residual. This is what makes contact differentiable for gradient-based control.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// One smoothed (central-path) solve at fixed `kappa`, optionally warm-started. `A` symmetric PSD.
pub fn solve_lcp_smoothed(a: &DMatrix<f64>, b: &[f64], kappa: f64, warm: Option<&DVector<f64>>) -> DVector<f64> {
    let n = b.len();
    let bv = DVector::from_row_slice(b);
    let mut lam = warm.cloned().unwrap_or_else(|| DVector::from_element(n, 1.0));
    for _ in 0..100 {
        let w = a * &lam + &bv;
        let f = lam.component_mul(&w) - DVector::from_element(n, kappa);
        if f.norm() < 1e-13 {
            break;
        }
        // Jacobian ∂F/∂λ = diag(w) + diag(λ)·A.
        let mut j = DMatrix::from_diagonal(&w);
        for i in 0..n {
            for k in 0..n {
                j[(i, k)] += lam[i] * a[(i, k)];
            }
        }
        let step = match j.lu().solve(&f) {
            Some(d) => d,
            None => break,
        };
        // Newton direction is −J⁻¹F; fraction-to-boundary keeps λ > 0.
        let mut alpha: f64 = 1.0;
        for i in 0..n {
            if step[i] > 0.0 {
                alpha = alpha.min(0.99 * lam[i] / step[i]);
            }
        }
        lam -= alpha * step;
    }
    lam
}

/// Hard LCP solution via a κ-continuation (1 → 1e-14), warm-started down the central path.
pub fn solve_lcp(a: &DMatrix<f64>, b: &[f64]) -> DVector<f64> {
    let mut lam = DVector::from_element(b.len(), 1.0);
    let mut kappa = 1.0;
    for _ in 0..14 {
        lam = solve_lcp_smoothed(a, b, kappa, Some(&lam));
        kappa *= 0.1;
    }
    lam
}

/// Smoothed solve at `kappa`, plus the analytic gradient `∂λ/∂b` via the implicit function theorem:
/// `∂λ/∂b = −(diag(w) + diag(λ)·A)⁻¹ · diag(λ)`, where `w = Aλ + b`.
pub fn solve_lcp_diff(a: &DMatrix<f64>, b: &[f64], kappa: f64) -> (DVector<f64>, DMatrix<f64>) {
    let n = b.len();
    let lam = solve_lcp_smoothed(a, b, kappa, None);
    let w = a * &lam + DVector::from_row_slice(b);
    let mut j = DMatrix::from_diagonal(&w);
    for i in 0..n {
        for k in 0..n {
            j[(i, k)] += lam[i] * a[(i, k)];
        }
    }
    let jinv = j.try_inverse().expect("central-path Jacobian invertible");
    let dlam_db = -&jinv * DMatrix::from_diagonal(&lam);
    (lam, dlam_db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_point_recovers_the_lcp_solution() {
        // A PSD, b chosen so the unconstrained optimum has a negative component (constraint active).
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let b = [-1.0, 0.5];
        let lam = solve_lcp(&a, &b);
        // Analytic LCP solution: λ = [0.5, 0] (second contact inactive).
        assert!((lam[0] - 0.5).abs() < 1e-5 && lam[1].abs() < 1e-5, "λ = {lam:?}");
        // Feasibility + complementarity: λ ≥ 0, w ≥ 0, λ∘w ≈ 0.
        let w = &a * &lam + DVector::from_row_slice(&b);
        let lo = |v: &DVector<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(lo(&lam) > -1e-9 && lo(&w) > -1e-6, "infeasible: w = {w:?}");
        assert!(lam.component_mul(&w).norm() < 1e-5, "complementarity violated");
    }

    #[test]
    fn smoothed_gradient_matches_finite_difference() {
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let b = [-1.0, 0.5];
        let kappa = 1e-2;
        let (_lam, dlam_db) = solve_lcp_diff(&a, &b, kappa);
        let eps = 1e-6;
        for jcol in 0..2 {
            let mut bp = b;
            bp[jcol] += eps;
            let fd = (solve_lcp_smoothed(&a, &bp, kappa, None) - solve_lcp_smoothed(&a, &b, kappa, None)) / eps;
            for irow in 0..2 {
                assert!(
                    (dlam_db[(irow, jcol)] - fd[irow]).abs() < 1e-4,
                    "∂λ/∂b[{irow},{jcol}]: analytic {} vs fd {}",
                    dlam_db[(irow, jcol)],
                    fd[irow]
                );
            }
        }
    }

    #[test]
    fn smoothing_stays_differentiable_at_the_contact_boundary() {
        // b2 = 0 puts the second contact exactly at the active/inactive boundary — where the
        // active-set QP gradient is undefined. The κ-smoothed gradient must be finite (no blow-up).
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let b = [-1.0, 0.0];
        let (_lam, dlam_db) = solve_lcp_diff(&a, &b, 1e-2);
        assert!(dlam_db.iter().all(|v| v.is_finite()), "gradient not finite at the boundary");
    }
}
