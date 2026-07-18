//! **Graduated Non-Convexity (GNC)** for outlier-robust estimation (Yang, Antonante, Tzoumas & Carlone,
//! RA-L 2020; Black–Rangarajan duality). Plain least-squares trusts every measurement, so a single gross
//! outlier — a bad loop closure, a mis-association — corrupts the whole solve. GNC makes the estimate
//! **robust without an initial guess**: it optimizes a *truncated-least-squares* cost by starting from a
//! near-convex surrogate and gradually sharpening it (the "graduated" schedule), re-deriving a per-
//! measurement weight `wᵢ ∈ [0,1]` in closed form at each step. Inliers converge to weight 1, outliers to
//! 0 — routinely rejecting 70–80% outliers.
//!
//! This is the robust wrapper the least-squares / factor-graph layer wants (it drops onto
//! [`crate::solve_factor_graph`] as a per-factor weight). Here it wraps a linear model `rᵢ = aᵢᵀx − bᵢ`
//! (so robust averaging, line fitting, and translation averaging are all special cases), solved by
//! iteratively-reweighted least squares. Verified: it recovers the clean, outlier-free solution from data
//! that is 40% gross outliers, while ordinary least squares is corrupted. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// The result of a GNC solve.
#[derive(Clone, Debug)]
pub struct GncResult {
    pub x: DVector<f64>,
    /// Final per-measurement weight (≈1 inlier, ≈0 outlier).
    pub weights: Vec<f64>,
    pub iterations: usize,
}

/// GNC-TLS closed-form weight for squared residual `r2` at control `mu` and truncation `cbar2 = c̄²`.
fn gnc_weight(r2: f64, mu: f64, cbar2: f64) -> f64 {
    let lo = mu / (mu + 1.0) * cbar2;
    let hi = (mu + 1.0) / mu * cbar2;
    if r2 <= lo {
        1.0
    } else if r2 >= hi {
        0.0
    } else {
        (cbar2 * mu * (mu + 1.0) / r2).sqrt() - mu
    }
}

/// Weighted linear least squares `min Σ wᵢ (aᵢᵀx − bᵢ)²` → `x = (AᵀWA)⁻¹ AᵀWb`.
fn weighted_ls(a: &DMatrix<f64>, b: &DVector<f64>, w: &[f64]) -> DVector<f64> {
    let wd = DVector::from_row_slice(w);
    let aw = DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)] * wd[i]);
    let ata = aw.transpose() * a;
    let atb = aw.transpose() * b;
    ata.lu().solve(&atb).expect("weighted normal equations singular")
}

/// Robustly solve the linear model `A x ≈ b` (row `i` is measurement `aᵢᵀx = bᵢ`) with GNC-TLS. `c_bar` is
/// the maximum residual an inlier may have (the truncation threshold). Returns the estimate and the
/// converged inlier/outlier weights.
pub fn gnc_solve(a: &DMatrix<f64>, b: &DVector<f64>, c_bar: f64, max_iter: usize) -> GncResult {
    let m = a.nrows();
    let cbar2 = c_bar * c_bar;
    // start from the plain least-squares fit (all weights 1)
    let mut x = weighted_ls(a, b, &vec![1.0; m]);
    let r = a * &x - b;
    let rmax2 = r.iter().map(|ri| ri * ri).fold(0.0, f64::max);
    // convex-surrogate initialization of the control parameter (Yang et al. 2020)
    let mut mu = (cbar2 / (2.0 * rmax2 - cbar2)).max(1e-4);
    let mut weights = vec![1.0; m];
    let mut it = 0;
    for _ in 0..max_iter {
        it += 1;
        // update weights from the current residuals
        let r = a * &x - b;
        for (i, wi) in weights.iter_mut().enumerate() {
            *wi = gnc_weight(r[i] * r[i], mu, cbar2);
        }
        // re-solve, then sharpen the surrogate
        x = weighted_ls(a, b, &weights);
        mu *= 1.4;
        // converged once the surrogate is essentially the hard truncation
        if mu > 1e6 {
            break;
        }
    }
    // final hard weights at the truncation
    let r = a * &x - b;
    for (i, wi) in weights.iter_mut().enumerate() {
        *wi = if r[i] * r[i] <= cbar2 { 1.0 } else { 0.0 };
    }
    GncResult { x, weights, iterations: it }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a 2-parameter line-fit system y = s·x + b, sampling `n_in` inliers on the true line (+small
    // noise) and `n_out` gross outliers far off it. Returns (A, b_vec, inlier_flags).
    fn line_data(slope: f64, intercept: f64, n_in: usize, n_out: usize) -> (DMatrix<f64>, DVector<f64>, Vec<bool>) {
        let m = n_in + n_out;
        let mut a = DMatrix::zeros(m, 2);
        let mut bv = DVector::zeros(m);
        let mut flag = vec![false; m];
        // deterministic pseudo-noise
        let noise = |k: usize| ((k as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5;
        for k in 0..n_in {
            let x = -2.0 + 4.0 * k as f64 / n_in as f64;
            a[(k, 0)] = x;
            a[(k, 1)] = 1.0;
            bv[k] = slope * x + intercept + 0.02 * noise(k);
            flag[k] = true;
        }
        for k in 0..n_out {
            let idx = n_in + k;
            let x = -2.0 + 4.0 * k as f64 / n_out as f64;
            a[(idx, 0)] = x;
            a[(idx, 1)] = 1.0;
            bv[idx] = slope * x + intercept + 5.0 + 3.0 * noise(idx + 999); // way off the line
        }
        (a, bv, flag)
    }

    #[test]
    fn gnc_recovers_the_line_through_forty_percent_outliers() {
        // THE HEADLINE. 20 inliers + 13 gross outliers (≈40%); GNC must recover the true (slope, intercept)
        // while ordinary least squares is dragged off by the outliers.
        let (a, b, _flag) = line_data(1.5, -0.4, 20, 13);
        let res = gnc_solve(&a, &b, 0.2, 60);
        assert!((res.x[0] - 1.5).abs() < 0.05, "slope {} vs 1.5", res.x[0]);
        assert!((res.x[1] + 0.4).abs() < 0.05, "intercept {} vs -0.4", res.x[1]);
        // ordinary least squares is corrupted
        let ls = weighted_ls(&a, &b, &vec![1.0; a.nrows()]);
        assert!((ls[0] - 1.5).abs() > 0.15 || (ls[1] + 0.4).abs() > 0.3, "plain LS should be corrupted: {ls:?}");
    }

    #[test]
    fn inliers_keep_weight_one_and_outliers_are_rejected() {
        let (a, b, flag) = line_data(1.5, -0.4, 20, 13);
        let res = gnc_solve(&a, &b, 0.2, 60);
        for (i, &is_in) in flag.iter().enumerate() {
            if is_in {
                assert!(res.weights[i] > 0.5, "inlier {i} was rejected (w={})", res.weights[i]);
            } else {
                assert!(res.weights[i] < 0.5, "outlier {i} was kept (w={})", res.weights[i]);
            }
        }
    }

    #[test]
    fn with_no_outliers_gnc_equals_least_squares() {
        let (a, b, _) = line_data(0.8, 1.2, 30, 0);
        let res = gnc_solve(&a, &b, 0.2, 60);
        let ls = weighted_ls(&a, &b, &vec![1.0; a.nrows()]);
        assert!((&res.x - &ls).norm() < 1e-3, "with no outliers GNC should match LS: {:?} vs {:?}", res.x, ls);
        assert!(res.weights.iter().all(|&w| w > 0.5), "all measurements should be inliers");
    }
}
