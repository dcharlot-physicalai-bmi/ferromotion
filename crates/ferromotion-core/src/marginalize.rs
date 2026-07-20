//! **Gaussian marginalization (Schur complement) & fixed-lag smoothing** — the operation that lets a
//! windowed estimator run in *constant time*. A linearized factor graph is a Gaussian in **information
//! form**, `Λx = η` (mean `Λ⁻¹η`, covariance `Λ⁻¹`). To bound the window you cannot just *delete* old states
//! — that throws away their information; instead you **marginalize** them, collapsing their influence into a
//! dense **prior** on the states they touched via the Schur complement `Λ′ = Λ_kk − Λ_km Λ_mm⁻¹ Λ_mk`,
//! `η′ = η_k − Λ_km Λ_mm⁻¹ η_m`. This is the backbone of every real-time sliding-window VIO/LIO backend
//! (OpenVINS, OKVIS, VINS-Fusion) and the principled complement to the crate's batch [`crate::solve_factor_graph`]
//! and incremental [`crate::IncrementalLeastSquares`] (iSAM).
//!
//! Verified: the marginal's mean and covariance exactly equal the kept sub-blocks of the full joint solve,
//! and a fixed-lag smoother (marginalizing the oldest state each step) reproduces the full-batch MAP on a
//! linear-Gaussian chain to machine precision. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A Gaussian in information form: `Λ x = η`, so the mean is `Λ⁻¹η` and the covariance is `Λ⁻¹`.
#[derive(Clone, Debug)]
pub struct GaussianInfo {
    pub lambda: DMatrix<f64>,
    pub eta: DVector<f64>,
}

impl GaussianInfo {
    /// A zero (uninformative) Gaussian over `n` scalar variables.
    pub fn zeros(n: usize) -> Self {
        GaussianInfo { lambda: DMatrix::zeros(n, n), eta: DVector::zeros(n) }
    }

    /// The mean `Λ⁻¹ η` (the MAP estimate).
    pub fn mean(&self) -> DVector<f64> {
        self.lambda.clone().lu().solve(&self.eta).expect("information matrix must be invertible")
    }

    /// The covariance `Λ⁻¹`.
    pub fn covariance(&self) -> DMatrix<f64> {
        self.lambda.clone().try_inverse().expect("information matrix must be invertible")
    }

    /// Marginalize out every variable **not** in `keep`, returning the Gaussian over the kept variables (in
    /// the order given by `keep`). Uses the Schur complement — the marginal information, not a deletion.
    pub fn marginalize(&self, keep: &[usize]) -> GaussianInfo {
        let n = self.eta.len();
        let marg: Vec<usize> = (0..n).filter(|i| !keep.contains(i)).collect();
        if marg.is_empty() {
            return GaussianInfo { lambda: select(&self.lambda, keep, keep), eta: select_vec(&self.eta, keep) };
        }
        let l_kk = select(&self.lambda, keep, keep);
        let l_km = select(&self.lambda, keep, &marg);
        let l_mk = select(&self.lambda, &marg, keep);
        let l_mm = select(&self.lambda, &marg, &marg);
        let e_k = select_vec(&self.eta, keep);
        let e_m = select_vec(&self.eta, &marg);
        let l_mm_inv = l_mm.try_inverse().expect("marginalized block must be invertible");
        let k = &l_km * &l_mm_inv;
        GaussianInfo { lambda: &l_kk - &k * &l_mk, eta: &e_k - &k * &e_m }
    }
}

fn select(m: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(rows.len(), cols.len(), |i, j| m[(rows[i], cols[j])])
}
fn select_vec(v: &DVector<f64>, idx: &[usize]) -> DVector<f64> {
    DVector::from_fn(idx.len(), |i, _| v[idx[i]])
}

/// Add a linear Gaussian factor `½ w (aᵀx − b)²` to an information system (scatter `w·aaᵀ` into `Λ` and
/// `w·a·b` into `η`), where `a` is nonzero only on the variable indices `vars`.
pub fn add_factor(info: &mut GaussianInfo, vars: &[usize], a: &[f64], b: f64, w: f64) {
    for (p, &vi) in vars.iter().enumerate() {
        info.eta[vi] += w * a[p] * b;
        for (q, &vj) in vars.iter().enumerate() {
            info.lambda[(vi, vj)] += w * a[p] * a[q];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a small dense linear-Gaussian system over 4 scalar variables
    fn system() -> GaussianInfo {
        let mut g = GaussianInfo::zeros(4);
        add_factor(&mut g, &[0], &[1.0], 1.0, 4.0); // prior x0 ≈ 1
        add_factor(&mut g, &[0, 1], &[-1.0, 1.0], 0.5, 2.0); // x1 − x0 ≈ 0.5
        add_factor(&mut g, &[1, 2], &[-1.0, 1.0], -0.3, 2.0); // x2 − x1 ≈ −0.3
        add_factor(&mut g, &[2, 3], &[-1.0, 1.0], 0.8, 2.0); // x3 − x2 ≈ 0.8
        add_factor(&mut g, &[3], &[1.0], 2.0, 1.0); // soft prior x3 ≈ 2
        g
    }

    #[test]
    fn the_marginal_matches_the_kept_blocks_of_the_full_solve() {
        // THE ORACLE. Marginalizing x1,x2 must leave a Gaussian on {x0,x3} whose mean and covariance equal
        // the corresponding entries of the full joint solve.
        let g = system();
        let full_mean = g.mean();
        let full_cov = g.covariance();
        let keep = [0usize, 3];
        let m = g.marginalize(&keep);
        let mm = m.mean();
        assert!((mm[0] - full_mean[0]).abs() < 1e-10 && (mm[1] - full_mean[3]).abs() < 1e-10, "marginal mean {mm} vs full {full_mean}");
        let mc = m.covariance();
        assert!((mc[(0, 0)] - full_cov[(0, 0)]).abs() < 1e-10, "var(x0) {} vs {}", mc[(0, 0)], full_cov[(0, 0)]);
        assert!((mc[(1, 1)] - full_cov[(3, 3)]).abs() < 1e-10, "var(x3)");
        assert!((mc[(0, 1)] - full_cov[(0, 3)]).abs() < 1e-10, "cov(x0,x3)");
    }

    #[test]
    fn marginalizing_all_but_one_gives_its_scalar_marginal() {
        let g = system();
        let full_cov = g.covariance();
        let m = g.marginalize(&[2]);
        assert!((m.covariance()[(0, 0)] - full_cov[(2, 2)]).abs() < 1e-10, "scalar marginal variance");
        assert!((m.mean()[0] - g.mean()[2]).abs() < 1e-10, "scalar marginal mean");
    }

    #[test]
    fn fixed_lag_smoothing_equals_the_full_batch_map() {
        // THE HEADLINE. A linear-Gaussian odometry chain of 8 states. A fixed-lag smoother that keeps a
        // window of 3 and marginalizes the oldest each step must estimate the newest states exactly as the
        // full-batch solve over all 8.
        let n = 8;
        let odom = [0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1];
        let w = 2.0;
        // full batch over all n states
        let mut full = GaussianInfo::zeros(n);
        add_factor(&mut full, &[0], &[1.0], 0.0, 5.0); // anchor x0 ≈ 0
        for (i, &o) in odom.iter().enumerate() {
            add_factor(&mut full, &[i, i + 1], &[-1.0, 1.0], o, w);
        }
        let full_mean = full.mean();

        // fixed-lag: window info over the *current* window variables, marginalizing the oldest when full
        let win = 3usize;
        // window holds global states [lo, hi]; local index = global − lo
        let mut lo = 0usize;
        let mut info = GaussianInfo::zeros(1);
        add_factor(&mut info, &[0], &[1.0], 0.0, 5.0); // anchor
        for (i, &o) in odom.iter().enumerate() {
            // grow the window by one state (append a column/row)
            let size = info.eta.len();
            let mut bigger = GaussianInfo::zeros(size + 1);
            bigger.lambda.view_mut((0, 0), (size, size)).copy_from(&info.lambda);
            bigger.eta.rows_mut(0, size).copy_from(&info.eta);
            info = bigger;
            // odometry factor between local (i−lo) and (i+1−lo)
            add_factor(&mut info, &[i - lo, i + 1 - lo], &[-1.0, 1.0], o, w);
            // marginalize the oldest while the window is too big
            while info.eta.len() > win {
                let keep: Vec<usize> = (1..info.eta.len()).collect();
                info = info.marginalize(&keep);
                lo += 1;
            }
        }
        // the window now covers global states [lo .. n-1]; compare to the batch
        let wmean = info.mean();
        for (local, global) in (lo..n).enumerate() {
            assert!((wmean[local] - full_mean[global]).abs() < 1e-8, "state {global}: fixed-lag {} vs batch {}", wmean[local], full_mean[global]);
        }
    }
}
