//! **Optimal transport** — the distance a closed loop actually charges for a policy's error.
//!
//! A generative policy is judged by how well its action distribution matches an expert's, and the
//! natural-looking choice of divergence is the wrong one. Total variation and KL are insensitive to
//! *how far* the mass moved: two distributions can be maximally far apart in total variation while
//! being a hair apart in space. A closed loop does not care about set overlap, it cares about the size
//! of the action error, and the second moment of that error is what shows up in the cost. That is a
//! Wasserstein quantity, not a TV one, which is why a composition result stated in TV cannot control
//! closed-loop regret.
//!
//! This module supplies the three pieces that get used in practice: the closed-form `W₂` between
//! Gaussians (the case a linearised analysis reduces to), `W₁` between empirical samples on the line
//! (exact by sorting), and entropic optimal transport by [`sinkhorn`] for general discrete measures.

use nalgebra::{DMatrix, DVector};

/// **Wasserstein-2 distance between two Gaussians**, in closed form:
///
/// `W₂² = ‖m₁ − m₂‖² + tr(Σ₁ + Σ₂ − 2(Σ₂^{1/2} Σ₁ Σ₂^{1/2})^{1/2})`
///
/// The mean term is the part a bias contributes and the covariance term is the part sampling spread
/// contributes, and they enter separately — which is the formal reason a systematic offset and a noisy
/// sample of the same magnitude are not interchangeable. Covariances must be symmetric positive
/// semi-definite; returns `None` otherwise.
pub fn w2_gaussian(m1: &DVector<f64>, s1: &DMatrix<f64>, m2: &DVector<f64>, s2: &DMatrix<f64>) -> Option<f64> {
    let mean_term = (m1 - m2).norm_squared();
    let root2 = psd_sqrt(s2)?;
    let inner = &root2 * s1 * &root2;
    let cross = psd_sqrt(&inner)?;
    let cov_term = s1.trace() + s2.trace() - 2.0 * cross.trace();
    Some((mean_term + cov_term.max(0.0)).max(0.0).sqrt())
}

/// The symmetric positive semi-definite square root, by eigendecomposition with negative eigenvalues
/// clamped to zero (they only ever arise from round-off on a PSD input).
fn psd_sqrt(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if a.nrows() != a.ncols() {
        return None;
    }
    let sym = (a + a.transpose()) * 0.5;
    let e = sym.symmetric_eigen();
    if e.eigenvalues.iter().any(|&l| l < -1e-8) {
        return None; // genuinely indefinite: not a covariance
    }
    let d = DMatrix::from_diagonal(&e.eigenvalues.map(|l| l.max(0.0).sqrt()));
    Some(&e.eigenvectors * d * e.eigenvectors.transpose())
}

/// **Wasserstein-1 between two equally-weighted empirical samples on the line**, exact. On the real
/// line the optimal coupling is monotone, so sorting both samples and averaging the paired gaps is the
/// answer — no solver needed. Samples must be the same length.
pub fn w1_empirical_1d(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() || !a.iter().chain(b).all(|v| v.is_finite()) {
        return None;
    }
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_by(f64::total_cmp);
    y.sort_by(f64::total_cmp);
    Some(x.iter().zip(&y).map(|(p, q)| (p - q).abs()).sum::<f64>() / x.len() as f64)
}

/// The result of an entropic optimal-transport solve.
#[derive(Clone, Debug)]
pub struct SinkhornPlan {
    /// The transport plan; `plan[(i, j)]` is the mass moved from `i` to `j`.
    pub plan: DMatrix<f64>,
    /// `⟨plan, cost⟩`, the transport cost of the plan.
    pub cost: f64,
    /// Worst violation of the two marginal constraints at the returned plan.
    pub marginal_error: f64,
    pub iters: usize,
}

/// **Entropic optimal transport** between discrete measures `a` and `b` under `cost`, by Sinkhorn
/// iteration on `K = exp(−cost/reg)`.
///
/// The entropic regulariser makes the problem strictly convex and the iteration a pair of
/// alternating rescalings, which is why this is the workhorse: it is a few matrix-vector products per
/// step and it warm-starts. Smaller `reg` approaches the true optimal transport cost and converges more
/// slowly; the returned `cost` is the plan's transport cost, without the entropy term, so it is
/// comparable to an unregularised optimum. Marginals must be positive and sum to the same total.
pub fn sinkhorn(a: &[f64], b: &[f64], cost: &DMatrix<f64>, reg: f64, iters: usize) -> Option<SinkhornPlan> {
    let (n, m) = (a.len(), b.len());
    if cost.nrows() != n || cost.ncols() != m || reg <= 0.0 {
        return None;
    }
    let (sa, sb) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    if sa <= 0.0 || sb <= 0.0 || (sa - sb).abs() > 1e-9 * sa.max(sb) {
        return None; // the measures must have the same total mass
    }
    // K = exp(−C/reg), stabilised by removing the smallest cost so the exponentials stay in range
    let cmin = cost.iter().cloned().fold(f64::INFINITY, f64::min);
    let k = cost.map(|c| ((cmin - c) / reg).exp());

    let mut u = DVector::from_element(n, 1.0);
    let mut v = DVector::from_element(m, 1.0);
    let av = DVector::from_row_slice(a);
    let bv = DVector::from_row_slice(b);
    let mut used = 0;
    for it in 0..iters {
        // v = b / (Kᵀu), u = a / (Kv)
        let ktu = k.transpose() * &u;
        for j in 0..m {
            v[j] = if ktu[j] > 1e-300 { bv[j] / ktu[j] } else { 0.0 };
        }
        let kv = &k * &v;
        for i in 0..n {
            u[i] = if kv[i] > 1e-300 { av[i] / kv[i] } else { 0.0 };
        }
        used = it + 1;
        if !u.iter().chain(v.iter()).all(|x| x.is_finite()) {
            return None;
        }
    }

    let mut plan = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            plan[(i, j)] = u[i] * k[(i, j)] * v[j];
        }
    }
    let row_err = (0..n).fold(0.0f64, |e, i| e.max((plan.row(i).sum() - a[i]).abs()));
    let col_err = (0..m).fold(0.0f64, |e, j| e.max((plan.column(j).sum() - b[j]).abs()));
    let tc = plan.iter().zip(cost.iter()).map(|(p, c)| p * c).sum::<f64>();
    Some(SinkhornPlan { plan, cost: tc, marginal_error: row_err.max(col_err), iters: used })
}

/// Squared-Euclidean cost matrix between two point sets, the usual choice for a `W₂` problem.
pub fn squared_cost(x: &[DVector<f64>], y: &[DVector<f64>]) -> DMatrix<f64> {
    let mut c = DMatrix::zeros(x.len(), y.len());
    for (i, xi) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            c[(i, j)] = (xi - yj).norm_squared();
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }
    fn eye(n: usize, s: f64) -> DMatrix<f64> {
        DMatrix::identity(n, n) * s
    }

    /// Between Gaussians with the same covariance, `W₂` is exactly the distance between the means — the
    /// case that makes a policy's systematic bias directly visible as a distance.
    #[test]
    fn w2_between_shifted_gaussians_is_the_mean_distance() {
        let s = eye(3, 0.7);
        let d = w2_gaussian(&dv(&[0.0, 0.0, 0.0]), &s, &dv(&[3.0, 4.0, 0.0]), &s).unwrap();
        assert!((d - 5.0).abs() < 1e-9, "shifted by 5, W2 = {d}");
        // identical distributions are at distance zero
        let z = w2_gaussian(&dv(&[1.0, 2.0, 3.0]), &s, &dv(&[1.0, 2.0, 3.0]), &s).unwrap();
        assert!(z < 1e-9, "a distribution is zero distance from itself, got {z}");
    }

    /// For scalar Gaussians the closed form reduces to `W₂² = (m₁−m₂)² + (σ₁−σ₂)²`, which is a sharp
    /// check on the matrix-root path.
    #[test]
    fn w2_scalar_case_matches_the_textbook_formula() {
        for &(m1, s1, m2, s2) in &[(0.0, 1.0, 0.0, 2.0), (1.0, 0.5, -1.0, 1.5), (2.0, 3.0, 2.0, 3.0)] {
            let got = w2_gaussian(&dv(&[m1]), &eye(1, s1 * s1), &dv(&[m2]), &eye(1, s2 * s2)).unwrap();
            let want = ((m1 - m2).powi(2) + (s1 - s2).powi(2)).sqrt();
            assert!((got - want).abs() < 1e-9, "W2 {got} vs textbook {want}");
        }
    }

    /// The separation total variation cannot see: two distributions with disjoint support are at TV
    /// distance 1 no matter how close they sit, while `W₂` reports the actual gap and shrinks with it.
    /// This is why a closed-loop bound has to be stated in Wasserstein.
    #[test]
    fn wasserstein_sees_distance_where_total_variation_saturates() {
        let tiny = eye(1, 1e-12); // effectively point masses, so TV between distinct ones is 1
        let mut last = f64::INFINITY;
        for gap in [1.0, 0.1, 0.01, 0.001] {
            let d = w2_gaussian(&dv(&[0.0]), &tiny, &dv(&[gap]), &tiny).unwrap();
            assert!((d - gap).abs() < 1e-6, "W2 should track the gap: {d} vs {gap}");
            assert!(d < last, "W2 must shrink as the gap shrinks");
            last = d;
        }
        eprintln!("W2 tracked the gap down to {last:.4} while TV would have read 1 throughout");
    }

    /// `W₁` on the line is exact by sorting, so it can be checked against a hand-computed value.
    #[test]
    fn w1_on_the_line_is_the_sorted_pairing() {
        // sorted: [1,2,3] vs [2,4,6] → gaps 1,2,3 → mean 2
        let d = w1_empirical_1d(&[3.0, 1.0, 2.0], &[6.0, 2.0, 4.0]).unwrap();
        assert!((d - 2.0).abs() < 1e-12, "W1 = {d}");
        // a sample against itself is zero, and order does not matter
        assert!(w1_empirical_1d(&[5.0, 1.0, 3.0], &[3.0, 5.0, 1.0]).unwrap() < 1e-12);
    }

    /// Sinkhorn must satisfy the marginals and, as the regulariser shrinks, approach the exact optimum.
    /// The exact answer here is available: transporting a point set onto a shift of itself costs the
    /// squared shift, since the identity coupling is optimal.
    #[test]
    fn sinkhorn_meets_the_marginals_and_approaches_the_exact_optimum() {
        let x: Vec<DVector<f64>> = (0..6).map(|i| dv(&[i as f64])).collect();
        let shift = 0.5;
        let y: Vec<DVector<f64>> = (0..6).map(|i| dv(&[i as f64 + shift])).collect();
        let c = squared_cost(&x, &y);
        let w = vec![1.0 / 6.0; 6];
        let exact = shift * shift; // identity coupling, each unit of mass moves `shift`

        let mut prev = f64::INFINITY;
        for &reg in &[0.5_f64, 0.1, 0.02] {
            let p = sinkhorn(&w, &w, &c, reg, 4000).expect("valid problem");
            eprintln!("sinkhorn reg {reg:.2}: cost {:.6} (exact {exact:.6}), marginal error {:.2e}, iters {}", p.cost, p.marginal_error, p.iters);
            // a smaller regulariser buys a tighter cost and pays in marginal convergence
            assert!(p.marginal_error < 1e-2, "marginals badly violated: {}", p.marginal_error);
            assert!(p.cost >= exact - 1e-3, "entropic cost cannot meaningfully beat the exact optimum");
            assert!(p.cost < prev, "shrinking the regulariser should tighten the cost");
            prev = p.cost;
        }
        assert!((prev - exact).abs() < 0.02, "should be close to exact at the smallest reg: {prev} vs {exact}");

        // and the honest characterisation of the trade: more iterations tighten the marginals
        let coarse = sinkhorn(&w, &w, &c, 0.05, 500).unwrap();
        let fine = sinkhorn(&w, &w, &c, 0.05, 8000).unwrap();
        eprintln!("at reg 0.05: marginal error {:.2e} after 500 iters, {:.2e} after 8000", coarse.marginal_error, fine.marginal_error);
        assert!(fine.marginal_error <= coarse.marginal_error, "more iterations must not make the marginals worse");
    }

    /// Mismatched or non-conformable inputs are refused rather than silently producing a number.
    #[test]
    fn malformed_transport_problems_are_refused() {
        let c = DMatrix::zeros(2, 3);
        assert!(sinkhorn(&[0.5, 0.5], &[0.4, 0.3, 0.3], &c, 0.0, 10).is_none(), "reg must be positive");
        assert!(sinkhorn(&[0.5, 0.5], &[0.4, 0.3], &c, 0.1, 10).is_none(), "shape mismatch");
        assert!(sinkhorn(&[0.5, 0.9], &[0.4, 0.3, 0.3], &c, 0.1, 10).is_none(), "unequal total mass");
        assert!(w1_empirical_1d(&[1.0], &[1.0, 2.0]).is_none(), "unequal sample sizes");
    }
}
