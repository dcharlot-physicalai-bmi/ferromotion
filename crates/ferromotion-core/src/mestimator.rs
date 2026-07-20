//! **Robust M-estimator kernels** — the per-residual robust losses that make least-squares survive
//! outliers, the everyday tool in Ceres/GTSAM/g2o. A kernel replaces the quadratic `½r²` with a loss `ρ(r)`
//! that grows sub-quadratically (or saturates) for large residuals, so a few gross errors (a bad
//! correspondence, a wrong loop closure) can't dominate the fit. Each kernel exposes `ρ` (loss), `ψ = ρ′`
//! (**influence**), and `w = ψ/r` (the **IRLS weight**): plugging `w` as a per-residual multiplier into a
//! Gauss–Newton/LM step (like [`crate::solve_factor_graph`]) performs iteratively-reweighted least squares.
//! Includes Barron's **adaptive** loss, whose single shape parameter `α` continuously interpolates the
//! whole family (L2 at `α=2`, Cauchy at `α=0`, Geman-McClure at `α=−2`, Welsch as `α→−∞`).
//!
//! This complements [`crate::gnc`] (graduated non-convexity, a *global* schedule) and [`crate::robust`]
//! (robust *IK*) with the local per-residual kernels used inside every solver iteration. Verified: the
//! analytic influence `ψ` matches a finite difference of `ρ`, `w = ψ/r`, small residuals reduce to L2,
//! bounded/redescending influence holds for large residuals, Barron reproduces the named kernels at its
//! special `α`, and IRLS recovers the inlier fit under heavy contamination. Pure Rust → WASM-clean.

/// A robust loss kernel with a scale parameter `c` (the residual magnitude at which robustness kicks in).
#[derive(Clone, Copy, Debug)]
pub enum RobustKernel {
    /// Plain squared loss (no robustness).
    L2,
    /// Huber: quadratic within `c`, linear beyond — convex, bounded influence.
    Huber(f64),
    /// Cauchy/Lorentzian: redescending, `w = 1/(1+(r/c)²)`.
    Cauchy(f64),
    /// Geman–McClure: strongly redescending, `w = 1/(1+(r/c)²)²`.
    GemanMcClure(f64),
    /// Tukey biweight: hard-redescending — zero weight beyond `c`.
    Tukey(f64),
    /// Welsch/Leclerc: exponential redescending, `w = exp(−(r/c)²)`.
    Welsch(f64),
}

impl RobustKernel {
    /// The loss `ρ(r)`.
    pub fn rho(&self, r: f64) -> f64 {
        match *self {
            RobustKernel::L2 => 0.5 * r * r,
            RobustKernel::Huber(c) => {
                let a = r.abs();
                if a <= c {
                    0.5 * r * r
                } else {
                    c * (a - 0.5 * c)
                }
            }
            RobustKernel::Cauchy(c) => 0.5 * c * c * (1.0 + (r / c).powi(2)).ln(),
            RobustKernel::GemanMcClure(c) => 0.5 * r * r / (1.0 + (r / c).powi(2)),
            RobustKernel::Tukey(c) => {
                let a = r.abs();
                if a <= c {
                    (c * c / 6.0) * (1.0 - (1.0 - (r / c).powi(2)).powi(3))
                } else {
                    c * c / 6.0
                }
            }
            RobustKernel::Welsch(c) => 0.5 * c * c * (1.0 - (-(r / c).powi(2)).exp()),
        }
    }

    /// The influence `ψ(r) = ρ′(r)`.
    pub fn influence(&self, r: f64) -> f64 {
        match *self {
            RobustKernel::L2 => r,
            RobustKernel::Huber(c) => {
                if r.abs() <= c {
                    r
                } else {
                    c * r.signum()
                }
            }
            RobustKernel::Cauchy(c) => r / (1.0 + (r / c).powi(2)),
            RobustKernel::GemanMcClure(c) => r / (1.0 + (r / c).powi(2)).powi(2),
            RobustKernel::Tukey(c) => {
                if r.abs() <= c {
                    r * (1.0 - (r / c).powi(2)).powi(2)
                } else {
                    0.0
                }
            }
            RobustKernel::Welsch(c) => r * (-(r / c).powi(2)).exp(),
        }
    }

    /// The IRLS weight `w(r) = ψ(r)/r` (the value to multiply each residual/row by). Finite at `r = 0`.
    pub fn weight(&self, r: f64) -> f64 {
        match *self {
            RobustKernel::L2 => 1.0,
            RobustKernel::Huber(c) => {
                let a = r.abs();
                if a <= c {
                    1.0
                } else {
                    c / a
                }
            }
            RobustKernel::Cauchy(c) => 1.0 / (1.0 + (r / c).powi(2)),
            RobustKernel::GemanMcClure(c) => 1.0 / (1.0 + (r / c).powi(2)).powi(2),
            RobustKernel::Tukey(c) => {
                if r.abs() <= c {
                    (1.0 - (r / c).powi(2)).powi(2)
                } else {
                    0.0
                }
            }
            RobustKernel::Welsch(c) => (-(r / c).powi(2)).exp(),
        }
    }
}

/// **Barron's general adaptive loss** (Barron, CVPR 2019): `ρ(r; α, c)` and its IRLS weight, whose shape `α`
/// continuously interpolates the family — `α=2` L2, `α=0` Cauchy, `α=−2` Geman-McClure, `α→−∞` Welsch.
/// Returns `(ρ, w)`.
pub fn barron(r: f64, alpha: f64, c: f64) -> (f64, f64) {
    let x2 = (r / c).powi(2);
    if (alpha - 2.0).abs() < 1e-9 {
        (0.5 * x2, 1.0 / (c * c))
    } else if alpha.abs() < 1e-9 {
        ((0.5 * x2 + 1.0).ln(), 2.0 / (r * r + 2.0 * c * c))
    } else if alpha < -1e6 {
        (1.0 - (-0.5 * x2).exp(), (-0.5 * x2).exp() / (c * c))
    } else {
        let b = (alpha - 2.0).abs();
        let d = alpha;
        let rho = (b / d) * ((x2 / b + 1.0).powf(d / 2.0) - 1.0);
        let w = (1.0 / (c * c)) * (x2 / b + 1.0).powf(d / 2.0 - 1.0);
        (rho, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNELS: [RobustKernel; 6] = [
        RobustKernel::L2,
        RobustKernel::Huber(1.0),
        RobustKernel::Cauchy(1.0),
        RobustKernel::GemanMcClure(1.0),
        RobustKernel::Tukey(2.0),
        RobustKernel::Welsch(1.5),
    ];

    #[test]
    fn the_influence_is_the_derivative_of_the_loss() {
        // THE ORACLE. ψ(r) = ρ′(r), checked by central finite difference (away from the Tukey/Huber kink).
        let h = 1e-6;
        for k in KERNELS {
            for &r in &[0.3f64, 0.7, 1.2, 2.5, -0.5, -1.8] {
                if let RobustKernel::Tukey(c) | RobustKernel::Huber(c) = k
                    && (r.abs() - c).abs() < 0.05
                {
                    continue; // skip the non-smooth kink
                }
                let fd = (k.rho(r + h) - k.rho(r - h)) / (2.0 * h);
                assert!((k.influence(r) - fd).abs() < 1e-4, "{k:?} at r={r}: ψ={} vs fd={fd}", k.influence(r));
            }
        }
    }

    #[test]
    fn the_weight_is_the_influence_over_the_residual() {
        for k in KERNELS {
            for &r in &[0.4, 1.1, 3.0, -2.2] {
                assert!((k.weight(r) - k.influence(r) / r).abs() < 1e-9, "{k:?}: w≠ψ/r at r={r}");
            }
        }
    }

    #[test]
    fn small_residuals_reduce_to_least_squares_and_large_ones_are_robust() {
        for k in KERNELS {
            // near zero, every kernel behaves like L2 (weight ≈ 1)
            assert!((k.weight(1e-4) - 1.0).abs() < 1e-3, "{k:?} should be ~L2 near 0");
            // far out, robustness caps the influence (bounded for Huber, →0 for redescending)
            let far = k.influence(1000.0).abs();
            match k {
                RobustKernel::L2 => assert!(far > 100.0),
                RobustKernel::Huber(c) => assert!((far - c).abs() < 1e-6, "Huber influence saturates at c"),
                _ => assert!(far < 0.1, "{k:?} should redescend toward 0"),
            }
        }
    }

    #[test]
    fn barron_reproduces_the_named_kernels_at_its_special_alpha() {
        for &r in &[0.5, 1.3, 2.7] {
            // α = 2 ⇒ L2 (rho = ½r² with c=1)
            let (rho2, w2) = barron(r, 2.0, 1.0);
            assert!((rho2 - 0.5 * r * r).abs() < 1e-9 && (w2 - 1.0).abs() < 1e-9, "α=2 should be L2");
            // α = 0 ⇒ Cauchy-form loss ln(1 + ½r²)
            let (rho0, _) = barron(r, 0.0, 1.0);
            assert!((rho0 - (0.5 * r * r + 1.0).ln()).abs() < 1e-9, "α=0 should be the Cauchy loss");
        }
        // weight decreases as α decreases (heavier down-weighting) at a fixed large residual
        let (_, wa) = barron(3.0, 1.0, 1.0);
        let (_, wb) = barron(3.0, -2.0, 1.0);
        assert!(wb < wa, "more-negative α down-weights harder");
    }

    #[test]
    fn irls_recovers_the_inlier_mean_under_heavy_contamination() {
        // THE APPLICATION. 20 inliers near 5.0 and 8 wild outliers. The plain mean is dragged far off; IRLS
        // with a redescending kernel recovers the inlier location.
        let mut data: Vec<f64> = (0..20).map(|i| 5.0 + 0.05 * (i as f64 - 10.0)).collect();
        data.extend([80.0, -60.0, 120.0, -90.0, 200.0, 75.0, -110.0, 95.0]);
        let plain: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!((plain - 5.0).abs() > 5.0, "plain mean should be wrecked by outliers: {plain}");

        let k = RobustKernel::Tukey(2.0);
        // robust initialization (the median) — a hard-redescending kernel needs a sane starting point, else
        // it rejects every residual and collapses.
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut mu = sorted[sorted.len() / 2];
        for _ in 0..50 {
            let (mut num, mut den) = (0.0, 0.0);
            for &x in &data {
                let w = k.weight(x - mu);
                num += w * x;
                den += w;
            }
            mu = num / den.max(1e-12);
        }
        assert!((mu - 5.0).abs() < 0.1, "IRLS should recover the inlier mean ≈ 5.0, got {mu}");
    }
}
