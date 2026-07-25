//! **Smoothed gradients through contact make/break** — the correct-gradients-through-contact
//! frontier (Dojo; "Differentiable Physics Simulations with Contacts: do they have correct
//! gradients?"). An analytic contact adjoint with the *frozen active set* (the OptNet/theseus
//! convention ferromotion uses in its constraint solver) is exact everywhere except *at* the
//! activation boundaries — where a contact makes or breaks, stick switches to slip, or a body
//! grazes an obstacle. There the true gradient is a subgradient or a cliff, and a naive descent
//! stalls (flat pre-contact region) or oscillates (a jump).
//!
//! Randomized smoothing repairs this: replace `f(θ)` by its Gaussian convolution
//! `f_σ(θ) = E_{ξ∼𝒩(0,1)}[f(θ+σξ)]`, whose gradient is informative *through* the switch. The
//! ferromotion-flavored estimator is **deterministic**: Gauss–Hermite quadrature of the smoothing
//! integral, with the gradient obtained in closed form by Stein's identity
//! `d/dθ E[f(θ+σξ)] = (1/σ) E[ξ·f(θ+σξ)]`. No RNG, reproducible bit-for-bit — the determinism the
//! rest of the stack is built on. A quasi-Monte-Carlo estimator covers higher dimensions.

use nalgebra::DMatrix;

/// Gauss–Hermite nodes and weights for the *probabilists'* Gaussian `𝒩(0,1)` (so
/// `E[g] ≈ Σ wₖ g(xₖ)`), via Golub–Welsch: the symmetric tridiagonal Jacobi matrix has zero
/// diagonal and sub-diagonal `√k`; eigenvalues are the nodes, weights are the squared first
/// components of the (unit) eigenvectors.
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        let b = (k as f64).sqrt();
        j[(k, k - 1)] = b;
        j[(k - 1, k)] = b;
    }
    let eig = j.symmetric_eigen();
    let nodes: Vec<f64> = (0..n).map(|i| eig.eigenvalues[i]).collect();
    let weights: Vec<f64> = (0..n).map(|i| eig.eigenvectors[(0, i)].powi(2)).collect();
    (nodes, weights)
}

/// The Gaussian-smoothed value `f_σ(θ) = E[f(θ+σξ)]` in 1-D, by `n`-node Gauss–Hermite quadrature.
pub fn smoothed_value_1d(f: impl Fn(f64) -> f64, theta: f64, sigma: f64, n: usize) -> f64 {
    let (x, w) = gauss_hermite(n);
    (0..n).map(|k| w[k] * f(theta + sigma * x[k])).sum()
}

/// The smoothed gradient `d/dθ f_σ(θ) = (1/σ) E[ξ·f(θ+σξ)]` in 1-D (Stein's identity), by
/// Gauss–Hermite quadrature — deterministic and exact for the smoothing, informative through a
/// make/break switch where the raw gradient is not.
pub fn smoothed_grad_1d(f: impl Fn(f64) -> f64, theta: f64, sigma: f64, n: usize) -> f64 {
    let (x, w) = gauss_hermite(n);
    (0..n).map(|k| w[k] * x[k] * f(theta + sigma * x[k])).sum::<f64>() / sigma
}

/// A deterministic standard-normal sample sequence (Box–Muller over a splitmix64 stream) for the
/// higher-dimensional quasi-Monte-Carlo smoothed gradient — reproducible, no RNG crate.
fn normal_samples(count: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z ^ (z >> 31)) as f64) / (u64::MAX as f64)
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = next().max(1e-12);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f64::consts::TAU * u2).cos());
        out.push(r * (std::f64::consts::TAU * u2).sin());
    }
    out.truncate(count);
    out
}

/// The smoothed gradient `∇f_σ(θ) = (1/σ) E[ξ·f(θ+σξ)]` for a vector parameter, by deterministic
/// quasi-Monte-Carlo (Stein estimator, `n_samples` normal draws), with antithetic variates for
/// variance reduction.
pub fn smoothed_grad(f: impl Fn(&[f64]) -> f64, theta: &[f64], sigma: f64, n_samples: usize) -> Vec<f64> {
    let d = theta.len();
    let xi = normal_samples(n_samples * d, 0xF1u64);
    let mut g = vec![0.0f64; d];
    let mut cand = vec![0.0f64; d];
    for s in 0..n_samples {
        // antithetic pair ±ξ
        for sign in [1.0f64, -1.0] {
            for i in 0..d {
                cand[i] = theta[i] + sign * sigma * xi[s * d + i];
            }
            let fv = f(&cand);
            for i in 0..d {
                g[i] += sign * xi[s * d + i] * fv;
            }
        }
    }
    let norm = 1.0 / (sigma * (2 * n_samples) as f64);
    g.iter_mut().for_each(|v| *v *= norm);
    g
}

#[cfg(test)]
mod verification {
    use super::*;

    /// The smoothed gradient of a SMOOTH function recovers the true derivative as `σ → 0`.
    #[test]
    fn smoothed_grad_recovers_true_derivative() {
        let f = |t: f64| (0.7 * t).sin() + 0.3 * t * t; // f'(t) = 0.7cos(0.7t) + 0.6t
        for &t in &[-1.0f64, 0.0, 0.6, 1.5] {
            let exact = 0.7 * (0.7 * t).cos() + 0.6 * t;
            let g = smoothed_grad_1d(f, t, 1e-3, 21);
            assert!((g - exact).abs() < 1e-4, "smoothed grad {g} vs exact {exact} at {t}");
        }
    }

    // A projectile launched at speed θ (fixed 45° angle) over a wall of height h at distance d.
    // If it clears the wall it lands at range R(θ)=θ²/g; if not, it is blocked at the wall base d.
    // f is FLAT below the clearing threshold θ* and jumps up at θ* — the canonical make/break
    // discontinuity where an analytic gradient is zero (blocked) or a cliff.
    fn projectile(theta: f64) -> f64 {
        let (d, h, g) = (5.0, 2.0, 9.8);
        let y_at_wall = d - g * d * d / (theta * theta); // 45°: y(d) = d − g d²/v²
        if y_at_wall > h {
            theta * theta / g // range, clears
        } else {
            d // blocked at the wall
        }
    }

    /// **The core claim.** In the blocked region the true gradient of the objective is exactly zero
    /// (the landing point is flat at the wall), so naive descent cannot escape; the smoothed
    /// gradient sees the cliff and points the right way.
    #[test]
    fn smoothed_grad_is_informative_across_make_break() {
        let target = 10.0;
        let j = |theta: f64| (projectile(theta) - target).powi(2);
        // θ* (clearing) is at v² = g d²/(d−h) = 9.8·25/3 ≈ 81.7 → θ* ≈ 9.04. Start below it.
        let theta0 = 8.0;
        // raw one-sided gradient in the blocked region is 0 (f is flat) → descent stalls
        let raw = (j(theta0 + 1e-6) - j(theta0 - 1e-6)) / 2e-6;
        let smoothed = smoothed_grad_1d(j, theta0, 0.6, 41);
        eprintln!("at θ={theta0} (blocked): raw grad {raw:.3e}, smoothed grad {smoothed:.3e}");
        assert!(raw.abs() < 1e-6, "raw gradient should be ~0 in the flat region");
        assert!(smoothed < -1.0, "smoothed gradient should point toward clearing (negative J-grad): {smoothed}");
    }

    /// **The payoff.** Gradient descent with the smoothed gradient crosses the make/break boundary
    /// and hits a target only reachable by clearing the wall — where naive descent stays blocked.
    #[test]
    fn smoothed_descent_crosses_the_boundary() {
        let target = 10.0;
        let j = |theta: f64| (projectile(theta) - target).powi(2);

        // naive descent (exact a.e. gradient) — stuck in the blocked flat region
        let mut tn = 8.0;
        for _ in 0..200 {
            let g = (j(tn + 1e-6) - j(tn - 1e-6)) / 2e-6;
            tn -= 0.02 * g;
        }
        let naive_blocked = projectile(tn) < 5.5; // still at the wall base ≈ d = 5

        // smoothed descent — anneal σ, escapes and converges to the target
        let mut ts = 8.0;
        let mut sigma = 0.8;
        for it in 0..200 {
            let g = smoothed_grad_1d(j, ts, sigma, 41);
            ts -= 0.05 * g;
            ts = ts.clamp(6.0, 14.0);
            if it % 40 == 39 {
                sigma *= 0.6; // sharpen as it approaches the solution
            }
        }
        let landed = projectile(ts);
        eprintln!("naive final θ={tn:.3} (blocked={naive_blocked}); smoothed final θ={ts:.3}, landed {landed:.3} (target {target})");
        assert!(naive_blocked, "naive descent unexpectedly escaped — weaken the test scenario");
        assert!((landed - target).abs() < 0.3, "smoothed descent did not reach the target: landed {landed}");
    }
}
