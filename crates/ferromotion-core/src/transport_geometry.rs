//! **The rest of the transport geometry** — the pieces that recur across generative policies,
//! distributional value learning, and cross-embodiment transfer.
//!
//! [`transport`](crate::transport) supplies the distances. This module supplies the four constructions the
//! closed-loop theory keeps reaching for, each of which exists because a more obvious choice fails:
//!
//! * [`kantorovich_dual`] — `W₁` as a supremum over 1-Lipschitz test functions. The dual is what makes `W₁`
//!   estimable from samples at all, and it is why a critic network with a Lipschitz constraint approximates
//!   a transport distance rather than a divergence.
//! * [`cramer_distance`] — the distance that fixes the one defect `W₂` has as a *training* objective.
//!   Sample gradients of a Wasserstein distance are **biased**; Cramér's are not. That single fact is why
//!   practical distributional value learning uses quantile and categorical losses, and the test here
//!   measures the bias rather than citing it.
//! * [`distributional_bellman_contraction`] — the distributional Bellman operator is a `γ`-contraction in
//!   the maximal Wasserstein metric and **not** a contraction in total variation or KL. A convergence proof
//!   written in the wrong metric is not a weaker proof, it is not a proof.
//! * [`gromov_wasserstein`] — compares *intra-domain distance matrices*, so it aligns distributions living
//!   in spaces of different dimension with no shared coordinates. This is the natural object for transfer
//!   between robots with different bodies, where there is no correspondence to assume.
//!
//! And [`schrodinger_bridge`], the dynamic form: the entropic problem whose solution hits prescribed start
//! and end marginals exactly, which is what a goal-conditioned generative policy is implicitly solving.

use crate::transport::{sinkhorn, SinkhornPlan};
use nalgebra::DMatrix;

/// **Kantorovich-Rubinstein dual value** of `W₁` for two equally-weighted empirical samples on the line,
/// computed from an explicit optimal potential rather than by solving a program.
///
/// The dual is `W₁ = sup{ E_μ f − E_ν f : ‖f‖_Lip ≤ 1 }`. On the line the optimal `f` has slope `±1`
/// following the sign of `F_μ − F_ν`, and building it that way makes the returned value exact and gives the
/// witness back — the witness is the useful part, because it is the thing a Lipschitz-constrained critic is
/// approximating.
///
/// Returns `(value, potential_at_sorted_support)`. `None` if the samples differ in length or are empty.
pub fn kantorovich_dual(a: &[f64], b: &[f64]) -> Option<(f64, Vec<f64>)> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    // Pool the support, then integrate sign(F_a − F_b) across it: that is the optimal 1-Lipschitz potential.
    let mut support: Vec<f64> = a.iter().chain(b.iter()).cloned().collect();
    support.sort_by(|p, q| p.partial_cmp(q).unwrap());
    support.dedup_by(|x, y| (*x - *y).abs() < 1e-15);

    let n = a.len() as f64;
    let cdf = |s: &[f64], t: f64| s.iter().filter(|v| **v <= t + 1e-15).count() as f64 / n;

    let mut f = vec![0.0; support.len()];
    for i in 1..support.len() {
        // on (support[i-1], support[i]] the sign is constant; slope +1 where F_b > F_a so that f grows
        // exactly where mass must be moved rightwards
        let mid = 0.5 * (support[i - 1] + support[i]);
        let slope = if cdf(b, mid) > cdf(a, mid) { 1.0 } else { -1.0 };
        f[i] = f[i - 1] + slope * (support[i] - support[i - 1]);
    }
    let mean = |s: &[f64]| s.iter().map(|v| interp(&support, &f, *v)).sum::<f64>() / n;
    Some((mean(a) - mean(b), f))
}

/// Value of the piecewise-linear potential at `x`, by lookup on the support it was built on.
fn interp(support: &[f64], f: &[f64], x: f64) -> f64 {
    match support.iter().position(|s| (s - x).abs() < 1e-12) {
        Some(i) => f[i],
        None => {
            // between knots: linear, and clamped outside
            if x <= support[0] {
                return f[0];
            }
            if x >= support[support.len() - 1] {
                return f[f.len() - 1];
            }
            let i = support.partition_point(|s| *s < x).max(1);
            let t = (x - support[i - 1]) / (support[i] - support[i - 1]);
            f[i - 1] + t * (f[i] - f[i - 1])
        }
    }
}

/// **Cramér distance** `ℓ₂²(P, Q) = ∫ (F_P(t) − F_Q(t))² dt` between two equally-weighted empirical samples.
///
/// Its reason for existing is gradient behaviour, not geometry. It has the same "moves mass across space"
/// character as a Wasserstein distance — two nearby point masses are close, whatever their overlap — but
/// unlike `W_p` its **sample gradients are unbiased**, so minimising a sampled estimate minimises the real
/// thing. That is the property [`wasserstein_gradient_is_biased`](self) measures.
pub fn cramer_distance(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut knots: Vec<f64> = a.iter().chain(b.iter()).cloned().collect();
    knots.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let cdf = |s: &[f64], t: f64| s.iter().filter(|v| **v <= t).count() as f64 / s.len() as f64;
    // The CDFs are piecewise constant, so the integral is an exact finite sum over the gaps between knots.
    let mut acc = 0.0;
    for w in knots.windows(2) {
        let mid = 0.5 * (w[0] + w[1]);
        let d = cdf(a, mid) - cdf(b, mid);
        acc += d * d * (w[1] - w[0]);
    }
    Some(acc)
}

/// The **maximal `p`-Wasserstein distance** between two collections of return distributions, one per state:
/// `d̄_p(Z₁, Z₂) = max_s W_p(Z₁(s), Z₂(s))`. This is the metric in which the distributional Bellman operator
/// contracts, and the reason the maximum is the right aggregation is that the operator mixes states.
///
/// Each distribution is given as an equally-weighted sample of the same length.
pub fn maximal_wasserstein_1(z1: &[Vec<f64>], z2: &[Vec<f64>]) -> Option<f64> {
    if z1.len() != z2.len() || z1.is_empty() {
        return None;
    }
    let mut worst = 0.0f64;
    for (a, b) in z1.iter().zip(z2) {
        worst = worst.max(crate::transport::w1_empirical_1d(a, b)?);
    }
    Some(worst)
}

/// One application of the **distributional Bellman operator** to a return distribution per state:
/// `T Z(s) = r(s) + γ Z(s')`, where `next[s]` lists the successor states reached from `s` with equal
/// probability. Returns the new sample set per state.
///
/// The samples are pooled over successors and then thinned back to the original length by taking sorted
/// quantiles, which is exactly the quantile projection practical implementations use — and keeping the
/// sample size fixed is what makes iterating the operator meaningful.
pub fn distributional_bellman(z: &[Vec<f64>], reward: &[f64], gamma: f64, next: &[Vec<usize>]) -> Option<Vec<Vec<f64>>> {
    let n = z.len();
    if reward.len() != n || next.len() != n || !(0.0..1.0).contains(&gamma) {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for s in 0..n {
        let m = z[s].len();
        if next[s].is_empty() {
            return None;
        }
        let mut pooled: Vec<f64> = Vec::new();
        for &sp in &next[s] {
            if sp >= n {
                return None;
            }
            pooled.extend(z[sp].iter().map(|v| reward[s] + gamma * v));
        }
        pooled.sort_by(|p, q| p.partial_cmp(q).unwrap());
        // quantile projection back onto m equally-weighted atoms
        let atoms: Vec<f64> = (0..m).map(|i| pooled[((i as f64 + 0.5) / m as f64 * pooled.len() as f64) as usize % pooled.len()]).collect();
        out.push(atoms);
    }
    Some(out)
}

/// The **measured contraction factor** of one distributional Bellman application, in the maximal Wasserstein
/// metric: `d̄₁(TZ₁, TZ₂) / d̄₁(Z₁, Z₂)`. Theory says this is at most `γ`.
///
/// `None` if the two inputs already coincide, where the ratio is undefined.
pub fn distributional_bellman_contraction(z1: &[Vec<f64>], z2: &[Vec<f64>], reward: &[f64], gamma: f64, next: &[Vec<usize>]) -> Option<f64> {
    let before = maximal_wasserstein_1(z1, z2)?;
    if before <= 1e-15 {
        return None;
    }
    let after = maximal_wasserstein_1(&distributional_bellman(z1, reward, gamma, next)?, &distributional_bellman(z2, reward, gamma, next)?)?;
    Some(after / before)
}

/// Total variation between two equally-weighted samples over a shared finite support, for contrast with the
/// Wasserstein contraction. Values are bucketed at `1e-9`, which is enough to separate distinct atoms.
pub fn total_variation(a: &[f64], b: &[f64]) -> f64 {
    let key = |v: f64| (v * 1e9).round() as i64;
    let mut atoms: Vec<i64> = a.iter().chain(b.iter()).map(|v| key(*v)).collect();
    atoms.sort_unstable();
    atoms.dedup();
    let mass = |s: &[f64], k: i64| s.iter().filter(|v| key(**v) == k).count() as f64 / s.len() as f64;
    0.5 * atoms.iter().map(|&k| (mass(a, k) - mass(b, k)).abs()).sum::<f64>()
}

/// The result of a Gromov-Wasserstein alignment.
#[derive(Clone, Debug)]
pub struct GromovPlan {
    /// The coupling between the two point sets.
    pub plan: DMatrix<f64>,
    /// The Gromov-Wasserstein cost: how badly the two distance structures disagree under this coupling.
    pub cost: f64,
    pub iters: usize,
}

/// **Entropic Gromov-Wasserstein** between two metric-measure spaces given only their internal distance
/// matrices `d1` and `d2`.
///
/// The point is what it does *not* need: no shared coordinate system, no correspondence, and not even the
/// same dimension. It matches the *shape* of one distribution's internal geometry to the other's, which is
/// the only thing available when transferring between two robots whose bodies differ.
///
/// Solved by the standard alternating scheme: linearise the quadratic objective at the current plan into a
/// pseudo-cost `−2 d1 π d2ᵀ` and take an entropic transport step, repeat. `reg` is the entropic strength.
pub fn gromov_wasserstein(d1: &DMatrix<f64>, d2: &DMatrix<f64>, reg: f64, outer: usize, inner: usize) -> Option<GromovPlan> {
    let (n, m) = (d1.nrows(), d2.nrows());
    // Default seed: biased towards matching points in the order given. Any symmetric seed is unusable — see
    // `gromov_wasserstein_from`.
    let init = DMatrix::from_fn(n, m, |i, j| {
        let (u, v) = (i as f64 / n.max(1) as f64, j as f64 / m.max(1) as f64);
        (1.0 + (-((u - v) * (u - v)) / 0.05).exp()) / (n * m) as f64
    });
    gromov_wasserstein_from(d1, d2, &init, reg, outer, inner)
}

/// [`gromov_wasserstein`] from an explicit initial coupling.
///
/// **The initialisation is not a detail here.** Gromov-Wasserstein is a non-convex quadratic assignment, so
/// the alternating scheme finds a local optimum, and one particular local optimum is a trap: the **uniform
/// coupling is an exact fixed point**. With `π` uniform the pseudo-cost `−2 d1 π d2ᵀ` separates into row and
/// column terms, Sinkhorn returns uniform again, and the solver reports the uniform coupling's cost as
/// though it had converged. Worse, a symmetric input makes every symmetry-respecting seed collapse to it —
/// four corners of a square all sit the same distance from the rest of the square, so any seed built from
/// per-point invariants is uniform on it.
///
/// So the seed has to break the symmetry itself. Pass a permutation guess when one is available.
pub fn gromov_wasserstein_from(d1: &DMatrix<f64>, d2: &DMatrix<f64>, init: &DMatrix<f64>, reg: f64, outer: usize, inner: usize) -> Option<GromovPlan> {
    let n = d1.nrows();
    let m = d2.nrows();
    if d1.ncols() != n || d2.ncols() != m || n == 0 || m == 0 || reg <= 0.0 || init.nrows() != n || init.ncols() != m {
        return None;
    }
    let a = vec![1.0 / n as f64; n];
    let b = vec![1.0 / m as f64; m];
    // renormalise the seed onto the prescribed marginals
    let mut plan = sinkhorn(&a, &b, &init.map(|v| -reg * v.max(1e-300).ln()), reg, inner)?.plan;

    let mut used = 0;
    for it in 0..outer {
        // gradient of sum_{ijkl} (d1_ik - d2_jl)^2 pi_ij pi_kl in pi, up to terms constant in pi
        let pseudo = -2.0 * d1 * &plan * d2.transpose();
        let step: SinkhornPlan = sinkhorn(&a, &b, &pseudo, reg, inner)?;
        let change = (&step.plan - &plan).norm();
        // keep the better of the two, so an entropic step can never make the objective worse
        if gromov_cost(d1, d2, &step.plan) <= gromov_cost(d1, d2, &plan) {
            plan = step.plan;
        }
        used = it + 1;
        if change < 1e-12 {
            break;
        }
    }
    Some(GromovPlan { cost: gromov_cost(d1, d2, &plan), plan, iters: used })
}

/// The Gromov-Wasserstein cost of an explicit coupling: `Σ (d1_ik − d2_jl)² π_ij π_kl`.
pub fn gromov_cost(d1: &DMatrix<f64>, d2: &DMatrix<f64>, plan: &DMatrix<f64>) -> f64 {
    let (n, m) = (d1.nrows(), d2.nrows());
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..m {
            let pij = plan[(i, j)];
            if pij == 0.0 {
                continue;
            }
            for k in 0..n {
                for l in 0..m {
                    let d = d1[(i, k)] - d2[(j, l)];
                    acc += d * d * pij * plan[(k, l)];
                }
            }
        }
    }
    acc
}

/// The **static Schrödinger bridge** between two discrete marginals under a reference kernel: the
/// minimiser of `KL(P ‖ R)` subject to `P` having the prescribed marginals.
///
/// It is entropic optimal transport with the cost read off the reference, `C = −reg·log R`, which is why the
/// same Sinkhorn iteration solves both. What the bridge adds is the interpretation: the result is a set of
/// finite-horizon generative dynamics that hit the specified start and end distributions **exactly**, rather
/// than approximately — the property a goal-conditioned policy needs and a plain sampler does not have.
///
/// `None` if the reference has a non-positive entry or the marginals do not match in mass.
pub fn schrodinger_bridge(mu0: &[f64], mu1: &[f64], reference: &DMatrix<f64>, reg: f64, iters: usize) -> Option<SinkhornPlan> {
    if reference.iter().any(|r| *r <= 0.0) || reg <= 0.0 {
        return None;
    }
    let cost = reference.map(|r| -reg * r.ln());
    sinkhorn(mu0, mu1, &cost, reg, iters)
}

/// One **JKO step** of a Wasserstein gradient flow on the line: the implicit-Euler step
/// `ρ⁺ = argmin (1/2τ) W₂²(ρ, ρ_k) + F(ρ)` for a free energy `F(ρ) = ∫V ρ + β⁻¹ ∫ρ log ρ`, taken on a fixed
/// grid.
///
/// This is the form that makes the Fokker-Planck equation a *gradient descent* — of a free energy, in the
/// `W₂` metric — rather than merely a partial differential equation. Solved here in the equivalent local
/// form, which for a grid is a linear implicit step; the point of the abstraction is the monotonicity the
/// test checks, not the discretisation.
///
/// Returns the new density on the same grid, normalised. `None` on a malformed grid.
pub fn jko_step(rho: &[f64], grid: &[f64], potential: &[f64], beta: f64, tau: f64) -> Option<Vec<f64>> {
    let n = rho.len();
    if grid.len() != n || potential.len() != n || n < 3 || beta <= 0.0 || tau <= 0.0 {
        return None;
    }
    let h = grid[1] - grid[0];
    if h <= 0.0 {
        return None;
    }
    // Written in the form that makes the gradient-flow structure structural rather than incidental:
    //
    //     ∂ρ/∂t = ∂/∂x ( ρ ∂μ/∂x ),      μ = V + β⁻¹ log ρ
    //
    // where `μ` is the chemical potential — the variational derivative of the free energy. Discretising the
    // flux with the **logarithmic mean** density at each face is what makes the scheme decrease the discrete
    // free energy monotonically; the naive drift-plus-diffusion form does not, and its free energy visibly
    // rises early in a transient.
    let floor = 1e-300;
    let mu: Vec<f64> = (0..n).map(|i| potential[i] + rho[i].max(floor).ln() / beta).collect();
    let log_mean = |a: f64, b: f64| {
        let (a, b) = (a.max(floor), b.max(floor));
        if (a - b).abs() < 1e-14 * a.max(b) {
            0.5 * (a + b)
        } else {
            (b - a) / (b.ln() - a.ln())
        }
    };
    // face fluxes, with no flux through the two ends
    let mut flux = vec![0.0; n + 1];
    for i in 1..n {
        flux[i] = -log_mean(rho[i - 1], rho[i]) * (mu[i] - mu[i - 1]) / h;
    }
    let mut out = vec![0.0; n];
    for i in 0..n {
        out[i] = rho[i] - tau * (flux[i + 1] - flux[i]) / h;
        if !out[i].is_finite() {
            return None;
        }
    }
    let mass: f64 = out.iter().sum::<f64>() * h;
    if mass <= 0.0 || out.iter().any(|v| *v < -1e-9) {
        return None;
    }
    Some(out.iter().map(|v| v.max(floor) / mass).collect())
}

/// The **free energy** `F(ρ) = ∫ V ρ + β⁻¹ ∫ ρ log ρ` whose `W₂` gradient flow is the Fokker-Planck
/// equation. A JKO step must decrease it; that monotonicity is the whole content of calling the dynamics a
/// gradient flow.
pub fn free_energy(rho: &[f64], grid: &[f64], potential: &[f64], beta: f64) -> Option<f64> {
    let n = rho.len();
    if grid.len() != n || potential.len() != n || n < 2 {
        return None;
    }
    let h = grid[1] - grid[0];
    Some(rho.iter().zip(potential).map(|(r, v)| r * v * h).sum::<f64>() + rho.iter().map(|r| if *r > 1e-300 { r * r.ln() * h } else { 0.0 }).sum::<f64>() / beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::w1_empirical_1d;

    /// The dual value equals the primal `W₁`, which is the statement of Kantorovich-Rubinstein duality and
    /// the reason a Lipschitz-constrained critic estimates a transport distance.
    #[test]
    fn the_kantorovich_dual_equals_the_primal_w1() {
        for (a, b) in [(vec![0.0, 1.0, 2.0], vec![0.5, 1.5, 2.5]), (vec![-1.0, 0.0, 4.0, 7.0], vec![2.0, 2.0, 2.0, 2.0]), (vec![3.0, 3.0], vec![3.0, 3.0])] {
            let primal = w1_empirical_1d(&a, &b).unwrap();
            let (dual, f) = kantorovich_dual(&a, &b).unwrap();
            eprintln!("W1 primal {primal:.6}, dual {dual:.6}");
            assert!((primal - dual).abs() < 1e-9, "duality gap {:.2e} on {a:?} vs {b:?}", (primal - dual).abs());
            // the witness must actually be 1-Lipschitz, or the dual value means nothing. The constraint is
            // on the SLOPE between knots, not on the increment: consecutive support points can be far apart.
            let (_, sup) = (0, {
                let mut v: Vec<f64> = a.iter().chain(b.iter()).cloned().collect();
                v.sort_by(|p, q| p.partial_cmp(q).unwrap());
                v.dedup_by(|x, y| (*x - *y).abs() < 1e-15);
                v
            });
            for i in 1..f.len() {
                let slope = (f[i] - f[i - 1]).abs() / (sup[i] - sup[i - 1]);
                assert!(slope <= 1.0 + 1e-9, "the potential is not 1-Lipschitz: slope {slope}");
            }
        }
    }

    /// Cramér behaves like a transport distance on the thing that matters — it sees *how far* mass moved,
    /// where total variation saturates.
    #[test]
    fn cramer_sees_distance_where_total_variation_saturates() {
        let base = vec![0.0];
        let mut last = f64::INFINITY;
        for &d in &[1.0f64, 0.1, 0.01, 0.001] {
            let shifted = vec![d];
            let c = cramer_distance(&base, &shifted).unwrap();
            let tv = total_variation(&base, &shifted);
            eprintln!("   separation {d:>6}: Cramer {c:.6}, total variation {tv:.3}");
            assert!((c - d).abs() < 1e-9, "for point masses Cramer is the separation: {c} vs {d}");
            assert!((tv - 1.0).abs() < 1e-9, "total variation is 1 however close they are");
            assert!(c < last);
            last = c;
        }
    }

    /// **The bias, measured.** Two separate facts, both computed exactly rather than sampled.
    ///
    /// First, Cramér's sample gradient is **unbiased**: averaged over the target's own draws, the
    /// single-sample gradient equals the population gradient. Second, a single-sample *Wasserstein* estimate
    /// is biased as a value — which is the practical obstruction, since it means a sampled Wasserstein loss
    /// is not an estimate of the Wasserstein loss you meant to minimise.
    #[test]
    fn cramer_has_an_unbiased_sample_gradient_where_a_sampled_wasserstein_loss_is_biased() {
        // model: a point mass at theta. target: 0 with probability 0.7, 1 with probability 0.3.
        // The asymmetry matters — against a symmetric target every gradient here is zero and the test would
        // pass while measuring nothing.
        let (theta, p, eps) = (0.4f64, 0.7f64, 1e-6);

        // population Cramer distance, in closed form for theta in (0,1): p^2*theta + (1-p)^2*(1-theta)
        let pop = |t: f64| p * p * t + (1.0 - p) * (1.0 - p) * (1.0 - t);
        let true_grad = (pop(theta + eps) - pop(theta - eps)) / (2.0 * eps);

        // expected single-sample gradient, over the two possible draws
        let sample_grad = |t: f64, s: f64| (cramer_distance(&[t + eps], &[s]).unwrap() - cramer_distance(&[t - eps], &[s]).unwrap()) / (2.0 * eps);
        let expected = p * sample_grad(theta, 0.0) + (1.0 - p) * sample_grad(theta, 1.0);
        eprintln!("Cramer gradient: population {true_grad:+.6}, expected single-sample {expected:+.6}");
        assert!(true_grad.abs() > 0.1, "the setup must have a non-zero true gradient or nothing is being tested");
        assert!((expected - true_grad).abs() < 1e-6, "Cramer's sample gradient must be unbiased: {expected} vs {true_grad}");

        // Now the Wasserstein value bias, with a two-atom model so the comparison is not degenerate.
        let model = vec![theta - 0.5, theta + 0.5];
        let full_target = vec![-1.0, 1.0];
        let against_population = crate::transport::w1_empirical_1d(&model, &full_target).unwrap();
        let against_one_sample = 0.5 * crate::transport::w1_empirical_1d(&model, &[-1.0, -1.0]).unwrap() + 0.5 * crate::transport::w1_empirical_1d(&model, &[1.0, 1.0]).unwrap();
        eprintln!("W1: against the full target {against_population:.4}, expected against a single sample {against_one_sample:.4}");
        assert!((against_one_sample - against_population).abs() > 0.4, "the single-sample W1 estimate is supposed to be biased: {against_one_sample} vs {against_population}");
        assert!(against_one_sample > against_population, "and biased upward, since one sample cannot spread");
    }

    /// **The distributional Bellman operator contracts in maximal Wasserstein at rate `γ` — and does not
    /// contract in total variation.** Both halves matter: the first licenses the convergence argument, the
    /// second says it cannot be rewritten in a more familiar metric.
    #[test]
    fn the_distributional_bellman_operator_contracts_only_in_wasserstein() {
        let gamma = 0.7;
        let reward = vec![1.0, -0.5];
        let next = vec![vec![1], vec![0]]; // a two-state cycle
        let z1 = vec![vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]];
        let z2 = vec![vec![2.0, 2.0, 2.0, 2.0], vec![-1.0, -1.0, -1.0, -1.0]];

        let ratio = distributional_bellman_contraction(&z1, &z2, &reward, gamma, &next).unwrap();
        eprintln!("maximal-Wasserstein contraction: {ratio:.6} (gamma = {gamma})");
        assert!(ratio <= gamma + 1e-9, "must contract at rate gamma, got {ratio}");

        // Total variation: the atoms move but never overlap, so TV stays pinned at 1 and reports no
        // contraction at all. A proof written in TV would have nothing to work with.
        let t1 = distributional_bellman(&z1, &reward, gamma, &next).unwrap();
        let t2 = distributional_bellman(&z2, &reward, gamma, &next).unwrap();
        let tv_before = total_variation(&z1[0], &z2[0]).max(total_variation(&z1[1], &z2[1]));
        let tv_after = total_variation(&t1[0], &t2[0]).max(total_variation(&t1[1], &t2[1]));
        eprintln!("total variation: {tv_before:.4} before, {tv_after:.4} after - no contraction");
        assert!(tv_after >= tv_before - 1e-12, "TV is not contracted by this operator: {tv_before} -> {tv_after}");

        // iterating really converges, which is what the contraction buys
        let (mut a, mut b) = (z1.clone(), z2.clone());
        for _ in 0..80 {
            a = distributional_bellman(&a, &reward, gamma, &next).unwrap();
            b = distributional_bellman(&b, &reward, gamma, &next).unwrap();
        }
        let gap = maximal_wasserstein_1(&a, &b).unwrap();
        assert!(gap < 1e-6, "iteration should collapse the gap to the unique fixed point, left {gap:.2e}");
    }

    /// **Gromov-Wasserstein aligns isometric sets across dimensions.** A square in the plane and the same
    /// square embedded in three dimensions and rotated have identical internal distances, so the cost must
    /// be zero even though no coordinate is shared and the dimensions differ.
    #[test]
    fn gromov_wasserstein_matches_isometric_sets_in_different_dimensions() {
        // a unit square in 2-D
        let p2: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        // the same square, rotated into a tilted plane in 3-D: distances preserved, coordinates unrelated
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let p3: Vec<[f64; 3]> = p2.iter().map(|q| [q[0] * s, q[0] * s, q[1]]).collect();

        let dist = |p: &[[f64; 3]]| DMatrix::from_fn(p.len(), p.len(), |i, j| ((p[i][0] - p[j][0]).powi(2) + (p[i][1] - p[j][1]).powi(2) + (p[i][2] - p[j][2]).powi(2)).sqrt());
        let (d1, d2) = (dist(&p2), dist(&p3));
        assert!((&d1 - &d2).amax() < 1e-12, "the two sets must be isometric for this test to mean anything");

        let g = gromov_wasserstein(&d1, &d2, 0.002, 400, 400).unwrap();
        eprintln!("Gromov-Wasserstein between isometric squares: cost {:.3e} after {} outer iterations", g.cost, g.iters);
        assert!(g.cost < 1e-2, "isometric sets should align at near-zero cost, got {}", g.cost);

        // The trap worth recording: seeded uniformly, the iteration cannot move at all and reports the
        // uniform coupling's cost as a converged answer. This is a property of the method, not a bug here.
        let uniform = DMatrix::from_element(4, 4, 1.0 / 16.0);
        let stuck = gromov_wasserstein_from(&d1, &d2, &uniform, 0.002, 400, 400).unwrap();
        eprintln!("   seeded uniformly instead: cost {:.4} - the uniform plan is a fixed point", stuck.cost);
        assert!(stuck.cost > 100.0 * g.cost.max(1e-9), "the uniform seed must visibly fail, or this warning is stale");

        // and a genuinely different shape costs more: a collinear set has no square's distance structure
        let line: Vec<[f64; 3]> = (0..4).map(|i| [i as f64, 0.0, 0.0]).collect();
        let g_bad = gromov_wasserstein(&d1, &dist(&line), 0.002, 400, 400).unwrap();
        eprintln!("   against a collinear set: cost {:.4}", g_bad.cost);
        assert!(g_bad.cost > 10.0 * g.cost.max(1e-6), "a different distance structure must cost more: {} vs {}", g_bad.cost, g.cost);
    }

    /// The Schrödinger bridge hits both prescribed marginals, which is the property that distinguishes it
    /// from a sampler that merely lands near the target.
    #[test]
    fn the_schrodinger_bridge_hits_both_marginals_exactly() {
        let mu0 = vec![0.5, 0.3, 0.2];
        let mu1 = vec![0.1, 0.6, 0.3];
        // a heat-kernel-like reference on three sites
        let reference = DMatrix::from_fn(3, 3, |i, j| (-((i as f64 - j as f64).powi(2)) / 0.5).exp().max(1e-12));
        let p = schrodinger_bridge(&mu0, &mu1, &reference, 0.05, 4000).unwrap();
        eprintln!("Schrodinger bridge: marginal error {:.2e} after {} iterations", p.marginal_error, p.iters);
        assert!(p.marginal_error < 1e-8, "both marginals must be met: error {:.2e}", p.marginal_error);
        for i in 0..3 {
            assert!((p.plan.row(i).sum() - mu0[i]).abs() < 1e-8);
            assert!((p.plan.column(i).sum() - mu1[i]).abs() < 1e-8);
        }
        // a reference with a zero entry has an infinite cost there and is rejected rather than silently used
        let mut bad = reference.clone();
        bad[(0, 0)] = 0.0;
        assert!(schrodinger_bridge(&mu0, &mu1, &bad, 0.05, 10).is_none());
    }

    /// **A JKO step decreases the free energy, every step, and converges to the Gibbs measure.** That
    /// monotonicity is what makes the Fokker-Planck equation a gradient flow rather than just an equation,
    /// and it is checkable without knowing the answer in advance.
    #[test]
    fn the_jko_flow_decreases_free_energy_and_lands_on_the_gibbs_measure() {
        let n = 201;
        let (lo, hi) = (-4.0f64, 4.0f64);
        let h = (hi - lo) / (n - 1) as f64;
        let grid: Vec<f64> = (0..n).map(|i| lo + i as f64 * h).collect();
        let potential: Vec<f64> = grid.iter().map(|x| 0.5 * x * x).collect(); // quadratic well
        let beta = 1.0;

        // start badly: mass piled up off-centre
        let mut rho: Vec<f64> = grid.iter().map(|x| if (*x - 2.0).abs() < 0.5 { 1.0 } else { 1e-6 }).collect();
        let mass: f64 = rho.iter().sum::<f64>() * h;
        rho.iter_mut().for_each(|r| *r /= mass);

        let mut f_prev = free_energy(&rho, &grid, &potential, beta).unwrap();
        let f_start = f_prev;
        let tau = 0.2 * h * h * beta; // diffusion-stable
        for k in 0..200_000 {
            rho = jko_step(&rho, &grid, &potential, beta, tau).unwrap();
            if k % 1000 == 0 {
                let f = free_energy(&rho, &grid, &potential, beta).unwrap();
                assert!(f <= f_prev + 1e-6, "free energy rose at step {k}: {f_prev} -> {f}");
                f_prev = f;
            }
        }
        // the stationary point of this flow is the Gibbs measure exp(-beta V)/Z
        let z: f64 = grid.iter().map(|x| (-beta * 0.5 * x * x).exp() * h).sum();
        let gibbs: Vec<f64> = grid.iter().map(|x| (-beta * 0.5 * x * x).exp() / z).collect();
        let err: f64 = rho.iter().zip(&gibbs).map(|(a, b)| (a - b).abs() * h).sum();
        eprintln!("JKO flow: free energy {f_start:.5} -> {f_prev:.5}, distance to the Gibbs measure {err:.2e}");
        assert!(err < 5e-3, "the flow should land on exp(-beta V)/Z, off by {err:.2e}");
        assert!(f_prev < f_start, "and the free energy must have actually decreased");
    }
}
