//! **What a trained score buys you** — the bounds that convert a network's training error into a distance
//! between the sampled action law and the one you wanted.
//!
//! These are the raw material for any closed-loop statement about a diffusion or flow-matching policy, and
//! they all have the same shape: an **additive** decomposition into initialisation, discretisation, and
//! network error, with the network error entering **linearly**.
//!
//! That last word is the load-bearing one, and it is worth stating why. The pessimistic prior for a learned
//! component inside a dynamical system is exponential amplification, and for a *closed loop* that prior is
//! correct — see [`iss`](../../ferromotion_control/iss/index.html) for the theorem and the escapes. But the
//! sampler is a different object. Its error is amplified only by the **diffusion** horizon `T`, and only as
//! `√T`. Sampling error and closed-loop error compound in genuinely different ways, and conflating them is
//! how a policy gets blamed for a control problem or vice versa.
//!
//! * [`score_to_tv`] — Chen et al.: `TV ≲ √KL·e^{−T} + (L√d + L m₂)√(Th) + ε_score √T`.
//! * [`flow_matching_w2`] — Benton et al.: velocity `L²` error times a Grönwall factor `e^{∫Lip(v)}`. Note
//!   what is *not* benign here: the bound is exponential in the true field's Lipschitz constant. A stiff
//!   target costs exponentially, which is a statement about the data and not about the network.
//! * [`minimax_rate`] — Oko et al.: `n^{−s/(2s+d)}`, with `d` the **intrinsic** dimension of the data
//!   manifold, not the ambient one. This is why action-space dimension is much less punishing than it looks.
//! * [`consistency_error`] — Dou et al.: few-step generation converges to the ODE solution as steps refine.
//!
//! The test module does not restate these. It integrates a probability-flow ODE with a *deliberately
//! corrupted* score, measures the resulting law's actual `W₂` to the target, and checks the growth really is
//! linear in the injected error and really does scale with the diffusion horizon rather than exploding in it.

/// The three additive terms of a sampler error bound, kept separate because they are reduced by three
/// different things — longer forward noising, finer steps, and better training.
#[derive(Clone, Copy, Debug)]
pub struct SamplerError {
    /// Failure of the forward process to reach the reference Gaussian: `√KL · e^{−T}`. Reduced by noising
    /// for longer.
    pub mixing: f64,
    /// Discretisation of the reverse process: `(L√d + L m₂)√(Th)`. Reduced by more sampling steps.
    pub discretization: f64,
    /// The network's own error: `ε_score √T`. Reduced by training, and **linear** in the score error.
    pub score: f64,
}

impl SamplerError {
    pub fn total(&self) -> f64 {
        self.mixing + self.discretization + self.score
    }

    /// Which term dominates, as a label. Useful because the three call for completely different work and the
    /// usual instinct — train more — is often the wrong one.
    pub fn dominant(&self) -> &'static str {
        if self.mixing >= self.discretization && self.mixing >= self.score {
            "mixing: noise for longer"
        } else if self.discretization >= self.score {
            "discretisation: take more sampling steps"
        } else {
            "score error: train better"
        }
    }
}

/// **Score error to total variation** (Chen-Chewi-Li-Li-Salim-Zhang). `kl_init` is `KL(p₀‖γ_d)`, `t_diffusion`
/// the forward horizon `T`, `step` the reverse step size `h`, `lipschitz` the score's Lipschitz constant `L`,
/// `dim` the ambient dimension, `second_moment` the target's `m₂`, and `eps_score` the `L²` score error.
///
/// Needs no log-concavity and no functional inequality — only a Lipschitz score, a finite second moment, and
/// `L²` accuracy. That is what makes it applicable to a real action distribution rather than a convenient one.
pub fn score_to_tv(kl_init: f64, t_diffusion: f64, step: f64, lipschitz: f64, dim: usize, second_moment: f64, eps_score: f64) -> Option<SamplerError> {
    if kl_init < 0.0 || t_diffusion <= 0.0 || step <= 0.0 || lipschitz < 0.0 || second_moment < 0.0 || eps_score < 0.0 {
        return None;
    }
    Some(SamplerError {
        mixing: kl_init.sqrt() * (-t_diffusion).exp(),
        discretization: (lipschitz * (dim as f64).sqrt() + lipschitz * second_moment) * (t_diffusion * step).sqrt(),
        score: eps_score * t_diffusion.sqrt(),
    })
}

/// **Flow-matching Wasserstein bound** (Benton-Deligiannidis-Doucet): `W₂ ≲ ‖v̂ − v‖_{L²} · e^{∫Lip(v)}`.
///
/// Two things to read off it. The velocity error enters linearly, as in the score bounds — good. But the
/// Grönwall factor is **exponential in the true velocity field's Lipschitz constant**, so a sharply-peaked or
/// nearly-deterministic action distribution is expensive to hit regardless of how well the network fits. That
/// exponential is a property of the target, and no amount of training touches it.
///
/// Unlike the total-variation bounds, this one holds for data **without full support** — the
/// manifold-supported case, which is what an action distribution concentrated on a feasible set actually is.
pub fn flow_matching_w2(velocity_l2_error: f64, lipschitz_integral: f64) -> Option<f64> {
    if velocity_l2_error < 0.0 || lipschitz_integral < 0.0 {
        return None;
    }
    Some(velocity_l2_error * lipschitz_integral.exp())
}

/// **Minimax estimation rate** (Oko-Akiyama-Suzuki): `n^{−s/(2s+d)}` for a target of smoothness `s` in
/// `intrinsic_dim` dimensions, up to logarithmic factors.
///
/// The point is which `d` appears. The rate adapts to the **intrinsic** dimension of the data manifold, so a
/// 30-dimensional action vector whose reachable set is a 4-dimensional manifold estimates at the
/// 4-dimensional rate. That is the formal reason high-dimensional action spaces are not hopeless, and it is
/// also the reason the effective sample complexity cannot be read off the action dimension.
pub fn minimax_rate(n_samples: usize, smoothness: f64, intrinsic_dim: usize) -> Option<f64> {
    if n_samples == 0 || smoothness <= 0.0 || intrinsic_dim == 0 {
        return None;
    }
    Some((n_samples as f64).powf(-smoothness / (2.0 * smoothness + intrinsic_dim as f64)))
}

/// **Consistency-model error** (Dou et al.): a training term plus a step-discretisation term, so few-step
/// generation converges to the underlying ODE solution as the discretisation refines. `steps` is the number
/// of generation steps.
pub fn consistency_error(training_error: f64, steps: usize) -> Option<f64> {
    if training_error < 0.0 || steps == 0 {
        return None;
    }
    Some(training_error + 1.0 / steps as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::w2_gaussian;
    use nalgebra::{DMatrix, DVector};

    /// **The empirical check on the load-bearing claim.**
    ///
    /// Run an actual probability-flow ODE with a corrupted score and measure how far the sampled law lands
    /// from the target. Setup: forward Ornstein-Uhlenbeck `dx = −x dt + √2 dw`, whose stationary law is
    /// `N(0,1)`; a Gaussian target, so every intermediate law is Gaussian with a known score `−x/v(t)`; and
    /// the reverse ODE `ẋ = −x − s(x,t)` integrated from `T` to `0`.
    ///
    /// Corrupting the score by a constant `ε` keeps the law Gaussian, so the sampled mean and variance can be
    /// propagated exactly and the distance read from the closed-form `W₂`. The claim under test is that the
    /// error grows **linearly** in `ε` — not exponentially — which is what makes "reduce the training loss" a
    /// well-posed instruction.
    #[test]
    fn sampler_error_grows_linearly_in_the_score_error() {
        let target_var = 0.25f64;
        let t_diffusion = 3.0f64;
        let steps = 30_000;
        let dt = t_diffusion / steps as f64;

        // variance of the forward law at time t, starting from the target
        let v = |t: f64| 1.0 + (target_var - 1.0) * (-2.0 * t).exp();

        // Reverse-integrate the probability-flow ODE for a Gaussian, tracking (mean, variance). With the true
        // score the result is the target exactly; with a constant offset `eps` added to the score the mean is
        // driven away while the variance is untouched.
        let sample_law = |eps: f64| {
            let (mut m, mut var) = (0.0f64, 1.0f64); // start from the reference N(0,1)
            for k in 0..steps {
                let t = t_diffusion - k as f64 * dt;
                let vt = v(t);
                // dx/dt = -x - s(x,t), s = -x/vt + eps  =>  dx/dt = -x + x/vt - eps, integrated backwards
                let a = -1.0 + 1.0 / vt;
                m -= (a * m - eps) * dt;
                var -= (2.0 * a * var) * dt;
            }
            (m, var.max(1e-12))
        };

        let w2_to_target = |(m, var): (f64, f64)| {
            w2_gaussian(&DVector::from_row_slice(&[m]), &DMatrix::from_row_slice(1, 1, &[var]), &DVector::from_row_slice(&[0.0]), &DMatrix::from_row_slice(1, 1, &[target_var])).unwrap()
        };

        // with the true score the sampler reproduces the target
        let exact = sample_law(0.0);
        let base = w2_to_target(exact);
        eprintln!("true score: sampled law N({:.6}, {:.6}), target variance {target_var}, W2 = {base:.2e}", exact.0, exact.1);
        assert!(base < 1e-3, "an exact score must reproduce the target, W2 = {base:.2e}");

        // The response is linear in the injected score error, but reading that off requires separating the
        // score contribution from the discretisation floor (`base` above) **in quadrature**, not by
        // subtraction. The reason is the structure of `w2_gaussian` itself: for Gaussians the mean term and
        // the covariance term enter as a sum of squares. Here the floor is a variance error and the injected
        // score error produces a mean error, so
        //
        //     W₂(eps)² = W₂(0)² + (k·eps)²
        //
        // and the score part is the square root of the difference. Subtracting linearly instead reads 1.218
        // at the smallest eps against 1.639 at the largest, and would look like a broken linearity claim.
        let mut ratios = Vec::new();
        for eps in [1e-3f64, 1e-2, 1e-1] {
            let w2 = w2_to_target(sample_law(eps));
            let score_part = (w2 * w2 - base * base).max(0.0).sqrt();
            ratios.push(score_part / eps);
            eprintln!("   score error {eps:>6.0e}: W2 = {w2:.6}, score part (in quadrature) = {score_part:.6}, ratio {:.4}", score_part / eps);
        }
        for r in &ratios {
            assert!((r - ratios[0]).abs() / ratios[0] < 0.02, "the response must be linear in the score error, ratios {ratios:?}");
        }
        // an exponential response would multiply the ratio by orders of magnitude across those decades
        assert!(ratios.last().unwrap() / ratios[0] < 1.05, "linear, not exponential: {ratios:?}");
    }

    /// The second structural claim: the amplification is set by the **diffusion** horizon and grows like
    /// `√T`, not like `e^T`. Measured on the same ODE by sweeping the horizon at a fixed score error.
    #[test]
    fn the_amplification_grows_with_the_diffusion_horizon_not_exponentially_in_it() {
        let target_var = 0.25f64;
        let eps = 1e-2f64;
        let v = |t: f64, tv: f64| 1.0 + (tv - 1.0) * (-2.0 * t).exp();

        let run = |t_diffusion: f64| {
            let steps = (t_diffusion * 10_000.0) as usize;
            let dt = t_diffusion / steps as f64;
            let (mut m, mut var) = (0.0f64, 1.0f64);
            for k in 0..steps {
                let t = t_diffusion - k as f64 * dt;
                let a = -1.0 + 1.0 / v(t, target_var);
                m -= (a * m - eps) * dt;
                var -= (2.0 * a * var) * dt;
            }
            m.abs()
        };

        let mut prev = 0.0;
        for &t in &[1.0f64, 2.0, 4.0, 8.0] {
            let dev = run(t);
            let per_sqrt_t = dev / t.sqrt();
            eprintln!("   diffusion horizon T = {t:>4}: mean deviation {dev:.5}, per sqrt(T) {per_sqrt_t:.5}");
            assert!(dev >= prev, "a longer horizon should not reduce the accumulated error");
            // the decisive contrast: e^T over T = 1 to 8 is a factor of 1100. This must be far smaller.
            assert!(dev < 20.0 * eps, "the amplification must stay polynomial, got {} at T = {t}", dev / eps);
            prev = dev;
        }
        let ratio = run(8.0) / run(1.0);
        eprintln!("   amplification from T=1 to T=8: {ratio:.3}x (exponential would be e^7 = {:.0}x)", 7.0f64.exp());
        assert!(ratio < 20.0, "growth in the diffusion horizon must be polynomial, got {ratio}x");
    }

    /// The additive decomposition is the useful part of the bound, because the three terms are reduced by
    /// three different actions. This checks each term responds only to its own lever, and that `dominant`
    /// points at the right work.
    #[test]
    fn the_three_error_terms_respond_to_their_own_levers() {
        let base = score_to_tv(2.0, 5.0, 1e-4, 1.0, 16, 1.0, 1e-2).unwrap();
        eprintln!("baseline: mixing {:.4e}, discretisation {:.4e}, score {:.4e} -> {}", base.mixing, base.discretization, base.score, base.dominant());

        // noising longer kills only the mixing term
        let longer = score_to_tv(2.0, 20.0, 1e-4, 1.0, 16, 1.0, 1e-2).unwrap();
        assert!(longer.mixing < base.mixing * 1e-6, "mixing decays exponentially in T");
        // finer steps kill only the discretisation term
        let finer = score_to_tv(2.0, 5.0, 1e-8, 1.0, 16, 1.0, 1e-2).unwrap();
        assert!((finer.discretization / base.discretization - 0.01).abs() < 1e-9, "discretisation scales as sqrt(h)");
        assert_eq!(finer.score, base.score, "a finer step does not change the score term");
        // training reduces the score term linearly, and only it
        let trained = score_to_tv(2.0, 5.0, 1e-4, 1.0, 16, 1.0, 1e-4).unwrap();
        assert!((trained.score / base.score - 0.01).abs() < 1e-12, "the score term is linear in eps_score");
        assert_eq!(trained.mixing, base.mixing);

        // and the label points at the binding term rather than the habitual one
        assert_eq!(score_to_tv(2.0, 5.0, 1e-12, 1.0, 16, 1.0, 1.0).unwrap().dominant(), "score error: train better");
        assert_eq!(score_to_tv(2.0, 5.0, 1e-2, 1.0, 16, 1.0, 1e-9).unwrap().dominant(), "discretisation: take more sampling steps");
        assert_eq!(score_to_tv(1e6, 0.1, 1e-12, 1e-6, 1, 1e-6, 1e-9).unwrap().dominant(), "mixing: noise for longer");
    }

    /// The flow-matching bound is linear in the network error and **exponential in the target's stiffness**.
    /// Both halves are worth having as numbers: the first says training helps proportionally, the second says
    /// a nearly-deterministic action distribution is intrinsically expensive.
    #[test]
    fn the_flow_matching_bound_is_linear_in_error_and_exponential_in_stiffness() {
        // linear in the velocity error at fixed stiffness
        let a = flow_matching_w2(1e-3, 2.0).unwrap();
        let b = flow_matching_w2(1e-2, 2.0).unwrap();
        assert!((b / a - 10.0).abs() < 1e-9, "linear in the velocity error");

        // exponential in the Lipschitz integral, which is a property of the data
        let stiffs: Vec<f64> = [1.0f64, 3.0, 6.0, 10.0].iter().map(|l| flow_matching_w2(1e-3, *l).unwrap()).collect();
        eprintln!("velocity error 1e-3 at Lipschitz integrals 1/3/6/10: {:.3e} {:.3e} {:.3e} {:.3e}", stiffs[0], stiffs[1], stiffs[2], stiffs[3]);
        assert!(stiffs[3] / stiffs[0] > 8000.0, "the Gronwall factor is exponential in stiffness: {}", stiffs[3] / stiffs[0]);
        // so no amount of training rescues a stiff target: a 1000x better network at stiffness 10 is still
        // worse than an untrained-by-comparison one at stiffness 1
        assert!(flow_matching_w2(1e-6, 10.0).unwrap() > flow_matching_w2(1e-3, 1.0).unwrap(), "stiffness dominates training");
    }

    /// The estimation rate follows the **intrinsic** dimension. A wide action vector on a thin manifold
    /// estimates at the manifold's rate, and the gap between the two readings is large enough to change
    /// whether a data budget looks feasible.
    #[test]
    fn the_minimax_rate_follows_the_intrinsic_dimension_not_the_ambient_one() {
        let (n, s) = (100_000usize, 2.0f64);
        let intrinsic = minimax_rate(n, s, 4).unwrap();
        let ambient = minimax_rate(n, s, 30).unwrap();
        eprintln!("n = {n}, smoothness {s}: rate at intrinsic dim 4 = {intrinsic:.5}, at ambient dim 30 = {ambient:.5} ({:.1}x apart)", intrinsic / ambient);
        assert!(intrinsic < ambient, "a thinner manifold estimates faster");
        assert!(ambient / intrinsic > 3.0, "and the gap is large enough to matter for a data budget");

        // more data always helps, and smoothness helps
        assert!(minimax_rate(10 * n, s, 4).unwrap() < intrinsic);
        assert!(minimax_rate(n, 2.0 * s, 4).unwrap() < intrinsic);
        // the sample count needed to match the intrinsic rate at the ambient dimension is the honest cost
        let needed = (intrinsic.ln() / (-s / (2.0 * s + 30.0))).exp();
        eprintln!("   to reach the same accuracy at ambient dim 30 would need about {needed:.3e} samples");
        assert!(needed > 1e8, "the ambient reading is a much larger budget: {needed:.2e}");
    }

    /// Few-step generation converges to the ODE solution as the step count grows, and is floored by the
    /// training error — so distillation to one step is bounded by how well the consistency was trained, not by
    /// the step count.
    #[test]
    fn consistency_error_is_floored_by_training_and_falls_with_steps() {
        let train = 1e-3;
        let mut prev = f64::INFINITY;
        for steps in [1usize, 2, 4, 16, 256, 1_000_000] {
            let e = consistency_error(train, steps).unwrap();
            assert!(e < prev, "more steps must not be worse");
            assert!(e > train, "and the training error is a floor");
            prev = e;
        }
        assert!((prev - train) / train < 0.01, "at many steps the error is the training error: {prev}");
        assert!(consistency_error(train, 0).is_none());
    }
}
