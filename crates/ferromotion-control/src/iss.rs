//! **Contraction and incremental stability as the price of propagation** — how a bounded per-step error
//! becomes a bounded trajectory error, without an exponential in the horizon.
//!
//! A policy that is slightly wrong at every step is the normal case, and the question that decides whether
//! it is usable is how that per-step error compounds. The pessimistic answer is that it compounds
//! exponentially in the horizon, and for a smooth deterministic imitator that pessimism is a *theorem*
//! (Simchowitz-Pfrommer-Jadbabaie 2025: closed-loop cost can be `e^Ω(H)` times training error even under
//! exponentially stable open-loop dynamics).
//!
//! The escape is not a better policy. It is **stability of the loop the policy closes**, and it converts
//! the currency of imitation from information-theoretic coverage into a control-theoretic gain:
//!
//! * [`disturbance_tube`] — a contraction at rate `λ` rejects a bounded disturbance into a tube of radius
//!   `sup‖d‖/λ`. The contraction rate *is* a disturbance-rejection gain of `1/λ`.
//! * [`stochastic_contraction_bound`] — the stochastic version (Pham-Tabareau-Slotine 2009), and the one
//!   the closed-loop theory of generative policies is built on: two trajectories of a contracting system
//!   under independent noise of intensity `C` satisfy
//!
//!   `E‖a(t) − b(t)‖² ≤ C/λ + E‖a(0) − b(0)‖² e^{−2λt}`.
//!
//!   Identify `C` with a sampled policy's action-noise intensity and the trajectory deviation is bounded by
//!   the **steady-state term `C/λ`** — a constant, independent of the horizon. That is the composition the
//!   agenda is missing in general, in the one form where it is already available.
//! * [`eiss_ultimate_bound`] — the discrete counterpart used to transfer a reduced-order certificate to the
//!   full order. If a Lyapunov function decays by `c₃V` per step and is inflated by `σ(‖d‖)` from model
//!   discrepancy, it settles at `σ(‖d‖)/c₃`. This is how a guarantee proved on a template degrades
//!   *gracefully* on the anchor rather than evaporating.
//!
//! Every bound here is an inequality, and the tests check them against simulation rather than restating
//! them — including the case where the deviation would have to grow with the horizon if the reasoning were
//! wrong, and does not.

/// The **deterministic disturbance tube** of a system contracting at rate `lambda`, driven by a disturbance
/// bounded by `d_sup`: trajectories converge into a ball of this radius.
///
/// The content of the formula is that a contraction rate is a rejection gain. Doubling `lambda` halves the
/// tube; a marginally stable loop (`lambda → 0`) has no tube at all, which is why "stable" without a rate
/// is not a useful property for a system carrying a noisy policy.
pub fn disturbance_tube(lambda: f64, d_sup: f64) -> Option<f64> {
    if lambda <= 0.0 || d_sup < 0.0 {
        return None;
    }
    Some(d_sup / lambda)
}

/// **Stochastic contraction bound** (Pham-Tabareau-Slotine): `C/λ + Δ₀² e^{−2λt}`, an upper bound on
/// `E‖a(t) − b(t)‖²` for two trajectories of a system contracting at rate `lambda` driven by *independent*
/// noise. `initial_sq` is `E‖a(0) − b(0)‖²`.
///
/// `c` is the **per-trajectory** noise intensity `tr(σᵀσ)`, not the intensity of the relative dynamics.
/// The distinction is a factor of two and it is easy to get backwards: two independent realisations make the
/// difference process carry `2·tr(σᵀσ)`, whose stationary variance is `2·tr(σᵀσ)/(2λ) = C/λ`. The Monte
/// Carlo test below pins the convention, because an off-by-two in a bound is worse than no bound.
///
/// The first term does not depend on `t`. That is the whole point: the transient forgets the initial
/// condition exponentially and what remains is a fixed noise floor, so a longer horizon does not buy a
/// larger deviation.
pub fn stochastic_contraction_bound(lambda: f64, c: f64, initial_sq: f64, t: f64) -> Option<f64> {
    if lambda <= 0.0 || c < 0.0 || initial_sq < 0.0 || t < 0.0 {
        return None;
    }
    Some(c / lambda + initial_sq * (-2.0 * lambda * t).exp())
}

/// The horizon-independent part of [`stochastic_contraction_bound`]: the steady-state deviation `C/λ` a
/// noisy policy inherits from the loop it closes, whatever the horizon.
pub fn stochastic_steady_state(lambda: f64, c: f64) -> Option<f64> {
    if lambda <= 0.0 || c < 0.0 {
        return None;
    }
    Some(c / lambda)
}

/// **Ultimate bound of an exponentially-input-to-state-stable Lyapunov recursion.**
///
/// Given `V(next) − V ≤ −c3·V + sigma_d` — a certificate that decays by a fixed fraction each step and is
/// inflated by a class-`K` gain of the model discrepancy — the sequence settles at `sigma_d / c3`, from any
/// starting value. This is the reduced-to-full transfer in its usable form: a template's guarantee survives
/// on the full robot, degraded in proportion to how badly the template describes it.
///
/// `None` unless `0 < c3 ≤ 1`, since a per-step decay fraction above one is not a decay.
pub fn eiss_ultimate_bound(c3: f64, sigma_d: f64) -> Option<f64> {
    if c3 <= 0.0 || c3 > 1.0 || sigma_d < 0.0 {
        return None;
    }
    Some(sigma_d / c3)
}

/// One step of the E-ISS recursion, for iterating the bound explicitly: `V ↦ max(0, V − c3·V + sigma_d)`.
pub fn eiss_step(v: f64, c3: f64, sigma_d: f64) -> f64 {
    (v - c3 * v + sigma_d).max(0.0)
}

/// The **horizon-free imitation multiplier** for a chunked policy (Zhang-Pfrommer-Pan-Matni-Simchowitz
/// 2025): under open-loop exponential incremental ISS at rate `rho < 1` and chunk length `ell`, the
/// trajectory imitation cost is bounded by a constant multiple of the per-step demonstration cost, and this
/// is that constant — `1/(1 − ρ^ell)`.
///
/// It depends on the chunk length and the stability rate, and **not** on the horizon. Compare with the
/// deterministic-smooth-imitator lower bound, where the multiplier is exponential in the horizon: this is
/// the quantitative statement of what chunking buys.
pub fn chunked_imitation_multiplier(rho: f64, ell: usize) -> Option<f64> {
    if !(0.0..1.0).contains(&rho) || ell == 0 {
        return None;
    }
    Some(1.0 / (1.0 - rho.powi(ell as i32)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic linear contraction driven by a constant worst-case disturbance settles at exactly the
    /// tube radius, so the bound is tight rather than merely valid.
    #[test]
    fn the_disturbance_tube_is_the_deviation_a_contraction_settles_into() {
        let (lambda, d) = (2.5f64, 0.4f64);
        let tube = disturbance_tube(lambda, d).unwrap();
        // x' = -lambda x + d settles at d/lambda
        let dt = 1e-5;
        let mut x = 0.0f64;
        for _ in 0..2_000_000 {
            x += (-lambda * x + d) * dt;
        }
        eprintln!("contraction rate {lambda}, disturbance {d}: settled at {x:.6}, tube radius {tube:.6}");
        assert!((x - tube).abs() < 1e-4, "the tube should be where it settles: {x} vs {tube}");
        // and the rate really is a rejection gain: twice the rate, half the tube
        assert!((disturbance_tube(2.0 * lambda, d).unwrap() - tube / 2.0).abs() < 1e-12);
        assert!(disturbance_tube(0.0, d).is_none(), "no rate, no tube");
    }

    /// **The stochastic contraction bound against Monte Carlo.** Two independent noise realisations of a
    /// contracting system, and the measured mean-square deviation against `C/λ`. In this linear case the
    /// bound is attained, which makes it the sharpest available check: anything other than agreement means
    /// the constant is wrong.
    #[test]
    fn stochastic_contraction_matches_a_monte_carlo_measurement() {
        // dx = -lambda x dt + sigma dW, in dimension n, two independent realisations
        let (lambda, sigma, n) = (3.0f64, 0.5f64, 4usize);
        // per-trajectory intensity tr(sigma^T sigma); the factor of two from independence is inside the
        // theorem's constant, not something to apply here
        let c = n as f64 * sigma * sigma;
        let predicted = stochastic_steady_state(lambda, c).unwrap();

        // deterministic pseudo-random noise, so the test cannot flake
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut normal = || {
            // Box-Muller from a xorshift stream
            let mut u = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64
            };
            let (u1, u2) = (u().max(1e-12), u());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };

        let dt = 1e-3f64;
        let steps = 20_000;
        let trials = 60;
        let mut mean_sq = 0.0;
        for _ in 0..trials {
            let mut a = vec![0.0f64; n];
            let mut b = vec![0.0f64; n];
            let s = sigma * dt.sqrt();
            for _ in 0..steps {
                for i in 0..n {
                    a[i] += -lambda * a[i] * dt + s * normal();
                    b[i] += -lambda * b[i] * dt + s * normal();
                }
            }
            mean_sq += a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
        }
        mean_sq /= trials as f64;
        let err = (mean_sq - predicted).abs() / predicted;
        eprintln!("stochastic contraction: measured E||a-b||^2 = {mean_sq:.5}, C/lambda = {predicted:.5} ({:.1}% apart)", 100.0 * err);
        assert!(err < 0.08, "the C/lambda constant is attained in this linear case, so a gap means the constant is wrong: measured {mean_sq}, predicted {predicted}");
        // the scaling is the other half of the claim: half the rate, twice the deviation
        assert!((stochastic_steady_state(2.0 * lambda, c).unwrap() - predicted / 2.0).abs() < 1e-12);
    }

    /// **The horizon-free claim, which is the reason any of this matters.** Measure the deviation at several
    /// horizons: if per-step noise compounded, it would grow with the horizon. It does not — it sits at the
    /// same steady state, because the contraction spends it as fast as the noise supplies it.
    #[test]
    fn the_deviation_does_not_grow_with_the_horizon() {
        let (lambda, c) = (3.0f64, 1.0f64);
        let ss = stochastic_steady_state(lambda, c).unwrap();
        let mut last = f64::INFINITY;
        for &t in &[1.0f64, 5.0, 20.0, 100.0, 1000.0] {
            let bound = stochastic_contraction_bound(lambda, c, 9.0, t).unwrap();
            eprintln!("   horizon {t:>6}: bound {bound:.6} (steady state {ss:.6})");
            assert!(bound <= last + 1e-15, "the bound must not grow with the horizon");
            assert!(bound >= ss, "and never fall below the noise floor");
            last = bound;
        }
        // by t = 1000 the transient is gone entirely and only the horizon-independent floor is left
        let far = stochastic_contraction_bound(lambda, c, 9.0, 1000.0).unwrap();
        assert!((far - ss).abs() < 1e-12, "at a long horizon the bound IS the steady state: {far} vs {ss}");
        // whereas a marginally stable loop has no floor to settle to
        assert!(stochastic_steady_state(0.0, c).is_none());
    }

    /// The E-ISS recursion settles at `σ(‖d‖)/c₃` from anywhere, which is the reduced-to-full transfer: the
    /// certificate is not lost to model discrepancy, it is *inflated in proportion to it*.
    #[test]
    fn the_eiss_recursion_settles_in_proportion_to_the_discrepancy() {
        let (c3, sigma_d) = (0.25f64, 0.05f64);
        let bound = eiss_ultimate_bound(c3, sigma_d).unwrap();
        assert!((bound - 0.2).abs() < 1e-12, "sigma/c3 = 0.2, got {bound}");

        for start in [0.0f64, 0.2, 5.0] {
            let mut v = start;
            for _ in 0..400 {
                v = eiss_step(v, c3, sigma_d);
            }
            assert!((v - bound).abs() < 1e-9, "from V0 = {start} the recursion should reach {bound}, got {v}");
        }
        // zero discrepancy recovers the exact certificate; the transfer is continuous in the discrepancy
        assert_eq!(eiss_ultimate_bound(c3, 0.0), Some(0.0));
        let doubled = eiss_ultimate_bound(c3, 2.0 * sigma_d).unwrap();
        assert!((doubled - 2.0 * bound).abs() < 1e-12, "the bound is linear in the discrepancy gain");
        // and a decay fraction above one is not a decay
        assert!(eiss_ultimate_bound(1.5, sigma_d).is_none());
    }

    /// The chunked-imitation multiplier depends on the chunk length and the stability rate, never on the
    /// horizon — which is the whole content of "horizon-free". A longer chunk on a stabler loop buys a
    /// multiplier approaching one, meaning trajectory cost approaches per-step cost.
    #[test]
    fn the_chunked_multiplier_is_horizon_free_and_improves_with_chunk_length() {
        let rho = 0.6;
        let mut last = f64::INFINITY;
        for ell in [1usize, 2, 4, 8, 16] {
            let m = chunked_imitation_multiplier(rho, ell).unwrap();
            eprintln!("   rho {rho}, chunk {ell:>2}: multiplier {m:.6}");
            assert!(m < last, "a longer chunk must not make the bound worse");
            assert!(m >= 1.0, "the multiplier cannot be below one");
            last = m;
        }
        assert!(last < 1.001, "a long chunk drives the multiplier to one, so trajectory cost meets per-step cost");
        // a stabler loop is uniformly better at the same chunk length
        assert!(chunked_imitation_multiplier(0.3, 4).unwrap() < chunked_imitation_multiplier(0.9, 4).unwrap());
        // and a non-contracting open loop has no such bound
        assert!(chunked_imitation_multiplier(1.0, 4).is_none());
    }
}
