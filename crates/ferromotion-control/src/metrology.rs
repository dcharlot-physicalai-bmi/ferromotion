//! **Reliability metrology** — what it costs to *measure* the quantity that decides long-horizon reliability, and
//! why that cost rises as the robot gets better.
//!
//! [`reliability`](crate::reliability) shows that detection quality dominates skill success: 90% detection is worth
//! roughly a 10× lower skill failure rate, and hitting 99% on a 100-step task needs `p ≥ 0.9999` undetected against
//! `p ≥ 0.99005` at `d = 0.99`. So detection is the number to know. This module is about the awkward fact that
//! follows.
//!
//! **You only learn about a detector from failures.** Every trial where the skill succeeds tells you nothing about
//! whether a failure would have been caught. So the effective sample size for estimating `d` is not the number of
//! trials but the number of *failures*, which is `n(1−p)` — and that shrinks exactly as the skill improves. The
//! consequence, made precise by [`trials_to_certify_detection`]:
//!
//! ```text
//! trials ≈ z²·d(1−d) / (ε²·(1−p))
//! ```
//!
//! The `1/(1−p)` is the whole problem. **Improving the skill makes the dominant quantity harder to certify**, and
//! at `p = 0.999` a detector needs ten times the trials it needed at `p = 0.99` for the same confidence. A
//! programme that improves skills and reports task success will find its predictions getting *less* trustworthy
//! even as its numbers improve, because the interval on `d` is widening while nobody is looking at it.
//!
//! # What the interval is, and why Wilson
//!
//! [`wilson_interval`] rather than the textbook `p̂ ± z√(p̂(1−p̂)/n)`, because the normal approximation is exactly
//! wrong in the regime that matters here: few observations, and a proportion near one. At `9/9` successes it gives
//! the interval `[1, 1]` — a claim of certainty from nine samples. Wilson's interval stays inside `(0,1)`, keeps
//! roughly nominal coverage at small `n`, and is a closed form. The coverage test below measures that rather than
//! asserting it.

/// A two-sided confidence interval on a proportion.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
    /// Point estimate, `successes / n`.
    pub point: f64,
    /// The nominal confidence level, e.g. `0.95`.
    pub confidence: f64,
    pub samples: usize,
}

impl Interval {
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
    /// Whether a value lies inside, used by coverage checks.
    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }
}

/// The normal quantile `z` for a two-sided interval at the given confidence, by a rational approximation of the
/// inverse normal CDF (Acklam's, accurate to ~1e-9 over the useful range — far tighter than any interval needs).
pub fn z_for(confidence: f64) -> Option<f64> {
    // strictly inside (0, 1): a confidence of zero gives a zero-width interval, which is arithmetically fine and
    // not a usable input, and one of exactly one is unattainable
    if !(confidence > 0.0 && confidence < 1.0) {
        return None;
    }
    let p = 0.5 * (1.0 + confidence); // one-sided tail
    let (a, b) = (
        [-3.969_683_028_665_376e1, 2.209_460_984_245_205e2, -2.759_285_104_469_687e2, 1.383_577_518_672_69e2, -3.066_479_806_614_716e1, 2.506_628_277_459_239],
        [-5.447_609_879_822_406e1, 1.615_858_368_580_409e2, -1.556_989_798_598_866e2, 6.680_131_188_771_972e1, -1.328_068_155_288_572e1],
    );
    let (c, d) = (
        [-7.784_894_002_430_293e-3, -3.223_964_580_411_365e-1, -2.400_758_277_161_838, -2.549_732_539_343_734, 4.374_664_141_464_968, 2.938_163_982_698_783],
        [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416],
    );
    let plow = 0.02425;
    let x = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };
    Some(x)
}

/// **Wilson score interval** on a proportion: `successes` out of `n`, at the given confidence.
///
/// Preferred over the normal approximation because the regime here is small `n` and a proportion near one, which is
/// where the normal interval fails outright — it returns `[1, 1]` on `n/n` successes, claiming certainty from a
/// handful of samples. Wilson stays strictly inside `(0,1)` and keeps roughly nominal coverage.
pub fn wilson_interval(successes: usize, n: usize, confidence: f64) -> Option<Interval> {
    if n == 0 || successes > n {
        return None;
    }
    let z = z_for(confidence)?;
    let (nf, s) = (n as f64, successes as f64);
    let phat = s / nf;
    let denom = 1.0 + z * z / nf;
    let centre = (phat + z * z / (2.0 * nf)) / denom;
    let half = z * ((phat * (1.0 - phat) / nf + z * z / (4.0 * nf * nf)).sqrt()) / denom;
    Some(Interval { lo: (centre - half).max(0.0), hi: (centre + half).min(1.0), point: phat, confidence, samples: n })
}

/// **How many task trials are needed to certify a detection probability to `±precision`.**
///
/// Only failures carry information about a detector, so the effective sample size is `n(1−p)` and
///
/// ```text
/// n ≈ z²·d(1−d) / (precision²·(1−p))
/// ```
///
/// The `1/(1−p)` is the finding: the cost of certifying the dominant quantity is inversely proportional to the
/// skill's failure rate, so **it rises as the skill improves**. `None` for a perfect skill, where no failure is
/// ever observed and `d` is unmeasurable at any budget.
pub fn trials_to_certify_detection(p_success: f64, d: f64, precision: f64, confidence: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&p_success) || !(0.0..=1.0).contains(&d) || precision <= 0.0 {
        return None;
    }
    let z = z_for(confidence)?;
    let failure_rate = 1.0 - p_success;
    (failure_rate > 0.0).then(|| z * z * d * (1.0 - d) / (precision * precision * failure_rate))
}

/// A skill's measured interface with **intervals** rather than point estimates: what a finite experiment actually
/// licenses.
#[derive(Clone, Copy, Debug)]
pub struct MeasuredSkill {
    pub p_success: Interval,
    pub p_detect: Interval,
}

/// Task-level reliability as an **interval**, propagated from skill-level intervals through
/// `∏ p/(p + (1−p)(1−d))`.
///
/// The map is monotone increasing in both `p` and `d`, so the endpoints propagate directly — no sampling needed and
/// no linearisation to be wrong about. That monotonicity is worth stating because it is what makes an interval
/// answer available at all: a non-monotone map would need the joint distribution.
pub fn task_interval(skills: &[MeasuredSkill], n_repeats: usize) -> Option<(f64, f64)> {
    if skills.is_empty() || n_repeats == 0 {
        return None;
    }
    let factor = |p: f64, d: f64| {
        let denom = p + (1.0 - p) * (1.0 - d);
        if denom <= 0.0 { 0.0 } else { p / denom }
    };
    let mut lo = 1.0;
    let mut hi = 1.0;
    for s in skills {
        lo *= factor(s.p_success.lo, s.p_detect.lo).powi(n_repeats as i32);
        hi *= factor(s.p_success.hi, s.p_detect.hi).powi(n_repeats as i32);
    }
    Some((lo, hi))
}

/// Which measurement the task-level interval is **limited by**: widen each input to its own interval in turn and
/// see which one moves the task answer more.
///
/// This is the number that says where the next thousand trials should go, and the point of computing it is that the
/// answer is usually not the one being optimised.
pub fn dominant_uncertainty(skill: &MeasuredSkill, n_steps: usize) -> Option<(&'static str, f64, f64)> {
    let factor = |p: f64, d: f64| {
        let denom = p + (1.0 - p) * (1.0 - d);
        if denom <= 0.0 { 0.0 } else { p / denom }
    };
    let n = n_steps as i32;
    if n_steps == 0 {
        return None;
    }
    // vary p alone, holding d at its point estimate, and vice versa
    let from_p = factor(skill.p_success.hi, skill.p_detect.point).powi(n) - factor(skill.p_success.lo, skill.p_detect.point).powi(n);
    let from_d = factor(skill.p_success.point, skill.p_detect.hi).powi(n) - factor(skill.p_success.point, skill.p_detect.lo).powi(n);
    Some((if from_d >= from_p { "detection" } else { "skill success" }, from_p, from_d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{uniform_chain, Xorshift};

    /// The normal quantiles, against values that can be checked by eye.
    #[test]
    fn the_normal_quantiles_are_right() {
        for (conf, want) in [(0.90f64, 1.644_854), (0.95, 1.959_964), (0.99, 2.575_829)] {
            let z = z_for(conf).unwrap();
            assert!((z - want).abs() < 1e-5, "z for {conf} should be {want}, got {z}");
        }
        assert!(z_for(1.0).is_none() && z_for(0.0).is_none());
    }

    /// **Why Wilson and not the normal approximation.** On `9/9` successes the normal interval collapses to a point
    /// and claims certainty from nine samples; Wilson does not.
    #[test]
    fn wilson_does_not_claim_certainty_from_nine_samples() {
        let w = wilson_interval(9, 9, 0.95).unwrap();
        let naive_half = 1.959_964 * (1.0f64 * 0.0 / 9.0).sqrt(); // p̂(1−p̂) = 0 at 9/9
        eprintln!("9 of 9 successes: Wilson [{:.4}, {:.4}], normal approximation [1.0000, 1.0000] (half-width {naive_half})", w.lo, w.hi);
        assert_eq!(naive_half, 0.0, "the normal interval really is degenerate here");
        assert!(w.lo < 0.75 && w.hi >= 1.0 - 1e-12, "Wilson must stay honest about nine samples: [{}, {}]", w.lo, w.hi);
        // and it tightens properly with more data
        let big = wilson_interval(900, 900, 0.95).unwrap();
        assert!(big.lo > w.lo, "900 of 900 licenses a tighter claim than 9 of 9");
        assert!(big.width() < w.width());
    }

    /// **Coverage, measured.** A confidence interval's only real property is that it contains the truth about as
    /// often as it claims. Simulated across proportions and sample sizes, including the near-one regime where the
    /// normal interval fails.
    #[test]
    fn the_interval_covers_at_about_its_nominal_rate() {
        let mut rng = Xorshift::new(0x5EED_C0DE_F00D_1234);
        for &truth in &[0.5f64, 0.9, 0.99] {
            for &n in &[10usize, 50, 200] {
                let trials = 4000;
                let mut covered = 0;
                for _ in 0..trials {
                    let k = (0..n).filter(|_| rng.uniform() < truth).count();
                    if wilson_interval(k, n, 0.95).unwrap().contains(truth) {
                        covered += 1;
                    }
                }
                let rate = covered as f64 / trials as f64;
                eprintln!("truth {truth}, n = {n:>3}: coverage {:.3} (nominal 0.95)", rate);
                assert!(rate > 0.90, "coverage must not fall far below nominal: {rate:.3} at truth {truth}, n {n}");
            }
        }
    }

    /// **The finding: certifying detection gets HARDER as the skill improves.** The trial budget scales as
    /// `1/(1−p)`, so a tenfold better skill needs ten times the data to say the same thing about its detector.
    #[test]
    fn improving_the_skill_makes_detection_harder_to_certify() {
        eprintln!("trials needed to pin d = 0.9 to +/-0.05 at 95% confidence:");
        let mut previous = 0.0;
        for &p in &[0.9f64, 0.99, 0.999, 0.9999] {
            let n = trials_to_certify_detection(p, 0.9, 0.05, 0.95).unwrap();
            eprintln!("      skill p = {p:<8} needs {n:>12.0} trials   ({:>10.0} failures observed)", n * (1.0 - p));
            if previous > 0.0 {
                assert!((n / previous - 10.0).abs() < 0.5, "each decade of skill should cost a decade of trials: {n} vs {previous}");
            }
            previous = n;
        }
        // the number of FAILURES observed is what stays constant - that is the effective sample size
        let a = trials_to_certify_detection(0.9, 0.9, 0.05, 0.95).unwrap() * 0.1;
        let b = trials_to_certify_detection(0.999, 0.9, 0.05, 0.95).unwrap() * 0.001;
        assert!((a - b).abs() / a < 1e-9, "the failure count is the invariant: {a} vs {b}");
        // a perfect skill makes its detector unmeasurable at any budget
        assert!(trials_to_certify_detection(1.0, 0.9, 0.05, 0.95).is_none());
    }

    /// **The task-level interval, and which measurement limits it.** With a realistic experiment — many trials of a
    /// good skill, so plenty of data on `p` and little on `d` — the task answer is dominated by the detector's
    /// uncertainty, which is the opposite of where effort usually goes.
    #[test]
    fn the_task_interval_is_limited_by_the_detector_not_the_skill() {
        // 2000 trials of a 99% skill: 1980 successes, and only ~20 failures to learn detection from
        let p = wilson_interval(1980, 2000, 0.95).unwrap();
        let d = wilson_interval(18, 20, 0.95).unwrap();
        eprintln!("from 2000 trials: p in [{:.4}, {:.4}] (width {:.4}), d in [{:.4}, {:.4}] (width {:.4})", p.lo, p.hi, p.width(), d.lo, d.hi, d.width());
        assert!(d.width() > 5.0 * p.width(), "the detector is far less well pinned: {:.4} vs {:.4}", d.width(), p.width());

        let skill = MeasuredSkill { p_success: p, p_detect: d };
        let (lo, hi) = task_interval(&[skill], 100).unwrap();
        let point = uniform_chain(100, p.point, d.point, 1.0).unwrap().success;
        eprintln!("100-step task: point estimate {point:.4}, interval [{lo:.4}, {hi:.4}] (width {:.4})", hi - lo);
        assert!(lo < point && point < hi, "the point estimate must lie inside its own interval");
        assert!(hi - lo > 0.2, "and the interval is WIDE - a 2000-trial experiment does not pin a 100-step task");

        let (which, from_p, from_d) = dominant_uncertainty(&skill, 100).unwrap();
        eprintln!("task-interval contribution: skill {from_p:.4}, detection {from_d:.4}  ->  limited by {which}");
        assert_eq!(which, "detection", "the detector is the binding measurement");
        assert!(from_d > 2.0 * from_p, "and by a real margin: {from_d:.4} vs {from_p:.4}");
    }

    /// The interval propagation is sound at its endpoints, checked against direct evaluation — the monotonicity
    /// argument is what licenses propagating endpoints at all, so it is worth confirming rather than assuming.
    #[test]
    fn the_propagation_matches_direct_evaluation_at_the_endpoints() {
        let p = wilson_interval(95, 100, 0.95).unwrap();
        let d = wilson_interval(4, 5, 0.95).unwrap();
        let (lo, hi) = task_interval(&[MeasuredSkill { p_success: p, p_detect: d }], 10).unwrap();
        let direct_lo = uniform_chain(10, p.lo, d.lo, 1.0).unwrap().success;
        let direct_hi = uniform_chain(10, p.hi, d.hi, 1.0).unwrap().success;
        assert!((lo - direct_lo).abs() < 1e-12 && (hi - direct_hi).abs() < 1e-12, "endpoints must match direct evaluation");
        // monotone in both arguments, which is why endpoints suffice
        for (a, b) in [(0.9f64, 0.5f64), (0.95, 0.5), (0.9, 0.8)] {
            let base = uniform_chain(10, 0.9, 0.5, 1.0).unwrap().success;
            let moved = uniform_chain(10, a, b, 1.0).unwrap().success;
            assert!(moved >= base - 1e-12, "raising p or d must not lower task reliability");
        }
    }
}
