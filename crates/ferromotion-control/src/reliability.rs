//! **Compositional reliability** — why a 100-step task built from 99%-reliable skills fails most of the time, and
//! what actually fixes it.
//!
//! Chain `n` independent skills and the task succeeds with probability `∏pᵢ`. At `p = 0.99` and `n = 100` that is
//! **37%**, which is the arithmetic behind every demo that does not survive being made longer. The obvious
//! response is to improve the skills, and it is close to the worst available use of effort.
//!
//! Add failure **detection** with probability `d` and a recovery routine that returns the system to a retryable
//! state. The chain becomes an absorbing Markov chain: a detected failure costs *time* and an undetected one costs
//! the *task*. At each step the outcome is a three-way race — pass with `p`, retry with `(1−p)d`, die with
//! `(1−p)(1−d)` — and because retries loop back to the same step, the probability of eventually passing it is a
//! geometric sum with a closed form:
//!
//! ```text
//! P(pass step i) = pᵢ / (pᵢ + (1−pᵢ)(1−dᵢ))
//! P(task)        = ∏ᵢ pᵢ / (pᵢ + (1−pᵢ)(1−dᵢ))
//! ```
//!
//! Read the two limits. At `d = 0` this is the naive product. At `d = 1` **every factor is exactly 1**: with
//! perfect detection a chain of arbitrarily unreliable skills completes with probability one, paying only in time.
//! Detection does not improve the skills; it converts a fatal error into an expensive one, and that is a different
//! kind of change than making `p` larger.
//!
//! The consequence is quantitative and it is the point of this module: **detection quality dominates skill
//! success at long horizons**, by margins large enough to redirect where effort goes.
//! [`detection_equivalent_skill`] states the exchange rate.
//!
//! # What this does not paper over
//!
//! The closed form assumes failures are independent and that recovery returns the system to a genuinely retryable
//! state. [`with_common_cause`] adds a shared failure mode, and it caps task reliability at `1 − q` **no matter
//! how good the detection is** — because a common cause is not something a per-step retry can retry its way out
//! of. That cap is the honest limit on the whole calculus, and it is the reason "just add recovery" is not a
//! strategy on its own.

/// A skill's measured interface: how often it works, how often its failures are noticed, and what a retry costs.
#[derive(Clone, Copy, Debug)]
pub struct Skill {
    /// Probability the skill succeeds on an attempt.
    pub p_success: f64,
    /// Probability a failure is **detected**. This is the quantity the arithmetic says dominates, and the one
    /// robot pipelines least often measure.
    pub p_detect: f64,
    /// Time cost of one attempt, in whatever unit the task budget is in.
    pub attempt_cost: f64,
}

impl Skill {
    pub fn new(p_success: f64, p_detect: f64, attempt_cost: f64) -> Option<Skill> {
        ((0.0..=1.0).contains(&p_success) && (0.0..=1.0).contains(&p_detect) && attempt_cost >= 0.0).then_some(Skill { p_success, p_detect, attempt_cost })
    }

    /// Probability of eventually passing this step, allowing unlimited detected-and-recovered retries:
    /// `p / (p + (1−p)(1−d))`.
    pub fn pass_probability(&self) -> f64 {
        let fatal = (1.0 - self.p_success) * (1.0 - self.p_detect);
        let denom = self.p_success + fatal;
        if denom <= 0.0 { 0.0 } else { self.p_success / denom }
    }

    /// Expected attempts spent on this step, conditional on reaching it: `1 / (p + (1−p)(1−d))`.
    ///
    /// Note this counts attempts on *both* the runs that pass and the runs that die here, which is what a time
    /// budget actually experiences.
    pub fn expected_attempts(&self) -> f64 {
        let denom = self.p_success + (1.0 - self.p_success) * (1.0 - self.p_detect);
        if denom <= 0.0 { f64::INFINITY } else { 1.0 / denom }
    }
}

/// The outcome of composing a chain of skills.
#[derive(Clone, Copy, Debug)]
pub struct ChainOutcome {
    /// Probability the whole chain completes.
    pub success: f64,
    /// Probability the chain completes with **no** detection anywhere — the naive `∏pᵢ`, for comparison.
    pub success_without_detection: f64,
    /// Expected total time, counting retries, over the runs that reach each step.
    pub expected_cost: f64,
    pub steps: usize,
}

/// Compose a chain of skills, with detection and recovery as first-class operators.
pub fn compose(skills: &[Skill]) -> ChainOutcome {
    let mut success = 1.0;
    let mut naive = 1.0;
    let mut cost = 0.0;
    let mut reach = 1.0; // probability of reaching this step at all
    for s in skills {
        cost += reach * s.expected_attempts() * s.attempt_cost;
        let pass = s.pass_probability();
        success *= pass;
        naive *= s.p_success;
        reach *= pass;
    }
    ChainOutcome { success, success_without_detection: naive, expected_cost: cost, steps: skills.len() }
}

/// A uniform chain: `n` copies of the same skill.
pub fn uniform_chain(n: usize, p_success: f64, p_detect: f64, attempt_cost: f64) -> Option<ChainOutcome> {
    let s = Skill::new(p_success, p_detect, attempt_cost)?;
    Some(compose(&vec![s; n]))
}

/// **The exchange rate between detection and skill.**
///
/// Given a chain of `n` skills at success `p` with detection `d`, returns the success probability `p'` an
/// undetected chain would need to match it. The ratio of failure rates `(1−p)/(1−p')` is how many times better the
/// skills would have to get to substitute for the detection — and at long horizons it is large.
///
/// `None` if the detected chain's reliability is unattainable without detection (which happens whenever `d` is
/// high enough that `p'` would have to exceed one).
pub fn detection_equivalent_skill(n: usize, p: f64, d: f64) -> Option<f64> {
    if n == 0 || !(0.0..=1.0).contains(&p) || !(0.0..=1.0).contains(&d) {
        return None;
    }
    let target = Skill::new(p, d, 0.0)?.pass_probability().powi(n as i32);
    // an undetected chain of n skills at p' has reliability p'^n
    let equivalent = target.powf(1.0 / n as f64);
    (equivalent < 1.0).then_some(equivalent)
}

/// The **skill success needed to hit a task-reliability target**, at a given detection quality and horizon.
///
/// This is the design question in the direction it is usually asked: "we need 99% on a 100-step task, what does
/// each skill have to do?" `None` if no `p ≤ 1` suffices.
pub fn required_skill(n: usize, target: f64, d: f64) -> Option<f64> {
    if n == 0 || !(0.0..1.0).contains(&target) || !(0.0..=1.0).contains(&d) {
        return None;
    }
    let per_step = target.powf(1.0 / n as f64);
    // invert p/(p + (1-p)(1-d)) = per_step for p
    let k = 1.0 - d;
    if k <= 0.0 {
        return Some(0.0); // perfect detection: any positive skill suffices
    }
    // per_step·(p + (1−p)k) = p  ⟹  p(1 − per_step + per_step·k) = per_step·k
    let denom = 1.0 - per_step + per_step * k;
    if denom <= 0.0 {
        return None;
    }
    let p = per_step * k / denom;
    (p <= 1.0).then_some(p)
}

/// Add a **common-cause failure** of probability `q` to a chain outcome: a shared mode that defeats the whole task
/// at once, independent of per-step retries.
///
/// This is the honest cap on the calculus. Recovery retries a *step*; it cannot retry its way out of a cause that
/// applies to every step, so task reliability is bounded by `1 − q` however perfect the detection. A reliability
/// model without a common-cause term will happily promise numbers that shared failure modes make unreachable.
pub fn with_common_cause(outcome: &ChainOutcome, q: f64) -> Option<ChainOutcome> {
    if !(0.0..=1.0).contains(&q) {
        return None;
    }
    Some(ChainOutcome { success: outcome.success * (1.0 - q), success_without_detection: outcome.success_without_detection * (1.0 - q), ..*outcome })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Xorshift;

    /// **The headline arithmetic**, and the limits that make it interesting.
    #[test]
    fn a_hundred_step_chain_of_ninety_nine_percent_skills() {
        let naive = uniform_chain(100, 0.99, 0.0, 1.0).unwrap();
        assert!((naive.success - 0.99f64.powi(100)).abs() < 1e-12);
        eprintln!("100 steps at p = 0.99, no detection:      {:.4}", naive.success);
        assert!((naive.success - 0.366).abs() < 0.002, "the famous 37%: got {:.4}", naive.success);

        for d in [0.5f64, 0.9, 0.99, 0.999] {
            let c = uniform_chain(100, 0.99, d, 1.0).unwrap();
            eprintln!("100 steps at p = 0.99, detection {d:<6}: {:.4}   (expected cost {:.2} attempts)", c.success, c.expected_cost);
            assert!(c.success > naive.success, "detection must help");
        }
        // perfect detection makes every factor exactly one: the task completes, paying only in time
        let perfect = uniform_chain(100, 0.5, 1.0, 1.0).unwrap();
        eprintln!("100 steps at p = 0.50 with PERFECT detection: {:.6}, cost {:.1} attempts", perfect.success, perfect.expected_cost);
        assert!((perfect.success - 1.0).abs() < 1e-12, "with d = 1 a chain of coin flips still finishes");
        assert!(perfect.expected_cost > 150.0, "but it pays in time: {:.1} attempts for 100 steps", perfect.expected_cost);
    }

    /// **The exchange rate**: how much better the skills would have to be to substitute for detection. This is the
    /// number that says where effort should go.
    #[test]
    fn detection_is_worth_more_than_skill_at_a_long_horizon() {
        eprintln!("100 steps at p = 0.99. Skill needed to match each detection level WITHOUT detection:");
        let mut previous_gain = 1.0;
        for d in [0.5f64, 0.9, 0.99] {
            let eq = detection_equivalent_skill(100, 0.99, d).expect("attainable");
            let gain = 0.01 / (1.0 - eq); // how many times lower the failure rate must be
            eprintln!("      d = {d:<6} equivalent p' = {eq:.6}  -> failure rate must fall {gain:.1}x");
            assert!(gain > previous_gain, "more detection must be worth more skill");
            previous_gain = gain;
        }
        // the specific claim worth remembering
        let eq = detection_equivalent_skill(100, 0.99, 0.9).unwrap();
        assert!(eq > 0.998, "90% detection is worth roughly a 10x better skill: p' = {eq:.6}");

        // and past some detection level, no attainable skill substitutes at all
        assert!(detection_equivalent_skill(100, 0.99, 1.0).is_none(), "perfect detection cannot be bought with skill");
    }

    /// The design question in the direction it is asked: what must each skill do to hit a task target?
    #[test]
    fn the_required_skill_falls_steeply_with_detection_quality() {
        eprintln!("target 99% on a 100-step task. Required per-skill success, by detection quality:");
        let mut prev = 1.0;
        for d in [0.0f64, 0.5, 0.9, 0.99] {
            let p = required_skill(100, 0.99, d).expect("attainable");
            eprintln!("      d = {d:<6} requires p >= {p:.6}   (failure rate {:.2e})", 1.0 - p);
            assert!(p < prev, "better detection must relax the skill requirement");
            prev = p;
            // and the requirement is self-consistent
            let achieved = uniform_chain(100, p, d, 1.0).unwrap().success;
            assert!((achieved - 0.99).abs() < 1e-6, "the inverse must round-trip: {achieved} vs 0.99");
        }
        assert!(required_skill(100, 0.99, 1.0).unwrap() < 1e-12, "perfect detection needs no skill at all");
    }

    /// **The closed form against simulation.** The geometric-race argument is the whole basis of the module, so it
    /// is worth checking against an explicit Markov rollout rather than trusting the algebra.
    #[test]
    fn the_closed_form_matches_a_simulated_markov_chain() {
        let (n, p, d) = (20usize, 0.85f64, 0.7f64);
        let predicted = uniform_chain(n, p, d, 1.0).unwrap();
        let mut rng = Xorshift::new(0xBEEF_5EED_1234_9999);
        let mut u = || rng.uniform();
        let trials = 200_000;
        let mut wins = 0;
        let mut attempts_total = 0u64;
        for _ in 0..trials {
            let mut step = 0;
            let mut alive = true;
            let mut attempts = 0u64;
            while step < n && alive {
                attempts += 1;
                if u() < p {
                    step += 1;
                } else if u() >= d {
                    alive = false;
                }
            }
            attempts_total += attempts;
            if alive {
                wins += 1;
            }
        }
        let measured = wins as f64 / trials as f64;
        let rel = (measured - predicted.success).abs() / predicted.success;
        eprintln!("n = {n}, p = {p}, d = {d}: closed form {:.5}, simulated {measured:.5} over {trials} trials ({:.2}%)", predicted.success, 100.0 * rel);
        eprintln!("   expected attempts: closed form {:.2}, simulated {:.2}", predicted.expected_cost, attempts_total as f64 / trials as f64);
        assert!(rel < 0.05, "the closed form must match the chain: {} vs {measured}", predicted.success);
    }

    /// **The honest cap.** A common-cause failure bounds task reliability at `1 − q` whatever the detection, because
    /// recovery retries a step and a shared cause is not a step. Any reliability model without this term overpromises.
    #[test]
    fn a_common_cause_caps_reliability_however_good_the_detection() {
        let q = 0.02;
        eprintln!("a 2% common-cause failure caps the task at {:.4} regardless of detection:", 1.0 - q);
        for d in [0.0f64, 0.9, 0.99, 1.0] {
            let base = uniform_chain(100, 0.99, d, 1.0).unwrap();
            let capped = with_common_cause(&base, q).unwrap();
            eprintln!("      d = {d:<6} without the common cause {:.4}, with it {:.4}", base.success, capped.success);
            assert!(capped.success <= 1.0 - q + 1e-12, "the cap must hold at d = {d}");
        }
        // perfect detection reaches the cap exactly and cannot pass it
        let perfect = with_common_cause(&uniform_chain(100, 0.99, 1.0, 1.0).unwrap(), q).unwrap();
        assert!((perfect.success - (1.0 - q)).abs() < 1e-12, "with d = 1 the common cause IS the whole failure rate");
        // so a target above the cap is unreachable, and the calculus should be read as saying so
        assert!(1.0 - q < 0.99, "a 2% common cause makes a 99% task target impossible at any skill or detection");
    }
}
