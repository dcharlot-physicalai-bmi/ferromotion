//! **Where to cut a long task into skills** — decomposition granularity as a certified design variable.
//!
//! A long-horizon task can be one monolithic skill or a hundred short ones. The usual justification for cutting it up
//! is semantic ("reach, then grasp, then lift"), and semantics gives no way to decide between ten skills and twenty.
//! This module derives the answer from the two certificates already in hand.
//!
//! Start with what decomposition does *not* buy. If the task fails at rate `q` per step and failures are independent,
//! then `k` skills of length `H/k` each succeed with probability `(1−q)^(H/k)`, and the product over the chain is
//! `(1−q)^H` for **every** `k`. Cutting a task up does not make it more likely to succeed. That is worth stating
//! plainly because it is the intuition most decomposition arguments quietly rely on.
//!
//! What decomposition buys is **cheaper recovery**. A detected failure inside skill `i` costs retrying skill `i`, not
//! the task, so with the retry calculus of [`reliability`](crate::reliability) finer cuts genuinely win: on a
//! 500-step task at `q = 0.002` with 90% detection, `k = 1` passes at 0.8531 and `k = 500` at 0.9048.
//!
//! What decomposition costs is **handoffs**. Each cut adds a funnel that has to be measured, and with a fixed rollout
//! budget `N` split across `k` skills each funnel gets `N/k` samples, so by [`sampled_set_escape_rate`] the
//! per-handoff escape rate grows about linearly in `k`. Recovery saturates and measurement cost does not, which puts
//! the optimum in the interior — and the optimum is set by the rollout budget, not by task semantics.
//!
//! A runtime containment monitor ([`set_aware_reliability`](crate::set_aware_reliability)) moves every breach out of
//! the fatal column and into the retry column. That shifts the optimum much finer — but not to infinity, and not for
//! free: `eps` survives in `p = r(1−eps)`, and the **retries** it buys are unbounded. At `k = 500` the monitored
//! optimum needs over a hundred retries per task, so under any finite retry budget the optimum is firmly interior
//! again: `examples/r7_decomposition.rs` measures `k = 8` at a cap of one retry and `k = 16` at caps of three and ten,
//! with `k = 500` failing outright at every cap. [`Granularity::expected_retries`] is the number that decides this.
//!
//! [`sampled_set_escape_rate`]: crate::sampled_set_escape_rate

use crate::{sampled_set_escape_rate, Skill};

/// A long-horizon task, described by what makes it fail rather than by what it means.
#[derive(Clone, Copy, Debug)]
pub struct TaskDecomposition {
    /// Total control steps.
    pub horizon: usize,
    /// Probability the task fails on any one step, independent across steps.
    pub per_step_failure: f64,
    /// Probability a skill's own failure is detected, so a retry is possible.
    pub detection: f64,
    /// Dimension of the handoff state, which sets how expensive a funnel is to measure.
    pub dim: usize,
    /// Total rollouts available to measure funnels, split across however many skills there are.
    pub rollout_budget: usize,
}

/// What a given granularity delivers.
#[derive(Clone, Copy, Debug)]
pub struct Granularity {
    pub skills: usize,
    /// Steps per skill.
    pub sub_horizon: f64,
    /// Per-skill success before handoffs.
    pub skill_success: f64,
    /// Per-handoff containment escape rate at this budget split.
    pub escape_rate: f64,
    /// End-to-end pass probability with retries.
    pub pass: f64,
    /// Retries the whole task is expected to need. This is the cost a monitor charges instead of a failure, and
    /// under a finite retry budget it is what decides the decomposition.
    pub expected_retries: f64,
}

impl TaskDecomposition {
    /// Pass probability at granularity `k`. `monitored` says whether a runtime containment check runs on each handoff.
    ///
    /// Returns `None` for `k = 0`, `k` beyond the horizon, or parameters outside `[0, 1]`.
    pub fn at(&self, k: usize, monitored: bool) -> Option<Granularity> {
        if k == 0 || k > self.horizon || self.dim == 0 || !(0.0..=1.0).contains(&self.per_step_failure) || !(0.0..=1.0).contains(&self.detection) {
            return None;
        }
        let sub_horizon = self.horizon as f64 / k as f64;
        let skill_success = (1.0 - self.per_step_failure).powf(sub_horizon);
        // the measurement budget is split across the skills, so finer cuts get coarser funnels
        let escape_rate = sampled_set_escape_rate(self.dim, self.rollout_budget / k);
        // a link succeeds only if the skill succeeds AND the handoff stays inside the certified inflow
        let p = skill_success * (1.0 - escape_rate);
        // what ends the task: a skill failure its detector missed, plus every silent breach if nothing is watching
        let dead = (1.0 - skill_success) * (1.0 - self.detection) + if monitored { 0.0 } else { skill_success * escape_rate };
        let d = 1.0 - dead / (1.0 - p).max(1e-300);
        let link = Skill::new(p, d.clamp(0.0, 1.0), 1.0)?;
        let ki = i32::try_from(k).ok()?;
        Some(Granularity {
            skills: k,
            sub_horizon,
            skill_success,
            escape_rate,
            pass: link.pass_probability().powi(ki),
            expected_retries: (link.expected_attempts() - 1.0) * k as f64,
        })
    }

    /// The granularity that maximises pass probability, searching `1..=max_k`.
    pub fn optimal_granularity(&self, max_k: usize, monitored: bool) -> Option<Granularity> {
        (1..=max_k.min(self.horizon)).filter_map(|k| self.at(k, monitored)).max_by(|a, b| a.pass.total_cmp(&b.pass))
    }

    /// Pass probability ignoring handoff cost entirely — the recovery benefit alone, which saturates.
    pub fn recovery_only(&self, k: usize) -> Option<f64> {
        if k == 0 || k > self.horizon {
            return None;
        }
        let p = (1.0 - self.per_step_failure).powf(self.horizon as f64 / k as f64);
        Some(Skill::new(p, self.detection, 1.0)?.pass_probability().powi(i32::try_from(k).ok()?))
    }
}

/// **Decomposition buys nothing without recovery.** The product of per-skill successes is `(1−q)^H` at every
/// granularity, so this returns one number, and it is the same number for all `k`.
pub fn no_recovery_success(horizon: usize, per_step_failure: f64) -> f64 {
    (1.0 - per_step_failure).powi(i32::try_from(horizon).unwrap_or(i32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> TaskDecomposition {
        TaskDecomposition { horizon: 500, per_step_failure: 0.002, detection: 0.9, dim: 4, rollout_budget: 20_000 }
    }

    /// The negative result first, because it is what most decomposition arguments assume away.
    #[test]
    fn decomposition_alone_changes_nothing() {
        let t = task();
        let monolith = no_recovery_success(t.horizon, t.per_step_failure);
        for k in [1usize, 5, 10, 50, 500] {
            let g = t.at(k, false).unwrap();
            let product = g.skill_success.powi(i32::try_from(k).unwrap());
            eprintln!("k = {k:>3}: per-skill {:.6} over {:.1} steps, product {product:.6}", g.skill_success, g.sub_horizon);
            assert!((product - monolith).abs() < 1e-9, "the product is (1-q)^H at every k");
        }
        eprintln!("   All equal {monolith:.6}. Cutting a task into skills does not make it more likely to succeed.");
    }

    /// Recovery is where the benefit lives, and it saturates.
    #[test]
    fn recovery_is_the_benefit_and_it_saturates() {
        let t = task();
        eprintln!("recovery benefit alone (no handoff cost), 500 steps at q = 0.002, detection 0.9:");
        let mut prev = 0.0;
        for k in [1usize, 2, 5, 10, 50, 100, 500] {
            let p = t.recovery_only(k).unwrap();
            eprintln!("    k = {k:>3}: pass {p:.4}");
            assert!(p >= prev - 1e-12, "finer cuts never hurt when handoffs are free");
            prev = p;
        }
        assert!((t.recovery_only(1).unwrap() - 0.8531).abs() < 5e-4);
        assert!((t.recovery_only(500).unwrap() - 0.9048).abs() < 5e-4);
        // saturation: the last 5x of granularity buys almost nothing
        let (a, b) = (t.recovery_only(100).unwrap(), t.recovery_only(500).unwrap());
        eprintln!("   k = 100 -> 500 gains only {:.5}, so the benefit is spent well before the horizon", b - a);
        assert!(b - a < 1e-3);
    }

    /// **The interior optimum**, and that it is set by the rollout budget.
    #[test]
    fn the_optimum_is_set_by_the_measurement_budget() {
        eprintln!("optimal granularity vs rollout budget (500 steps, q = 0.002, d = 0.9, dim 4, unmonitored):");
        eprintln!("    {:>10}  {:>7}  {:>10}  {:>12}", "budget", "best k", "pass", "escape rate");
        let mut best_ks = Vec::new();
        for budget in [2_000usize, 20_000, 200_000, 2_000_000] {
            let t = TaskDecomposition { rollout_budget: budget, ..task() };
            let g = t.optimal_granularity(500, false).unwrap();
            eprintln!("    {budget:>10}  {:>7}  {:>10.4}  {:>11.4}%", g.skills, g.pass, 100.0 * g.escape_rate);
            best_ks.push(g.skills);
        }
        // a bigger measurement budget affords a finer decomposition
        assert!(best_ks.windows(2).all(|w| w[1] >= w[0]), "optimal k is non-decreasing in budget: {best_ks:?}");
        assert!(best_ks[3] > best_ks[0], "and strictly grows across a 1000x budget range: {best_ks:?}");
        // the optimum is interior, not at either end
        let t = task();
        let g = t.optimal_granularity(500, false).unwrap();
        assert!(g.skills > 1 && g.skills < 500, "interior optimum at k = {}", g.skills);
        assert!(g.pass > t.at(1, false).unwrap().pass && g.pass > t.at(500, false).unwrap().pass);
        eprintln!("\n   The best decomposition is set by how many rollouts you can afford, not by task semantics.");
    }

    /// **A containment monitor shifts the optimum much finer, and charges retries for it.**
    ///
    /// An earlier version of [`TaskDecomposition::at`] had a spurious `(1−eps)` in the detection term, which made
    /// `eps` divide out of `p/(p + dead)` and reported that a monitor recovers exact sets for free. The Monte Carlo
    /// in `examples/r7_decomposition.rs` disagreed at `k = 500` (0.8822 measured against 0.9047 predicted) and the
    /// cause was that a breach draw never happens when the skill has already failed. It does not cancel.
    #[test]
    fn a_monitor_shifts_the_optimum_finer_and_charges_retries() {
        let t = task();
        let unmonitored = t.optimal_granularity(500, false).unwrap();
        let monitored = t.optimal_granularity(500, true).unwrap();
        eprintln!("unmonitored: best k = {:>3}, pass {:.4}, expected retries {:.2}", unmonitored.skills, unmonitored.pass, unmonitored.expected_retries);
        eprintln!("  monitored: best k = {:>3}, pass {:.4}, expected retries {:.2}", monitored.skills, monitored.pass, monitored.expected_retries);
        assert!(monitored.skills > unmonitored.skills, "the monitor affords finer cuts: {} -> {}", unmonitored.skills, monitored.skills);
        assert!(monitored.pass > unmonitored.pass, "and a better task rate");

        // it does NOT recover exact sets: eps survives in p = r(1-eps)
        let ideal = t.recovery_only(monitored.skills).unwrap();
        eprintln!("  monitored pass {:.6} vs handoff-free ideal {ideal:.6} at the same k - short by {:.6}", monitored.pass, ideal - monitored.pass);
        assert!(monitored.pass < ideal, "a monitor is not a substitute for measuring the funnel");

        // and the retry cost is what it charges instead
        assert!(monitored.expected_retries > unmonitored.expected_retries, "the monitor pays in attempts");
        let finest = t.at(500, true).unwrap();
        eprintln!("\n  at the finest cut k = 500 the monitor needs {:.1} retries per task for pass {:.4}", finest.expected_retries, finest.pass);
        assert!(finest.expected_retries > 100.0, "over a hundred attempts, which no finite time budget affords");
        assert!(finest.pass < monitored.pass, "so the finest cut is not even the best unconstrained choice");
        eprintln!("  The monitor converts set-measurement error from a failure into a delay. That is a large gain and");
        eprintln!("  not a free one: under a retry cap the simulation puts the optimum back at k = 8 to 16.");
    }
}
