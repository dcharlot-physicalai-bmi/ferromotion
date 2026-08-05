//! **A locomotion task suite, and what a suite score can actually support.**
//!
//! Benchmark suites for whole-body humanoid control report a success rate per task across a handful of seeds, and
//! conclusions get drawn from the ranking: this method beats that one, flat reinforcement learning struggles where a
//! hierarchical decomposition succeeds. Both halves of that sentence are measurements, and the second half is a
//! measurement of a *difference between two measurements*, which needs far more evidence than either alone.
//!
//! This module supplies both halves and keeps them separate. [`Task`] and [`run_suite`] give five locomotion tasks on
//! the verified ALIP step-to-step dynamics ([`Alip`](crate::Alip)) — real dynamics with real disturbances, rather than
//! thirty-one tasks nobody checked. [`seeds_to_rank`] and [`family_corrected_confidence`] give the metrology: how many
//! seeds a claimed ranking needs, and what a 31-task suite does to that number.
//!
//! **The number is uncomfortable.** Distinguishing a 70% task from an 80% one at 95% confidence takes on the order of
//! 300 seeds per method, and a suite of 31 tasks compared at 95% each expects about 1.6 false rankings from chance
//! alone. A suite is not a way to get more confidence from the same compute; it is a way to spend confidence on
//! breadth. Whether that is the right trade is a choice, but it should be a made choice.

use crate::{wilson_interval, Alip, Interval, Xorshift};
use nalgebra::Vector2;

/// One episode's outcome. A benchmark needs the success bit; the return is kept because a task can be survived
/// badly and the two orderings are not the same.
#[derive(Clone, Copy, Debug)]
pub struct Episode {
    pub success: bool,
    pub total_reward: f64,
    pub steps: usize,
}

/// A foot-placement policy: given the ALIP state `[x, L]` and the commanded velocity, return the step length.
pub trait StepPolicy {
    fn action(&self, sigma: &Vector2<f64>, command: f64) -> f64;
    fn name(&self) -> &'static str;
}

/// The deadbeat foot-placement law, which is the strong low-level policy a hierarchy would sit on top of.
pub struct Deadbeat {
    pub gain: Vector2<f64>,
    /// Feed-forward step length for the commanded velocity.
    pub stride: f64,
}

impl StepPolicy for Deadbeat {
    fn action(&self, sigma: &Vector2<f64>, command: f64) -> f64 {
        self.gain.dot(sigma) + command * self.stride
    }
    fn name(&self) -> &'static str {
        "deadbeat"
    }
}

/// A **detuned** deadbeat: the angular-momentum gain scaled by `detune`. This is the realistic model of an imperfect
/// policy — one that has the right structure and the wrong numbers — and it is what the suite ranks against the exact
/// law.
///
/// A gain on CoM offset *alone* is not an option worth ranking. At a 1 m height and a 0.35 s step the ALIP
/// step-to-step matrix is
///
/// ```text
///   M = [ [   1.6635,   0.0094 ],
///         [ 187.3669,   1.6635 ] ]
/// ```
///
/// so a CoM offset builds angular momentum with a gain of **187** (`M21`) while angular momentum feeds back into
/// offset at only `0.0094` (`M12`) — a scaling ratio of `2e4` across one 2x2 matrix. A law that does not feed back `L`
/// never corrects the momentum its own offset generated, and it fails at every gain from 0.5 to 5.0 (measured: 0.000
/// success throughout). The same scaling is why the deadbeat gain reads `(3.327, 0.0242)`: the tiny second entry is
/// not a negligible term, it is a large correction to a large-valued state.
pub struct DetunedDeadbeat {
    pub gain: Vector2<f64>,
    pub stride: f64,
    /// Scale applied to the angular-momentum gain. `1.0` is the exact deadbeat law.
    pub detune: f64,
}

impl StepPolicy for DetunedDeadbeat {
    fn action(&self, sigma: &Vector2<f64>, command: f64) -> f64 {
        Vector2::new(self.gain.x, self.detune * self.gain.y).dot(sigma) + command * self.stride
    }
    fn name(&self) -> &'static str {
        "detuned deadbeat"
    }
}

/// What makes a task hard, and when it counts as passed.
#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub name: &'static str,
    pub walker: Alip,
    /// Steps the episode must survive.
    pub horizon: usize,
    /// Commanded forward velocity, in stride units.
    pub command: f64,
    /// Per-step disturbance on the angular momentum, as a standard deviation.
    pub push_sigma: f64,
    /// A one-off impulse applied at step `impulse_at`, if any.
    pub impulse: Option<(usize, f64)>,
    /// The CoM offset past which the walker has fallen.
    pub fall_offset: f64,
    /// Multiplicative step-timing jitter, as a standard deviation on `t_step`.
    pub timing_jitter: f64,
}

impl Task {
    /// Run one episode. Deterministic in `seed`, so two policies can be compared on identical disturbances — common
    /// random numbers, without which the seed noise swamps the difference being measured.
    pub fn episode(&self, policy: &dyn StepPolicy, seed: u64) -> Episode {
        let mut rng = Xorshift::new(seed);
        let mut sigma = Vector2::new(0.0, 0.0);
        let mut reward = 0.0;
        for k in 0..self.horizon {
            // a jittered step duration changes the transition matrix, so build the walker per step
            let mut walker = self.walker;
            if self.timing_jitter > 0.0 {
                walker.t_step = (self.walker.t_step * (1.0 + self.timing_jitter * rng.normal())).max(1e-3);
            }
            let u = policy.action(&sigma, self.command);
            sigma = walker.step(&sigma, u);
            if self.push_sigma > 0.0 {
                sigma.y += self.push_sigma * rng.normal();
            }
            if self.impulse.is_some_and(|(at, _)| k == at) {
                sigma.y += self.impulse.expect("checked").1;
            }
            if !sigma.x.is_finite() || sigma.x.abs() > self.fall_offset {
                return Episode { success: false, total_reward: reward, steps: k + 1 };
            }
            // reward staying near the nominal orbit, which is what "walking well" means here
            reward += 1.0 - (sigma.x.abs() / self.fall_offset).min(1.0);
        }
        Episode { success: true, total_reward: reward, steps: self.horizon }
    }

    /// Success rate over `seeds` episodes, with its Wilson interval — a rate without one is not a measurement.
    pub fn score(&self, policy: &dyn StepPolicy, seeds: usize, confidence: f64) -> Option<(f64, Interval)> {
        if seeds == 0 {
            return None;
        }
        let wins = (0..seeds).filter(|s| self.episode(policy, *s as u64 + 1).success).count();
        Some((wins as f64 / seeds as f64, wilson_interval(wins, seeds, confidence)?))
    }
}

/// The five locomotion tasks, on a 1.0 m ALIP walker with a 0.35 s step.
///
/// Difficulties were chosen from a measured grid rather than guessed, so that each task lands in the informative
/// middle for at least one policy. A task everything passes and a task nothing passes both measure zero, and a suite
/// built by intuition tends to be made of those.
pub fn locomotion_suite() -> Vec<Task> {
    let walker = Alip { mass: 45.0, height: 1.0, g: 9.81, t_step: 0.35 };
    vec![
        Task { name: "walk-flat", walker, horizon: 40, command: 1.0, push_sigma: 0.0, impulse: None, fall_offset: 0.5, timing_jitter: 0.0 },
        Task { name: "walk-pushed", walker, horizon: 40, command: 1.0, push_sigma: 6.0, impulse: None, fall_offset: 0.5, timing_jitter: 0.0 },
        Task { name: "walk-shoved", walker, horizon: 40, command: 1.0, push_sigma: 8.0, impulse: None, fall_offset: 0.5, timing_jitter: 0.0 },
        Task { name: "walk-jitter", walker, horizon: 40, command: 1.0, push_sigma: 2.0, impulse: None, fall_offset: 0.5, timing_jitter: 0.05 },
        Task { name: "walk-jitter-hard", walker, horizon: 40, command: 1.0, push_sigma: 2.0, impulse: None, fall_offset: 0.5, timing_jitter: 0.10 },
    ]
}

/// A whole suite's result for one policy.
#[derive(Clone, Debug)]
pub struct SuiteScore {
    pub policy: &'static str,
    pub per_task: Vec<(&'static str, f64, Interval)>,
    pub seeds: usize,
}

impl SuiteScore {
    /// Mean success across tasks. Reported because it is what gets quoted, and it is the number that hides the
    /// per-task intervals.
    pub fn headline(&self) -> f64 {
        if self.per_task.is_empty() {
            return f64::NAN;
        }
        self.per_task.iter().map(|(_, p, _)| p).sum::<f64>() / self.per_task.len() as f64
    }

    /// Tasks whose interval is wide enough that the rate carries almost no information.
    pub fn uninformative(&self, width: f64) -> Vec<&'static str> {
        self.per_task.iter().filter(|(_, _, i)| i.width() > width).map(|(n, _, _)| *n).collect()
    }
}

/// Run a suite for one policy.
pub fn run_suite(tasks: &[Task], policy: &dyn StepPolicy, seeds: usize, confidence: f64) -> SuiteScore {
    SuiteScore {
        policy: policy.name(),
        per_task: tasks.iter().filter_map(|t| t.score(policy, seeds, confidence).map(|(p, i)| (t.name, p, i))).collect(),
        seeds,
    }
}

/// **How many seeds are needed before two success rates are distinguishable** at the given confidence, using the
/// normal-approximation two-proportion test:
///
/// ```text
///   n >= ( z * ( sqrt(p1(1-p1)) + sqrt(p2(1-p2)) ) / (p1 - p2) )^2
/// ```
///
/// Returns `None` when the rates are equal (no `n` separates them) or the inputs are outside `[0, 1]`.
pub fn seeds_to_rank(p_a: f64, p_b: f64, confidence: f64) -> Option<usize> {
    if !(0.0..=1.0).contains(&p_a) || !(0.0..=1.0).contains(&p_b) || (p_a - p_b).abs() < 1e-12 {
        return None;
    }
    let z = crate::z_for(confidence)?;
    let s = (p_a * (1.0 - p_a)).sqrt() + (p_b * (1.0 - p_b)).sqrt();
    Some((z * s / (p_a - p_b)).powi(2).ceil() as usize)
}

/// **What comparing `tasks` tasks at `per_task` confidence costs.** Returns `(family_confidence,
/// expected_false_rankings, bonferroni_per_task)`.
///
/// A suite is a family of simultaneous comparisons. At 95% per task, 31 tasks expect `31 * 0.05 = 1.55` rankings that
/// are pure chance, and the probability that *every* ranking is sound is `0.95^31 = 0.20`. Holding the family at 95%
/// requires each task to be tested at `1 - 0.05/31`, which raises the seed requirement.
pub fn family_corrected_confidence(tasks: usize, per_task: f64) -> Option<(f64, f64, f64)> {
    if tasks == 0 || !(0.0..1.0).contains(&per_task) {
        return None;
    }
    let alpha = 1.0 - per_task;
    Some((per_task.powi(i32::try_from(tasks).ok()?), tasks as f64 * alpha, 1.0 - alpha / tasks as f64))
}

/// Seeds needed to rank two rates while holding the **whole suite** at `family_confidence` — the Bonferroni-corrected
/// version of [`seeds_to_rank`].
pub fn seeds_to_rank_across_suite(p_a: f64, p_b: f64, tasks: usize, family_confidence: f64) -> Option<usize> {
    let (_, _, per_task) = family_corrected_confidence(tasks, family_confidence)?;
    seeds_to_rank(p_a, p_b, per_task)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walker() -> Alip {
        Alip { mass: 45.0, height: 1.0, g: 9.81, t_step: 0.35 }
    }

    fn policies() -> (Deadbeat, DetunedDeadbeat) {
        let w = walker();
        let g = w.deadbeat_gain();
        (Deadbeat { gain: g, stride: 0.35 }, DetunedDeadbeat { gain: g, stride: 0.35, detune: 0.95 })
    }

    /// A gain on CoM offset alone cannot work, at any gain, because the state is scaled so badly that the two
    /// coordinates are not comparable: `M21 = 187` builds momentum from offset while `M12 = 0.0094` feeds it back.
    ///
    /// (An earlier version of this test asserted `M12 > 100`. `nalgebra`'s `as_slice()` is column-major, so a probe
    /// that read the matrix as a flat slice put `M21` in the `M12` position and the interpretation came out reversed.)
    #[test]
    fn the_angular_momentum_gain_is_not_optional() {
        let w = walker();
        let m = w.s2s_matrix();
        eprintln!("s2s matrix: [[{:.4}, {:.4}], [{:.6}, {:.4}]]", m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]);
        assert!(m[(1, 0)] > 100.0, "offset builds momentum: M21 = {:.1}", m[(1, 0)]);
        assert!(m[(0, 1)] < 0.1, "momentum feeds back weakly: M12 = {:.5}", m[(0, 1)]);
        eprintln!("   scaling ratio M21/M12 = {:.0}, across a single 2x2 matrix", m[(1, 0)] / m[(0, 1)]);
        let k = w.deadbeat_gain();
        eprintln!("   deadbeat gain ({:.4}, {:.4}) - the small entry is a large correction to a large-valued state", k.x, k.y);

        struct XOnly(f64);
        impl StepPolicy for XOnly {
            fn action(&self, s: &Vector2<f64>, c: f64) -> f64 {
                self.0 * s.x + c * 0.35
            }
            fn name(&self) -> &'static str {
                "x-only"
            }
        }
        let task = locomotion_suite()[1];
        for gain in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0] {
            let (p, _) = task.score(&XOnly(gain), 200, 0.95).unwrap();
            assert!(p < 1e-12, "x-only at gain {gain} should never survive, got {p}");
        }
        eprintln!("   x-only foot placement: 0.000 success at every gain from 0.5 to 5.0");
    }

    /// **A stable closed loop that fails every episode.** Detuning to `0.80` leaves a spectral radius of `0.952`, so
    /// the walker is asymptotically stable and still falls with zero disturbance, because `M12 = 187` turns a small
    /// angular-momentum transient into an offset excursion past the boundary before it decays.
    ///
    /// A contraction rate bounds where a trajectory ends up, not how far it travels first.
    #[test]
    fn spectral_radius_below_one_does_not_keep_the_walker_up() {
        let w = walker();
        let g = w.deadbeat_gain();
        let flat = locomotion_suite()[0]; // zero disturbance, so the outcome is deterministic
        eprintln!("{:>7}  {:>9}  {:>10}", "detune", "rho", "success");
        let mut stable_and_failing = 0usize;
        for detune in [0.80, 0.90, 0.95, 1.00] {
            let k = Vector2::new(g.x, detune * g.y);
            let rho = w.closed_loop(&k).complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0f64, f64::max);
            let policy = DetunedDeadbeat { gain: g, stride: 0.35, detune };
            let (p, _) = flat.score(&policy, 20, 0.95).unwrap();
            eprintln!("{detune:>7.2}  {rho:>9.5}  {p:>10.3}");
            if rho < 1.0 && p < 0.5 {
                stable_and_failing += 1;
            }
        }
        assert!(stable_and_failing >= 1, "at least one asymptotically stable gain fails the task outright");
        let k80 = Vector2::new(g.x, 0.80 * g.y);
        let rho80 = w.closed_loop(&k80).complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0f64, f64::max);
        assert!(rho80 < 1.0 && rho80 > 0.9, "rho = {rho80:.5}");
        assert!(flat.score(&DetunedDeadbeat { gain: g, stride: 0.35, detune: 0.80 }, 20, 0.95).unwrap().0 < 1e-12);
        eprintln!("\n   rho = {rho80:.4} < 1 and success = 0.000 with no disturbance at all. The task boundary is");
        eprintln!("   crossed in transient, which a spectral radius says nothing about.");
    }

    /// The suite has to actually discriminate, with a spread of gaps rather than all-or-nothing.
    #[test]
    fn the_suite_discriminates_with_a_spread_of_gaps() {
        let (deadbeat, detuned) = policies();
        let tasks = locomotion_suite();
        let a = run_suite(&tasks, &deadbeat, 400, 0.95);
        let b = run_suite(&tasks, &detuned, 400, 0.95);
        eprintln!("{:<20} {:>10} {:>10} {:>8}", "task", "deadbeat", "detuned", "gap");
        let mut gaps = Vec::new();
        for ((n, pa, _), (_, pb, _)) in a.per_task.iter().zip(&b.per_task) {
            eprintln!("{n:<20} {pa:>10.3} {pb:>10.3} {:>8.3}", pa - pb);
            gaps.push(pa - pb);
        }
        eprintln!("headline: deadbeat {:.3}, detuned {:.3}", a.headline(), b.headline());
        assert!(a.headline() > b.headline(), "the exact law is stronger and the suite must see it");
        assert!(gaps.iter().any(|g| g.abs() < 0.05), "at least one task ties, which is what a suite looks like");
        assert!(gaps.iter().any(|g| *g > 0.2), "and at least one separates them clearly");
        // no task may be uninformative for BOTH policies at this seed count
        for ((n, pa, _), (_, pb, _)) in a.per_task.iter().zip(&b.per_task) {
            assert!(pa.max(*pb) > 1e-9, "task {n} is passed by nobody and measures nothing");
        }
    }

    /// **The seed count this suite's own rankings need.**
    #[test]
    fn this_suite_needs_hundreds_of_seeds_for_its_own_rankings() {
        let (deadbeat, detuned) = policies();
        let tasks = locomotion_suite();
        let a = run_suite(&tasks, &deadbeat, 400, 0.95);
        let b = run_suite(&tasks, &detuned, 400, 0.95);
        eprintln!("{:<20} {:>9} {:>9}  {:>10}  {:>14}", "task", "deadbeat", "detuned", "seeds@95%", "seeds@family95%");
        let mut worst = 0usize;
        for ((n, pa, _), (_, pb, _)) in a.per_task.iter().zip(&b.per_task) {
            match (seeds_to_rank(*pa, *pb, 0.95), seeds_to_rank_across_suite(*pa, *pb, tasks.len(), 0.95)) {
                (Some(s), Some(f)) => {
                    eprintln!("{n:<20} {pa:>9.3} {pb:>9.3}  {s:>10}  {f:>14}");
                    worst = worst.max(f);
                }
                _ => eprintln!("{n:<20} {pa:>9.3} {pb:>9.3}  {:>10}  {:>14}", "tied", "-"),
            }
        }
        eprintln!("\n   worst case across the suite: {worst} seeds per policy per task to hold the family at 95%.");
        assert!(worst > 100, "a handful of seeds cannot support these rankings: {worst}");
    }

    /// **How many seeds a ranking needs at all.** This is the number that the practice of reporting 3 to 10 seeds
    /// runs into.
    #[test]
    fn ranking_two_rates_needs_far_more_seeds_than_are_usually_run() {
        eprintln!("seeds per method to distinguish two success rates at 95% confidence:");
        eprintln!("  {:>8} vs {:>8}  {:>10}", "rate A", "rate B", "seeds");
        for (a, b) in [(0.9, 0.5), (0.8, 0.6), (0.8, 0.7), (0.75, 0.70), (0.52, 0.50)] {
            eprintln!("  {a:>8.2} vs {b:>8.2}  {:>10}", seeds_to_rank(a, b, 0.95).unwrap());
        }
        let n = seeds_to_rank(0.8, 0.7, 0.95).unwrap();
        assert!((280..=340).contains(&n), "0.8 vs 0.7 takes about 300 seeds, not 3: {n}");
        assert!(seeds_to_rank(0.9, 0.5, 0.95).unwrap() < 30, "a 0.4 gap is visible in a handful of seeds");
        assert!(seeds_to_rank(0.5, 0.5, 0.95).is_none(), "equal rates are never distinguishable");
    }

    /// **A 31-task suite spends confidence on breadth.**
    #[test]
    fn a_thirty_one_task_suite_expects_false_rankings() {
        let (family, false_rankings, per_task) = family_corrected_confidence(31, 0.95).unwrap();
        eprintln!("31 tasks compared at 95% each:");
        eprintln!("    probability every ranking is sound: {family:.4}");
        eprintln!("    expected rankings that are chance:  {false_rankings:.2}");
        eprintln!("    per-task confidence to hold the family at 95%: {per_task:.5}");
        assert!(family < 0.21, "0.95^31 = {family:.4}, so four times in five at least one ranking is noise");
        assert!((false_rankings - 1.55).abs() < 1e-9);

        let plain = seeds_to_rank(0.8, 0.7, 0.95).unwrap();
        let corrected = seeds_to_rank_across_suite(0.8, 0.7, 31, 0.95).unwrap();
        eprintln!("\n    seeds for 0.8 vs 0.7 on one task:        {plain}");
        eprintln!("    seeds for the same claim across 31 tasks: {corrected}  ({:.2}x)", corrected as f64 / plain as f64);
        assert!(corrected > plain, "breadth costs seeds");
        eprintln!("\n    A suite is not a way to get more confidence from the same compute. It is a way to spend");
        eprintln!("    confidence on breadth, at {:.2}x the seeds per claim.", corrected as f64 / plain as f64);
    }

    /// A score with an interval can say when it has measured nothing; a bare rate cannot.
    #[test]
    fn a_small_seed_count_produces_uninformative_scores() {
        let (deadbeat, _) = policies();
        let tasks = locomotion_suite();
        let mut widths = Vec::new();
        for seeds in [3usize, 10, 100, 1000] {
            let s = run_suite(&tasks, &deadbeat, seeds, 0.95);
            let widest = s.per_task.iter().map(|(_, _, i)| i.width()).fold(0.0, f64::max);
            eprintln!("{seeds:>5} seeds: headline {:.3}, widest interval {widest:.3}, {} tasks wider than 0.25", s.headline(), s.uninformative(0.25).len());
            widths.push(widest);
        }
        assert!(widths[0] > 0.5, "at 3 seeds an interval spans more than half the range: {:.3}", widths[0]);
        assert!(widths[3] < 0.1, "at 1000 seeds every interval is tight: {:.3}", widths[3]);
        assert!(widths.windows(2).all(|w| w[1] < w[0]), "more seeds is always narrower");
        eprintln!("   The headline barely moves while the widest interval shrinks {:.0}x. Quoting the headline", widths[0] / widths[3]);
        eprintln!("   without the interval quotes the one part that does not depend on the evidence.");
    }

    /// Common random numbers: the same seed must give the same disturbances, and different seeds must differ, or the
    /// comparison is measuring the seeds instead of the policies.
    #[test]
    fn seeds_are_reproducible_and_distinct() {
        let task = locomotion_suite()[4]; // the jitter task, where the outcome genuinely depends on the seed
        let (_, detuned) = policies();
        let a = task.episode(&detuned, 7);
        let b = task.episode(&detuned, 7);
        assert_eq!((a.steps, a.success), (b.steps, b.success), "the same seed is the same episode");
        assert!((a.total_reward - b.total_reward).abs() < 1e-15);

        // across many seeds the outcomes must actually vary, or the task is deterministic and the seeds are theatre
        let outcomes: Vec<usize> = (1..40).map(|s| task.episode(&detuned, s).steps).collect();
        let distinct = outcomes.iter().collect::<std::collections::BTreeSet<_>>().len();
        eprintln!("39 seeds on {} produced {distinct} distinct episode lengths", task.name);
        assert!(distinct > 3, "the seeds have to do something: {distinct} distinct outcomes");
    }
}
