//! **Embodied energy — the term that turns `E_task` from a rate into a crossover.**
//!
//! [`Battery`](crate::Battery) prices the joules a task draws from the pack and
//! [`MotorThermal`](crate::MotorThermal) prices where they leave as heat. Neither prices the joules it
//! took to *build* the actuator, and without that term `E_task` is a **rate**: a bench can rank two
//! designs at one operating point but cannot say at what mission duration the ranking **inverts**. That
//! threshold is the useful output, and it was unstateable.
//!
//! # Biology's own ratio inverts across taxa
//!
//! Flagellar construction is 5–16.5% of a cell-cycle energy budget against 0.73–5.2% for operation, so
//! build-to-operate runs about **23:1** for *Pyrococcus furiosus* archaella and about **1.2:1** for
//! *Chlamydomonas*. Life therefore sits at build-cheap/run-hot for large cells and build-expensive/run-free
//! for small ones. Which side a design is on is a decision with a threshold, not a preference.
//!
//! # The arithmetic, and the one input this module refuses to hide
//!
//! ```text
//! E_task(T) = E_build + T·P_operate,     E_build = mass · intensity
//! ```
//!
//! Two designs tie at `T* = (E_build_b − E_build_a) / (P_a − P_b)`, which is positive only when the
//! design that costs more to build costs less to run. Otherwise one design dominates at every duration
//! and there is no crossover to report, which [`crossover_seconds`] says with `None` rather than a number.
//!
//! **Embodied intensity is an input, never a default.** Commonly quoted figures for processed metals span
//! roughly an order of magnitude and depend on the process, the recycled fraction and the boundary of the
//! accounting, so a hidden constant here would be a number with no protocol. The caller supplies it and
//! states its source; this module only does the arithmetic that turns it into a threshold.

/// An actuator priced on both sides of the ledger: what it cost to build, and what it costs to run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbodiedActuator {
    /// Mass (kg).
    pub mass_kg: f64,
    /// Embodied energy intensity (MJ/kg). **Supply this with its source.** See the module note on why
    /// there is no default.
    pub intensity_mj_per_kg: f64,
    /// Mean electrical power while on task (W), from whatever the rest of the stack measures — a holding
    /// posture, a duty-cycled gait, a full mission profile.
    pub operating_w: f64,
}

impl EmbodiedActuator {
    /// Joules it took to build. `None` unless mass and intensity are finite and non-negative.
    pub fn build_j(&self) -> Option<f64> {
        let ok = [self.mass_kg, self.intensity_mj_per_kg].iter().all(|v| v.is_finite() && *v >= 0.0);
        ok.then_some(self.mass_kg * self.intensity_mj_per_kg * 1.0e6)
    }

    /// Total joules over a mission of `mission_s` seconds: build plus operation.
    pub fn task_j(&self, mission_s: f64) -> Option<f64> {
        if !mission_s.is_finite() || mission_s < 0.0 || !self.operating_w.is_finite() || self.operating_w < 0.0 {
            return None;
        }
        let b = self.build_j()?;
        let t = b + mission_s * self.operating_w;
        t.is_finite().then_some(t)
    }

    /// The fraction of a mission's total energy that was spent before the machine moved.
    ///
    /// This is the quantity directly comparable with the biological figures in the module note, and it is
    /// the number that says which regime a design is in. It falls monotonically with mission duration, so
    /// quoting it without the duration it was computed over is meaningless.
    pub fn build_share(&self, mission_s: f64) -> Option<f64> {
        let total = self.task_j(mission_s)?;
        if total <= 0.0 {
            return None;
        }
        Some(self.build_j()? / total)
    }
}

/// **The mission duration at which two designs tie, and past which the ranking inverts.**
///
/// `None` when there is no crossover to report: identical operating power, or one design both cheaper to
/// build *and* cheaper to run, in which case it wins at every duration and a threshold would be fiction.
///
/// This is the whole point of carrying a build term. Without it a bench compares operating power and
/// declares a winner; with it the bench can say *for how long* that winner holds.
pub fn crossover_seconds(a: &EmbodiedActuator, b: &EmbodiedActuator) -> Option<f64> {
    let (ba, bb) = (a.build_j()?, b.build_j()?);
    if !a.operating_w.is_finite() || !b.operating_w.is_finite() || a.operating_w < 0.0 || b.operating_w < 0.0 {
        return None;
    }
    let dp = a.operating_w - b.operating_w;
    if dp == 0.0 {
        return None; // parallel lines: whichever is cheaper to build is cheaper forever
    }
    let t = (bb - ba) / dp;
    (t.is_finite() && t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heavy and efficient against light and thirsty. Intensities are stated inputs, not defaults.
    fn pair() -> (EmbodiedActuator, EmbodiedActuator) {
        // Heavier, built with more energy, but holds its posture on less power.
        let heavy = EmbodiedActuator { mass_kg: 2.4, intensity_mj_per_kg: 60.0, operating_w: 1.9 };
        // Lighter and cheaper to build, but draws more to do the same job.
        let light = EmbodiedActuator { mass_kg: 0.8, intensity_mj_per_kg: 45.0, operating_w: 6.4 };
        (heavy, light)
    }

    /// **THE POINT: the ranking inverts across the crossover.** A bench that reports only a rate cannot
    /// express this, which is why the build term has to be carried rather than assumed negligible.
    #[test]
    fn the_ranking_inverts_across_the_crossover_and_a_rate_cannot_say_so() {
        let (heavy, light) = pair();
        let t = crossover_seconds(&heavy, &light).expect("these two do cross");

        // Before the crossover the light design wins; after it the heavy one does. Same two machines.
        let before = t * 0.5;
        let after = t * 2.0;
        assert!(
            light.task_j(before).unwrap() < heavy.task_j(before).unwrap(),
            "light must win early"
        );
        assert!(
            heavy.task_j(after).unwrap() < light.task_j(after).unwrap(),
            "heavy must win late — if it does not, there is no inversion and no reason to carry the term"
        );
        // And they tie at the crossover, to floating-point.
        let (a, b) = (heavy.task_j(t).unwrap(), light.task_j(t).unwrap());
        assert!((a - b).abs() < 1e-6 * a.max(b), "they must tie at t*: {a} vs {b}");

        eprintln!(
            "  crossover {:.0} s ({:.1} h): light {:.0} kJ vs heavy {:.0} kJ at the tie",
            t, t / 3600.0, b / 1e3, a / 1e3
        );
        // Operating power alone ranks these the other way round, which is the error the term removes.
        assert!(heavy.operating_w < light.operating_w, "the rate-only ranking favours heavy at every duration");
        assert!(heavy.build_j().unwrap() > light.build_j().unwrap(), "and the build term is what opposes it");
    }

    /// No crossover to report when one design dominates both terms, or when the rates are equal.
    #[test]
    fn a_dominating_design_has_no_crossover_and_says_so() {
        let good = EmbodiedActuator { mass_kg: 0.8, intensity_mj_per_kg: 45.0, operating_w: 1.9 };
        let bad = EmbodiedActuator { mass_kg: 2.4, intensity_mj_per_kg: 60.0, operating_w: 6.4 };
        assert!(crossover_seconds(&good, &bad).is_none(), "cheaper to build AND to run: it wins at every duration");
        let twin = EmbodiedActuator { mass_kg: 1.0, intensity_mj_per_kg: 50.0, operating_w: 1.9 };
        let other = EmbodiedActuator { mass_kg: 2.0, intensity_mj_per_kg: 50.0, operating_w: 1.9 };
        assert!(crossover_seconds(&twin, &other).is_none(), "equal rates never cross");
    }

    /// `build_share` is the biology-comparable number, and it is meaningless without its duration.
    #[test]
    fn build_share_falls_with_mission_duration_and_spans_the_biological_regimes() {
        let (heavy, _) = pair();
        let short = heavy.build_share(60.0).expect("share");
        let long = heavy.build_share(60.0 * 60.0 * 24.0 * 365.0).expect("share");
        assert!(short > long, "the build share must fall as the mission lengthens: {short} vs {long}");
        assert!(short > 0.99, "over a minute this design is almost all build cost: {short:.4}");
        assert!(long < 0.75, "over a year operation has taken over: {long:.4}");

        // The regimes named in the module note, reached by choosing the duration rather than the machine:
        // build:operate 23:1 is the archaellum end, 1.2:1 the Chlamydomonas end.
        let ratio_at = |s: f64| {
            let sh = heavy.build_share(s).unwrap();
            sh / (1.0 - sh)
        };
        assert!(ratio_at(60.0) > 23.0, "a one-minute mission is past the build-expensive end");
        let mid = ratio_at(heavy.build_j().unwrap() / heavy.operating_w);
        assert!((mid - 1.0).abs() < 1e-9, "by construction build:operate is 1:1 when T = E_build/P");
        // `mid` is a RATIO, not a time. The first version of this line printed it under a seconds label
        // and reported "1:1 at 1 s" for a design whose 1:1 duration is 2.4 years.
        let t_unity = heavy.build_j().unwrap() / heavy.operating_w;
        eprintln!(
            "  build:operate 23:1 at {:.1} d, 1.2:1 at {:.1} d, 1:1 at {:.1} d (ratio there = {:.3})",
            heavy.build_j().unwrap() / (23.0 * heavy.operating_w) / 86400.0,
            heavy.build_j().unwrap() / (1.2 * heavy.operating_w) / 86400.0,
            t_unity / 86400.0,
            mid
        );
    }

    #[test]
    fn it_refuses_what_it_cannot_answer_for() {
        let a = EmbodiedActuator { mass_kg: 1.0, intensity_mj_per_kg: 50.0, operating_w: 2.0 };
        assert!(a.task_j(0.0).is_some(), "a zero-length mission is still just the build cost");
        assert_eq!(a.task_j(0.0), a.build_j());
        assert!(a.task_j(-1.0).is_none(), "negative mission");
        assert!(a.task_j(f64::NAN).is_none(), "non-finite mission");
        assert!(EmbodiedActuator { mass_kg: -1.0, ..a }.build_j().is_none(), "negative mass");
        assert!(EmbodiedActuator { intensity_mj_per_kg: f64::NAN, ..a }.build_j().is_none(), "non-finite intensity");
        assert!(EmbodiedActuator { operating_w: -1.0, ..a }.task_j(10.0).is_none(), "negative power");
        assert!(a.build_share(0.0).is_some_and(|s| (s - 1.0).abs() < 1e-12), "at T=0 the share is all build");
    }
}
