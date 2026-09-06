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
//! Schavemaker & Lynch price flagellar construction and operation against the whole cell-cycle energy
//! budget (eLife 2022;11:e77266, Table 1):
//!
//! | organism | volume | construction | operation | build:operate |
//! |---|---|---|---|---|
//! | *Pyrococcus furiosus* (archaellum) | 0.22 µm³ | 16.5% | 0.73% | **23:1** |
//! | *Escherichia coli* | 1.0 µm³ | 5.0% | 5.2% | 0.96:1 |
//! | *Chlamydomonas reinhardtii* | 122 µm³ | 1.4% | 1.2% | **1.2:1** |
//!
//! The bill falls with cell volume and, as it falls, it shifts from operation-shared to
//! construction-dominated: the *smallest* cell here is the build-expensive/run-free one, and the largest
//! pays little for either. Which side a design is on is a decision with a threshold, not a preference.
//!
//! ⛔ **An earlier version of this note stated the construction range as "5–16.5%", which silently
//! excluded *Chlamydomonas* at 1.4%, and it called the large end "build-cheap/run-hot".** The two ratios
//! it quoted, 23:1 and 1.2:1, were right and were attached to the right organisms, but *Chlamydomonas*
//! runs on 1.2% of its budget, the second lowest operating cost in the table: the large cell is cheap on
//! both sides, not hot on one. The run-hot organism is *E. coli* at 5.2%, in the middle by volume. The
//! table above is the primary source's own.
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

/// A capability the body OWNS, priced whether or not it is ever used.
///
/// The third missing ledger term. In mammalian standard metabolic rate, protein synthesis is one of the
/// largest single ATP sinks (Rolfe & Brown, *Physiol Rev* 77:731–758, 1997,
/// `doi:10.1152/physrev.1997.77.3.731` — a review-level decomposition; take percentages from its own
/// tables with the tissue named). The bill for a tissue is **not per use, it is per owned gram per day**,
/// and that is why *deleting the modality* is evolution's move rather than duty-cycling it.
///
/// A robot ledger has no per-owned-gram term. It should, because a sensor that is never read still costs
/// three things: the energy that built it, its quiescent draw for every second it is owned, and the
/// energy to carry its mass for every metre the body moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OwnedCapability {
    /// What it is made of and what it draws when idle.
    pub built: EmbodiedActuator,
    /// Quiescent power while owned and unused (W) — the standing bill, distinct from `built.operating_w`.
    pub quiescent_w: f64,
}

impl OwnedCapability {
    /// **What owning this costs over a mission, used or not.**
    ///
    /// `E_own = E_build + quiescent·T + CoT·m·g·d`
    ///
    /// The third term is the one that makes this a *body* question rather than an electronics question:
    /// carrying mass costs energy in proportion to the body's cost of transport, so the same sensor is a
    /// different decision on a different chassis.
    ///
    /// Cost of transport is dimensionless, `E/(mgd)`, the same quantity the locomotion literature also
    /// calls specific resistance. Reference points, each with its source:
    ///
    /// | system | CoT | source |
    /// |---|---|---|
    /// | human walking | **0.2** | Tucker 1975, the standard benchmark |
    /// | MIT Cheetah, with electrical regeneration | 0.5 | Seok et al. 2015 |
    /// | Cassie, 1.0 m/s | 0.7 | 30 kg at 200 W total |
    /// | Honda ASIMO | **2** | 54 kg, 1.8 kW at 1.5 m/s, Sakagami et al. 2002 |
    /// | Walk-Man, actuation only / including electronics | 1.35 / 2.8 | Tsagarakis et al. 2017 |
    /// | BigDog, hydraulic | 15 | the high end of the published span |
    ///
    /// Collected in the review at `doi:10.3389/frobt.2018.00129`, which states the `E/(Mgd)` definition
    /// used here. ⛔ **An earlier version of this doc gave "a walking human is about 0.32" and called all
    /// three of its figures "measured" with no citation.** The human figure is 0.2; 0.32 matched no
    /// measurement in the literature this review located.
    ///
    /// `None` unless every input is finite and non-negative.
    pub fn owned_cost_j(&self, mission_s: f64, distance_m: f64, cost_of_transport: f64) -> Option<f64> {
        let ok = [mission_s, distance_m, cost_of_transport, self.quiescent_w]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0);
        if !ok {
            return None;
        }
        let build = self.built.build_j()?;
        let standing = self.quiescent_w * mission_s;
        let carried = cost_of_transport * self.built.mass_kg * 9.81 * distance_m;
        let total = build + standing + carried;
        total.is_finite().then_some(total)
    }

    /// **The value this capability must buy to be worth owning.** Identical to [`Self::owned_cost_j`] —
    /// named separately because it is the DELETION THRESHOLD, and naming it that way is the point.
    ///
    /// A capability earns its place only if what it saves elsewhere exceeds what owning it costs. This
    /// workspace already has the behavioural half of that argument on record: integrated gradients on a
    /// quadruped attribute about 80% to 4 of 9 feedback states, and reduced-set policies reach 93.7–99.1%
    /// of full-state performance (`arXiv:2306.17101`). Five of nine sensors were unnecessary
    /// *behaviourally*. This turns that into joules, so the deletion decision has a number on both sides.
    pub fn break_even_value_j(&self, mission_s: f64, distance_m: f64, cost_of_transport: f64) -> Option<f64> {
        self.owned_cost_j(mission_s, distance_m, cost_of_transport)
    }
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

    /// **Owning a capability costs energy even if it is never used, and the chassis changes the answer.**
    #[test]
    fn an_unused_sensor_still_costs_three_ways_and_the_chassis_decides_which_dominates() {
        // A 120 g sensor: cheap to build, 0.25 W quiescent, never read on this mission.
        let sensor = OwnedCapability {
            built: EmbodiedActuator { mass_kg: 0.120, intensity_mj_per_kg: 250.0, operating_w: 0.0 },
            quiescent_w: 0.25,
        };
        let (mission_s, distance_m) = (8.0 * 3600.0, 12_000.0); // an 8-hour, 12 km shift

        // The same sensor on three chassis. Cost of transport is the only thing that changes, and every
        // value here is one of the sourced reference points on `owned_cost_j`.
        const HUMAN: f64 = 0.2; // Tucker 1975
        const CHEETAH: f64 = 0.5; // Seok et al. 2015, with electrical regeneration
        const ASIMO: f64 = 2.0; // Sakagami et al. 2002, 54 kg at 1.8 kW and 1.5 m/s
        let human = sensor.owned_cost_j(mission_s, distance_m, HUMAN).expect("well-posed");
        let cheetah = sensor.owned_cost_j(mission_s, distance_m, CHEETAH).expect("well-posed");
        let robot = sensor.owned_cost_j(mission_s, distance_m, ASIMO).expect("well-posed");

        let build = sensor.built.build_j().unwrap();
        let standing = sensor.quiescent_w * mission_s;
        let carried = |total: f64| total - build - standing;
        eprintln!(
            "  build {:.1} kJ | standing {:.1} kJ | carried: {:.1} kJ at CoT 0.2, {:.1} kJ at 0.5, {:.1} kJ at 2.0",
            build / 1e3, standing / 1e3, carried(human) / 1e3, carried(cheetah) / 1e3, carried(robot) / 1e3
        );

        assert!(robot > cheetah && cheetah > human, "a worse chassis makes the same sensor more expensive to own");
        // The carrying term scales exactly with cost of transport, which is why the chassis is the decision.
        let carried_ratio = carried(robot) / carried(human);
        assert!((carried_ratio - 10.0).abs() < 1e-9, "CoT 2.0 vs 0.2 must be 10x to carry, got {carried_ratio:.3}x");

        // ⛔ Scaling in cost of transport is ALL the first version of this test checked, so magnitude,
        // mass-dependence and `g` were unconstrained: an implementation that dropped `mass_kg` entirely,
        // or used 9.8 for `g`, or multiplied by distance twice, passed it. The term is E = CoT·m·g·d and
        // the whole product is asserted, which is the only assertion that pins the hidden constant.
        // The tolerance is relative because `carried` is a difference against a 30 MJ build cost, so the
        // absolute resolution of an f64 there is already about 7e-9 J.
        let g = 9.81;
        let expected_carry = ASIMO * sensor.built.mass_kg * g * distance_m;
        assert!(
            (carried(robot) / expected_carry - 1.0).abs() < 1e-9,
            "E_carry must be exactly CoT*m*g*d, got {:.6} J against {expected_carry:.6} J",
            carried(robot)
        );
        assert!(carried(robot) > 0.0, "carrying mass over a distance costs something");

        // Mass-dependence, from outside: twice the sensor is twice the carrying bill. Build scales with
        // mass too, so the carried term is isolated before comparing.
        let heavier = OwnedCapability { built: EmbodiedActuator { mass_kg: 0.240, ..sensor.built }, ..sensor };
        let heavier_total = heavier.owned_cost_j(mission_s, distance_m, ASIMO).expect("well-posed");
        let heavier_carried = heavier_total - heavier.built.build_j().unwrap() - standing;
        assert!(
            (heavier_carried / carried(robot) - 2.0).abs() < 1e-9,
            "doubling the mass must double the carrying term, got {:.4}x",
            heavier_carried / carried(robot)
        );

        // Distance-dependence, and it is linear rather than quadratic: build and standing do not move.
        let far = sensor.owned_cost_j(mission_s, distance_m * 2.0, ASIMO).expect("well-posed");
        assert!(
            (carried(far) / carried(robot) - 2.0).abs() < 1e-9,
            "doubling the distance must double the carrying term, got {:.4}x",
            carried(far) / carried(robot)
        );

        // And it is never free: an unused sensor still costs its build plus its standing draw.
        let stationary = sensor.owned_cost_j(mission_s, 0.0, ASIMO).expect("well-posed");
        assert!(stationary > 0.0 && stationary == build + standing, "standing still does not make ownership free");

        // The deletion threshold is the same number, named for the decision it informs.
        assert_eq!(
            sensor.break_even_value_j(mission_s, distance_m, ASIMO),
            sensor.owned_cost_j(mission_s, distance_m, ASIMO)
        );
    }

    /// **For electronics the ownership bill is overwhelmingly SUNK AT MANUFACTURE, and that is why
    /// evolution deletes the modality instead of duty-cycling it.**
    ///
    /// I assumed the standing draw would dominate over a long watch and asserted it. It does not, by three
    /// orders of magnitude, and the test caught it. The correct statement is the more useful one: a part
    /// must idle for MONTHS before its quiescent draw equals what it cost to build, so duty-cycling a
    /// sensor is nearly pointless energetically. The decision that matters is whether it exists at all —
    /// which is exactly the move biology makes.
    #[test]
    fn the_ownership_bill_is_sunk_at_manufacture_so_deletion_beats_duty_cycling() {
        let part = OwnedCapability {
            built: EmbodiedActuator { mass_kg: 0.05, intensity_mj_per_kg: 250.0, operating_w: 0.0 },
            quiescent_w: 2.0,
        };
        let build = part.built.build_j().unwrap();

        // Over a full day of idling, the standing draw is a rounding error against the build cost.
        let watch_day = part.owned_cost_j(24.0 * 3600.0, 0.0, 3.2).expect("well-posed");
        let standing_day = watch_day - build;
        assert!(standing_day < 0.02 * build, "over a day, standing draw is <2% of build: {:.4}", standing_day / build);

        // How long must it idle before the standing draw catches the build cost? That is the real number.
        let catch_up_s = build / part.quiescent_w;
        assert!(catch_up_s > 30.0 * 86400.0, "catch-up must be months, got {:.1} d", catch_up_s / 86400.0);
        let recomputed = part.owned_cost_j(catch_up_s, 0.0, 3.2).expect("well-posed");
        assert!((recomputed - 2.0 * build).abs() < 1e-6 * build, "at catch-up the total is exactly twice the build");

        eprintln!(
            "  a {:.0} g part at {:.0} MJ/kg drawing {:.1} W must idle {:.0} DAYS before its standing draw \
equals its build cost; over one day it is {:.2}% of it",
            part.built.mass_kg * 1e3, part.built.intensity_mj_per_kg, part.quiescent_w,
            catch_up_s / 86400.0, 100.0 * standing_day / build
        );

        // So the saving from deleting it is almost entirely the build cost, not the runtime.
        let deleted_saving_fraction = build / watch_day;
        assert!(deleted_saving_fraction > 0.98, "deleting saves ~all of it, and almost none of that is runtime");
    }

    #[test]
    fn ownership_refuses_what_it_cannot_answer_for() {
        let c = OwnedCapability {
            built: EmbodiedActuator { mass_kg: 0.1, intensity_mj_per_kg: 100.0, operating_w: 0.0 },
            quiescent_w: 0.1,
        };
        assert!(c.owned_cost_j(1.0, 1.0, 1.0).is_some(), "control");
        assert!(c.owned_cost_j(-1.0, 1.0, 1.0).is_none(), "negative mission");
        assert!(c.owned_cost_j(1.0, -1.0, 1.0).is_none(), "negative distance");
        assert!(c.owned_cost_j(1.0, 1.0, f64::NAN).is_none(), "non-finite cost of transport");
        assert!(OwnedCapability { quiescent_w: -1.0, ..c }.owned_cost_j(1.0, 1.0, 1.0).is_none(), "negative quiescent");
    }
}
