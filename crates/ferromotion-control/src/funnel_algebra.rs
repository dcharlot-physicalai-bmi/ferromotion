//! **A funnel algebra for learned skills** — composing certified basins, and what has to change when the skills are
//! stochastic.
//!
//! Burridge, Rizzi and Koditschek's result is the classical composition theorem: controllers with certified domains
//! of attraction compose by **preimage ordering** — if one skill's goal set lies inside the next skill's domain of
//! attraction, switching on containment yields a certified composite. That is the funnel algebra, and it is exact.
//! It is also all-or-nothing: one containment failure anywhere in a chain and there is no certificate at all.
//!
//! Learned skills do not satisfy it. Their basins hold with a *probability*, estimated from finite trials
//! ([`metrology`](crate::metrology)), so a chain of ten skills each certified "with 95% confidence" carries a
//! deterministic certificate for none of them. The naive repair — multiply the confidences — is exactly the serial
//! fragility of [`reliability`](crate::reliability): `0.95¹⁰ = 0.60`, and a longer task becomes hopeless.
//!
//! The repair that works is the one the reliability calculus already identified, and it is not "add probabilities to
//! the funnels". It is **detection**. A containment failure that is *noticed* costs a retry; one that is not costs
//! the task. So the composition operator this module supplies carries three things per skill — a certified inflow
//! set, an outflow set, and a detection probability — and the resulting chain certificate degrades gracefully in a
//! way the deterministic algebra cannot and the naive probabilistic one does not.
//!
//! The gap between the three is measured in the tests, and it is large: on a ten-skill chain the deterministic
//! algebra refuses, the naive product gives 0.5987, and the detection-aware composition gives 0.9489 at the same
//! per-skill numbers.
//!
//! How the sets were *measured* then turns out to matter more than any of that — see [`sampled_set_escape_rate`] and
//! [`set_aware_reliability`].

use crate::Skill;

/// An axis-aligned box, which is what a certified basin usually is in practice — an interval per coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}

impl Region {
    /// A box, or `None` if any interval is inverted.
    pub fn new(lo: Vec<f64>, hi: Vec<f64>) -> Option<Region> {
        (lo.len() == hi.len() && !lo.is_empty() && lo.iter().zip(&hi).all(|(a, b)| a <= b)).then_some(Region { lo, hi })
    }

    /// A one-dimensional interval, the common case — a section coordinate's basin.
    pub fn interval(lo: f64, hi: f64) -> Option<Region> {
        Region::new(vec![lo], vec![hi])
    }

    pub fn dim(&self) -> usize {
        self.lo.len()
    }

    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dim() && self.lo.iter().zip(&self.hi).zip(x).all(|((l, h), v)| l <= v && v <= h)
    }

    /// Whether this region is contained in `other` — the **preimage ordering** the composition theorem switches on.
    pub fn subset_of(&self, other: &Region) -> bool {
        self.dim() == other.dim() && self.lo.iter().zip(&other.lo).all(|(a, b)| a >= b) && self.hi.iter().zip(&other.hi).all(|(a, b)| a <= b)
    }

    /// How far outside `other` this region reaches, per coordinate; zero everywhere iff contained. Reported rather
    /// than a bare boolean, because "by how much does the chain break" is the actionable number.
    pub fn overhang(&self, other: &Region) -> Vec<f64> {
        if self.dim() != other.dim() {
            return vec![f64::INFINITY; self.dim()];
        }
        (0..self.dim()).map(|i| (other.lo[i] - self.lo[i]).max(0.0).max((self.hi[i] - other.hi[i]).max(0.0))).collect()
    }
}

/// A skill with a certified funnel: it carries any state in `inflow` into `outflow`, and does so with probability at
/// least `reliability`. `detection` is the probability a failure to do so is noticed.
#[derive(Clone, Debug)]
pub struct SkillFunnel {
    pub name: &'static str,
    pub inflow: Region,
    pub outflow: Region,
    /// Probability the skill delivers its outflow guarantee, given an inflow state.
    pub reliability: f64,
    /// Probability a failure is detected, so that a recovery can retry instead of the task ending.
    pub detection: f64,
}

impl SkillFunnel {
    /// Whether this skill's outflow lies inside `next`'s inflow — interface compatibility, and the only *structural*
    /// condition the composition needs.
    pub fn composes_with(&self, next: &SkillFunnel) -> bool {
        self.outflow.subset_of(&next.inflow)
    }
}

/// Why a chain failed to compose structurally.
#[derive(Clone, Debug, PartialEq)]
pub struct Incompatibility {
    /// Index of the skill whose outflow does not fit the next skill's inflow.
    pub at: usize,
    pub from: &'static str,
    pub to: &'static str,
    /// How far outside, per coordinate — what would have to shrink for the chain to certify.
    pub overhang: Vec<f64>,
}

/// The certificate a chain carries.
#[derive(Clone, Debug)]
pub struct ChainCertificate {
    /// The composite's inflow: the first skill's, unchanged.
    pub inflow: Region,
    /// The composite's outflow: the last skill's.
    pub outflow: Region,
    /// **Deterministic** algebra: certified only if every skill is certain. All-or-nothing.
    pub deterministic: bool,
    /// **Naive probabilistic**: the product of reliabilities, which is the serial-fragility answer.
    pub naive_reliability: f64,
    /// **Detection-aware**: `∏ p/(p + (1−p)(1−d))`, treating a detected containment failure as a retry.
    pub with_detection: f64,
    pub skills: usize,
}

/// **Compose a chain of skill funnels.**
///
/// Returns the composite certificate, or the first structural incompatibility. Structural compatibility is checked
/// first and separately from reliability, because they fail for different reasons and admit different fixes: an
/// overhang is a *design* problem (shrink an outflow or widen an inflow) while a low reliability is a *training or
/// detection* problem.
pub fn compose_chain(skills: &[SkillFunnel]) -> Result<ChainCertificate, Incompatibility> {
    if skills.is_empty() {
        return Err(Incompatibility { at: 0, from: "", to: "", overhang: vec![] });
    }
    for i in 0..skills.len() - 1 {
        if !skills[i].composes_with(&skills[i + 1]) {
            return Err(Incompatibility { at: i, from: skills[i].name, to: skills[i + 1].name, overhang: skills[i].outflow.overhang(&skills[i + 1].inflow) });
        }
    }
    let naive = skills.iter().map(|s| s.reliability).product();
    let detected: Vec<Skill> = skills.iter().filter_map(|s| Skill::new(s.reliability, s.detection, 1.0)).collect();
    let with_detection = if detected.len() == skills.len() { crate::compose(&detected).success } else { naive };
    Ok(ChainCertificate {
        inflow: skills[0].inflow.clone(),
        outflow: skills[skills.len() - 1].outflow.clone(),
        deterministic: skills.iter().all(|s| s.reliability >= 1.0),
        naive_reliability: naive,
        with_detection,
        skills: skills.len(),
    })
}

/// **Widen a skill's inflow to accept its predecessor's outflow**, returning the widened region — the repair a
/// structural incompatibility calls for, stated as a requirement on the skill rather than as a failure.
///
/// This is what makes the algebra usable as a design tool: an overhang says *how much more basin* the next skill
/// needs, which is a specification a controller can be built or retrained against.
pub fn required_inflow(previous_outflow: &Region, current_inflow: &Region) -> Option<Region> {
    if previous_outflow.dim() != current_inflow.dim() {
        return None;
    }
    Region::new(
        (0..current_inflow.dim()).map(|i| current_inflow.lo[i].min(previous_outflow.lo[i])).collect(),
        (0..current_inflow.dim()).map(|i| current_inflow.hi[i].max(previous_outflow.hi[i])).collect(),
    )
}

/// **The escape rate of a sampled set-certificate**: `2d/(n+1)`.
///
/// A funnel measured by running a skill from `n` sampled starts and taking the bounding box of the images has `2d`
/// faces, and a fresh image exceeds the sample maximum along any one direction with probability `1/(n+1)`. So a
/// certificate built this way is violated at rate about `2d/(n+1)` — a fact about bounding boxes, not about the
/// dynamics. Verified over an ensemble in `examples/r7_funnel_chain.rs` for `d = 2, 4, 8`.
///
/// This is the slot the classical composition theorem lacks. It takes the inflow and outflow sets as exact, and the
/// reliability arithmetic in [`compose_chain`] never sees how they were obtained, so a 50-sample basin estimate and a
/// sound one produce identical certificates that differ by 15 percentage points on hardware.
pub fn sampled_set_escape_rate(dim: usize, samples: usize) -> f64 {
    (2.0 * dim as f64 / (samples as f64 + 1.0)).min(1.0)
}

/// How many rollouts a skill's funnel needs for its handoff to hold at rate `target`. Linear in dimension, which is
/// what makes it affordable: a 30-DOF humanoid at 0.1% needs 120_000 rollouts per skill, not `2^60` corner probes.
pub fn samples_for_handoff(dim: usize, target: f64) -> Option<usize> {
    (target > 0.0 && target < 1.0 && dim > 0).then(|| (2.0 * dim as f64 / target).ceil() as usize)
}

/// **A chain certificate that accounts for how its sets were measured.**
///
/// Each link carries a containment failure at rate [`sampled_set_escape_rate`] on top of the skill's own
/// unreliability. The important part is the second argument: a containment breach is **checkable at runtime** — the
/// box is right there, and testing membership costs `2d` comparisons. So unlike a policy failure, this failure mode
/// comes with detection for free, and `monitored` says what that is worth.
///
/// What a monitor does, precisely. A link ends three ways: it succeeds with `p = r(1-eps)`, it dies with
/// `dead`, or it retries. A monitor moves the breach out of `dead` and into `retry`:
///
/// ```text
///   unmonitored:  dead = (1-r)(1-d_skill) + r*eps      <- a silent breach ends the task
///     monitored:  dead = (1-r)(1-d_skill)              <- a caught breach costs an attempt
///   pass = p / (p + dead)
/// ```
///
/// So `eps` does **not** cancel: it stays in the numerator through `p = r(1-eps)`. A monitor converts
/// set-measurement error from a *failure* into a *delay*, which is a real and large gain but not a free one. On a
/// 500-link chain at `eps = 0.195` the monitored chain passes at 0.883 against 0.905 with exact sets, while the
/// retries needed climb to over a hundred — and under any finite retry budget that is what binds. See
/// [`TaskDecomposition`](crate::TaskDecomposition) and `examples/r7_decomposition.rs`, where the simulation caught an
/// earlier version of this function that had a spurious `(1-eps)` in the detection term and therefore reported an
/// exact cancellation that does not exist.
///
/// A box-membership test is exact only in the coordinates you can see. Under partial observation the breach is
/// detected on an estimate, so pass `monitored = false` when the handoff state is not measured.
pub fn set_aware_reliability(skills: &[SkillFunnel], dim: usize, samples: usize, monitored: bool) -> Option<f64> {
    let eps = sampled_set_escape_rate(dim, samples);
    let links: Vec<Skill> = skills
        .iter()
        .map(|s| {
            // the link succeeds only if the skill succeeds AND the handoff stays inside the certified inflow
            let p = s.reliability * (1.0 - eps);
            // what ends the task outright: a skill failure its own detector missed, plus - without a monitor - every
            // silent containment breach. A breach the skill itself never sees is not covered by its detector.
            let dead = (1.0 - s.reliability) * (1.0 - s.detection) + if monitored { 0.0 } else { s.reliability * eps };
            let d = 1.0 - dead / (1.0 - p).max(1e-300);
            Skill::new(p, d.clamp(0.0, 1.0), 1.0)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(crate::compose(&links).success)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(name: &'static str, in_lo: f64, in_hi: f64, out_lo: f64, out_hi: f64, rel: f64, det: f64) -> SkillFunnel {
        SkillFunnel { name, inflow: Region::interval(in_lo, in_hi).unwrap(), outflow: Region::interval(out_lo, out_hi).unwrap(), reliability: rel, detection: det }
    }

    /// The preimage ordering is the structural condition, and containment is what it checks.
    #[test]
    fn preimage_ordering_is_containment() {
        let a = skill("reach", 0.0, 1.0, 0.4, 0.6, 1.0, 1.0);
        let b = skill("grasp", 0.3, 0.7, 0.0, 0.1, 1.0, 1.0);
        let c = skill("lift", 0.45, 0.7, 0.0, 0.1, 1.0, 1.0); // inflow too narrow for a's outflow
        assert!(a.composes_with(&b), "a's outflow [0.4, 0.6] sits inside b's inflow [0.3, 0.7]");
        assert!(!a.composes_with(&c), "and does not sit inside [0.45, 0.7]");
        // the overhang says by how much, which is the actionable number
        let over = a.outflow.overhang(&c.inflow);
        eprintln!("reach -> lift overhang: {over:?} (a's outflow starts 0.05 below lift's inflow)");
        assert!((over[0] - 0.05).abs() < 1e-12);
        assert!(a.outflow.overhang(&b.inflow).iter().all(|v| *v == 0.0), "a compatible pair has zero overhang");
    }

    /// A chain composes or names the first break, with the repair stated as a requirement rather than a refusal.
    #[test]
    fn a_broken_chain_names_the_break_and_the_repair() {
        let chain = vec![skill("reach", 0.0, 1.0, 0.4, 0.6, 1.0, 1.0), skill("grasp", 0.45, 0.7, 0.2, 0.3, 1.0, 1.0), skill("place", 0.0, 1.0, 0.0, 0.1, 1.0, 1.0)];
        let err = compose_chain(&chain).expect_err("this chain does not compose");
        eprintln!("break at index {}: {} -> {}, overhang {:?}", err.at, err.from, err.to, err.overhang);
        assert_eq!((err.at, err.from, err.to), (0, "reach", "grasp"));
        // and the repair: what grasp's inflow would have to be
        let need = required_inflow(&chain[0].outflow, &chain[1].inflow).unwrap();
        eprintln!("   grasp's inflow must widen from [{}, {}] to [{}, {}]", chain[1].inflow.lo[0], chain[1].inflow.hi[0], need.lo[0], need.hi[0]);
        assert!((need.lo[0] - 0.4).abs() < 1e-12 && (need.hi[0] - 0.7).abs() < 1e-12);
        // with that widening the chain certifies
        let mut fixed = chain.clone();
        fixed[1].inflow = need;
        let cert = compose_chain(&fixed).expect("now it composes");
        assert!(cert.deterministic, "all reliabilities are 1, so the deterministic certificate holds");
        assert_eq!((cert.inflow.lo[0], cert.outflow.hi[0]), (0.0, 0.1), "the composite inherits the ends");
    }

    /// **The three algebras on the same chain**, which is the point of the module. Ten learned skills at 95%
    /// reliability and 90% detection: the deterministic algebra refuses outright, the naive product collapses, and
    /// the detection-aware composition holds.
    #[test]
    fn detection_is_what_makes_a_learned_funnel_algebra_work() {
        let chain: Vec<SkillFunnel> = (0..10).map(|_| skill("step", 0.0, 1.0, 0.2, 0.8, 0.95, 0.9)).collect();
        let cert = compose_chain(&chain).expect("structurally compatible");
        eprintln!("ten skills, each 95% reliable with 90% detection:");
        eprintln!("      deterministic algebra: {}   (certified at all? no - not one skill is certain)", cert.deterministic);
        eprintln!("      naive product:         {:.4}", cert.naive_reliability);
        eprintln!("      with detection:        {:.4}", cert.with_detection);
        assert!(!cert.deterministic, "the classical algebra has nothing to say about a 95% skill");
        assert!((cert.naive_reliability - 0.95f64.powi(10)).abs() < 1e-12);
        assert!(cert.naive_reliability < 0.61, "serial fragility: {:.4}", cert.naive_reliability);
        // 0.95 / (0.95 + 0.05 * 0.1) = 0.994764 per skill, so 0.994764^10 = 0.9489
        assert!((cert.with_detection - 0.9489).abs() < 5e-5, "detection rescues it: {:.4}", cert.with_detection);
        eprintln!("\n      the same per-skill numbers give 0.5987 or 0.9489 depending only on whether failures are NOTICED.");
        eprintln!("      That is the whole content of a stochastic funnel calculus: detection is the certified channel,");
        eprintln!("      and adding probabilities to funnels without it just reproduces serial fragility.");

        // and the gap widens with the chain length, which is the regime that matters
        for n in [5usize, 10, 20, 50] {
            let c = compose_chain(&(0..n).map(|_| skill("step", 0.0, 1.0, 0.2, 0.8, 0.95, 0.9)).collect::<Vec<_>>()).unwrap();
            eprintln!("      n = {n:>2}: naive {:.4}, with detection {:.4}  (ratio {:.1}x)", c.naive_reliability, c.with_detection, c.with_detection / c.naive_reliability);
        }
        let long = compose_chain(&(0..50).map(|_| skill("step", 0.0, 1.0, 0.2, 0.8, 0.95, 0.9)).collect::<Vec<_>>()).unwrap();
        // 0.994764^50 / 0.95^50 = 9.996 - the printed "10.0x" above is rounding, not the true ratio
        assert!(long.with_detection / long.naive_reliability > 9.9, "the advantage compounds with length: {:.3}x", long.with_detection / long.naive_reliability);
    }

    /// The set-certification law, and what it costs at scale.
    #[test]
    fn a_sampled_funnel_escapes_at_two_d_over_n() {
        // matches the ensemble measurement in examples/r7_funnel_chain.rs
        for (dim, n, expect) in [(2usize, 50usize, 0.078431), (4, 200, 0.039801), (8, 1000, 0.015984)] {
            assert!((sampled_set_escape_rate(dim, n) - expect).abs() < 1e-6);
        }
        assert_eq!(samples_for_handoff(4, 0.001), Some(8_000), "a 4-D funnel at 0.1%");
        assert_eq!(samples_for_handoff(60, 0.001), Some(120_000), "a 30-DOF humanoid at 0.1%");
        eprintln!("set certification is LINEAR in dimension: d=4 -> 8000 rollouts, d=60 -> 120000.");
        eprintln!("   The exact route costs 2^d corner probes: d=60 -> 1.15e18. Linear is the affordable one.");
        assert!(sampled_set_escape_rate(100, 10) <= 1.0, "the rate is a probability, so it clamps");
    }

    /// **The runtime containment monitor.** A containment breach is the one failure mode always cheap to detect, so
    /// the set-level error can be moved from the undetected column to the detected one for 2d comparisons.
    #[test]
    fn a_containment_monitor_pays_for_itself() {
        let chain: Vec<SkillFunnel> = (0..10).map(|_| skill("step", 0.0, 1.0, 0.2, 0.8, 0.95, 0.9)).collect();
        let ideal = compose_chain(&chain).unwrap().with_detection;
        for n in [200usize, 1_000, 8_000] {
            let unmonitored = set_aware_reliability(&chain, 4, n, false).unwrap();
            let monitored = set_aware_reliability(&chain, 4, n, true).unwrap();
            eprintln!("n = {n:>5} (escape {:.4}%): unmonitored {unmonitored:.4}, monitored {monitored:.4}", 100.0 * sampled_set_escape_rate(4, n));
            assert!(monitored >= unmonitored - 1e-12, "a monitor never hurts");
            // but it does NOT recover the ideal: eps stays in the numerator through p = r(1-eps)
            assert!(monitored <= ideal + 1e-12, "and never beats exact sets: {monitored} vs {ideal}");
            assert!(unmonitored <= ideal + 1e-12, "accounting for the sets never flatters the chain");
        }
        eprintln!("   (ideal, sets taken as exact: {ideal:.4})");
        // the monitor matters most where the set error is largest
        let (m_lo, u_lo) = (set_aware_reliability(&chain, 4, 200, true).unwrap(), set_aware_reliability(&chain, 4, 200, false).unwrap());
        let (m_hi, u_hi) = (set_aware_reliability(&chain, 4, 8_000, true).unwrap(), set_aware_reliability(&chain, 4, 8_000, false).unwrap());
        eprintln!("   monitor gains {:.4} at n=200 and {:.4} at n=8000 - it buys the most where the sets are worst", m_lo - u_lo, m_hi - u_hi);
        assert!(m_lo - u_lo > m_hi - u_hi, "so the cheap fix substitutes for expensive measurement");
        eprintln!("\n   The monitor moves a breach from the fatal column to the retry column, so it recovers most");
        eprintln!("   but not all of the gap to exact sets - eps stays in p = r(1-eps). A 200-rollout basin estimate");
        eprintln!("   with a monitor still beats an 8000-rollout one without: {m_lo:.4} vs {u_hi:.4}.");
        assert!(m_lo > u_hi, "cheap sets plus a monitor beat expensive sets without one");
        assert!(m_lo < ideal, "but a monitor is not a substitute for measuring the set: {m_lo:.6} < {ideal:.6}");
    }

    /// Structural compatibility and reliability fail for different reasons and are reported separately, because the
    /// fixes are different: an overhang is a design problem, a low reliability is a training or detection one.
    #[test]
    fn the_two_failure_modes_stay_separate() {
        // structurally broken but individually perfect
        let broken = vec![skill("a", 0.0, 1.0, 0.0, 0.9, 1.0, 1.0), skill("b", 0.0, 0.5, 0.0, 0.1, 1.0, 1.0)];
        assert!(compose_chain(&broken).is_err(), "perfect skills that do not fit still do not compose");

        // structurally fine but unreliable
        let shaky = vec![skill("a", 0.0, 1.0, 0.2, 0.4, 0.7, 0.0), skill("b", 0.0, 1.0, 0.0, 0.1, 0.7, 0.0)];
        let cert = compose_chain(&shaky).expect("structurally fine");
        assert!(!cert.deterministic && (cert.naive_reliability - 0.49).abs() < 1e-12);
        assert!((cert.with_detection - 0.49).abs() < 1e-12, "with zero detection, detection-aware reduces to naive");
        eprintln!("structurally fine, 70% skills, no detection: naive {:.2} = with-detection {:.2}", cert.naive_reliability, cert.with_detection);
        eprintln!("   so the detection-aware operator is a strict generalisation - it agrees with the naive product");
        eprintln!("   exactly when nothing is detected, which is the sanity check that it is not double-counting.");
    }
}
