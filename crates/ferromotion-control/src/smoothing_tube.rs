//! **Certifying a policy designed on a smoothed model, against the nonsmooth dynamics it will actually meet.**
//!
//! # The gap this closes
//!
//! [`crate::qp_certificate`] certifies a controller against the model it was designed on.
//! [`ferromotion_core::hybrid_jacobian`] gives the exact derivative of a nonsmooth trajectory. Neither answers the
//! question a practitioner actually has: *I optimised this policy through a smoothed contact because that is what
//! gives usable gradients. What happens when I run it on the real thing?*
//!
//! The published shape of the answer is to treat the deviation between the smoothed and nonsmooth dynamics as a
//! set-valued disturbance, propagate it through the closed loop as a reachable tube, and check constraints against
//! the tube rather than against the nominal trajectory. That is what this module does, on top of
//! [`crate::zonotope::Zonotope`].
//!
//! # Why the evidence type is not decoration
//!
//! Bounding the smoothing gap is the hard part, and it is where a certificate quietly stops being one. Measuring the
//! gap at a thousand sampled states gives a **lower** bound on its supremum; a tube built from it is an estimate
//! wearing a certificate's clothes. So [`GapEvidence`] is carried through every function here, and the asymmetry is
//! enforced by [`certify`] rather than left to the caller's discipline:
//!
//! - **Sampling can refute.** One sampled state whose tube violates a constraint is a real counterexample, and
//!   [`TubeVerdict::Refuted`] is sound on sampled evidence.
//! - **A bound still tagged [`GapEvidence::Sampled`] can never certify.** The best available verdict is
//!   [`TubeVerdict::Undecided`], at any margin, and [`certify`] enforces it.
//!
//! **What this does NOT do, stated plainly because an adversarial audit of this module caught the overstatement.**
//! Soundness is *relocated*, not eliminated. [`GapBound::assume_bound`] verifies nothing: it stamps
//! [`GapEvidence::Proved`] on whatever half-widths it is handed, so passing sampled numbers through it certifies. The
//! accompanying example and the shipped `certificate_lab` do exactly that on purpose, to show what a proof *would*
//! buy, and both label it as assumed. The type moves the unsupported step to one named, auditable call site; it does
//! not discharge it. Deriving a real Lipschitz constant for a stiff penalty contact remains open.
//!
//! Two defects the same audit found here, both now fixed and pinned by tests: [`certify`] read a trusted `sound`
//! boolean and never consulted the evidence, so a hand-built report certified regardless of its gaps; and
//! `propagate_tube(x0, &[])` certified every constraint because the loop that checks evidence never ran.
//!
//! # What is certified
//!
//! Given a nominal trajectory, a per-step closed-loop map, and a per-step gap set: that **every** state reachable
//! under the nonsmooth dynamics satisfies each half-space constraint at every step, and that the final reachable set
//! lies inside a goal set. The certificate is only as good as the gap bound and the linearisation, and
//! [`TubeReport`] reports both so neither is implicit.

use crate::zonotope::Zonotope;
use nalgebra::{DMatrix, DVector};

/// How a gap bound was obtained. A tube is only a certificate if this is [`GapEvidence::Proved`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GapEvidence {
    /// The maximum over a finite set of sampled states: a **lower** bound on the true supremum. Sound for refuting,
    /// never for certifying.
    Sampled { samples: usize },
    /// A bound valid over the whole domain, from a Lipschitz constant on a bounded region: `gap <= nominal + L * r`.
    Proved { lipschitz: f64, radius: f64 },
}

impl GapEvidence {
    /// Whether a certificate may rest on this.
    pub fn is_sound(&self) -> bool {
        matches!(self, GapEvidence::Proved { .. })
    }
}

/// A bound on the per-dimension half-width of the smoothing gap, with the evidence that produced it.
#[derive(Clone, Debug)]
pub struct GapBound {
    /// Half-width per state dimension. The gap set is the box `[-half_width, half_width]`.
    pub half_width: DVector<f64>,
    pub evidence: GapEvidence,
}

impl GapBound {
    /// From measurements at sampled states. `residuals` are the per-sample, per-dimension deviations of the smoothed
    /// one-step map from the nonsmooth one.
    ///
    /// Returns `None` on an empty sample or a non-finite residual, because a bound computed from a NaN is worse than
    /// no bound: it silently reads as zero under `max`.
    pub fn from_samples(residuals: &[DVector<f64>]) -> Option<GapBound> {
        let first = residuals.first()?;
        let n = first.len();
        let mut half: DVector<f64> = DVector::zeros(n);
        for r in residuals {
            if r.len() != n {
                return None;
            }
            for i in 0..n {
                if !r[i].is_finite() {
                    return None;
                }
                half[i] = half[i].max(r[i].abs());
            }
        }
        Some(GapBound { half_width: half, evidence: GapEvidence::Sampled { samples: residuals.len() } })
    }

    /// **Assume** a bound: the gap is at most `nominal + lipschitz * radius` in every dimension, over a ball of the
    /// given radius around the nominal trajectory.
    ///
    /// # This function verifies nothing
    ///
    /// It stamps [`GapEvidence::Proved`] on whatever half-widths it is handed. Pass it sampled numbers and they will
    /// certify. It was called `from_lipschitz` until an adversarial audit pointed out that the name reads as *derived
    /// from a Lipschitz argument* when the operation is *assert a bound*, and that the module's own example and shipped
    /// lab both route sampled gaps through it. The rename makes every call site self-indicting; it does not make the
    /// assertion any more discharged.
    ///
    /// It exists for two legitimate uses:
    /// - a gap that is **structurally** zero, e.g. a free-flight step both models integrate identically;
    /// - a **conditional** result, to show what a proof would buy before anyone spends the effort proving it.
    ///
    /// Any other use is a certificate resting on an unverified claim, and the claim is the caller's.
    // Negated comparisons are deliberate: `!(lipschitz >= 0.0)` rejects a NaN constant, where `lipschitz < 0.0`
    // would accept it and build a gap bound out of it.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn assume_bound(nominal: &DVector<f64>, lipschitz: f64, radius: f64) -> Option<GapBound> {
        if !(lipschitz >= 0.0) || !(radius >= 0.0) || nominal.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let half = nominal.map(|v| v.abs() + lipschitz * radius);
        Some(GapBound { half_width: half, evidence: GapEvidence::Proved { lipschitz, radius } })
    }

    /// **Rescale a bound while PRESERVING its evidence.**
    ///
    /// Needed because the tube's disturbance set is per step (`R_{k+1} = A R_k (+) W`), so a mismatch measured over a
    /// whole contact has to be divided across the steps the contact spans. Doing that through
    /// [`Self::assume_bound`] would stamp `Proved` on the slices and launder sampled numbers into a certificate — a
    /// bug that was written and caught by the evidence tests within minutes, which is the entire argument for having
    /// the evidence be a type rather than a comment.
    ///
    /// Returns `None` for a non-positive or non-finite divisor.
    // `!(n > 0.0)` rather than `n <= 0.0` so a NaN divisor is refused rather than used to build a bound.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn divided_by(&self, n: f64) -> Option<GapBound> {
        if !(n > 0.0) || !n.is_finite() {
            return None;
        }
        Some(GapBound { half_width: &self.half_width / n, evidence: self.evidence })
    }

    /// The gap set as a zonotope centred at the origin.
    pub fn as_set(&self) -> Zonotope {
        let n = self.half_width.len();
        let lo = -&self.half_width;
        let hi = self.half_width.clone();
        let _ = n;
        Zonotope::from_interval(&lo, &hi)
    }

    /// The largest half-width across dimensions, for reporting.
    pub fn magnitude(&self) -> f64 {
        self.half_width.iter().fold(0.0, |a: f64, v| a.max(v.abs()))
    }
}

/// One step of the linearised closed loop: `x_{t+1} = M x_t + w`, with `w` in the gap set.
///
/// `closed_loop` is `A + B K` for a time-varying affine policy, or a saltation-composed map across an impact. The
/// module does not care which, and that is the point: an exact hybrid Jacobian drops straight in.
#[derive(Clone, Debug)]
pub struct TubeStep {
    pub closed_loop: DMatrix<f64>,
    /// How far the smoothed dynamics depart from the nonsmooth ones over this step.
    pub gap: GapBound,
    /// **The linearisation residual**: how far the true map departs from `closed_loop * x` over this step's reachable
    /// set.
    ///
    /// This field exists because an adversarial review pointed out that the tube propagated `M*x + w` with *nothing*
    /// bounding the true map's departure from `M*x` — so proving the gap would have moved the leak one field over
    /// rather than closing it. Zero here is a **claim**, and like every other claim in this module it needs evidence:
    /// legitimate when the step is exactly linear (free flight is), and an assertion otherwise.
    pub residual: GapBound,
    /// A precondition the reachable set must satisfy for this step's bounds to hold at all, as `normal . x <= bound`.
    ///
    /// The motivating case: a free-flight step is given a structurally-zero gap because both models integrate the same
    /// quadratic — but only **above the plane**. The same review found that the first three flight steps of this
    /// module's own example have reachable sets containing `h < 0`, where free flight is the flow of neither model, so
    /// the zero was false. Attaching `-h <= 0` here makes the claim checkable, and [`certify`] refuses when it fails.
    pub precondition: Option<HalfSpace>,
}

impl TubeStep {
    /// A step whose gap and residual both need stating. Verbose on purpose: every argument is a claim.
    pub fn new(closed_loop: DMatrix<f64>, gap: GapBound, residual: GapBound) -> TubeStep {
        TubeStep { closed_loop, gap, residual, precondition: None }
    }

    /// Attach a precondition the reachable set must satisfy for this step's bounds to mean anything.
    pub fn requiring(mut self, precondition: HalfSpace) -> TubeStep {
        self.precondition = Some(precondition);
        self
    }

    /// An **exactly linear** step: free flight, where the map is linear by construction so the residual is
    /// structurally zero rather than assumed. Still takes a gap, and still wants a precondition if the linearity is
    /// only valid in a region.
    pub fn linear(closed_loop: DMatrix<f64>, gap: GapBound) -> Option<TubeStep> {
        let n = closed_loop.nrows();
        Some(TubeStep {
            closed_loop,
            gap,
            residual: GapBound::assume_bound(&DVector::zeros(n), 0.0, 0.0)?,
            precondition: None,
        })
    }
}

/// A half-space constraint `normal . x <= bound`.
#[derive(Clone, Debug)]
pub struct HalfSpace {
    pub normal: DVector<f64>,
    pub bound: f64,
}

impl HalfSpace {
    pub fn new(normal: DVector<f64>, bound: f64) -> HalfSpace {
        HalfSpace { normal, bound }
    }

    /// Worst-case violation over a set: positive means some reachable state violates the constraint.
    ///
    /// Uses the zonotope support function, which is exact, so no conservatism enters here. Any looseness in the
    /// verdict comes from the gap bound or the linearisation, not from this test.
    pub fn worst_case(&self, set: &Zonotope) -> f64 {
        set.support(&self.normal) - self.bound
    }
}

/// The verdict. Deliberately three-valued: "I could not show this" is a different answer from "this is false", and
/// collapsing them is how an unsound certificate gets shipped.
#[derive(Clone, Debug, PartialEq)]
pub enum TubeVerdict {
    /// Every reachable state satisfies every constraint, on sound evidence. `margin` is the smallest slack.
    Certified { margin: f64 },
    /// Some reachable state violates a constraint. Sound on sampled evidence too, since a sampled gap is a lower
    /// bound and a violation of a lower-bound tube is a violation of the true one.
    Refuted { step: usize, constraint: usize, violation: f64 },
    /// No constraint was shown to fail, but the evidence does not support a certificate.
    Undecided { reason: UndecidedReason },
}

/// Why a tube could not be certified. Named rather than a string so a caller can react to the specific reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UndecidedReason {
    /// The gap bound came from sampling, so it is a lower bound on the true gap.
    GapOnlySampled,
    /// The tube grew without bound over the horizon: no gap bound would help.
    TubeDiverged,
    /// A map or bound was not finite.
    NonFinite,
    /// The tube propagated no steps, so it establishes nothing however wide the constraint margin looks.
    EmptyTube,
    /// A step's bounds were only valid in a region and the reachable set left it. The classic case is a free-flight
    /// step given a structurally-zero gap whose reachable set dips below the plane.
    PreconditionViolated { step: usize },
    /// Dimensions did not agree across the inputs.
    DimensionMismatch,
}

/// The propagated tube plus everything needed to judge it.
#[derive(Clone, Debug)]
pub struct TubeReport {
    /// Reachable sets `R_0 … R_T`, in the coordinates of the deviation from the nominal trajectory.
    pub sets: Vec<Zonotope>,
    /// Per-step largest half-width of the tube, the honest one-number summary of how much the certificate gave away.
    pub widths: Vec<f64>,
    /// The evidence behind **every** step's gap bound, in order.
    ///
    /// Recorded per step rather than collapsed into a `sound: bool`, because a single trusted flag is one assignment
    /// away from certifying anything: an adversarial audit of this module found that [`certify`] read such a flag and
    /// never consulted the evidence, so a hand-built report with `sound: true` certified regardless of its gaps. The
    /// flag is gone and [`Self::is_sound`] derives the answer instead.
    pub evidence: Vec<GapEvidence>,
    /// Per-step preconditions, in order, so [`certify`] can check that the reachable set stayed where each step's
    /// bounds were valid.
    pub preconditions: Vec<Option<HalfSpace>>,
}

impl TubeReport {
    /// **Whether a certificate may rest on this tube.** Derived, never stored.
    ///
    /// A tube over **zero** steps is not sound. That is not pedantry: the same audit found that
    /// `propagate_tube(x0, &[])` produced a report that certified every constraint, because the loop that would have
    /// noticed weak evidence never ran. A tube that propagated nothing establishes nothing.
    pub fn is_sound(&self) -> bool {
        !self.evidence.is_empty() && self.evidence.iter().all(GapEvidence::is_sound)
    }

    /// The weakest evidence any step rested on, or `None` for a tube over no steps. Sampled evidence is weaker than
    /// proved, so a single sampled step is what this returns.
    pub fn weakest_evidence(&self) -> Option<GapEvidence> {
        self.evidence.iter().find(|e| !e.is_sound()).copied().or_else(|| self.evidence.first().copied())
    }

    /// The final tube width: how much uncertainty the policy is carrying at the end of the horizon.
    pub fn final_width(&self) -> f64 {
        self.widths.last().copied().unwrap_or(f64::NAN)
    }

    /// Growth factor from first to last step. Above one means the closed loop is amplifying the smoothing gap faster
    /// than it contracts, and the certificate will not close over a long horizon however tight the gap is.
    pub fn growth(&self) -> f64 {
        match (self.widths.first(), self.widths.last()) {
            (Some(a), Some(b)) if *a > 0.0 => b / a,
            _ => f64::NAN,
        }
    }
}

/// Propagate the tube: `R_0 = x0`, `R_{t+1} = M_t R_t (+) W_t`.
///
/// `x0` is the initial deviation set, in deviation coordinates, so it is usually a small box or a point.
pub fn propagate_tube(x0: &Zonotope, steps: &[TubeStep]) -> Option<TubeReport> {
    let n = x0.center.len();
    let mut sets = vec![x0.clone()];
    let mut widths = vec![half_width(x0)];
    let mut evidence: Vec<GapEvidence> = Vec::with_capacity(2 * steps.len());
    let mut preconditions: Vec<Option<HalfSpace>> = Vec::with_capacity(steps.len());

    for s in steps {
        let (r, c) = s.closed_loop.shape();
        if r != n || c != n || s.gap.half_width.len() != n {
            return None;
        }
        if s.residual.half_width.len() != n {
            return None;
        }
        if s.closed_loop.iter().any(|v| !v.is_finite())
            || s.gap.half_width.iter().any(|v| !v.is_finite())
            || s.residual.half_width.iter().any(|v| !v.is_finite())
        {
            return None;
        }
        // BOTH bounds are evidence the certificate rests on, so both are recorded. Omitting the residual is how the
        // linearisation leak stayed invisible.
        evidence.push(s.gap.evidence);
        evidence.push(s.residual.evidence);
        let next = sets
            .last()?
            .linear_map(&s.closed_loop)
            .minkowski_sum(&s.gap.as_set())
            .minkowski_sum(&s.residual.as_set());
        preconditions.push(s.precondition.clone());
        widths.push(half_width(&next));
        sets.push(next);
    }

    Some(TubeReport { sets, widths, evidence, preconditions })
}

/// Largest per-dimension half-width of a zonotope.
fn half_width(z: &Zonotope) -> f64 {
    let (lo, hi) = z.interval_hull();
    (0..lo.len()).map(|i| (hi[i] - lo[i]) * 0.5).fold(0.0, f64::max)
}

/// Certify a nominal trajectory plus tube against half-space constraints.
///
/// `nominal[t]` is the nominal state at step `t` and `tube.sets[t]` the deviation set around it, so the reachable set
/// at step `t` is `nominal[t] (+) sets[t]`.
///
/// The order of checks is deliberate. Refutation is attempted **first**, because it is sound on any evidence; only
/// once nothing can be refuted does the evidence quality decide between `Certified` and `Undecided`.
pub fn certify(nominal: &[DVector<f64>], tube: &TubeReport, constraints: &[HalfSpace]) -> TubeVerdict {
    if nominal.len() != tube.sets.len() {
        return TubeVerdict::Undecided { reason: UndecidedReason::DimensionMismatch };
    }
    if nominal.iter().any(|x| x.iter().any(|v| !v.is_finite())) || tube.widths.iter().any(|w| !w.is_finite()) {
        return TubeVerdict::Undecided { reason: UndecidedReason::NonFinite };
    }

    // Preconditions first: a step whose bounds are only valid in a region must have stayed in it, or every number
    // downstream is about a system the caller was not simulating.
    for (t, pre) in tube.preconditions.iter().enumerate() {
        if let Some(c) = pre {
            let (Some(x), Some(set)) = (nominal.get(t + 1), tube.sets.get(t + 1)) else { continue };
            if c.normal.len() != x.len() {
                return TubeVerdict::Undecided { reason: UndecidedReason::DimensionMismatch };
            }
            let shifted = Zonotope::new(&set.center + x, set.generators.clone());
            if c.worst_case(&shifted) > 0.0 {
                return TubeVerdict::Undecided { reason: UndecidedReason::PreconditionViolated { step: t + 1 } };
            }
        }
    }

    let mut margin = f64::INFINITY;
    for (t, (x, set)) in nominal.iter().zip(&tube.sets).enumerate() {
        for (i, c) in constraints.iter().enumerate() {
            if c.normal.len() != x.len() {
                return TubeVerdict::Undecided { reason: UndecidedReason::DimensionMismatch };
            }
            // Shift the deviation set to the nominal state, then take the exact support.
            let shifted = Zonotope::new(&set.center + x, set.generators.clone());
            let v = c.worst_case(&shifted);
            if v > 0.0 {
                return TubeVerdict::Refuted { step: t, constraint: i, violation: v };
            }
            margin = margin.min(-v);
        }
    }

    // Nothing refutable. Now the evidence decides, and a wide margin does not upgrade a sampled bound.
    if tube.evidence.is_empty() {
        return TubeVerdict::Undecided { reason: UndecidedReason::EmptyTube };
    }
    if !tube.is_sound() {
        return TubeVerdict::Undecided { reason: UndecidedReason::GapOnlySampled };
    }
    if !margin.is_finite() {
        return TubeVerdict::Undecided { reason: UndecidedReason::NonFinite };
    }
    TubeVerdict::Certified { margin }
}

/// The smallest slack the **nominal** trajectory leaves against any constraint, ignoring the tube entirely.
///
/// A certificate is worthless if the constraint could not have been violated over the horizon in the first place. That
/// failure is invisible in [`TubeVerdict::Certified`], which is perfectly true and says nothing. Comparing this number
/// against [`TubeReport::final_width`] is the check: if the nominal slack everywhere exceeds the tube width by a wide
/// factor, the horizon is too short or the bound too loose for the certificate to mean anything.
///
/// This exists because the first version of the accompanying example certified a ceiling constraint over a 60 ms
/// horizon when the mass needed 285 ms to reach its apex. The verdict was `Certified`. The constraint was unreachable.
pub fn nominal_activity(nominal: &[DVector<f64>], constraints: &[HalfSpace]) -> f64 {
    let mut slack = f64::INFINITY;
    for x in nominal {
        for c in constraints {
            if c.normal.len() == x.len() {
                slack = slack.min(c.bound - c.normal.dot(x));
            }
        }
    }
    slack
}

/// Whether the final reachable set lies inside a goal set, tested along the given probe directions.
///
/// Returns `None` if the dimensions disagree. The test is exact when `dirs` spans the goal set's facet normals; with
/// fewer directions it is a necessary condition only, which is why the directions are the caller's explicit input
/// rather than a hidden default.
pub fn reaches_goal(
    nominal_final: &DVector<f64>,
    tube: &TubeReport,
    goal: &Zonotope,
    dirs: &[DVector<f64>],
) -> Option<bool> {
    let set = tube.sets.last()?;
    if set.center.len() != nominal_final.len() || goal.center.len() != nominal_final.len() {
        return None;
    }
    let shifted = Zonotope::new(&set.center + nominal_final, set.generators.clone());
    // Every direction: the reachable set's support must not exceed the goal's.
    Some(dirs.iter().all(|d| shifted.support(d) <= goal.support(d) + 1e-12))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m2(a: f64, b: f64, c: f64, d: f64) -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[a, b, c, d])
    }

    /// **The soundness rule, pinned.** A sampled gap must never produce `Certified`, however wide the margin. This is
    /// the test that would have caught the unsound interval certificate this workspace shipped once before.
    #[test]
    fn a_sampled_gap_can_never_certify() {
        let gap = GapBound::from_samples(&[DVector::from_vec(vec![1e-9, 1e-9])]).unwrap();
        assert!(!gap.evidence.is_sound());
        let steps = vec![TubeStep::linear(m2(0.5, 0.0, 0.0, 0.5), gap).unwrap()];
        let tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).unwrap();
        let nominal = vec![DVector::zeros(2), DVector::zeros(2)];
        // A constraint with an enormous margin.
        let c = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 1e6)];
        match certify(&nominal, &tube, &c) {
            TubeVerdict::Undecided { reason } => assert_eq!(reason, UndecidedReason::GapOnlySampled),
            other => panic!("a sampled bound certified something: {other:?}"),
        }
    }

    /// The other half of the asymmetry: sampling **can** refute, because a sampled gap under-states the true gap and
    /// a violation of an under-stated tube is a real violation.
    #[test]
    fn a_sampled_gap_can_refute() {
        let gap = GapBound::from_samples(&[DVector::from_vec(vec![2.0, 0.0])]).unwrap();
        let steps = vec![TubeStep::linear(m2(1.0, 0.0, 0.0, 1.0), gap).unwrap()];
        let tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).unwrap();
        let nominal = vec![DVector::zeros(2), DVector::zeros(2)];
        let c = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 1.0)];
        match certify(&nominal, &tube, &c) {
            TubeVerdict::Refuted { step, violation, .. } => {
                assert_eq!(step, 1);
                assert!((violation - 1.0).abs() < 1e-12, "violation {violation}");
            }
            other => panic!("expected refutation, got {other:?}"),
        }
    }

    /// A proved gap on a contracting loop certifies, and the margin is the real slack rather than a placeholder.
    #[test]
    fn a_proved_gap_on_a_contracting_loop_certifies() {
        let gap = GapBound::assume_bound(&DVector::from_vec(vec![0.01, 0.01]), 0.0, 0.0).unwrap();
        assert!(gap.evidence.is_sound());
        let steps: Vec<TubeStep> =
            (0..20).map(|_| TubeStep::linear(m2(0.5, 0.0, 0.0, 0.5), gap.clone()).unwrap()).collect();
        let tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).unwrap();
        let nominal: Vec<DVector<f64>> = (0..21).map(|_| DVector::zeros(2)).collect();
        let c = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 0.1)];
        match certify(&nominal, &tube, &c) {
            TubeVerdict::Certified { margin } => {
                // Geometric series: the tube half-width tends to 0.01/(1-0.5) = 0.02, so the slack tends to 0.08.
                assert!((margin - 0.08).abs() < 1e-3, "margin {margin}");
            }
            other => panic!("expected a certificate, got {other:?}"),
        }
    }

    /// The tube must contract on a stable loop and grow on an unstable one, and the report must say which. A tube
    /// that grows cannot be rescued by a tighter gap, so this is the first thing to look at.
    #[test]
    fn tube_growth_tracks_the_closed_loop() {
        let gap = GapBound::assume_bound(&DVector::from_vec(vec![0.01, 0.01]), 0.0, 0.0).unwrap();
        let stable: Vec<TubeStep> =
            (0..30).map(|_| TubeStep::linear(m2(0.5, 0.0, 0.0, 0.5), gap.clone()).unwrap()).collect();
        let unstable: Vec<TubeStep> =
            (0..30).map(|_| TubeStep::linear(m2(1.2, 0.0, 0.0, 1.2), gap.clone()).unwrap()).collect();

        let x0 = Zonotope::from_interval(&DVector::from_vec(vec![-0.05, -0.05]), &DVector::from_vec(vec![0.05, 0.05]));
        let a = propagate_tube(&x0, &stable).unwrap();
        let b = propagate_tube(&x0, &unstable).unwrap();
        assert!(a.final_width() < 0.05, "stable loop should contract: {}", a.final_width());
        assert!(b.final_width() > 1.0, "unstable loop should blow up: {}", b.final_width());
        assert!(a.growth() < 1.0 && b.growth() > 1.0, "growth {} vs {}", a.growth(), b.growth());
    }

    /// A NaN residual must not read as a zero gap under `max`. This is the exact shape of a bug this workspace has
    /// hit before, where `f64::max` swallowed a NaN and a divergence was reported as zero error.
    #[test]
    fn a_nan_residual_is_refused_not_maxed_away() {
        let bad = vec![DVector::from_vec(vec![1.0, f64::NAN])];
        assert!(GapBound::from_samples(&bad).is_none());
        assert!(GapBound::from_samples(&[]).is_none());
        // and a mismatched dimension is refused rather than truncated
        let ragged = vec![DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![1.0])];
        assert!(GapBound::from_samples(&ragged).is_none());
    }

    /// The support-function test is exact for a zonotope, so a box tube touching a constraint exactly must read as
    /// zero violation rather than a small positive or negative number.
    #[test]
    fn the_constraint_test_is_exact_on_a_box() {
        let z = Zonotope::from_interval(&DVector::from_vec(vec![-1.0, -2.0]), &DVector::from_vec(vec![1.0, 2.0]));
        let c = HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 1.0);
        assert!(c.worst_case(&z).abs() < 1e-15, "{}", c.worst_case(&z));
        // and a diagonal direction picks up both half-widths
        let diag = HalfSpace::new(DVector::from_vec(vec![1.0, 1.0]), 0.0);
        assert!((diag.worst_case(&z) - 3.0).abs() < 1e-15, "{}", diag.worst_case(&z));
    }

    /// A certificate over a horizon where the constraint cannot be reached is true and worthless. The activity check
    /// has to expose that, because the verdict never will.
    #[test]
    fn a_vacuous_certificate_is_visible_in_the_activity_check() {
        let gap = GapBound::assume_bound(&DVector::from_vec(vec![0.01, 0.0]), 0.0, 0.0).unwrap();
        let steps: Vec<TubeStep> =
            (0..10).map(|_| TubeStep::linear(m2(1.0, 0.0, 0.0, 1.0), gap.clone()).unwrap()).collect();
        let tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).unwrap();

        // A nominal trajectory that never goes anywhere near the constraint.
        let far: Vec<DVector<f64>> = (0..11).map(|_| DVector::from_vec(vec![0.0, 0.0])).collect();
        let c = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 100.0)];
        assert!(matches!(certify(&far, &tube, &c), TubeVerdict::Certified { .. }));
        // ... and the activity check shows why that certificate says nothing: 100 of slack against a 0.1 tube.
        let slack = nominal_activity(&far, &c);
        assert!(slack / tube.final_width() > 100.0, "slack {slack} vs width {}", tube.final_width());

        // A nominal that actually approaches the constraint gives a slack comparable to the tube, which is when the
        // certificate is worth having.
        let near: Vec<DVector<f64>> = (0..11).map(|_| DVector::from_vec(vec![99.95, 0.0])).collect();
        assert!(nominal_activity(&near, &c) < tube.final_width());
    }

    /// **Defect found by adversarial audit.** A tube that propagated no steps used to certify every constraint,
    /// because the loop that would have noticed weak evidence never ran and the trusted `sound` flag stayed `true`.
    /// A tube over nothing establishes nothing.
    #[test]
    fn a_tube_over_no_steps_certifies_nothing() {
        let tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &[]).unwrap();
        assert!(!tube.is_sound(), "an empty tube must not be sound");
        assert_eq!(tube.weakest_evidence(), None);
        let nominal = vec![DVector::zeros(2)];
        let c = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), 1e6)];
        match certify(&nominal, &tube, &c) {
            TubeVerdict::Undecided { reason } => assert_eq!(reason, UndecidedReason::EmptyTube),
            other => panic!("an empty tube returned {other:?}"),
        }
    }

    /// **Defect found by adversarial audit.** Soundness used to be a single `pub bool` that `certify` trusted, so a
    /// hand-built report could flip it and certify anything. It is now derived from the per-step evidence, so a forgery
    /// has to rewrite the whole record rather than one field.
    #[test]
    fn soundness_is_derived_from_the_evidence_not_a_trusted_flag() {
        let sampled = GapBound::from_samples(&[DVector::from_vec(vec![1e-9, 1e-9])]).unwrap();
        let steps = vec![TubeStep::linear(m2(0.5, 0.0, 0.0, 0.5), sampled).unwrap()];
        let mut tube = propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).unwrap();
        assert!(!tube.is_sound());
        assert!(matches!(tube.weakest_evidence(), Some(GapEvidence::Sampled { .. })));

        // The only way to make this tube read as sound is to replace the recorded evidence, which is exactly the
        // explicit act the old boolean let a caller perform by accident.
        tube.evidence = vec![GapEvidence::Proved { lipschitz: 0.0, radius: 0.0 }];
        assert!(tube.is_sound(), "an honest record is what soundness now means");
    }

    /// Dimension disagreement is refused, not silently broadcast.
    #[test]
    fn mismatched_dimensions_are_refused() {
        let gap = GapBound::assume_bound(&DVector::from_vec(vec![0.01, 0.01]), 0.0, 0.0).unwrap();
        let steps = vec![TubeStep::linear(DMatrix::identity(3, 3), gap).unwrap()];
        assert!(propagate_tube(&Zonotope::point(DVector::zeros(2)), &steps).is_none());
    }
}
