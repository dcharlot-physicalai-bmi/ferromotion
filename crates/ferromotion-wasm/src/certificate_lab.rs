//! **What a certificate can and cannot claim** — the on-device lab behind the verification lesson.
//!
//! Most people meet verification as a number: a margin, a bound, a percentage. This lab is built to break that habit.
//! The learner gets three controls and watches a verdict move between three values that are *not* a spectrum:
//!
//! - **Certified** — every reachable state satisfies the constraint, on evidence that supports the claim.
//! - **Refuted** — some reachable state violates it. A real counterexample.
//! - **Undecided** — nothing was shown to fail, and nothing was shown to hold either.
//!
//! The third one is the lesson. A learner who has only ever seen pass/fail will read `Undecided` as a soft pass, and
//! the whole point is that it is not one.
//!
//! The system underneath is real: a mass dropped onto a plane, a ceiling it must not exceed on the way back up, and
//! the gap between a smoothed contact model and the rigid one propagated as a reachable tube
//! ([`ferromotion_control::propagate_tube`]). Three things are directly manipulable, and each one teaches a different
//! way a certificate fails:
//!
//! | control | what it demonstrates |
//! |---|---|
//! | contact stiffness | the tube is only tight enough to certify at high stiffness |
//! | evidence quality | a sampled gap can refute and can never certify, at any margin |
//! | horizon | a certificate over a horizon where the constraint cannot bite is true and worthless |

use ferromotion_control::{
    certify, nominal_activity, propagate_tube, GapBound, HalfSpace, TubeStep, TubeVerdict, Zonotope,
};
use ferromotion_core::{AffineContact, BouncingMass, PenaltyMass, GRAVITY};
use nalgebra::{DMatrix, DVector};
use wasm_bindgen::prelude::*;

const H0: f64 = 1.0;
const DT: f64 = 1e-3;
/// Damping ratio held fixed as stiffness sweeps, so the realised restitution stays put and only the contact
/// resolution changes.
const ZETA: f64 = 0.1606;

#[wasm_bindgen]
pub struct CertificateLab {
    log_stiffness: f64,
    ceiling: f64,
    horizon: usize,
    /// Whether the gap bound is presented as proved. Sampled is the honest default, because that is what a
    /// measurement actually gives you.
    proved: bool,
    /// Whether the flight steps' preconditions are checked.
    ///
    /// A fourth control, and the one that found a real modelling error in this very lab. The flight steps claim a zero
    /// gap because both models integrate the same quadratic — true only ABOVE the plane. With checking on, the tube's
    /// reachable set is found to dip below it and the verdict is honestly `PreconditionViolated`. With it off you get
    /// the verdict the lab used to report, which is what an unchecked assumption buys you.
    check_preconditions: bool,
}

#[wasm_bindgen]
impl CertificateLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CertificateLab {
        CertificateLab { log_stiffness: 6.0, ceiling: 0.46, horizon: 340, proved: false, check_preconditions: true }
    }

    pub fn set_log_stiffness(&mut self, v: f64) {
        self.log_stiffness = v.clamp(3.0, 8.0);
    }

    pub fn set_ceiling(&mut self, v: f64) {
        self.ceiling = v.clamp(0.05, 1.2);
    }

    /// Horizon in control steps. Short horizons are where vacuous certificates come from, so this is exposed rather
    /// than fixed at a value that always works.
    pub fn set_horizon(&mut self, steps: f64) {
        self.horizon = (steps.max(1.0) as usize).clamp(1, 800);
    }

    /// Claim the gap bound is proved rather than sampled. Nothing about the geometry changes when this flips, which is
    /// the point: the verdict tracks the evidence, not the numbers.
    pub fn set_proved(&mut self, proved: bool) {
        self.proved = proved;
    }

    pub fn stiffness(&self) -> f64 {
        10f64.powf(self.log_stiffness)
    }

    pub fn horizon(&self) -> f64 {
        self.horizon as f64
    }

    pub fn is_proved(&self) -> bool {
        self.proved
    }

    /// Turn the flight steps' precondition checking on or off. On is the honest default.
    pub fn set_check_preconditions(&mut self, on: bool) {
        self.check_preconditions = on;
    }

    pub fn checks_preconditions(&self) -> bool {
        self.check_preconditions
    }

    fn impact_speed() -> f64 {
        (2.0 * GRAVITY * H0).sqrt()
    }

    fn models(&self) -> Option<(PenaltyMass, BouncingMass, f64)> {
        let k = self.stiffness();
        let d = 2.0 * ZETA * k.sqrt();
        let penalty = PenaltyMass::new(GRAVITY, k, d, DT.min(0.2 / k.sqrt()))?;
        let e = penalty.effective_restitution(Self::impact_speed())?;
        let rigid = BouncingMass::new(GRAVITY, e.clamp(0.0, 1.0))?;
        Some((penalty, rigid, e))
    }

    /// The restitution the penalty pair actually realises, measured rather than assumed.
    pub fn measured_restitution(&self) -> f64 {
        self.models().map_or(f64::NAN, |(_, _, e)| e)
    }

    /// **The smoothing gap**: how far the smoothed model's post-impact state lands from the rigid one, over a spread
    /// of entry speeds. This is the quantity the whole certificate rests on, and it shrinks as the contact stiffens.
    fn gap(&self) -> Option<GapBound> {
        let (penalty, rigid, _) = self.models()?;
        let v = Self::impact_speed();
        let h_start = 1e-4;
        let v_start = -(v * v - 2.0 * GRAVITY * h_start).sqrt();
        let window = (v - v_start.abs()) / GRAVITY + 8.0 * core::f64::consts::PI / self.stiffness().sqrt();

        let mut residuals = Vec::new();
        for i in 0..25 {
            let scale = 0.9 + 0.2 * (i as f64) / 24.0;
            let entry = [h_start, v_start * scale];
            let smooth = penalty.rollout(entry, window);
            let (flown, _) = rigid.flow(entry, window);
            let r = DVector::from_vec(vec![smooth[0] - flown[0], smooth[1] - flown[1]]);
            if r.iter().all(|x| x.is_finite()) {
                residuals.push(r);
            }
        }
        let sampled = GapBound::from_samples(&residuals)?;
        if self.proved {
            // The same magnitude, presented as proved. Deriving a real Lipschitz constant for a stiff penalty contact
            // is an open problem, so this is a conditional: it shows what a proof would buy, not that one exists.
            GapBound::assume_bound(&sampled.half_width, 0.0, 0.0)
        } else {
            Some(sampled)
        }
    }

    /// The largest half-width of the measured gap.
    pub fn gap_magnitude(&self) -> f64 {
        self.gap().map_or(f64::NAN, |g| g.magnitude())
    }

    fn build(&self) -> Option<(Vec<DVector<f64>>, ferromotion_control::TubeReport, Vec<HalfSpace>)> {
        let (_, rigid, _) = self.models()?;
        let gap = self.gap()?;
        let exact = rigid.jacobian_saltation(
            [1e-4, -((Self::impact_speed().powi(2) - 2.0 * GRAVITY * 1e-4).sqrt())],
            (Self::impact_speed() - (Self::impact_speed().powi(2) - 2.0 * GRAVITY * 1e-4).sqrt()) / GRAVITY
                + 8.0 * core::f64::consts::PI / self.stiffness().sqrt(),
        )?;
        let impact = DMatrix::from_row_slice(2, 2, &[exact[0][0], exact[0][1], exact[1][0], exact[1][1]]);
        let flight = DMatrix::from_row_slice(2, 2, &[1.0, DT, 0.0, 1.0]);
        let zero = GapBound::assume_bound(&DVector::zeros(2), 0.0, 0.0)?;
        let _ = &flight;

        // The impact map is a linearisation, so its residual is asserted; the flight steps are exactly linear but
        // their zero gap only holds above the plane, which the precondition makes checkable.
        let asserted = GapBound::assume_bound(&DVector::zeros(2), 0.0, 0.0)?;
        let above_plane = HalfSpace::new(DVector::from_vec(vec![-1.0, 0.0]), 0.0);

        // **The disturbance set is PER STEP.** The standard tube recursion is R_{k+1} = A R_k (+) W with W the
        // one-step mismatch (Mayne et al. 2005). The measured gap is the mismatch accumulated over the WHOLE contact,
        // so injecting it at a single control step over-states the one-step disturbance by the number of steps the
        // contact spans — and that over-statement is what used to put spurious below-plane states in the tube.
        //
        // The contact duration is known exactly, from the closed-form solve, so the step count is computed rather
        // than guessed.
        let k = self.stiffness();
        let contact = AffineContact::new(GRAVITY, k, 2.0 * ZETA * k.sqrt())?;
        let duration = contact.solve(Self::impact_speed())?.duration;
        let n_contact = ((duration / DT).ceil().max(1.0) as usize).min(self.horizon);
        // divided_by, NOT assume_bound: dividing through the assertion constructor would stamp Proved on a sampled
        // gap and launder it into a certificate.
        let per_step = gap.divided_by(n_contact as f64)?;

        // During the contact the two models genuinely differ, and the mismatch accrues; the total injected over those
        // steps is the measured gap. Once the contact is over both models are in free flight and there is no new
        // mismatch, which is when the zero gap and its above-plane precondition become true.
        let mut steps = vec![TubeStep::new(impact, per_step.clone(), asserted.clone())];
        for _ in 1..n_contact {
            steps.push(TubeStep::new(flight.clone(), per_step.clone(), asserted.clone()));
        }
        // Free flight is exactly linear, so the residual is structural. Its ZERO GAP, however, is justified only by
        // "both models integrate the same quadratic", which holds only above the plane — so every flight step carries
        // that precondition and certify() refuses when the reachable set dips below it.
        //
        // On this fixture it DOES dip, and the refusal is correct. The gap is injected as a fixed height half-width at
        // the impact step, but the contact exits at h = 0 BY CONSTRUCTION: the real difference is downstream, growing
        // from the velocity and timing mismatch. Modelling it as an offset at the exit puts penetrating states in the
        // reachable set that the system never visits, and the honest consequence is UNDECIDED rather than a certificate.
        // Fixing it needs the exact gap identity, which is not built. Until then this lab shows the refusal.
        for _ in n_contact..self.horizon {
            let step = TubeStep::linear(flight.clone(), zero.clone())?;
            steps.push(if self.check_preconditions { step.requiring(above_plane.clone()) } else { step });
        }
        let x0 = Zonotope::from_interval(
            &DVector::from_vec(vec![-1e-4, -1e-3]),
            &DVector::from_vec(vec![1e-4, 1e-3]),
        );
        let tube = propagate_tube(&x0, &steps)?;

        // Nominal: rebound at the realised restitution, then free flight.
        let v_plus = rigid.restitution * Self::impact_speed();
        let nominal: Vec<DVector<f64>> = (0..tube.sets.len())
            .map(|i| {
                let t = i as f64 * DT;
                DVector::from_vec(vec![(v_plus * t - 0.5 * GRAVITY * t * t).max(0.0), v_plus - GRAVITY * t])
            })
            .collect();
        let constraints = vec![HalfSpace::new(DVector::from_vec(vec![1.0, 0.0]), self.ceiling)];
        Some((nominal, tube, constraints))
    }

    /// **The verdict, as a word.** `0 = Certified`, `1 = Refuted`, `2 = Undecided`, `-1 = the model failed to build`.
    /// Returned as a code so the page can colour it; the three are not ordered and the lab never draws them on a scale.
    pub fn verdict_code(&self) -> i32 {
        match self.build().map(|(n, t, c)| certify(&n, &t, &c)) {
            Some(TubeVerdict::Certified { .. }) => 0,
            Some(TubeVerdict::Refuted { .. }) => 1,
            Some(TubeVerdict::Undecided { .. }) => 2,
            None => -1,
        }
    }

    /// A sentence a learner can read, including the reason when there is one.
    pub fn verdict_text(&self) -> String {
        match self.build().map(|(n, t, c)| certify(&n, &t, &c)) {
            Some(TubeVerdict::Certified { margin }) => format!("CERTIFIED with margin {margin:.4} m"),
            Some(TubeVerdict::Refuted { step, violation, .. }) => {
                format!("REFUTED at step {step}, exceeds the ceiling by {violation:.4} m")
            }
            Some(TubeVerdict::Undecided { reason }) => format!("UNDECIDED: {reason:?}"),
            None => "the model could not be built at this setting".to_string(),
        }
    }

    /// Final tube half-width: how much uncertainty the certificate had to carry.
    pub fn tube_width(&self) -> f64 {
        self.build().map_or(f64::NAN, |(_, t, _)| t.final_width())
    }

    /// The nominal trajectory's smallest slack against the ceiling, ignoring the tube.
    pub fn nominal_slack(&self) -> f64 {
        self.build().map_or(f64::NAN, |(n, _, c)| nominal_activity(&n, &c))
    }

    /// **The vacuity ratio**: nominal slack divided by tube width. Large means the constraint was never in danger and
    /// the certificate says nothing, however green it looks.
    pub fn vacuity_ratio(&self) -> f64 {
        let (s, w) = (self.nominal_slack(), self.tube_width());
        if w > 0.0 { s / w } else { f64::NAN }
    }

    /// Whether the constraint is close enough to active for the certificate to be worth having.
    pub fn constraint_is_active(&self) -> bool {
        let r = self.vacuity_ratio();
        r.is_finite() && r < 10.0
    }

    /// Apex height of the nominal trajectory, so a learner can see when the horizon is too short to reach it.
    pub fn nominal_apex(&self) -> f64 {
        self.models().map_or(f64::NAN, |(_, rigid, _)| {
            let v_plus = rigid.restitution * Self::impact_speed();
            v_plus * v_plus / (2.0 * GRAVITY)
        })
    }

    /// Control steps needed to reach the apex. Compare against the horizon: below it, the ceiling cannot bite.
    pub fn steps_to_apex(&self) -> f64 {
        self.models().map_or(f64::NAN, |(_, rigid, _)| {
            (rigid.restitution * Self::impact_speed() / GRAVITY / DT).round()
        })
    }

    /// Nominal height at a given step, for plotting the trajectory.
    pub fn nominal_height_at(&self, step: f64) -> f64 {
        self.models().map_or(f64::NAN, |(_, rigid, _)| {
            let v_plus = rigid.restitution * Self::impact_speed();
            let t = step.max(0.0) * DT;
            (v_plus * t - 0.5 * GRAVITY * t * t).max(0.0)
        })
    }

    /// Tube half-width at a given step, for drawing the envelope around the trajectory.
    pub fn tube_width_at(&self, step: f64) -> f64 {
        let i = step.max(0.0) as usize;
        self.build().map_or(f64::NAN, |(_, t, _)| t.widths.get(i).copied().unwrap_or(f64::NAN))
    }

    pub fn ceiling(&self) -> f64 {
        self.ceiling
    }
}

impl Default for CertificateLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The lesson's load-bearing claim.** Flipping the evidence flag changes no geometry at all, and changes the
    /// verdict from Undecided to Certified. If this ever stops holding, the lab is teaching that a certificate is a
    /// number.
    #[test]
    fn the_verdict_tracks_the_evidence_not_the_geometry() {
        let mut lab = CertificateLab::new();
        lab.set_log_stiffness(6.0);
        let (w, s) = (lab.tube_width(), lab.nominal_slack());
        assert_eq!(lab.verdict_code(), 2, "sampled evidence must be Undecided: {}", lab.verdict_text());

        lab.set_proved(true);
        assert!((lab.tube_width() - w).abs() < 1e-15, "the geometry moved: {} vs {w}", lab.tube_width());
        assert!((lab.nominal_slack() - s).abs() < 1e-15, "the nominal moved");
        assert_eq!(lab.verdict_code(), 0, "proved evidence should certify: {}", lab.verdict_text());
    }

    /// Stiffness is what buys the certificate: the gap shrinks and a refuted constraint becomes certifiable.
    #[test]
    fn stiffening_the_contact_shrinks_the_gap_and_changes_the_verdict() {
        let mut lab = CertificateLab::new();
        lab.set_proved(true);

        lab.set_log_stiffness(4.0);
        let soft_gap = lab.gap_magnitude();
        let soft = lab.verdict_code();

        lab.set_log_stiffness(6.0);
        let stiff_gap = lab.gap_magnitude();
        let stiff = lab.verdict_code();

        eprintln!("k=1e4: gap {soft_gap:.3e} -> {}", { lab.set_log_stiffness(4.0); lab.verdict_text() });
        lab.set_log_stiffness(6.0);
        eprintln!("k=1e6: gap {stiff_gap:.3e} -> {}", lab.verdict_text());

        assert!(stiff_gap < soft_gap / 10.0, "gap should fall by a decade: {soft_gap:.3e} -> {stiff_gap:.3e}");
        assert_eq!(soft, 1, "the soft contact should be refuted");
        assert_eq!(stiff, 0, "the stiff contact should certify");
    }

    /// A short horizon makes the ceiling unreachable, so the certificate becomes true and worthless. The lab has to
    /// expose that rather than show a green verdict.
    #[test]
    fn a_short_horizon_produces_a_vacuous_certificate() {
        let mut lab = CertificateLab::new();
        lab.set_proved(true);
        lab.set_log_stiffness(6.0);

        lab.set_horizon(60.0);
        assert!(lab.horizon() < lab.steps_to_apex(), "60 steps should be short of the apex at {}", lab.steps_to_apex());
        assert_eq!(lab.verdict_code(), 0, "a short horizon certifies: {}", lab.verdict_text());
        assert!(!lab.constraint_is_active(), "and the lab must flag it: ratio {}", lab.vacuity_ratio());

        lab.set_horizon(340.0);
        assert!(lab.constraint_is_active(), "a full horizon makes it active: ratio {}", lab.vacuity_ratio());
    }

    /// **The certificate closes, with the precondition CHECKED.**
    ///
    /// This test exists because the certificate was briefly withdrawn on the strength of a bug in this fixture rather
    /// than in the certificate. The disturbance set in `R_{k+1} = A R_k (+) W` is PER STEP; the measured gap is the
    /// mismatch accumulated over the whole contact. Injecting it at one control step over-stated the one-step
    /// disturbance by the number of steps the contact spans, which put below-plane states in the tube and made the
    /// flight steps' above-plane precondition fail. Scoping it correctly makes the precondition hold and the
    /// certificate close.
    #[test]
    fn the_certificate_closes_with_preconditions_checked() {
        let mut lab = CertificateLab::new();
        lab.set_log_stiffness(6.0);
        lab.set_proved(true);
        assert!(lab.checks_preconditions(), "checking must be the default");
        eprintln!("k=1e6, proved, checked: {}", lab.verdict_text());
        assert_eq!(lab.verdict_code(), 0, "expected a certificate: {}", lab.verdict_text());
        assert!(lab.constraint_is_active(), "and a non-vacuous one: ratio {}", lab.vacuity_ratio());
    }

    /// **The scope bug, locked out.** Injecting the whole-contact mismatch at a single step is exactly the error that
    /// cost a correct result. The tube's own half-width is the witness: correctly scoped it must be far smaller than
    /// the whole gap right after the impact, because the mismatch has only had one step to accrue.
    #[test]
    fn the_gap_is_injected_per_step_not_all_at_once() {
        let mut lab = CertificateLab::new();
        lab.set_log_stiffness(6.0);
        let gap = lab.gap_magnitude();
        let first = lab.tube_width_at(1.0);
        eprintln!("gap {gap:.4e} over the contact; tube half-width after ONE step {first:.4e}");
        assert!(first < gap, "a per-step injection cannot reach the whole gap in one step: {first:.3e} vs {gap:.3e}");
        // And by the end of the contact the accrued width is on the order of the whole gap, which is the total the
        // measurement actually supports.
        let after = lab.tube_width_at(6.0);
        assert!(after > 0.5 * gap, "by the contact's end the accrued width should approach the gap: {after:.3e}");
    }

    /// Raising the ceiling out of reach must certify; dropping it below the nominal path must refute. Both on proved
    /// evidence, so the verdict is about the geometry and not the bound.
    #[test]
    fn the_ceiling_moves_the_verdict_in_the_obvious_direction() {
        let mut lab = CertificateLab::new();
        lab.set_proved(true);
        lab.set_log_stiffness(6.0);

        lab.set_ceiling(1.2);
        assert_eq!(lab.verdict_code(), 0, "a high ceiling certifies: {}", lab.verdict_text());
        lab.set_ceiling(0.1);
        assert_eq!(lab.verdict_code(), 1, "a low ceiling refutes: {}", lab.verdict_text());
    }
}
