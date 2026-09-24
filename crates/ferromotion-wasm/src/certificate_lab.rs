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
//!
//! ⛔ **And a fourth thing no control can fix: the tube does not contain the real trajectory.** Commit `97a7307`
//! withdrew both verdicts this fixture produces, the `Certified` at `k = 1e6` and the `Refuted` at `k = 1e4`,
//! because the penalty model's own trajectory, the one its gap was measured from, leaves the tube one control step
//! after the bounce. The gap is a TIME offset of one contact duration, and no per-step additive box can represent a
//! timing term. The lab keeps producing the algebra's verdicts on purpose, and runs
//! [`ferromotion_control::escaping_sample`] beside them ([`CertificateLab::contains_true_trajectory`]), so the learner
//! sees the word CERTIFIED on screen and the check that voids it. A tube is not a certificate until that check passes.

use ferromotion_control::{
    certify, escaping_sample, nominal_activity, propagate_tube, GapBound, HalfSpace, TubeReport, TubeStep,
    TubeVerdict, UndecidedReason, Zonotope,
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
    /// The flight steps claim a zero gap because both models integrate the same quadratic, which is true only ABOVE
    /// the plane, so each carries that precondition. It fired once, when the whole gap was injected at one step; the
    /// gap is now spread over the contact's steps and it passes. Passing it does not make the tube sound: see
    /// [`CertificateLab::contains_true_trajectory`], which it does not.
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

        // The tube recursion is R_{k+1} = A R_k (+) W with W the one-step disturbance (Mayne et al. 2005). This fixture
        // spreads the measured gap over the steps the contact spans (the duration comes from the closed-form solve),
        // which keeps the reachable set above the plane and lets the flight preconditions pass. ⛔ It does NOT make W
        // bound the one-step mismatch: the gap measures a TIME offset of one contact duration, and the true one-step
        // velocity mismatch at k = 1e6 is ~4.9 m/s where ~2.3e-3 is injected (`97a7307`). No per-step box can hold a
        // timing term. `contains_true_trajectory` is the check that shows it.
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
        // that precondition and certify() refuses when the reachable set dips below it. With the gap spread over the
        // contact it no longer dips, so the precondition passes; representing the mismatch soundly needs a
        // saltation-style timing term, which is not built.
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
            // `certify` reports the FIRST step the plan crosses, where it has only just crossed, so the size of the
            // violation is read at the plan's peak instead.
            Some(TubeVerdict::Refuted { step, .. }) => format!(
                "REFUTED: the planned trajectory itself crosses the ceiling at step {step} and peaks {:.3} m above it",
                self.nominal_apex() - self.ceiling
            ),
            Some(TubeVerdict::Undecided { reason: UndecidedReason::EnvelopeReachesConstraint { step, .. } }) => format!(
                "UNDECIDED: the envelope reaches the ceiling at step {step} but the planned trajectory clears it; an \
                 envelope is not a counterexample"
            ),
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

    /// The penalty model's own trajectory from the impact, sampled at every control step: element `i` is the state
    /// at `i * DT`. It is advanced on the model's own internal step, so the sample times do not drift at stiffnesses
    /// where `DT` is not a whole number of internal steps.
    fn true_trajectory(&self, n: usize) -> Option<Vec<DVector<f64>>> {
        let (penalty, _, _) = self.models()?;
        let mut x = [0.0, -Self::impact_speed()];
        let mut done = 0usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let target = (i as f64 * DT / penalty.dt).round() as usize;
            if target > done {
                x = penalty.rollout(x, (target - done) as f64 * penalty.dt);
                done = target;
            }
            out.push(DVector::from_vec(vec![x[0], x[1]]));
        }
        Some(out)
    }

    fn escape(&self) -> Option<Option<(usize, f64)>> {
        let (nominal, tube, _) = self.build()?;
        let samples = self.true_trajectory(nominal.len())?;
        Some(escape_after_impact(&nominal, &tube, &samples))
    }

    /// **The reality check.** The tube claims to contain every state the system can reach, so the penalty model's own
    /// trajectory, the one its gap was measured from, must lie inside it at every step. On this fixture it does not,
    /// at any stiffness, which is why both of the lab's verdicts were withdrawn. `false` means no verdict from this
    /// tube holds, whatever the word above it says.
    pub fn contains_true_trajectory(&self) -> bool {
        matches!(self.escape(), Some(None))
    }

    /// Control step of the worst escape, or `NaN` if the tube contains the trajectory or the model failed.
    pub fn escape_step(&self) -> f64 {
        match self.escape() {
            Some(Some((step, _))) => step as f64,
            _ => f64::NAN,
        }
    }

    /// The worst escape in multiples of the tube's own half-width: `0` if contained, `NaN` if the model failed.
    pub fn escape_ratio(&self) -> f64 {
        match self.escape() {
            Some(Some((_, ratio))) => ratio,
            Some(None) => 0.0,
            None => f64::NAN,
        }
    }

    /// Highest point the penalty model's own trajectory reaches within the horizon: what the real system does, as
    /// against the tube's verdict about it.
    pub fn true_apex(&self) -> f64 {
        self.true_trajectory(self.horizon + 1)
            .map_or(f64::NAN, |xs| xs.iter().map(|x| x[0]).fold(f64::NEG_INFINITY, f64::max))
    }
}

/// [`escaping_sample`] from one control step after the impact. Step 0 is the impact instant, where the rigid state is
/// two-valued (it reverses instantly: `-v` before, `+e v` after) and the nominal takes the post-rebound value; from step
/// 1 both trajectories are single-valued. Step 0 is replaced by the nominal itself, the tube's centre, so it can never
/// be the reported escape.
fn escape_after_impact(nominal: &[DVector<f64>], tube: &TubeReport, samples: &[DVector<f64>]) -> Option<(usize, f64)> {
    let mut samples = samples.to_vec();
    if let (Some(first), Some(n0)) = (samples.first_mut(), nominal.first()) {
        *first = n0.clone();
    }
    escaping_sample(nominal, tube, &samples)
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

    /// Stiffness narrows the envelope: at 1e4 it reaches the ceiling, which is UNDECIDED and not Refuted, because the
    /// planned trajectory clears the ceiling and an envelope is not a counterexample; at 1e6 it clears it.
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
        assert_eq!(soft, 2, "the soft envelope reaching the ceiling is not a refutation");
        lab.set_log_stiffness(4.0);
        assert!(lab.verdict_text().contains("envelope"), "and the reason must say so: {}", lab.verdict_text());
        lab.set_log_stiffness(6.0);
        assert_eq!(stiff, 0, "the stiff envelope clears the ceiling");
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

    /// ⛔ **The certificate the algebra issues at the lab's default is VOID, and this pins why.** Commit `97a7307`
    /// withdrew it: the penalty model's own trajectory, the one the gap was measured from, leaves the tube one control
    /// step after the bounce. This test used to be `the_certificate_closes_with_preconditions_checked`, asserting the
    /// withdrawn verdict as a result, with a doc comment restating the fix the withdrawal called "the same mistake".
    /// The word CERTIFIED is kept on purpose; the reality check beside it is what a learner has to read.
    #[test]
    fn the_certificate_at_the_default_is_void_because_the_tube_misses_the_real_trajectory() {
        let mut lab = CertificateLab::new();
        lab.set_log_stiffness(6.0);
        lab.set_proved(true);
        assert_eq!(lab.verdict_code(), 0, "the tube algebra still says Certified: {}", lab.verdict_text());
        assert!(!lab.contains_true_trajectory(), "the penalty trajectory must be seen leaving the tube");
        let (step, ratio) = (lab.escape_step(), lab.escape_ratio());
        eprintln!("k=1e6: worst escape at step {step}, {ratio:.1}x the tube's own half-width");
        assert_eq!(step, 1.0, "the escape is one step after the bounce, where the timing offset lives");
        // Measured 1456.9: the velocity, which the rigid model reverses instantly and the penalty model does not.
        assert!((1400.0..1520.0).contains(&ratio), "escape ratio {ratio}");
    }

    /// The soft contact's old REFUTED was fabricated: it reported the 0.46 m ceiling breached while the real mass
    /// peaks near 0.396 m. With `certify` refuting only on a trajectory, the envelope reaching the ceiling is Undecided,
    /// and the real apex is exposed so a learner can see that nothing was violated.
    #[test]
    fn the_soft_contact_does_not_refute_a_ceiling_the_real_mass_never_reaches() {
        let mut lab = CertificateLab::new();
        lab.set_log_stiffness(4.0);
        lab.set_proved(true);
        assert_eq!(lab.verdict_code(), 2, "{}", lab.verdict_text());
        let apex = lab.true_apex();
        // Measured 0.3964, against the rigid plan's 0.3978: pinned tightly enough to tell the real mass from the plan.
        assert!((apex - 0.3964).abs() < 5e-4, "the penalty mass peaks near 0.3964 m: {apex}");
        assert!(apex < lab.ceiling() - 0.05, "and clears the ceiling by several centimetres");
        assert!(!lab.contains_true_trajectory(), "the soft tube misses the real trajectory too");
    }

    /// The reality check passes exactly where the contact fits inside one control step, which is the withdrawal's own
    /// explanation: a per-step box cannot hold a timing offset spread over several steps, and can when the whole offset
    /// happens within one. At 1e7.0 the contact lasts ~1.007 ms (two steps); at 1e7.1 ~0.90 ms (one).
    #[test]
    fn a_contact_shorter_than_one_control_step_passes_the_reality_check() {
        let mut lab = CertificateLab::new();
        lab.set_proved(true);
        lab.set_log_stiffness(7.0);
        assert!(!lab.contains_true_trajectory(), "two contact steps: escape ratio {}", lab.escape_ratio());
        for lk in [7.1, 8.0] {
            lab.set_log_stiffness(lk);
            assert!(lab.contains_true_trajectory(), "at 1e{lk} the contact fits in one step: {}", lab.escape_ratio());
            assert_eq!(lab.escape_ratio(), 0.0);
        }
    }

    /// A real refutation: a ceiling below the planned apex is crossed by the plan itself, and that holds on sampled
    /// evidence too, because it rests on no bound.
    #[test]
    fn a_ceiling_below_the_planned_apex_is_refuted_on_any_evidence() {
        for proved in [false, true] {
            let mut lab = CertificateLab::new();
            lab.set_proved(proved);
            lab.set_ceiling(0.3);
            assert_eq!(lab.verdict_code(), 1, "proved {proved}: {}", lab.verdict_text());
            assert!(lab.verdict_text().contains("planned trajectory itself"), "{}", lab.verdict_text());
        }
    }

    /// The check can say "contained": the nominal is the tube's centre, so it must never be reported as escaping.
    /// Without this, `contains_true_trajectory() == false` could be a check that cannot say anything else.
    #[test]
    fn the_reality_check_passes_a_trajectory_that_is_inside() {
        let (nominal, tube, _) = CertificateLab::new().build().unwrap();
        assert_eq!(escape_after_impact(&nominal, &tube, &nominal), None);
    }

    /// The fixture spreads the gap over the contact's steps rather than injecting it at one. This pins that it does,
    /// and nothing more: spreading it is what lets the flight preconditions pass, and it does NOT make the tube sound
    /// (see `the_certificate_at_the_default_is_void_because_the_tube_misses_the_real_trajectory`).
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
