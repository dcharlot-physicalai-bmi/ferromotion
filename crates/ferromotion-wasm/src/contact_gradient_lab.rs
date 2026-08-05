//! **Where the gradient lies** — the on-device lab behind the contact-gradient lesson.
//!
//! Drag the contact stiffness and watch two numbers move in opposite directions: the forward model gets *more*
//! faithful while its derivative gets *less* correct, and past a point the derivative points the wrong way entirely.
//! Both are computed live from `ferromotion-core`, against a rigid impact whose Jacobian is known in closed form and
//! verified against finite differences to `8e-10`.
//!
//! The learner is meant to arrive at the shape of it themselves: at no stiffness is a *fixed-step* penalty rollout
//! both faithful and differentiable, because its derivative grows as `sqrt(stiffness)`.
//!
//! # The third route, and the correction it forced
//!
//! The lab originally offered two answers, penalty and exact, and invited the conclusion that the penalty *model* was
//! the problem. Measurement says otherwise, so there is now a third: the same penalty physics integrated under an
//! error tolerance instead of a fixed step ([`ContactGradientLab::adaptive_entry`]). It lands on the exact answer.
//!
//! That makes the lesson sharper rather than softer. The divergence is real and belongs to the fixed-step
//! integrator's autodiff, not to the spring. A learner can now check the timestep buttons (uniform refinement, which
//! does not help) against the tolerance route (which does), and the thing being indicted is correctly identified.

use ferromotion_core::{
    jacobian_relative_error, AdaptiveOptions, AdaptivePenalty, BouncingMass, PenaltyMass, GRAVITY,
};
use wasm_bindgen::prelude::*;

/// Damping ratio held fixed as the stiffness sweeps, so the realised restitution stays put and only the contact
/// resolution changes. Without this the sweep would be comparing different physics.
const ZETA: f64 = 0.1606;
const H0: f64 = 1.0;
const HORIZON: f64 = 0.8;

#[wasm_bindgen]
pub struct ContactGradientLab {
    /// `log10` of the penalty stiffness.
    log_stiffness: f64,
    dt: f64,
}

#[wasm_bindgen]
impl ContactGradientLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ContactGradientLab {
        ContactGradientLab { log_stiffness: 4.0, dt: 1e-4 }
    }

    /// Set `log10(stiffness)`; the slider's only input.
    pub fn set_log_stiffness(&mut self, v: f64) {
        self.log_stiffness = v.clamp(2.0, 8.0);
    }

    pub fn stiffness(&self) -> f64 {
        10f64.powf(self.log_stiffness)
    }

    /// Timestep, so a learner can check whether resolving the contact more finely rescues the gradient. It does not.
    pub fn set_dt(&mut self, dt: f64) {
        self.dt = dt.clamp(1e-7, 1e-3);
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    fn penalty(&self) -> Option<PenaltyMass> {
        let k = self.stiffness();
        PenaltyMass::new(GRAVITY, k, 2.0 * ZETA * k.sqrt(), self.dt)
    }

    /// The restitution this stiffness/damping pair actually realises, measured by a drop test rather than assumed.
    pub fn measured_restitution(&self) -> f64 {
        let speed = (2.0 * GRAVITY * H0).sqrt();
        self.penalty().and_then(|p| p.effective_restitution(speed)).unwrap_or(f64::NAN)
    }

    fn rigid(&self) -> Option<BouncingMass> {
        BouncingMass::new(GRAVITY, self.measured_restitution().clamp(0.0, 1.0))
    }

    /// **Axis one: forward fidelity.** How far the penalty trajectory ends from the rigid one, in metres. Falls as the
    /// contact stiffens.
    pub fn forward_error(&self) -> f64 {
        let Some((p, r)) = self.penalty().zip(self.rigid()) else { return f64::NAN };
        let a = r.flow([H0, 0.0], HORIZON).0;
        let b = p.rollout([H0, 0.0], HORIZON);
        (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
    }

    /// **Axis two: derivative error.** Relative error of the penalty Jacobian against the exact one. Rises as the
    /// contact stiffens, which is the whole point of the lab.
    pub fn gradient_error(&self) -> f64 {
        let Some((p, r)) = self.penalty().zip(self.rigid()) else { return f64::NAN };
        let Some(exact) = r.jacobian_saltation([H0, 0.0], HORIZON) else { return f64::NAN };
        jacobian_relative_error(p.jacobian_autodiff([H0, 0.0], HORIZON), exact)
    }

    /// The dominant Jacobian entry the penalty model reports.
    pub fn penalty_entry(&self) -> f64 {
        self.penalty().map_or(f64::NAN, |p| p.jacobian_autodiff([H0, 0.0], HORIZON)[1][0])
    }

    /// The same entry, exactly, from the saltation matrix.
    pub fn exact_entry(&self) -> f64 {
        self.rigid().and_then(|r| r.jacobian_saltation([H0, 0.0], HORIZON)).map_or(f64::NAN, |j| j[1][0])
    }

    /// **The headline the learner should notice**: whether the penalty gradient even points the right way.
    pub fn sign_agrees(&self) -> bool {
        let (a, b) = (self.penalty_entry(), self.exact_entry());
        a.is_finite() && b.is_finite() && a.signum() == b.signum()
    }

    /// The saltation matrix's time-of-impact term at impact speed `v` — the term no smooth contact model contains.
    /// Unbounded as the impact grazes, which is why softening cannot recover it.
    pub fn timing_term(&self, impact_speed: f64) -> f64 {
        self.rigid().and_then(|r| r.saltation(-impact_speed.abs().max(1e-9))).map_or(f64::NAN, |xi| xi[1][0])
    }

    /// Penalty Jacobian entry at an arbitrary `log10` stiffness, for plotting the divergence curve without disturbing
    /// the slider's own state.
    pub fn entry_at(&self, log_stiffness: f64) -> f64 {
        let k = 10f64.powf(log_stiffness);
        PenaltyMass::new(GRAVITY, k, 2.0 * ZETA * k.sqrt(), self.dt)
            .map_or(f64::NAN, |p| p.jacobian_autodiff([H0, 0.0], HORIZON)[1][0])
    }

    /// Gradient error at an arbitrary `log10` stiffness, for the same reason.
    pub fn gradient_error_at(&self, log_stiffness: f64) -> f64 {
        let k = 10f64.powf(log_stiffness);
        let speed = (2.0 * GRAVITY * H0).sqrt();
        let Some(p) = PenaltyMass::new(GRAVITY, k, 2.0 * ZETA * k.sqrt(), self.dt) else { return f64::NAN };
        let Some(e) = p.effective_restitution(speed) else { return f64::NAN };
        let Some(r) = BouncingMass::new(GRAVITY, e.clamp(0.0, 1.0)) else { return f64::NAN };
        let Some(exact) = r.jacobian_saltation([H0, 0.0], HORIZON) else { return f64::NAN };
        jacobian_relative_error(p.jacobian_autodiff([H0, 0.0], HORIZON), exact)
    }

    /// Forward error at an arbitrary `log10` stiffness — the other axis of the plot.
    pub fn forward_error_at(&self, log_stiffness: f64) -> f64 {
        let k = 10f64.powf(log_stiffness);
        let speed = (2.0 * GRAVITY * H0).sqrt();
        let Some(p) = PenaltyMass::new(GRAVITY, k, 2.0 * ZETA * k.sqrt(), self.dt) else { return f64::NAN };
        let Some(e) = p.effective_restitution(speed) else { return f64::NAN };
        let Some(r) = BouncingMass::new(GRAVITY, e.clamp(0.0, 1.0)) else { return f64::NAN };
        let a = r.flow([H0, 0.0], HORIZON).0;
        let b = p.rollout([H0, 0.0], HORIZON);
        (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
    }

    /// The same penalty physics as a continuous ODE, with no timestep baked in.
    fn adaptive(&self) -> Option<AdaptivePenalty> {
        let k = self.stiffness();
        AdaptivePenalty::new(GRAVITY, k, 2.0 * ZETA * k.sqrt())
    }

    /// Tolerance driving the adaptive route. Fixed rather than exposed: the point of the third route is that it does
    /// not need tuning, and a slider would suggest it does.
    fn adaptive_options() -> AdaptiveOptions {
        AdaptiveOptions::with_tolerance(1e-11)
    }

    /// **The third route.** The dominant Jacobian entry from the *same* penalty model, integrated under an error
    /// tolerance rather than at a fixed step. Compare against [`Self::penalty_entry`] and [`Self::exact_entry`]: it
    /// agrees with the exact one, which is what reassigns the blame from the contact model to the integrator.
    pub fn adaptive_entry(&self) -> f64 {
        self.adaptive().map_or(f64::NAN, |p| {
            p.jacobian_richardson([H0, 0.0], HORIZON, 1e-6, Self::adaptive_options())
                .map_or(f64::NAN, |j| j[1][0])
        })
    }

    /// Relative error of the adaptive Jacobian against the same exact reference the penalty route is scored on, so
    /// the two are directly comparable.
    pub fn adaptive_error(&self) -> f64 {
        let Some((p, r)) = self.adaptive().zip(self.rigid()) else { return f64::NAN };
        let Some(exact) = r.jacobian_saltation([H0, 0.0], HORIZON) else { return f64::NAN };
        match p.jacobian_richardson([H0, 0.0], HORIZON, 1e-6, Self::adaptive_options()) {
            Ok(j) => jacobian_relative_error(j, exact),
            Err(_) => f64::NAN,
        }
    }

    /// What the adaptive route costs, in accepted steps. The honest comparison is cost against accuracy, and a
    /// learner should be able to see it is not buying the answer with a huge budget.
    pub fn adaptive_steps(&self) -> f64 {
        self.adaptive().map_or(f64::NAN, |p| {
            p.rollout([H0, 0.0], HORIZON, Self::adaptive_options()).map_or(f64::NAN, |(_, s)| s.accepted as f64)
        })
    }

    /// How many fixed steps the timestep buttons are spending, for the same comparison.
    pub fn fixed_steps(&self) -> f64 {
        (HORIZON / self.dt).round()
    }

    /// Adaptive-route error at an arbitrary `log10` stiffness, so the third curve can be drawn across the sweep
    /// without disturbing the slider. Scored against the same exact reference as the other two.
    pub fn adaptive_error_at(&self, log_stiffness: f64) -> f64 {
        let k = 10f64.powf(log_stiffness);
        let speed = (2.0 * GRAVITY * H0).sqrt();
        let Some(fixed) = PenaltyMass::new(GRAVITY, k, 2.0 * ZETA * k.sqrt(), self.dt) else { return f64::NAN };
        let Some(e) = fixed.effective_restitution(speed) else { return f64::NAN };
        let Some(r) = BouncingMass::new(GRAVITY, e.clamp(0.0, 1.0)) else { return f64::NAN };
        let Some(exact) = r.jacobian_saltation([H0, 0.0], HORIZON) else { return f64::NAN };
        let Some(p) = AdaptivePenalty::new(GRAVITY, k, 2.0 * ZETA * k.sqrt()) else { return f64::NAN };
        match p.jacobian_richardson([H0, 0.0], HORIZON, 1e-6, Self::adaptive_options()) {
            Ok(j) => jacobian_relative_error(j, exact),
            Err(_) => f64::NAN,
        }
    }

    /// The **model** share of the error at a given `log10` stiffness: how far the converged penalty gradient sits from
    /// the exact rigid one. This is the part no integrator can remove, and the lab shows it shrinking.
    pub fn model_error_at(&self, log_stiffness: f64) -> f64 {
        self.decomposed_at(log_stiffness).map_or(f64::NAN, |d| d.model)
    }

    /// The **discretisation** share at a given `log10` stiffness: what the fixed-step integrator adds on top. This is
    /// the part a tolerance removes, and the lab shows it dominating.
    pub fn discretisation_error_at(&self, log_stiffness: f64) -> f64 {
        self.decomposed_at(log_stiffness).map_or(f64::NAN, |d| d.discretisation)
    }

    fn decomposed_at(&self, log_stiffness: f64) -> Option<ferromotion_core::ErrorDecomposition> {
        let k = 10f64.powf(log_stiffness);
        ferromotion_core::decompose_gradient_error(
            GRAVITY,
            k,
            2.0 * ZETA * k.sqrt(),
            self.dt,
            [H0, 0.0],
            HORIZON,
            1e-6,
            Self::adaptive_options(),
        )
    }

    /// Measured exponent of `|penalty entry|` against stiffness over `[lo, hi]` in `log10`. The exact value is `1/2`,
    /// so a learner can read the divergence law off the lab instead of being told it.
    pub fn divergence_exponent(&self, lo: f64, hi: f64) -> f64 {
        let (a, b) = (self.entry_at(lo).abs(), self.entry_at(hi).abs());
        if !(a > 0.0 && b > 0.0) {
            return f64::NAN;
        }
        (b / a).ln() / (10f64.powf(hi) / 10f64.powf(lo)).ln()
    }
}

impl Default for ContactGradientLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lab must reproduce the numbers the audit example measured, or the lesson teaches something the library does
    /// not do.
    #[test]
    fn the_lab_reproduces_the_measured_result() {
        let mut lab = ContactGradientLab::new();
        lab.set_dt(1e-4);
        eprintln!("{:>10}  {:>10}  {:>12}  {:>14}  {:>12}  sign", "stiffness", "meas. e", "fwd error", "penalty entry", "grad rel err");
        let mut fwd = Vec::new();
        let mut grad = Vec::new();
        for l in [2.0, 3.0, 4.0, 5.0, 6.0] {
            lab.set_log_stiffness(l);
            eprintln!(
                "{:>10.0e}  {:>10.4}  {:>12.2e}  {:>14.4}  {:>12.2e}  {}",
                lab.stiffness(),
                lab.measured_restitution(),
                lab.forward_error(),
                lab.penalty_entry(),
                lab.gradient_error(),
                if lab.sign_agrees() { "ok" } else { "WRONG" }
            );
            fwd.push(lab.forward_error());
            grad.push(lab.gradient_error());
        }
        // the two axes move in opposite directions, which is the lesson
        assert!(fwd[4] < fwd[0], "forward fidelity improves with stiffness: {:.2e} -> {:.2e}", fwd[0], fwd[4]);
        assert!(grad[4] > grad[0], "gradient accuracy does not: {:.2e} -> {:.2e}", grad[0], grad[4]);

        // The exact entry is +3.5905 here against the audit example's +3.5436, because this lab gives the rigid
        // reference the restitution the penalty pair actually REALISES (0.6212 at k = 1e4) rather than a nominal 0.6.
        // Matching the physics is the right choice and it moves the reference slightly; the numbers are consistent.
        lab.set_log_stiffness(4.0);
        assert!((lab.measured_restitution() - 0.6212).abs() < 1e-3, "measured e {:.4}", lab.measured_restitution());
        assert!((lab.exact_entry() - 3.5905).abs() < 5e-3, "exact entry {:.4}", lab.exact_entry());
        assert!(!lab.sign_agrees(), "at k = 1e4 the penalty gradient points the wrong way");

        // and the divergence law is readable off the lab
        let e = lab.divergence_exponent(4.0, 8.0);
        eprintln!("\n   divergence exponent over 1e4..1e8: {e:.4} against an exact 0.5");
        assert!((e - 0.5).abs() < 0.05, "{e:.4}");
    }

    /// The timing term is what a smooth contact cannot represent, and the lab has to show it growing.
    #[test]
    fn the_timing_term_is_exposed_and_unbounded() {
        let lab = ContactGradientLab::new();
        let mut last = 0.0f64;
        for v in [4.0, 1.0, 0.1, 0.01] {
            let t = lab.timing_term(v);
            eprintln!("impact speed {v:>5}: timing term {t:>12.2}");
            assert!(t.abs() > last.abs());
            last = t;
        }
        // 1/|v| exactly: each tenfold slower impact multiplies the term by ten
        assert!((lab.timing_term(0.1) / lab.timing_term(1.0) - 10.0).abs() < 1e-9, "the term goes as 1/|v|");
        assert!(last.abs() > 1e3, "unbounded as the impact grazes: {last:.1}");
    }

    /// **The correction, pinned.** The third route must land on the exact answer while the fixed-step route does not,
    /// at a stiffness where the fixed-step route is not merely inaccurate but pointing the wrong way. If this ever
    /// stops holding, the lesson attached to this lab is teaching the wrong cause.
    #[test]
    fn the_tolerance_route_lands_on_the_exact_answer() {
        let mut lab = ContactGradientLab::new();
        lab.set_log_stiffness(6.0);
        lab.set_dt(1e-4);

        eprintln!("at stiffness 1e6, one reference, three routes:");
        eprintln!("   exact (saltation)    {:>12.5}", lab.exact_entry());
        eprintln!("   fixed-step autodiff  {:>12.5}   rel err {:.2e}   {} steps", lab.penalty_entry(), lab.gradient_error(), lab.fixed_steps());
        eprintln!("   tolerance-driven     {:>12.5}   rel err {:.2e}   {} steps", lab.adaptive_entry(), lab.adaptive_error(), lab.adaptive_steps());

        assert!(lab.adaptive_error() < 1e-2, "the tolerance route should recover the exact answer: {:.3e}", lab.adaptive_error());
        assert!(lab.gradient_error() > 1.0, "the fixed-step route should still be wrong: {:.3e}", lab.gradient_error());
        assert!(
            lab.adaptive_error() < lab.gradient_error() / 100.0,
            "the two routes should differ by orders of magnitude, not a little: {:.3e} vs {:.3e}",
            lab.adaptive_error(),
            lab.gradient_error()
        );
        // And it is not buying the answer with a giant budget.
        assert!(lab.adaptive_steps() < lab.fixed_steps(), "adaptive {} steps vs fixed {}", lab.adaptive_steps(), lab.fixed_steps());
    }

    /// The decomposition the third curve rests on: the integrator's share must dominate the model's share, and the
    /// model's share must shrink as the contact stiffens rather than grow.
    #[test]
    fn the_decomposition_blames_the_integrator() {
        let mut lab = ContactGradientLab::new();
        lab.set_dt(1e-4);
        eprintln!("{:>10}  {:>13}  {:>13}  {:>9}", "stiffness", "discretisation", "model", "ratio");
        let mut models = Vec::new();
        for l in [4.0, 5.0, 6.0] {
            let (d, m) = (lab.discretisation_error_at(l), lab.model_error_at(l));
            eprintln!("{:>10.0e}  {d:>13.3e}  {m:>13.3e}  {:>9.1e}", 10f64.powf(l), d / m);
            assert!(d > 10.0 * m, "at 1e{l}: discretisation {d:.3e} does not dominate model {m:.3e}");
            models.push(m);
        }
        assert!(models[2] < models[0], "the model share should shrink with stiffness: {:.3e} -> {:.3e}", models[0], models[2]);
    }

    /// Finer timesteps do not rescue the gradient, so a learner who tries that must see it fail.
    #[test]
    fn a_finer_timestep_does_not_rescue_it() {
        let mut lab = ContactGradientLab::new();
        lab.set_log_stiffness(6.0);
        eprintln!("at stiffness 1e6, gradient error by timestep:");
        for dt in [1e-4, 1e-5, 1e-6] {
            lab.set_dt(dt);
            eprintln!("   dt {dt:.0e}: grad rel err {:.2e}, sign {}", lab.gradient_error(), if lab.sign_agrees() { "ok" } else { "WRONG" });
            assert!(lab.gradient_error() > 1.0, "still wrong by more than the whole answer at dt = {dt:.0e}");
        }
    }
}
