//! **Contact Trust Region (CTR)** — contact-rich manipulation planning on a *smoothed, quasi-dynamic*
//! contact model (Suh, Pang, Zhao, Tedrake, IJRR 2025; building on Pang & Tedrake's convex quasi-dynamic
//! contact). Two ideas make gradient planning through contact work:
//!
//! * a **smoothed contact force** — a softplus of penetration, `λ = (k/κ)·log(1 + e^{κd})` — which is
//!   nonzero and differentiable even at the contact boundary (hard contact is the `κ → ∞` limit), so a
//!   planner gets a usable gradient where a rigid model gives none; and
//! * a **contact trust region** — the crucial fix over an ellipsoidal (`‖Δu‖ ≤ ρ`) trust region, which
//!   is geometrically inconsistent with contact: a plain ball-shaped step can jump *across* the contact
//!   boundary, where the linearization is meaningless. The CTR instead bounds the step so the contact
//!   configuration changes only within the model's smoothing bandwidth, keeping the linear model valid.
//!
//! Demonstrated on a quasi-dynamic pusher–slider (the slider has no actuator and moves only when
//! pushed). Pure Rust → WASM-clean; reuses no external solver.

/// A softplus-smoothed unilateral contact: force as a function of penetration depth `d` (`d > 0` ⇒
/// penetrating). `κ` is the smoothing sharpness (→ ∞ recovers rigid contact `k·max(0,d)`), `k` the
/// stiffness.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedContact {
    pub k: f64,
    pub kappa: f64,
}

impl SmoothedContact {
    /// Contact force `λ(d) = (k/κ)·log(1 + e^{κd})` (numerically-stable softplus).
    pub fn force(&self, d: f64) -> f64 {
        let x = self.kappa * d;
        let sp = if x > 30.0 { x } else { (1.0 + x.exp()).ln() };
        self.k * sp / self.kappa
    }
    /// `dλ/dd = k·σ(κd)` — the (logistic) contact stiffness, smooth everywhere.
    pub fn dforce(&self, d: f64) -> f64 {
        self.k / (1.0 + (-self.kappa * d).exp())
    }
}

/// A 1-D quasi-dynamic pusher–slider. The pusher is position-controlled (`p⁺ = p + h·u`); the slider,
/// unactuated, is moved only by the contact force (`s⁺ = s + h·λ/b`, quasi-dynamic: damping `b`, no
/// inertia). Penetration is `d = p⁺ − s` (the pusher overlapping the slider from behind).
#[derive(Clone, Copy, Debug)]
pub struct PusherSlider {
    pub contact: SmoothedContact,
    pub h: f64,
    pub b: f64,
}

impl PusherSlider {
    /// One step: returns `(p⁺, s⁺)`.
    pub fn step(&self, p: f64, s: f64, u: f64) -> (f64, f64) {
        let pn = p + self.h * u;
        let d = pn - s;
        let sn = s + self.h * self.contact.force(d) / self.b;
        (pn, sn)
    }

    /// `∂s⁺/∂u = (h²/b)·λ'(d)` at `(p, s, u)` — the analytic control sensitivity of the slider.
    pub fn dslider_du(&self, p: f64, s: f64, u: f64) -> f64 {
        let d = (p + self.h * u) - s;
        self.h * self.h * self.contact.dforce(d) / self.b
    }

    /// The **contact trust radius** at `(p, s)`: the largest control step that keeps the penetration
    /// change within one smoothing bandwidth `1/κ`, so the linear model stays valid across it.
    /// (`Δd = h·Δu ≤ 1/κ` ⇒ `Δu ≤ 1/(κ·h)`.) An ellipsoidal trust region ignores this and may step clear
    /// across the contact boundary.
    pub fn contact_trust_radius(&self) -> f64 {
        1.0 / (self.contact.kappa * self.h)
    }

    /// Roll a control sequence out from `(p0, s0)`, returning the final slider position.
    pub fn rollout_final(&self, p0: f64, s0: f64, controls: &[f64]) -> f64 {
        let (mut p, mut s) = (p0, s0);
        for &u in controls {
            let (pn, sn) = self.step(p, s, u);
            p = pn;
            s = sn;
        }
        s
    }

    /// Plan a constant pushing control to drive the slider to `s_target`, by gradient descent with the
    /// step at each iteration capped to the contact trust region — the CTR keeps every step inside the
    /// region where the linearization is trustworthy. Returns the control and achieved slider position.
    pub fn plan_push(&self, p0: f64, s0: f64, s_target: f64, steps: usize, iters: usize) -> (f64, f64) {
        let mut u = 0.0;
        let radius = self.contact_trust_radius();
        for _ in 0..iters {
            // gradient of (s_final − target)² w.r.t. a constant control u, via the chain over the rollout
            let (mut p, mut s) = (p0, s0);
            let mut ds_du = 0.0; // d s / d u accumulated
            for _ in 0..steps {
                // s⁺ = s + h·λ(p+hu−s)/b ; dp/du carries forward (pusher integrates u)
                let dsn = self.dslider_du(p, s, u) + (self.h * self.contact.dforce((p + self.h * u) - s) / self.b) * 0.0;
                // chain: ds_du_next = ds_du·(1 − hλ'/b) + (∂s⁺/∂u direct); pusher p depends on all past u
                let dpen_ds = -self.h * self.contact.dforce((p + self.h * u) - s) / self.b;
                ds_du = ds_du * (1.0 + dpen_ds) + dsn;
                let (pn, sn) = self.step(p, s, u);
                p = pn;
                s = sn;
            }
            let grad = 2.0 * (s - s_target) * ds_du;
            let mut step = -0.3 * grad;
            step = step.clamp(-radius, radius); // the contact trust region
            u += step;
        }
        (u, self.rollout_final(p0, s0, &vec![u; steps]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ps() -> PusherSlider {
        PusherSlider { contact: SmoothedContact { k: 20.0, kappa: 40.0 }, h: 0.05, b: 4.0 }
    }

    #[test]
    fn the_smoothed_force_approaches_rigid_contact_as_kappa_grows() {
        // λ(d) → k·max(0,d) as κ → ∞, and is a smooth, monotone, nonnegative surrogate for any κ.
        for &d in &[-0.2_f64, -0.02, 0.0, 0.02, 0.2] {
            let sharp = SmoothedContact { k: 20.0, kappa: 2000.0 };
            let hard = 20.0 * d.max(0.0);
            assert!((sharp.force(d) - hard).abs() < 0.05, "sharp κ should approach rigid at d={d}: {} vs {hard}", sharp.force(d));
        }
        let c = SmoothedContact { k: 20.0, kappa: 40.0 };
        assert!(c.force(-1.0) >= 0.0 && c.force(-1.0) < 1e-3, "well-separated ⇒ ~zero force");
        assert!(c.force(0.5) > c.force(0.1), "force monotone increasing in penetration");
    }

    #[test]
    fn the_contact_force_gradient_matches_finite_differences() {
        let c = SmoothedContact { k: 20.0, kappa: 40.0 };
        let eps = 1e-6;
        for &d in &[-0.05, 0.0, 0.03, 0.1] {
            let fd = (c.force(d + eps) - c.force(d - eps)) / (2.0 * eps);
            assert!((c.dforce(d) - fd).abs() < 1e-4, "dforce({d}) {} vs fd {fd}", c.dforce(d));
        }
    }

    #[test]
    fn contact_transmits_a_push_only_when_engaged() {
        let r = ps();
        // pusher overlapping the slider (d>0) drives it forward
        let (_p, s1) = r.step(0.05, 0.0, 5.0);
        assert!(s1 > 1e-3, "an engaged push should move the slider: {s1}");
        // pusher well behind the slider (large gap) moves it negligibly (displacement ≈ 0)
        let (_p2, s2) = r.step(0.0, 0.5, 5.0);
        assert!((s2 - 0.5).abs() < 1e-3, "a push across a big gap should barely move the slider: Δ={}", s2 - 0.5);
    }

    #[test]
    fn the_contact_trust_region_keeps_the_linearization_valid_where_an_ellipsoidal_step_does_not() {
        // THE CHAPTER. At a contact transition (gap ≈ 0) the dynamics are sharply nonlinear. A step
        // sized to the contact trust region keeps the linear prediction of s⁺ accurate; a much larger
        // (ellipsoidal-style, contact-blind) step of the same planner leaps across the transition and
        // the linear model is badly wrong.
        let r = ps();
        let (p, s, u0) = (-0.001, 0.0, 0.0); // right at contact onset (d ≈ 0)
        let base = r.step(p, s, u0).1;
        let slope = r.dslider_du(p, s, u0);
        let predict = |du: f64| base + slope * du;
        let actual = |du: f64| r.step(p, s, u0 + du).1;

        let ctr = r.contact_trust_radius();
        let err_ctr = (predict(ctr) - actual(ctr)).abs();
        let err_big = (predict(6.0 * ctr) - actual(6.0 * ctr)).abs();
        assert!(err_ctr < 0.2 * err_big, "CTR step should keep the linearization far more valid: {err_ctr} vs {err_big}");
    }

    #[test]
    fn ctr_planning_pushes_the_slider_to_the_target() {
        // The trust-region planner discovers the pushing control that drives the unactuated slider to a
        // target reachable only through contact.
        let r = ps();
        let (u, s_final) = r.plan_push(-0.05, 0.0, 0.3, 40, 200);
        assert!((s_final - 0.3).abs() < 0.03, "slider should reach the target: {s_final}");
        assert!(u > 0.0, "the plan should push forward: u = {u}");
    }
}
