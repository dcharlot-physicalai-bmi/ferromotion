//! **Contact trust-region lab** — the rig behind the textbook chapter on planning *through* contact.
//! Drives the real [`ferromotion_control::PusherSlider`] and [`ferromotion_control::SmoothedContact`] so
//! the reader can (1) watch the smoothed contact force morph toward rigid `k·max(0,d)` as the sharpness
//! `κ` grows, and (2) drag a control step `Δu` at a contact transition and see the linear prediction of
//! the slider's next position stay honest inside the contact trust region and diverge outside it.

use ferromotion_control::{PusherSlider, SmoothedContact};
use wasm_bindgen::prelude::*;

// The linearization is probed right at contact onset (pusher a hair behind the slider, no nominal push).
const P0: f64 = -0.001;
const S0: f64 = 0.0;
const U0: f64 = 0.0;

#[wasm_bindgen]
pub struct CtrLab {
    ps: PusherSlider,
}

#[wasm_bindgen]
impl CtrLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CtrLab {
        CtrLab { ps: PusherSlider { contact: SmoothedContact { k: 20.0, kappa: 40.0 }, h: 0.05, b: 4.0 } }
    }

    /// Contact sharpness `κ` (→ ∞ recovers rigid contact).
    pub fn set_kappa(&mut self, kappa: f64) {
        self.ps.contact.kappa = kappa.max(1.0);
    }
    pub fn kappa(&self) -> f64 {
        self.ps.contact.kappa
    }

    /// The smoothed contact force `λ(d)` at penetration `d`.
    pub fn contact_force(&self, d: f64) -> f64 {
        self.ps.contact.force(d)
    }
    /// The rigid reference `k·max(0,d)` the smoothing approaches as `κ → ∞`.
    pub fn rigid_force(&self, d: f64) -> f64 {
        self.ps.contact.k * d.max(0.0)
    }

    /// The **contact trust radius** `1/(κ·h)`: the largest control step that keeps the penetration change
    /// within one smoothing bandwidth, so the linear model stays valid across it.
    pub fn trust_radius(&self) -> f64 {
        self.ps.contact_trust_radius()
    }

    /// The slider position after one step at the onset config with nominal control (the linearization
    /// base point).
    pub fn base_s(&self) -> f64 {
        self.ps.step(P0, S0, U0).1
    }
    /// The linear sensitivity `∂s⁺/∂u` at the onset config (the slope of the model).
    pub fn slope(&self) -> f64 {
        self.ps.dslider_du(P0, S0, U0)
    }
    /// The linear model's prediction of the next slider position for a control step `Δu`.
    pub fn predict_s(&self, du: f64) -> f64 {
        self.base_s() + self.slope() * du
    }
    /// The *true* next slider position for a control step `Δu` (one real step of the smoothed dynamics).
    pub fn actual_s(&self, du: f64) -> f64 {
        self.ps.step(P0, S0, U0 + du).1
    }
    /// Absolute linear-model error at step `Δu` (`|predict − actual|`).
    pub fn lin_error(&self, du: f64) -> f64 {
        (self.predict_s(du) - self.actual_s(du)).abs()
    }

    /// Plan a constant push (trust-region gradient descent) to drive the unactuated slider to `target`
    /// from a small standoff; returns the achieved slider position.
    pub fn plan_final(&self, target: f64) -> f64 {
        self.ps.plan_push(-0.05, 0.0, target, 40, 200).1
    }
    /// The control the planner found for `target`.
    pub fn plan_control(&self, target: f64) -> f64 {
        self.ps.plan_push(-0.05, 0.0, target, 40, 200).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharper_kappa_pushes_the_force_toward_rigid() {
        let mut lab = CtrLab::new();
        lab.set_kappa(2000.0);
        // deep in contact the smoothed force ≈ the rigid force
        assert!((lab.contact_force(0.3) - lab.rigid_force(0.3)).abs() < 0.1);
    }

    #[test]
    fn the_ctr_step_is_more_valid_than_a_big_step() {
        let lab = CtrLab::new();
        let r = lab.trust_radius();
        assert!(lab.lin_error(r) < lab.lin_error(6.0 * r), "linearization must be more valid inside the CTR");
    }

    #[test]
    fn the_planner_reaches_a_contact_only_target() {
        let lab = CtrLab::new();
        assert!((lab.plan_final(0.3) - 0.3).abs() < 0.03, "planner should drive the slider to the target");
    }
}
