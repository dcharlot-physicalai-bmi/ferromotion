//! **Walking lab** — the rig behind the textbook chapter on catching a fall and turning it into a step.
//!
//! A walking robot is a controlled fall: its centre of mass topples forward and a foot is thrown out to
//! catch it. The question is *where*. Model the body as a point mass on a massless leg — the linear
//! inverted pendulum — and there is exactly one point on the ground, the **capture point**
//! `ξ = x + ẋ/ω` (with `ω = √(g/z)`), where planting the foot brings the body to a complete stop.
//! Step short of it and you keep toppling; step past and you rock back. The capture point is the whole
//! game: the divergent part of the dynamics, `ξ̇ = ω(ξ − foot)`, is unstable and must be steered by
//! foot placement, while the centre of mass converges to `ξ` for free.
//!
//! The rig runs the real [`ferromotion_control`] `capture_point`/`dcm`/`plan_dcm`, so the reader can
//! push the body, drop the foot on the capture point, and watch the fall resolve into balance — then
//! chain the same idea into a walk.

use ferromotion_control::{capture_point, dcm, lipm_omega, plan_dcm, DcmStep};
use nalgebra::Vector2;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WalkLab {
    z: f64,
    g: f64,
    omega: f64,
    // sagittal LIPM state (forward axis)
    x: f64,
    vx: f64,
    foot: f64,
    // walking plan
    feet: Vec<f64>,
    dur: f64,
}

#[wasm_bindgen]
impl WalkLab {
    #[wasm_bindgen(constructor)]
    pub fn new(height: f64, g: f64) -> WalkLab {
        WalkLab { z: height, g, omega: lipm_omega(height, g), x: 0.0, vx: 0.0, foot: 0.0, feet: vec![], dur: 0.6 }
    }

    pub fn omega(&self) -> f64 { self.omega }
    pub fn com_x(&self) -> f64 { self.x }
    pub fn com_vx(&self) -> f64 { self.vx }
    pub fn foot_x(&self) -> f64 { self.foot }
    pub fn height(&self) -> f64 { self.z }

    pub fn set_state(&mut self, x: f64, vx: f64) {
        self.x = x;
        self.vx = vx;
    }
    pub fn set_foot(&mut self, f: f64) {
        self.foot = f;
    }
    pub fn push(&mut self, dv: f64) {
        self.vx += dv;
    }

    /// The **capture point** `ξ = x + ẋ/ω` (the real library `capture_point`), projected on the forward
    /// axis — the ground point where a step brings the body to rest.
    pub fn capture_point(&self) -> f64 {
        capture_point([self.x, 0.0], [self.vx, 0.0], self.z, self.g)[0]
    }

    /// Same quantity via the DCM (`ξ = x + ẋ/ω`) — for the readout, to show they agree.
    pub fn dcm_x(&self) -> f64 {
        dcm(Vector2::new(self.x, 0.0), Vector2::new(self.vx, 0.0), self.omega).x
    }

    /// Plant the support foot on the capture point.
    pub fn step_to_capture(&mut self) {
        self.foot = self.capture_point();
    }

    /// Advance the linear inverted pendulum `ẍ = ω²(x − foot)` by `dt`.
    pub fn advance(&mut self, dt: f64) {
        let acc = self.omega * self.omega * (self.x - self.foot);
        self.vx += acc * dt;
        self.x += self.vx * dt;
    }

    pub fn com_speed(&self) -> f64 {
        self.vx.abs()
    }

    // ---------- walking ----------

    /// Plan a walk: footstep positions (forward) held `step_dur` each; the DCM is planned backward to
    /// end at rest over the last foot (real library `plan_dcm`).
    pub fn plan(&mut self, feet: &[f64], step_dur: f64) {
        self.feet = feet.to_vec();
        self.dur = step_dur;
    }

    pub fn total_time(&self) -> f64 {
        self.feet.len() as f64 * self.dur
    }

    fn dcm_plan(&self) -> ferromotion_control::DcmPlan {
        let steps: Vec<DcmStep> = self.feet.iter().map(|&f| DcmStep { vrp: Vector2::new(f, 0.0), duration: self.dur }).collect();
        plan_dcm(self.omega, &steps)
    }

    /// The planned DCM reference (forward) at time `t`.
    pub fn dcm_ref_at(&self, t: f64) -> f64 {
        if self.feet.is_empty() { return 0.0; }
        self.dcm_plan().reference(t).0.x
    }

    /// Which foot (VRP) is the support at time `t`.
    pub fn support_foot_at(&self, t: f64) -> f64 {
        let i = ((t / self.dur).floor() as usize).min(self.feet.len().saturating_sub(1));
        self.feet.get(i).copied().unwrap_or(0.0)
    }

    /// The CoM path over the walk: it follows the stable dynamics `ẋ = −ω(x − ξ_ref)` toward the DCM.
    /// Returns the CoM forward position sampled every `dt`.
    pub fn walk_com(&self, dt: f64) -> Vec<f64> {
        if self.feet.is_empty() { return vec![]; }
        let plan = self.dcm_plan();
        let t_end = self.total_time();
        let mut x = 0.0; // start at the first foot
        let mut out = vec![x];
        let mut t = 0.0;
        while t < t_end {
            let xi = plan.reference(t).0.x;
            x += -self.omega * (x - xi) * dt;
            t += dt;
            out.push(x);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lab() -> WalkLab {
        WalkLab::new(0.9, 9.81)
    }

    #[test]
    fn the_capture_point_is_com_plus_velocity_over_omega() {
        let mut l = lab();
        l.set_state(0.1, 0.5);
        let cp = 0.1 + 0.5 / l.omega();
        assert!((l.capture_point() - cp).abs() < 1e-12, "capture point {} vs {cp}", l.capture_point());
        // the DCM is the same quantity
        assert!((l.dcm_x() - l.capture_point()).abs() < 1e-12, "DCM should equal the capture point");
    }

    #[test]
    fn stepping_to_the_capture_point_brings_the_body_to_rest() {
        // THE CHAPTER. A body toppling forward; drop the foot on the capture point and it stops —
        // the CoM converges to the foot with zero velocity, no further control.
        let mut l = lab();
        l.set_state(0.0, 0.6); // moving forward at 0.6 m/s
        l.step_to_capture();
        let foot = l.foot_x();
        // The capture point puts the body on the stable manifold, so it decays to rest over the leg's
        // time constant (~1/ω). Integrate until it settles — the point is a saddle, so we stop once
        // balanced rather than integrating the (physically re-controlled) divergent mode forever.
        let mut settled = false;
        for _ in 0..30_000 {
            l.advance(1e-4);
            if l.com_speed() < 5e-3 && (l.com_x() - foot).abs() < 5e-3 {
                settled = true;
                break;
            }
        }
        assert!(settled, "body did not settle to rest over the foot: v = {}, x−foot = {}", l.com_speed(), l.com_x() - foot);
    }

    #[test]
    fn stepping_short_of_the_capture_point_keeps_toppling() {
        // Place the foot short of ξ and the body keeps falling forward — the divergent mode is not
        // arrested. This is why the capture point, not just "somewhere ahead", is the answer.
        let mut l = lab();
        l.set_state(0.0, 0.6);
        let cp = l.capture_point();
        l.set_foot(cp - 0.1); // short
        for _ in 0..15_000 {
            l.advance(1e-4);
        }
        assert!(l.com_x() - l.foot_x() > 0.2, "CoM should have toppled past the short foot");
        assert!(l.com_speed() > 0.3, "a short step should leave it still falling: v = {}", l.com_speed());
    }

    #[test]
    fn stepping_past_the_capture_point_rocks_back() {
        let mut l = lab();
        l.set_state(0.0, 0.6);
        let cp = l.capture_point();
        l.set_foot(cp + 0.15); // overstep
        // It should decelerate, stop short of the foot, and reverse (velocity goes negative).
        let mut min_v = f64::INFINITY;
        for _ in 0..15_000 {
            l.advance(1e-4);
            min_v = min_v.min(l.com_vx());
        }
        assert!(min_v < -0.05, "overstepping should rock the body back: min v = {min_v}");
    }

    #[test]
    fn a_planned_walk_leads_the_com_forward_and_ends_at_rest() {
        // Chain captures into a walk: footsteps forward, DCM planned backward to rest, CoM follows.
        let mut l = lab();
        let feet = [0.0, 0.25, 0.5, 0.75, 0.75];
        l.plan(&feet, 0.6);
        // The DCM reference ends at rest over the last foot.
        let t_end = l.total_time();
        assert!((l.dcm_ref_at(t_end) - 0.75).abs() < 1e-6, "DCM should end at the last foot: {}", l.dcm_ref_at(t_end));
        // The CoM walks forward and finishes near the last foot.
        let com = l.walk_com(1e-3);
        assert!(com.len() > 10);
        let end = *com.last().unwrap();
        assert!(end > 0.5 && (end - 0.75).abs() < 0.1, "CoM should finish near the last foot: {end}");
        // …and it is monotone-ish forward (never falls back past the start).
        assert!(com.iter().all(|&x| x > -0.05), "CoM should not walk backward");
    }
}
