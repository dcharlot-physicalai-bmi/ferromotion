//! **Grasp lab** — the rig behind the textbook chapter on force closure.
//!
//! Fingers press on an object. The question that decides whether you actually have a grip is not
//! "are you touching it" but "can you resist a push — or a twist — from *any* direction at once?"
//! That property is **force closure**, and it is entirely geometric: each frictional contact can only
//! push inside its **friction cone**, and the grasp holds against every possible disturbance exactly
//! when those cones' wrenches positively span wrench space — when the origin sits strictly inside their
//! convex hull. The **Ferrari–Canny Q1** metric measures how deep inside: the radius of the largest
//! wrench-ball the grasp can resist in every direction. `Q1 > 0` ⟺ force closure; larger is firmer.
//!
//! The rig places contacts on a disk and runs the real [`ferromotion_core::force_closure_q1`], and also
//! returns the *weakest direction* — the wrench the grasp is closest to failing against — so the reader
//! can see where a grip is about to slip.

use ferromotion_core::{force_closure_q1, primitive_wrenches, GraspContact};
use nalgebra::{Vector2, Vector3};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GraspLab {
    r: f64,           // object (disk) radius
    thetas: Vec<f64>, // contact angles on the boundary
    mu: f64,
}

fn fib_dirs(n: usize) -> Vec<Vector3<f64>> {
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = ga * k as f64;
            Vector3::new(r * th.cos(), r * th.sin(), z)
        })
        .collect()
}

#[wasm_bindgen]
impl GraspLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GraspLab {
        GraspLab { r: 1.0, thetas: Vec::new(), mu: 0.5 }
    }

    pub fn add_contact(&mut self, theta: f64) {
        self.thetas.push(theta);
    }
    pub fn set_theta(&mut self, i: usize, theta: f64) {
        self.thetas[i] = theta;
    }
    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu.max(0.0);
    }
    pub fn clear(&mut self) {
        self.thetas.clear();
    }
    pub fn n(&self) -> usize {
        self.thetas.len()
    }
    pub fn theta(&self, i: usize) -> f64 {
        self.thetas[i]
    }
    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn radius(&self) -> f64 {
        self.r
    }

    fn contacts(&self) -> Vec<GraspContact> {
        self.thetas
            .iter()
            .map(|&t| {
                let (c, s) = (t.cos(), t.sin());
                GraspContact {
                    pos: Vector2::new(self.r * c, self.r * s),
                    normal: Vector2::new(-c, -s), // inward
                    mu: self.mu,
                }
            })
            .collect()
    }

    /// Ferrari–Canny Q1 (the real library metric). `> 0` ⟺ force closure; the value is the robustness
    /// margin — the radius of the largest wrench-ball the grasp resists in every direction.
    pub fn q1(&self) -> f64 {
        if self.thetas.len() < 2 {
            return -1.0;
        }
        force_closure_q1(&self.contacts(), 1200)
    }

    pub fn is_force_closure(&self) -> bool {
        self.q1() > 1e-6
    }

    /// The **weakest wrench direction** `[fx, fy, τ]` (unit) — the disturbance the grasp resists least,
    /// i.e. the `argmin_d max_i (w_i·d)` that realises Q1. Where the grip is about to give.
    pub fn weakest_dir(&self) -> Vec<f64> {
        let ws = primitive_wrenches(&self.contacts());
        if ws.is_empty() {
            return vec![0.0, 0.0, 0.0];
        }
        let mut best = f64::INFINITY;
        let mut arg = Vector3::new(1.0, 0.0, 0.0);
        for d in fib_dirs(1200) {
            let m = ws.iter().map(|w| w.dot(&d)).fold(f64::NEG_INFINITY, f64::max);
            if m < best {
                best = m;
                arg = d;
            }
        }
        vec![arg.x, arg.y, arg.z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn two_finger(a: f64, b: f64, mu: f64) -> GraspLab {
        let mut g = GraspLab::new();
        g.set_mu(mu);
        g.add_contact(a);
        g.add_contact(b);
        g
    }

    #[test]
    fn an_antipodal_pinch_is_force_closure_a_same_side_one_is_not() {
        // THE CHAPTER. Two fingers on opposite sides grip; two on the same side cannot — the object
        // can always escape sideways.
        let antipodal = two_finger(0.0, PI, 0.5);
        assert!(antipodal.q1() > 1e-3, "antipodal pinch should be force closure: Q1 = {}", antipodal.q1());
        assert!(antipodal.is_force_closure());

        let same_side = two_finger(0.3, -0.3, 0.5); // both near θ = 0
        assert!(same_side.q1() < 0.0, "same-side grip should NOT be force closure: Q1 = {}", same_side.q1());
        assert!(!same_side.is_force_closure());
    }

    #[test]
    fn a_narrow_pinch_needs_enough_friction_to_close() {
        // Two fingers 60° apart: frictionless they cannot close; enough friction and they can.
        // (Widening the friction cone lets the same geometry positively span wrench space.)
        let slick = two_finger(0.0, PI / 3.0, 0.05);
        let grippy = two_finger(0.0, PI / 3.0, 1.2);
        assert!(slick.q1() < grippy.q1(), "more friction must not lower Q1: {} vs {}", slick.q1(), grippy.q1());
        assert!(grippy.q1() > slick.q1() + 1e-3, "enough friction should improve a narrow pinch");
    }

    #[test]
    fn three_fingers_120_apart_grip_more_firmly_than_a_two_finger_pinch() {
        let mut tri = GraspLab::new();
        tri.set_mu(0.5);
        for k in 0..3 {
            tri.add_contact(k as f64 * 2.0 * PI / 3.0);
        }
        let pinch = two_finger(0.0, PI, 0.5);
        assert!(tri.is_force_closure(), "a symmetric tripod should be force closure");
        assert!(tri.q1() > pinch.q1(), "tripod Q1 {} should beat the pinch {}", tri.q1(), pinch.q1());
    }

    #[test]
    fn more_friction_never_lowers_the_quality() {
        let q = |mu: f64| two_finger(0.0, PI, mu).q1();
        assert!(q(0.8) > q(0.4) && q(0.4) > q(0.1), "Q1 should rise with μ: {} {} {}", q(0.1), q(0.4), q(0.8));
    }

    #[test]
    fn the_weakest_direction_realises_q1() {
        // The reported weakest direction d* should be exactly where the grasp's support is smallest,
        // i.e. max_i (w_i · d*) ≈ Q1. This is the wrench the grip is closest to failing against.
        let g = two_finger(0.0, PI, 0.4);
        let d = g.weakest_dir();
        let dv = Vector3::new(d[0], d[1], d[2]);
        assert!((dv.norm() - 1.0).abs() < 1e-9, "weakest direction must be a unit wrench");
        let ws = primitive_wrenches(&g.contacts());
        let support = ws.iter().map(|w| w.dot(&dv)).fold(f64::NEG_INFINITY, f64::max);
        assert!((support - g.q1()).abs() < 5e-3, "support along d* {support} should equal Q1 {}", g.q1());
    }
}
