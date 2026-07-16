//! **Safety-filter lab** — the rig behind the textbook chapter on control barrier functions.
//!
//! A point robot is driven by whatever command the reader gives it (a velocity toward a target). Between
//! that command and the robot sits a **control barrier function** filter: for each hazard it forms the
//! barrier `h(x) = ‖x − c‖ − r` (positive outside the disk) and folds `ḣ + α·h ≥ 0` into an affine
//! constraint on the command, then solves `min ½‖u − u_nom‖²` subject to those constraints. A safe
//! command passes through untouched; an unsafe one is projected to the nearest command that keeps the
//! robot out. The reader can aim straight at a hazard and the robot simply *cannot* be driven in — it
//! slides along the boundary. Runs the real [`ferromotion_control::CbfFilter`].

use ferromotion_control::{CbfConstraint, CbfFilter};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CbfLab {
    x: f64,
    y: f64,
    tx: f64,
    ty: f64,
    gain: f64,
    alpha: f64,
    obs: Vec<(f64, f64, f64)>, // (cx, cy, r)
}

#[wasm_bindgen]
impl CbfLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CbfLab {
        CbfLab { x: 0.0, y: 0.0, tx: 0.0, ty: 0.0, gain: 2.0, alpha: 4.0, obs: Vec::new() }
    }

    pub fn set_pos(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }
    pub fn set_target(&mut self, tx: f64, ty: f64) {
        self.tx = tx;
        self.ty = ty;
    }
    pub fn set_gain(&mut self, k: f64) {
        self.gain = k;
    }
    /// Class-K gain α: how hard the filter lets the robot approach a boundary. Larger α ⇒ it may ride
    /// closer before the barrier bites; smaller α ⇒ a wider, more cautious berth.
    pub fn set_alpha(&mut self, a: f64) {
        self.alpha = a.max(0.0);
    }
    pub fn add_obstacle(&mut self, cx: f64, cy: f64, r: f64) {
        self.obs.push((cx, cy, r));
    }
    pub fn clear_obstacles(&mut self) {
        self.obs.clear();
    }

    /// The nominal (unfiltered) command: a proportional pull toward the target.
    fn nominal(&self) -> [f64; 2] {
        [self.gain * (self.tx - self.x), self.gain * (self.ty - self.y)]
    }

    /// Build the CBF row for one hazard. Single integrator `ẋ = u` ⇒ `f = 0`, `g = I`, so
    /// `L_g h = ∂h`, `L_f h = 0`, and the row enforces `ḣ + α·h ≥ 0`.
    fn constraint_for(&self, o: (f64, f64, f64)) -> CbfConstraint {
        let (cx, cy, r) = o;
        let (dx, dy) = (self.x - cx, self.y - cy);
        let d = (dx * dx + dy * dy).sqrt().max(1e-9);
        let h = d - r;
        let grad = [dx / d, dy / d]; // ∂h = (x − c)/‖x − c‖
        CbfConstraint::relative_degree1(&grad, 0.0, h, self.alpha)
    }

    fn constraints(&self) -> Vec<CbfConstraint> {
        self.obs.iter().map(|&o| self.constraint_for(o)).collect()
    }

    /// The filtered, safe command actually applied to the robot.
    fn filtered(&self) -> Vec<f64> {
        CbfFilter::new().filter(&self.nominal(), &self.constraints())
    }

    // --- exposed to JS for drawing/readout ---
    pub fn nominal_x(&self) -> f64 { self.nominal()[0] }
    pub fn nominal_y(&self) -> f64 { self.nominal()[1] }
    pub fn filtered_x(&self) -> f64 { self.filtered()[0] }
    pub fn filtered_y(&self) -> f64 { self.filtered()[1] }
    pub fn x(&self) -> f64 { self.x }
    pub fn y(&self) -> f64 { self.y }

    /// How much the filter altered the command — 0 when the nominal command was already safe.
    pub fn correction(&self) -> f64 {
        let (n, f) = (self.nominal(), self.filtered());
        ((n[0] - f[0]).powi(2) + (n[1] - f[1]).powi(2)).sqrt()
    }

    /// Safety margin: the smallest barrier value over all hazards (≥ 0 means outside every disk).
    pub fn min_h(&self) -> f64 {
        self.obs
            .iter()
            .map(|&(cx, cy, r)| ((self.x - cx).powi(2) + (self.y - cy).powi(2)).sqrt() - r)
            .fold(f64::INFINITY, f64::min)
    }

    /// Advance one step under the *filtered* command (single integrator).
    pub fn step(&mut self, dt: f64) {
        let u = self.filtered();
        self.x += u[0] * dt;
        self.y += u[1] * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_obstacle() -> CbfLab {
        let mut l = CbfLab::new();
        l.set_pos(-2.0, 0.0);
        l.add_obstacle(0.0, 0.0, 1.0);
        l.set_alpha(4.0);
        l.set_gain(2.0);
        l
    }

    #[test]
    fn the_robot_cannot_be_driven_into_the_hazard() {
        // THE CHAPTER. Aim the target dead centre of the hazard and integrate; the barrier must hold
        // for every step, no matter that the command points straight in.
        let mut l = one_obstacle();
        l.set_target(0.0, 0.0); // straight into the disk
        let mut worst = f64::INFINITY;
        for _ in 0..20_000 {
            l.step(1e-3);
            worst = worst.min(l.min_h());
        }
        assert!(worst > -1e-6, "robot penetrated the hazard: min h = {worst:.2e}");
        // It should press right up to the boundary (h ≈ 0), not stop short.
        assert!(l.min_h() < 0.02, "robot did not reach the boundary: h = {}", l.min_h());
    }

    #[test]
    fn a_safe_command_passes_through_untouched() {
        // Target in the open; nothing between robot and goal ⇒ the filter is a no-op.
        let mut l = one_obstacle();
        l.set_pos(-2.0, 3.0);
        l.set_target(-4.0, 3.0); // pulling away from the hazard
        assert!(l.correction() < 1e-9, "a safe command was altered: {}", l.correction());
        assert!((l.filtered_x() - l.nominal_x()).abs() < 1e-9);
    }

    #[test]
    fn the_intervention_is_the_minimal_orthogonal_projection() {
        // On the boundary, pushing inward: the safe command is u_nom with only its inward (normal)
        // component removed — the tangential part survives, so the robot slides, not stops.
        let mut l = one_obstacle();
        l.set_pos(-1.0, 0.0); // on the boundary, left of centre (h = 0)
        l.set_target(2.0, 0.0); // straight through the disk
        let n = [l.nominal_x(), l.nominal_y()];
        let f = [l.filtered_x(), l.filtered_y()];
        // The normal at (−1,0) points in −x (outward). The inward part of u_nom (+x) must be gone…
        assert!(f[0].abs() < 1e-6, "inward component not removed: {}", f[0]);
        // …and the removed vector (n − f) is parallel to the barrier gradient (the x axis here).
        assert!((n[1] - f[1]).abs() < 1e-6, "correction had a spurious tangential part");
        // The correction is exactly the projection: ‖u* − u_nom‖ equals the inward speed it cancelled.
        assert!((l.correction() - n[0].abs()).abs() < 1e-6);
    }

    #[test]
    fn a_tangential_command_slides_freely_along_the_boundary() {
        // Moving along the boundary is safe (ḣ = 0), so a tangential command is untouched.
        let mut l = one_obstacle();
        l.set_pos(-1.0, 0.0);
        l.set_target(-1.0, 3.0); // straight up — tangent to the circle here
        assert!(l.correction() < 1e-6, "tangential motion should be unfiltered: {}", l.correction());
    }

    #[test]
    fn many_hazards_are_all_respected_at_once() {
        // A field of overlapping hazards; drive across it and never enter any of them.
        let mut l = CbfLab::new();
        l.set_pos(-3.0, -3.0);
        for &(cx, cy) in &[(-1.0, -1.0), (0.0, 0.0), (1.0, 0.5), (-0.5, 1.0)] {
            l.add_obstacle(cx, cy, 0.8);
        }
        l.set_alpha(5.0);
        l.set_gain(2.5);
        l.set_target(3.0, 3.0); // goal on the far side of the field
        let mut worst = f64::INFINITY;
        for _ in 0..30_000 {
            l.step(1e-3);
            worst = worst.min(l.min_h());
        }
        assert!(worst > -1e-6, "penetrated a hazard in the field: min h = {worst:.2e}");
    }

    #[test]
    fn alpha_sets_how_close_it_rides() {
        // Larger α lets the robot commit later and pass nearer the boundary. Compare the closest
        // approach for a small vs large α on a grazing trajectory.
        let closest = |alpha: f64| -> f64 {
            let mut l = CbfLab::new();
            l.set_pos(-3.0, -1.05);
            l.add_obstacle(0.0, 0.0, 1.0);
            l.set_alpha(alpha);
            l.set_gain(2.0);
            l.set_target(3.0, -1.05); // skim just past the bottom of the disk
            let mut worst = f64::INFINITY;
            for _ in 0..20_000 {
                l.step(1e-3);
                worst = worst.min(l.min_h());
            }
            worst
        };
        let cautious = closest(1.0);
        let aggressive = closest(12.0);
        assert!(cautious > -1e-6 && aggressive > -1e-6, "both must stay safe");
        assert!(aggressive < cautious, "larger α should ride closer: {aggressive} vs {cautious}");
    }
}
