//! **Movement-primitive lab** — the rig behind the textbook chapter on learning from one demonstration.
//!
//! Show a robot a motion *once* and it should be able to do it again — and, more usefully, do the same
//! *shape* of motion to a place you never demonstrated. A Dynamic Movement Primitive is the classic way
//! to get both. It is a stable spring-damper pulling toward the goal, plus a learned forcing term that
//! bends the path into the demonstrated shape — and that forcing term is gated by a **phase** variable
//! that decays to zero, so it always switches off before the end. Whatever the robot learned, the goal
//! is therefore a *structural* attractor: the motion cannot fail to arrive.
//!
//! The rig fits two real [`ferromotion_control::Dmp`]s (one per axis) to a 2-D demonstration and rolls
//! them out to new goals, so the reader can drag the target and watch the learned shape follow it.

use ferromotion_control::Dmp;
use wasm_bindgen::prelude::*;

/// Steps integrated per demonstration-duration, extended past the demo so the phase decays and the
/// goal is actually reached (the demo shape occupies the first `steps`, then it settles on g).
const CONVERGE: f64 = 2.5;

#[wasm_bindgen]
pub struct DmpLab {
    dx: Dmp,
    dy: Dmp,
    dt: f64,
    steps: usize,
    x0: f64,
    y0: f64,
    gx: f64,
    gy: f64,
    fitted: bool,
}

#[wasm_bindgen]
impl DmpLab {
    #[wasm_bindgen(constructor)]
    pub fn new(n_basis: usize) -> DmpLab {
        DmpLab {
            dx: Dmp::new(n_basis),
            dy: Dmp::new(n_basis),
            dt: 0.01,
            steps: 100,
            x0: 0.0,
            y0: 0.0,
            gx: 1.0,
            gy: 0.0,
            fitted: false,
        }
    }

    /// Fit both axes to a 2-D demonstration (`xs[i], ys[i]` sampled every `dt`).
    pub fn fit(&mut self, xs: &[f64], ys: &[f64], dt: f64) {
        let n = xs.len().min(ys.len());
        self.dx.fit(&xs[..n], dt);
        self.dy.fit(&ys[..n], dt);
        self.dt = dt;
        self.steps = n.saturating_sub(1).max(1);
        self.x0 = xs[0];
        self.y0 = ys[0];
        self.gx = xs[n - 1];
        self.gy = ys[n - 1];
        self.fitted = true;
    }

    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
    pub fn demo_start_x(&self) -> f64 { self.x0 }
    pub fn demo_start_y(&self) -> f64 { self.y0 }
    pub fn demo_goal_x(&self) -> f64 { self.gx }
    pub fn demo_goal_y(&self) -> f64 { self.gy }

    /// Number of demonstration samples (the first this-many rollout points trace the learned shape).
    pub fn demo_steps(&self) -> usize {
        self.steps
    }

    /// Roll the learned primitive out to a (possibly new) start, goal, and time constant. Returns the
    /// 2-D path interleaved `[x0, y0, x1, y1, …]`. `tau_scale` multiplies the demonstrated duration.
    pub fn rollout(&self, sx: f64, sy: f64, gx: f64, gy: f64, tau_scale: f64) -> Vec<f64> {
        let tau = self.dx.tau * tau_scale;
        let steps = ((self.steps as f64) * tau_scale * CONVERGE).round().max(1.0) as usize;
        let px = self.dx.rollout(sx, gx, tau, self.dt, steps);
        let py = self.dy.rollout(sy, gy, tau, self.dt, steps);
        let mut out = Vec::with_capacity(2 * px.len());
        for i in 0..px.len() {
            out.push(px[i]);
            out.push(py[i]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2-D demonstration with *distinct* endpoints on both axes (start (0,0) → goal (2, 1.2)) plus a
    /// sideways bump, so the "shape" is the deviation from the chord and neither axis degenerates.
    /// (A DMP scales each axis by its own g − y₀, so a shape on an axis whose endpoints coincide would
    /// be scaled to nothing — the classic gotcha this demo avoids.)
    fn arc_demo(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let s = i as f64 / (n - 1) as f64;
            xs.push(2.0 * s); // 0 → 2 along x
            ys.push(1.2 * s + (std::f64::consts::PI * s).sin() * 0.7); // 0 → 1.2 with a bump
        }
        (xs, ys)
    }

    /// Max perpendicular deviation of a path from its own start→goal chord, over the chord length —
    /// a scale-and-translation-invariant fingerprint of the motion's *shape*.
    fn shape_signature(path: &[f64]) -> f64 {
        let n = path.len() / 2;
        let (x0, y0) = (path[0], path[1]);
        let (gx, gy) = (path[2 * (n - 1)], path[2 * (n - 1) + 1]);
        let (cx, cy) = (gx - x0, gy - y0);
        let clen = (cx * cx + cy * cy).sqrt().max(1e-9);
        let mut worst: f64 = 0.0;
        for i in 0..n {
            let (px, py) = (path[2 * i] - x0, path[2 * i + 1] - y0);
            let perp = (px * cy - py * cx).abs() / clen; // |cross| / |chord|
            worst = worst.max(perp);
        }
        worst / clen
    }

    #[test]
    fn it_reproduces_the_demonstration_it_was_shown() {
        let (xs, ys) = arc_demo(120);
        let mut lab = DmpLab::new(20);
        lab.fit(&xs, &ys, 0.01);
        let path = lab.rollout(lab.demo_start_x(), lab.demo_start_y(), lab.demo_goal_x(), lab.demo_goal_y(), 1.0);
        // The demo shape occupies the first `demo_steps` rollout points (same dt, same duration).
        let m = lab.demo_steps();
        let mut worst: f64 = 0.0;
        for i in 0..=m.min(xs.len() - 1) {
            worst = worst.max((path[2 * i] - xs[i]).abs().max((path[2 * i + 1] - ys[i]).abs()));
        }
        assert!(worst < 0.08, "rollout did not reproduce the demo: worst dev {worst}");
    }

    #[test]
    fn the_goal_is_a_structural_attractor_whatever_was_learned() {
        // THE CHAPTER. The forcing term is gated by a phase that decays to zero, so the motion must
        // end at the goal — for the demo's goal, for a goal never demonstrated, even with garbage
        // weights. Convergence does not depend on what was learned.
        let (xs, ys) = arc_demo(120);
        let mut lab = DmpLab::new(20);
        lab.fit(&xs, &ys, 0.01);
        for &(gx, gy) in &[(2.0, 0.0), (3.5, 1.2), (-1.0, -2.0), (0.5, 3.0)] {
            let path = lab.rollout(0.0, 0.0, gx, gy, 1.0);
            let n = path.len() / 2;
            let (ex, ey) = (path[2 * (n - 1)], path[2 * (n - 1) + 1]);
            assert!((ex - gx).abs() < 0.02 && (ey - gy).abs() < 0.02, "did not reach goal ({gx},{gy}): ended ({ex},{ey})");
        }
        // Even with the weights scrambled, the spring-damper still lands on the goal.
        let mut wild = DmpLab::new(20);
        wild.fit(&xs, &ys, 0.01);
        wild.dx.w.iter_mut().enumerate().for_each(|(i, w)| *w = 500.0 * ((i as f64).sin()));
        let path = wild.rollout(0.0, 0.0, 2.0, 1.0, 1.0);
        let n = path.len() / 2;
        assert!((path[2 * (n - 1)] - 2.0).abs() < 0.05, "structural convergence must survive any weights");
    }

    #[test]
    fn a_new_goal_reproduces_the_same_shape() {
        // Generalization: roll out to goals of different length and direction; the shape fingerprint
        // (bump as a fraction of the chord) is preserved — that is what "the same motion, elsewhere"
        // means for a DMP.
        let (xs, ys) = arc_demo(120);
        let mut lab = DmpLab::new(20);
        lab.fit(&xs, &ys, 0.01);
        // Uniform rescalings of the demo displacement (2, 1.2) — same direction, new distance. A DMP
        // scales each axis by its own g − y₀, so along the demo direction the shape is preserved
        // exactly; the reader dragging the goal off-axis will see it shear, which is real DMP behaviour.
        let base = shape_signature(&lab.rollout(0.0, 0.0, 2.0, 1.2, 1.0));
        assert!(base > 0.05, "the demo should have a real bump to preserve: {base}");
        for k in [0.5, 1.5, 2.0, 3.0] {
            let (gx, gy) = (2.0 * k, 1.2 * k);
            let sig = shape_signature(&lab.rollout(0.0, 0.0, gx, gy, 1.0));
            let rel = (sig - base).abs() / base;
            assert!(rel < 0.05, "shape not preserved to a rescaled goal ×{k}: signature {sig} vs {base}");
        }
    }

    #[test]
    fn tau_rescales_duration_but_not_the_path() {
        // Slowing the motion (larger τ) traces the same geometric curve, just with more samples.
        let (xs, ys) = arc_demo(120);
        let mut lab = DmpLab::new(20);
        lab.fit(&xs, &ys, 0.01);
        let fast = lab.rollout(0.0, 0.0, 2.0, 1.2, 1.0);
        let slow = lab.rollout(0.0, 0.0, 2.0, 1.2, 2.0);
        // Same shape fingerprint despite different lengths.
        let (sf, ss) = (shape_signature(&fast), shape_signature(&slow));
        assert!((sf - ss).abs() / sf < 0.05, "τ changed the path shape: {sf} vs {ss}");
        assert!(slow.len() > fast.len(), "larger τ should take more steps");
    }
}
