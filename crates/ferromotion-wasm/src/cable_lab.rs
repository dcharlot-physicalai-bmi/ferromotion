//! **Cable lab** — the rig behind the textbook chapter on cable-driven parallel robots. Drives the real
//! [`ferromotion_control::Cdpr`] closed-form tension distribution so the reader can drag a platform hung
//! from four cables and watch the pull redistribute — taut and feasible in the middle, slack or
//! over-tensioned at the edges of the workspace.

use ferromotion_control::Cdpr;
use nalgebra::{Vector2, Vector3};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct CableLab {
    cdpr: Cdpr,
    x: f64,
    y: f64,
    theta: f64,
    weight: f64,
    torque: f64,
    tensions: Vec<f64>,
    feasible: bool,
    residual: f64,
}

#[wasm_bindgen]
impl CableLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CableLab {
        let cdpr = Cdpr {
            anchors: vec![
                Vector2::new(-2.0, -2.0),
                Vector2::new(2.0, -2.0),
                Vector2::new(2.0, 2.0),
                Vector2::new(-2.0, 2.0),
            ],
            attach: vec![
                Vector2::new(-0.35, -0.35),
                Vector2::new(0.35, -0.35),
                Vector2::new(0.35, 0.35),
                Vector2::new(-0.35, 0.35),
            ],
            t_min: 1.0,
            t_max: 40.0,
        };
        let mut lab = CableLab { cdpr, x: 0.0, y: 0.0, theta: 0.0, weight: 10.0, torque: 0.0, tensions: vec![], feasible: false, residual: 0.0 };
        lab.solve();
        lab
    }

    pub fn set_pose(&mut self, x: f64, y: f64, theta: f64) {
        self.x = x;
        self.y = y;
        self.theta = theta;
    }
    pub fn set_weight(&mut self, w: f64) {
        self.weight = w;
    }
    pub fn set_torque(&mut self, t: f64) {
        self.torque = t;
    }

    /// Solve the tension distribution to hold the platform (cables must apply +weight upward, plus any
    /// commanded torque).
    pub fn solve(&mut self) {
        let res = self.cdpr.tension_distribution(self.x, self.y, self.theta, Vector3::new(0.0, self.weight, self.torque));
        self.tensions = res.tensions.iter().cloned().collect();
        self.feasible = res.feasible;
        self.residual = res.residual;
    }

    pub fn n(&self) -> usize {
        self.cdpr.anchors.len()
    }
    pub fn anchor_x(&self, i: usize) -> f64 { self.cdpr.anchors[i].x }
    pub fn anchor_y(&self, i: usize) -> f64 { self.cdpr.anchors[i].y }
    pub fn t_max(&self) -> f64 { self.cdpr.t_max }
    pub fn t_min(&self) -> f64 { self.cdpr.t_min }

    /// World position of platform attachment `i` at the current pose.
    pub fn attach_x(&self, i: usize) -> f64 {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        let l = self.cdpr.attach[i];
        self.x + c * l.x - s * l.y
    }
    pub fn attach_y(&self, i: usize) -> f64 {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        let l = self.cdpr.attach[i];
        self.y + s * l.x + c * l.y
    }
    pub fn tension(&self, i: usize) -> f64 {
        self.tensions.get(i).copied().unwrap_or(0.0)
    }
    pub fn pose_x(&self) -> f64 { self.x }
    pub fn pose_y(&self) -> f64 { self.y }
    pub fn feasible(&self) -> bool { self.feasible }
    pub fn max_tension(&self) -> f64 {
        self.tensions.iter().cloned().fold(0.0, f64::max)
    }
    pub fn min_tension(&self) -> f64 {
        self.tensions.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_hold_is_feasible_and_balanced() {
        let mut lab = CableLab::new();
        lab.set_pose(0.0, 0.0, 0.0);
        lab.solve();
        assert!(lab.feasible, "centered hold should be feasible: {:?}", lab.tensions);
        assert!(lab.residual < 1e-9, "equilibrium residual {}", lab.residual);
        assert!(lab.min_tension() >= lab.t_min() - 1e-9, "all cables taut");
    }

    #[test]
    fn dragging_toward_a_corner_eventually_becomes_infeasible() {
        // Near the frame edge one cable must slacken (or another over-tension) → outside the
        // wrench-feasible workspace.
        let mut lab = CableLab::new();
        lab.set_pose(1.85, 1.85, 0.0); // almost at the top-right anchor
        lab.solve();
        assert!(!lab.feasible, "a pose jammed into the corner should be infeasible: {:?}", lab.tensions);
    }
}
