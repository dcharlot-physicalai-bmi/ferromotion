//! **Reactive-motion lab** — the rig behind the textbook chapter on composing behaviors.
//!
//! A robot arm usually has to do several things at once: reach a goal, keep its whole body clear of an
//! obstacle, stay smooth. The old answer was to plan a path that trades these off ahead of time; RMPflow
//! does it *reactively* and geometrically. Each behavior is a **Riemannian Motion Policy** — a desired
//! acceleration plus a state-dependent **metric** saying how much it matters right now — and the
//! behaviors are fused by a metric-weighted pullback to the joints. The trick is the metric: an
//! obstacle's metric blows up as the arm nears it, so that behavior *dominates* close in and *vanishes*
//! far away, with no planner and no mode switch.
//!
//! The rig drives the real [`ferromotion_control::RmpArm`] on a two-link arm built from URDF through the
//! real `ferromotion-core` kinematics, so the reader can drag the obstacle into the arm's path and watch
//! the whole arm bend around it while the hand still reaches the goal.

use ferromotion_control::RmpArm;
use ferromotion_core::{from_urdf_str, Robot};
use nalgebra::{Vector2, Vector3};
use wasm_bindgen::prelude::*;

const ARM2: &str = r#"<robot name="a2">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="1 0 0"/></joint>
</robot>"#;

#[wasm_bindgen]
pub struct RmpLab {
    robot: Robot,
    q: Vec<f64>,
    qd: Vec<f64>,
    goal: Vector2<f64>,
    obstacle: Vector2<f64>,
    obstacle_r: f64,
}

fn control_points() -> Vec<(usize, Vector3<f64>)> {
    let mut cps = Vec::new();
    for s in [0.5, 1.0] {
        cps.push((1usize, Vector3::new(s, 0.0, 0.0))); // along link 1
    }
    for s in [0.5, 1.0] {
        cps.push((2usize, Vector3::new(s, 0.0, 0.0))); // along link 2 (1.0 = EE)
    }
    cps
}

#[wasm_bindgen]
impl RmpLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RmpLab {
        let robot = from_urdf_str(ARM2, "base", "tool").expect("valid arm URDF");
        RmpLab {
            robot,
            q: vec![0.9, 0.6],
            qd: vec![0.0, 0.0],
            goal: Vector2::new(0.6, -1.4),
            obstacle: Vector2::new(1.5, 0.0),
            obstacle_r: 0.4,
        }
    }

    fn arm(&self) -> RmpArm<'_> {
        RmpArm {
            robot: &self.robot,
            goal: self.goal,
            obstacle: self.obstacle,
            obstacle_r: self.obstacle_r,
            control_points: control_points(),
            kp: 8.0,
            kd: 5.0,
            d0: 0.5,
            k_rep: 0.5,
            kd_obs: 2.0,
            w_attract: 1.0,
        }
    }

    pub fn set_goal(&mut self, x: f64, y: f64) {
        self.goal = Vector2::new(x, y);
    }
    pub fn set_obstacle(&mut self, x: f64, y: f64) {
        self.obstacle = Vector2::new(x, y);
    }
    pub fn set_obstacle_r(&mut self, r: f64) {
        self.obstacle_r = r.max(0.05);
    }
    pub fn reset(&mut self, q0: f64, q1: f64) {
        self.q = vec![q0, q1];
        self.qd = vec![0.0, 0.0];
    }

    /// Advance the reactive policy one step.
    pub fn step(&mut self, dt: f64) {
        let a = self.arm().accel(&self.q, &self.qd);
        for i in 0..2 {
            self.qd[i] += a[i] * dt;
            self.q[i] += self.qd[i] * dt;
        }
    }

    // --- geometry for drawing ---
    fn point(&self, frame: usize, off: f64) -> [f64; 2] {
        let w = (self.robot.frame_pose(&self.q, frame) * nalgebra::Point3::new(off, 0.0, 0.0)).coords;
        [w.x, w.y]
    }

    /// Joint chain `[base, elbow, hand]` interleaved `[x,y,…]` for drawing the arm.
    pub fn joints_xy(&self) -> Vec<f64> {
        let base = [0.0, 0.0];
        let elbow = self.point(1, 1.0);
        let hand = self.point(2, 1.0);
        vec![base[0], base[1], elbow[0], elbow[1], hand[0], hand[1]]
    }

    /// The control points sampled along the arm (interleaved) — the places that watch the obstacle.
    pub fn control_points_xy(&self) -> Vec<f64> {
        let mut o = Vec::new();
        for &(f, off) in &control_points() {
            let p = self.point(f, off.x);
            o.push(p[0]);
            o.push(p[1]);
        }
        o
    }

    pub fn ee_x(&self) -> f64 { self.point(2, 1.0)[0] }
    pub fn ee_y(&self) -> f64 { self.point(2, 1.0)[1] }
    pub fn goal_x(&self) -> f64 { self.goal.x }
    pub fn goal_y(&self) -> f64 { self.goal.y }
    pub fn obstacle_x(&self) -> f64 { self.obstacle.x }
    pub fn obstacle_y(&self) -> f64 { self.obstacle.y }
    pub fn obstacle_radius(&self) -> f64 { self.obstacle_r }

    /// Smallest clearance of any arm point to the obstacle surface (negative ⇒ collision).
    pub fn min_clearance(&self) -> f64 {
        self.arm().min_clearance(&self.q)
    }

    pub fn goal_error(&self) -> f64 {
        ((self.ee_x() - self.goal.x).powi(2) + (self.ee_y() - self.goal.y).powi(2)).sqrt()
    }

    /// Whether the obstacle-avoidance behavior is currently engaged (any point within influence).
    pub fn avoidance_active(&self) -> bool {
        self.min_clearance() < 0.5 // d0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(lab: &mut RmpLab, steps: usize) -> f64 {
        let mut worst = lab.min_clearance();
        for _ in 0..steps {
            lab.step(2e-3);
            worst = worst.min(lab.min_clearance());
        }
        worst
    }

    #[test]
    fn it_reaches_the_goal_while_the_whole_arm_avoids_the_obstacle() {
        // THE CHAPTER. Goal on the far side of an obstacle that sits on the direct path; the arm must
        // bend around it and still land the hand on the goal — no planner, just the reactive fusion.
        let mut lab = RmpLab::new();
        lab.reset(0.9, 0.6);
        lab.set_goal(0.6, -1.4);
        lab.set_obstacle(1.5, 0.0);
        let worst = settle(&mut lab, 6000);
        assert!(worst > 0.0, "arm collided with the obstacle: min clearance {worst:.3}");
        assert!(worst < 0.5, "obstacle never engaged (clearance {worst:.3}) — not exercising avoidance");
        assert!(lab.goal_error() < 0.1, "hand did not reach the goal: error {:.3}", lab.goal_error());
    }

    #[test]
    fn a_far_obstacle_does_not_perturb_the_reach() {
        // The metric makes avoidance vanish far away: with the obstacle off in a corner, the arm
        // reaches exactly as if it were not there (no residual push distorting the motion).
        let mut lab = RmpLab::new();
        lab.reset(0.9, 0.6);
        lab.set_goal(0.6, -1.4);
        lab.set_obstacle(-1.6, 1.6); // far from any reasonable path
        let worst = settle(&mut lab, 6000);
        assert!(worst > 0.5, "a far obstacle should never engage: clearance {worst:.3}");
        assert!(lab.goal_error() < 0.05, "far obstacle perturbed the reach: error {:.3}", lab.goal_error());
    }

    #[test]
    fn the_arm_bends_more_the_closer_the_obstacle_sits_to_the_path() {
        // Avoidance is graded, not on/off: an obstacle right on the path forces a smaller clearance
        // (a bigger detour) than one grazing the edge — the metric scales the response with proximity.
        let reach = |ox: f64, oy: f64| {
            let mut lab = RmpLab::new();
            lab.reset(0.9, 0.6);
            lab.set_goal(0.6, -1.4);
            lab.set_obstacle(ox, oy);
            settle(&mut lab, 6000)
        };
        let on_path = reach(1.2, -0.2); // squarely in the way
        let grazing = reach(1.7, 0.6); // off to the side
        assert!(on_path > 0.0 && grazing > 0.0, "both must stay collision-free");
        assert!(on_path < grazing, "a closer obstacle should force a tighter pass: {on_path:.3} vs {grazing:.3}");
    }
}
