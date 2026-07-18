//! **FOCI lab** — the rig behind the textbook chapter on collision on Gaussian-splat maps. Drives the real
//! [`ferromotion_core::foci`] overlap-integral collision so the reader can slide an *elongated* robot
//! toward a slot between two obstacle Gaussians and rotate it: head-on it jams, but turned to align with
//! the gap it slips through — the orientation-aware collision the paper is about.

use ferromotion_core::{collision_cost, Gaussian3, RobotSplat};
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

const GAP: f64 = 0.6; // half-gap of the slot in y

#[wasm_bindgen]
pub struct FociLab {
    env: Vec<Gaussian3>,
    robot: RobotSplat,
    // robot body-frame std devs (for drawing)
    rsx: f64,
    rsy: f64,
    x: f64,
    y: f64,
    yaw: f64,
}

#[wasm_bindgen]
impl FociLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> FociLab {
        // a corridor along world-x, walled off in ±y by two obstacle Gaussians
        let env = vec![
            Gaussian3::axis_aligned(Vector3::new(0.0, GAP, 0.0), Vector3::new(0.4, 0.3, 0.4)),
            Gaussian3::axis_aligned(Vector3::new(0.0, -GAP, 0.0), Vector3::new(0.4, 0.3, 0.4)),
        ];
        // robot elongated along its BODY-y axis (long and thin)
        let (rsx, rsy) = (0.12, 0.7);
        let robot = RobotSplat::single(Vector3::new(rsx, rsy, 0.12));
        FociLab { env, robot, rsx, rsy, x: -2.2, y: 0.0, yaw: 0.0 }
    }

    pub fn set_pose(&mut self, x: f64, y: f64, yaw: f64) {
        self.x = x;
        self.y = y;
        self.yaw = yaw;
    }

    /// The FOCI collision cost (summed overlap kernel) at the current pose.
    pub fn cost(&self) -> f64 {
        collision_cost(&self.env, &self.robot.posed(&Vector3::new(self.x, self.y, 0.0), self.yaw))
    }
    /// Collision cost at an arbitrary pose (for probing / the self-check).
    pub fn cost_at(&self, x: f64, y: f64, yaw: f64) -> f64 {
        collision_cost(&self.env, &self.robot.posed(&Vector3::new(x, y, 0.0), yaw))
    }

    // ---- geometry for drawing ----
    pub fn n_obstacles(&self) -> usize {
        self.env.len()
    }
    pub fn obs_x(&self, i: usize) -> f64 {
        self.env[i].mu.x
    }
    pub fn obs_y(&self, i: usize) -> f64 {
        self.env[i].mu.y
    }
    pub fn obs_sx(&self, i: usize) -> f64 {
        self.env[i].sigma[(0, 0)].sqrt()
    }
    pub fn obs_sy(&self, i: usize) -> f64 {
        self.env[i].sigma[(1, 1)].sqrt()
    }
    pub fn robot_sx(&self) -> f64 {
        self.rsx
    }
    pub fn robot_sy(&self) -> f64 {
        self.rsy
    }
    pub fn pose_x(&self) -> f64 {
        self.x
    }
    pub fn pose_y(&self) -> f64 {
        self.y
    }
    pub fn pose_yaw(&self) -> f64 {
        self.yaw
    }
    pub fn gap(&self) -> f64 {
        GAP
    }

    /// Cost of trying to pass the slot head-on (ψ=0) vs turned to align (ψ=90°), at the slot center — the
    /// headline invariant.
    pub fn head_on_cost(&self) -> f64 {
        self.cost_at(0.0, 0.0, 0.0)
    }
    pub fn turned_cost(&self) -> f64 {
        self.cost_at(0.0, 0.0, std::f64::consts::FRAC_PI_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turning_to_align_with_the_slot_cuts_the_cost() {
        let lab = FociLab::new();
        assert!(lab.turned_cost() < 0.25 * lab.head_on_cost(), "yaw-to-fit must cut cost: {} vs {}", lab.turned_cost(), lab.head_on_cost());
    }
}
