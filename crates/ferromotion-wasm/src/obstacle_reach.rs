//! **Obstacle-aware reach bench** — the sampling planner, on a real arm. A wall stands between the
//! arm and a target behind it; a naive straight reach would drive the arm *through* the wall, so the
//! arm plans a collision-free joint-space path around it ([`ferromotion_core::plan_arm_reach`], an
//! RRT* over the arm's swept collision geometry) and follows it. The plan is computed once and
//! replayed, so the animation stays smooth. Self-verifying: the executed path's minimum clearance to
//! the wall stays positive, where the straight path would penetrate it. Pure Rust → WebAssembly.

use ferromotion_core::{arm_clearance, from_urdf_str, plan_arm_reach, solve_diffik, DiffIkOptions, FrameTaskDef, ReachPlanOptions, Robot, Sdf, SdfScene};
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

const ARM: &str = r#"<robot name="a"><link name="world"/><link name="base"/>
  <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
  <joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

const TIP: Vector3<f64> = Vector3::new(0.0, 0.0, 0.05);
const LINK_R: f64 = 0.03;

#[wasm_bindgen]
pub struct ObstacleReachLab {
    robot: Robot,
    q: Vec<f64>,
    q_start: Vec<f64>,
    ee: usize,
    wall: SdfScene,
    wall_c: Vector3<f64>,
    wall_h: Vector3<f64>,
    target: Vector3<f64>,
    path: Vec<Vec<f64>>,
    planned: bool, // false ⇒ the planner fell back to the (colliding) straight line
    naive_min: f64,
    wp: usize,
    min_clear: f64,
    hold: u32,
}

fn build_reach() -> ObstacleReachLab {
    let robot = from_urdf_str(ARM, "world", "tool").unwrap();
    let ee = robot.dof();
    let q0 = vec![0.0; ee];
    let target = Vector3::new(0.42, 0.0, 0.42);
    let q_goal = solve_diffik(&robot, &[FrameTaskDef::new(ee, TIP, target, 2.0, 1.0)], &q0, &DiffIkOptions::default()).q;

    // wall at the tool midpoint of the naive interpolation ⇒ the straight path passes through it
    let q_mid: Vec<f64> = (0..ee).map(|i| 0.5 * (q0[i] + q_goal[i])).collect();
    let tool_mid = (robot.fk(&q_mid) * nalgebra::Point3::from(TIP)).coords;
    let wall_c = tool_mid;
    let wall_h = Vector3::new(0.06, 0.16, 0.11);
    let wall = SdfScene { prims: vec![Sdf::Box { center: wall_c, half: wall_h }] };

    // naive straight-path clearance (the receipt for "why plan")
    let mut naive_min = f64::INFINITY;
    for k in 0..=24 {
        let t = k as f64 / 24.0;
        let q: Vec<f64> = (0..ee).map(|i| q0[i] + t * (q_goal[i] - q0[i])).collect();
        naive_min = naive_min.min(arm_clearance(&robot, &q, &wall, LINK_R, 3));
    }

    // plan around the wall
    let pad = 0.7;
    let lo: Vec<f64> = (0..ee).map(|i| q0[i].min(q_goal[i]) - pad).collect();
    let hi: Vec<f64> = (0..ee).map(|i| q0[i].max(q_goal[i]) + pad).collect();
    let opts = ReachPlanOptions { max_iters: 14000, margin: 0.012, edge_res: 0.03, ..Default::default() };
    let (path, planned) = match plan_arm_reach(&robot, &q0, &q_goal, &wall, &lo, &hi, &opts) {
        Some(mut p) => {
            let far = p.last().map(|w| w.iter().zip(&q_goal).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).unwrap_or(1.0);
            if far > 1e-3 {
                p.push(q_goal.clone());
            }
            (p, true)
        }
        None => (vec![q0.clone(), q_goal.clone()], false),
    };

    ObstacleReachLab { robot, q: q0.clone(), q_start: q0, ee, wall, wall_c, wall_h, target, path, planned, naive_min, wp: 1, min_clear: f64::INFINITY, hold: 0 }
}

#[wasm_bindgen]
impl ObstacleReachLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ObstacleReachLab {
        build_reach()
    }

    /// Advance the arm `n` frames along the planned path; loop when it arrives.
    #[allow(clippy::needless_range_loop)]
    pub fn tick(&mut self, n: usize) {
        let max_step = 0.03;
        for _ in 0..n {
            if self.wp >= self.path.len() {
                // arrived: hold, then replay the (cached) plan from the start
                self.hold += 1;
                if self.hold > 60 {
                    self.q = self.q_start.clone();
                    self.wp = 1;
                    self.hold = 0;
                    self.min_clear = f64::INFINITY;
                }
                continue;
            }
            let goal = &self.path[self.wp];
            let mut arrived = true;
            for i in 0..self.ee {
                let d = (goal[i] - self.q[i]).clamp(-max_step, max_step);
                self.q[i] += d;
                if (goal[i] - self.q[i]).abs() > 1e-3 {
                    arrived = false;
                }
            }
            self.min_clear = self.min_clear.min(arm_clearance(&self.robot, &self.q, &self.wall, LINK_R, 3));
            if arrived {
                self.wp += 1;
            }
        }
    }

    /// Arm skeleton as flat `[x, z, …]` (side view): base → joint frames → tool.
    pub fn skeleton_xz(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..=self.ee {
            let p = self.robot.frame_pose(&self.q, i).translation.vector;
            out.push(p.x);
            out.push(p.z);
        }
        let tool = (self.robot.fk(&self.q) * nalgebra::Point3::from(TIP)).coords;
        out.push(tool.x);
        out.push(tool.z);
        out
    }

    /// The planned tool trail as flat `[x, z, …]` — the collision-free route the planner found.
    pub fn trail_xz(&self) -> Vec<f64> {
        self.path.iter().flat_map(|q| { let t = (self.robot.fk(q) * nalgebra::Point3::from(TIP)).coords; [t.x, t.z] }).collect()
    }

    /// Wall `[center_x, center_z, half_x, half_z]`.
    pub fn wall_xz(&self) -> Vec<f64> {
        vec![self.wall_c.x, self.wall_c.z, self.wall_h.x, self.wall_h.z]
    }
    pub fn target_xz(&self) -> Vec<f64> {
        vec![self.target.x, self.target.z]
    }
    /// Minimum clearance the executed path has kept to the wall so far (metres; > 0 ⇒ never touched it).
    pub fn min_clearance(&self) -> f64 {
        if self.min_clear.is_finite() {
            self.min_clear
        } else {
            1.0
        }
    }
    /// The naive straight-line reach's clearance (negative ⇒ it would drive the arm into the wall).
    pub fn naive_clearance(&self) -> f64 {
        self.naive_min
    }
    pub fn planned(&self) -> bool {
        self.planned
    }
    pub fn n_waypoints(&self) -> usize {
        self.path.len()
    }
}

impl Default for ObstacleReachLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bench plans a genuine detour: the naive straight reach hits the wall, the planned path
    /// does not, and executing it keeps the whole arm clear.
    #[test]
    fn bench_plans_around_the_wall() {
        let mut lab = ObstacleReachLab::new();
        assert!(lab.planned, "planner fell back to the straight line");
        assert!(lab.naive_min < 0.0, "the wall should block the straight reach: {}", lab.naive_min);
        assert!(lab.path.len() > 2, "expected a multi-waypoint detour, got {}", lab.path.len());
        // execute the whole path and confirm it never touches the wall
        for _ in 0..4000 {
            lab.tick(1);
            if lab.wp >= lab.path.len() {
                break;
            }
        }
        eprintln!("obstacle-reach: naive {:.3} m, executed min clearance {:.3} m, {} waypoints", lab.naive_min, lab.min_clear, lab.path.len());
        assert!(lab.min_clear > 0.0, "the arm hit the wall on the planned path: {:.3}", lab.min_clear);
    }
}
