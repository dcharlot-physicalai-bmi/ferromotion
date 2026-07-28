//! **Planned grasp** — the sampling planner and the deformable grasp, in one loop. A wall stands
//! between the arm and a soft block; the arm plans a collision-free joint path AROUND the wall
//! ([`plan_arm_reach`], RRT* over the arm's swept geometry, avoiding both the wall and the block) to a
//! pre-grasp pose above the block, follows it, then descends, closes on the block, and lifts it away.
//! This is the honest resolution of the "two-obstacle planned grasp": use the *proven* RRT reach for
//! the global detour and the *proven* friction grasp for the pickup — no fragile trajectory optimizer.
//! Pure Rust → WebAssembly.

use ferromotion_core::{arm_clearance, from_urdf_str, plan_arm_reach, solve_diffik, DiffIkOptions, FrameTaskDef, ReachPlanOptions, Robot, Sdf, SdfScene};
use ferromotion_coupled::GraspFemSim;
use nalgebra::Vector3;

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

const TIP: Vector3<f64> = Vector3::new(0.0, 0.0, 0.08);
const LINK_R: f64 = 0.03;

const EXECUTE: u8 = 0; // follow the planned path around the wall
const DESCEND: u8 = 1;
const GRASP: u8 = 2;
const LIFT: u8 = 3;
const HOLD: u8 = 4;

pub struct PlannedGrasp {
    pub robot: Robot,
    pub q: Vec<f64>,
    ee: usize,
    pub grasp: GraspFemSim,
    pub radius: f64,
    pub object_c: Vector3<f64>,
    wall_c: Vector3<f64>,
    wall_h: Vector3<f64>,
    scene: SdfScene, // wall + object, for planning + clearance receipts
    pub path: Vec<Vec<f64>>,
    pub planned: bool,
    pub naive_min: f64,
    pub min_clear: f64,
    wp: usize,
    q_goal: Vec<f64>,
    prev_grip: Vector3<f64>,
    open_hw: f64,
    grip_hw: f64,
    pub lift_h: f64,
    sub: usize,
    pub phase: u8,
    pub t: u32,
}

fn tool(robot: &Robot, q: &[f64]) -> Vector3<f64> {
    (robot.frame_pose(q, robot.dof()) * nalgebra::Point3::from(TIP)).coords
}

impl PlannedGrasp {
    pub fn new() -> Self {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let ee = robot.dof();
        let q0 = vec![0.0; ee];
        let radius = 0.09;
        let object_c = Vector3::new(0.42, 0.0, 0.30);
        let floor = object_c.z - radius;
        let pregrasp = object_c + Vector3::new(0.0, 0.0, 0.22);
        let q_goal = solve_diffik(&robot, &[FrameTaskDef::new(ee, TIP, pregrasp, 2.0, 1.0)], &q0, &DiffIkOptions::default()).q;

        // wall at the tool midpoint of the naive q0→pre-grasp interpolation (⇒ the straight reach
        // passes through it); the object also goes in the collision scene so the plan avoids both.
        let q_mid: Vec<f64> = (0..ee).map(|i| 0.5 * (q0[i] + q_goal[i])).collect();
        let wall_c = (robot.fk(&q_mid) * nalgebra::Point3::from(TIP)).coords;
        let wall_h = Vector3::new(0.06, 0.16, 0.11);
        let scene = SdfScene {
            prims: vec![
                Sdf::Box { center: wall_c, half: wall_h },
                Sdf::Box { center: object_c, half: Vector3::repeat(radius) },
            ],
        };

        let mut naive_min = f64::INFINITY;
        for k in 0..=24 {
            let t = k as f64 / 24.0;
            let q: Vec<f64> = (0..ee).map(|i| q0[i] + t * (q_goal[i] - q0[i])).collect();
            naive_min = naive_min.min(arm_clearance(&robot, &q, &scene, LINK_R, 3));
        }

        let pad = 0.7;
        let lo: Vec<f64> = (0..ee).map(|i| q0[i].min(q_goal[i]) - pad).collect();
        let hi: Vec<f64> = (0..ee).map(|i| q0[i].max(q_goal[i]) + pad).collect();
        let opts = ReachPlanOptions { max_iters: 16000, margin: 0.012, edge_res: 0.03, ..Default::default() };
        let (path, planned) = match plan_arm_reach(&robot, &q0, &q_goal, &scene, &lo, &hi, &opts) {
            Some(mut p) => {
                let far = p.last().map(|w| w.iter().zip(&q_goal).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).unwrap_or(1.0);
                if far > 1e-3 {
                    p.push(q_goal.clone());
                }
                (p, true)
            }
            None => (vec![q0.clone(), q_goal.clone()], false),
        };

        // the soft block, as a FEM body at `object_c`
        let mut fem = ferromotion_fem::FemSim::box_grid(3, 3, 3, 0.06, 0.02, 1.0e4, 6.0e3, 2.0e-4);
        fem.damping = 0.02;
        let n = fem.n_verts() as f64;
        let c0: Vector3<f64> = fem.x.iter().sum::<Vector3<f64>>() / n;
        let off = object_c - c0;
        for v in fem.x.iter_mut() {
            *v += off;
        }
        let mut grasp = GraspFemSim::new(fem, object_c, radius + 0.03, 5.0e4, 0.9);
        grasp.floor = Some(floor);
        grasp.pad = radius * 1.7;

        let prev_grip = tool(&robot, &q0);
        PlannedGrasp {
            robot,
            q: q0,
            ee,
            grasp,
            radius,
            object_c,
            wall_c,
            wall_h,
            scene,
            path,
            planned,
            naive_min,
            min_clear: f64::INFINITY,
            wp: 1,
            q_goal: vec![0.0; ee],
            prev_grip,
            open_hw: radius + 0.03,
            grip_hw: radius - 0.016,
            lift_h: 0.20,
            sub: 30,
            phase: EXECUTE,
            t: 0,
        }
    }

    fn solve_to(&self, target: Vector3<f64>) -> Vec<f64> {
        let opts = DiffIkOptions { dt: 0.05, vmax: 3.0, max_iters: 200, use_limits: true, ..Default::default() };
        solve_diffik(&self.robot, &[FrameTaskDef::new(self.ee, TIP, target, 2.0, 1.0)], &self.q, &opts).q
    }

    /// One frame: advance the arm (planned-path replay in EXECUTE, IK-goal slew afterwards), then step
    /// the FEM block under gravity + floor + (once grasping) the gripper contact.
    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let contact = self.phase == GRASP || self.phase == LIFT;

        // ---- arm motion ----
        let mut arrived = false;
        if self.phase == EXECUTE {
            // replay the planned path around the wall
            let max_step = 0.05;
            if self.wp < self.path.len() {
                let goal = self.path[self.wp].clone();
                let mut at = true;
                for i in 0..self.ee {
                    let d = (goal[i] - self.q[i]).clamp(-max_step, max_step);
                    self.q[i] += d;
                    if (goal[i] - self.q[i]).abs() > 1e-3 {
                        at = false;
                    }
                }
                self.min_clear = self.min_clear.min(arm_clearance(&self.robot, &self.q, &self.scene, LINK_R, 3));
                if at {
                    self.wp += 1;
                }
            } else {
                arrived = true;
            }
        } else {
            // slew toward the phase's IK goal, capping Cartesian speed in contact
            let max_step = if contact { 0.03 } else { 0.06 };
            let mut dq = vec![0.0; self.ee];
            arrived = true;
            for i in 0..self.ee {
                dq[i] = (self.q_goal[i] - self.q[i]).clamp(-max_step, max_step);
                if (self.q_goal[i] - self.q[i]).abs() > 2e-3 {
                    arrived = false;
                }
            }
            let q_try: Vec<f64> = (0..self.ee).map(|i| self.q[i] + dq[i]).collect();
            if contact {
                let disp = (tool(&self.robot, &q_try) - self.prev_grip).norm();
                let scale = if disp > 0.0016 { 0.0016 / disp } else { 1.0 };
                for i in 0..self.ee {
                    self.q[i] += dq[i] * scale;
                }
            } else {
                self.q = q_try;
            }
        }

        // ---- FEM block: gripper contact only once grasping ----
        let g_now = tool(&self.robot, &self.q);
        let dt = self.grasp.fem.dt;
        self.grasp.grip_axis = Vector3::new(1.0, 0.0, 0.0);
        self.grasp.pad = if self.phase >= GRASP { self.radius * 1.7 } else { 0.0 };
        self.grasp.jaw_vel = (g_now - self.prev_grip) / (self.sub as f64 * dt);
        self.grasp.half_width = match self.phase {
            GRASP => {
                let a = (self.t as f64 / 45.0).min(1.0);
                self.open_hw * (1.0 - a) + self.grip_hw * a
            }
            LIFT | HOLD => self.grip_hw,
            _ => self.open_hw,
        };
        for k in 0..self.sub {
            let a = (k as f64 + 1.0) / self.sub as f64;
            self.grasp.center = self.prev_grip + a * (g_now - self.prev_grip);
            self.grasp.step();
        }
        self.prev_grip = g_now;
        self.t += 1;

        // ---- phase machine ----
        match self.phase {
            EXECUTE if arrived => {
                self.phase = DESCEND;
                self.t = 0;
                self.q_goal = self.solve_to(self.object_c);
            }
            DESCEND if arrived => {
                self.phase = GRASP;
                self.t = 0;
                self.q_goal = self.q.clone();
            }
            GRASP if self.t > 60 => {
                self.phase = LIFT;
                self.t = 0;
                self.q_goal = self.solve_to(self.object_c + Vector3::new(0.0, 0.0, self.lift_h));
            }
            LIFT if arrived => {
                self.phase = HOLD;
                self.t = 0;
            }
            _ => {}
        }
    }

    pub fn object_centroid(&self) -> Vector3<f64> {
        let n = self.grasp.fem.n_verts().max(1) as f64;
        self.grasp.fem.x.iter().sum::<Vector3<f64>>() / n
    }
    pub fn lift_off_rest(&self) -> f64 {
        self.object_centroid().z - self.object_c.z
    }
}

impl Default for PlannedGrasp {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Live bench wrapper.
// ---------------------------------------------------------------------------------------------
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PlannedGraspLab {
    sim: PlannedGrasp,
    edges: Vec<[usize; 2]>,
}

#[wasm_bindgen]
impl PlannedGraspLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PlannedGraspLab {
        let sim = PlannedGrasp::new();
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for a in 0..sim.grasp.fem.n_verts() {
            for b in (a + 1)..sim.grasp.fem.n_verts() {
                if (sim.grasp.fem.x[a] - sim.grasp.fem.x[b]).norm() < 1.05 * 0.06 {
                    set.insert((a, b));
                }
            }
        }
        let edges = set.into_iter().map(|(a, b)| [a, b]).collect();
        PlannedGraspLab { sim, edges }
    }

    pub fn tick(&mut self, n: usize) {
        for _ in 0..n {
            self.sim.step();
        }
    }
    /// 0 executing-plan · 1 descending · 2 grasping · 3 lifting · 4 holding.
    pub fn phase(&self) -> u8 {
        self.sim.phase
    }
    pub fn skeleton_xz(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..=self.sim.ee {
            let p = self.sim.robot.frame_pose(&self.sim.q, i).translation.vector;
            out.push(p.x);
            out.push(p.z);
        }
        let g = tool(&self.sim.robot, &self.sim.q);
        out.push(g.x);
        out.push(g.z);
        out
    }
    /// The planned tool route as flat `[x, z, …]` (the collision-free path around the wall).
    pub fn trail_xz(&self) -> Vec<f64> {
        self.sim.path.iter().flat_map(|q| { let t = tool(&self.sim.robot, q); [t.x, t.z] }).collect()
    }
    pub fn object_verts_xz(&self) -> Vec<f64> {
        self.sim.grasp.fem.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }
    pub fn object_edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }
    pub fn wall_xz(&self) -> Vec<f64> {
        vec![self.sim.wall_c.x, self.sim.wall_c.z, self.sim.wall_h.x, self.sim.wall_h.z]
    }
    pub fn gripper_xz(&self) -> Vec<f64> {
        let g = tool(&self.sim.robot, &self.sim.q);
        vec![g.x, g.z]
    }
    pub fn floor_z(&self) -> f64 {
        self.sim.object_c.z - self.sim.radius
    }
    pub fn planned(&self) -> bool {
        self.sim.planned
    }
    pub fn min_clearance(&self) -> f64 {
        if self.sim.min_clear.is_finite() {
            self.sim.min_clear
        } else {
            1.0
        }
    }
    pub fn naive_clearance(&self) -> f64 {
        self.sim.naive_min
    }
    pub fn lift(&self) -> f64 {
        self.sim.lift_off_rest()
    }
    pub fn reset(&mut self) {
        *self = PlannedGraspLab::new();
    }
}

impl Default for PlannedGraspLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Planned-grasp oracle.** The straight reach hits the wall; the arm plans a detour around both
    /// the wall and the block, follows it collision-free to a pre-grasp pose, then descends, grasps,
    /// and lifts the block clear of the floor.
    #[test]
    fn plans_around_the_wall_then_grasps() {
        let mut sim = PlannedGrasp::new();
        assert!(sim.planned, "planner fell back to the straight line");
        assert!(sim.naive_min < 0.0, "the wall should block the straight reach: {}", sim.naive_min);
        assert!(sim.path.len() > 2, "expected a multi-waypoint detour, got {}", sim.path.len());

        let mut grasped = false;
        for _ in 0..6000 {
            sim.step();
            if sim.phase >= GRASP {
                grasped = true;
            }
            if sim.phase == HOLD && sim.t > 300 {
                break;
            }
        }
        assert!(grasped, "never reached the grasp phase");
        let lift = sim.lift_off_rest();
        let x_off = (sim.object_centroid().x - sim.object_c.x).abs();
        eprintln!(
            "planned-grasp: naive {:.3} m, executed reach min-clearance {:.3} m, {} waypoints; lift {:.3} m, x-drift {:.3}",
            sim.naive_min, sim.min_clear, sim.path.len(), lift, x_off
        );
        assert!(sim.min_clear > 0.0, "the arm hit an obstacle on the planned reach: {:.3}", sim.min_clear);
        assert!(lift > 0.05 && lift < 0.4, "the block was not cleanly lifted: {lift:.3} m");
        assert!(x_off < 0.06, "the block slipped out of the gripper: {x_off:.3}");
    }
}
