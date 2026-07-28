//! **Dynamic-object grasping** — catching a *moving* block. A soft (FEM) block glides across a
//! frictionless floor at constant speed; a fixed camera perceives it; the arm waits at an interception
//! point, and as the block arrives it descends, closes on it *while matching its velocity* (so the
//! jaws meet the block at near-zero relative speed — no slam), and lifts it away, carrying it along.
//! Velocity matching is the trick: track the perceived block position so the gripper moves *with* the
//! block through contact. Composes perception + kinematics + the deformable grasp. Pure Rust → WASM.

use ferromotion_core::{from_urdf_str, solve_diffik, DepthCamera, DiffIkOptions, FrameTaskDef, Robot, Sdf, SdfScene};
use ferromotion_coupled::GraspFemSim;
use nalgebra::{Isometry3, Vector3};

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

const WAIT: u8 = 0;
const DESCEND: u8 = 1;
const CLOSE: u8 = 2;
const LIFT: u8 = 3;
const HOLD: u8 = 4;

pub struct DynamicGrasp {
    pub robot: Robot,
    pub q: Vec<f64>,
    pub ee: usize,
    pub tip: Vector3<f64>,
    pub grasp: GraspFemSim,
    pub radius: f64,
    pub v_drift: f64,
    pub rest_z: f64,       // the block's floating height (centre) as it glides
    pub y0: f64,
    pub intercept_x: f64,  // where the gripper waits for the block
    pub cam: DepthCamera,
    pub est_x: f64,        // perceived block x (world)
    pub phase: u8,
    pub t: u32,
    q_goal: Vec<f64>,
    prev_grip: Vector3<f64>,
    pub open_hw: f64,
    pub grip_hw: f64,
    pub standoff: f64,
    pub lift_h: f64,
    pub sub: usize,
}

fn look_at(eye: Vector3<f64>, target: Vector3<f64>) -> Isometry3<f64> {
    use nalgebra::{Matrix3, Translation3, UnitQuaternion};
    let up = Vector3::new(0.0, 0.0, 1.0);
    let fwd = (target - eye).normalize();
    let right = fwd.cross(&up).normalize();
    let down = fwd.cross(&right);
    let r = Matrix3::from_columns(&[right, down, fwd]);
    Isometry3::from_parts(Translation3::from(eye), UnitQuaternion::from_matrix(&r))
}

impl DynamicGrasp {
    pub fn new() -> Self {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let ee = robot.dof();
        let radius = 0.09;
        let y0 = 0.0;
        let start_x = 0.12;
        let rest_z = 0.42;
        let v_drift = 0.15;
        let intercept_x = 0.40;

        // a soft cube, translated to the start, given a constant horizontal glide
        let mut fem = ferromotion_fem::FemSim::box_grid(3, 3, 3, 0.06, 0.02, 1.0e4, 6.0e3, 2.0e-4);
        fem.damping = 0.04;
        let n = fem.n_verts() as f64;
        let c0: Vector3<f64> = fem.x.iter().sum::<Vector3<f64>>() / n;
        let off = Vector3::new(start_x, y0, rest_z) - c0;
        for v in fem.x.iter_mut() {
            *v += off;
        }
        for vv in fem.v.iter_mut() {
            vv.x = v_drift; // constant glide along +x (frictionless floor keeps it constant)
        }
        let mut grasp = GraspFemSim::new(fem, Vector3::new(start_x, y0, rest_z), radius + 0.03, 5.0e4, 0.9);
        grasp.floor = Some(rest_z - radius);
        grasp.pad = radius * 1.7;

        let cam = DepthCamera { pose: look_at(Vector3::new(0.34, -1.1, 0.5), Vector3::new(0.34, 0.0, rest_z)), fx: 90.0, fy: 90.0, cx: 47.5, cy: 35.5, width: 96, height: 72, far: 6.0 };

        let q = vec![0.0, 0.55, -0.7, 0.35, 0.0, 0.2];
        let tip = Vector3::new(0.0, 0.0, 0.08);
        let prev_grip = (robot.frame_pose(&q, ee) * nalgebra::Point3::from(tip)).coords;
        let standoff = 0.22;
        let mut s = DynamicGrasp {
            robot,
            q: q.clone(),
            ee,
            tip,
            grasp,
            radius,
            v_drift,
            rest_z,
            y0,
            intercept_x,
            cam,
            est_x: start_x,
            phase: WAIT,
            t: 0,
            q_goal: q,
            prev_grip,
            open_hw: radius + 0.03,
            grip_hw: radius - 0.016,
            standoff,
            lift_h: 0.20,
            sub: 30,
        };
        // pre-position the gripper above the interception point
        s.q_goal = s.solve_to(Vector3::new(intercept_x, y0, rest_z + standoff), 200);
        s
    }

    fn solve_to(&self, target: Vector3<f64>, iters: usize) -> Vec<f64> {
        let tasks = [FrameTaskDef::new(self.ee, self.tip, target, 2.0, 1.0)];
        let opts = DiffIkOptions { dt: 0.05, vmax: 3.0, max_iters: iters, use_limits: true, ..Default::default() };
        solve_diffik(&self.robot, &tasks, &self.q, &opts).q
    }

    fn perceive_block(&mut self) {
        let mut lo = Vector3::repeat(f64::INFINITY);
        let mut hi = Vector3::repeat(f64::NEG_INFINITY);
        for p in &self.grasp.fem.x {
            lo = lo.inf(p);
            hi = hi.sup(p);
        }
        let proxy = SdfScene { prims: vec![Sdf::Box { center: 0.5 * (lo + hi), half: 0.5 * (hi - lo) }] };
        let p = ferromotion_control::perceive(&self.cam, &proxy, 0);
        if p.seen {
            let cam_pos = self.cam.pose.translation.vector;
            let dir = (p.point_world - cam_pos).normalize();
            self.est_x = (cam_pos + (p.z + self.radius) * dir).x;
        }
    }

    fn gripper_pos(&self) -> Vector3<f64> {
        (self.robot.frame_pose(&self.q, self.ee) * nalgebra::Point3::from(self.tip)).coords
    }

    /// The current tracking target: hover above the interception point until the block arrives, then
    /// follow the perceived block position (so the gripper moves *with* it through the grasp).
    fn target(&self) -> Vector3<f64> {
        match self.phase {
            WAIT => Vector3::new(self.intercept_x, self.y0, self.rest_z + self.standoff),
            DESCEND | CLOSE => Vector3::new(self.est_x, self.y0, self.rest_z),
            _ => Vector3::new(self.est_x, self.y0, self.rest_z + self.lift_h),
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        self.perceive_block();
        // WAIT holds a fixed goal (set at construction) — no need to re-solve; only re-solve to
        // track the moving block once the catch begins
        if self.phase != WAIT {
            self.q_goal = self.solve_to(self.target(), 40);
        }

        let max_step = 0.03;
        let mut dq = vec![0.0; self.q.len()];
        let mut arrived = true;
        for i in 0..self.q.len() {
            dq[i] = (self.q_goal[i] - self.q[i]).clamp(-max_step, max_step);
            if (self.q_goal[i] - self.q[i]).abs() > 2e-3 {
                arrived = false;
            }
        }
        // cap the gripper's Cartesian speed once in contact (CLOSE/LIFT), but allow the x-tracking
        // velocity (matching the block) as a floor so the gripper can still keep pace with the glide
        let contact = self.phase == CLOSE || self.phase == LIFT;
        let q_try: Vec<f64> = (0..self.q.len()).map(|i| self.q[i] + dq[i]).collect();
        if contact {
            let g_try = (self.robot.frame_pose(&q_try, self.ee) * nalgebra::Point3::from(self.tip)).coords;
            let disp = (g_try - self.prev_grip).norm();
            let cap = 0.0016 + self.v_drift * self.sub as f64 * self.grasp.fem.dt; // allow the glide-matching motion
            let scale = if disp > cap { cap / disp } else { 1.0 };
            for i in 0..self.q.len() {
                self.q[i] += dq[i] * scale;
            }
        } else {
            self.q = q_try;
        }

        // drive the world-x gripper from the arm, interpolating across the FEM sub-steps
        let g_now = self.gripper_pos();
        let dt = self.grasp.fem.dt;
        self.grasp.grip_axis = Vector3::new(1.0, 0.0, 0.0);
        // enable jaw contact only once we are grasping — otherwise the gripper sweeping to the
        // interception point would drag its open jaws through the gliding block
        self.grasp.pad = if self.phase >= CLOSE { self.radius * 1.7 } else { 0.0 };
        self.grasp.center = self.prev_grip;
        self.grasp.jaw_vel = (g_now - self.prev_grip) / (self.sub as f64 * dt);
        self.grasp.half_width = match self.phase {
            CLOSE => {
                let a = (self.t as f64 / 45.0).min(1.0);
                self.open_hw * (1.0 - a) + self.grip_hw * a
            }
            LIFT | HOLD => self.grip_hw,
            _ => self.open_hw,
        };
        for _ in 0..self.sub {
            self.grasp.step();
            // maintain the constant glide until grasped (the FEM damping would otherwise kill it);
            // once the jaws close, the contact physics carries the block instead
            if self.phase < CLOSE {
                for vv in self.grasp.fem.v.iter_mut() {
                    vv.x = self.v_drift;
                }
            }
        }
        self.prev_grip = g_now;
        self.t += 1;

        // phase machine
        match self.phase {
            WAIT if self.est_x >= self.intercept_x - 0.01 => {
                self.phase = DESCEND;
                self.t = 0;
            }
            DESCEND if arrived => {
                self.phase = CLOSE;
                self.t = 0;
            }
            CLOSE if self.t > 60 => {
                self.phase = LIFT;
                self.t = 0;
            }
            LIFT if arrived => {
                self.phase = HOLD;
                self.t = 0;
            }
            _ => {}
        }
    }

    pub fn block_centroid(&self) -> Vector3<f64> {
        let n = self.grasp.fem.n_verts().max(1) as f64;
        self.grasp.fem.x.iter().sum::<Vector3<f64>>() / n
    }
    pub fn lift_off_rest(&self) -> f64 {
        self.block_centroid().z - self.rest_z
    }
}

impl Default for DynamicGrasp {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Live bench wrapper.
// ---------------------------------------------------------------------------------------------
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DynamicGraspLab {
    sim: DynamicGrasp,
    edges: Vec<[usize; 2]>,
}

#[wasm_bindgen]
impl DynamicGraspLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> DynamicGraspLab {
        let sim = DynamicGrasp::new();
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
        DynamicGraspLab { sim, edges }
    }

    pub fn tick(&mut self, n: usize) {
        for _ in 0..n {
            self.sim.step();
        }
    }
    /// 0 waiting · 1 descending · 2 closing · 3 lifting · 4 holding.
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
        let g = self.sim.gripper_pos();
        out.push(g.x);
        out.push(g.z);
        out
    }
    pub fn block_verts_xz(&self) -> Vec<f64> {
        self.sim.grasp.fem.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }
    pub fn block_edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }
    pub fn jaws_xz(&self) -> Vec<f64> {
        let g = self.sim.gripper_pos();
        vec![g.x - self.sim.grasp.half_width, g.x + self.sim.grasp.half_width, g.z, self.sim.grasp.pad.max(self.sim.radius * 1.7)]
    }
    pub fn floor_z(&self) -> f64 {
        self.sim.rest_z - self.sim.radius
    }
    pub fn block_xz(&self) -> Vec<f64> {
        let c = self.sim.block_centroid();
        vec![c.x, c.z]
    }
    pub fn lift(&self) -> f64 {
        self.sim.lift_off_rest()
    }
    pub fn v_drift(&self) -> f64 {
        self.sim.v_drift
    }
    pub fn reset(&mut self) {
        *self = DynamicGraspLab::new();
    }
}

impl Default for DynamicGraspLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Dynamic-grasp oracle.** The block glides in; the arm intercepts, grasps it while matching its
    /// motion, and lifts it clear of the floor — retained (held near the gripper, lifted).
    #[test]
    fn arm_catches_and_lifts_the_gliding_block() {
        let mut sim = DynamicGrasp::new();
        assert!(sim.grasp.fem.v.iter().all(|v| (v.x - sim.v_drift).abs() < 1e-9), "block should start gliding");

        let mut grasped = false;
        for _ in 0..3000 {
            sim.step();
            if sim.phase >= CLOSE {
                grasped = true;
            }
            if sim.phase == HOLD && sim.t > 400 {
                break;
            }
        }
        assert!(grasped, "the arm never reached the grasp phase");
        let lift = sim.lift_off_rest();
        let gx = sim.gripper_pos().x;
        let x_off = (sim.block_centroid().x - gx).abs();
        eprintln!("dynamic-grasp: phase {}, lift {lift:.3} m, block↔gripper x-offset {x_off:.3} m", sim.phase);
        assert!(lift > 0.05 && lift < 0.4, "the arm did not cleanly lift the gliding block: {lift:.3} m");
        assert!(x_off < 0.06, "the block slipped out of the gripper: x-offset {x_off:.3}");
    }
}
