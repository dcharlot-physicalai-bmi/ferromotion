//! **See → reach → grasp, closed on one arm.** The three legs of the phase change, integrated: a
//! fixed camera perceives a soft (FEM) sphere *only* through its raytraced image (an SDF proxy of the
//! object's live extent), the arm drives its joints — differential IK — to bring its gripper over the
//! perceived object, the parallel jaws close on the deformable sphere, and the arm lifts it away, held
//! by friction. Perception (`sensor_render`+`perceive`), kinematics (`Robot`+`solve_diffik`), and
//! deformable contact (`GraspFemSim`) in a single loop, pure Rust → WASM. A sphere is used so the
//! (uncontrolled) approach roll never matters — any closing axis grips it.

use ferromotion_control::{perceive, Perception};
use ferromotion_coupled::GraspFemSim;
use ferromotion_core::{from_urdf_str, solve_diffik, DepthCamera, DiffIkOptions, FrameTaskDef, Robot, Sdf, SdfScene};
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

// phases
pub const APPROACH: u8 = 0; // move the open gripper above the perceived sphere
pub const DESCEND: u8 = 1; // lower onto it
pub const GRASP: u8 = 2; // close the jaws
pub const LIFT: u8 = 3; // carry it up
pub const HOLD: u8 = 4; // hold aloft

pub struct SeeReachGrasp {
    pub robot: Robot,
    pub q: Vec<f64>,
    pub ee: usize,
    pub tip: Vector3<f64>, // gripper-centre offset in the wrist frame
    pub grasp: GraspFemSim,
    pub radius: f64,       // nominal object radius (for the SDF perception proxy)
    pub rest_center: Vector3<f64>,
    pub distractor_c: Vector3<f64>, // a decoy object beside the target (perception clutter, label 1)
    pub distractor_h: Vector3<f64>,
    pub cam: DepthCamera, // fixed eye-to-hand camera
    pub phase: u8,
    pub t: u32,
    pub perceived: Perception,
    pub center_est: Vector3<f64>, // perceived sphere centre (world)
    pub grasp_center: Vector3<f64>,
    q_goal: Vec<f64>, // the converged IK solution for the current phase; the arm slews toward it
    prev_grip: Vector3<f64>,
    pub open_hw: f64,
    pub grip_hw: f64,
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

impl SeeReachGrasp {
    pub fn new() -> Self {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let ee = robot.dof();
        let radius = 0.09; // nominal half-size: the perception proxy half-extent + gripper sizing
        let center = Vector3::new(0.34, 0.0, 0.42);
        let floor = center.z - radius; // the block rests on this floor
        // A soft cube (clean box-grid tets — stable under squeeze, unlike a coarse implicit mesh),
        // translated so its centroid sits at `center`.
        let mut fem = ferromotion_fem::FemSim::box_grid(3, 3, 3, 0.06, 0.02, 1.0e4, 6.0e3, 2.0e-4);
        fem.damping_rate = 102.040_816_326_530_6; // old per-step 0.02 at dt = 2.0e-4
        let n = fem.n_verts() as f64;
        let c0: Vector3<f64> = fem.x.iter().sum::<Vector3<f64>>() / n;
        let off = center - c0;
        for v in fem.x.iter_mut() {
            *v += off;
        }
        let mut grasp = GraspFemSim::new(fem, center, radius + 0.03, 5.0e4, 0.9); // jaws start open
        grasp.floor = Some(floor);
        grasp.pad = radius * 1.7; // finite jaws: only grip near the gripper, so approach doesn't clip the block
        // fixed camera, off to the side, looking at the object
        let cam = DepthCamera { pose: look_at(Vector3::new(0.34, -1.1, 0.5), center), fx: 90.0, fy: 90.0, cx: 47.5, cy: 35.5, width: 96, height: 72, far: 6.0 };
        let q = vec![0.0, 0.55, -0.7, 0.35, 0.0, 0.2];
        let tip = Vector3::new(0.0, 0.0, 0.08);
        let prev_grip = (robot.frame_pose(&q, ee) * nalgebra::Point3::from(tip)).coords;
        let mut s = SeeReachGrasp {
            robot,
            q: q.clone(),
            ee,
            tip,
            grasp,
            radius,
            rest_center: center,
            // a larger decoy just beyond the target (off the reach path, still in the camera's view):
            // a naive "grab the biggest blob" would take this, but segmentation targets label 0.
            distractor_c: Vector3::new(0.55, 0.03, 0.42),
            distractor_h: Vector3::new(0.11, 0.11, 0.13),
            cam,
            phase: APPROACH,
            t: 0,
            perceived: nothing(),
            center_est: center,
            grasp_center: center,
            q_goal: q,
            prev_grip,
            open_hw: radius + 0.03,
            grip_hw: radius - 0.016,
            sub: 30,
        };
        s.perceive_object();
        s.q_goal = s.solve_to(s.target());
        s
    }

    /// Full differential-IK solve bringing the gripper point to `target` (a converged goal the arm
    /// then slews toward, so its motion stays smooth and never traps against a joint limit).
    fn solve_to(&self, target: Vector3<f64>) -> Vec<f64> {
        let tasks = [FrameTaskDef::new(self.ee, self.tip, target, 2.0, 1.0)];
        let opts = DiffIkOptions { dt: 0.05, vmax: 3.0, max_iters: 200, use_limits: true, ..Default::default() };
        solve_diffik(&self.robot, &tasks, &self.q, &opts).q
    }

    /// Perceive the sphere through the fixed camera using an SDF proxy of its current extent, and
    /// estimate the world-space centre from the raytraced image alone.
    fn perceive_object(&mut self) {
        // an axis-aligned box proxy of the object's current (deformed) extent for the camera to see
        let mut lo = Vector3::repeat(f64::INFINITY);
        let mut hi = Vector3::repeat(f64::NEG_INFINITY);
        for p in &self.grasp.fem.x {
            lo = lo.inf(p);
            hi = hi.sup(p);
        }
        // prim 0 = the target (what we grasp); prim 1 = a decoy the camera also sees. `perceive`
        // segments label 0 out of the raytraced image, so the arm is never fooled by the clutter.
        let proxy = SdfScene {
            prims: vec![
                Sdf::Box { center: 0.5 * (lo + hi), half: 0.5 * (hi - lo) },
                Sdf::Box { center: self.distractor_c, half: self.distractor_h },
            ],
        };
        let p = perceive(&self.cam, &proxy, 0);
        if p.seen {
            let cam_pos = self.cam.pose.translation.vector;
            let dir = (p.point_world - cam_pos).normalize();
            self.center_est = cam_pos + (p.z + self.radius) * dir; // front surface + one radius ≈ centre
        }
        self.perceived = p;
    }

    fn gripper_pos(&self) -> Vector3<f64> {
        (self.robot.frame_pose(&self.q, self.ee) * nalgebra::Point3::from(self.tip)).coords
    }

    /// The diffik target for the current phase.
    fn target(&self) -> Vector3<f64> {
        match self.phase {
            APPROACH => self.center_est + Vector3::new(0.0, 0.0, 0.22),
            DESCEND => self.center_est,
            GRASP => self.grasp_center,
            _ => self.grasp_center + Vector3::new(0.0, 0.0, 0.20), // LIFT / HOLD
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        self.perceive_object();
        // slew the joints toward the phase's converged IK goal (smooth, trap-free)
        let max_step = 0.03; // rad per frame
        let mut dq = vec![0.0; self.q.len()];
        let mut arrived = true;
        for i in 0..self.q.len() {
            dq[i] = (self.q_goal[i] - self.q[i]).clamp(-max_step, max_step);
            if (self.q_goal[i] - self.q[i]).abs() > 1e-3 {
                arrived = false;
            }
        }
        // during contact phases, cap the gripper's Cartesian speed so the FEM sees a gentle jaw
        // (the arm moves at most `cap` metres per frame ≈ `cap`/(sub·dt) m/s against the soft body)
        let contact_phase = self.phase >= GRASP;
        let q_try: Vec<f64> = (0..self.q.len()).map(|i| self.q[i] + dq[i]).collect();
        if contact_phase {
            let g_try = (self.robot.frame_pose(&q_try, self.ee) * nalgebra::Point3::from(self.tip)).coords;
            let disp = (g_try - self.prev_grip).norm();
            let cap = 0.0016;
            let scale = if disp > cap { cap / disp } else { 1.0 };
            for i in 0..self.q.len() {
                self.q[i] += dq[i] * scale;
            }
        } else {
            self.q = q_try;
        }

        // drive the (world-x) gripper from the arm; interpolate its motion across the FEM sub-steps
        let g_now = self.gripper_pos();
        let dt = self.grasp.fem.dt;
        self.grasp.grip_axis = Vector3::new(1.0, 0.0, 0.0);
        self.grasp.center = self.prev_grip;
        self.grasp.jaw_vel = (g_now - self.prev_grip) / (self.sub as f64 * dt);
        // jaw opening by phase
        let hw = match self.phase {
            GRASP | LIFT | HOLD => self.grip_hw,
            _ => self.open_hw,
        };
        // close smoothly during GRASP
        if self.phase == GRASP {
            let a = (self.t as f64 / 40.0).min(1.0);
            self.grasp.half_width = self.open_hw * (1.0 - a) + self.grip_hw * a;
        } else {
            self.grasp.half_width = hw;
        }
        for _ in 0..self.sub {
            self.grasp.step();
        }
        self.prev_grip = g_now;
        self.t += 1;

        // advance the phase machine (each transition recomputes the IK goal for the next phase)
        match self.phase {
            APPROACH if arrived => {
                self.phase = DESCEND;
                self.t = 0;
                self.q_goal = self.solve_to(self.target());
            }
            DESCEND if arrived => {
                self.grasp_center = self.gripper_pos();
                self.phase = GRASP;
                self.t = 0;
                self.q_goal = self.q.clone(); // hold position while closing
            }
            GRASP if self.t > 70 => {
                self.phase = LIFT;
                self.t = 0;
                self.q_goal = self.solve_to(self.grasp_center + Vector3::new(0.0, 0.0, 0.20));
            }
            LIFT if arrived => {
                self.phase = HOLD;
                self.t = 0;
            }
            _ => {}
        }
    }

    pub fn object_centroid(&self) -> Vector3<f64> {
        self.grasp.object_centroid()
    }
    pub fn lift_off_rest(&self) -> f64 {
        self.grasp.object_centroid().z - self.rest_center.z
    }
}

impl Default for SeeReachGrasp {
    fn default() -> Self {
        Self::new()
    }
}

fn nothing() -> Perception {
    Perception { seen: false, n_pixels: 0, u: 0.0, v: 0.0, x: 0.0, y: 0.0, z: 0.0, point_world: Vector3::zeros(), center_err: f64::INFINITY }
}

// ---------------------------------------------------------------------------------------------
// Live bench wrapper.
// ---------------------------------------------------------------------------------------------
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SeeReachGraspLab {
    sim: SeeReachGrasp,
    edges: Vec<[usize; 2]>,
}

#[wasm_bindgen]
impl SeeReachGraspLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SeeReachGraspLab {
        let sim = SeeReachGrasp::new();
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
        SeeReachGraspLab { sim, edges }
    }

    /// Advance `n` control frames (each is a joint slew + the FEM sub-steps).
    pub fn tick(&mut self, n: usize) {
        for _ in 0..n {
            self.sim.step();
        }
    }

    /// 0 approach · 1 descend · 2 grasp · 3 lift · 4 hold.
    pub fn phase(&self) -> u8 {
        self.sim.phase
    }

    /// Arm skeleton as flat `[x, z, …]` (side view): base → joint frames → gripper.
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

    pub fn verts_xz(&self) -> Vec<f64> {
        self.sim.grasp.fem.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }
    pub fn edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }
    /// `[left_x, right_x, gripper_z, pad_half_z]` for drawing the two jaw bars.
    pub fn jaws_xz(&self) -> Vec<f64> {
        let g = self.sim.gripper_pos();
        vec![g.x - self.sim.grasp.half_width, g.x + self.sim.grasp.half_width, g.z, self.sim.grasp.pad]
    }
    pub fn floor_z(&self) -> f64 {
        self.sim.rest_center.z - self.sim.radius
    }
    pub fn object_xz(&self) -> Vec<f64> {
        let c = self.sim.object_centroid();
        vec![c.x, c.z]
    }
    /// The decoy object `[center_x, center_z, half_x, half_z]` (perception clutter the arm ignores).
    pub fn decoy_xz(&self) -> Vec<f64> {
        vec![self.sim.distractor_c.x, self.sim.distractor_c.z, self.sim.distractor_h.x, self.sim.distractor_h.z]
    }
    /// Perceived-vs-true centre error — the live perception receipt (metres).
    pub fn perc_err(&self) -> f64 {
        (self.sim.center_est - self.sim.object_centroid()).norm()
    }
    /// Height the block has been lifted off its rest position (metres).
    pub fn lift(&self) -> f64 {
        self.sim.lift_off_rest()
    }
    pub fn reset(&mut self) {
        *self = SeeReachGraspLab::new();
    }
}

impl Default for SeeReachGraspLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Integrated oracle.** One arm: it perceives the soft sphere (from the raytraced image alone),
    /// reaches its gripper over it, closes on it, and lifts it clear of the floor — retained. Ground
    /// truth only grades: the perceived centre matches the true one, and the object rises with the arm.
    #[test]
    fn one_arm_sees_reaches_and_lifts_the_soft_sphere() {
        let mut sim = SeeReachGrasp::new();
        let true_c0 = sim.rest_center;

        // perception fidelity (before the object is disturbed): estimate vs ground truth
        sim.perceive_object();
        let perc_err = (sim.center_est - sim.object_centroid()).norm();
        let dist_to_decoy = (sim.center_est - sim.distractor_c).norm();
        assert!(sim.perceived.seen, "the camera did not see the object");
        eprintln!("integrated: perceived centre err {perc_err:.3} m; distance from the decoy {dist_to_decoy:.3} m");
        assert!(perc_err < 0.04, "perceived centre off by {perc_err:.3} m");
        // discrimination: it locked onto the target, not the larger decoy beside it
        assert!(dist_to_decoy > 0.15, "perception was fooled by the decoy: only {dist_to_decoy:.3} m from it");

        let mut reached_grasp = false;
        for _ in 0..1500 {
            sim.step();
            if sim.phase >= GRASP {
                reached_grasp = true;
            }
            if sim.phase == HOLD && sim.t > 400 {
                break;
            }
        }
        assert!(reached_grasp, "the arm never reached the grasp phase");

        let lift = sim.lift_off_rest();
        let held_x = (sim.object_centroid().x - sim.rest_center.x).abs();
        eprintln!("integrated: phase {}, object lift {lift:.3} m off rest, x-drift {held_x:.3}, true c0 {:?}", sim.phase, true_c0);
        assert!(lift > 0.06, "the arm did not lift the sphere clear of the floor: {lift:.3} m");
        assert!(held_x < 0.06, "the sphere was knocked out of the gripper: x-drift {held_x:.3}");
    }
}
