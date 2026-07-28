//! **Bimanual manipulation** — two arms lifting one soft object together. A deformable beam is too
//! long for a single grasp; the two arms each take an end (jaws closing across the beam, finite pads
//! keeping each gripper to its own region), then lift in coordination so the beam rises held at both
//! ends while its ungripped middle sags between them. Two `Robot`s + two grippers acting on one shared
//! `FemSim`, perception-free (the grip points are commanded). Pure Rust → WASM.

use ferromotion_core::{from_urdf_str, solve_diffik, DiffIkOptions, FrameTaskDef, Robot};
use ferromotion_fem::FemSim;
use nalgebra::Vector3;

fn arm_urdf(base_x: f64) -> String {
    format!(
        r#"<robot name="a"><link name="world"/><link name="base"/>
  <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
  <joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="{base_x} 0 0.05" rpy="0 0 0"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
  <joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#
    )
}

const REACH: u8 = 0;
const DESCEND: u8 = 1;
const GRIP: u8 = 2;
const LIFT: u8 = 3;
const HOLD: u8 = 4;

struct Arm {
    robot: Robot,
    q: Vec<f64>,
    q_goal: Vec<f64>,
    grip_x: f64,   // where along the beam this arm grips
    prev_grip: Vector3<f64>,
}

pub struct Bimanual {
    pub fem: FemSim,
    tip: Vector3<f64>,
    ee: usize,
    left: Arm,
    right: Arm,
    pub rest_z: f64,
    pub floor_z: f64,
    pub phase: u8,
    pub t: u32,
    pub open_hw: f64,
    pub grip_hw: f64,
    pub k_contact: f64,
    pub mu: f64,
    pub pad: f64,
    pub standoff: f64,
    pub lift_h: f64,
    pub sub: usize,
}

impl Bimanual {
    pub fn new() -> Self {
        // a soft beam, wide in x, translated so its centroid sits at the work height
        let mut fem = FemSim::box_grid(6, 2, 2, 0.06, 0.02, 1.0e4, 6.0e3, 2.0e-4);
        fem.damping = 0.04;
        let center = Vector3::new(0.45, 0.0, 0.42);
        let n = fem.n_verts() as f64;
        let c0: Vector3<f64> = fem.x.iter().sum::<Vector3<f64>>() / n;
        let off = center - c0;
        for v in fem.x.iter_mut() {
            *v += off;
        }
        fem.gravity = Vector3::zeros(); // this integrator owns gravity + floor
        let half_x = 0.5 * (fem.x.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max) - fem.x.iter().map(|p| p.x).fold(f64::INFINITY, f64::min));
        let half_y = 0.5 * (fem.x.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max) - fem.x.iter().map(|p| p.y).fold(f64::INFINITY, f64::min));
        let half_z = 0.5 * (fem.x.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max) - fem.x.iter().map(|p| p.z).fold(f64::INFINITY, f64::min));
        let rest_z = center.z;
        let floor_z = center.z - half_z;

        let ee = 6;
        let tip = Vector3::new(0.0, 0.0, 0.08);
        // two arms flanking the beam; each grips a point a third of the way in from its end
        let robot_l = from_urdf_str(&arm_urdf(0.0), "world", "tool").unwrap();
        let robot_r = from_urdf_str(&arm_urdf(0.90), "world", "tool").unwrap();
        let grip_lx = center.x - 0.55 * half_x;
        let grip_rx = center.x + 0.55 * half_x;
        let q_l = vec![0.0, 0.6, -0.7, 0.35, 0.0, 0.2];
        let q_r = vec![0.0, 0.6, -0.7, 0.35, 0.0, 0.2];
        let pl = (robot_l.frame_pose(&q_l, ee) * nalgebra::Point3::from(tip)).coords;
        let pr = (robot_r.frame_pose(&q_r, ee) * nalgebra::Point3::from(tip)).coords;
        let left = Arm { robot: robot_l, q: q_l.clone(), q_goal: q_l, grip_x: grip_lx, prev_grip: pl };
        let right = Arm { robot: robot_r, q: q_r.clone(), q_goal: q_r, grip_x: grip_rx, prev_grip: pr };

        let mut s = Bimanual {
            fem,
            tip,
            ee,
            left,
            right,
            rest_z,
            floor_z,
            phase: REACH,
            t: 0,
            open_hw: half_y + 0.03,
            grip_hw: half_y - 0.014,
            k_contact: 5.0e4,
            mu: 0.9,
            pad: 0.11,
            standoff: 0.22,
            lift_h: 0.18,
            sub: 30,
        };
        s.retarget();
        s
    }

    fn tool(robot: &Robot, q: &[f64], ee: usize, tip: Vector3<f64>) -> Vector3<f64> {
        (robot.frame_pose(q, ee) * nalgebra::Point3::from(tip)).coords
    }

    fn solve(robot: &Robot, q: &[f64], ee: usize, tip: Vector3<f64>, target: Vector3<f64>, iters: usize) -> Vec<f64> {
        let tasks = [FrameTaskDef::new(ee, tip, target, 2.0, 1.0)];
        let opts = DiffIkOptions { dt: 0.05, vmax: 3.0, max_iters: iters, use_limits: true, ..Default::default() };
        solve_diffik(robot, &tasks, q, &opts).q
    }

    /// The grip target for an arm at the current phase.
    fn target_for(&self, grip_x: f64) -> Vector3<f64> {
        match self.phase {
            REACH => Vector3::new(grip_x, 0.0, self.rest_z + self.standoff),
            DESCEND | GRIP => Vector3::new(grip_x, 0.0, self.rest_z),
            _ => Vector3::new(grip_x, 0.0, self.rest_z + self.lift_h),
        }
    }

    /// Recompute both arms' IK goals for the current phase.
    fn retarget(&mut self) {
        let (ee, tip) = (self.ee, self.tip);
        let tl = self.target_for(self.left.grip_x);
        let tr = self.target_for(self.right.grip_x);
        self.left.q_goal = Self::solve(&self.left.robot, &self.left.q, ee, tip, tl, 200);
        self.right.q_goal = Self::solve(&self.right.robot, &self.right.q, ee, tip, tr, 200);
    }

    /// Add one gripper's jaw contact (jaws close along world-y at `gc`, limited to the beam verts
    /// within `pad` of the gripper in the x–z plane) to the force accumulator.
    #[allow(clippy::needless_range_loop)]
    fn apply_gripper(&self, gc: Vector3<f64>, gv: Vector3<f64>, hw: f64, f: &mut [Vector3<f64>]) {
        let g = Vector3::new(0.0, 1.0, 0.0);
        let gamma = 0.5 * (self.k_contact * self.fem.mass).sqrt();
        for i in 0..self.fem.n_verts() {
            let p = self.fem.x[i];
            let s = (p - gc).dot(&g);
            let perp = ((p.x - gc.x).powi(2) + (p.z - gc.z).powi(2)).sqrt();
            if perp > self.pad {
                continue;
            }
            let contact = if s > hw {
                Some((-g, s - hw))
            } else if s < -hw {
                Some((g, -hw - s))
            } else {
                None
            };
            if let Some((n, pen)) = contact {
                let vrel = self.fem.v[i] - gv;
                let vn = vrel.dot(&n);
                let n_mag = (self.k_contact * pen - gamma * vn.min(0.0)).max(0.0);
                let f_n = n_mag * n;
                let v_t = vrel - vn * n;
                let sspeed = v_t.norm();
                let f_t = if sspeed > 1e-12 { -(self.mu * n_mag) * (v_t / (sspeed + 1e-3)) } else { Vector3::zeros() };
                f[i] += f_n + f_t;
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn step(&mut self) {
        let (ee, tip) = (self.ee, self.tip);
        // Each phase has FIXED grip targets (no moving object to track), so the IK goals are solved
        // once per phase in `retarget()`; here we just slew the arms toward them (cheap — no per-frame
        // IK, which matters with two arms).
        let contact = self.phase == GRIP || self.phase == LIFT;
        let arrived_l = Self::slew(&mut self.left, contact, ee, tip);
        let arrived_r = Self::slew(&mut self.right, contact, ee, tip);

        // gripper endpoints + interpolated motion across the FEM sub-steps
        let gl_new = Self::tool(&self.left.robot, &self.left.q, ee, tip);
        let gr_new = Self::tool(&self.right.robot, &self.right.q, ee, tip);
        let dt = self.fem.dt;
        let vl = (gl_new - self.left.prev_grip) / (self.sub as f64 * dt);
        let vr = (gr_new - self.right.prev_grip) / (self.sub as f64 * dt);
        let hw = if self.phase == GRIP {
            let a = (self.t as f64 / 45.0).min(1.0);
            self.open_hw * (1.0 - a) + self.grip_hw * a
        } else if self.phase == LIFT || self.phase == HOLD {
            self.grip_hw
        } else {
            self.open_hw
        };

        let inv_m = 1.0 / self.fem.mass;
        let fdamp = self.fem.damping;
        let grav = Vector3::new(0.0, 0.0, -9.81);
        let gamma = 0.5 * (self.k_contact * self.fem.mass).sqrt();
        for k in 0..self.sub {
            let a = (k as f64 + 1.0) / self.sub as f64;
            let cl = self.left.prev_grip + a * (gl_new - self.left.prev_grip);
            let cr = self.right.prev_grip + a * (gr_new - self.right.prev_grip);
            let mut f = self.fem.forces();
            for fi in f.iter_mut() {
                *fi += self.fem.mass * grav;
            }
            // floor
            for i in 0..self.fem.n_verts() {
                let pen = self.floor_z - self.fem.x[i].z;
                if pen > 0.0 {
                    let vnf = self.fem.v[i].z.min(0.0);
                    f[i].z += self.k_contact * pen - gamma * vnf;
                }
            }
            self.apply_gripper(cl, vl, hw, &mut f);
            self.apply_gripper(cr, vr, hw, &mut f);
            for i in 0..self.fem.n_verts() {
                if self.fem.pinned[i] {
                    continue;
                }
                self.fem.v[i] = (self.fem.v[i] + dt * f[i] * inv_m) * (1.0 - fdamp);
                self.fem.x[i] += dt * self.fem.v[i];
            }
        }
        self.left.prev_grip = gl_new;
        self.right.prev_grip = gr_new;
        self.t += 1;

        // phase machine — both arms move in lock-step
        let both = arrived_l && arrived_r;
        match self.phase {
            REACH if both => {
                self.phase = DESCEND;
                self.t = 0;
                self.retarget();
            }
            DESCEND if both => {
                self.phase = GRIP;
                self.t = 0;
                self.left.q_goal = self.left.q.clone();
                self.right.q_goal = self.right.q.clone();
            }
            GRIP if self.t > 70 => {
                self.phase = LIFT;
                self.t = 0;
                self.retarget();
            }
            LIFT if both => {
                self.phase = HOLD;
                self.t = 0;
            }
            _ => {}
        }
    }

    /// Slew one arm one frame toward its goal; returns whether it arrived. Caps Cartesian speed while
    /// in contact so the FEM sees a gentle gripper.
    #[allow(clippy::needless_range_loop)]
    fn slew(arm: &mut Arm, contact: bool, ee: usize, tip: Vector3<f64>) -> bool {
        // move briskly while approaching (no contact), gently once gripping
        let max_step = if contact { 0.03 } else { 0.06 };
        let mut dq = vec![0.0; arm.q.len()];
        let mut arrived = true;
        for i in 0..arm.q.len() {
            dq[i] = (arm.q_goal[i] - arm.q[i]).clamp(-max_step, max_step);
            if (arm.q_goal[i] - arm.q[i]).abs() > 2e-3 {
                arrived = false;
            }
        }
        let q_try: Vec<f64> = (0..arm.q.len()).map(|i| arm.q[i] + dq[i]).collect();
        if contact {
            let g_try = (arm.robot.frame_pose(&q_try, ee) * nalgebra::Point3::from(tip)).coords;
            let disp = (g_try - arm.prev_grip).norm();
            let cap = 0.0016;
            let scale = if disp > cap { cap / disp } else { 1.0 };
            for i in 0..arm.q.len() {
                arm.q[i] += dq[i] * scale;
            }
        } else {
            arm.q = q_try;
        }
        arrived
    }

    pub fn beam_centroid(&self) -> Vector3<f64> {
        let n = self.fem.n_verts().max(1) as f64;
        self.fem.x.iter().sum::<Vector3<f64>>() / n
    }
    /// Mean height the beam has risen off its rest position.
    pub fn lift_off_rest(&self) -> f64 {
        self.beam_centroid().z - self.rest_z
    }
    pub fn left_gripper(&self) -> Vector3<f64> {
        Self::tool(&self.left.robot, &self.left.q, self.ee, self.tip)
    }
    pub fn right_gripper(&self) -> Vector3<f64> {
        Self::tool(&self.right.robot, &self.right.q, self.ee, self.tip)
    }
}

impl Default for Bimanual {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Live bench wrapper.
// ---------------------------------------------------------------------------------------------
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BimanualLab {
    sim: Bimanual,
    edges: Vec<[usize; 2]>,
}

fn skeleton(robot: &Robot, q: &[f64], ee: usize, tip: Vector3<f64>) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..=ee {
        let p = robot.frame_pose(q, i).translation.vector;
        out.push(p.x);
        out.push(p.z);
    }
    let g = (robot.frame_pose(q, ee) * nalgebra::Point3::from(tip)).coords;
    out.push(g.x);
    out.push(g.z);
    out
}

#[wasm_bindgen]
impl BimanualLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> BimanualLab {
        let sim = Bimanual::new();
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for a in 0..sim.fem.n_verts() {
            for b in (a + 1)..sim.fem.n_verts() {
                if (sim.fem.x[a] - sim.fem.x[b]).norm() < 1.05 * 0.06 {
                    set.insert((a, b));
                }
            }
        }
        let edges = set.into_iter().map(|(a, b)| [a, b]).collect();
        BimanualLab { sim, edges }
    }

    pub fn tick(&mut self, n: usize) {
        for _ in 0..n {
            self.sim.step();
        }
    }
    /// 0 reach · 1 descend · 2 grip · 3 lift · 4 hold.
    pub fn phase(&self) -> u8 {
        self.sim.phase
    }
    pub fn skeleton_l(&self) -> Vec<f64> {
        skeleton(&self.sim.left.robot, &self.sim.left.q, self.sim.ee, self.sim.tip)
    }
    pub fn skeleton_r(&self) -> Vec<f64> {
        skeleton(&self.sim.right.robot, &self.sim.right.q, self.sim.ee, self.sim.tip)
    }
    pub fn beam_verts_xz(&self) -> Vec<f64> {
        self.sim.fem.x.iter().flat_map(|p| [p.x, p.z]).collect()
    }
    pub fn beam_edges(&self) -> Vec<u32> {
        self.edges.iter().flat_map(|e| [e[0] as u32, e[1] as u32]).collect()
    }
    /// Both gripper points as `[lx, lz, rx, rz]`.
    pub fn grippers_xz(&self) -> Vec<f64> {
        let l = self.sim.left_gripper();
        let r = self.sim.right_gripper();
        vec![l.x, l.z, r.x, r.z]
    }
    pub fn floor_z(&self) -> f64 {
        self.sim.floor_z
    }
    pub fn lift(&self) -> f64 {
        self.sim.lift_off_rest()
    }
    pub fn reset(&mut self) {
        *self = BimanualLab::new();
    }
}

impl Default for BimanualLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Bimanual oracle.** Two arms take the two ends of a soft beam and lift it together off the
    /// floor — both ends rise, the beam is retained, and it never explodes.
    #[test]
    fn two_arms_lift_the_beam_together() {
        let mut sim = Bimanual::new();
        let z0 = sim.beam_centroid().z;

        let mut gripped = false;
        for _ in 0..2500 {
            sim.step();
            if sim.phase >= GRIP {
                gripped = true;
            }
            if sim.phase == HOLD && sim.t > 400 {
                break;
            }
        }
        assert!(gripped, "the arms never reached the grip phase");

        let lift = sim.lift_off_rest();
        // both ends lifted: the beam verts nearest each gripper rose
        let left_end = sim.fem.x.iter().filter(|p| p.x < 0.4).map(|p| p.z).sum::<f64>() / sim.fem.x.iter().filter(|p| p.x < 0.4).count().max(1) as f64;
        let right_end = sim.fem.x.iter().filter(|p| p.x > 0.5).map(|p| p.z).sum::<f64>() / sim.fem.x.iter().filter(|p| p.x > 0.5).count().max(1) as f64;
        eprintln!("bimanual: lift {lift:.3} m (z {z0:.3}→{:.3}), left end z {left_end:.3}, right end z {right_end:.3}", sim.beam_centroid().z);
        assert!(lift > 0.05 && lift < 0.4, "the beam was not cleanly lifted: {lift:.3} m");
        assert!(left_end > sim.rest_z + 0.03 && right_end > sim.rest_z + 0.03, "a beam end was not lifted: L {left_end:.3} R {right_end:.3}");
    }
}
