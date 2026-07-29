//! **WebGPU batch collision-checker** — the data half of the honest GPU win for sampling-based
//! planning. An RRT tree is sequential (each sample extends the nearest existing node), a poor GPU
//! fit; but the planner's *hot loop* — testing whether a candidate joint configuration drives the arm
//! through an obstacle — is embarrassingly parallel across candidates. Here a whole batch of configs
//! is checked at once: one GPU thread per config runs forward kinematics, places the arm's swept
//! collision spheres, and min-reduces their signed distance to an [`SdfScene`] — exactly what
//! [`arm_clearance`] does on the CPU, which is the reference the GPU is verified against.

use ferromotion_core::{arm_clearance, from_urdf_str, JointKind, Robot, Sdf, SdfScene};
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

const LINK_R: f64 = 0.03;
const PER_LINK: usize = 3;
const MARGIN: f64 = 0.012;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[wasm_bindgen]
pub struct GpuPlanRef {
    robot: Robot,
    dof: usize,
    configs: Vec<f64>,   // n_configs × dof
    n_configs: usize,
    tool_pts: Vec<f32>,  // n_configs × 3 (CPU FK, for the scatter viz)
    wall_base: Vector3<f64>,
    wall_half: Vector3<f64>,
    joints_flat: Vec<f32>,
    ee_flat: Vec<f32>,
}

#[wasm_bindgen]
impl GpuPlanRef {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GpuPlanRef {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let dof = robot.dof();
        let n_configs = 4096usize;

        // a coherent cloud of candidate configs in front of the arm (deterministic)
        let mut s = 0xC0FFEEu64;
        let mut configs = Vec::with_capacity(n_configs * dof);
        for _ in 0..n_configs {
            for _ in 0..dof {
                let u = (splitmix64(&mut s) as f64) / (u64::MAX as f64); // [0,1)
                configs.push((u * 2.0 - 1.0) * 1.1); // [-1.1, 1.1] rad
            }
        }

        // tool point per config, and the cloud centroid → where to seat the sweeping obstacle
        let mut tool_pts = Vec::with_capacity(n_configs * 3);
        let mut c = Vector3::zeros();
        for k in 0..n_configs {
            let q = &configs[k * dof..(k + 1) * dof];
            let p = robot.fk(q).translation.vector;
            tool_pts.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
            c += p;
        }
        let wall_base = c / n_configs as f64;
        let wall_half = Vector3::new(0.05, 0.20, 0.14);

        // per-joint FK data for the GPU: origin rotation (col-major 3×3), origin translation,
        // axis, kind (0 = revolute, 1 = prismatic)
        let mut joints_flat = Vec::with_capacity(dof * 16);
        for j in &robot.joints {
            let r = j.origin.rotation.to_rotation_matrix();
            joints_flat.extend(r.matrix().as_slice().iter().map(|&v| v as f32)); // 9, col-major
            let t = j.origin.translation.vector;
            joints_flat.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);
            let a = j.axis.into_inner();
            joints_flat.extend_from_slice(&[a.x as f32, a.y as f32, a.z as f32]);
            joints_flat.push(match j.kind {
                JointKind::Revolute => 0.0,
                JointKind::Prismatic => 1.0,
            });
        }
        let r = robot.ee_offset.rotation.to_rotation_matrix();
        let mut ee_flat: Vec<f32> = r.matrix().as_slice().iter().map(|&v| v as f32).collect();
        let t = robot.ee_offset.translation.vector;
        ee_flat.extend_from_slice(&[t.x as f32, t.y as f32, t.z as f32]);

        GpuPlanRef { robot, dof, configs, n_configs, tool_pts, wall_base, wall_half, joints_flat, ee_flat }
    }

    pub fn n_configs(&self) -> usize {
        self.n_configs
    }
    pub fn dof(&self) -> usize {
        self.dof
    }
    pub fn joints_flat(&self) -> Vec<f32> {
        self.joints_flat.clone()
    }
    pub fn ee_flat(&self) -> Vec<f32> {
        self.ee_flat.clone()
    }
    /// All candidate configs, flat `[q0,q1,…]` of length `n_configs × dof` (f32 for the GPU).
    pub fn configs(&self) -> Vec<f32> {
        self.configs.iter().map(|&v| v as f32).collect()
    }
    /// Tool world position per config, flat `[x,y,z,…]` — for the top-down scatter.
    pub fn tool_pts(&self) -> Vec<f32> {
        self.tool_pts.clone()
    }
    /// `[dof, per_link, link_radius, n_configs, margin]`.
    pub fn params(&self) -> Vec<f32> {
        vec![self.dof as f32, PER_LINK as f32, LINK_R as f32, self.n_configs as f32, MARGIN as f32]
    }
    /// Base obstacle box `[cx,cy,cz, hx,hy,hz]` (before the animation sweep).
    pub fn wall(&self) -> Vec<f32> {
        vec![
            self.wall_base.x as f32, self.wall_base.y as f32, self.wall_base.z as f32,
            self.wall_half.x as f32, self.wall_half.y as f32, self.wall_half.z as f32,
        ]
    }

    /// The VERIFIED CPU render: min clearance of every config to the obstacle box swept by
    /// `(dx,dy,dz)`. Returns `n_configs` clearances — the reference the GPU is checked against.
    pub fn cpu_clear(&self, dx: f32, dy: f32, dz: f32) -> Vec<f32> {
        let center = self.wall_base + Vector3::new(dx as f64, dy as f64, dz as f64);
        let scene = SdfScene { prims: vec![Sdf::Box { center, half: self.wall_half }] };
        (0..self.n_configs)
            .map(|k| {
                let q = &self.configs[k * self.dof..(k + 1) * self.dof];
                arm_clearance(&self.robot, q, &scene, LINK_R, PER_LINK) as f32
            })
            .collect()
    }
}

impl Default for GpuPlanRef {
    fn default() -> Self {
        Self::new()
    }
}
