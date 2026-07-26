//! **Eye-in-hand bench** — the sensorimotor loop on a real manipulator. A 6-DOF arm carries a
//! [`DepthCamera`] on its wrist; it sees a target only through that raytraced image and drives its
//! joints (differential IK on two look-at tasks) until the wrist camera faces the target at a
//! standoff. Two panels: the arm reaching (side view) and the robot's own wrist view centering the
//! target. Click for a new target and it reaches again — [`ferromotion_control::EyeInHand`] live.

use ferromotion_control::{EyeInHand, Perception};
use ferromotion_core::{from_urdf_str, DepthCamera, Sdf, SdfScene};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
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

fn wrist_eye() -> DepthCamera {
    // Wide field of view (edge ≈ 0.76 in normalized coords) so the target stays in frame while the arm reaches.
    DepthCamera { pose: Isometry3::identity(), fx: 64.0, fy: 64.0, cx: 47.5, cy: 35.5, width: 96, height: 72, far: 8.0 }
}

#[wasm_bindgen]
pub struct EyeInHandLab {
    eih: EyeInHand,
    scene: SdfScene,
    center: Vector3<f64>,
    radius: f64,
    // a few reachable targets (each defined one standoff ahead of a known pose) to cycle through
    targets: Vec<Vector3<f64>>,
    ti: usize,
    width: usize,
    height: usize,
    range: Vec<f64>,
    seg: Vec<i32>,
    last: Perception,
}

fn nothing() -> Perception {
    Perception { seen: false, n_pixels: 0, u: 0.0, v: 0.0, x: 0.0, y: 0.0, z: 0.0, point_world: Vector3::zeros(), center_err: f64::INFINITY }
}

#[wasm_bindgen]
impl EyeInHandLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> EyeInHandLab {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let ee = robot.dof();
        let mount = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.05), UnitQuaternion::identity());
        let standoff = 0.25;
        let radius = 0.07;

        // Reachable targets: anchor one standoff ahead of the wrist camera at a known, test-proven
        // pose, then hop it a little within that view. Small in-frame hops keep every transition
        // visible (a target the arm can't see would freeze the loop) and every pose reachable.
        let pose0 = vec![0.15, 0.70, -0.50, 0.30, 0.05, 0.10];
        let cam0 = robot.frame_pose(&pose0, ee) * mount;
        let fwd0 = cam0.rotation * Vector3::new(0.0, 0.0, 1.0);
        let right0 = cam0.rotation * Vector3::new(1.0, 0.0, 0.0);
        let up0 = cam0.rotation * Vector3::new(0.0, -1.0, 0.0);
        let c0 = cam0.translation.vector + standoff * fwd0;
        let targets: Vec<Vector3<f64>> = vec![
            c0,
            c0 + 0.11 * right0 - 0.05 * up0,
            c0 - 0.10 * right0 + 0.06 * up0,
        ];
        let center = targets[0];
        let scene = SdfScene { prims: vec![Sdf::Sphere { center, radius }] };

        // Start off the solution but with the target in the (wide) field of view (test-proven q0).
        let q0 = vec![0.40, 0.35, -0.15, 0.10, 0.20, 0.25];
        let eih = EyeInHand { robot, q: q0, ee_frame: ee, mount, cam: wrist_eye(), standoff, gain: 2.5, dt: 0.05, inner_iters: 6 };
        let (width, height) = (eih.cam.width, eih.cam.height);
        EyeInHandLab { eih, scene, center, radius, targets, ti: 0, width, height, range: vec![], seg: vec![], last: nothing() }
    }

    /// Advance `sub` perception→reach cycles, then render the wrist view for display.
    pub fn step(&mut self, sub: usize) {
        for _ in 0..sub {
            self.last = self.eih.step(&self.scene, 0);
        }
        let img = self.eih.camera().render(&self.scene);
        self.range = img.range;
        self.seg = img.seg;
    }

    /// Cycle to the next reachable target; the arm reaches for it from wherever it is.
    pub fn new_target(&mut self) {
        self.ti = (self.ti + 1) % self.targets.len();
        self.center = self.targets[self.ti];
        self.scene = SdfScene { prims: vec![Sdf::Sphere { center: self.center, radius: self.radius }] };
    }

    /// Arm skeleton as flat `[x, z, …]` (side view): base → each joint frame → wrist → camera.
    pub fn skeleton_xz(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..=self.eih.ee_frame {
            let p = self.eih.robot.frame_pose(&self.eih.q, i).translation.vector;
            out.push(p.x);
            out.push(p.z);
        }
        let cam = self.eih.camera_pos();
        out.push(cam.x);
        out.push(cam.z);
        out
    }

    pub fn target_xz(&self) -> Vec<f64> {
        vec![self.center.x, self.center.z]
    }
    pub fn target_radius(&self) -> f64 {
        self.radius
    }
    pub fn cam_xz(&self) -> Vec<f64> {
        let c = self.eih.camera_pos();
        vec![c.x, c.z]
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    /// Depth normalized to `[0,1]` (near→0, far/miss→1).
    pub fn depth_normalized(&self) -> Vec<f64> {
        let far = self.eih.cam.far;
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &r in &self.range {
            if r < far {
                lo = lo.min(r);
                hi = hi.max(r);
            }
        }
        if !lo.is_finite() {
            return vec![1.0; self.range.len()];
        }
        self.range.iter().map(|&r| if r >= far { 1.0 } else { (r - lo) / (hi - lo + 1e-9) }).collect()
    }

    pub fn seg(&self) -> Vec<i32> {
        self.seg.clone()
    }
    pub fn seen(&self) -> bool {
        self.last.seen
    }
    pub fn center_err(&self) -> f64 {
        if self.last.center_err.is_finite() {
            self.last.center_err
        } else {
            0.0
        }
    }
    pub fn range_m(&self) -> f64 {
        self.last.z
    }
    pub fn standoff(&self) -> f64 {
        self.eih.standoff
    }
    /// Image-centre pixel — the crosshair.
    pub fn center_px(&self) -> Vec<f64> {
        vec![self.eih.cam.cx, self.eih.cam.cy]
    }
    pub fn target_px(&self) -> Vec<f64> {
        if self.last.seen {
            vec![self.last.u, self.last.v]
        } else {
            vec![-1.0, -1.0]
        }
    }
}

impl Default for EyeInHandLab {
    fn default() -> Self {
        Self::new()
    }
}
