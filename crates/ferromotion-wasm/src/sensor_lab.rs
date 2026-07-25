//! **Sensor-rendering bench** — the wasm rig behind the verified browser sensor renderer. Isaac,
//! ManiSkill, and Genesis render RGB-D / lidar / segmentation with CUDA ray tracers; this is the
//! browser-native, hardware-unbound counterpart, and — like the fluid bench — it renders *and
//! self-verifies in the page*: the depth of the tracked sphere is checked against the closed-form
//! ray–sphere intersection every frame. Depth camera, per-object segmentation, and a top-down lidar
//! slice over the analytic [`ferromotion_core::SdfScene`].

use ferromotion_core::{DepthCamera, Lidar, Sdf, SdfScene};
use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SensorLab {
    scene: SdfScene,
    width: usize,
    height: usize,
    // last render
    range: Vec<f64>,
    seg: Vec<i32>,
    far: f64,
    range_err: f64,
    // the tracked sphere (segmentation id 0) for the self-check
    sphere_c: Vector3<f64>,
    sphere_r: f64,
    // last camera
    cam: Option<DepthCamera>,
}

fn look_at_pose(eye: Vector3<f64>, target: Vector3<f64>) -> Isometry3<f64> {
    // camera frame: +x right, +y down, +z forward (CV convention)
    let up = Vector3::new(0.0, 0.0, 1.0);
    let forward = (target - eye).normalize();
    let right = forward.cross(&up).normalize();
    let down = forward.cross(&right);
    let r = Matrix3::from_columns(&[right, down, forward]);
    Isometry3::from_parts(Translation3::from(eye), UnitQuaternion::from_matrix(&r))
}

#[wasm_bindgen]
impl SensorLab {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> SensorLab {
        // scene: prim 0 = tracked sphere, 1 = second sphere, 2 = box, 3 = floor plane
        let sphere_c = Vector3::new(0.0, 0.0, 0.6);
        let sphere_r = 0.5;
        let scene = SdfScene {
            prims: vec![
                Sdf::Sphere { center: sphere_c, radius: sphere_r },
                Sdf::Sphere { center: Vector3::new(1.1, 0.4, 0.35), radius: 0.3 },
                Sdf::Box { center: Vector3::new(-1.0, -0.2, 0.4), half: Vector3::new(0.35, 0.35, 0.4) },
                Sdf::Plane { normal: Vector3::new(0.0, 0.0, 1.0), offset: 0.0 },
            ],
        };
        SensorLab { scene, width, height, range: vec![], seg: vec![], far: 12.0, range_err: 0.0, sphere_c, sphere_r, cam: None }
    }

    /// Render from a camera orbiting the scene at `azimuth`/`elevation` (radians) and `radius`.
    pub fn render(&mut self, azimuth: f64, elevation: f64, radius: f64) {
        let target = Vector3::new(0.0, 0.0, 0.55);
        let eye = target + radius * Vector3::new(elevation.cos() * azimuth.cos(), elevation.cos() * azimuth.sin(), elevation.sin());
        let f = 0.9 * self.width as f64; // focal length
        let cam = DepthCamera {
            pose: look_at_pose(eye, target),
            fx: f,
            fy: f,
            cx: self.width as f64 / 2.0,
            cy: self.height as f64 / 2.0,
            width: self.width,
            height: self.height,
            far: self.far,
        };
        let img = cam.render(&self.scene);
        // self-verification: rendered range vs analytic ray-sphere for the tracked sphere (seg 0)
        let mut worst = 0.0f64;
        for v in 0..self.height {
            for u in 0..self.width {
                let idx = v * self.width + u;
                if img.seg[idx] == 0 {
                    let (o, d) = cam.pixel_ray(u as f64 + 0.5, v as f64 + 0.5);
                    let oc = o - self.sphere_c;
                    let b = d.dot(&oc);
                    let disc = b * b - (oc.norm_squared() - self.sphere_r * self.sphere_r);
                    if disc >= 0.0 {
                        let t = -b - disc.sqrt();
                        if t > 0.0 {
                            worst = worst.max((img.range[idx] - t).abs());
                        }
                    }
                }
            }
        }
        self.range = img.range;
        self.seg = img.seg;
        self.range_err = worst;
        self.cam = Some(cam);
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    /// Depth normalized to `[0,1]` (near→0, far/miss→1) for a grayscale image.
    pub fn depth_normalized(&self) -> Vec<f64> {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &r in &self.range {
            if r < self.far {
                lo = lo.min(r);
                hi = hi.max(r);
            }
        }
        if !lo.is_finite() {
            return vec![1.0; self.range.len()];
        }
        self.range.iter().map(|&r| if r >= self.far { 1.0 } else { (r - lo) / (hi - lo + 1e-9) }).collect()
    }

    /// Per-pixel segmentation label (`-1` background).
    pub fn seg(&self) -> Vec<i32> {
        self.seg.clone()
    }

    /// Max depth error vs the analytic ray–sphere intersection over the tracked sphere — the live
    /// verification receipt.
    pub fn range_error(&self) -> f64 {
        self.range_err
    }

    /// A top-down lidar slice (single elevation ring) from a sensor placed outside the objects,
    /// returning hit points as flat `[x, y, …]` in world coordinates.
    pub fn lidar_topdown(&self, n_rays: usize) -> Vec<f64> {
        let lidar = Lidar {
            pose: Isometry3::from_parts(Translation3::new(0.0, -2.6, 0.5), UnitQuaternion::identity()),
            n_azimuth: n_rays,
            n_elevation: 1,
            az_min: 0.0,
            az_max: std::f64::consts::TAU * (1.0 - 1.0 / n_rays as f64),
            el_min: 0.0,
            el_max: 0.0,
            far: self.far,
        };
        lidar.scan(&self.scene).points.iter().flat_map(|p| [p.x, p.y]).collect()
    }
}
