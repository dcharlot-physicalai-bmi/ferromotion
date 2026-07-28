//! **WebGPU sensor reference** — the data half of running the signed-distance sensor renderer on the
//! *local GPU*. Holds a fixed SDF-sphere scene + camera, exposes the scene to a WGSL compute shader
//! (sphere list + camera as flat float arrays), and renders the SAME scene on the CPU with the
//! verified core ray marcher ([`ferromotion_core::sensor_render`]). The island runs the WGSL path on
//! whatever local GPU the browser exposes and checks it against `cpu_depth()` — so "it ran on your
//! GPU" comes with "and it matches the CPU reference to N decimals", not just a claim.

use ferromotion_core::{DepthCamera, Sdf, SdfScene};
use nalgebra::{Isometry3, Vector3};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GpuSensorRef {
    scene: SdfScene,
    cam: DepthCamera,
    spheres: Vec<f32>, // [cx,cy,cz,r, …]
}

#[wasm_bindgen]
impl GpuSensorRef {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GpuSensorRef {
        // three well-separated spheres in front of the camera (camera at origin, looking +z) — no
        // screen-space overlap, so a ray that hits a surface in both renders hits the SAME surface,
        // and the GPU↔CPU comparison isolates float precision from the silhouette boundary.
        let prims = vec![
            Sdf::Sphere { center: Vector3::new(-0.9, 0.0, 2.8), radius: 0.40 },
            Sdf::Sphere { center: Vector3::new(0.0, 0.0, 4.0), radius: 0.45 },
            Sdf::Sphere { center: Vector3::new(0.9, 0.0, 2.2), radius: 0.38 },
        ];
        let spheres: Vec<f32> = prims
            .iter()
            .flat_map(|p| match p {
                Sdf::Sphere { center, radius } => [center.x as f32, center.y as f32, center.z as f32, *radius as f32],
                _ => [0.0; 4],
            })
            .collect();
        let scene = SdfScene { prims };
        let cam = DepthCamera { pose: Isometry3::identity(), fx: 96.0, fy: 96.0, cx: 79.5, cy: 59.5, width: 160, height: 120, far: 8.0 };
        GpuSensorRef { scene, cam, spheres }
    }

    pub fn width(&self) -> usize {
        self.cam.width
    }
    pub fn height(&self) -> usize {
        self.cam.height
    }
    pub fn far(&self) -> f64 {
        self.cam.far
    }
    /// Flat `[cx, cy, cz, r, …]` for each sphere (for the WGSL storage buffer).
    pub fn spheres(&self) -> Vec<f32> {
        self.spheres.clone()
    }
    pub fn n_spheres(&self) -> usize {
        self.spheres.len() / 4
    }
    /// Camera intrinsics `[fx, fy, cx, cy]`.
    pub fn intrinsics(&self) -> Vec<f32> {
        vec![self.cam.fx as f32, self.cam.fy as f32, self.cam.cx as f32, self.cam.cy as f32]
    }
    /// Camera origin (world) `[x, y, z]`.
    pub fn cam_origin(&self) -> Vec<f32> {
        let t = self.cam.pose.translation.vector;
        vec![t.x as f32, t.y as f32, t.z as f32]
    }
    /// Camera rotation as a 3×3 row-major matrix (9 floats) — maps camera-frame rays to the world.
    pub fn cam_rot(&self) -> Vec<f32> {
        let r = self.cam.pose.rotation.to_rotation_matrix();
        let m = r.matrix();
        (0..3).flat_map(|i| (0..3).map(move |j| m[(i, j)] as f32)).collect()
    }
    /// The VERIFIED CPU render (Euclidean range per pixel, `far` where nothing was hit), row-major.
    pub fn cpu_depth(&self) -> Vec<f32> {
        self.scene_render().0
    }
    /// Per-pixel segmentation label (`-1` background), row-major.
    pub fn cpu_seg(&self) -> Vec<i32> {
        self.scene_render().1
    }

    fn scene_render(&self) -> (Vec<f32>, Vec<i32>) {
        let img = self.cam.render(&self.scene);
        (img.range.iter().map(|&r| r as f32).collect(), img.seg)
    }
}

impl Default for GpuSensorRef {
    fn default() -> Self {
        Self::new()
    }
}
