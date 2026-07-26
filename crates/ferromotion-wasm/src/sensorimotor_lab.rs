//! **Sensorimotor-loop bench** — the flagship raytraced sensor, now *closing a control loop*. The
//! robot flies a free eye that sees a labelled target only through its own [`DepthCamera`] (segment
//! the target out of the rendered image, rebuild its position from the range channel) and visual-
//! servos to face it and close to a standoff — ignoring the unlabelled distractors. Nothing reads the
//! scene's ground truth. Toggle the target into motion and the eye tracks it. `render → perceive →
//! servo → move → render`, pure Rust → WebAssembly. This is [`ferromotion_control::FreeEye`] live.

use ferromotion_control::{FreeEye, Perception};
use ferromotion_core::{DepthCamera, Sdf, SdfScene};
use nalgebra::{Isometry3, Vector3};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SensorimotorLab {
    eye: FreeEye,
    scene: SdfScene,
    base: Vector3<f64>,   // nominal target centre
    target: Vector3<f64>, // current target centre
    radius: f64,
    t: f64,
    moving: bool,
    home: Isometry3<f64>, // camera home pose for reset
    last: Perception,
    width: usize,
    height: usize,
    range: Vec<f64>,
    seg: Vec<i32>,
}

fn nothing() -> Perception {
    // a zero-perception stand-in until the first frame is rendered
    Perception { seen: false, n_pixels: 0, u: 0.0, v: 0.0, x: 0.0, y: 0.0, z: 0.0, point_world: Vector3::zeros(), center_err: f64::INFINITY }
}

#[wasm_bindgen]
impl SensorimotorLab {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> SensorimotorLab {
        // The eye starts at the origin looking down world +z; the target sits off-axis so the loop is
        // seen to acquire it. ~53° horizontal field of view.
        let f = width as f64;
        let home = Isometry3::identity();
        let cam = DepthCamera {
            pose: home,
            fx: f,
            fy: f,
            cx: width as f64 / 2.0,
            cy: height as f64 / 2.0,
            width,
            height,
            far: 20.0,
        };
        let base = Vector3::new(0.9, -0.6, 5.0); // world (camera-frame +y is down, so −y is up)
        let radius = 0.5;
        // prim 0 = labelled target; 1,2 = unlabelled distractors the loop must ignore
        let scene = SdfScene {
            prims: vec![
                Sdf::Sphere { center: base, radius },
                Sdf::Sphere { center: Vector3::new(-1.5, 0.4, 6.6), radius: 0.6 },
                Sdf::Box { center: Vector3::new(1.9, 1.0, 7.6), half: Vector3::new(0.5, 0.5, 0.6) },
            ],
        };
        SensorimotorLab {
            eye: FreeEye::new(cam, 2.5),
            scene,
            base,
            target: base,
            radius,
            t: 0.0,
            moving: false,
            home,
            last: nothing(),
            width,
            height,
            range: vec![],
            seg: vec![],
        }
    }

    fn weave(&self) -> Vector3<f64> {
        let t = self.t;
        self.base + Vector3::new(0.95 * (0.6 * t).sin(), 0.55 * (0.4 * t + 1.0).sin(), 0.6 * (0.3 * t).sin())
    }

    /// Advance `sub` perception→action cycles, then render the current eye view for display.
    pub fn step(&mut self, sub: usize) {
        for _ in 0..sub {
            if self.moving {
                self.t += self.eye.dt;
                self.target = self.weave();
            }
            self.scene.prims[0] = Sdf::Sphere { center: self.target, radius: self.radius };
            self.last = self.eye.step(&self.scene, 0);
        }
        let img = self.eye.cam.render(&self.scene);
        self.range = img.range;
        self.seg = img.seg;
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    /// Depth normalized to `[0,1]` (near→0, far/miss→1) for a grayscale image.
    pub fn depth_normalized(&self) -> Vec<f64> {
        let far = self.eye.cam.far;
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

    /// Per-pixel segmentation label (`-1` background; `0` is the tracked target).
    pub fn seg(&self) -> Vec<i32> {
        self.seg.clone()
    }

    pub fn seen(&self) -> bool {
        self.last.seen
    }
    pub fn n_pixels(&self) -> usize {
        self.last.n_pixels
    }
    /// Target centroid in pixels (from perception); `(-1,-1)` if unseen.
    pub fn target_px(&self) -> Vec<f64> {
        if self.last.seen {
            vec![self.last.u, self.last.v]
        } else {
            vec![-1.0, -1.0]
        }
    }
    /// Optical-axis (image centre) in pixels — the crosshair.
    pub fn center_px(&self) -> Vec<f64> {
        vec![self.eye.cam.cx, self.eye.cam.cy]
    }
    /// Centering error `‖(x,y)‖` in normalized image units — 0 when the target is on the optical axis.
    pub fn center_err(&self) -> f64 {
        if self.last.center_err.is_finite() {
            self.last.center_err
        } else {
            0.0
        }
    }
    /// Perceived surface range to the target (metres) — the quantity the approach term drives to standoff.
    pub fn range_m(&self) -> f64 {
        self.last.z
    }
    pub fn standoff(&self) -> f64 {
        self.eye.standoff
    }
    pub fn cam_pos(&self) -> Vec<f64> {
        let p = self.eye.position();
        vec![p.x, p.y, p.z]
    }
    pub fn target_pos(&self) -> Vec<f64> {
        vec![self.target.x, self.target.y, self.target.z]
    }
    pub fn moving(&self) -> bool {
        self.moving
    }
    pub fn set_moving(&mut self, on: bool) {
        self.moving = on;
    }

    pub fn reset(&mut self) {
        self.eye.cam.pose = self.home;
        self.target = self.base;
        self.t = 0.0;
        self.moving = false;
        self.last = nothing();
    }
}
