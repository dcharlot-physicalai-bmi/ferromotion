//! **The sensorimotor loop** — perception and action closed on each other. Every other controller in
//! this crate servos on *state you are handed*; this one servos on *what the robot's own eye sees*.
//! The eye is [`ferromotion_core::DepthCamera`], the flagship browser-native raytraced depth +
//! segmentation sensor: rays are marched through the [`SdfScene`], and the target is recovered **only
//! from that rendered image** — its segmentation mask gives the pixels, their per-pixel range
//! reconstructs its position. Nothing reads the scene's ground truth. Then image-based visual servoing
//! ([`crate::visual_servo`]) drives a camera-frame twist that centers the target in view and closes to
//! a standoff. The result is the full `render → perceive → servo → move → render` cycle — a robot
//! that finds and tracks a target through its own synthetic retina, entirely pure `nalgebra` → WASM.
//!
//! The honest part is that perception is *earned* from the raytraced buffer, not read off the model:
//! [`perceive`] segments the target out of the [`DepthImage`] and rebuilds its 3-D position from the
//! range channel, and the oracle checks that recovered feature against the ground-truth projection.

use crate::visual_servo::Camera;
use ferromotion_core::{solve_diffik, DepthCamera, DiffIkOptions, FrameTaskDef, Robot, SdfScene};
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3, Vector6};

/// What the robot infers about the target **from the raytraced image alone**.
#[derive(Clone, Debug)]
pub struct Perception {
    /// Whether the target's segmentation label appeared in the frame at all.
    pub seen: bool,
    /// Number of pixels the target covered (its apparent area).
    pub n_pixels: usize,
    /// Target centroid in pixels.
    pub u: f64,
    pub v: f64,
    /// Target centroid in normalized image coordinates `(x, y) = ((u−cx)/fx, (v−cy)/fy)` — the IBVS feature.
    pub x: f64,
    pub y: f64,
    /// Mean camera-frame depth of the target's surface (metres along the optical `+z`).
    pub z: f64,
    /// Reconstructed target position in the **world**, averaged over its visible surface hits.
    pub point_world: Vector3<f64>,
    /// Centering error `‖(x, y)‖` — 0 when the target sits on the optical axis.
    pub center_err: f64,
}

impl Perception {
    fn nothing() -> Self {
        Perception {
            seen: false,
            n_pixels: 0,
            u: 0.0,
            v: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            point_world: Vector3::zeros(),
            center_err: f64::INFINITY,
        }
    }
}

/// Segment `target_label` out of the camera's raytraced frame and recover its image feature and
/// 3-D position from the range channel. Reads only the rendered [`DepthImage`], never the scene.
pub fn perceive(cam: &DepthCamera, scene: &SdfScene, target_label: i32) -> Perception {
    let img = cam.render(scene);
    let (mut su, mut sv, mut sz) = (0.0f64, 0.0f64, 0.0f64);
    let mut sworld = Vector3::zeros();
    let mut n = 0usize;
    for vy in 0..img.height {
        for ux in 0..img.width {
            if img.seg_at(ux, vy) != target_label {
                continue;
            }
            let rng = img.range_at(ux, vy);
            if !(rng.is_finite() && rng < cam.far) {
                continue;
            }
            // camera-frame ray for this pixel (unit); range is Euclidean along it
            let dir_cam =
                Vector3::new((ux as f64 + 0.5 - cam.cx) / cam.fx, (vy as f64 + 0.5 - cam.cy) / cam.fy, 1.0).normalize();
            let (o, d_world) = cam.pixel_ray(ux as f64 + 0.5, vy as f64 + 0.5);
            su += ux as f64 + 0.5;
            sv += vy as f64 + 0.5;
            sz += rng * dir_cam.z; // optical-axis depth of this surface hit
            sworld += o + rng * d_world; // world-space hit point
            n += 1;
        }
    }
    if n == 0 {
        return Perception::nothing();
    }
    let inv = 1.0 / n as f64;
    let (u, v, z) = (su * inv, sv * inv, sz * inv);
    let x = (u - cam.cx) / cam.fx;
    let y = (v - cam.cy) / cam.fy;
    Perception {
        seen: true,
        n_pixels: n,
        u,
        v,
        x,
        y,
        z,
        point_world: sworld * inv,
        center_err: (x * x + y * y).sqrt(),
    }
}

/// The camera-frame twist `(v, ω)` that servos the eye onto the target. This is the **decoupled gaze
/// law** — the near-axis reduction of the IBVS interaction matrix `L(x, y, Z)`: a single centroid
/// feature gives two constraints, and the well-conditioned way to spend them is pan/tilt rotation, not
/// the min-norm 6-DoF pseudo-inverse (whose null space commands a spurious optical-axis screw that can
/// throw the target out of frame). From `L`, near the axis `ẋ ≈ −ω_y`, `ẏ ≈ ω_x`, so `ω_y = λx`,
/// `ω_x = −λy` drive the feature down an exponential envelope; a forward term closes the range to
/// `standoff`. At the fixed point the target is centered (`x = y = 0`) at range `standoff`.
pub fn servo_twist(p: &Perception, standoff: f64, lambda: f64, k_range: f64) -> Vector6<f64> {
    if !p.seen {
        return Vector6::zeros();
    }
    let mut tw = Vector6::zeros();
    tw[2] = k_range * (p.z - standoff); // v_z: approach along the optical axis
    tw[3] = -lambda * p.y; // ω_x: tilt to null the vertical feature error
    tw[4] = lambda * p.x; // ω_y: pan to null the horizontal feature error
    tw
}

/// A free-flying eye that finds and tracks a target through its own raytraced retina. The camera is
/// the sensor *and* the moving body; each [`step`](FreeEye::step) renders, perceives, and integrates
/// the servo twist. Mount the same loop on a manipulator's wrist frame to get eye-in-hand reaching.
#[derive(Clone, Debug)]
pub struct FreeEye {
    pub cam: DepthCamera,
    /// IBVS centering gain.
    pub lambda: f64,
    /// Range-approach gain.
    pub k_range: f64,
    /// Desired standoff range to the target (metres).
    pub standoff: f64,
    pub dt: f64,
}

impl FreeEye {
    pub fn new(cam: DepthCamera, standoff: f64) -> Self {
        FreeEye { cam, lambda: 4.0, k_range: 2.5, standoff, dt: 0.02 }
    }

    /// One perception→action cycle. Returns what was perceived this frame.
    pub fn step(&mut self, scene: &SdfScene, target_label: i32) -> Perception {
        let p = perceive(&self.cam, scene, target_label);
        if p.seen {
            let tw = servo_twist(&p, self.standoff, self.lambda, self.k_range);
            // Integrate the camera-frame twist through the shared visual-servo camera model.
            let mut c = Camera { r: self.cam.pose.rotation.to_rotation_matrix().into_inner(), t: self.cam.pose.translation.vector };
            c.integrate(&tw, self.dt);
            self.cam.pose = Isometry3::from_parts(Translation3::from(c.t), UnitQuaternion::from_matrix(&c.r));
        }
        p
    }

    /// Camera position in the world.
    pub fn position(&self) -> Vector3<f64> {
        self.cam.pose.translation.vector
    }
}

/// **Eye-in-hand reaching** — the sensorimotor loop mounted on a real manipulator. A [`DepthCamera`]
/// rides the arm's wrist frame; each step forward-kinematics the arm, renders that wrist view,
/// perceives the target from the raytraced image (via [`perceive`]), and drives the joints so the
/// camera holds a **standoff facing the target** — a look-at reach solved through differential IK
/// ([`solve_diffik`]). "Look-at + standoff" is expressed as two position tasks on the wrist frame: the
/// camera origin reaches a point one standoff *behind* the target along the view line, and a look-point
/// one standoff *ahead* of the camera reaches the target. Together they pin the camera's position and
/// aim (5 DoF; the arm's spare DoF is free), so the target stays centered as the arm extends toward it.
/// This turns the flying eye of [`FreeEye`] into an arm that reaches for what it sees. Pure Rust → WASM.
pub struct EyeInHand {
    pub robot: Robot,
    pub q: Vec<f64>,
    /// Frame index the camera mounts on (the chain end = `robot.dof()`).
    pub ee_frame: usize,
    /// Fixed wrist→camera transform. The camera looks along the mounted frame's `+z`.
    pub mount: Isometry3<f64>,
    /// Carries the intrinsics/resolution; its pose is overwritten from FK each step.
    pub cam: DepthCamera,
    pub standoff: f64,
    pub gain: f64,
    pub dt: f64,
    /// Inner differential-IK iterations per perception cycle (kept small so the eye re-perceives often).
    pub inner_iters: usize,
}

impl EyeInHand {
    /// The current wrist-mounted camera, its pose taken from forward kinematics.
    pub fn camera(&self) -> DepthCamera {
        let mut c = self.cam.clone();
        c.pose = self.robot.frame_pose(&self.q, self.ee_frame) * self.mount;
        c
    }

    /// Camera position in the world.
    pub fn camera_pos(&self) -> Vector3<f64> {
        (self.robot.frame_pose(&self.q, self.ee_frame) * self.mount).translation.vector
    }

    /// One perception→reach cycle. Returns what the wrist camera perceived this frame.
    pub fn step(&mut self, scene: &SdfScene, label: i32) -> Perception {
        let cam = self.camera();
        let p = perceive(&cam, scene, label);
        if !p.seen {
            return p;
        }
        let cam_pos = cam.pose.translation.vector;
        let d = p.point_world - cam_pos;
        let dist = d.norm();
        if dist < 1e-9 {
            return p;
        }
        let u = d / dist; // view direction toward the perceived target
        // Two look-at tasks on the wrist frame: camera origin one standoff behind the target, and a
        // point one standoff ahead of the camera landing on the target.
        let o_cam = self.mount.translation.vector;
        let o_look = (self.mount * Point3::new(0.0, 0.0, self.standoff)).coords;
        let goal_cam = p.point_world - self.standoff * u;
        let goal_look = p.point_world;
        let tasks = [
            FrameTaskDef::new(self.ee_frame, o_cam, goal_cam, self.gain, 1.0),
            FrameTaskDef::new(self.ee_frame, o_look, goal_look, self.gain, 1.0),
        ];
        let opts = DiffIkOptions { dt: self.dt, max_iters: self.inner_iters, use_limits: true, ..Default::default() };
        self.q = solve_diffik(&self.robot, &tasks, &self.q, &opts).q;
        p
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use ferromotion_core::{from_urdf_str, Sdf};

    /// A small imaging camera looking down its `+z` axis from the origin (kept low-res so the
    /// render-in-the-loop tests stay fast; still tens-to-hundreds of target pixels).
    fn eye() -> DepthCamera {
        DepthCamera {
            pose: Isometry3::identity(),
            fx: 60.0,
            fy: 60.0,
            cx: 31.5,
            cy: 23.5,
            width: 64,
            height: 48,
            far: 20.0,
        }
    }

    /// Ground-truth projection of a world point into normalized image coordinates (CV frame, +z fwd).
    fn project(cam: &DepthCamera, p: &Vector3<f64>) -> (f64, f64, f64) {
        let pc = cam.pose.inverse_transform_point(&p.clone().into());
        (pc.x / pc.z, pc.y / pc.z, pc.z)
    }

    /// **Perception oracle.** The target recovered *from the raytraced segmentation + range image*
    /// must agree with the ground-truth projection of the true sphere centre — to sub-pixel accuracy
    /// in the image, and to within one radius in 3-D (a surface centroid sits a little in front of
    /// the centre). Nothing here reads the sphere's parameters; it is all decoded from the image.
    #[test]
    fn perception_recovers_target_from_the_raytraced_image() {
        let cam = eye();
        let center = Vector3::new(0.6, -0.3, 4.0);
        let radius = 0.5;
        let scene = SdfScene { prims: vec![Sdf::Sphere { center, radius }] };
        let p = perceive(&cam, &scene, 0);
        assert!(p.seen && p.n_pixels > 50, "target barely visible: {} px", p.n_pixels);

        // Image feature vs ground-truth projection of the centre — within ~1 pixel.
        let (gx, gy, gz) = project(&cam, &center);
        let feat_px = ((p.x - gx) * cam.fx).hypot((p.y - gy) * cam.fy);
        eprintln!("perception: feature err {feat_px:.3} px, depth {:.3} (true centre {gz:.3}), |P̂−C| {:.3}", p.z, (p.point_world - center).norm());
        assert!(feat_px < 1.5, "image feature off by {feat_px:.2} px");

        // Recovered depth is a front surface, so a little nearer than the centre, by < one radius.
        assert!(p.z < gz && gz - p.z < radius + 1e-3, "depth {} not a front surface of centre {gz}", p.z);
        // 3-D reconstruction within one radius of the true centre.
        assert!((p.point_world - center).norm() < radius, "3-D recon too far: {}", (p.point_world - center).norm());
    }

    /// **Closed-loop oracle.** Starting with the target off-axis and too far, the free eye must, using
    /// only its raytraced perception, servo until the *true* target centre projects onto the optical
    /// axis and the range reaches the standoff. Ground truth is used only to grade the result.
    #[test]
    fn free_eye_centers_true_target_and_reaches_standoff() {
        let center = Vector3::new(1.1, 0.7, 5.0);
        let radius = 0.45;
        let scene = SdfScene { prims: vec![Sdf::Sphere { center, radius }] };
        let mut fe = FreeEye::new(eye(), 2.5);
        let (gx0, gy0, gz0) = project(&fe.cam, &center);
        let start_off = (gx0 * gx0 + gy0 * gy0).sqrt();
        assert!(start_off > 0.15, "target started too close to axis to be a real test: {start_off}");

        let mut last = perceive(&fe.cam, &scene, 0);
        for _ in 0..1200 {
            last = fe.step(&scene, 0);
            assert!(last.seen, "the robot lost sight of the target mid-servo");
        }

        // Grade against ground truth: the TRUE centre now sits on the optical axis …
        let (gx, gy, gz) = project(&fe.cam, &center);
        let axis_err = (gx * gx + gy * gy).sqrt();
        eprintln!(
            "servo: on-axis err {start_off:.3} → {axis_err:.4}; perceived range {gz0:.2} → {:.3} (standoff {:.2}); centre range {gz:.3}",
            last.z, fe.standoff
        );
        assert!(axis_err < 5e-3, "true target not centered: {axis_err:.2e}");
        // … and the perceived surface range — the quantity the loop controls — reached the standoff.
        // (The centre sits ~one front-surface offset beyond that, gz ≈ standoff + offset, as expected.)
        assert!((last.z - fe.standoff).abs() < 0.05, "did not reach standoff: perceived range {} vs {}", last.z, fe.standoff);
        assert!(gz > fe.standoff, "centre should lie beyond the perceived front surface");
    }

    /// **Tracking oracle.** With the target drifting along a path, the eye keeps it centered: the
    /// steady-state on-axis error of the *true* centre stays small frame to frame.
    #[test]
    fn free_eye_tracks_a_moving_target() {
        let mut center = Vector3::new(0.0, 0.0, 4.0);
        let radius = 0.4;
        let mut fe = FreeEye::new(eye(), 2.2);
        // settle onto the initially static target
        for _ in 0..800 {
            let scene = SdfScene { prims: vec![Sdf::Sphere { center, radius }] };
            fe.step(&scene, 0);
        }
        // now drift the target and watch tracking error
        let mut worst = 0.0f64;
        for k in 0..600 {
            let t = k as f64 * 0.02;
            // a slow lateral + depth weave, kept inside the frustum
            center = Vector3::new(0.9 * (0.7 * t).sin(), 0.6 * (0.5 * t).cos() - 0.3, 4.0 + 0.8 * (0.4 * t).sin());
            let scene = SdfScene { prims: vec![Sdf::Sphere { center, radius }] };
            let p = fe.step(&scene, 0);
            assert!(p.seen, "lost the moving target at t={t:.2}");
            if k > 100 {
                let (gx, gy, _) = project(&fe.cam, &center);
                worst = worst.max((gx * gx + gy * gy).sqrt());
            }
        }
        eprintln!("tracking: worst steady-state on-axis error {worst:.4}");
        assert!(worst < 0.08, "tracking error too large: {worst}");
    }

    // The diffik reference arm: a compact 6-DOF chain, world → … → tool.
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
        // Matches the live EyeInHandLab intrinsics (wide FOV, edge ≈ 0.76 normalized).
        DepthCamera { pose: Isometry3::identity(), fx: 64.0, fy: 64.0, cx: 47.5, cy: 35.5, width: 96, height: 72, far: 8.0 }
    }

    fn on_axis(cam: &DepthCamera, p: &Vector3<f64>) -> f64 {
        let pc = cam.pose.inverse_transform_point(&(*p).into());
        (pc.x / pc.z).hypot(pc.y / pc.z)
    }

    /// **Eye-in-hand oracle.** A 6-DOF arm with a wrist camera, starting with the target in view,
    /// drives its joints — through differential IK on the two look-at tasks — until the camera holds a
    /// standoff facing the target: the target stays visible the whole reach, the TRUE target ends up
    /// centered in the wrist image, and the camera sits ~one standoff from it. Ground truth only grades.
    #[test]
    fn eye_in_hand_reaches_and_faces_the_seen_target() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let ee = robot.dof();
        let mount = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.05), UnitQuaternion::identity());
        let standoff = 0.25;
        let radius = 0.07;

        // Define the target from a KNOWN-reachable "reached" pose: place the sphere exactly one
        // standoff ahead of the camera at `q_tgt`, on its optical axis. Then a facing-at-standoff
        // solution (q_tgt itself) provably exists inside the workspace — the reach is solvable.
        let q_tgt = vec![0.15, 0.70, -0.50, 0.30, 0.05, 0.10];
        let cam_t = { let mut c = wrist_eye(); c.pose = robot.frame_pose(&q_tgt, ee) * mount; c };
        let fwd_t = cam_t.pose.rotation * Vector3::new(0.0, 0.0, 1.0);
        let center = cam_t.pose.translation.vector + standoff * fwd_t;
        let scene = SdfScene { prims: vec![Sdf::Sphere { center, radius }] };

        // Start from a perturbed configuration — the arm is off the solution, but (wide FOV) the
        // target is still in view — and let the loop drive it back to a facing-at-standoff pose.
        let q0 = vec![0.40, 0.35, -0.15, 0.10, 0.20, 0.25];
        let mut eih = EyeInHand { robot, q: q0, ee_frame: ee, mount, cam: wrist_eye(), standoff, gain: 2.5, dt: 0.05, inner_iters: 6 };
        let start_axis = on_axis(&eih.camera(), &center);
        let start_dist = (eih.camera_pos() - center).norm();

        let mut last = perceive(&eih.camera(), &scene, 0);
        for _ in 0..300 {
            last = eih.step(&scene, 0);
            assert!(last.seen, "the arm lost sight of the target mid-reach");
        }

        let camf = eih.camera();
        let axis_err = on_axis(&camf, &center);
        let dist = (eih.camera_pos() - center).norm();
        eprintln!("eye-in-hand: on-axis {start_axis:.3} → {axis_err:.4}; camera↔target {start_dist:.3} → {dist:.3} (standoff {standoff}); perceived range {:.3}", last.z);
        assert!(axis_err < 0.03, "true target not centered by the arm: {axis_err:.3}");
        assert!((dist - standoff).abs() < radius + 0.06, "arm did not reach ~standoff: {dist} vs {standoff}");
        assert!((last.point_world - center).norm() < radius, "3-D perception off: {}", (last.point_world - center).norm());
    }
}
