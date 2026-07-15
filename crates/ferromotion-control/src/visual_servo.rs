//! **Image-based visual servoing (IBVS)** (Chaumette & Hutchinson) — closing the loop through a
//! camera. Everything else in this crate servos on state; IBVS servos directly on *what the camera
//! sees*, never reconstructing pose.
//!
//! The object is the **interaction matrix** (image Jacobian) `L`, mapping a camera twist
//! `v_c = (v, ω)` (camera frame) to the motion of a projected point feature `s = (x, y)` at depth `Z`:
//!
//! ```text
//!   ṡ = L(s, Z) · v_c,    L = [ −1/Z   0     x/Z    x·y     −(1+x²)   y  ]
//!                             [  0    −1/Z   y/Z   1+y²      −x·y    −x  ]
//! ```
//!
//! Stacking several features and inverting gives the control law `v_c = −λ L⁺ (s − s*)`, which drives
//! the feature error down an (approximately) exponential envelope — the camera finds the pose that
//! makes the image match, without ever solving for that pose. Pure `nalgebra` → WASM-clean.

use crate::exp_so3;
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3, Vector6};

/// A camera pose: `r`/`t` are the camera's orientation and position **in the world** (world←camera).
/// The camera looks along its own `+z` axis.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
}

impl Camera {
    /// Project a world point to normalized image coordinates `(x, y)` plus its depth `Z`.
    /// `None` if the point is at or behind the image plane.
    pub fn project(&self, p_world: &Vector3<f64>) -> Option<(f64, f64, f64)> {
        let pc = self.r.transpose() * (p_world - self.t); // point in the camera frame
        if pc.z <= 1e-9 {
            return None;
        }
        Some((pc.x / pc.z, pc.y / pc.z, pc.z))
    }

    /// Move the camera by a **camera-frame** twist `(v, ω)` for `dt`.
    pub fn integrate(&mut self, twist: &Vector6<f64>, dt: f64) {
        let v = Vector3::new(twist[0], twist[1], twist[2]);
        let w = Vector3::new(twist[3], twist[4], twist[5]);
        self.t += self.r * v * dt;
        self.r *= exp_so3(w * dt);
    }
}

/// The interaction matrix for one point feature at normalized coords `(x, y)` and depth `z`.
pub fn interaction_matrix(x: f64, y: f64, z: f64) -> SMatrix<f64, 2, 6> {
    SMatrix::<f64, 2, 6>::from_row_slice(&[
        -1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y, //
        0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x,
    ])
}

/// IBVS control law: the camera twist that drives the observed features toward `desired`.
/// `v_c = −λ L⁺ (s − s*)`. `None` if any feature is not visible.
pub fn ibvs_twist(cam: &Camera, points: &[Vector3<f64>], desired: &[(f64, f64)], lambda: f64) -> Option<Vector6<f64>> {
    let n = points.len();
    let mut l = DMatrix::zeros(2 * n, 6);
    let mut e = DVector::zeros(2 * n);
    for (i, p) in points.iter().enumerate() {
        let (x, y, z) = cam.project(p)?;
        l.view_mut((2 * i, 0), (2, 6)).copy_from(&interaction_matrix(x, y, z));
        e[2 * i] = x - desired[i].0;
        e[2 * i + 1] = y - desired[i].1;
    }
    let lp = l.pseudo_inverse(1e-9).ok()?;
    let v = -lambda * (lp * e);
    Some(Vector6::new(v[0], v[1], v[2], v[3], v[4], v[5]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four world points forming a square in the z = 0 plane.
    fn square() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(-0.1, -0.1, 0.0),
            Vector3::new(0.1, -0.1, 0.0),
            Vector3::new(0.1, 0.1, 0.0),
            Vector3::new(-0.1, 0.1, 0.0),
        ]
    }

    fn goal_cam() -> Camera {
        Camera { r: Matrix3::identity(), t: Vector3::new(0.0, 0.0, -0.5) }
    }

    #[test]
    fn interaction_matrix_matches_finite_differences() {
        // The defining property: ṡ = L·v_c. Move the camera by a small twist and compare.
        let p = Vector3::new(0.07, -0.04, 0.0);
        let cam = Camera { r: exp_so3(Vector3::new(0.1, -0.2, 0.15)), t: Vector3::new(0.05, -0.03, -0.6) };
        let (x, y, z) = cam.project(&p).unwrap();
        let l = interaction_matrix(x, y, z);

        let dt = 1e-7;
        for k in 0..6 {
            let mut tw = Vector6::zeros();
            tw[k] = 1.0; // unit twist along each of the six camera DoF
            let (mut cp, mut cm) = (cam, cam);
            cp.integrate(&tw, dt);
            cm.integrate(&tw, -dt);
            let (xp, yp, _) = cp.project(&p).unwrap();
            let (xm, ym, _) = cm.project(&p).unwrap();
            let sdot_num = ((xp - xm) / (2.0 * dt), (yp - ym) / (2.0 * dt));
            assert!((l[(0, k)] - sdot_num.0).abs() < 1e-4, "L[0,{k}]: {} vs fd {}", l[(0, k)], sdot_num.0);
            assert!((l[(1, k)] - sdot_num.1).abs() < 1e-4, "L[1,{k}]: {} vs fd {}", l[(1, k)], sdot_num.1);
        }
    }

    #[test]
    fn ibvs_converges_to_the_desired_view_and_pose() {
        let pts = square();
        let goal = goal_cam();
        let desired: Vec<(f64, f64)> = pts.iter().map(|p| { let (x, y, _) = goal.project(p).unwrap(); (x, y) }).collect();

        // Start from a wrong pose (translated and rotated).
        let mut cam = Camera { r: exp_so3(Vector3::new(0.08, -0.12, 0.2)), t: Vector3::new(0.12, -0.07, -0.62) };
        let err0: f64 = (cam.t - goal.t).norm();
        let (lambda, dt) = (2.0, 0.01);
        for _ in 0..2000 {
            let Some(tw) = ibvs_twist(&cam, &pts, &desired, lambda) else { break };
            cam.integrate(&tw, dt);
        }
        // Features converged …
        let feat_err: f64 = pts
            .iter()
            .zip(&desired)
            .map(|(p, d)| { let (x, y, _) = cam.project(p).unwrap(); ((x - d.0).powi(2) + (y - d.1).powi(2)).sqrt() })
            .fold(0.0, f64::max);
        assert!(feat_err < 1e-4, "features did not converge: {feat_err:.2e}");
        // … and since four non-degenerate points fix the pose, the camera pose converged too.
        assert!((cam.t - goal.t).norm() < 1e-3, "pose did not converge: {:?} vs {:?}", cam.t, goal.t);
        assert!((cam.r - goal.r).norm() < 1e-3, "orientation did not converge");
        assert!(err0 > 0.1, "test started too close to the goal to be meaningful");
    }

    #[test]
    fn feature_error_decays_exponentially_at_the_gain_rate() {
        // v_c = −λL⁺e ⇒ ė ≈ −λe, so ‖e‖ should follow e^{−λt}.
        let pts = square();
        let goal = goal_cam();
        let desired: Vec<(f64, f64)> = pts.iter().map(|p| { let (x, y, _) = goal.project(p).unwrap(); (x, y) }).collect();
        let mut cam = Camera { r: Matrix3::identity(), t: Vector3::new(0.06, -0.04, -0.56) };
        let norm_e = |c: &Camera| -> f64 {
            pts.iter().zip(&desired).map(|(p, d)| { let (x, y, _) = c.project(p).unwrap(); (x - d.0).powi(2) + (y - d.1).powi(2) }).sum::<f64>().sqrt()
        };
        let (lambda, dt) = (1.5, 0.005);
        let e0 = norm_e(&cam);
        for k in 0..400 {
            let t = k as f64 * dt;
            let predicted = e0 * (-lambda * t).exp();
            let actual = norm_e(&cam);
            assert!((actual - predicted).abs() < 0.15 * e0, "error envelope off at t={t:.2}: {actual:.4} vs {predicted:.4}");
            let tw = ibvs_twist(&cam, &pts, &desired, lambda).unwrap();
            cam.integrate(&tw, dt);
        }
    }
}
