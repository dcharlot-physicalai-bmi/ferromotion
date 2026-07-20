//! **Pinhole camera model with radial-tangential distortion** (Brown–Conrady). This is the entry point to
//! the crate's geometric-vision stack: [`crate::pnp`], [`crate::essential`], [`crate::homography`], and
//! [`crate::bundle`] all operate on *calibrated, normalized* image coordinates — this model is what converts
//! raw **pixels** to and from those normalized rays, applying the intrinsics `(fx, fy, cx, cy)` and the lens
//! distortion `(k1, k2, p1, p2, k3)` that every real camera has. `project` maps a 3-D camera-frame point to a
//! pixel; `unproject` inverts it (distortion removed by fixed-point iteration) to a bearing ray.
//!
//! Verified: a point on the optical axis lands on the principal point; `project`∘`unproject` round-trips to
//! machine precision without distortion and to a tight tolerance with it (the iterative undistort); and the
//! pinhole projection Jacobian matches a finite difference (the block bundle adjustment needs for real
//! cameras). Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix2x3, Vector2, Vector3};

/// A pinhole camera: focal lengths, principal point, and Brown–Conrady distortion `[k1, k2, p1, p2, k3]`.
#[derive(Clone, Copy, Debug)]
pub struct PinholeCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub dist: [f64; 5],
}

impl PinholeCamera {
    /// A distortion-free pinhole.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        PinholeCamera { fx, fy, cx, cy, dist: [0.0; 5] }
    }

    /// Apply radial-tangential distortion to a normalized image point `(x, y)`.
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let [k1, k2, p1, p2, k3] = self.dist;
        let r2 = x * x + y * y;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        (xd, yd)
    }

    /// Remove distortion from a distorted normalized point `(xd, yd)` by fixed-point iteration.
    fn undistort(&self, xd: f64, yd: f64) -> (f64, f64) {
        let [k1, k2, p1, p2, k3] = self.dist;
        let (mut x, mut y) = (xd, yd);
        for _ in 0..30 {
            let r2 = x * x + y * y;
            let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            x = (xd - dx) / radial;
            y = (yd - dy) / radial;
        }
        (x, y)
    }

    /// Project a 3-D point in the camera frame to a pixel (`z > 0` required).
    pub fn project(&self, p: &Vector3<f64>) -> Vector2<f64> {
        let (xn, yn) = (p.x / p.z, p.y / p.z);
        let (xd, yd) = self.distort(xn, yn);
        Vector2::new(self.fx * xd + self.cx, self.fy * yd + self.cy)
    }

    /// Unproject a pixel to a normalized bearing ray `(x, y, 1)` in the camera frame (distortion removed).
    pub fn unproject(&self, px: &Vector2<f64>) -> Vector3<f64> {
        let xd = (px.x - self.cx) / self.fx;
        let yd = (px.y - self.cy) / self.fy;
        let (xn, yn) = self.undistort(xd, yd);
        Vector3::new(xn, yn, 1.0)
    }

    /// The pinhole projection Jacobian `∂pixel/∂point` (2×3), for the undistorted model — the block bundle
    /// adjustment / PnP needs when a real camera's pixels feed the reprojection residual.
    pub fn project_jacobian(&self, p: &Vector3<f64>) -> Matrix2x3<f64> {
        let z = p.z;
        Matrix2x3::new(
            self.fx / z,
            0.0,
            -self.fx * p.x / (z * z),
            0.0,
            self.fy / z,
            -self.fy * p.y / (z * z),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> PinholeCamera {
        PinholeCamera { fx: 520.0, fy: 515.0, cx: 320.0, cy: 240.0, dist: [-0.28, 0.09, 0.001, -0.0008, 0.0] }
    }

    #[test]
    fn an_axial_point_lands_on_the_principal_point() {
        // A point straight down the optical axis projects to (cx, cy) regardless of distortion.
        let c = cam();
        let px = c.project(&Vector3::new(0.0, 0.0, 3.0));
        assert!((px - Vector2::new(320.0, 240.0)).norm() < 1e-9, "axial point → principal point: {px}");
    }

    #[test]
    fn project_then_unproject_round_trips_without_distortion() {
        // THE ORACLE. With no distortion the pixel↔ray maps are exact inverses.
        let c = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        for p in [Vector3::new(0.3, -0.2, 2.0), Vector3::new(-0.5, 0.4, 5.0), Vector3::new(0.1, 0.7, 1.5)] {
            let px = c.project(&p);
            let ray = c.unproject(&px);
            // the ray is the point scaled to z=1
            assert!((ray - p / p.z).norm() < 1e-12, "round trip failed: {ray} vs {}", p / p.z);
        }
    }

    #[test]
    fn distortion_and_undistortion_are_inverse_to_tolerance() {
        // With real distortion, unproject∘project recovers the bearing via the iterative undistort.
        let c = cam();
        for p in [Vector3::new(0.4, -0.3, 2.0), Vector3::new(-0.6, 0.5, 3.0), Vector3::new(0.2, 0.6, 1.8)] {
            let px = c.project(&p);
            let ray = c.unproject(&px);
            assert!((ray - p / p.z).norm() < 1e-8, "distorted round trip: {ray} vs {}", p / p.z);
        }
    }

    #[test]
    fn the_projection_jacobian_matches_a_finite_difference() {
        let c = PinholeCamera::new(520.0, 515.0, 320.0, 240.0);
        let p = Vector3::new(0.3, -0.2, 2.5);
        let j = c.project_jacobian(&p);
        let h = 1e-6;
        for col in 0..3 {
            let mut pp = p;
            pp[col] += h;
            let mut pm = p;
            pm[col] -= h;
            let fd = (c.project(&pp) - c.project(&pm)) / (2.0 * h);
            assert!((j.column(col) - fd).norm() < 1e-3, "Jacobian column {col}: {} vs {fd}", j.column(col));
        }
    }

    #[test]
    fn a_known_point_projects_to_the_expected_pixel() {
        // Distortion-free: pixel = (fx·x/z + cx, fy·y/z + cy).
        let c = PinholeCamera::new(500.0, 500.0, 320.0, 240.0);
        let px = c.project(&Vector3::new(1.0, 0.5, 2.0));
        // x/z = 0.5 → 500·0.5 + 320 = 570 ; y/z = 0.25 → 500·0.25 + 240 = 365
        assert!((px - Vector2::new(570.0, 365.0)).norm() < 1e-9, "pixel {px}");
    }
}
