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

// the Zhang constraint row v_ij from a homography's columns i, j (0-indexed)
fn zhang_v(h: &nalgebra::Matrix3<f64>, i: usize, j: usize) -> nalgebra::SVector<f64, 6> {
    nalgebra::SVector::<f64, 6>::from_row_slice(&[
        h[(0, i)] * h[(0, j)],
        h[(0, i)] * h[(1, j)] + h[(1, i)] * h[(0, j)],
        h[(1, i)] * h[(1, j)],
        h[(2, i)] * h[(0, j)] + h[(0, i)] * h[(2, j)],
        h[(2, i)] * h[(1, j)] + h[(1, i)] * h[(2, j)],
        h[(2, i)] * h[(2, j)],
    ])
}

/// One calibration view: the planar target's object points (on `Z = 0`) paired with their observed pixels.
pub type CalibrationView = (Vec<Vector2<f64>>, Vec<Vector2<f64>>);

/// **Zhang's camera calibration** (Zhang, 2000): recover a pinhole's intrinsics `(fx, fy, cx, cy)` from
/// `≥ 3` views of a **planar** target (a checkerboard on the `Z = 0` plane). Each view gives a plane→image
/// homography (via [`crate::homography_dlt`]); each homography imposes two linear constraints on the image of
/// the absolute conic `B = K⁻ᵀK⁻¹`, and stacking them over views solves for `B` (SVD null space), from which
/// the intrinsics follow in closed form. Distortion is not estimated here (linear method); pass to a
/// nonlinear refinement for that. `views[k] = (planar object points on Z=0, their pixels)`.
pub fn calibrate(views: &[CalibrationView]) -> Option<PinholeCamera> {
    use nalgebra::DMatrix;
    if views.len() < 3 {
        return None;
    }
    let mut a = DMatrix::zeros(2 * views.len(), 6);
    for (k, (obj, img)) in views.iter().enumerate() {
        let h = crate::homography::homography_dlt(obj, img);
        let v01 = zhang_v(&h, 0, 1);
        let v00 = zhang_v(&h, 0, 0);
        let v11 = zhang_v(&h, 1, 1);
        let d = v00 - v11;
        for c in 0..6 {
            a[(2 * k, c)] = v01[c];
            a[(2 * k + 1, c)] = d[c];
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let mut b: Vec<f64> = (0..6).map(|c| vt[(5, c)]).collect();
    if b[0] < 0.0 {
        b.iter_mut().for_each(|x| *x = -*x); // fix sign so B11 > 0
    }
    let (b11, b12, b22, b13, b23, b33) = (b[0], b[1], b[2], b[3], b[4], b[5]);
    let denom = b11 * b22 - b12 * b12;
    if denom.abs() < 1e-15 {
        return None;
    }
    let cy = (b12 * b13 - b11 * b23) / denom;
    let lambda = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11;
    if lambda / b11 <= 0.0 || lambda * b11 / denom <= 0.0 {
        return None;
    }
    let fx = (lambda / b11).sqrt();
    let fy = (lambda * b11 / denom).sqrt();
    let gamma = -b12 * fx * fx * fy / lambda;
    let cx = gamma * cy / fy - b13 * fx * fx / lambda;
    Some(PinholeCamera::new(fx, fy, cx, cy))
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

    #[test]
    fn zhang_calibration_recovers_known_intrinsics() {
        // THE ORACLE. Render a planar checkerboard from several poses with a known camera, then Zhang's
        // method must recover its intrinsics from the corner pixels alone.
        use super::calibrate;
        use nalgebra::{Matrix3, Vector3};
        let truth = PinholeCamera::new(500.0, 520.0, 320.0, 240.0);
        // a 7×7 planar grid of corners on Z=0, spaced 0.05 m
        let obj: Vec<Vector2<f64>> = (0..7).flat_map(|i| (0..7).map(move |j| Vector2::new(i as f64 * 0.05, j as f64 * 0.05))).collect();
        // small SO(3) exp for varied board orientations
        let so3 = |w: Vector3<f64>| -> Matrix3<f64> {
            let th = w.norm();
            if th < 1e-12 {
                return Matrix3::identity();
            }
            let k = w / th;
            let kx = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
            Matrix3::identity() + th.sin() * kx + (1.0 - th.cos()) * kx * kx
        };
        // six varied camera poses viewing the board ~0.5 m away
        let poses = [
            (Vector3::new(0.1, 0.0, 0.0), Vector3::new(-0.15, -0.15, 0.5)),
            (Vector3::new(-0.2, 0.15, 0.0), Vector3::new(-0.1, -0.2, 0.55)),
            (Vector3::new(0.15, -0.2, 0.1), Vector3::new(-0.2, -0.1, 0.45)),
            (Vector3::new(-0.1, -0.25, -0.1), Vector3::new(-0.15, -0.15, 0.5)),
            (Vector3::new(0.25, 0.1, 0.0), Vector3::new(-0.18, -0.12, 0.6)),
            (Vector3::new(0.0, 0.3, 0.15), Vector3::new(-0.12, -0.18, 0.5)),
        ];
        let views: Vec<super::CalibrationView> = poses
            .iter()
            .map(|&(w, t)| {
                let r = so3(w);
                let img: Vec<Vector2<f64>> = obj
                    .iter()
                    .map(|p| {
                        let pc = r * Vector3::new(p.x, p.y, 0.0) + t;
                        truth.project(&pc)
                    })
                    .collect();
                (obj.clone(), img)
            })
            .collect();
        let est = calibrate(&views).expect("calibration should succeed");
        assert!((est.fx - 500.0).abs() < 1.0, "fx {} vs 500", est.fx);
        assert!((est.fy - 520.0).abs() < 1.0, "fy {} vs 520", est.fy);
        assert!((est.cx - 320.0).abs() < 1.0, "cx {} vs 320", est.cx);
        assert!((est.cy - 240.0).abs() < 1.0, "cy {} vs 240", est.cy);
    }
}
