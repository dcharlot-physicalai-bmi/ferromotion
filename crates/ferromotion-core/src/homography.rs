//! **Homography estimation** (the normalized DLT). A homography `H` is the projective `3×3` map between two
//! images of a **planar** scene, or between two views related by **pure rotation** — `x₂ ≃ H x₁` in
//! homogeneous coordinates. It is the model behind planar augmented reality, image stitching / panoramas,
//! marker rectification, and planar visual odometry. It complements [`crate::essential`] (general
//! non-planar motion): when the scene is planar or the camera only rotates, the essential-matrix estimate
//! degenerates and the homography is the correct model.
//!
//! `H` has 8 degrees of freedom (defined up to scale), so `n ≥ 4` correspondences determine it. Each pair
//! contributes two rows to a linear system solved by SVD null-space. Verified: the fitted homography maps
//! planar correspondences to machine precision (the oracle); for a pure rotation it recovers `H ∝ R`; four
//! points suffice; and wrapped in [`crate::ransac`] it is robust to mismatched correspondences. Pure
//! `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};

/// Estimate the homography `H` (with `x₂ ≃ H x₁`) from `n ≥ 4` correspondences by the direct linear
/// transform. Coordinates are used directly; pre-normalize (Hartley) externally if conditioning matters.
pub fn homography_dlt(x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> Matrix3<f64> {
    let n = x1.len();
    // 2 rows per correspondence from x₂ × (H x₁) = 0; allocate ≥9 rows so the thin SVD exposes the full 9×9
    // Vᵀ even with exactly 4 points (2n = 8).
    let mut a = DMatrix::zeros((2 * n).max(9), 9);
    for (i, (p1, p2)) in x1.iter().zip(x2).enumerate() {
        let (u1, v1) = (p1.x, p1.y);
        let (u2, v2) = (p2.x, p2.y);
        // row for the x-component: [-u1,-v1,-1, 0,0,0, u2u1,u2v1,u2]
        let r0 = [-u1, -v1, -1.0, 0.0, 0.0, 0.0, u2 * u1, u2 * v1, u2];
        // row for the y-component: [0,0,0, -u1,-v1,-1, v2u1,v2v1,v2]
        let r1 = [0.0, 0.0, 0.0, -u1, -v1, -1.0, v2 * u1, v2 * v1, v2];
        for k in 0..9 {
            a[(2 * i, k)] = r0[k];
            a[(2 * i + 1, k)] = r1[k];
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t.expect("SVD V^T");
    let h = vt.row(8); // null-space vector
    Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8])
}

/// Apply a homography to an image point: `H x` in homogeneous coordinates, dehomogenized.
pub fn apply_homography(h: &Matrix3<f64>, x: &Vector2<f64>) -> Vector2<f64> {
    let v = h * Vector3::new(x.x, x.y, 1.0);
    Vector2::new(v.x / v.z, v.y / v.z)
}

/// Symmetric-ish transfer error: RMS reprojection of `x₁` through `H` onto `x₂`.
pub fn transfer_error(h: &Matrix3<f64>, x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> f64 {
    let acc: f64 = x1.iter().zip(x2).map(|(a, b)| (apply_homography(h, a) - b).norm_squared()).sum();
    (acc / x1.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }
    fn so3(phi: &Vector3<f64>) -> Matrix3<f64> {
        let th = phi.norm();
        if th < 1e-12 {
            return Matrix3::identity();
        }
        let k = phi / th;
        let kx = skew(&k);
        Matrix3::identity() + th.sin() * kx + (1.0 - th.cos()) * kx * kx
    }

    #[test]
    fn it_maps_planar_correspondences_exactly() {
        // THE ORACLE. Coplanar 3-D points (z = 0 plane) seen from two cameras induce a homography between the
        // views; the fitted H must map view-1 points onto view-2 points to machine precision.
        let r = so3(&Vector3::new(0.05, 0.1, -0.08));
        let t = Vector3::new(0.6, 0.1, 0.2);
        // planar scene: points on z=0, lifted to camera 1 at distance ~5 by an offset
        let plane_pts = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.3, -0.6), (-0.7, 0.4), (0.5, 0.9)];
        let cam1_offset = Vector3::new(0.0, 0.0, 5.0);
        let x1: Vec<Vector2<f64>> = plane_pts.iter().map(|&(a, b)| { let p = Vector3::new(a, b, 0.0) + cam1_offset; Vector2::new(p.x / p.z, p.y / p.z) }).collect();
        let x2: Vec<Vector2<f64>> = plane_pts.iter().map(|&(a, b)| { let p = r * (Vector3::new(a, b, 0.0) + cam1_offset) + t; Vector2::new(p.x / p.z, p.y / p.z) }).collect();
        let h = homography_dlt(&x1, &x2);
        assert!(transfer_error(&h, &x1, &x2) < 1e-9, "homography should map planar points exactly: {}", transfer_error(&h, &x1, &x2));
    }

    #[test]
    fn pure_rotation_recovers_the_rotation_homography() {
        // With zero translation the induced homography is H ∝ R for ANY points (the plane-at-infinity
        // homography). Recover it and confirm the normalized H equals R.
        let r = so3(&Vector3::new(0.1, -0.2, 0.15));
        let pts = [Vector3::new(-1.0, -0.5, 5.0), Vector3::new(1.2, -0.7, 6.0), Vector3::new(0.8, 1.0, 4.5), Vector3::new(-0.9, 0.9, 5.5), Vector3::new(0.2, -1.1, 7.0)];
        let x1: Vec<Vector2<f64>> = pts.iter().map(|p| Vector2::new(p.x / p.z, p.y / p.z)).collect();
        let x2: Vec<Vector2<f64>> = pts.iter().map(|p| { let pc = r * p; Vector2::new(pc.x / pc.z, pc.y / pc.z) }).collect();
        let h = homography_dlt(&x1, &x2);
        // scale so det = 1 (H = s R ⇒ det = s³), then fix sign
        let s = h.determinant().cbrt();
        let mut hn = h / s;
        if (hn - r).norm() > ((-hn) - r).norm() {
            hn = -hn;
        }
        assert!((hn - r).norm() < 1e-8, "pure-rotation homography should equal R: err {}", (hn - r).norm());
    }

    #[test]
    fn four_points_are_enough() {
        // The minimal case: 4 correspondences (2n = 8 rows, padded to 9) still yield a valid homography.
        let h_true = Matrix3::new(1.2, 0.1, 0.3, -0.05, 0.9, -0.2, 0.001, -0.002, 1.0);
        let src = [Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0), Vector2::new(1.0, 1.0), Vector2::new(0.0, 1.0)];
        let dst: Vec<Vector2<f64>> = src.iter().map(|x| apply_homography(&h_true, x)).collect();
        let h = homography_dlt(&src, &dst);
        assert!(transfer_error(&h, &src, &dst) < 1e-9, "4-point homography should be exact: {}", transfer_error(&h, &src, &dst));
    }

    #[test]
    fn ransac_makes_it_robust_to_mismatches() {
        // THE HEADLINE. Planar correspondences with 6 of 30 wrong; RANSAC over 4-point homography samples
        // recovers the true map and flags the inliers.
        use crate::ransac::ransac;
        let h_true = Matrix3::new(1.1, 0.08, 0.4, -0.06, 1.05, -0.25, 0.002, -0.001, 1.0);
        let mut seed = 7u64;
        let ur = |s: &mut u64| { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((*s >> 33) as f64 / (1u64 << 31) as f64) - 1.0 };
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for _ in 0..30 {
            let p = Vector2::new(2.0 * ur(&mut seed), 2.0 * ur(&mut seed));
            src.push(p);
            dst.push(apply_homography(&h_true, &p));
        }
        for slot in dst.iter_mut().take(6) {
            *slot = Vector2::new(3.0 * ur(&mut seed), 3.0 * ur(&mut seed)); // corrupt
        }
        let (s1, d1) = (src.clone(), dst.clone());
        let fit = move |idx: &[usize]| -> Option<Matrix3<f64>> {
            let a: Vec<Vector2<f64>> = idx.iter().map(|&i| s1[i]).collect();
            let b: Vec<Vector2<f64>> = idx.iter().map(|&i| d1[i]).collect();
            Some(homography_dlt(&a, &b))
        };
        let (s2, d2) = (src.clone(), dst.clone());
        let is_inlier = move |h: &Matrix3<f64>, i: usize| (apply_homography(h, &s2[i]) - d2[i]).norm() < 1e-3;
        let res = ransac(30, 4, 500, 1, fit, is_inlier).expect("robust H found");
        assert!(res.n_inliers >= 22, "should keep the ~24 good matches: {}", res.n_inliers);
        assert!(res.inliers[6..].iter().filter(|&&b| b).count() >= 22, "true inliers flagged");
    }
}
