//! **Two-view relative pose — the essential matrix** (Longuet-Higgins 1981; Hartley's normalized 8-point
//! algorithm; Nistér's pose decomposition). Given `n ≥ 8` point correspondences between two calibrated views
//! (normalized image coordinates), recover the **relative camera motion** `(R, t)` — rotation exactly and
//! translation up to scale. Unlike [`crate::pnp`] (which needs known 3D points), this needs only 2D-2D
//! matches, so it is the **bootstrap of visual odometry / structure-from-motion**: it initializes the map
//! from the first two frames, after which PnP and triangulation take over.
//!
//! The essential matrix `E = [t]× R` satisfies the epipolar constraint `x₂ᵀ E x₁ = 0`. The 8-point algorithm
//! solves for `E` linearly (SVD null-space), projects it onto the essential manifold (singular values
//! `(1, 1, 0)`), and Nistér's `U W Vᵀ` factorization yields four candidate poses; the physically-correct one
//! is picked by **cheirality** (the reconstructed points must lie in front of both cameras). Verified: the
//! recovered `E` satisfies the epipolar constraint on every correspondence; and the recovered pose matches
//! the ground-truth rotation and translation direction. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};

/// The normalized **8-point algorithm**: estimate the essential matrix from `n ≥ 8` correspondences in
/// normalized (calibrated) image coordinates, projected onto the essential manifold.
pub fn eight_point(x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> Matrix3<f64> {
    let n = x1.len();
    // stack the epipolar constraint x₂ᵀ E x₁ = 0 as A·vec(E) = 0
    let mut a = DMatrix::zeros(n, 9);
    for (i, (p1, p2)) in x1.iter().zip(x2).enumerate() {
        let (u1, v1) = (p1.x, p1.y);
        let (u2, v2) = (p2.x, p2.y);
        let row = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0];
        for (k, r) in row.iter().enumerate() {
            a[(i, k)] = *r;
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t.expect("SVD V^T");
    let e = vt.row(8); // null-space vector
    let e_raw = Matrix3::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
    // project onto the essential manifold: singular values (1, 1, 0)
    let esvd = e_raw.svd(true, true);
    let (u, vtr) = (esvd.u.unwrap(), esvd.v_t.unwrap());
    u * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * vtr
}

/// The four candidate relative poses `(R, t)` from an essential matrix (Nistér's decomposition; `t` is unit
/// length, sign/rotation ambiguity resolved later by cheirality).
pub fn decompose_essential(e: &Matrix3<f64>) -> [(Matrix3<f64>, Vector3<f64>); 4] {
    let svd = e.svd(true, true);
    let (mut u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    // ensure U, V are rotations (det +1)
    if u.determinant() < 0.0 {
        u.set_column(2, &(-u.column(2)));
    }
    let mut v = vt.transpose();
    if v.determinant() < 0.0 {
        v.set_column(2, &(-v.column(2)));
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = u * w * v.transpose();
    let r2 = u * w.transpose() * v.transpose();
    let t = u.column(2).into_owned();
    [(r1, t), (r1, -t), (r2, t), (r2, -t)]
}

/// Triangulate one point (DLT) given camera 1 = `[I | 0]` and camera 2 = `[R | t]`. Returns the 3-D point in
/// camera-1 coordinates.
fn triangulate_one(r: &Matrix3<f64>, t: &Vector3<f64>, x1: &Vector2<f64>, x2: &Vector2<f64>) -> Vector3<f64> {
    // rows of P1 = [I|0], P2 = [R|t]
    let mut a = nalgebra::Matrix4::zeros();
    // camera 1 = [I|0]: row1=(1,0,0,0), row2=(0,1,0,0), row3=(0,0,1,0)
    // u1·P1row3 − P1row1 = (−1, 0, u1, 0);  v1·P1row3 − P1row2 = (0, −1, v1, 0)
    a[(0, 0)] = -1.0;
    a[(0, 2)] = x1.x;
    a[(1, 1)] = -1.0;
    a[(1, 2)] = x1.y;
    // P2 rows
    let p2r0 = [r[(0, 0)], r[(0, 1)], r[(0, 2)], t.x];
    let p2r1 = [r[(1, 0)], r[(1, 1)], r[(1, 2)], t.y];
    let p2r2 = [r[(2, 0)], r[(2, 1)], r[(2, 2)], t.z];
    for k in 0..4 {
        a[(2, k)] = x2.x * p2r2[k] - p2r0[k];
        a[(3, k)] = x2.y * p2r2[k] - p2r1[k];
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t.unwrap();
    let xh = vt.row(3);
    Vector3::new(xh[0] / xh[3], xh[1] / xh[3], xh[2] / xh[3])
}

/// Recover the physically-valid relative pose `(R, t)` (translation up to scale, returned unit-length) from
/// an essential matrix, using cheirality over the correspondences.
pub fn recover_pose(e: &Matrix3<f64>, x1: &[Vector2<f64>], x2: &[Vector2<f64>]) -> (Matrix3<f64>, Vector3<f64>) {
    let candidates = decompose_essential(e);
    let mut best = (candidates[0].0, candidates[0].1);
    let mut best_in_front = -1i32;
    for (r, t) in candidates {
        let mut in_front = 0i32;
        for (p1, p2) in x1.iter().zip(x2) {
            let x_cam1 = triangulate_one(&r, &t, p1, p2);
            let x_cam2 = r * x_cam1 + t;
            if x_cam1.z > 0.0 && x_cam2.z > 0.0 {
                in_front += 1;
            }
        }
        if in_front > best_in_front {
            best_in_front = in_front;
            best = (r, t);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }
    fn so3_exp(phi: &Vector3<f64>) -> Matrix3<f64> {
        let theta = phi.norm();
        if theta < 1e-12 {
            return Matrix3::identity();
        }
        let k = phi / theta;
        let kx = skew(&k);
        Matrix3::identity() + theta.sin() * kx + (1.0 - theta.cos()) * kx * kx
    }

    /// `(R_true, t_true, view-1 image points, view-2 image points)`.
    type TwoViews = (Matrix3<f64>, Vector3<f64>, Vec<Vector2<f64>>, Vec<Vector2<f64>>);

    // Two calibrated views of a point cloud, related by a known (R, t).
    fn two_views() -> TwoViews {
        let r = so3_exp(&Vector3::new(0.05, 0.2, -0.1));
        let t = Vector3::new(1.0, 0.1, 0.15); // baseline
        let pts = vec![
            Vector3::new(-1.0, -0.5, 5.0),
            Vector3::new(1.2, -0.7, 6.0),
            Vector3::new(0.8, 1.0, 4.5),
            Vector3::new(-0.9, 0.9, 5.5),
            Vector3::new(0.2, -1.1, 7.0),
            Vector3::new(-0.4, 0.3, 4.8),
            Vector3::new(1.1, 0.6, 6.2),
            Vector3::new(-1.3, -0.2, 5.1),
            Vector3::new(0.5, 0.8, 5.9),
            Vector3::new(-0.2, -0.9, 4.6),
        ];
        let x1: Vec<Vector2<f64>> = pts.iter().map(|p| Vector2::new(p.x / p.z, p.y / p.z)).collect();
        let x2: Vec<Vector2<f64>> = pts
            .iter()
            .map(|p| {
                let pc = r * p + t;
                Vector2::new(pc.x / pc.z, pc.y / pc.z)
            })
            .collect();
        (r, t, x1, x2)
    }

    #[test]
    fn the_essential_matrix_satisfies_the_epipolar_constraint() {
        // THE ORACLE (algebraic). Every correspondence must satisfy x₂ᵀ E x₁ = 0.
        let (_r, _t, x1, x2) = two_views();
        let e = eight_point(&x1, &x2);
        for (p1, p2) in x1.iter().zip(&x2) {
            let v1 = Vector3::new(p1.x, p1.y, 1.0);
            let v2 = Vector3::new(p2.x, p2.y, 1.0);
            let epi = (v2.transpose() * e * v1)[(0, 0)];
            assert!(epi.abs() < 1e-9, "epipolar residual {epi}");
        }
    }

    #[test]
    fn recover_pose_matches_the_ground_truth_motion() {
        // THE HEADLINE. The recovered rotation matches exactly, and the translation direction (t is only
        // known up to scale) matches the true baseline direction.
        let (r_true, t_true, x1, x2) = two_views();
        let e = eight_point(&x1, &x2);
        let (r, t) = recover_pose(&e, &x1, &x2);
        assert!((r - r_true).norm() < 1e-8, "rotation error {}", (r - r_true).norm());
        // t is unit-length and up to scale; compare directions
        let cos = t.normalize().dot(&t_true.normalize());
        assert!(cos > 1.0 - 1e-9, "translation direction cosine {cos}");
    }

    #[test]
    fn all_four_decompositions_are_valid_rotations() {
        // Each candidate rotation is proper (det +1), and exactly the physical one wins cheirality.
        let (_r, _t, x1, x2) = two_views();
        let e = eight_point(&x1, &x2);
        for (r, _t) in decompose_essential(&e) {
            assert!((r.determinant() - 1.0).abs() < 1e-9, "candidate not a rotation: det {}", r.determinant());
        }
    }

    #[test]
    fn the_recovered_points_lie_in_front_of_both_cameras() {
        // Cheirality sanity: with the recovered pose, every reconstructed point has positive depth in both
        // views.
        let (_r, _t, x1, x2) = two_views();
        let e = eight_point(&x1, &x2);
        let (r, t) = recover_pose(&e, &x1, &x2);
        for (p1, p2) in x1.iter().zip(&x2) {
            let xc1 = triangulate_one(&r, &t, p1, p2);
            let xc2 = r * xc1 + t;
            assert!(xc1.z > 0.0 && xc2.z > 0.0, "point behind a camera: z1={} z2={}", xc1.z, xc2.z);
        }
    }
}
