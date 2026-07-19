//! **Perspective-n-Point (PnP)** — recover a calibrated camera's 6-DoF pose `(R, t)` from `n ≥ 6` known
//! 3D world points and their 2D image projections. PnP is the geometric core of visual localization,
//! augmented reality, marker/AprilTag tracking, and the camera-resection step of visual SLAM & structure-
//! from-motion. Given normalized image coordinates `xᵢ` (pixel coordinates pre-multiplied by `K⁻¹`), it
//! finds `(R, t)` such that `xᵢ ≈ π(R Xᵢ + t)`, where `π(X, Y, Z) = (X/Z, Y/Z)`.
//!
//! Two stages: a linear **DLT** (Direct Linear Transform) recovers the projection matrix from the
//! correspondences by SVD and factors out `(R, t)` (rotation projected to `SO(3)`); then a **Gauss–Newton**
//! refinement minimizes the true reprojection error on the `SE(3)` manifold, which is what makes it robust
//! to measurement noise. Verified: from noise-free projections it recovers the exact pose (the oracle); the
//! reprojection error at the solution is ~0; and under pixel noise the GN refinement lowers the error below
//! the DLT-only estimate. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, Matrix3, Vector2, Vector3, Vector6};

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// `SO(3)` exponential (Rodrigues): `exp([φ]×)`.
fn so3_exp(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();
    if theta < 1e-12 {
        return Matrix3::identity() + skew(phi);
    }
    let k = phi / theta;
    let kx = skew(&k);
    Matrix3::identity() + theta.sin() * kx + (1.0 - theta.cos()) * kx * kx
}

/// Root-mean-square reprojection error of pose `(R, t)` over the correspondences (normalized coords).
pub fn reprojection_error(r: &Matrix3<f64>, t: &Vector3<f64>, x3d: &[Vector3<f64>], x2d: &[Vector2<f64>]) -> f64 {
    let mut acc = 0.0;
    for (xw, xi) in x3d.iter().zip(x2d) {
        let pc = r * xw + t;
        let proj = Vector2::new(pc.x / pc.z, pc.y / pc.z);
        acc += (proj - xi).norm_squared();
    }
    (acc / x3d.len() as f64).sqrt()
}

/// **DLT PnP**: linear pose estimate from `n ≥ 6` correspondences (calibrated / normalized coordinates).
pub fn pnp_dlt(x3d: &[Vector3<f64>], x2d: &[Vector2<f64>]) -> (Matrix3<f64>, Vector3<f64>) {
    let n = x3d.len();
    // Build the 2n×12 DLT system A p = 0 with p = [m1; m2; m3] (rows of the 3×4 projection matrix M).
    let mut a = DMatrix::zeros(2 * n, 12);
    for (i, (xw, xi)) in x3d.iter().zip(x2d).enumerate() {
        let xt = [xw.x, xw.y, xw.z, 1.0];
        // u·(m3·X̃) − (m1·X̃) = 0
        for k in 0..4 {
            a[(2 * i, k)] = -xt[k];
            a[(2 * i, 8 + k)] = xi.x * xt[k];
            a[(2 * i + 1, 4 + k)] = -xt[k];
            a[(2 * i + 1, 8 + k)] = xi.y * xt[k];
        }
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t.expect("SVD V^T");
    let p = vt.row(11); // null-space vector (smallest singular value)
    let m = Matrix3::from_columns(&[
        Vector3::new(p[0], p[4], p[8]),  // column 0 of [R|t]
        Vector3::new(p[1], p[5], p[9]),  // column 1
        Vector3::new(p[2], p[6], p[10]), // column 2
    ]);
    let tcol = Vector3::new(p[3], p[7], p[11]);
    // scale = norm of the third row's rotational part (‖s·r3‖ = |s|)
    let s = Vector3::new(p[8], p[9], p[10]).norm();
    // cheirality: choose the sign so the points lie in front of the camera (positive depth)
    let depth_sum: f64 = x3d.iter().map(|xw| p[8] * xw.x + p[9] * xw.y + p[10] * xw.z + p[11]).sum();
    let sign = if depth_sum < 0.0 { -1.0 } else { 1.0 };
    let mn = m * (sign / s);
    let t = tcol * (sign / s);
    // project the (scaled) rotation block to the nearest proper rotation
    let rsvd = mn.svd(true, true);
    let (u, vtr) = (rsvd.u.unwrap(), rsvd.v_t.unwrap());
    let mut r = u * vtr;
    if r.determinant() < 0.0 {
        let mut uu = u;
        uu.set_column(2, &(-u.column(2)));
        r = uu * vtr;
    }
    (r, t)
}

/// **Gauss–Newton refinement** of a pose `(R, t)` minimizing the reprojection error on `SE(3)`.
pub fn pnp_gn(x3d: &[Vector3<f64>], x2d: &[Vector2<f64>], r0: Matrix3<f64>, t0: Vector3<f64>, iters: usize) -> (Matrix3<f64>, Vector3<f64>) {
    let mut r = r0;
    let mut t = t0;
    for _ in 0..iters {
        let mut h = nalgebra::Matrix6::zeros();
        let mut g = Vector6::zeros();
        for (xw, xi) in x3d.iter().zip(x2d) {
            let pc = r * xw + t;
            let (xc, yc, zc) = (pc.x, pc.y, pc.z);
            if zc.abs() < 1e-9 {
                continue;
            }
            let proj = Vector2::new(xc / zc, yc / zc);
            let res = proj - xi;
            // ∂proj/∂P_c (2×3)
            let dpi = nalgebra::Matrix2x3::new(1.0 / zc, 0.0, -xc / (zc * zc), 0.0, 1.0 / zc, -yc / (zc * zc));
            // ∂P_c/∂ξ = [I | -[P_c]×]  (3×6), left-perturbation ξ = (ρ, φ)
            let mut dpc = nalgebra::Matrix3x6::zeros();
            dpc.fixed_view_mut::<3, 3>(0, 0).copy_from(&Matrix3::identity());
            dpc.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&pc)));
            let j = dpi * dpc; // 2×6
            h += j.transpose() * j;
            g += j.transpose() * res;
        }
        let Some(delta) = (h + nalgebra::Matrix6::identity() * 1e-9).lu().solve(&(-g)) else { break };
        let rho = Vector3::new(delta[0], delta[1], delta[2]);
        let phi = Vector3::new(delta[3], delta[4], delta[5]);
        let dr = so3_exp(&phi);
        r = dr * r;
        t = dr * t + rho;
        if delta.norm() < 1e-12 {
            break;
        }
    }
    (r, t)
}

/// Full PnP: DLT initialization followed by Gauss–Newton refinement.
pub fn pnp(x3d: &[Vector3<f64>], x2d: &[Vector2<f64>], iters: usize) -> (Matrix3<f64>, Vector3<f64>) {
    let (r0, t0) = pnp_dlt(x3d, x2d);
    pnp_gn(x3d, x2d, r0, t0, iters)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(R_true, t_true, world points, normalized image points)`.
    type Scene = (Matrix3<f64>, Vector3<f64>, Vec<Vector3<f64>>, Vec<Vector2<f64>>);

    // A fixed "true" pose and a cloud of world points, projected to normalized image coords.
    fn scene() -> Scene {
        let r_true = so3_exp(&Vector3::new(0.2, -0.35, 0.15));
        let t_true = Vector3::new(0.3, -0.2, 4.0); // camera looking down +z, points ~4 m away
        let pts = vec![
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.2),
            Vector3::new(1.0, 1.0, -0.1),
            Vector3::new(-1.0, 1.0, 0.3),
            Vector3::new(0.5, -0.5, 1.0),
            Vector3::new(-0.6, 0.7, -0.5),
            Vector3::new(0.2, 0.1, 0.6),
            Vector3::new(-0.3, -0.8, 0.4),
        ];
        let img: Vec<Vector2<f64>> = pts
            .iter()
            .map(|xw| {
                let pc = r_true * xw + t_true;
                Vector2::new(pc.x / pc.z, pc.y / pc.z)
            })
            .collect();
        (r_true, t_true, pts, img)
    }

    #[test]
    fn dlt_recovers_the_exact_pose_from_clean_projections() {
        // THE ORACLE. Noise-free projections ⇒ the DLT recovers the true pose.
        let (r_true, t_true, pts, img) = scene();
        let (r, t) = pnp_dlt(&pts, &img);
        assert!((r - r_true).norm() < 1e-8, "rotation error {}", (r - r_true).norm());
        assert!((t - t_true).norm() < 1e-8, "translation error {}", (t - t_true).norm());
    }

    #[test]
    fn the_reprojection_error_is_zero_at_the_solution() {
        let (_r_true, _t_true, pts, img) = scene();
        let (r, t) = pnp(&pts, &img, 10);
        assert!(reprojection_error(&r, &t, &pts, &img) < 1e-10, "reprojection error should vanish");
    }

    #[test]
    fn gauss_newton_refines_a_perturbed_pose_back_to_truth() {
        // Starting from a pose perturbed away from the truth, GN on the clean projections returns to it —
        // exercising the SE(3) update independent of the DLT.
        let (r_true, t_true, pts, img) = scene();
        let r0 = so3_exp(&Vector3::new(0.1, 0.0, 0.0)) * r_true;
        let t0 = t_true + Vector3::new(0.2, -0.15, 0.3);
        let (r, t) = pnp_gn(&pts, &img, r0, t0, 30);
        assert!((r - r_true).norm() < 1e-6 && (t - t_true).norm() < 1e-6, "GN should converge to truth: R {} t {}", (r - r_true).norm(), (t - t_true).norm());
    }

    #[test]
    fn refinement_beats_dlt_under_pixel_noise() {
        // THE HEADLINE. With deterministic noise on the image points, GN refinement yields a lower
        // reprojection error (and pose error) than the linear DLT alone.
        let (r_true, t_true, pts, img) = scene();
        let noise = [0.004, -0.003, 0.005, -0.002, 0.0035, -0.0045, 0.0025, -0.005, 0.003, 0.004, -0.0035, 0.0045, 0.002, -0.003, 0.0038, -0.0042];
        let noisy: Vec<Vector2<f64>> = img.iter().enumerate().map(|(i, x)| x + Vector2::new(noise[2 * i], noise[2 * i + 1])).collect();
        let (rd, td) = pnp_dlt(&pts, &noisy);
        let (rr, tr) = pnp_gn(&pts, &noisy, rd, td, 30);
        let err_dlt = (rd - r_true).norm() + (td - t_true).norm();
        let err_ref = (rr - r_true).norm() + (tr - t_true).norm();
        assert!(err_ref < err_dlt, "GN refinement should beat DLT: {err_ref} vs {err_dlt}");
    }
}
