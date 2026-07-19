//! **ICP — Iterative Closest Point** point-cloud registration (Besl & McKay 1992 point-to-point via
//! Arun/Umeyama; Chen & Medioni 1991 point-to-plane): align two point clouds by alternating
//! nearest-neighbour correspondence with a rigid-transform solve, until they lock together. This is the
//! geometric-perception counterpart to [`crate::Msckf`]'s visual odometry — the workhorse behind LiDAR/
//! depth scan-matching and mapping.
//!
//! Two variants: **point-to-point** (closed-form SVD alignment of matched points — the Umeyama solution)
//! and **point-to-plane** (minimize the distance along the *target's surface normal*, which converges far
//! faster on structured scenes — a 6-DOF linearized least-squares per iteration). Verified by recovering a
//! known SE(3) transform from a perturbed cloud. Pure `nalgebra` → WASM-clean.
//!
//! (Nearest neighbours are brute-force here; a k-d tree is the standard acceleration for large clouds.)

use crate::screw::exp_so3;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// The result of a registration: the rigid transform aligning `source` onto `target` (`q ≈ R·p + t`), the
/// per-correspondence RMSE, and the iteration count.
#[derive(Clone, Debug)]
pub struct IcpResult {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// **Umeyama / Arun** closed-form rigid alignment: the `(R, t)` minimizing `Σ‖(R·src_i + t) − dst_i‖²` for
/// *given* correspondences, via the SVD of the cross-covariance (reflection-free).
pub fn umeyama(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> (Matrix3<f64>, Vector3<f64>) {
    let n = src.len() as f64;
    let mu_s: Vector3<f64> = src.iter().sum::<Vector3<f64>>() / n;
    let mu_d: Vector3<f64> = dst.iter().sum::<Vector3<f64>>() / n;
    let mut h = Matrix3::zeros();
    for (s, d) in src.iter().zip(dst) {
        h += (d - mu_d) * (s - mu_s).transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    // R = U · diag(1,1,det(U·Vᵀ)) · Vᵀ  (avoids a reflection)
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let r = u * d * vt;
    let t = mu_d - r * mu_s;
    (r, t)
}

/// Brute-force index of the nearest target point to `p`.
fn nearest(p: &Vector3<f64>, target: &[Vector3<f64>]) -> usize {
    target
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - p).norm_squared().partial_cmp(&(b.1 - p).norm_squared()).unwrap())
        .unwrap()
        .0
}

/// Registration settings.
#[derive(Clone, Copy, Debug)]
pub struct Icp {
    pub max_iters: usize,
    /// Convergence tolerance on the incremental motion (‖δθ,δt‖).
    pub tol: f64,
}

impl Default for Icp {
    fn default() -> Self {
        Icp { max_iters: 50, tol: 1e-8 }
    }
}

impl Icp {
    /// **Point-to-point ICP**: alternate nearest-neighbour matching with the closed-form Umeyama solve.
    pub fn point_to_point(&self, source: &[Vector3<f64>], target: &[Vector3<f64>]) -> IcpResult {
        let (mut r, mut t) = (Matrix3::identity(), Vector3::zeros());
        let mut converged = false;
        let mut iterations = 0;
        for _ in 0..self.max_iters {
            iterations += 1;
            let moved: Vec<Vector3<f64>> = source.iter().map(|p| r * p + t).collect();
            let matched: Vec<Vector3<f64>> = moved.iter().map(|p| target[nearest(p, target)]).collect();
            let (dr, dt) = umeyama(&moved, &matched); // incremental correction on the moved cloud
            r = dr * r;
            t = dr * t + dt;
            // incremental motion size
            let motion = crate::screw::log_so3(&dr).norm() + dt.norm();
            if motion < self.tol {
                converged = true;
                break;
            }
        }
        IcpResult { rmse: rmse(source, target, &r, &t), rotation: r, translation: t, iterations, converged }
    }

    /// **Point-to-plane ICP**: minimize the residual along each matched target point's surface `normals`,
    /// a 6-DOF linearized least-squares (`R ≈ I + [δθ]`) per iteration — faster on structured scenes.
    pub fn point_to_plane(&self, source: &[Vector3<f64>], target: &[Vector3<f64>], normals: &[Vector3<f64>]) -> IcpResult {
        let (mut r, mut t) = (Matrix3::identity(), Vector3::zeros());
        let mut converged = false;
        let mut iterations = 0;
        for _ in 0..self.max_iters {
            iterations += 1;
            // build the 6×6 normal equations for [δθ; δt]
            let mut a = DMatrix::<f64>::zeros(6, 6);
            let mut b = DVector::<f64>::zeros(6);
            for p in source {
                let pw = r * p + t;
                let j = nearest(&pw, target);
                let (q, n) = (target[j], normals[j]);
                // r_i = n·(pw − q) + (pw×n)·δθ + n·δt  → row = [(pw×n)ᵀ, nᵀ], rhs = −n·(pw−q)
                let cross = pw.cross(&n);
                let mut row = DVector::<f64>::zeros(6);
                row.fixed_rows_mut::<3>(0).copy_from(&cross);
                row.fixed_rows_mut::<3>(3).copy_from(&n);
                let e = n.dot(&(pw - q));
                a += &row * row.transpose();
                b -= &row * e;
            }
            let Some(x) = a.lu().solve(&b) else { break };
            let dth = Vector3::new(x[0], x[1], x[2]);
            let dt = Vector3::new(x[3], x[4], x[5]);
            let dr = exp_so3(&dth);
            r = dr * r;
            t = dr * t + dt;
            if dth.norm() + dt.norm() < self.tol {
                converged = true;
                break;
            }
        }
        IcpResult { rmse: rmse(source, target, &r, &t), rotation: r, translation: t, iterations, converged }
    }
}

/// Root-mean-square nearest-neighbour distance of the transformed source to the target.
fn rmse(source: &[Vector3<f64>], target: &[Vector3<f64>], r: &Matrix3<f64>, t: &Vector3<f64>) -> f64 {
    let s: f64 = source
        .iter()
        .map(|p| {
            let pw = r * p + t;
            (pw - target[nearest(&pw, target)]).norm_squared()
        })
        .sum();
    (s / source.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot(axis: Vector3<f64>, ang: f64) -> Matrix3<f64> {
        exp_so3(&(axis.normalize() * ang))
    }

    // A structured cloud: points on three mutually-perpendicular walls (so registration is well-constrained).
    fn walls() -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let mut pts = Vec::new();
        let mut nrm = Vec::new();
        let g = 6;
        for i in 0..g {
            for j in 0..g {
                let (a, b) = (i as f64 / g as f64, j as f64 / g as f64);
                pts.push(Vector3::new(0.0, a, b)); // x=0 wall
                nrm.push(Vector3::new(1.0, 0.0, 0.0));
                pts.push(Vector3::new(a, 0.0, b)); // y=0 wall
                nrm.push(Vector3::new(0.0, 1.0, 0.0));
                pts.push(Vector3::new(a, b, 0.0)); // z=0 floor
                nrm.push(Vector3::new(0.0, 0.0, 1.0));
            }
        }
        (pts, nrm)
    }

    #[test]
    fn umeyama_recovers_a_known_transform_exactly() {
        let (src, _) = walls();
        let r_true = rot(Vector3::new(0.3, -0.5, 0.8), 0.4);
        let t_true = Vector3::new(0.7, -0.2, 0.5);
        let dst: Vec<Vector3<f64>> = src.iter().map(|p| r_true * p + t_true).collect();
        let (r, t) = umeyama(&src, &dst);
        assert!((r - r_true).abs().max() < 1e-10, "rotation off: {}", (r - r_true).abs().max());
        assert!((t - t_true).norm() < 1e-10, "translation off: {}", (t - t_true).norm());
    }

    #[test]
    fn point_to_point_icp_recovers_a_small_transform() {
        // No correspondences given — ICP finds them by NN. A modest transform converges to the truth.
        let (src, _) = walls();
        let r_true = rot(Vector3::new(0.2, 0.3, 1.0), 0.12);
        let t_true = Vector3::new(0.06, -0.05, 0.04);
        let target: Vec<Vector3<f64>> = src.iter().map(|p| r_true * p + t_true).collect();
        let res = Icp::default().point_to_point(&src, &target);
        assert!(res.rmse < 1e-6, "should register to ~0 RMSE: {}", res.rmse);
        assert!((res.rotation - r_true).abs().max() < 1e-4 && (res.translation - t_true).norm() < 1e-4, "did not recover the transform");
    }

    #[test]
    fn point_to_plane_icp_recovers_the_transform() {
        // Point-to-plane minimizes distance along the target normals; on the structured wall scene it
        // recovers the transform to high accuracy in a handful of iterations.
        let (src, nrm) = walls();
        let r_true = rot(Vector3::new(0.1, 0.4, 0.9), 0.1);
        let t_true = Vector3::new(0.05, 0.04, -0.03);
        let target: Vec<Vector3<f64>> = src.iter().map(|p| r_true * p + t_true).collect();
        let target_normals: Vec<Vector3<f64>> = nrm.iter().map(|n| r_true * n).collect();
        let plane = Icp::default().point_to_plane(&src, &target, &target_normals);
        assert!(plane.rmse < 1e-6, "point-to-plane should register to ~0: {}", plane.rmse);
        assert!((plane.rotation - r_true).abs().max() < 1e-4 && (plane.translation - t_true).norm() < 1e-4, "did not recover the transform");
        assert!(plane.iterations <= 10, "should converge quickly on a structured scene: {}", plane.iterations);
    }

    #[test]
    fn a_perfect_alignment_is_a_fixed_point() {
        // If the clouds already coincide, ICP returns identity immediately.
        let (src, _) = walls();
        let res = Icp::default().point_to_point(&src, &src);
        assert!((res.rotation - Matrix3::identity()).abs().max() < 1e-9 && res.translation.norm() < 1e-9, "identity expected");
        assert!(res.rmse < 1e-12);
    }
}
