//! **Continuous-time SE(3) trajectory — cumulative cubic B-spline** (Kim, Kaess & Leonard 2016; Sommer et
//! al. 2020). A pose trajectory parameterized by control poses `Pᵢ ∈ SE(3)` on a uniform time grid, using
//! the **cumulative** B-spline form so the interpolation stays on the manifold:
//! `T(u) = Pⱼ · ∏ₖ exp( B̃ₖ(u) · log(P_{j+k−1}⁻¹ P_{j+k}) )`. Because it is smooth (C²) and has a *closed-form
//! body velocity*, it is the standard representation for **asynchronous / high-rate sensor fusion** — event
//! cameras, rolling-shutter and LiDAR de-skew, and targetless spatiotemporal calibration — where the
//! discrete-keyframe pose graph cannot represent motion *between* frames.
//!
//! Builds on the crate's [`crate::exp_se3`]/[`crate::log_se3`]/[`crate::adjoint`] Lie-group tools and the
//! Euclidean [`crate::BSpline`]. The analytic body twist `ξ = T⁻¹Ṫ` follows from differentiating the
//! cumulative product (each `exp` commutes with its own generator), giving
//! `ξ = Ad_{(A₂A₃)⁻¹} B̃₁′Ω₁ + Ad_{A₃⁻¹} B̃₂′Ω₂ + B̃₃′Ω₃`. Verified: the analytic velocity matches a finite
//! difference; the trajectory is C¹ across segment joins; and control poses equally spaced along a single
//! screw reproduce that screw at constant twist. Pure `nalgebra` → WASM-clean.

use crate::screw::{adjoint, exp_se3, log_se3};
use nalgebra::{Matrix4, Vector6};

/// A cubic cumulative B-spline over `≥ 4` SE(3) control poses (`4×4` homogeneous). The time domain is
/// `[0, k−3]` for `k` control poses (`k−3` segments, one time unit each).
#[derive(Clone, Debug)]
pub struct SplineSE3 {
    pub control: Vec<Matrix4<f64>>,
}

// cumulative cubic basis B̃₁,B̃₂,B̃₃ and their derivatives at local u ∈ [0,1]
fn basis(u: f64) -> ([f64; 3], [f64; 3]) {
    let (u2, u3) = (u * u, u * u * u);
    let b = [(5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0, (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0, u3 / 6.0];
    let db = [(1.0 - u) * (1.0 - u) / 2.0, (1.0 + 2.0 * u - 2.0 * u2) / 2.0, u2 / 2.0];
    (b, db)
}

impl SplineSE3 {
    pub fn segments(&self) -> usize {
        self.control.len().saturating_sub(3)
    }

    fn locate(&self, t: f64) -> (usize, f64) {
        let n = self.segments();
        let tt = t.clamp(0.0, n as f64 - 1e-12);
        let seg = (tt.floor() as usize).min(n - 1);
        (seg, tt - seg as f64)
    }

    // the three inter-control-pose increments Ωₖ = log(P_{j+k-1}⁻¹ P_{j+k}) for segment j
    fn omegas(&self, seg: usize) -> [Vector6<f64>; 3] {
        let p = &self.control;
        std::array::from_fn(|k| {
            let a = p[seg + k].try_inverse().unwrap() * p[seg + k + 1];
            log_se3(&a)
        })
    }

    /// The pose `T(t)` on the trajectory.
    pub fn pose(&self, t: f64) -> Matrix4<f64> {
        let (seg, u) = self.locate(t);
        let (b, _) = basis(u);
        let om = self.omegas(seg);
        self.control[seg] * exp_se3(&(om[0] * b[0])) * exp_se3(&(om[1] * b[1])) * exp_se3(&(om[2] * b[2]))
    }

    /// The **body twist** `ξ = T⁻¹Ṫ = [ω; v]` at `t` (angular first, per the crate's screw convention).
    pub fn body_velocity(&self, t: f64) -> Vector6<f64> {
        let (seg, u) = self.locate(t);
        let (b, db) = basis(u);
        let om = self.omegas(seg);
        let a2 = exp_se3(&(om[1] * b[1]));
        let a3 = exp_se3(&(om[2] * b[2]));
        let ad_23 = adjoint(&(a2 * a3).try_inverse().unwrap());
        let ad_3 = adjoint(&a3.try_inverse().unwrap());
        ad_23 * (om[0] * db[0]) + ad_3 * (om[1] * db[1]) + om[2] * db[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screw::{exp_so3, pose};
    use nalgebra::Vector3;

    fn ctrl(seed: f64) -> Matrix4<f64> {
        let w = Vector3::new(0.2 * seed, -0.4 + 0.1 * seed, 0.3 * seed);
        pose(&exp_so3(&w), &Vector3::new(seed, 0.5 * seed - 0.3, -0.2 * seed))
    }

    fn spline() -> SplineSE3 {
        SplineSE3 { control: (0..6).map(|i| ctrl(i as f64 * 0.5)).collect() }
    }

    // body twist by finite difference: vee(T⁻¹ (T(t+h)−T(t−h))/2h)
    fn fd_twist(s: &SplineSE3, t: f64) -> Vector6<f64> {
        let h = 1e-6;
        let tdot = (s.pose(t + h) - s.pose(t - h)) / (2.0 * h);
        let m = s.pose(t).try_inverse().unwrap() * tdot; // se(3) matrix
        Vector6::new(m[(2, 1)], m[(0, 2)], m[(1, 0)], m[(0, 3)], m[(1, 3)], m[(2, 3)])
    }

    #[test]
    fn the_analytic_body_velocity_matches_a_finite_difference() {
        // THE ORACLE. The closed-form twist equals the numerical T⁻¹Ṫ across the trajectory.
        let s = spline();
        for i in 1..30 {
            let t = i as f64 / 30.0 * s.segments() as f64 * 0.99;
            let (an, fd) = (s.body_velocity(t), fd_twist(&s, t));
            assert!((an - fd).norm() < 1e-4, "twist at t={t}: {an} vs fd {fd}");
        }
    }

    #[test]
    fn the_trajectory_is_c1_across_segment_joins() {
        let s = spline();
        for seg in 1..s.segments() {
            let t = seg as f64;
            let eps = 1e-6;
            // pose continuity
            assert!((s.pose(t - eps) - s.pose(t + eps)).abs().max() < 1e-4, "pose discontinuous at join {seg}");
            // velocity continuity
            assert!((s.body_velocity(t - eps) - s.body_velocity(t + eps)).norm() < 1e-3, "velocity discontinuous at join {seg}");
        }
    }

    #[test]
    fn control_poses_along_one_screw_reproduce_it_at_constant_twist() {
        // Equally-spaced control poses on a single screw: Pᵢ = P₀·exp(i·Ω). The cumulative spline should
        // trace that screw with (nearly) constant body twist Ω.
        let omega = Vector6::new(0.1, -0.2, 0.15, 0.4, -0.1, 0.3); // [ω; v]
        let step = exp_se3(&omega);
        let mut control = vec![Matrix4::identity()];
        for _ in 0..5 {
            control.push(control.last().unwrap() * step);
        }
        let s = SplineSE3 { control };
        // in the interior (away from the ends) the twist should be ≈ Ω
        for &t in &[1.2, 1.5, 1.8] {
            let xi = s.body_velocity(t);
            assert!((xi - omega).norm() < 1e-6, "twist should equal the generating screw: {xi} vs {omega}");
        }
    }
}
