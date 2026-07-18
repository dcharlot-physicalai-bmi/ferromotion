//! **Manipulability** (Yoshikawa, IJRR 1985) — the standard scalar/geometric measures of how freely a
//! manipulator can move (or exert force) at a configuration, read straight off its Jacobian `J`. The
//! **velocity manipulability ellipsoid** is the image of the unit joint-velocity ball under `J`; its axes
//! are the singular values `σᵢ` (directions the columns of `U`). From it come the **Yoshikawa index**
//! `w = √det(JJᵀ) = ∏σᵢ` (ellipsoid volume — zero at a singularity), the **condition number** `σ_max/σ_min`
//! (how anisotropic / how close to singular), and its reciprocal **isotropy**. The **force** ellipsoid is
//! the dual (`1/σᵢ`) — easy where motion is hard and vice versa. The **manipulability gradient** `∂w/∂q`
//! is the redundancy-resolution signal that steers a robot away from singularities. Pure `nalgebra`,
//! verified against the closed-form 2R planar arm. → WASM-clean.

use crate::Robot;
use nalgebra::DMatrix;

/// The Yoshikawa manipulability index `w = √det(JJᵀ)` = product of the singular values of `J` (the
/// velocity-ellipsoid volume). Zero exactly at a kinematic singularity.
pub fn yoshikawa(j: &DMatrix<f64>) -> f64 {
    singular_values(j).iter().product()
}

/// Singular values of `J`, largest first (the velocity-ellipsoid semi-axis lengths).
pub fn singular_values(j: &DMatrix<f64>) -> Vec<f64> {
    let mut s: Vec<f64> = j.clone().singular_values().iter().cloned().collect();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    s
}

/// Condition number `σ_max / σ_min` (≥ 1; → ∞ at a singularity). A round ellipsoid (isotropic) is 1.
pub fn condition_number(j: &DMatrix<f64>) -> f64 {
    let s = singular_values(j);
    let (mx, mn) = (s.first().copied().unwrap_or(0.0), s.last().copied().unwrap_or(0.0));
    if mn < 1e-15 { f64::INFINITY } else { mx / mn }
}

/// Isotropy `σ_min / σ_max ∈ [0,1]` (1 = perfectly isotropic, 0 = singular) — the reciprocal condition.
pub fn isotropy(j: &DMatrix<f64>) -> f64 {
    let s = singular_values(j);
    let (mx, mn) = (s.first().copied().unwrap_or(0.0), s.last().copied().unwrap_or(0.0));
    if mx < 1e-15 { 0.0 } else { mn / mx }
}

/// Force-ellipsoid semi-axis lengths `1/σᵢ` — the dual of the velocity ellipsoid (large where velocity is
/// constrained). Returns `∞` for any zero singular value.
pub fn force_ellipsoid_axes(j: &DMatrix<f64>) -> Vec<f64> {
    singular_values(j).into_iter().map(|s| if s < 1e-15 { f64::INFINITY } else { 1.0 / s }).collect()
}

/// The manipulability gradient `∂w/∂q` (finite differences of [`yoshikawa`] over the robot's Jacobian) —
/// ascend it to move away from singularities in a null-space / redundancy-resolution task.
pub fn manipulability_gradient(robot: &Robot, q: &[f64]) -> Vec<f64> {
    let eps = 1e-6;
    let mut g = vec![0.0; q.len()];
    for i in 0..q.len() {
        let mut qp = q.to_vec();
        let mut qm = q.to_vec();
        qp[i] += eps;
        qm[i] -= eps;
        g[i] = (yoshikawa(&robot.jacobian(&qp)) - yoshikawa(&robot.jacobian(&qm))) / (2.0 * eps);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Translation3, Vector3};

    // Position (2×2) Jacobian of a 2R planar arm at joint angles (t1, t2).
    fn planar_2r_jac(l1: f64, l2: f64, t1: f64, t2: f64) -> DMatrix<f64> {
        let (s1, c1) = t1.sin_cos();
        let (s12, c12) = (t1 + t2).sin_cos();
        DMatrix::from_row_slice(2, 2, &[-l1 * s1 - l2 * s12, -l2 * s12, l1 * c1 + l2 * c12, l2 * c12])
    }

    #[test]
    fn yoshikawa_matches_the_closed_form_2r_arm() {
        // For a 2R planar arm, w = |L1·L2·sin θ2| exactly.
        let (l1, l2) = (1.0, 0.7);
        for &(t1, t2) in &[(0.3, 0.9), (1.2, -0.6), (0.0, 1.5)] {
            let w = yoshikawa(&planar_2r_jac(l1, l2, t1, t2));
            let expect = (l1 * l2 * t2.sin()).abs();
            assert!((w - expect).abs() < 1e-9, "w {w} vs {expect}");
        }
    }

    #[test]
    fn manipulability_vanishes_at_the_stretched_singularity() {
        // θ2 = 0 ⇒ arm straight ⇒ w = 0 and the condition number blows up.
        let j = planar_2r_jac(1.0, 0.7, 0.5, 0.0);
        assert!(yoshikawa(&j) < 1e-12, "straight arm should be singular: w = {}", yoshikawa(&j));
        assert!(condition_number(&j) > 1e6, "condition number should blow up");
        assert!(isotropy(&j) < 1e-6, "isotropy should vanish");
    }

    #[test]
    fn the_force_ellipsoid_is_the_dual_of_the_velocity_ellipsoid() {
        let j = planar_2r_jac(1.0, 0.7, 0.6, 0.8);
        let v = singular_values(&j);
        let f = force_ellipsoid_axes(&j);
        for (vi, fi) in v.iter().zip(&f) {
            assert!((vi * fi - 1.0).abs() < 1e-9, "force axis should be 1/velocity axis");
        }
        // where velocity is largest, force is smallest (dual)
        assert!(f[0] <= *f.last().unwrap(), "force ellipsoid inverts the velocity ordering");
    }

    #[test]
    fn the_gradient_points_away_from_a_singularity() {
        // A near-straight 2R arm: ascending ∂w/∂q must increase manipulability (bend the elbow).
        let robot = Robot {
            joints: vec![
                crate::Joint::revolute(Isometry3::identity(), Vector3::z()),
                crate::Joint::revolute(Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0).into(), nalgebra::UnitQuaternion::identity()), Vector3::z()),
            ],
            ee_offset: Isometry3::from_parts(Translation3::new(0.7, 0.0, 0.0).into(), nalgebra::UnitQuaternion::identity()),
        };
        let q = [0.3, 0.15]; // elbow nearly straight ⇒ low manipulability
        let g = manipulability_gradient(&robot, &q);
        let w0 = yoshikawa(&robot.jacobian(&q));
        let step = 0.05;
        let q1 = [q[0] + step * g[0], q[1] + step * g[1]];
        let w1 = yoshikawa(&robot.jacobian(&q1));
        assert!(w1 > w0, "ascending the manipulability gradient should raise w: {w0} → {w1}");
    }
}
