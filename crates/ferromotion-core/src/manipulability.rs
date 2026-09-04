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
///
/// A non-finite entry yields all-`NaN` singular values. This is not defensive tidying: nalgebra's SVD
/// iteration **never terminates** on a matrix holding a `NaN`, so without this guard every function in
/// this module spins forever instead of returning, measured at 5 s with no result in
/// `tests/nonfinite_public_api.rs`. A hang in a control loop is worse than a wrong number.
pub fn singular_values(j: &DMatrix<f64>) -> Vec<f64> {
    crate::finite_singular_values(j).unwrap_or_else(|| vec![f64::NAN; j.nrows().min(j.ncols())])
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

/// The manipulability gradient `∂w/∂q`, **analytically**, from the kinematic Hessian.
///
/// With `w = √det(A)`, `∂det(A)/∂q_j = det(A)·tr(A⁻¹ ∂A/∂q_j)` gives
/// `∂w/∂q_j = w·tr(A⁻¹ H_j Jᵀ)` after the two trace terms collapse (they are transposes of each other
/// and `A⁻¹` is symmetric). `H_j = ∂J/∂q_j` comes from [`Robot::kinematic_hessian`].
///
/// **Which `A` depends on the arm, and getting it wrong is silent.** [`yoshikawa`] is the product of
/// *every* singular value of `J`, which is `√det(JJᵀ)` only when the arm has at least 6 joints. Below
/// that, `J` is `6 × n` with `n` singular values and the index is `√det(JᵀJ)`, so the gradient is
/// `w·tr(A⁻¹ Jᵀ H_j)` with `A = JᵀJ`. A 5-DoF arm is the common case, not the exotic one.
///
/// Measured on a 6-DoF arm: **14.08 us** against **66.96 us** for [`manipulability_gradient`], a 4.8x
/// saving, and the two agree to 1e-5 relative. The gap is not the Hessian, which costs 1.56 us against
/// 0.71 us for a single Jacobian; it is that differencing the index needs `2·dof` **SVDs**, and this
/// needs one.
///
/// Returns `None` at a singularity, where `A` is not invertible and `w = 0` has no gradient in this
/// form. [`manipulability_gradient`] differences the index instead and returns a number there, which is
/// the reason to keep both: the finite-difference version degrades where this one refuses.
pub fn manipulability_gradient_analytic(robot: &Robot, q: &[f64]) -> Option<Vec<f64>> {
    let j = robot.jacobian(q);
    let n = q.len();
    if n != robot.dof() {
        return None;
    }
    let w = yoshikawa(&j);
    let hs = robot.kinematic_hessian(q);
    // `A` is JJᵀ for a 6-or-more-joint arm and JᵀJ below that, matching what `yoshikawa` computes.
    let wide = n >= 6;
    let a = if wide { &j * j.transpose() } else { j.transpose() * &j };
    let ainv = a.try_inverse()?;
    let mut g = vec![0.0; n];
    for (k, h) in hs.iter().enumerate() {
        let m = if wide { &ainv * (h * j.transpose()) } else { &ainv * (j.transpose() * h) };
        let tr = m.trace();
        if !tr.is_finite() {
            return None;
        }
        g[k] = w * tr;
    }
    Some(g)
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
                crate::Joint::revolute(Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), nalgebra::UnitQuaternion::identity()), Vector3::z()),
            ],
            ee_offset: Isometry3::from_parts(Translation3::new(0.7, 0.0, 0.0), nalgebra::UnitQuaternion::identity()),
        };
        let q = [0.3, 0.15]; // elbow nearly straight ⇒ low manipulability
        let g = manipulability_gradient(&robot, &q);
        let w0 = yoshikawa(&robot.jacobian(&q));
        let step = 0.05;
        let q1 = [q[0] + step * g[0], q[1] + step * g[1]];
        let w1 = yoshikawa(&robot.jacobian(&q1));
        assert!(w1 > w0, "ascending the manipulability gradient should raise w: {w0} → {w1}");
    }

    /// A chain with BOTH joint kinds, so the prismatic branch of the Hessian is exercised. A
    /// revolute-only fixture would leave half the derivation untested.
    fn mixed_arm() -> Robot {
        const URDF: &str = r#"<robot name="mixed">
          <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="tool"/>
          <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/>
            <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
          <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.4 0 0.05"/>
            <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
          <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0.02 0"/>
            <axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" effort="10" velocity="3"/></joint>
          <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0.25 0 0.03"/>
            <axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
          <joint name="jt" type="fixed"><parent link="l4"/><child link="tool"/><origin xyz="0.2 0.01 0"/></joint>
        </robot>"#;
        crate::from_urdf_str(URDF, "base", "tool").expect("fixture urdf")
    }

    /// **The kinematic Hessian equals central differences of the Jacobian.**
    ///
    /// The whole derivation is geometric bookkeeping over which frames a joint moves, which is easy to
    /// get subtly wrong and impossible to eyeball. Finite differences adjudicate. The fixture mixes
    /// revolute and prismatic joints so both branches are covered, and asserts the Hessian is not
    /// trivially zero, since an all-zero tensor would match a Jacobian that happened not to vary.
    #[test]
    fn the_kinematic_hessian_matches_finite_differences() {
        let r = mixed_arm();
        let n = r.dof();
        for q in [vec![0.3, -0.5, 0.12, 0.7], vec![-1.1, 0.9, -0.2, 0.4], vec![0.0, 0.0, 0.0, 0.0]] {
            let h = r.kinematic_hessian(&q);
            assert_eq!(h.len(), n);
            let eps = 1e-6;
            let mut worst = 0.0f64;
            let mut scale = 0.0f64;
            for jj in 0..n {
                let (mut qp, mut qm) = (q.clone(), q.clone());
                qp[jj] += eps;
                qm[jj] -= eps;
                let fd = (r.jacobian(&qp) - r.jacobian(&qm)) / (2.0 * eps);
                for k in 0..6 * n {
                    worst = worst.max((h[jj][k] - fd[k]).abs());
                    scale = scale.max(fd[k].abs());
                }
            }
            assert!(scale > 0.1, "the Jacobian should actually vary at q = {q:?}, scale {scale:e}");
            assert!(worst < 1e-6 * scale.max(1.0), "Hessian vs central differences at q = {q:?}: worst {worst:e} on a scale of {scale:e}");
        }
    }

    /// `jacobian_dot` is the Jacobian's rate along a joint velocity, checked against differencing `J`
    /// over a small step of that same velocity.
    #[test]
    fn jacobian_dot_matches_differencing_along_the_velocity() {
        let r = mixed_arm();
        let q = vec![0.25, -0.4, 0.08, 0.6];
        let qd = vec![0.7, -1.3, 0.45, 0.9];
        let jd = r.jacobian_dot(&q, &qd).expect("dimensions match");
        let dt = 1e-6;
        let step = |sign: f64| -> Vec<f64> { (0..q.len()).map(|i| q[i] + sign * dt * qd[i]).collect() };
        let fd = (r.jacobian(&step(1.0)) - r.jacobian(&step(-1.0))) / (2.0 * dt);
        let scale = fd.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let worst = jd.iter().zip(fd.iter()).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(scale > 0.1, "J should be changing, scale {scale:e}");
        assert!(worst < 1e-6 * scale, "J-dot vs finite difference: worst {worst:e} on a scale of {scale:e}");
        assert!(r.jacobian_dot(&q, &[0.1]).is_none(), "a mismatched velocity length must be refused");
    }

    /// **The analytic manipulability gradient agrees with the finite-difference one.**
    ///
    /// Two independent routes to the same quantity: one differences [`yoshikawa`], the other goes
    /// through the Hessian and a trace identity. Checked on a 4-DoF arm, which takes the `JᵀJ` branch,
    /// because that is the branch a real 5-DoF arm uses and the one it is easy to get wrong by assuming
    /// `JJᵀ` throughout.
    #[test]
    fn the_analytic_manipulability_gradient_agrees_with_finite_differences() {
        let r = mixed_arm();
        assert!(r.dof() < 6, "this fixture is meant to exercise the J-transpose-J branch");
        for q in [vec![0.3, -0.5, 0.12, 0.7], vec![-0.8, 0.6, -0.15, 1.0]] {
            let fd = manipulability_gradient(&r, &q);
            let an = manipulability_gradient_analytic(&r, &q).expect("non-singular fixture");
            let scale = fd.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            let worst = an.iter().zip(&fd).fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
            assert!(scale > 1e-4, "the gradient should be non-trivial at q = {q:?}, scale {scale:e}");
            assert!(worst < 1e-5 * scale.max(1e-9), "analytic vs FD manipulability gradient at q = {q:?}: worst {worst:e} on a scale of {scale:e}");
        }
    }

}
