//! Resolved-rate motion control — a clean-room reimplementation of Orocos KDL's velocity IK
//! (`ChainIkSolverVel_wdls`). Given a desired end-effector twist, return the minimum-norm joint
//! velocity via a damped-least-squares pseudoinverse, with an optional secondary task projected
//! into the Jacobian nullspace (redundancy resolution). KDL is LGPL, so this is written from the
//! math, not its source. Pure `nalgebra` → WASM-clean.

use crate::Robot;
use nalgebra::{DMatrix, DVector, Vector6};

/// Damped-least-squares resolved-rate step: `q̇ = Jᵀ(JJᵀ + λ²I)⁻¹·twist`, plus an optional secondary
/// joint-velocity `q̇₀` projected into the nullspace `(I − J⁺J)·q̇₀` (leaves the task twist untouched).
/// `twist` is `[vx,vy,vz, ωx,ωy,ωz]` in the world frame.
pub fn resolved_rate(
    robot: &Robot,
    q: &[f64],
    twist: &Vector6<f64>,
    damping: f64,
    secondary: Option<&[f64]>,
) -> Vec<f64> {
    let n = robot.dof();
    let j = robot.jacobian(q); // 6×n
    let jt = j.transpose(); // n×6
    let mut jjt = &j * &jt; // 6×6
    for i in 0..6 {
        jjt[(i, i)] += damping * damping;
    }
    let inv = jjt.try_inverse().unwrap_or_else(|| DMatrix::identity(6, 6));
    let jpinv = &jt * &inv; // n×6 damped pseudoinverse

    let tw = DVector::from_column_slice(twist.as_slice());
    let mut qd = &jpinv * &tw;

    if let Some(sec) = secondary {
        let sec_v = DVector::from_row_slice(sec);
        let proj = DMatrix::<f64>::identity(n, n) - &jpinv * &j; // I − J⁺J
        qd += proj * sec_v;
    }
    qd.as_slice().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_str;

    const ARM: &str = r#"<robot name="a"><link name="world"/><link name="base"/>
      <link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
      <joint name="j0" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/></joint></robot>"#;

    #[test]
    fn resolved_rate_realizes_the_twist() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let q = [0.2, -0.5, 0.7, 0.3, 0.4, -0.2]; // non-singular
        let twist = Vector6::new(0.05, -0.03, 0.02, 0.01, 0.0, -0.01);
        let qd = resolved_rate(&robot, &q, &twist, 1e-4, None);
        // J·q̇ should reproduce the commanded twist (small damping).
        let jqd = robot.jacobian(&q) * DVector::from_row_slice(&qd);
        let realized = Vector6::from_column_slice(jqd.as_slice());
        assert!((realized - twist).norm() < 1e-3, "twist error {}", (realized - twist).norm());
    }

    #[test]
    fn nullspace_secondary_preserves_the_task() {
        let robot = from_urdf_str(ARM, "world", "tool").unwrap();
        let q = [0.2, -0.5, 0.7, 0.3, 0.4, -0.2];
        let twist = Vector6::new(0.04, 0.02, -0.03, 0.0, 0.01, 0.0);
        let secondary = [0.5, -0.4, 0.3, 0.2, -0.1, 0.6]; // arbitrary posture bias
        let qd_plain = resolved_rate(&robot, &q, &twist, 1e-4, None);
        let qd_null = resolved_rate(&robot, &q, &twist, 1e-4, Some(&secondary));
        // The secondary task changes the joint motion...
        let diff: f64 = qd_plain.iter().zip(&qd_null).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(diff.sqrt() > 1e-3, "secondary task had no effect");
        // ...but the realized end-effector twist is unchanged (it lives in the nullspace).
        let jqd = robot.jacobian(&q) * DVector::from_row_slice(&qd_null);
        let realized = Vector6::from_column_slice(jqd.as_slice());
        assert!((realized - twist).norm() < 1e-3, "nullspace term disturbed the task");
    }
}
