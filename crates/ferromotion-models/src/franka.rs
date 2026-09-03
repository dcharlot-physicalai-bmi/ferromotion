//! **Franka Robotics arms** — the Panda (FER) and the Research 3 (FR3), built from the manufacturer's
//! published modified-DH table.
//!
//! Both arms share one kinematic chain. Franka publishes it in **Craig's modified convention**,
//! `T_i = Rx(α_{i−1})·Tx(a_{i−1})·Rz(θ_i)·Tz(d_i)`, with a separate *Flange* row (`d = 0.107 m`) that
//! this module carries as the `tool` argument of [`Robot::from_dh`]. The two models differ only in
//! their joint limits, velocity limits and, for the FR3, the published inertial situation.
//!
//! Every number below is copied from the specification the doc comment names, in the units the source
//! uses (radians, metres, N·m, rad/s), so no unit conversion was applied in this file.
//!
//! **Verified against:** the flange position at `q = 0`, hand-computed from the table by chaining the
//! seven rows and the Flange row, `(0.088, 0, 0.926) m` with `R = diag(1, −1, −1)`; an in-limits pose
//! `q = [0, −π/4, 0, −3π/4, 0, π/2, π/4]` at `(0.306891, 0, 0.590282) m`, computed numerically from the
//! same table and reproduced here by an independent explicit 4×4 product chain to 1e-9 m; the
//! central-difference Hessian check every `Robot` in this workspace is held to; and a convention swap —
//! the same rows read as standard DH put the flange 0.703 m away at `q = 0` and 0.548 m away at the
//! in-limits pose, which is what makes the `DhConvention::Modified` argument load-bearing.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use nalgebra::{Translation3, UnitQuaternion};

/// The seven joint rows of the Franka table, `(θ_offset, d, a, α)` per [`DhRow::revolute`], in the
/// order Franka prints them: each row `i` holds `(a_{i−1}, α_{i−1}, d_i, θ_i)`.
///
/// Source: Franka Control Interface (FCI) documentation, 'Control Interface Specification and Robot
/// Limits', Denavit–Hartenberg Parameters,
/// <https://frankarobotics.github.io/docs/robot_specifications.html>. All lengths in metres, all
/// angles in radians, as printed.
///
/// | joint | a (m)   | α (rad) | d (m) |
/// |---|---|---|---|
/// | 1 | 0       | 0       | 0.333 |
/// | 2 | 0       | −π/2    | 0     |
/// | 3 | 0       | π/2     | 0.316 |
/// | 4 | 0.0825  | π/2     | 0     |
/// | 5 | −0.0825 | −π/2    | 0.384 |
/// | 6 | 0       | π/2     | 0     |
/// | 7 | 0.088   | π/2     | 0     |
/// | Flange | 0 | 0       | 0.107 |
fn chain(limits: &[(f64, f64); 7]) -> [DhRow; 7] {
    use std::f64::consts::FRAC_PI_2;
    let geometry = [
        (0.333, 0.0, 0.0),
        (0.0, 0.0, -FRAC_PI_2),
        (0.316, 0.0, FRAC_PI_2),
        (0.0, 0.0825, FRAC_PI_2),
        (0.384, -0.0825, -FRAC_PI_2),
        (0.0, 0.0, FRAC_PI_2),
        (0.0, 0.088, FRAC_PI_2),
    ];
    let mut rows = [DhRow::revolute(0.0, 0.0, 0.0, 0.0); 7];
    for (row, ((d, a, alpha), &(lo, hi))) in rows.iter_mut().zip(geometry.iter().zip(limits)) {
        *row = DhRow::revolute(0.0, *d, *a, *alpha).with_limits(lo, hi);
    }
    rows
}

/// The Flange row, `d = 0.107 m` along the joint-7 `z` axis, as a tool transform.
fn flange() -> Iso {
    Iso::from_parts(Translation3::new(0.0, 0.0, 0.107), UnitQuaternion::identity())
}

/// Build the chain and attach the per-joint effort (N·m) and velocity (rad/s) limits, which
/// [`Robot::from_dh`] has no row field for.
fn build(limits: &[(f64, f64); 7], effort: &[f64; 7], velocity: &[f64; 7]) -> Robot {
    let mut robot = Robot::from_dh(&chain(limits), DhConvention::Modified, flange()).expect("the Franka table is non-empty and finite");
    for (joint, (&tau, &dq)) in robot.joints.iter_mut().zip(effort.iter().zip(velocity)) {
        *joint = joint.clone().with_effort(tau).with_max_velocity(dq);
    }
    robot
}

/// **Franka Emika Panda (FER)**, 7 revolute joints, modified DH, flange at `d = 0.107 m`.
///
/// **Primary source** (title and URL as recorded in the specification): "Franka Control Interface
/// (FCI) documentation, 'Control Interface Specification and Robot Limits' (Denavit-Hartenberg
/// Parameters; Limits for Franka Emika Robot (FER)); corroborated by 'Data Sheet Robot Arm & Control',
/// Release April 2020", <https://frankarobotics.github.io/docs/robot_specifications.html>.
/// Confidence: **published primary source**. Convention: **modified DH (Craig)**, passed as
/// [`DhConvention::Modified`].
///
/// **Joint limits** (rad), from the FCI 'Limits for Franka Emika Robot (FER)' table, `q_min`/`q_max`:
/// joints 1, 3, 5, 7 `±2.8973`; joint 2 `±1.7628`; joint 4 `[−3.0718, −0.0698]`; joint 6
/// `[−0.0175, 3.7525]`. **Effort** (`tau_max`, N·m): `87` on joints 1–4, `12` on joints 5–7.
/// **Velocity** (`dq_max`, rad/s): `2.175` on joints 1–4, `2.61` on joints 5–7. All three are attached
/// to the returned joints; nothing was converted (the source states radians, N·m and rad/s directly).
/// The same table's acceleration and jerk limits and the Cartesian limits are not represented by
/// [`Robot`] and are not carried.
///
/// **Known-answer pose, computed by hand from the table** (not stated by the manufacturer — the FCI
/// page publishes no home pose): at `q = 0` the flange is at `(0.088, 0, 0.926) m` with
/// `R = diag(1, −1, −1)`, because `z = 0.333 + 0.316 + 0.384 − 0.107` (the flange `z` axis points down
/// at `q = 0`, so the Flange row subtracts) and `x = 0.088`. `q = 0` lies outside the joint-4 limit,
/// so it is a kinematic-chain check only; the in-limits pose `q = [0, −π/4, 0, −3π/4, 0, π/2, π/4]`
/// at `(0.306891, 0, 0.590282) m` is a secondary check computed numerically from the same table and
/// re-derived here by an explicit product chain. Both are verified in this module's tests, along with
/// the Hessian finite-difference check and the convention swap.
///
/// Inertial parameters: Franka publishes none in a document; the spec points to Gaz et al., IEEE RA-L
/// 4(4) 2019, DOI 10.1109/LRA.2019.2931248, which is not transcribed here.
pub fn panda() -> Robot {
    build(
        &[(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)],
        &[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
        &[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
    )
}

/// **Franka Research 3 (FR3)**, 7 revolute joints, modified DH, flange at `d = 0.107 m`. Same
/// kinematic table as [`panda`]; the limits differ.
///
/// **Primary source** (title and URL as recorded in the specification): "Franka Control Interface
/// (FCI) documentation, 'Control Interface Specification and Robot Limits' (Denavit-Hartenberg
/// Parameters for the Franka Research 3 kinematic chain, following Craig's convention; Limits for
/// Franka Research 3 (FR3)); corroborated by 'Datasheet Franka Research 3', Document number R02212,
/// release version 2.4 (October 2025)", <https://frankarobotics.github.io/docs/robot_specifications.html>.
/// Confidence: **published primary source**. Convention: **modified DH (Craig)**, passed as
/// [`DhConvention::Modified`].
///
/// **Joint limits** (rad), from the FCI 'Limits for Franka Research 3 (FR3)' table: joints 1 and 3
/// `±2.9007`; joint 2 `±1.8361`; joint 4 `[−3.0770, −0.1169]`; joint 5 `±2.8763`; joint 6
/// `[0.4398, 4.6216]`; joint 7 `±3.0508`. **Effort** (`tau_max`, N·m): `87` on joints 1–4, `12` on
/// joints 5–7. **Velocity** (`dq_max`, rad/s): `[2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]`. No unit
/// conversion was applied. The FCI page's position-dependent velocity limits, acceleration and jerk
/// limits, and Cartesian limits are not represented by [`Robot`] and are not carried. An independent
/// sourcing of the same arm (Datasheet R02212 v2.4, degrees converted to radians) gives limits that
/// differ from the FCI radian table by up to 0.0053 rad on joint 4 (`−0.1222` vs `−0.1169`) and by
/// 0.0034 rad on joint 1 (`2.8973` vs `2.9007`); the FCI table is what this constructor carries.
///
/// **Known-answer pose, computed by hand from the table** (not stated by the manufacturer): at
/// `q = 0` the flange is at `(0.088, 0, 0.926) m`, `R = diag(1, −1, −1)`, from
/// `z = 0.333 + 0.316 + 0.384 − 0.107`. `q = 0` is outside the joint-4 and joint-6 limits, so it is a
/// kinematic-chain check only; the in-limits pose `q = [0, −π/4, 0, −3π/4, 0, π/2, π/4]` at
/// `(0.306891, 0, 0.590282) m` is a secondary check computed numerically from the same table. Both
/// are verified in this module's tests.
///
/// Inertial parameters: the specification did not locate a primary document publishing FR3 link
/// inertias, so none are claimed.
pub fn fr3() -> Robot {
    build(
        &[(-2.9007, 2.9007), (-1.8361, 1.8361), (-2.9007, 2.9007), (-3.077, -0.1169), (-2.8763, 2.8763), (0.4398, 4.6216), (-3.0508, 3.0508)],
        &[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
        &[2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    /// The in-limits secondary pose from the specification: `[0, −π/4, 0, −3π/4, 0, π/2, π/4]`.
    fn ready() -> [f64; 7] {
        [0.0, -FRAC_PI_4, 0.0, -3.0 * FRAC_PI_4, 0.0, FRAC_PI_2, FRAC_PI_4]
    }

    /// The tests below run on both arms because the specification states they share one table.
    fn both() -> [(&'static str, Robot); 2] {
        [("panda", panda()), ("fr3", fr3())]
    }

    /// **Hand-computed known answer at `q = 0`.** The flange sits at `(0.088, 0, 0.926)`: the three
    /// `d` values stack to `1.033`, and the Flange row's `0.107` subtracts because the joint-7 `z`
    /// axis points down there, so `R` must be `diag(1, −1, −1)` too. The fixture is non-vacuous: the
    /// flange moves from this pose when joint 4 bends, so a chain that ignored its rows could not pass.
    #[test]
    fn q_zero_flange_pose_matches_the_hand_computation_from_the_table() {
        let want = Vector3::new(0.088, 0.0, 0.926);
        for (name, r) in both() {
            let t = r.fk(&[0.0; 7]);
            let moved = r.fk(&[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]).translation.vector;
            assert!((moved - want).norm() > 0.1, "{name}: bending joint 4 must move the flange, got {moved:?}");
            let p = t.translation.vector;
            assert!(close(&p, &want, 1e-9), "{name}: q = 0 flange at {p:?}, hand says {want:?}");
            let rot = t.rotation.to_rotation_matrix().into_inner();
            let want_r = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
            assert!((rot - want_r).norm() < 1e-12, "{name}: q = 0 flange orientation {rot} vs diag(1, −1, −1)");
        }
    }

    /// **The in-limits pose the specification computed numerically from the same table**, reproduced
    /// to the six decimals it states (1e-6 m), and to 1e-9 m against an explicit 4×4 product chain of
    /// Craig's formula written for this module: `(0.306890567, 0, 0.590282052)`.
    #[test]
    fn ready_pose_computed_from_the_table_matches_the_stated_position() {
        let stated = Vector3::new(0.306891, 0.0, 0.590282);
        let chain_product = Vector3::new(0.306_890_567, 0.0, 0.590_282_052);
        for (name, r) in both() {
            let q = ready();
            for (j, &qi) in q.iter().enumerate() {
                let (lo, hi) = r.joints[j].limits.expect("every Franka joint has a limit");
                assert!(lo <= qi && qi <= hi, "{name}: the ready pose must be inside the joint-{} limit [{lo}, {hi}], q = {qi}", j + 1);
            }
            let p = r.fk(&q).translation.vector;
            assert!(!close(&p, &Vector3::new(0.088, 0.0, 0.926), 1e-3), "{name}: the ready pose must differ from q = 0");
            assert!(close(&p, &stated, 1e-6), "{name}: ready pose at {p:?}, spec says {stated:?}");
            assert!(close(&p, &chain_product, 1e-9), "{name}: ready pose at {p:?}, product chain says {chain_product:?}");
        }
    }

    #[test]
    fn dof_is_seven() {
        for (name, r) in both() {
            assert_eq!(r.dof(), 7, "{name}");
        }
    }

    /// The analytic Hessian against central differences of the Jacobian, at the in-limits pose.
    #[test]
    fn a_franka_arm_passes_the_hessian_finite_difference_check() {
        for (name, r) in both() {
            let q = ready();
            let h = r.kinematic_hessian(&q);
            let eps = 1e-6;
            let (mut worst, mut scale) = (0.0f64, 0.0f64);
            for j in 0..q.len() {
                let (mut qp, mut qm) = (q.to_vec(), q.to_vec());
                qp[j] += eps;
                qm[j] -= eps;
                let fd = (r.jacobian(&qp) - r.jacobian(&qm)) / (2.0 * eps);
                for k in 0..fd.len() {
                    worst = worst.max((h[j][k] - fd[k]).abs());
                    scale = scale.max(fd[k].abs());
                }
            }
            assert!(scale > 0.1, "{name}: the Jacobian should vary, scale {scale:e}");
            assert!(worst < 1e-6 * scale, "{name}: Hessian vs FD: worst {worst:e} on a scale of {scale:e}");
        }
    }

    /// **The convention argument is load-bearing.** The same seven rows read as standard DH build a
    /// different arm: at `q = 0` the flange lands 0.703 m away (`(0.088, −0.068, 0.226)` by the
    /// product chain) and at the ready pose 0.548 m away. A table read under the wrong convention is
    /// the failure mode this crate's tests exist to catch.
    #[test]
    fn reading_the_table_as_standard_dh_moves_the_flange_by_more_than_a_millimetre() {
        let rows = chain(&[(-1.0, 1.0); 7]);
        let modified = Robot::from_dh(&rows, DhConvention::Modified, flange()).unwrap();
        let standard = Robot::from_dh(&rows, DhConvention::Standard, flange()).unwrap();
        for (q, expect) in [([0.0; 7], 0.703_295), (ready(), 0.548_395)] {
            let pm = modified.fk(&q).translation.vector;
            let ps = standard.fk(&q).translation.vector;
            let dist = (pm - ps).norm();
            assert!(dist > 1e-3, "convention swap at {q:?} moved the flange by only {dist} m");
            assert!((dist - expect).abs() < 1e-6, "convention swap at {q:?}: {dist} m, product chain says {expect} m");
        }
        assert!(close(&standard.fk(&[0.0; 7]).translation.vector, &Vector3::new(0.088, -0.068, 0.226), 1e-9));
    }

    /// The two constructors carry the limits, effort and velocity their sources state, and they carry
    /// them per joint: the FR3's joint 6 cannot reach `0` while the Panda's can, and the FR3's wrist is
    /// rated at more than twice the Panda's velocity.
    #[test]
    fn limits_effort_and_velocity_are_the_published_values() {
        let (p, f) = (panda(), fr3());
        assert_eq!(p.joints[3].limits, Some((-3.0718, -0.0698)), "Panda joint 4");
        assert_eq!(p.joints[5].limits, Some((-0.0175, 3.7525)), "Panda joint 6");
        assert_eq!(f.joints[3].limits, Some((-3.077, -0.1169)), "FR3 joint 4");
        assert_eq!(f.joints[5].limits, Some((0.4398, 4.6216)), "FR3 joint 6");
        assert_eq!(f.joints[6].limits, Some((-3.0508, 3.0508)), "FR3 joint 7");
        for (i, j) in p.joints.iter().enumerate() {
            assert_eq!(j.effort, Some(if i < 4 { 87.0 } else { 12.0 }), "Panda effort joint {}", i + 1);
            assert_eq!(j.max_velocity, Some(if i < 4 { 2.175 } else { 2.61 }), "Panda velocity joint {}", i + 1);
        }
        for (i, j) in f.joints.iter().enumerate() {
            assert_eq!(j.effort, Some(if i < 4 { 87.0 } else { 12.0 }), "FR3 effort joint {}", i + 1);
        }
        assert_eq!(f.joints.iter().map(|j| j.max_velocity.unwrap()).collect::<Vec<_>>(), vec![2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], "FR3 velocity");
        assert!(p.joints[5].limits.unwrap().0 < 0.0 && f.joints[5].limits.unwrap().0 > 0.0, "the two arms must differ where their sources do");
    }

    /// The specification states the two arms share one kinematic table, so their forward kinematics
    /// must agree everywhere — including at poses only one of them is allowed to reach.
    #[test]
    fn panda_and_fr3_share_one_kinematic_chain() {
        let (p, f) = (panda(), fr3());
        let poses = [[0.0; 7], ready(), [0.5, -1.0, 0.3, -2.0, 0.7, 2.5, -0.9], [-2.0, 1.2, 1.5, -0.5, -2.5, 3.0, PI]];
        let p0 = p.fk(&poses[0]).translation.vector;
        assert!(!close(&p.fk(&poses[2]).translation.vector, &p0, 1e-3), "the poses must differ");
        for q in poses {
            let (a, b) = (p.fk(&q), f.fk(&q));
            assert!((a.translation.vector - b.translation.vector).norm() < 1e-15 && (a.rotation.angle_to(&b.rotation)) < 1e-12, "at {q:?}: {a:?} vs {b:?}");
        }
    }
}
