//! **KUKA arms** — the LBR iiwa 7 R800, the LBR iiwa 14 R820 and the KR 5 arc, each built with
//! [`Robot::from_dh`] from a table that is **derived from published geometry**, not from a
//! manufacturer DH table: KUKA publishes dimension drawings and axis data for these arms and no DH
//! table, so the link lengths `d`/`a` below are read off the specification drawings, and the frame
//! assignment (the `alpha` signs, the `theta` offset on the KR 5 arc) is the reviewer's, following the
//! symbolic standard-DH structure printed in Beck et al. 2023 for the iiwa.
//!
//! Every constant in this file names the figure or section it came from. Joint position limits and
//! speeds are KUKA's published axis data converted from degrees (`x.to_radians()`, i.e. `x·π/180`)
//! and metres from the drawing's millimetres (`÷ 1000`); both are attached on the [`DhRow`]
//! ([`DhRow::with_limits`], [`DhRow::with_max_velocity`]) and carried onto the joint by
//! [`Robot::from_dh`], the same path the other DH-table models in this crate use. Per-axis torque
//! limits are not in the specifications and are left `None` on every joint.
//!
//! What is verified, and against what: forward kinematics at `q = 0` against the overall height (or
//! reach) figure printed on the working-envelope drawing, hand-chained in the doc of each constructor;
//! forward kinematics at a generic `q` against an independent 15-line matrix-product chain of the same
//! table (numpy, written for this file; values quoted in the tests); the analytic kinematic Hessian
//! against central differences of the Jacobian; and a convention swap showing that reading each table
//! as modified DH moves the known-answer pose by more than a metre, so the convention argument is
//! load-bearing.

use ferromotion_core::Iso;

use crate::{DhConvention, DhRow, Robot};
use std::f64::consts::FRAC_PI_2;

/// Height of the A6 axis above the flange face on both iiwa variants, from the KUKA working-envelope
/// drawings: Fig. 4-1 shows `(1140)` to A6 and `(1266)` overall for the 7 R800, Fig. 4-4 shows `(1180)`
/// and `(1306)` for the 14 R820; both differences are 126 mm.
const IIWA_FLANGE_M: f64 = 0.126;

/// The seven iiwa rows: the non-offset SRS structure (`a_i = 0`, `theta_i = q_i`) in standard DH as
/// printed symbolically in Beck et al. 2023 Table 1(b), `alpha = [π/2, −π/2, −π/2, π/2, π/2, −π/2, 0]`,
/// with `d1, d3, d5` supplied by the caller from the drawing. Limits are the same on both variants
/// (KUKA Sections 4.2.2 and 4.3.2): A1/A3/A5 ±170°, A2/A4/A6 ±120°, A7 ±175°, converted with
/// `to_radians()`. `speeds_deg_per_s` is KUKA's "speed with rated payload" per axis A1..A7, which
/// differs between the two variants; it is attached to each row with [`DhRow::with_max_velocity`]
/// in rad/s and carried onto the joint by [`Robot::from_dh`]. Effort is left `None`: no per-axis
/// torque is published.
fn iiwa_rows(d1: f64, d3: f64, d5: f64, speeds_deg_per_s: [f64; 7]) -> [DhRow; 7] {
    let (odd, even, last) = (170.0_f64.to_radians(), 120.0_f64.to_radians(), 175.0_f64.to_radians());
    let v = speeds_deg_per_s.map(f64::to_radians);
    [
        DhRow::revolute(0.0, d1, 0.0, FRAC_PI_2).with_limits(-odd, odd).with_max_velocity(v[0]),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(-even, even).with_max_velocity(v[1]),
        DhRow::revolute(0.0, d3, 0.0, -FRAC_PI_2).with_limits(-odd, odd).with_max_velocity(v[2]),
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2).with_limits(-even, even).with_max_velocity(v[3]),
        DhRow::revolute(0.0, d5, 0.0, FRAC_PI_2).with_limits(-odd, odd).with_max_velocity(v[4]),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(-even, even).with_max_velocity(v[5]),
        DhRow::revolute(0.0, 0.0, 0.0, 0.0).with_limits(-last, last).with_max_velocity(v[6]),
    ]
}

/// KUKA Section 4.2.2, speed with rated payload for the LBR iiwa 7 R800, A1..A7 in deg/s.
const IIWA_7_R800_SPEEDS_DEG: [f64; 7] = [98.0, 98.0, 100.0, 130.0, 140.0, 180.0, 180.0];

/// KUKA Section 4.3.2, speed with rated payload for the LBR iiwa 14 R820, A1..A7 in deg/s.
const IIWA_14_R820_SPEEDS_DEG: [f64; 7] = [85.0, 85.0, 100.0, 75.0, 130.0, 135.0, 135.0];

/// **KUKA LBR iiwa 7 R800**, 7 DoF, standard DH. Confidence: **derived from published geometry**.
///
/// Primary source: KUKA Roboter GmbH, 'Robots LBR iiwa: LBR iiwa 7 R800, LBR iiwa 14 R820
/// Specification', Version Spez LBR iiwa V5, Issued 28.01.2015 (Section 4.2.1 Basic data, 4.2.2 Axis
/// data, Fig. 4-1 Working envelope LBR iiwa 7 R800); DH frame convention and table structure from
/// F. Beck, M. N. Vu, C. Hartl-Nesic, A. Kugi, 'Singularity Avoidance with Application to Online
/// Trajectory Optimization for Serial Manipulators', arXiv:2211.02516v5 (accepted IFAC World Congress
/// 2023), Table 1(b).
/// KUKA Roboter GmbH, *LBR iiwa 7 R800 / LBR iiwa 14 R820 Specification* (Spez LBR iiwa, V5); read from an archived copy, <https://web.archive.org/web/20190819075249id_/http://www.oir.caltech.edu/twiki_oir/pub/Palomar/ZTF/KUKARoboticArmMaterial/Spez_LBR_iiwa_en.pdf>
///
/// KUKA publishes no DH table for this arm. The table here is **derived from published geometry**:
/// the drawing it is derived from is **KUKA Fig. 4-1, Working envelope LBR iiwa 7 R800**, which in the
/// fully extended vertical pose gives base to A2 axis 340 mm, A2 to A4 axis 400 mm, A4 to A6 axis
/// 400 mm (height to A6 shown as `(1140)`) and overall height to the flange face `(1266)`, so A6 axis
/// to flange face is 126 mm. Those become `d1 = 0.340`, `d3 = 0.400`, `d5 = 0.400` (metres, `÷ 1000`)
/// and a tool offset `Tz(0.126)`; every `a_i = 0` and every `theta` offset is 0, with
/// `alpha = [π/2, −π/2, −π/2, π/2, π/2, −π/2, 0]` as printed symbolically in Beck et al. Table 1(b).
///
/// Convention: [`DhConvention::Standard`], `T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`.
///
/// Limits (KUKA Section 4.2.2 'Axis data, LBR iiwa 7 R800', degrees → radians): A1 ±170°, A2 ±120°,
/// A3 ±170°, A4 ±120°, A5 ±170°, A6 ±120°, A7 ±175°. Speed with rated payload, deg/s → rad/s on
/// `max_velocity`: A1 98, A2 98, A3 100, A4 130, A5 140, A6 180, A7 180. Per-axis torque limits are
/// not given in the specification, so `effort` is `None` on every joint. Rated payload 7 kg, maximum
/// reach 800 mm (= 400 + 400, A2 to A6), from Section 4.2.1.
///
/// Known answer: at `q = 0` the flange is at `(0, 0, 1.266)` m. The spec entry states this is "stated
/// by the source and confirmed by hand": the `(1266)` figure is printed on Fig. 4-1, and with all
/// `a_i = 0` and `q = 0` the chain is `d1 + d3 + d5 + 0.126 = 0.340 + 0.400 + 0.400 + 0.126 = 1.266`
/// along `+z`. Flange orientation at `q = 0` is the identity (the twists sum to zero). Note the `q = 0`
/// answer is independent of the `alpha` signs taken pairwise, so the sign of each twist — which fixes
/// which way positive `q2/q4/q6` tilt the arm — was not checked against KUKA's A-axis rotation
/// direction; a comparison against controller readings at non-zero `q` must first establish that
/// mapping.
pub fn kuka_lbr_iiwa_7_r800() -> Robot {
    Robot::from_dh(&iiwa_rows(0.340, 0.400, 0.400, IIWA_7_R800_SPEEDS_DEG), DhConvention::Standard, Iso::translation(0.0, 0.0, IIWA_FLANGE_M))
        .expect("the iiwa 7 R800 table is finite and non-empty")
}

/// **KUKA LBR iiwa 14 R820**, 7 DoF, standard DH. Confidence: **derived from published geometry**.
///
/// Primary source: KUKA Roboter GmbH, 'Robots LBR iiwa: LBR iiwa 7 R800, LBR iiwa 14 R820
/// Specification', Version Spez LBR iiwa V5, Issued 28.01.2015 (Section 4.3.1 Basic data, 4.3.2 Axis
/// data, Fig. 4-4 Working envelope LBR iiwa 14 R820); DH frame convention and table structure from
/// F. Beck, M. N. Vu, C. Hartl-Nesic, A. Kugi, 'Singularity Avoidance with Application to Online
/// Trajectory Optimization for Serial Manipulators', arXiv:2211.02516v5 (accepted IFAC World Congress
/// 2023), Table 1(b) 'KUKA LBR iiwa 14 R820'.
/// KUKA Roboter GmbH, *LBR iiwa 7 R800 / LBR iiwa 14 R820 Specification* (Spez LBR iiwa, V5); read from an archived copy, <https://web.archive.org/web/20190819075249id_/http://www.oir.caltech.edu/twiki_oir/pub/Palomar/ZTF/KUKARoboticArmMaterial/Spez_LBR_iiwa_en.pdf>
///
/// KUKA publishes no DH table for this arm. The table is **derived from published geometry**: the
/// drawing it is derived from is **KUKA Fig. 4-4, Working envelope LBR iiwa 14 R820**, which in the
/// fully extended vertical pose gives base to A2 axis 360 mm, A2 to A4 axis 420 mm, A4 to A6 axis
/// 400 mm (height to A6 `(1180)`) and overall height to the flange face `(1306)`, so A6 axis to
/// flange face is 126 mm. Those become `d1 = 0.360`, `d3 = 0.420`, `d5 = 0.400` and a tool offset
/// `Tz(0.126)`. Beck et al. Table 1(b) prints the rows symbolically only — `(q1, d1, 0, π/2)`,
/// `(q2, 0, 0, −π/2)`, `(q3, d3, 0, −π/2)`, `(q4, 0, 0, π/2)`, `(q5, d5, 0, π/2)`, `(q6, 0, 0, −π/2)`,
/// `(q7, 0, 0, 0)` plus an end-effector row `(0, d, 0, 0)` — so no numeric value comes from the paper.
///
/// Convention: [`DhConvention::Standard`], `T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`.
///
/// Limits (KUKA Section 4.3.2 'Axis data, LBR iiwa 14 R820', degrees → radians): A1 ±170°, A2 ±120°,
/// A3 ±170°, A4 ±120°, A5 ±170°, A6 ±120°, A7 ±175°. Speed with rated payload, deg/s → rad/s on
/// `max_velocity`: A1 85, A2 85, A3 100, A4 75, A5 130, A6 135, A7 135. Per-axis torque limits are
/// not in the specification, so `effort` is `None`. Rated payload 14 kg, maximum reach 820 mm
/// (= 420 + 400), from Section 4.3.1.
///
/// Known answer: at `q = 0` the flange is at `(0, 0, 1.306)` m. The spec entry states this is "stated
/// by the source and confirmed by hand": the `(1306)` figure is printed on Fig. 4-4, and the chain is
/// `0.360 + 0.420 + 0.400 + 0.126 = 1.306` along `+z`, flange orientation identity. As for the
/// 7 R800, the `alpha` signs' mapping to KUKA's A-axis rotation direction was not verified; the `q = 0`
/// answer does not depend on it.
pub fn kuka_lbr_iiwa_14_r820() -> Robot {
    Robot::from_dh(&iiwa_rows(0.360, 0.420, 0.400, IIWA_14_R820_SPEEDS_DEG), DhConvention::Standard, Iso::translation(0.0, 0.0, IIWA_FLANGE_M))
        .expect("the iiwa 14 R820 table is finite and non-empty")
}

/// The six KR 5 arc rows; lengths from KUKA Fig. 4-2, limits and speeds with rated payload
/// (154, 154, 228, 343, 384, 721 deg/s) from Section 4.2, both converted with `to_radians()` and
/// attached on the row ([`DhRow::with_limits`], [`DhRow::with_max_velocity`]). See [`kuka_kr_5_arc`]
/// for the derivation and caveats.
fn kr_5_arc_rows() -> [DhRow; 6] {
    let deg = f64::to_radians;
    [
        DhRow::revolute(0.0, 0.400, 0.180, -FRAC_PI_2).with_limits(deg(-155.0), deg(155.0)).with_max_velocity(deg(154.0)),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.600, 0.0).with_limits(deg(-180.0), deg(65.0)).with_max_velocity(deg(154.0)),
        DhRow::revolute(0.0, 0.0, 0.120, -FRAC_PI_2).with_limits(deg(-15.0), deg(158.0)).with_max_velocity(deg(228.0)),
        DhRow::revolute(0.0, 0.620, 0.0, FRAC_PI_2).with_limits(deg(-350.0), deg(350.0)).with_max_velocity(deg(343.0)),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(deg(-130.0), deg(130.0)).with_max_velocity(deg(384.0)),
        DhRow::revolute(0.0, 0.115, 0.0, 0.0).with_limits(deg(-350.0), deg(350.0)).with_max_velocity(deg(721.0)),
    ]
}

/// **KUKA KR 5 arc**, 6 DoF, standard DH. Confidence: **derived from published geometry**.
///
/// Primary source: KUKA Roboter GmbH, 'Robots KR 5 arc Specification', Version Spez KR 5 arc V2,
/// Issued 23.05.2014 (Section 4.1 Basic data, 4.2 Axis data, Fig. 4-2 Working envelope, 4.3 Payloads);
/// identical axis data and dimensions in the 2011 edition (Spez KR 5 arc V1 en, issued 21.03.2011,
/// <https://www.irsrobotics.com/wp-content/uploads/2023/04/KR_5_arc_en.pdf>).
/// <https://robotosvarka.ru/upload/iblock/617/Spetsifikatsiya-KR-5-arc.pdf>
///
/// KUKA publishes no DH table for KR-series robots. The table is **derived from published geometry**:
/// the drawing it is derived from is **KUKA Fig. 4-2, Working envelope (side view, all dimensions
/// mm)**, which shows the arm vertical and the forearm horizontal with base to A2 axis 400, A1 axis to
/// A2 axis horizontal offset 180, A2 to A3 axis 600, A3 axis to forearm (A4) axis 120 (vertical
/// offset), A3/A4 to wrist centre (A5) 620, A5 to flange face 115. The lengths are primary
/// (`a = [0.180, 0.600, 0.120, 0, 0, 0]`, `d = [0.400, 0, 0, 0.620, 0, 0.115]`, metres); the frame
/// assignment — `alpha = [−π/2, 0, −π/2, π/2, −π/2, 0]` and a `theta_2` offset of `−π/2` — is the
/// spec reviewer's choice, made so that `q = 0` is the pose drawn in Fig. 4-2. `d6 = 0.115` sits in
/// row 6, so the joint-6 frame origin is on the flange face and the tool transform is identity.
///
/// Convention: [`DhConvention::Standard`], `T_i = Rz(θ_i + q_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`.
///
/// Limits (KUKA Section 4.2 Axis data, software-limited ranges, degrees → radians): A1 ±155°, A2 +65°
/// to −180°, A3 +158° to −15°, A4 ±350°, A5 ±130°, A6 ±350°; speed with rated payload, deg/s → rad/s
/// on `max_velocity`: 154, 154, 228, 343, 384, 721. These are in KUKA's axis convention; the spec
/// entry states the sign of each KUKA axis relative to this table and the controller's axis values at
/// the drawn pose were not verified, so the asymmetric A2/A3 ranges are attached as printed and may
/// need a sign/zero mapping before use against a controller. Per-axis torque limits are not in the
/// specification, so `effort` is `None`. Basic data: rated payload 5 kg, weight approx. 127 kg,
/// mounting flange DIN/ISO 9409-1-A40.
///
/// Known answer: the spec entry says it was **computed by hand** from the Fig. 4-2 dimensions and
/// confirmed against the drawing, not stated by the source as a coordinate. At `q = 0`:
/// `O1 = (0.180, 0, 0.400)`; row 2's `Rz(−π/2)` turns `x` to `+z_base` so `Tx(0.600)` climbs to
/// `O2 = (0.180, 0, 1.000)`; row 3's `Tx(0.120)` continues up to `O3 = (0.180, 0, 1.120)` and its
/// `Rx(−π/2)` points `z3` along `+x_base`; `Tz(0.620)` reaches the wrist centre `O4 = O5 =
/// (0.800, 0, 1.120)`; `Tz(0.115)` along `z5 = +x_base` puts the flange at `(0.915, 0, 1.120)` m,
/// i.e. `x = 180 + 620 + 115 = 915` mm and `z = 400 + 600 + 120 = 1120` mm. Flange orientation:
/// `z6 = +x_base`, `x6 = +z_base`, `y6 = −y_base`. Reach `180 + 600 + √(620² + 120²) = 1411.5` mm
/// matches the `R1412` in the drawing's top view.
pub fn kuka_kr_5_arc() -> Robot {
    Robot::from_dh(&kr_5_arc_rows(), DhConvention::Standard, Iso::identity()).expect("the KR 5 arc table is finite and non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// A generic, non-singular configuration used by the Hessian and numpy cross-checks; truncated to
    /// six entries for the KR 5 arc.
    const Q_GENERIC: [f64; 7] = [0.3, -0.5, 0.7, -1.1, 0.4, 0.9, -0.2];

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    /// Known-answer check shared by the three models: the fixture must vary (the flange at the generic
    /// `q` is at least 10 cm from the `q = 0` answer) before the `1e-9` m tolerance assertion.
    fn assert_known_answer(robot: &Robot, want: Vector3<f64>, dof: usize) {
        let q0 = vec![0.0; dof];
        let moved = robot.fk(&Q_GENERIC[..dof]).translation.vector;
        assert!((moved - want).norm() > 0.1, "non-vacuous fixture: the generic q must move the flange, got {moved:?}");
        let p = robot.fk(&q0).translation.vector;
        assert!(close(&p, &want, 1e-9), "FK at q = 0: {p:?} vs {want:?}");
    }

    /// Every joint carries exactly the KUKA "speed with rated payload" figure for its axis, in rad/s,
    /// through the [`DhRow::with_max_velocity`] → [`Robot::from_dh`] path: the list is compared
    /// axis by axis (a two-entry spot check would not see two rows' speeds swapped), and the fixture
    /// is non-vacuous because the list has at least two distinct values.
    fn assert_all_speeds_carried(robot: &Robot, speeds_deg_per_s: &[f64]) {
        assert_eq!(robot.dof(), speeds_deg_per_s.len());
        assert!(speeds_deg_per_s.iter().any(|s| *s != speeds_deg_per_s[0]), "non-vacuous fixture: the speeds must differ across axes");
        for (i, (j, s)) in robot.joints.iter().zip(speeds_deg_per_s).enumerate() {
            assert_eq!(j.max_velocity, Some(s.to_radians()), "axis A{} speed", i + 1);
        }
    }

    /// The convention swap: the same rows read as modified DH must move the `q = 0` flange by more
    /// than 1 mm, or the convention argument would not be load-bearing for this table.
    fn convention_swap_distance(rows: &[DhRow], tool: Iso) -> f64 {
        let std = Robot::from_dh(rows, DhConvention::Standard, tool).unwrap();
        let modified = Robot::from_dh(rows, DhConvention::Modified, tool).unwrap();
        let q0 = vec![0.0; rows.len()];
        (std.fk(&q0).translation.vector - modified.fk(&q0).translation.vector).norm()
    }

    /// The analytic kinematic Hessian against central differences of the Jacobian, copying
    /// `a_dh_arm_passes_the_hessian_finite_difference_check` in `ferromotion-core`'s `dh.rs`.
    fn assert_hessian_matches_finite_differences(robot: &Robot) {
        let q = &Q_GENERIC[..robot.dof()];
        let h = robot.kinematic_hessian(q);
        let eps = 1e-6;
        let (mut worst, mut scale) = (0.0f64, 0.0f64);
        for j in 0..q.len() {
            let (mut qp, mut qm) = (q.to_vec(), q.to_vec());
            qp[j] += eps;
            qm[j] -= eps;
            let fd = (robot.jacobian(&qp) - robot.jacobian(&qm)) / (2.0 * eps);
            for k in 0..fd.len() {
                worst = worst.max((h[j][k] - fd[k]).abs());
                scale = scale.max(fd[k].abs());
            }
        }
        assert!(scale > 0.1, "the Jacobian should vary, scale {scale:e}");
        assert!(worst < 1e-6 * scale, "Hessian vs FD: worst {worst:e} on a scale of {scale:e}");
    }

    // ---- LBR iiwa 7 R800 ----

    /// **Derived known answer, stated by the source and confirmed by hand**: the `(1266)` overall
    /// height on KUKA Fig. 4-1, `0.340 + 0.400 + 0.400 + 0.126`.
    #[test]
    fn iiwa_7_r800_derived_known_answer_at_q_zero_is_the_drawing_height_1266_mm() {
        assert_known_answer(&kuka_lbr_iiwa_7_r800(), Vector3::new(0.0, 0.0, 1.266), 7);
    }

    #[test]
    fn iiwa_7_r800_has_seven_dof_and_carries_the_published_limits_and_speeds() {
        let r = kuka_lbr_iiwa_7_r800();
        assert_eq!(r.dof(), 7);
        assert_eq!(r.joints[0].limits, Some((-170.0_f64.to_radians(), 170.0_f64.to_radians())));
        assert_eq!(r.joints[1].limits, Some((-120.0_f64.to_radians(), 120.0_f64.to_radians())));
        assert_eq!(r.joints[6].limits, Some((-175.0_f64.to_radians(), 175.0_f64.to_radians())));
        assert_eq!(r.joints[0].max_velocity, Some(98.0_f64.to_radians()));
        assert_eq!(r.joints[6].max_velocity, Some(180.0_f64.to_radians()));
        assert_all_speeds_carried(&r, &[98.0, 98.0, 100.0, 130.0, 140.0, 180.0, 180.0]);
        assert!(r.joints.iter().all(|j| j.effort.is_none()), "no per-axis torque is published, so none may be invented");
    }

    #[test]
    fn iiwa_7_r800_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences(&kuka_lbr_iiwa_7_r800());
    }

    /// Read as modified DH the first twist rotates `d1` onto `−y` and the flange lands at
    /// `(0, −0.340, 0.126)`, 1.19 m from the standard-DH answer.
    #[test]
    fn iiwa_7_r800_convention_swap_moves_the_known_answer_by_more_than_a_millimetre() {
        let dist = convention_swap_distance(&iiwa_rows(0.340, 0.400, 0.400, IIWA_7_R800_SPEEDS_DEG), Iso::translation(0.0, 0.0, IIWA_FLANGE_M));
        assert!(dist > 1e-3, "convention swap moved the flange by only {dist} m");
        assert!((dist - 1.189_621_788_637).abs() < 1e-9, "swap distance {dist} vs the hand-chained 1.189621788637");
    }

    /// The `q = 0` answer cannot see the twist signs pairwise, so this pins the flange at a generic
    /// `q` against an independent numpy product of the same table (`kuka_oracle.py`, printed to 12
    /// digits), which is where a wrong `alpha` sign would show.
    #[test]
    fn iiwa_7_r800_at_a_generic_q_matches_the_independent_numpy_chain() {
        let p = kuka_lbr_iiwa_7_r800().fk(&Q_GENERIC).translation.vector;
        let want = Vector3::new(0.064_133_711_862, -0.326_198_148_182, 0.969_899_951_495);
        assert!(close(&p, &want, 1e-9), "generic q: {p:?} vs numpy {want:?}");
    }

    // ---- LBR iiwa 14 R820 ----

    /// **Derived known answer, stated by the source and confirmed by hand**: the `(1306)` overall
    /// height on KUKA Fig. 4-4, `0.360 + 0.420 + 0.400 + 0.126`.
    #[test]
    fn iiwa_14_r820_derived_known_answer_at_q_zero_is_the_drawing_height_1306_mm() {
        assert_known_answer(&kuka_lbr_iiwa_14_r820(), Vector3::new(0.0, 0.0, 1.306), 7);
    }

    #[test]
    fn iiwa_14_r820_has_seven_dof_and_carries_the_published_limits_and_speeds() {
        let r = kuka_lbr_iiwa_14_r820();
        assert_eq!(r.dof(), 7);
        assert_eq!(r.joints[2].limits, Some((-170.0_f64.to_radians(), 170.0_f64.to_radians())));
        assert_eq!(r.joints[3].limits, Some((-120.0_f64.to_radians(), 120.0_f64.to_radians())));
        assert_eq!(r.joints[6].limits, Some((-175.0_f64.to_radians(), 175.0_f64.to_radians())));
        assert_eq!(r.joints[3].max_velocity, Some(75.0_f64.to_radians()));
        assert_eq!(r.joints[6].max_velocity, Some(135.0_f64.to_radians()));
        assert_all_speeds_carried(&r, &[85.0, 85.0, 100.0, 75.0, 130.0, 135.0, 135.0]);
        assert!(r.joints.iter().all(|j| j.effort.is_none()));
    }

    #[test]
    fn iiwa_14_r820_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences(&kuka_lbr_iiwa_14_r820());
    }

    /// Modified reading lands at `(0, −0.360 + 0.420 − 0.400, 0.126) = (0, −0.340, 0.126)`, 1.23 m away.
    #[test]
    fn iiwa_14_r820_convention_swap_moves_the_known_answer_by_more_than_a_millimetre() {
        let dist = convention_swap_distance(&iiwa_rows(0.360, 0.420, 0.400, IIWA_14_R820_SPEEDS_DEG), Iso::translation(0.0, 0.0, IIWA_FLANGE_M));
        assert!(dist > 1e-3, "convention swap moved the flange by only {dist} m");
        assert!((dist - 1.228_006_514_641).abs() < 1e-9, "swap distance {dist} vs the hand-chained 1.228006514641");
    }

    #[test]
    fn iiwa_14_r820_at_a_generic_q_matches_the_independent_numpy_chain() {
        let p = kuka_lbr_iiwa_14_r820().fk(&Q_GENERIC).translation.vector;
        let want = Vector3::new(0.073_293_966_079, -0.323_364_549_498, 1.007_451_602_733);
        assert!(close(&p, &want, 1e-9), "generic q: {p:?} vs numpy {want:?}");
    }

    // ---- KR 5 arc ----

    /// **Derived known answer, computed by hand** from the KUKA Fig. 4-2 dimensions:
    /// `x = 180 + 620 + 115`, `z = 400 + 600 + 120` mm. Also checks the flange orientation the hand
    /// chain gives (`z6 = +x_base`, `x6 = +z_base`, `y6 = −y_base`).
    #[test]
    fn kr_5_arc_derived_hand_computed_known_answer_at_q_zero_is_915_by_1120_mm() {
        let r = kuka_kr_5_arc();
        assert_known_answer(&r, Vector3::new(0.915, 0.0, 1.120), 6);
        let rot = r.fk(&[0.0; 6]).rotation;
        assert!(close(&(rot * Vector3::z()), &Vector3::x(), 1e-12), "z6 should point forward");
        assert!(close(&(rot * Vector3::x()), &Vector3::z(), 1e-12), "x6 should point up");
        assert!(close(&(rot * Vector3::y()), &-Vector3::y(), 1e-12), "y6 should point to −y");
    }

    #[test]
    fn kr_5_arc_has_six_dof_and_carries_the_published_limits_and_speeds() {
        let r = kuka_kr_5_arc();
        assert_eq!(r.dof(), 6);
        assert_eq!(r.joints[0].limits, Some((-155.0_f64.to_radians(), 155.0_f64.to_radians())));
        assert_eq!(r.joints[1].limits, Some((-180.0_f64.to_radians(), 65.0_f64.to_radians())));
        assert_eq!(r.joints[2].limits, Some((-15.0_f64.to_radians(), 158.0_f64.to_radians())));
        assert_eq!(r.joints[5].limits, Some((-350.0_f64.to_radians(), 350.0_f64.to_radians())));
        assert_eq!(r.joints[5].max_velocity, Some(721.0_f64.to_radians()));
        assert_all_speeds_carried(&r, &[154.0, 154.0, 228.0, 343.0, 384.0, 721.0]);
        assert!(r.joints.iter().all(|j| j.effort.is_none()));
    }

    #[test]
    fn kr_5_arc_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences(&kuka_kr_5_arc());
    }

    /// Modified reading puts the flange at `(0.895, 1.020, 0.120)`, 1.43 m away.
    #[test]
    fn kr_5_arc_convention_swap_moves_the_known_answer_by_more_than_a_millimetre() {
        let dist = convention_swap_distance(&kr_5_arc_rows(), Iso::identity());
        assert!(dist > 1e-3, "convention swap moved the flange by only {dist} m");
        assert!((dist - 1.428_565_714_274).abs() < 1e-9, "swap distance {dist} vs the hand-chained 1.428565714274");
    }

    #[test]
    fn kr_5_arc_at_a_generic_q_matches_the_independent_numpy_chain() {
        let p = kuka_kr_5_arc().fk(&Q_GENERIC[..6]).translation.vector;
        let want = Vector3::new(0.607_543_539_344, 0.146_158_298_379, 0.880_030_557_438);
        assert!(close(&p, &want, 1e-9), "generic q: {p:?} vs numpy {want:?}");
    }
}
