//! **UFACTORY, FANUC and DENSO arms** from their published tables and drawings.
//!
//! Six constructors: the UFACTORY xArm 5, xArm 6, xArm 7 and Lite 6, whose manuals print a standard
//! D-H table outright, and the FANUC LR Mate 200iD and DENSO VS-6556, whose makers publish a dimension
//! drawing and motion-range table but no D-H table, so the table here was derived from the drawing
//! and the doc comment says so. Every constructor states its source, its convention, the unit
//! conversions applied, and the pose it is verified against.
//!
//! **What is verified, and against what.** For each arm: forward kinematics at the mechanical zero
//! against a position computed by hand from the same table (the manuals do not print a pose; the
//! hand figure is an independent sum of the link lengths along the two non-zero axes); the
//! degree-of-freedom count and the limits and speeds carried through; the analytic kinematic Hessian
//! against central differences of the Jacobian; and a **convention swap** — the same rows read as
//! Craig's modified convention — which must move the zero pose by more than 1 mm, proving the
//! `DhConvention` argument is load-bearing for this table. All six tables move it by 0.25–0.94 m.
//!
//! **Units.** Every source prints millimetres and degrees; every value here is metres and radians.
//! Working ranges are written as the datasheet's degrees with `.to_radians()`, joint speeds as the
//! datasheet's deg/s with `.to_radians()`, and lengths as the datasheet's millimetres divided by 1000
//! (written out as metres). No source in this module publishes joint torques, so `effort` is `None`
//! on every joint.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use std::f64::consts::{FRAC_PI_2, PI};

/// Build the robot and stamp each joint's `max_velocity` (rad/s) from the datasheet. `from_dh`
/// returns `None` only for an empty or non-finite table, which a constant table in this file cannot be.
fn build(name: &str, rows: &[DhRow], convention: DhConvention, velocities_rad_s: &[f64]) -> Robot {
    let mut robot = Robot::from_dh(rows, convention, Iso::identity()).unwrap_or_else(|| panic!("{name}: the DH table is constant, finite and non-empty"));
    for (joint, &v) in robot.joints.iter_mut().zip(velocities_rad_s) {
        joint.max_velocity = Some(v);
    }
    robot
}

// ---------------------------------------------------------------------------------------------------
// UFACTORY xArm 5 / 6 / 7 — shared geometry from the xArm User Manual V2.3.0, Appendix 8
// ---------------------------------------------------------------------------------------------------

/// Base to joint-2 axis, `d1 = 267 mm` (manual pp.197, 201, 205), in metres.
const XARM_D1: f64 = 0.267;
/// Wrist-pitch link, `a = 76 mm` (row 4 of xArm 5, row 5 of xArm 6, row 6 of xArm 7), in metres.
const XARM_A_WRIST: f64 = 0.076;
/// Wrist-pitch axis to flange, `d = 97 mm` (last row of all three), in metres.
const XARM_D_FLANGE: f64 = 0.097;
/// Elbow rise, `a3 = 77.5 mm` (xArm 6 row 3, xArm 7 row 4), in metres.
const XARM_A_ELBOW: f64 = 0.0775;
/// Forearm, `d4 = 342.5 mm` (xArm 6 row 4, xArm 7 row 5), in metres.
const XARM_D_FOREARM: f64 = 0.3425;
/// Table 1.1 (manual p.9): maximum joint speed 180°/s on every joint of all three arms.
const XARM_SPEED_RAD_S: f64 = PI;

/// The xArm 5/6 upper-arm offset `T2_offset = -atan(284.5/53.5)` (manual pp.197, 202), radians.
fn xarm_t2() -> f64 {
    -(284.5f64 / 53.5).atan()
}

/// The xArm 5/6 upper-arm length `a2 = sqrt(284.5² + 53.5²)` (manual pp.197, 202), in metres.
///
/// The manual prints the result of its own formula as `289.48866` mm; the formula evaluates to
/// `289.486614…` mm, 2.05 µm shorter. The formula is used, because the manual's `T2_offset`, its
/// figure of the mechanical zero (upper arm 284.5 mm up and 53.5 mm forward) and the hand-computed
/// zero pose are all consistent with the formula and not with the printed decimal.
fn xarm_a2() -> f64 {
    284.5f64.hypot(53.5) / 1000.0
}

/// **UFACTORY xArm 5**, 5 revolute joints, standard D-H.
///
/// **Source (primary):** "xArm User Manual V2.3.0 (2024): Appendix 8 'DH Parameters of xArm Series'
/// pp.195-208 (xArm5 modified and standard D-H tables, T-offsets, mass parameters); Table 1.1 working
/// range and max speed p.10", <https://www.ufactory.cc/wp-content/uploads/2024/01/xArm-User-Manual-V2.3.0.pdf>.
/// Confidence: **published primary source**. Convention: [`DhConvention::Standard`] (the manual's
/// "Standard D-H Parameters" table, p.197; it also prints a modified table, which the spec reports
/// gives the identical pose at `q = 0` and at random `q`).
///
/// **Table (manual p.197, mm and rad → m and rad):**
/// `d1 = 267, α1 = -π/2; a2 = √(284.5²+53.5²), θ2 = -atan(284.5/53.5); a3 = √(77.5²+342.5²) =
/// 351.158796 mm, θ3 = atan(284.5/53.5) + atan(0.3425/0.0775) = 2.7331843 rad; a4 = 76, α4 = -π/2,
/// θ4 = -atan(342.5/77.5) = -1.3482664 rad; d5 = 97`. The `θ` column is the manual's `Tx_offset`,
/// "the offset joint angle from the mathematical zero position to the mechanical zero position", so
/// `q = 0` is the arm as drawn. See [`xarm_a2`] for the 2 µm discrepancy in the manual's printed `a2`.
///
/// **Limits (Table 1.1 p.9, degrees → radians):** J1 ±360°, J2 -118°…120°, J3 -225°…11°, J4 -97°…180°,
/// J5 ±360°. **Speed:** 180°/s on every joint → π rad/s. **Effort:** not published; left `None`.
///
/// **Known answer, computed by hand from the table (the manual prints no pose):** at the mechanical
/// zero `q = 0` the flange is at `(0.207, 0, 0.112)` m — `x = 53.5 + 77.5 + 76 = 207` mm, `z = 267 +
/// 284.5 - 342.5 - 97 = 112` mm. Verified by the FK test to 1e-9 m.
pub fn xarm5() -> Robot {
    build("xArm 5", &xarm5_rows(), DhConvention::Standard, &[XARM_SPEED_RAD_S; 5])
}

fn xarm5_rows() -> [DhRow; 5] {
    let t2 = xarm_t2();
    let t3 = (284.5f64 / 53.5).atan() + (0.3425f64 / 0.0775).atan();
    let t4 = -(342.5f64 / 77.5).atan();
    let a3 = 77.5f64.hypot(342.5) / 1000.0;
    [
        DhRow::revolute(0.0, XARM_D1, 0.0, -FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(t2, 0.0, xarm_a2(), 0.0).with_limits(-118f64.to_radians(), 120f64.to_radians()),
        DhRow::revolute(t3, 0.0, a3, 0.0).with_limits(-225f64.to_radians(), 11f64.to_radians()),
        DhRow::revolute(t4, 0.0, XARM_A_WRIST, -FRAC_PI_2).with_limits(-97f64.to_radians(), 180f64.to_radians()),
        DhRow::revolute(0.0, XARM_D_FLANGE, 0.0, 0.0).with_limits(-360f64.to_radians(), 360f64.to_radians()),
    ]
}

/// **UFACTORY xArm 6**, 6 revolute joints, standard D-H.
///
/// **Source (primary):** "xArm User Manual V2.3.0 (2024): Appendix 8 'DH Parameters of xArm Series'
/// pp.200-204 (xArm 6 modified and standard D-H tables; a2=289.48866 mm, T2=-1.3849179 rad, T3=-T2;
/// mass parameters); Table 1.1 working range and max speed p.10",
/// <https://www.ufactory.cc/wp-content/uploads/2024/01/xArm-User-Manual-V2.3.0.pdf>.
/// Confidence: **published primary source**. Convention: [`DhConvention::Standard`] (manual p.201).
///
/// **Table (manual p.201, mm and rad → m and rad):** `d1 = 267, α1 = -π/2; a2 = √(284.5²+53.5²),
/// θ2 = -atan(284.5/53.5); a3 = 77.5, α3 = -π/2, θ3 = -θ2; d4 = 342.5, α4 = π/2; a5 = 76, α5 = -π/2;
/// d6 = 97`. See [`xarm_a2`] for the 2 µm discrepancy in the manual's printed `a2`.
///
/// **Limits (Table 1.1 p.9, degrees → radians):** J1 ±360°, J2 -118°…120°, J3 -225°…11°, J4 ±360°,
/// J5 -97°…180°, J6 ±360°. **Speed:** 180°/s every joint → π rad/s. **Effort:** not published; `None`.
///
/// **Known answer, computed by hand from the table (the manual prints no pose):** at `q = 0` the
/// flange is at `(0.207, 0, 0.112)` m — `x = 53.5 + 77.5 + 76 = 207` mm, `z = 267 + 284.5 - 342.5 -
/// 97 = 112` mm, the same point as the xArm 5, whose wrist is the xArm 6's with joint 4 removed.
/// Verified by the FK test to 1e-9 m.
pub fn xarm6() -> Robot {
    build("xArm 6", &xarm6_rows(), DhConvention::Standard, &[XARM_SPEED_RAD_S; 6])
}

fn xarm6_rows() -> [DhRow; 6] {
    let t2 = xarm_t2();
    [
        DhRow::revolute(0.0, XARM_D1, 0.0, -FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(t2, 0.0, xarm_a2(), 0.0).with_limits(-118f64.to_radians(), 120f64.to_radians()),
        DhRow::revolute(-t2, 0.0, XARM_A_ELBOW, -FRAC_PI_2).with_limits(-225f64.to_radians(), 11f64.to_radians()),
        DhRow::revolute(0.0, XARM_D_FOREARM, 0.0, FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(0.0, 0.0, XARM_A_WRIST, -FRAC_PI_2).with_limits(-97f64.to_radians(), 180f64.to_radians()),
        DhRow::revolute(0.0, XARM_D_FLANGE, 0.0, 0.0).with_limits(-360f64.to_radians(), 360f64.to_radians()),
    ]
}

/// **UFACTORY xArm 7**, 7 revolute joints, standard D-H.
///
/// **Source (primary):** "xArm User Manual V2.3.0 (2024): Appendix 8 'DH Parameters of xArm Series'
/// pp.204-208 (xArm7 modified and standard D-H tables; mass parameters); Table 1.1 working range and
/// max speed p.10", <https://www.ufactory.cc/wp-content/uploads/2024/01/xArm-User-Manual-V2.3.0.pdf>.
/// Confidence: **published primary source**. Convention: [`DhConvention::Standard`] (manual pp.205-206).
///
/// **Table (manual pp.205-206, mm and rad → m and rad; every θ offset is 0):** `d1 = 267, α1 = -π/2;
/// α2 = π/2; d3 = 293, a3 = 52.5, α3 = π/2; a4 = 77.5, α4 = π/2; d5 = 342.5, α5 = π/2; a6 = 76,
/// α6 = -π/2; d7 = 97`.
///
/// **Limits (Table 1.1 p.9, degrees → radians):** J1 ±360°, J2 -118°…120°, J3 ±360°, J4 -11°…225°,
/// J5 ±360°, J6 -97°…180°, J7 ±360°. **Speed:** 180°/s every joint → π rad/s. **Effort:** not
/// published; `None`.
///
/// **Known answer, computed by hand from the table (the manual prints no pose):** at `q = 0` the
/// flange is at `(0.206, 0, 0.1205)` m — `x = 52.5 + 77.5 + 76 = 206` mm, `z = 267 + 293 - 342.5 -
/// 97 = 120.5` mm. Verified by the FK test to 1e-9 m.
pub fn xarm7() -> Robot {
    build("xArm 7", &xarm7_rows(), DhConvention::Standard, &[XARM_SPEED_RAD_S; 7])
}

fn xarm7_rows() -> [DhRow; 7] {
    [
        DhRow::revolute(0.0, XARM_D1, 0.0, -FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2).with_limits(-118f64.to_radians(), 120f64.to_radians()),
        DhRow::revolute(0.0, 0.293, 0.0525, FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(0.0, 0.0, XARM_A_ELBOW, FRAC_PI_2).with_limits(-11f64.to_radians(), 225f64.to_radians()),
        DhRow::revolute(0.0, XARM_D_FOREARM, 0.0, FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(0.0, 0.0, XARM_A_WRIST, -FRAC_PI_2).with_limits(-97f64.to_radians(), 180f64.to_radians()),
        DhRow::revolute(0.0, XARM_D_FLANGE, 0.0, 0.0).with_limits(-360f64.to_radians(), 360f64.to_radians()),
    ]
}

/// **UFACTORY Lite 6**, 6 revolute joints, standard D-H.
///
/// **Source (primary):** "UFACTORY Lite 6 User Manual V2.0.0 (2023): Appendix 7 'Kinematic and Dynamic
/// Parameters of UFACTORY Lite 6' pp.203-207 (modified and standard D-H, mass parameters); Table 1.1
/// working range and Table 1.2 motion parameters; specification table p.202",
/// <https://www.ufactory.cc/wp-content/uploads/2023/05/Lite6-User-Manual-V2.0.0.pdf>.
/// Confidence: **published primary source**. Convention: [`DhConvention::Standard`] (the manual's
/// "Standard D-H Parameters" table, printed with θ offset and α in **degrees**, d and a in mm).
///
/// **Table (manual Appendix 7, deg and mm → rad and m):** `d1 = 243.3, α1 = -90°; θ2 = -90°,
/// a2 = 200, α2 = 180°; θ3 = -90°, a3 = 87, α3 = 90°; d4 = 227.6, α4 = 90°; α5 = -90°; d6 = 61.5`.
///
/// **Limits (Table 1.1, degrees → radians):** J1 ±360°, J2 ±150°, J3 -3.5°…300°, J4 ±360°, J5 ±124°,
/// J6 ±360°. **Speed:** Table 1.2 gives joint motion 0…180°/s, used here as π rad/s on every joint;
/// the appendix specification table prints "Maximum Joint Speed 90°/s" for the same arm, so the
/// smaller figure is a documented alternative and the controller is the authority. **Effort:** not
/// published; `None`.
///
/// **Known answer, computed by hand from the table (the manual prints no pose):** at `q = 0` the
/// flange is at `(0.087, 0, 0.1542)` m — `x = a3 = 87` mm, `z = 243.3 + 200 - 227.6 - 61.5 = 154.2`
/// mm (the α2 = 180° twist reverses the later `d` offsets). Verified by the FK test to 1e-9 m.
pub fn lite6() -> Robot {
    build("Lite 6", &lite6_rows(), DhConvention::Standard, &[PI; 6])
}

fn lite6_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.2433, 0.0, -FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.2, PI).with_limits(-150f64.to_radians(), 150f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.087, FRAC_PI_2).with_limits(-3.5f64.to_radians(), 300f64.to_radians()),
        DhRow::revolute(0.0, 0.2276, 0.0, FRAC_PI_2).with_limits(-360f64.to_radians(), 360f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(-124f64.to_radians(), 124f64.to_radians()),
        DhRow::revolute(0.0, 0.0615, 0.0, 0.0).with_limits(-360f64.to_radians(), 360f64.to_radians()),
    ]
}

// ---------------------------------------------------------------------------------------------------
// FANUC LR Mate 200iD — derived from the data-sheet drawing
// ---------------------------------------------------------------------------------------------------

/// **FANUC LR Mate 200iD**, 6 revolute joints, standard D-H, **derived from published geometry**.
///
/// **Source (primary):** "FANUC America 'LR Mate 200iD Data Sheet' (dimension drawing: J1-J2 offset 50,
/// base-to-J2 330, J2-J3 330, J3 rise 35, J3-J5 335, J5-flange 80 mm, reach 717; motion range J1
/// 340(360 opt), J2 245, J3 420, J4 380, J5 250, J6 720 deg; max speed 450, 380, 520, 550, 545, 1000
/// deg/s; J4-J6 moments 16.6/16.6/9.4 N*m)",
/// <https://www.fanucamerica.com/docs/default-source/robotics-files/lr-mate-200id-data-sheet.pdf>.
/// Confidence: **derived from published geometry** — FANUC publishes no D-H table; the spec derived
/// this one by hand from the drawing with a conventional frame assignment (joint 1 vertical, joints 2
/// and 3 parallel pitch axes, wrist 4-5-6 with intersecting axes) and `θ2 = -π/2` so that `q = 0` is
/// FANUC's home posture (upper arm vertical, forearm horizontal). Convention: [`DhConvention::Standard`].
///
/// **Table (drawing mm → m):** `d1 = 330, a1 = 50, α1 = -π/2; θ2 = -π/2, a2 = 330; a3 = 35, α3 = -π/2;
/// d4 = 335, α4 = π/2; α5 = -π/2; d6 = 80`. Reach check from the spec: `50 + 330 + √(335² + 35²) =
/// 716.8` mm against the data sheet's 717 mm.
///
/// **Limits (data sheet, degrees → radians):** J1 ±170° and J2 -100°…145° read from the drawing;
/// J4 ±190°, J5 ±125°, J6 ±360° assume the data sheet's totals (380, 250, 720) are symmetric; **J3 is
/// left unset** because the data sheet gives only a 420° total and the split was not resolved.
/// **Speed (data sheet deg/s → rad/s):** 450, 380, 520, 550, 545, 1000. **Effort:** the data sheet's
/// J4-J6 figures are allowable wrist *moments*, not actuator torques, so `effort` is `None`.
///
/// **Known answer, computed by hand from the table (the data sheet prints no pose):** at `q = 0` the
/// flange is at `(0.465, 0, 0.695)` m — `x = 50 + 335 + 80 = 465` mm, `z = 330 + 330 + 35 = 695` mm.
/// Verified by the FK test to 1e-9 m. FANUC reports J3 relative to the horizontal (coupled with J2);
/// this table uses link-relative angles, and the controller conversion was not verified.
pub fn fanuc_lr_mate_200id() -> Robot {
    let speeds = [450f64, 380.0, 520.0, 550.0, 545.0, 1000.0].map(f64::to_radians);
    build("FANUC LR Mate 200iD", &fanuc_lr_mate_200id_rows(), DhConvention::Standard, &speeds)
}

fn fanuc_lr_mate_200id_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.33, 0.05, -FRAC_PI_2).with_limits(-170f64.to_radians(), 170f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.33, 0.0).with_limits(-100f64.to_radians(), 145f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.035, -FRAC_PI_2), // J3 total 420°, split unresolved: no limit
        DhRow::revolute(0.0, 0.335, 0.0, FRAC_PI_2).with_limits(-190f64.to_radians(), 190f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(-125f64.to_radians(), 125f64.to_radians()),
        DhRow::revolute(0.0, 0.08, 0.0, 0.0).with_limits(-360f64.to_radians(), 360f64.to_radians()),
    ]
}

// ---------------------------------------------------------------------------------------------------
// DENSO VS-6556 — derived from the technical specification and dimension drawing
// ---------------------------------------------------------------------------------------------------

/// **DENSO VS-6556**, 6 revolute joints, standard D-H, **derived from published geometry**.
///
/// **Source (primary):** "DENSO Robotics 'Technical Specifications VS-6556/6577' (arm lengths 270+295
/// mm, J1 arm offset 75 mm, J3 forearm offset 90 mm, motion ranges, max joint speeds; external
/// dimension drawing VS-6556-B); DH structure follows W. Khawla, 'Forward Kinematic Analysis of Denso
/// VS-6577 Robot Manipulator', JEET 8(1) 2021, Table 1 (symbolic standard DH for the sister VS-6577)",
/// <https://www.densorobotics-europe.com/fileadmin/Products/VS-6556/VS-6556_and_VS_6577_technical_Data_Sheet_-_Copy.pdf>.
/// Confidence: **derived from published geometry** — DENSO publishes no numeric D-H table. Convention:
/// [`DhConvention::Standard`] (the frame structure of Khawla's symbolic table, `d1, a2, a3, d4, d6`).
///
/// **Table (specification mm → m):** `a1 = 75` (J1 arm offset), `a2 = 270` (No.1 arm), `a3 = 90` (J3
/// forearm offset), `d4 = 295` (No.2 arm), all from the specification text; `d1 = 335` (mounting face
/// to J2 axis) and `d6 = 80` (J5 centre to flange face; the drawing's 375 = 295 + 80) read by the spec
/// from the VS-6556-B dimension drawing at 400 dpi, and stated there as needing confirmation against
/// the DENSO robot manual. Twists: `α1 = -π/2, α3 = -π/2, α4 = π/2, α5 = -π/2`; `θ2 = -π/2` so that
/// `q = 0` is upper arm vertical, forearm horizontal.
///
/// **Limits (specification, degrees → radians):** J1 ±170°, J2 -100°…135°, J3 -119°…166°, J4 ±190°,
/// J5 ±120°, J6 ±360°. **Speed (specification deg/s → rad/s):** 262.5, 240, 300, 300, 300, 480.
/// **Effort:** not published; `None`. Zero/sign correspondence with the DENSO controller's joint
/// angles (J3 in particular) was not verified.
///
/// **Known answer, computed by hand from the table (the specification prints no pose):** at `q = 0`
/// the flange is at `(0.45, 0, 0.695)` m — `x = 75 + 295 + 80 = 450` mm, `z = 335 + 270 + 90 = 695`
/// mm. Verified by the FK test to 1e-9 m.
pub fn denso_vs6556() -> Robot {
    let speeds = [262.5f64, 240.0, 300.0, 300.0, 300.0, 480.0].map(f64::to_radians);
    build("DENSO VS-6556", &denso_vs6556_rows(), DhConvention::Standard, &speeds)
}

fn denso_vs6556_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.335, 0.075, -FRAC_PI_2).with_limits(-170f64.to_radians(), 170f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.27, 0.0).with_limits(-100f64.to_radians(), 135f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.09, -FRAC_PI_2).with_limits(-119f64.to_radians(), 166f64.to_radians()),
        DhRow::revolute(0.0, 0.295, 0.0, FRAC_PI_2).with_limits(-190f64.to_radians(), 190f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(-120f64.to_radians(), 120f64.to_radians()),
        DhRow::revolute(0.0, 0.08, 0.0, 0.0).with_limits(-360f64.to_radians(), 360f64.to_radians()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// The pose every known-answer test is held to, in metres.
    const KNOWN_ANSWER_TOL: f64 = 1e-9;

    /// **Known-answer check with a non-vacuity guard.** Before comparing against `want`, the fixture
    /// proves the pose actually depends on `q`: moving every joint by 0.3 rad must move the flange by
    /// more than 1 mm, so a constructor that ignored its table could not pass on a constant.
    fn assert_known_answer(name: &str, r: &Robot, q: &[f64], want: Vector3<f64>) {
        assert!(want.norm() > 0.05, "{name}: the reference pose is not at the origin");
        let perturbed: Vec<f64> = q.iter().map(|v| v + 0.3).collect();
        let moved = (r.fk(&perturbed).translation.vector - r.fk(q).translation.vector).norm();
        assert!(moved > 1e-3, "{name}: the pose must depend on q, moved {moved:e}");
        let p = r.fk(q).translation.vector;
        let err = (p - want).norm();
        assert!(err < KNOWN_ANSWER_TOL, "{name}: FK {p:?} vs hand-computed {want:?}, error {err:e} m");
    }

    /// Limits and speeds carried through: every row with a limit has `lower < upper` on its joint,
    /// every joint has a speed, and no joint has an effort (none is published in this module).
    fn assert_limits_and_speeds(name: &str, r: &Robot, rows: &[DhRow], dof: usize) {
        assert_eq!(r.dof(), dof, "{name}: dof");
        assert_eq!(rows.len(), dof, "{name}: rows");
        let with_limits = rows.iter().filter(|row| row.limits.is_some()).count();
        assert!(with_limits >= dof - 1, "{name}: at most one joint may be without a limit, {with_limits} have one");
        for (i, (joint, row)) in r.joints.iter().zip(rows).enumerate() {
            assert_eq!(joint.limits, row.limits, "{name}: joint {i} limits");
            if let Some((lo, hi)) = joint.limits {
                assert!(lo < hi, "{name}: joint {i} limit ({lo}, {hi})");
            }
            let v = joint.max_velocity.unwrap_or_else(|| panic!("{name}: joint {i} has no speed"));
            assert!(v > 0.0 && v.is_finite(), "{name}: joint {i} speed {v}");
            assert!(joint.effort.is_none(), "{name}: joint {i} effort is not published and must stay None");
        }
    }

    /// The analytic Hessian against central differences of the Jacobian, the pattern of
    /// `a_dh_arm_passes_the_hessian_finite_difference_check` in `ferromotion_core::dh`.
    fn assert_hessian_matches_finite_differences(name: &str, r: &Robot, q: &[f64]) {
        assert_eq!(q.len(), r.dof(), "{name}: q length");
        let h = r.kinematic_hessian(q);
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

    /// **Convention swap.** The same rows read under the other convention must put the known-answer
    /// pose somewhere else by more than 1 mm, which is what makes the `DhConvention` argument
    /// load-bearing for this table. Returns the distance so the test can print it.
    fn assert_convention_swap_moves(name: &str, rows: &[DhRow], convention: DhConvention, q: &[f64]) -> f64 {
        let other = match convention {
            DhConvention::Standard => DhConvention::Modified,
            DhConvention::Modified => DhConvention::Standard,
        };
        let as_built = Robot::from_dh(rows, convention, Iso::identity()).unwrap();
        let swapped = Robot::from_dh(rows, other, Iso::identity()).unwrap();
        assert_eq!(as_built.dof(), swapped.dof(), "{name}: the swap keeps the joint count");
        let p = as_built.fk(q).translation.vector;
        let s = swapped.fk(q).translation.vector;
        assert!(s.iter().all(|v| v.is_finite()), "{name}: swapped pose {s:?}");
        let moved = (s - p).norm();
        assert!(moved > 1e-3, "{name}: reading the table as {other:?} moved the pose by only {moved:e} m");
        moved
    }

    const Q7: [f64; 7] = [0.3, -0.5, 0.7, -0.4, 0.6, 0.2, -0.3];

    // ---- xArm 5 -------------------------------------------------------------------------------

    #[test]
    fn xarm5_zero_pose_matches_the_hand_computation_from_the_manual_table() {
        assert_known_answer("xArm 5", &xarm5(), &[0.0; 5], Vector3::new(0.207, 0.0, 0.112));
    }

    #[test]
    fn xarm5_has_five_dof_with_the_manual_limits_and_speeds() {
        assert_limits_and_speeds("xArm 5", &xarm5(), &xarm5_rows(), 5);
    }

    #[test]
    fn xarm5_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences("xArm 5", &xarm5(), &Q7[..5]);
    }

    #[test]
    fn xarm5_read_as_modified_dh_is_a_different_arm() {
        let moved = assert_convention_swap_moves("xArm 5", &xarm5_rows(), DhConvention::Standard, &[0.0; 5]);
        assert!(moved > 0.3, "measured 0.308 m when written; now {moved}");
    }

    // ---- xArm 6 -------------------------------------------------------------------------------

    #[test]
    fn xarm6_zero_pose_matches_the_hand_computation_from_the_manual_table() {
        assert_known_answer("xArm 6", &xarm6(), &[0.0; 6], Vector3::new(0.207, 0.0, 0.112));
    }

    #[test]
    fn xarm6_has_six_dof_with_the_manual_limits_and_speeds() {
        assert_limits_and_speeds("xArm 6", &xarm6(), &xarm6_rows(), 6);
    }

    #[test]
    fn xarm6_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences("xArm 6", &xarm6(), &Q7[..6]);
    }

    #[test]
    fn xarm6_read_as_modified_dh_is_a_different_arm() {
        let moved = assert_convention_swap_moves("xArm 6", &xarm6_rows(), DhConvention::Standard, &[0.0; 6]);
        assert!(moved > 0.4, "measured 0.465 m when written; now {moved}");
    }

    // ---- xArm 7 -------------------------------------------------------------------------------

    #[test]
    fn xarm7_zero_pose_matches_the_hand_computation_from_the_manual_table() {
        assert_known_answer("xArm 7", &xarm7(), &[0.0; 7], Vector3::new(0.206, 0.0, 0.1205));
    }

    #[test]
    fn xarm7_has_seven_dof_with_the_manual_limits_and_speeds() {
        assert_limits_and_speeds("xArm 7", &xarm7(), &xarm7_rows(), 7);
    }

    #[test]
    fn xarm7_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences("xArm 7", &xarm7(), &Q7);
    }

    #[test]
    fn xarm7_read_as_modified_dh_is_a_different_arm() {
        let moved = assert_convention_swap_moves("xArm 7", &xarm7_rows(), DhConvention::Standard, &[0.0; 7]);
        assert!(moved > 0.3, "measured 0.384 m when written; now {moved}");
    }

    // ---- Lite 6 -------------------------------------------------------------------------------

    #[test]
    fn lite6_zero_pose_matches_the_hand_computation_from_the_manual_table() {
        assert_known_answer("Lite 6", &lite6(), &[0.0; 6], Vector3::new(0.087, 0.0, 0.1542));
    }

    #[test]
    fn lite6_has_six_dof_with_the_manual_limits_and_speeds() {
        assert_limits_and_speeds("Lite 6", &lite6(), &lite6_rows(), 6);
    }

    #[test]
    fn lite6_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences("Lite 6", &lite6(), &Q7[..6]);
    }

    #[test]
    fn lite6_read_as_modified_dh_is_a_different_arm() {
        let moved = assert_convention_swap_moves("Lite 6", &lite6_rows(), DhConvention::Standard, &[0.0; 6]);
        assert!(moved > 0.2, "measured 0.249 m when written; now {moved}");
    }

    // ---- FANUC LR Mate 200iD ---------------------------------------------------------------------

    #[test]
    fn fanuc_lr_mate_200id_home_pose_matches_the_hand_computation_from_the_drawing_derived_table() {
        assert_known_answer("FANUC LR Mate 200iD", &fanuc_lr_mate_200id(), &[0.0; 6], Vector3::new(0.465, 0.0, 0.695));
    }

    /// The spec's reach check: `a1 + a2 + √(d4² + a3²)` against the data sheet's 717 mm, as a second
    /// independent number from the same drawing.
    #[test]
    fn fanuc_lr_mate_200id_reach_from_the_table_is_the_data_sheet_reach_to_a_millimetre() {
        let rows = fanuc_lr_mate_200id_rows();
        let reach = rows[0].a + rows[1].a + rows[3].d.hypot(rows[2].a);
        assert!(reach > 0.7, "the reach is a real length, {reach}");
        assert!((reach - 0.717).abs() < 1e-3, "reach {reach} m vs data sheet 0.717 m");
    }

    #[test]
    fn fanuc_lr_mate_200id_has_six_dof_with_j3_unlimited() {
        let r = fanuc_lr_mate_200id();
        assert_limits_and_speeds("FANUC LR Mate 200iD", &r, &fanuc_lr_mate_200id_rows(), 6);
        assert!(r.joints[2].limits.is_none(), "J3's 420° total has no published split, so it must stay unset");
        assert!(r.joints[0].limits.is_some() && r.joints[5].limits.is_some());
    }

    #[test]
    fn fanuc_lr_mate_200id_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences("FANUC LR Mate 200iD", &fanuc_lr_mate_200id(), &Q7[..6]);
    }

    #[test]
    fn fanuc_lr_mate_200id_read_as_modified_dh_is_a_different_arm() {
        let moved = assert_convention_swap_moves("FANUC LR Mate 200iD", &fanuc_lr_mate_200id_rows(), DhConvention::Standard, &[0.0; 6]);
        assert!(moved > 0.9, "measured 0.937 m when written; now {moved}");
    }

    // ---- DENSO VS-6556 --------------------------------------------------------------------------

    #[test]
    fn denso_vs6556_home_pose_matches_the_hand_computation_from_the_drawing_derived_table() {
        assert_known_answer("DENSO VS-6556", &denso_vs6556(), &[0.0; 6], Vector3::new(0.45, 0.0, 0.695));
    }

    #[test]
    fn denso_vs6556_has_six_dof_with_the_specification_limits_and_speeds() {
        assert_limits_and_speeds("DENSO VS-6556", &denso_vs6556(), &denso_vs6556_rows(), 6);
    }

    #[test]
    fn denso_vs6556_passes_the_hessian_finite_difference_check() {
        assert_hessian_matches_finite_differences("DENSO VS-6556", &denso_vs6556(), &Q7[..6]);
    }

    #[test]
    fn denso_vs6556_read_as_modified_dh_is_a_different_arm() {
        let moved = assert_convention_swap_moves("DENSO VS-6556", &denso_vs6556_rows(), DhConvention::Standard, &[0.0; 6]);
        assert!(moved > 0.8, "measured 0.874 m when written; now {moved}");
    }

    /// The xArm 5 and xArm 6 share the upper arm, and the manual's `a2` formula and printed decimal
    /// disagree by 2.05 µm; this pins which one the code uses (the formula) and the size of the gap.
    #[test]
    fn xarm_a2_is_the_manuals_formula_not_its_printed_decimal() {
        let a2 = xarm_a2();
        assert!((a2 - 0.289_486_614_5).abs() < 1e-10, "formula √(284.5²+53.5²) = 289.4866145 mm, got {a2}");
        let printed = 0.289_488_66;
        let gap = (printed - a2).abs();
        assert!(gap > 2.0e-6 && gap < 2.1e-6, "the manual's printed 289.48866 mm is 2.05 µm longer, gap {gap:e}");
    }
}
