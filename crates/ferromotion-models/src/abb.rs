//! **ABB arms from their published tables**: IRB 140, IRB 120, IRB 1600-X/1.45 and one arm of the
//! YuMi IRB 14000.
//!
//! Every constructor is a [`Robot::from_dh`] call on the table its source prints, in that source's
//! convention, with the unit conversions stated on the row. ABB does not publish DH tables; the
//! kinematic rows come from papers and course material that measured or transcribed ABB's dimension
//! drawings, and the joint ranges come from ABB's own product specifications, converted here from
//! degrees with `f64::to_radians`. Where a source prints `π/2` to seven decimals it is a transcription
//! of the exact angle, and the exact constant is used.
//!
//! Each joint carries its **joint range and its datasheet speed**, the latter through
//! [`DhRow::with_max_velocity`], the same path `ur`, `kuka`, `kinova` and `others` use. Effort limits and
//! inertial parameters are not published by ABB for any of these arms and are left unset.
//!
//! ⛔ **This module said "[`DhRow`] carries a joint limit and nothing else, so the velocity limits quoted
//! in each doc comment ... are **not** attached to the model". That is false about the type**: `DhRow` has
//! `limits`, `effort` AND `max_velocity`, with `with_effort`/`with_max_velocity` builders that
//! [`Robot::from_dh`] copies onto every joint, and four sibling modules in this crate already use them.
//! The consequence was real data loss, not a wording slip: a caller reading
//! `joint.max_velocity.unwrap_or(f64::INFINITY)` got the datasheet rate for every UR, KUKA, Kinova and
//! xArm model and **unlimited** for every ABB one, while the doc told the reader the type could not carry
//! it, so they had no reason to look. No ABB test asserted `max_velocity` in either direction. The rates
//! are attached now, from the same product specifications each constructor's doc already cites, and
//! `every_abb_joint_carries_its_datasheet_speed` pins them.
//!
//! **What is verified, and against what.** For each model the tests check: the forward kinematics at
//! the all-zero pose against the position computed by hand from the table (the sources do not print
//! an end-effector position, so the oracle is a hand chain, written out in the doc comment and
//! cross-checked against a separately written numpy DH product); the degree-of-freedom count; the
//! analytic kinematic Hessian against central differences of the Jacobian; and a **convention swap**
//! — the same rows read under the other [`DhConvention`] — which must move the zero pose by more than
//! 1 mm, so that the convention argument is shown to be load-bearing for this particular table. Measured
//! swap displacements: IRB 140 0.6195 m, IRB 120 0.8155 m, IRB 1600 1.6119 m, YuMi 0.6328 m.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use nalgebra::{Translation3, UnitQuaternion};
use std::f64::consts::{FRAC_PI_2, PI};

/// A pure translation of `d` metres along the last DH frame's `z`: the flange offset a drawing gives
/// separately from the wrist centre.
fn tool_z(d: f64) -> Iso {
    Iso::from_parts(Translation3::new(0.0, 0.0, d), UnitQuaternion::identity())
}

/// The IRB 140 rows in the paper's **modified** (Craig) convention, without the flange, so the tests
/// can build the same table under either convention and with or without the tool.
/// Datasheet axis speeds (deg/s) per model, from the ABB product specifications each constructor's doc
/// cites. Attached to the rows through [`DhRow::with_max_velocity`] in radians per second.
fn with_speeds<const N: usize>(rows: [DhRow; N], deg_per_s: [f64; N]) -> Vec<DhRow> {
    rows.iter().zip(deg_per_s).map(|(r, v)| r.with_max_velocity(v.to_radians())).collect()
}

/// IRB 140-6/0.8, 'Product specification IRB 140' axis-speed table.
pub(crate) const IRB140_SPEEDS_DEG_S: [f64; 6] = [200.0, 200.0, 260.0, 360.0, 360.0, 450.0];
/// IRB 120-3/0.6, 'Product specification IRB 120' axis-speed table.
pub(crate) const IRB120_SPEEDS_DEG_S: [f64; 6] = [250.0, 250.0, 250.0, 320.0, 320.0, 420.0];
/// IRB 1600-6/1.45, 'Product specification IRB 1600' axis-speed table.
pub(crate) const IRB1600_SPEEDS_DEG_S: [f64; 6] = [150.0, 160.0, 170.0, 320.0, 400.0, 460.0];
/// IRB 14000 (YuMi), 3HAC052982-001 velocity table, in the PAPER's joint numbering. The zero and sign
/// correspondence to the controller's numbering was not verified by the specification, so this mapping is
/// **provisional in exactly the same way the joint limits above already are**.
pub(crate) const YUMI_SPEEDS_DEG_S: [f64; 7] = [180.0, 180.0, 180.0, 180.0, 400.0, 400.0, 400.0];

fn irb140_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.352, 0.0, 0.0).with_limits((-180.0f64).to_radians(), 180.0f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.070, -FRAC_PI_2).with_limits((-90.0f64).to_radians(), 110.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.360, 0.0).with_limits((-230.0f64).to_radians(), 50.0f64.to_radians()),
        DhRow::revolute(0.0, 0.380, 0.0, -FRAC_PI_2).with_limits((-200.0f64).to_radians(), 200.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2).with_limits((-115.0f64).to_radians(), 115.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits((-400.0f64).to_radians(), 400.0f64.to_radians()),
    ]
}

/// **ABB IRB 140**, 6 dof, **modified DH (Craig)**, tool frame at the flange centre.
///
/// Primary source (kinematic table): M. Almaged, 'Forward and Inverse Kinematic Analysis and Validation
/// of the ABB IRB 140 Industrial Robot', Int. J. Electronics, Mechanical and Mechatronics Engineering
/// 7(2), 2017, pp.1383-1401, Table 1 (modified D-H: d1=352, a1=70, a2=360, d4=380 mm, theta2-90);
/// limits, speeds and the 65 mm flange from ABB 'Product specification IRB 140' 3HAC041346-001 Rev. Q
/// (Dimensions p.13; Range of movement 1.8.x; Velocity 1.8.3).
/// <https://ijemme.aydin.edu.tr/wp-content/uploads/2020/04/ijemme_v07i2002.pdf>
///
/// Confidence: **published primary source**.
///
/// Rows are `(α_{i−1}, a_{i−1}, d_i, θ_i)`, millimetres in the paper converted to metres here
/// (352 → 0.352, 70 → 0.070, 360 → 0.360, 380 → 0.380); `θ₂ = q₂ − π/2`. The table ends at the wrist
/// centre (spherical wrist), so the tool transform is ABB's 65 mm axis-5-to-flange distance
/// (0.065 m) along the last frame's `z`.
///
/// Joint limits, ABB ranges in degrees converted with `to_radians`: 1 ±180, 2 +110/−90, 3 +50/−230,
/// 4 ±200 (default), 5 ±115, 6 ±400 (default). Velocities (IRB 140-6/0.8, not attached to the
/// model): 200, 200, 260, 360, 360, 450 deg/s. Effort: not published.
///
/// Known answer, **computed by hand from the table** (the paper prints no end-effector position):
/// at `q = 0` the wrist centre is `x = 0.070 + 0.380 = 0.450`, `z = 0.352 + 0.360 = 0.712` m, which
/// matches ABB's drawing (axis-1 to axis-5 = 70 + 380 mm; base to axis-2 352 mm plus 360 mm arm);
/// the flange is 0.065 m further along `x` at this pose, `(0.515, 0, 0.712)` m. Both are tested.
pub fn irb140() -> Robot {
    Robot::from_dh(&with_speeds(irb140_rows(), IRB140_SPEEDS_DEG_S), DhConvention::Modified, tool_z(0.065)).expect("the IRB 140 table is finite and non-empty")
}

/// The IRB 120 rows in **standard** DH.
fn irb120_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.290, 0.0, -FRAC_PI_2).with_limits((-165.0f64).to_radians(), 165.0f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.270, 0.0).with_limits((-110.0f64).to_radians(), 110.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.070, -FRAC_PI_2).with_limits((-110.0f64).to_radians(), 70.0f64.to_radians()),
        DhRow::revolute(0.0, 0.302, 0.0, FRAC_PI_2).with_limits((-160.0f64).to_radians(), 160.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits((-120.0f64).to_radians(), 120.0f64.to_radians()),
        DhRow::revolute(0.0, 0.072, 0.0, 0.0).with_limits((-400.0f64).to_radians(), 400.0f64.to_radians()),
    ]
}

/// **ABB IRB 120**, 6 dof, **standard DH**, tool frame at the flange centre.
///
/// Primary source: ETH Zurich Robot Dynamics course, 'Exercise 1a: Forward Kinematics of the ABB IRB
/// 120' (Hutter, Bloesch, Bellicoso, Bachmann, 2016) solution: explicit link transforms T01..T56
/// (0.145+0.145, 0.270, [0.134,0,0.070], 0.168, 0.072 m); limits and speeds from ABB 'Product
/// specification IRB 120' 3HAC035960-001 (Range of movement table; 1.8.4 Velocity).
/// <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/solution1a.pdf>
///
/// Confidence: **derived from published geometry**. The ETH solution gives the chain as explicit
/// homogeneous transforms, not a DH table; the standard-DH rows here were derived from that geometry
/// (`θ₂` offset `−π/2` because the axis-2 zero has the upper arm vertical), and the specification
/// records that the derived table reproduces the ETH transform product's end-effector position at
/// `q = 0` and at three random `q`. Lengths are already in metres in the source: `d₁ = 0.290`
/// (0.145 + 0.145), `a₂ = 0.270`, `a₃ = 0.070`, `d₄ = 0.302` (0.134 + 0.168), `d₆ = 0.072`.
///
/// Joint limits, ABB ranges in degrees converted with `to_radians`: 1 ±165, 2 ±110, 3 +70/−110,
/// 4 ±160, 5 ±120, 6 ±400 (default). Velocities (IRB 120-3/0.6, not attached to the model): 250, 250,
/// 250, 320, 320, 420 deg/s. Effort: not published.
///
/// Known answer, **computed by hand from the table** (the source prints no numeric position): at
/// `q = 0`, working back from the tip, the wrist stacks `d₄ + d₆ = 0.302 + 0.072 = 0.374` along the
/// row-3 `z`; the `α₃ = −π/2` twist and the `θ₂ = −π/2` offset turn that into world `x`, while
/// `a₂ + a₃ = 0.270 + 0.070` and `d₁ = 0.290` stack along world `z`. So `x = 0.374`,
/// `z = 0.290 + 0.270 + 0.070 = 0.630` m, the flange centre with the arm vertical; equal to the ETH
/// transform product `0.134 + 0.168 + 0.072 = 0.374`.
pub fn irb120() -> Robot {
    Robot::from_dh(&with_speeds(irb120_rows(), IRB120_SPEEDS_DEG_S), DhConvention::Standard, Iso::identity()).expect("the IRB 120 table is finite and non-empty")
}

/// The IRB 1600-X/1.45 rows in **standard** DH.
fn irb1600_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.4865, 0.150, -FRAC_PI_2).with_limits((-180.0f64).to_radians(), 180.0f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.700, 0.0).with_limits((-90.0f64).to_radians(), 120.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits((-245.0f64).to_radians(), 65.0f64.to_radians()),
        DhRow::revolute(0.0, 0.600, 0.0, FRAC_PI_2).with_limits((-200.0f64).to_radians(), 200.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits((-115.0f64).to_radians(), 115.0f64.to_radians()),
        DhRow::revolute(0.0, 0.065, 0.0, 0.0).with_limits((-400.0f64).to_radians(), 400.0f64.to_radians()),
    ]
}

/// **ABB IRB 1600-X/1.45**, 6 dof, **standard DH**, tool frame at the flange centre.
///
/// Primary source: N. A. Theissen, A. Mohammed, A. Archenti, 'Articulated industrial robots: An
/// approach to thermal compensation based on joint power consumption', euspen Laser Metrology and
/// Machine Performance XIII (LAMDAMAP 2019), Table 1 'Nominal DH parameters of the ABB IRB 1600 for
/// joint 1-4' (a=`[0,150,700,0]` mm, alpha=[0,-pi/2,0,-pi/2], theta2-pi/2, d=[486.5,0,0,600] mm);
/// joints 5-6, flange, ranges and speeds from ABB 'Product specification IRB 1600/1660'
/// 3HAC023604-001 Rev. AR (Dimensions IRB 1600-X/1.2 (1.45) p.16; Range of movement; Velocity).
/// <https://www.euspen.eu/knowledge-base/LAM19117.pdf>
///
/// Confidence: **derived from published geometry**. Rows 1-4 are Theissen's Table 1 re-indexed to
/// standard DH (their row-2 `a = 150` is the shoulder offset and row-3 `a = 700` the upper arm, so
/// `a₁ = 0.150`, `a₂ = 0.700` here); rows 5-6 (spherical wrist, `α₅ = −π/2`, `α₆ = 0`) and
/// `d₆ = 0.065` were added from ABB's drawing (axis-3 to axis-5 = 600 mm, axis-5 to flange = 65 mm,
/// base to axis-2 = 486.5 mm, upper arm 700 mm for the 1.45 m variant). Millimetres converted to
/// metres: 486.5 → 0.4865, 150 → 0.150, 700 → 0.700, 600 → 0.600, 65 → 0.065.
///
/// Joint limits for the 1.45 m variant, ABB ranges in degrees converted with `to_radians`: 1 ±180,
/// 2 +120/−90, 3 +65/−245, 4 ±200 (default), 5 ±115, 6 ±400 (default). Velocities (IRB 1600-6/1.45,
/// not attached to the model): 150, 160, 170, 320, 400, 460 deg/s. Effort: not published.
///
/// Known answer, **computed by hand from the table** (neither source prints a position): at `q = 0`,
/// `x = 0.150 + 0.600 + 0.065 = 0.815`, `z = 0.4865 + 0.700 = 1.1865` m, flange centre, arm vertical.
pub fn irb1600_1_45() -> Robot {
    Robot::from_dh(&with_speeds(irb1600_rows(), IRB1600_SPEEDS_DEG_S), DhConvention::Standard, Iso::identity()).expect("the IRB 1600 table is finite and non-empty")
}

/// One YuMi arm's rows in **standard** DH, in the paper's consecutive joint numbering.
fn yumi_rows() -> [DhRow; 7] {
    [
        DhRow::revolute(0.0, 0.166, -0.030, -FRAC_PI_2).with_limits((-168.5f64).to_radians(), 168.5f64.to_radians()),
        DhRow::revolute(0.0, 0.0, 0.030, FRAC_PI_2).with_limits((-143.5f64).to_radians(), 43.5f64.to_radians()),
        DhRow::revolute(0.0, 0.2515, 0.0405, -FRAC_PI_2).with_limits((-168.5f64).to_radians(), 168.5f64.to_radians()),
        DhRow::revolute(-FRAC_PI_2, 0.0, 0.0405, -FRAC_PI_2).with_limits((-123.5f64).to_radians(), 80.0f64.to_radians()),
        DhRow::revolute(PI, 0.265, 0.027, -FRAC_PI_2).with_limits((-290.0f64).to_radians(), 290.0f64.to_radians()),
        DhRow::revolute(0.0, 0.0, -0.027, FRAC_PI_2).with_limits((-88.0f64).to_radians(), 138.0f64.to_radians()),
        DhRow::revolute(0.0, 0.036, 0.0, 0.0).with_limits((-229.0f64).to_radians(), 229.0f64.to_radians()),
    ]
}

/// **ABB YuMi IRB 14000, one arm**, 7 dof, **standard DH**, tool frame at the paper's frame 7
/// (36 mm past axis 7; the flange offset is not given by the source and is not added).
///
/// Primary source: M. Asgari, I. A. Bonev, C. Gosselin, 'Singularities of ABB's YuMi 7-DOF robot arm',
/// Mechanism and Machine Theory 205 (2025) 105884, Table 1 'DH parameters of YuMi' and Fig. 1(b)
/// (a=30, b=40.5, c=27, d=166, e=251.5, f=265, g=36 mm); limits and speeds from ABB 'Product
/// specification IRB 14000 (YuMi)' 3HAC052982-001 section 1.8.1 and velocity table.
/// <https://espace2.etsmtl.ca/id/eprint/30327/1/Bonev-I-2025-30327.pdf>
///
/// Confidence: **published primary source**. The paper's transform is `Tz(d)·Rz(θ)·Tx(a)·Rx(α)`
/// (its eq. 1); `Tz` and `Rz` commute, so this is the standard convention. Fig. 1(b) lengths in
/// millimetres converted to metres: `a = 0.030`, `b = 0.0405`, `c = 0.027`, `d = 0.166`, `e = 0.2515`,
/// `f = 0.265`, `g = 0.036`; the signed entries `a₁ = −0.030`, `a₆ = −0.027` and the offsets
/// `θ₄ = −π/2`, `θ₅ = π` are the paper's Table 1.
///
/// Joint numbering is the paper's consecutive 1..7; the ABB controller numbers the same joints
/// 1, 2, 7, 3, 4, 5, 6. Joint limits, ABB ranges in degrees converted with `to_radians` and mapped to
/// the paper's numbering: 1 ±168.5, 2 −143.5/+43.5, 3 (ABB 7) ±168.5, 4 (ABB 3) −123.5/+80,
/// 5 (ABB 4) ±290, 6 (ABB 5) −88/+138, 7 (ABB 6) ±229. The paper warns that the real controller uses
/// a different convention, and the zero and sign correspondence between this table and the controller
/// was not verified by the specification, so the limit mapping is provisional. Velocities (not
/// attached to the model): 180 deg/s on paper joints 1-4, 400 deg/s on 5-7. Effort: not published.
///
/// Known answer, **computed by hand from the table** (the paper prints no numeric position): at
/// `q = 0`, working back from the tip, `Tx(−0.027)` and `Tx(+0.027)` cancel across the `θ₅ = π`
/// row, `d₇ = 0.036 + f = 0.265` stack to `0.301` along `x`, `b = 0.0405` twice gives
/// `x = 0.301 + 0.0405 = 0.3415` with the second `b` landing on `z`, and the `a = ±0.030` pair cancels
/// on `x`; `z = d + e + b = 0.166 + 0.2515 + 0.0405 = 0.458` m.
pub fn yumi_single_arm() -> Robot {
    Robot::from_dh(&with_speeds(yumi_rows(), YUMI_SPEEDS_DEG_S), DhConvention::Standard, Iso::identity()).expect("the YuMi table is finite and non-empty")
}

#[cfg(test)]
mod tests {

    /// **The reach ABB publishes, reproduced by the table to microns.**
    ///
    /// The IRB 120's 580 mm is from ABB's own product specification (3HAC035960); the IRB 1600's is in
    /// the model designation itself — ABB's suffix after the slash IS the reach in metres, and this
    /// crate builds the `IRB 1600-X/1.45` variant from the "Dimensions IRB 1600-X/1.2 (1.45)" drawing
    /// its constructor cites. Neither number is recomputed from this table, which is what makes them
    /// an audit of it rather than a restatement.
    ///
    /// Measured 2026-09-10 to the WRIST — `crate::envelope::wrist`, the frame `frame_pose(q, dof)`
    /// returns: 0.5800061 m against 0.580 (6.1 µm) and 1.4499999 m against 1.450 (0.1 µm). ⛔ ABB
    /// publishes reach to the wrist centre, NOT the flange: the flange envelopes are 0.652006 and
    /// 1.515 m, 72 and 65 mm further out, and match nothing. Kinova and Franka quote the flange
    /// instead, so the point has to be checked per manufacturer.
    /// ⭐ That is a different quality of agreement from the Universal Robots figures, which are rounded
    /// envelope numbers and miss by 0.7–11.5% — see `ur::the_ur_envelopes_bound_their_published_nominal_reach`.
    /// The bound here is 1 mm, matching the FANUC LR Mate precedent in `others.rs`, and the measured
    /// error is three orders inside it.
    /// ⛔ `#[ignore]`: the envelope search is a random-restart optimisation and costs seconds per arm
    /// in the debug profile the default lane uses — it took `ferromotion-models` from 1 s to 196 s.
    /// It runs in the `--release -- --ignored` lane, which `.github/workflows/ci.yml` invokes for this
    /// crate. An ignored test that no lane runs is worse than no test at all, so the two must never be
    /// separated.
    #[ignore = "envelope search: seconds per arm in debug; release --ignored lane"]
    #[test]
    fn the_abb_envelopes_are_the_published_reach_to_a_millimetre() {
        for (name, robot, want) in [
            ("IRB 120", irb120(), 0.580),
            ("IRB 1600-X/1.45", irb1600_1_45(), 1.450),
        ] {
            let e = crate::envelope::wrist(&robot);
            assert!(e > 0.3, "{name}: the envelope is a real length, {e}");
            assert!(
                (e - want).abs() < 1e-3,
                "{name}: envelope {e:.7} m vs ABB's published {want} m, off by {:.1} mm",
                (e - want).abs() * 1000.0
            );
        }
    }

    use super::*;
    use nalgebra::Vector3;

    /// FK position at `q`, as a vector.
    fn pos(r: &Robot, q: &[f64]) -> Vector3<f64> {
        r.fk(q).translation.vector
    }

    /// The known-answer check every model runs: the fixture is shown non-vacuous first (moving joint 1
    /// by 0.5 rad moves the tip by more than 50 mm, so a chain that ignored its joints could not pass),
    /// then the zero-pose position must match the hand computation to 1 nm.
    fn known_answer(name: &str, r: &Robot, want: Vector3<f64>) {
        let q0 = vec![0.0; r.dof()];
        let mut q1 = q0.clone();
        q1[0] = 0.5;
        let moved = (pos(r, &q1) - pos(r, &q0)).norm();
        assert!(moved > 0.05, "{name}: joint 1 must move the tip, moved {moved:e} m");
        let got = pos(r, &q0);
        let err = (got - want).norm();
        assert!(err < 1e-9, "{name} at q=0: got {got:?}, hand computation {want:?}, error {err:e} m");
    }

    /// The central-difference Hessian check, after `a_dh_arm_passes_the_hessian_finite_difference_check`
    /// in ferromotion-core's `dh.rs`: the analytic `kinematic_hessian` against `(J(q+ε) − J(q−ε)) / 2ε`.
    fn hessian_check(name: &str, r: &Robot, q: &[f64]) {
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
        assert!(worst < 1e-6 * scale, "{name}: Hessian vs FD, worst {worst:e} on a scale of {scale:e}");
    }

    /// The convention-swap check: the same rows under the other convention must put the zero pose
    /// somewhere else by more than 1 mm, or the convention argument would not be load-bearing.
    fn swap_moves(name: &str, rows: &[DhRow], used: DhConvention, tool: Iso, expect_m: f64) {
        let other = match used {
            DhConvention::Standard => DhConvention::Modified,
            DhConvention::Modified => DhConvention::Standard,
        };
        let a = Robot::from_dh(rows, used, tool).unwrap();
        let b = Robot::from_dh(rows, other, tool).unwrap();
        let q0 = vec![0.0; rows.len()];
        let moved = (pos(&a, &q0) - pos(&b, &q0)).norm();
        assert!(moved > 1e-3, "{name}: reading the table under {other:?} moved the zero pose by only {moved:e} m");
        // and the displacement is the one the numpy oracle measured, to the millimetre
        assert!((moved - expect_m).abs() < 1e-3, "{name}: swap displacement {moved:.4} m, oracle {expect_m:.4} m");
    }

    // ---- IRB 140 --------------------------------------------------------------------------------

    /// ⛔ **No ABB test asserted `max_velocity` in either direction**, so both the omission and a later
    /// edit attaching wrong speeds were invisible. This is the sibling of `ur.rs`'s
    /// `assert_datasheet_limits`, `kuka.rs`'s `assert_all_speeds_carried` and `others.rs`'s
    /// `assert_limits_and_speeds`.
    #[test]
    fn every_abb_joint_carries_its_datasheet_speed() {
        let check = |name: &str, r: &Robot, deg: &[f64]| {
            assert_eq!(r.dof(), deg.len(), "{name}: the speed fixture must cover every joint");
            for (i, j) in r.joints.iter().enumerate() {
                let v = j.max_velocity.unwrap_or_else(|| panic!("{name} joint {} carries no datasheet speed", i + 1));
                assert!((v - deg[i].to_radians()).abs() < 1e-12, "{name} joint {}: {v} rad/s vs {} deg/s", i + 1, deg[i]);
                assert!(v > 0.0, "{name} joint {}: a speed limit must be positive", i + 1);
                assert!(j.effort.is_none(), "{name} joint {}: ABB publishes no effort for these arms", i + 1);
                assert!(j.limits.is_some(), "{name} joint {}: the joint range must still be attached", i + 1);
            }
            eprintln!("  {name}: {} joints, speeds {:?} deg/s", r.dof(), deg);
        };
        check("IRB 140", &irb140(), &IRB140_SPEEDS_DEG_S);
        check("IRB 120", &irb120(), &IRB120_SPEEDS_DEG_S);
        check("IRB 1600-6/1.45", &irb1600_1_45(), &IRB1600_SPEEDS_DEG_S);
        check("YuMi", &yumi_single_arm(), &YUMI_SPEEDS_DEG_S);
        // and the speeds must not be all-equal placeholders: three of the four tables vary across joints
        assert!(IRB140_SPEEDS_DEG_S.iter().any(|v| *v != IRB140_SPEEDS_DEG_S[0]), "the IRB 140 table must vary by joint");
        assert!(IRB1600_SPEEDS_DEG_S.iter().any(|v| *v != IRB1600_SPEEDS_DEG_S[0]), "the IRB 1600 table must vary by joint");
    }

    #[test]
    fn irb140_zero_pose_matches_the_hand_computed_flange_and_wrist_centre() {
        known_answer("IRB 140 flange", &irb140(), Vector3::new(0.515, 0.0, 0.712));
        let wrist = Robot::from_dh(&irb140_rows(), DhConvention::Modified, Iso::identity()).unwrap();
        known_answer("IRB 140 wrist centre", &wrist, Vector3::new(0.450, 0.0, 0.712));
        // the flange offset is along the last frame's z, which at q=0 is world x: the two differ by exactly 65 mm in x
        let d = pos(&irb140(), &[0.0; 6]) - pos(&wrist, &[0.0; 6]);
        assert!((d - Vector3::new(0.065, 0.0, 0.0)).norm() < 1e-12, "flange offset in world at q=0: {d:?}");
    }

    #[test]
    fn irb140_has_six_dof_and_the_published_limits() {
        let r = irb140();
        assert_eq!(r.dof(), 6);
        let (lo, hi) = r.joints[2].limits.expect("axis 3 has an ABB range");
        assert!((lo - (-230.0f64).to_radians()).abs() < 1e-12 && (hi - 50.0f64.to_radians()).abs() < 1e-12, "axis 3: {lo} .. {hi}");
    }

    #[test]
    fn irb140_passes_the_hessian_finite_difference_check() {
        hessian_check("IRB 140", &irb140(), &[0.3, -0.5, 0.4, 0.7, -0.4, 0.2]);
    }

    #[test]
    fn irb140_read_as_standard_dh_is_a_different_arm() {
        swap_moves("IRB 140", &irb140_rows(), DhConvention::Modified, tool_z(0.065), 0.6195);
    }

    // ---- IRB 120 --------------------------------------------------------------------------------

    #[test]
    fn irb120_zero_pose_matches_the_hand_computed_flange_centre() {
        known_answer("IRB 120", &irb120(), Vector3::new(0.374, 0.0, 0.630));
    }

    #[test]
    fn irb120_has_six_dof_and_the_published_limits() {
        let r = irb120();
        assert_eq!(r.dof(), 6);
        let (lo, hi) = r.joints[2].limits.expect("axis 3 has an ABB range");
        assert!((lo - (-110.0f64).to_radians()).abs() < 1e-12 && (hi - 70.0f64.to_radians()).abs() < 1e-12, "axis 3: {lo} .. {hi}");
    }

    #[test]
    fn irb120_passes_the_hessian_finite_difference_check() {
        hessian_check("IRB 120", &irb120(), &[0.3, -0.5, 0.4, 0.7, -0.4, 0.2]);
    }

    #[test]
    fn irb120_read_as_modified_dh_is_a_different_arm() {
        swap_moves("IRB 120", &irb120_rows(), DhConvention::Standard, Iso::identity(), 0.8155);
    }

    // ---- IRB 1600-X/1.45 ------------------------------------------------------------------------

    #[test]
    fn irb1600_zero_pose_matches_the_hand_computed_flange_centre() {
        known_answer("IRB 1600-X/1.45", &irb1600_1_45(), Vector3::new(0.815, 0.0, 1.1865));
    }

    #[test]
    fn irb1600_has_six_dof_and_the_published_limits() {
        let r = irb1600_1_45();
        assert_eq!(r.dof(), 6);
        let (lo, hi) = r.joints[2].limits.expect("axis 3 has an ABB range");
        assert!((lo - (-245.0f64).to_radians()).abs() < 1e-12 && (hi - 65.0f64.to_radians()).abs() < 1e-12, "axis 3: {lo} .. {hi}");
    }

    #[test]
    fn irb1600_passes_the_hessian_finite_difference_check() {
        hessian_check("IRB 1600-X/1.45", &irb1600_1_45(), &[0.3, -0.5, 0.4, 0.7, -0.4, 0.2]);
    }

    #[test]
    fn irb1600_read_as_modified_dh_is_a_different_arm() {
        swap_moves("IRB 1600-X/1.45", &irb1600_rows(), DhConvention::Standard, Iso::identity(), 1.6119);
    }

    // ---- YuMi IRB 14000 -------------------------------------------------------------------------

    #[test]
    fn yumi_zero_pose_matches_the_hand_computed_frame_7() {
        known_answer("YuMi", &yumi_single_arm(), Vector3::new(0.3415, 0.0, 0.458));
    }

    #[test]
    fn yumi_has_seven_dof_and_the_published_limits() {
        let r = yumi_single_arm();
        assert_eq!(r.dof(), 7);
        let (lo, hi) = r.joints[1].limits.expect("paper joint 2 has an ABB range");
        assert!((lo - (-143.5f64).to_radians()).abs() < 1e-12 && (hi - 43.5f64.to_radians()).abs() < 1e-12, "joint 2: {lo} .. {hi}");
    }

    #[test]
    fn yumi_passes_the_hessian_finite_difference_check() {
        hessian_check("YuMi", &yumi_single_arm(), &[0.3, -0.5, 0.4, 0.7, -0.4, 0.2, 0.6]);
    }

    #[test]
    fn yumi_read_as_modified_dh_is_a_different_arm() {
        swap_moves("YuMi", &yumi_rows(), DhConvention::Standard, Iso::identity(), 0.6328);
    }

    /// **ABB's 559 mm for the YuMi is a bound on this table, not a millimetre check of it.**
    ///
    /// The IRB 140, 120 and 1600 reproduce ABB's published reach to microns, so the maker is capable of
    /// exactness and this crate normally demands it. The YuMi does not get that treatment, for three
    /// reasons that are all about the DOCUMENT rather than the table:
    ///
    /// 1. ⛔ **ABB's specification prints no arm link lengths.** Searched for the source paper's
    ///    Fig. 1(b) dimensions in 3HAC052982-001 Rev V — `251.5`, `166` and `40.5` mm appear nowhere in
    ///    it (read 2026-09-13). The only geometry it publishes is the 0.559 m reach and a working-range
    ///    drawing whose extents are measured from the TORSO.
    /// 2. ⚠ **It states no reference points for that 0.559 m.** For the IRB 140 the same maker gives the
    ///    axis-to-axis breakdown that fixes them; here it gives a single number.
    /// 3. ⚠ **This model is ONE ARM, based at the paper's arm-base frame.** YuMi's arms mount on an
    ///    angled dual-arm torso, so ABB's origin need not be this one. The paper's mapping onto ABB's
    ///    joint numbering is provisional for the same reason, and the limits ride on that mapping.
    ///
    /// So this is a gross-error bound in the spirit of the Universal Robots and Gen3 lite ones, set at
    /// 10% against a measured 5.84%. ⭐ It is still worth having: it is the difference between "a figure
    /// exists, it is 0.559 m, and the table sits 5.84% above it" and "nothing was found" — two states
    /// this crate deliberately keeps apart.
    ///
    /// ⚠ **And here is exactly what it catches, measured rather than assumed.** Lengthening the upper
    /// arm `e` and re-running: `+10 mm` survives, `+20 mm` survives, `+25 mm` fires. The published
    /// figure sits 32.7 mm below the envelope, so the bound has about 23 mm of headroom and that is the
    /// error it can see. It is a check against a transposed digit or a wrong unit, **not** against a
    /// millimetre transcription slip — no bound built on this document could be.
    #[ignore = "envelope search: seconds per arm in debug; release --ignored lane"]
    #[test]
    fn the_yumi_wrist_envelope_is_within_a_gross_error_of_abbs_published_559_mm() {
        let r = yumi_single_arm();
        let w = crate::envelope::wrist(&r);
        let f = crate::envelope::flange(&r);
        let rel = (w - 0.559) / 0.559;
        eprintln!("  YuMi: wrist {w:.6} m, flange {f:.6} m, against ABB's published 0.559 m ({:+.2}% at the wrist)", rel * 100.0);
        assert!(w > 0.3, "the envelope is a real length, got {w}");
        // ⛔ ABB quotes reach to the WRIST — the convention this crate records for ABB and DENSO, and
        // the reason the flange figure is printed but not asserted.
        assert!(w > 0.559, "the wrist must reach at least the published 0.559 m, got {w:.6}");
        assert!(rel < 0.10, "wrist envelope {w:.6} m is {:.2}% past the published 0.559 m", rel * 100.0);
    }
}
