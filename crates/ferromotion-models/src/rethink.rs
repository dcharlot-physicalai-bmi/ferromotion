//! **Rethink Robotics arms** — Baxter (one arm) and Sawyer — built from published tables.
//!
//! The two arms come in different conventions and with different provenance, and the docs on each
//! constructor say which. [`baxter`] is a **modified (Craig) DH** table transcribed from a published
//! primary source. [`sawyer`] is a **standard DH** table *derived* from published product-of-exponentials
//! geometry, not transcribed from a document; its test therefore carries the paper's own
//! product-of-exponentials forward kinematics as an independent oracle at random configurations, in
//! addition to the hand-computed home pose.
//!
//! Every number below states where it came from. Neither source gives effort or velocity limits, and
//! the Sawyer source gives no joint limits; those stay unset (`None`) rather than being invented.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use nalgebra::{Translation3, UnitQuaternion};
use std::f64::consts::FRAC_PI_2;

/// A pure `z` translation, used for the Baxter gripper offset along the wrist frame's `z₇`.
fn tz(d: f64) -> Iso {
    Iso::from_parts(Translation3::new(0.0, 0.0, d), UnitQuaternion::identity())
}

/// The Baxter arm's DH rows without the gripper offset, so a test can check the wrist-centre pose the
/// source states and the constructor can add the tool on top. Numbers documented on [`baxter`].
fn baxter_rows() -> [DhRow; 7] {
    // Williams' Table 4 joint limits, printed in degrees with his positive sense (see [`baxter`]);
    // converted here with `f64::to_radians`, which is the only unit conversion applied to the limits.
    let lim = |lo_deg: f64, hi_deg: f64| (lo_deg.to_radians(), hi_deg.to_radians());
    let (l1, l2, l3, l4, l5, l6, l7) =
        (lim(-141.0, 51.0), lim(-123.0, 60.0), lim(-173.0, 173.0), lim(-3.0, 150.0), lim(-175.0, 175.0), lim(-90.0, 120.0), lim(-175.0, 175.0));
    [
        // Each modified-DH row is (θ_i offset, d_i, a_{i−1}, α_{i−1}): the `a` and `α` describe the
        // link BEFORE joint i, which is what `DhConvention::Modified` expects.
        DhRow::revolute(0.0, 0.0, 0.0, 0.0).with_limits(l1.0, l1.1),
        DhRow::revolute(FRAC_PI_2, 0.0, 0.069, -FRAC_PI_2).with_limits(l2.0, l2.1),
        DhRow::revolute(0.0, 0.36435, 0.0, FRAC_PI_2).with_limits(l3.0, l3.1),
        DhRow::revolute(0.0, 0.0, 0.069, -FRAC_PI_2).with_limits(l4.0, l4.1),
        DhRow::revolute(0.0, 0.37429, 0.0, FRAC_PI_2).with_limits(l5.0, l5.1),
        DhRow::revolute(0.0, 0.0, 0.010, -FRAC_PI_2).with_limits(l6.0, l6.1),
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2).with_limits(l7.0, l7.1),
    ]
}

/// Gripper-finger centre offset along `z₇` from the wrist-pitch centre: Williams' L6 = 368.30 mm,
/// converted to metres (0.3683 m).
const BAXTER_TOOL_Z_M: f64 = 0.3683;

/// **Rethink Robotics Baxter, one 7-DoF arm (left and right are identical tables).**
///
/// **Convention: modified DH (Craig)** — each row is `(α_{i−1}, a_{i−1}, d_i, θ_i)`, built with
/// [`DhConvention::Modified`]. The convention-swap test below shows that reading this table as
/// standard DH moves the home pose by 1.56 m, so the argument is load-bearing.
///
/// **Primary source (verbatim from the specification):** "R. L. Williams II, 'Baxter Humanoid Robot
/// Kinematics', Ohio University, 2017: Tables 2-3 (Craig-convention DH, left and right arm identical),
/// Table 4 joint limits, Table 5 link lengths L0-L6; cross-checked with arXiv:2409.00867 'Kinematics &
/// Dynamics Library for Baxter Arm' Table I (standard DH, same lengths)",
/// <https://sites.ohio.edu/williams/html/PDF/BaxterKinematics.pdf>.
/// **Confidence: derived from a teaching document** (R. L. Williams II's Baxter kinematics notes), not a manufacturer table or a peer-reviewed paper.
///
/// **Table** (Williams' Table 2 with Table 5 lengths; lengths in metres, angles in radians):
///
/// | i | α_{i−1} | a_{i−1} (m) | d_i (m) | θ_i offset |
/// |---|---|---|---|---|
/// | 1 | 0 | 0 | 0 | 0 |
/// | 2 | −π/2 | 0.069 (L1) | 0 | +π/2 |
/// | 3 | +π/2 | 0 | 0.36435 (L2) | 0 |
/// | 4 | −π/2 | 0.069 (L3) | 0 | 0 |
/// | 5 | +π/2 | 0 | 0.37429 (L4) | 0 |
/// | 6 | −π/2 | 0.010 (L5) | 0 | 0 |
/// | 7 | +π/2 | 0 | 0 | 0 |
///
/// **Unit conversions applied.** Williams' 90° twists and the +90° offset on θ₂ are entered as the exact
/// `FRAC_PI_2` (the specification's truncated `1.5707963` would shift the home pose by 3.0e-8 m, measured
/// with an independent NumPy chain, which is why the exact constant is used against a 1e-9 m
/// tolerance). Link lengths are Williams' millimetres in metres (L1 = 69 mm → 0.069 m, and so on). Joint
/// limits are Williams' Table 4 in degrees, `[-141,+51], [-123,+60], [-173,+173], [-3,+150],
/// [-175,+175], [-90,+120], [-175,+175]`, converted with `f64::to_radians`; the specification notes
/// that the positive sense of Williams' joint variables is opposite to Rethink's, so these intervals are
/// in *his* convention, which is also the convention of the table. The tool is L6 = 368.30 mm =
/// 0.3683 m along `z₇`, from the wrist-pitch centre to the gripper-finger centre.
///
/// **Not set, because the source does not give them:** effort and velocity limits (the specification
/// says they live on the Rethink SDK wiki, which was not reachable during its review). The 0.27035 m
/// base-to-shoulder rise and the 45° base yaw of each arm are Williams' Figure 5/6, outside the table,
/// and are not included: the base frame here is his `{B}` at the S0 shoulder axis.
///
/// **Known-answer pose, computed by hand from the table** (the specification's own wording: "Computed
/// from Williams' Table 2 ... at q=0, expressed in the arm base frame {B} at the S0 shoulder axis"):
/// at `q = 0` the wrist centre, frame `{7}`, is at `(0.80764, 0, −0.079)` m, and adding L6 along `z₇`
/// gives the gripper-finger centre at `(1.17594, 0, −0.079)` m. Walking the frames: L1 and L2 and L4 lie
/// along base `x` (0.069 + 0.36435 + 0.37429 = 0.80764), while L3 and L5 lie along base `−z`
/// (0.069 + 0.010 = 0.079); note the specification's prose lists all five lengths in the `x` sum, which
/// would be 0.88664, but its stated value 0.80764 is the one the table produces. Verified in tests
/// against that hand computation to 1e-9 m for both the wrist centre (identity tool) and the gripper
/// (this constructor); the same chain evaluated independently in NumPy agreed to 1e-16 m.
pub fn baxter() -> Robot {
    Robot::from_dh(&baxter_rows(), DhConvention::Modified, tz(BAXTER_TOOL_Z_M)).expect("Baxter's DH table is non-empty and finite")
}

/// Sawyer rows, standard DH, derived from the paper's product-of-exponentials data. Documented on
/// [`sawyer`]; kept separate so the convention-swap test can rebuild the same rows.
fn sawyer_rows() -> [DhRow; 7] {
    // Each standard-DH row is (θ_i offset, d_i, a_i, α_i). No limits: the source gives none.
    [
        DhRow::revolute(0.0, 0.0, 0.081, -FRAC_PI_2),
        DhRow::revolute(FRAC_PI_2, 0.1925, 0.0, FRAC_PI_2),
        DhRow::revolute(0.0, 0.400, 0.0, -FRAC_PI_2),
        DhRow::revolute(0.0, -0.1685, 0.0, FRAC_PI_2),
        DhRow::revolute(0.0, 0.400, 0.0, -FRAC_PI_2),
        DhRow::revolute(0.0, 0.1363, 0.0, FRAC_PI_2),
        DhRow::revolute(0.0, 0.0, 0.0, 0.0),
    ]
}

/// **Rethink Robotics Sawyer, 7 DoF, to the wrist centre.**
///
/// **Convention: standard DH**, built with [`DhConvention::Standard`]. This table is **derived**, not
/// transcribed: the source publishes joint axes and inter-joint vectors (product-of-exponentials data),
/// and the specification derived the DH rows from them by hand, choosing `x_i = −z` for `i ≥ 2`, which
/// is where the `+π/2` offset on θ₂ comes from. The convention-swap test shows that reading these rows
/// as modified DH moves the home pose by 1.62 m.
///
/// **Primary source (verbatim from the specification):** "A. J. Elias and J. T. Wen, 'Redundancy
/// parameterization and inverse kinematics of 7-DOF revolute manipulators', arXiv:2307.13122 (v2),
/// Section 6.4, eq. (101): Sawyer kinematic parameters (joint axes h_i and inter-joint vectors p_ij in
/// mm, zero configuration), citing Rethink Robotics (2022)", <https://arxiv.org/pdf/2307.13122>.
/// **Confidence: derived from published geometry.**
///
/// **The source geometry** (eq. 101; millimetres, zero configuration, base frame): axes `h₁ = z`,
/// `h₂ = h₄ = h₆ = y`, `h₃ = h₅ = h₇ = x`; vectors `p₁₂ = (81, 192.5, 0)`, `p₃₄ = (400, −168.5, 0)`,
/// `p₅₆ = (400, 136.3, 0)`, `p₂₃ = p₄₅ = p₆₇ = 0`, tool at the wrist centre (`p₇T = 0`).
///
/// **Table** (derived; metres and radians):
///
/// | i | θ_i offset | d_i (m) | a_i (m) | α_i |
/// |---|---|---|---|---|
/// | 1 | 0 | 0 | 0.081 | −π/2 |
/// | 2 | +π/2 | 0.1925 | 0 | +π/2 |
/// | 3 | 0 | 0.400 | 0 | −π/2 |
/// | 4 | 0 | −0.1685 | 0 | +π/2 |
/// | 5 | 0 | 0.400 | 0 | −π/2 |
/// | 6 | 0 | 0.1363 | 0 | +π/2 |
/// | 7 | 0 | 0 | 0 | 0 |
///
/// **Unit conversions applied.** The paper's millimetres are entered in metres (81 mm → 0.081 m,
/// 192.5 mm → 0.1925 m, 400 mm → 0.400 m, 168.5 mm → 0.1685 m, 136.3 mm → 0.1363 m). Right-angle twists
/// and the θ₂ offset are the exact `FRAC_PI_2` (the specification's truncated `1.5707963` would shift
/// the home pose by 3.3e-8 m, measured with an independent NumPy chain). The tool is identity: frame 7
/// is the wrist centre, and the paper gives no flange or gripper offset.
///
/// **Not set, because the source does not give them:** joint limits, effort limits and velocity limits.
/// The specification records unverified web-snippet ranges for Sawyer and deliberately does not enter
/// them; neither does this constructor.
///
/// **Known-answer pose, computed by hand from the source geometry** (the specification: "At q=0 the
/// wrist centre is p07 = p12+p34+p56 = (881, 160.3, 0) mm"), i.e. `(0.881, 0.1603, 0)` m. Verified in
/// tests to 1e-9 m. Because the table is derived, the tests also carry the paper's own
/// product-of-exponentials forward kinematics as an independent oracle: DH and POE positions agree to
/// 1e-12 m at eight non-degenerate configurations (the NumPy version of the same comparison agreed to
/// 2.5e-16 m over twenty random configurations).
pub fn sawyer() -> Robot {
    Robot::from_dh(&sawyer_rows(), DhConvention::Standard, Iso::identity()).expect("Sawyer's DH table is non-empty and finite")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Unit, Vector3};

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    /// The central-difference Hessian check every `Robot` is held to, copied from
    /// `a_dh_arm_passes_the_hessian_finite_difference_check` in `ferromotion-core`'s `dh.rs`.
    fn hessian_matches_central_differences(r: &Robot, q: &[f64]) {
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
        assert!(scale > 0.1, "the Jacobian should vary, scale {scale:e}");
        assert!(worst < 1e-6 * scale, "Hessian vs FD: worst {worst:e} on a scale of {scale:e}");
    }

    const Q0: [f64; 7] = [0.0; 7];
    /// A non-degenerate configuration used wherever a fixture must be shown to actually vary.
    const Q_BENT: [f64; 7] = [0.3, -0.5, 0.7, 0.9, -0.4, 0.6, -1.1];

    // ---------------------------------------------------------------- Baxter

    /// Known answer from the specification's hand computation on Williams' Table 2 (modified DH): the
    /// wrist centre at `q = 0` is `(0.80764, 0, −0.079)` m and the gripper-finger centre, L6 = 0.3683 m
    /// further along `z₇`, is `(1.17594, 0, −0.079)` m.
    #[test]
    fn baxter_home_pose_matches_the_pose_computed_by_hand_from_williams_table() {
        let gripper = baxter();
        let wrist = Robot::from_dh(&baxter_rows(), DhConvention::Modified, Iso::identity()).unwrap();
        // non-vacuous: the pose depends on q, so a constructor that ignored q would be caught
        let home = gripper.fk(&Q0).translation.vector;
        let bent = gripper.fk(&Q_BENT).translation.vector;
        assert!((home - bent).norm() > 0.1, "the pose must vary with q: home {home:?}, bent {bent:?}");

        let p7 = wrist.fk(&Q0).translation.vector;
        assert!(close(&p7, &Vector3::new(0.80764, 0.0, -0.079), 1e-9), "wrist centre: {p7:?}");
        assert!(close(&home, &Vector3::new(1.17594, 0.0, -0.079), 1e-9), "gripper centre: {home:?}");
        // and the tool really is along z₇, which at home points along base +x
        let z7 = wrist.fk(&Q0).rotation * Vector3::z();
        assert!(close(&z7, &Vector3::x(), 1e-12), "z7 at home: {z7:?}");
    }

    #[test]
    fn baxter_has_seven_joints_with_williams_limits() {
        let r = baxter();
        assert_eq!(r.dof(), 7);
        // The specification's rounded radians (Williams' degrees), in units of 1e-4 rad so they are
        // an independent transcription rather than the same `to_radians` the constructor applied —
        // checked so that a limit on the wrong row, or with the wrong sign, is caught.
        let want_1e4: [(i32, i32); 7] = [(-24609, 8901), (-21468, 10472), (-30194, 30194), (-524, 26180), (-30543, 30543), (-15708, 20944), (-30543, 30543)];
        for (i, (j, (lo, hi))) in r.joints.iter().zip(want_1e4).enumerate() {
            let (lo, hi) = (f64::from(lo) * 1e-4, f64::from(hi) * 1e-4);
            let (l, h) = j.limits.unwrap_or_else(|| panic!("joint {i} must carry Williams' limit"));
            assert!((l - lo).abs() < 1e-4 && (h - hi).abs() < 1e-4, "joint {i} limits ({l}, {h}) vs spec ({lo}, {hi})");
            assert!(l < h, "joint {i} interval must be non-empty");
        }
    }

    #[test]
    fn baxter_passes_the_hessian_finite_difference_check() {
        hessian_matches_central_differences(&baxter(), &Q_BENT);
    }

    /// The convention argument is load-bearing: the same rows read as standard DH build a different arm.
    /// Measured displacement of the home pose: 1.555 m.
    #[test]
    fn baxter_read_as_standard_dh_is_a_different_arm() {
        let right = baxter().fk(&Q0).translation.vector;
        let swapped = Robot::from_dh(&baxter_rows(), DhConvention::Standard, tz(BAXTER_TOOL_Z_M)).unwrap();
        let wrong = swapped.fk(&Q0).translation.vector;
        let moved = (right - wrong).norm();
        assert!(moved > 1e-3, "convention swap moved the home pose by only {moved:e} m: {right:?} vs {wrong:?}");
    }

    // ---------------------------------------------------------------- Sawyer

    /// Known answer from the specification's hand computation on the paper's geometry:
    /// `p₀₇ = p₁₂ + p₃₄ + p₅₆ = (881, 160.3, 0)` mm at `q = 0`.
    #[test]
    fn sawyer_home_pose_matches_the_pose_computed_by_hand_from_the_paper_geometry() {
        let r = sawyer();
        let home = r.fk(&Q0).translation.vector;
        let bent = r.fk(&Q_BENT).translation.vector;
        assert!((home - bent).norm() > 0.1, "the pose must vary with q: home {home:?}, bent {bent:?}");
        assert!(close(&home, &Vector3::new(0.881, 0.1603, 0.0), 1e-9), "wrist centre: {home:?}");
    }

    #[test]
    fn sawyer_has_seven_joints_and_no_invented_limits() {
        let r = sawyer();
        assert_eq!(r.dof(), 7);
        assert!(r.joints.iter().all(|j| j.limits.is_none()), "the source gives no limits, so none may be set");
    }

    #[test]
    fn sawyer_passes_the_hessian_finite_difference_check() {
        hessian_matches_central_differences(&sawyer(), &Q_BENT);
    }

    /// Measured displacement of the home pose when the rows are read as modified DH: 1.616 m.
    #[test]
    fn sawyer_read_as_modified_dh_is_a_different_arm() {
        let right = sawyer().fk(&Q0).translation.vector;
        let wrong = Robot::from_dh(&sawyer_rows(), DhConvention::Modified, Iso::identity()).unwrap().fk(&Q0).translation.vector;
        let moved = (right - wrong).norm();
        assert!(moved > 1e-3, "convention swap moved the home pose by only {moved:e} m: {right:?} vs {wrong:?}");
    }

    /// The paper's own forward kinematics, written from its eq. (101) data rather than from the DH
    /// table: `p = R₁(p₁₂ + R₂(p₂₃ + ⋯ R₇ p₇T))` with `R_i` the rotation by `q_i` about `h_i`, all in the
    /// zero-configuration base frame. Because the DH table is derived, this is the oracle that ties it
    /// back to the source, at configurations where every joint is engaged.
    fn sawyer_poe_position(q: &[f64; 7]) -> Vector3<f64> {
        let (x, y, z) = (Vector3::x(), Vector3::y(), Vector3::z());
        let axes = [z, y, x, y, x, y, x];
        let p = [
            Vector3::new(0.081, 0.1925, 0.0),
            Vector3::zeros(),
            Vector3::new(0.400, -0.1685, 0.0),
            Vector3::zeros(),
            Vector3::new(0.400, 0.1363, 0.0),
            Vector3::zeros(),
            Vector3::zeros(),
        ];
        let mut tip = Vector3::zeros();
        for i in (0..7).rev() {
            tip = Rotation3::from_axis_angle(&Unit::new_normalize(axes[i]), q[i]) * (p[i] + tip);
        }
        tip
    }

    #[test]
    fn sawyer_dh_agrees_with_the_papers_product_of_exponentials_at_bent_configurations() {
        let r = sawyer();
        let qs: [[f64; 7]; 8] = [
            Q_BENT,
            [1.0, 0.2, -0.3, 1.4, 0.5, -0.6, 0.7],
            [-0.8, 1.1, 0.9, -1.2, -1.3, 0.4, 2.0],
            [2.0, -1.5, 1.7, 0.3, 1.9, -1.8, -0.2],
            [0.1, 0.9, -1.9, 2.1, -0.7, 1.3, 1.5],
            [-1.4, -0.9, 0.6, 1.8, 1.1, 0.8, -1.7],
            [0.5, 1.6, 2.2, -0.4, -1.6, -1.1, 0.9],
            [1.3, -0.2, -1.4, 0.8, 0.3, 1.9, -0.5],
        ];
        // non-vacuous: the oracle itself moves between configurations and away from home
        let home = sawyer_poe_position(&Q0);
        assert!(close(&home, &Vector3::new(0.881, 0.1603, 0.0), 1e-12), "POE home: {home:?}");
        let mut spread = 0.0f64;
        for q in &qs {
            spread = spread.max((sawyer_poe_position(q) - home).norm());
        }
        assert!(spread > 0.3, "the POE oracle should move the wrist well away from home, spread {spread}");
        let mut worst = 0.0f64;
        for q in &qs {
            let dh = r.fk(q).translation.vector;
            let poe = sawyer_poe_position(q);
            worst = worst.max((dh - poe).norm());
            assert!(close(&dh, &poe, 1e-12), "DH vs POE at {q:?}: {dh:?} vs {poe:?}");
        }
        assert!(worst < 1e-12, "worst DH-vs-POE gap {worst:e}");
    }
}
