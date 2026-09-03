//! **Universal Robots arms, from the tables Universal Robots publishes**: UR3, UR5 and UR10 (CB3 series), and
//! UR3e, UR5e, UR10e, UR16e and UR20 (e-Series).
//!
//! Every table here is transcribed from one manufacturer document, *Universal Robots support article: DH
//! Parameters for calculations of kinematics and dynamics*,
//! <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>
//! (confidence: published primary source). The article defines its parameters by reference to the classic
//! Denavit–Hartenberg definition ("UR a parameter = Wikipedia r parameter") and lists no theta offsets, so every
//! row is `DhRow::revolute(0, d, a, α)` read as [`DhConvention::Standard`]. The article already states `d` and `a`
//! in metres and `α` in radians; **no unit conversion was applied to any DH entry**. All eight tables share the
//! twist pattern `α = (π/2, 0, 0, π/2, −π/2, 0)`, carry the two link lengths on rows 2 and 3 with a negative sign,
//! and end frame 6 at the tool-flange centre (the controller's all-zero TCP), so the tool transform is identity.
//! The DH base frame coincides with the controller's *Base* feature frame, whose +x points away from the
//! stretched-out arm — an inference the specification records from the UR ROS 2 driver's *Robot Frames* page,
//! not a statement of the article.
//!
//! Joint ranges and speeds come from the Universal Robots datasheet named on each constructor, converted from
//! that sheet's degrees: `±360° → ±2π rad`, `180°/s → π rad/s`, `120°/s → 2π/3 rad/s`, `150°/s → 5π/6 rad/s`,
//! `210°/s → 7π/6 rad/s`, `360°/s → 2π rad/s`. Ranges are attached as joint limits and speeds as
//! [`ferromotion_core::Joint::max_velocity`]. Where a datasheet says the tool flange rotates without limit, the
//! wrist-3 limit is left unset. **Effort is unset on every joint of every model**: no Universal Robots datasheet,
//! user-manual technical-specification page or the DH article states a joint torque.
//!
//! # What is verified, against what
//!
//! No Universal Robots document located by the specification's review states an end-effector position for a
//! named joint configuration, so every known answer is **computed by hand from the table, not stated by the
//! source**. At `q = 0` the standard-DH product `Π Rz(0)·Tz(dᵢ)·Tx(aᵢ)·Rx(αᵢ)` collapses to
//! `p = (a₂ + a₃, −(d₄ + d₆), d₁ − d₅)` with `R = [[1,0,0],[0,0,−1],[0,1,0]]`, and at the teach-pendant home
//! pose `q = (0, −π/2, 0, −π/2, 0, 0)` to `p = (0, −(d₄ + d₆), d₁ − a₂ − a₃ + d₅)`. Those two poses see the
//! table only through `a₂ + a₃` and `d₄ + d₆`, so a third, **bent** pose `q = (0, 0, π/2, 0, π/2, 0)` is held
//! too: walking the chain from the flange, `Rx(−π/2)·(0,0,d₆) = (0,d₆,0)`, `Tz(d₅)`, `Rz(π/2)` → `(−d₆, 0, d₅)`;
//! `Rx(π/2)`, `Tz(d₄)` → `(−d₆, −d₅, d₄)`; `Tx(a₃)`, `Rz(π/2)` → `(d₅, a₃ − d₆, d₄)`; `Tx(a₂)` → `(a₂ + d₅,
//! a₃ − d₆, d₄)`; `Rx(π/2)`, `Tz(d₁)` → `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆)`, which separates `a₂` from `a₃`
//! and `d₄` from `d₆` (swapping either pair leaves `q = 0` exactly where it was and moves this pose by 0.04
//! to 0.19 m, measured). All three closed forms were checked against an independent 4×4 matrix chain (numpy),
//! which agreed to 1.2e-16 m on all eight models.
//! Each constructor's tests hold [`Robot::fk`] to those three poses and that rotation within 1e-9, hold the
//! analytic Hessian to central differences of the Jacobian, and read the same rows as
//! [`DhConvention::Modified`] to show the convention argument is load-bearing: measured, the swap moves the
//! `q = 0` pose by 0.128 m (UR5) to 0.339 m (UR20), never less.
//!
//! **Cross-check.** An independent second transcription of the same article (UR3e, UR5e, UR10e, UR16e and
//! UR5 in full, plus the UR3 and UR10 `d` and `a` columns) agrees with every `d`, `a` and theta entry exactly,
//! with every `α` to 2.7e-8 rad (it rounds π/2 to seven decimals), and with every known answer exactly. It
//! differs in one non-geometric entry: it leaves the UR16e wrist-3 range unset where the e-Series collective
//! datasheet's ±360° is used here; its own UR5e and UR10e entries carry that ±360°.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// `2π/3` rad/s, the datasheets' 120°/s.
const DEG_120_PER_S: f64 = 2.0 * PI / 3.0;
/// `5π/6` rad/s, the UR20 sheet's 150°/s.
const DEG_150_PER_S: f64 = 5.0 * PI / 6.0;
/// `7π/6` rad/s, the UR20 sheet's 210°/s.
const DEG_210_PER_S: f64 = 7.0 * PI / 6.0;

/// Build a UR arm: the six rows as standard DH with an identity tool, then the datasheet speed on each joint.
///
/// Every table here is six finite revolute rows, which is the only thing [`Robot::from_dh`] can refuse, so the
/// `expect` cannot fire on the constants in this module.
fn build(rows: &[DhRow; 6], speeds: [f64; 6]) -> Robot {
    // speeds ride the rows through `DhRow::with_max_velocity`, the one path every model in this crate uses
    let rows: Vec<DhRow> = rows.iter().zip(speeds).map(|(r, v)| r.with_max_velocity(v)).collect();
    Robot::from_dh(&rows, DhConvention::Standard, Iso::identity()).expect("six finite DH rows")
}

// ----------------------------------------------------------------------------------------------------------
// CB3 series
// ----------------------------------------------------------------------------------------------------------

/// **UR3** (CB3), 6 revolute joints, standard DH, identity tool.
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics (UR3
/// kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`] (the article's classic DH).
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.1519 | 0 | π/2 |
/// | 2 | 0 | −0.24365 | 0 |
/// | 3 | 0 | −0.21325 | 0 |
/// | 4 | 0.11235 | 0 | π/2 |
/// | 5 | 0.08535 | 0 | −π/2 |
/// | 6 | 0.0819 | 0 | 0 |
///
/// Limits and speeds: UR3 (CB3) Technical Specifications sheet, item no. 110103, EN 10/2016,
/// <https://www.universal-robots.com/media/240736/ur3_en.pdf> — joints 1–5 ±360° (±2π rad), wrist 3 "Infinite
/// rotation on end joint" so its limit is unset; speed 180°/s (π rad/s) on base, shoulder, elbow and 360°/s
/// (2π rad/s) on the three wrists. Effort: unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (a₂ + a₃, −(d₄ + d₆), d₁ − d₅) = (−0.4569, −0.19425, 0.06655)` m; at the home pose
/// `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.19425, 0.69415)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.1583, −0.11235, −0.14325)` m. The second sourcing states the same `q = 0`
/// position.
pub fn ur3() -> Robot {
    build(&ur3_rows(), [PI, PI, PI, TAU, TAU, TAU])
}

fn ur3_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.1519, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.24365, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.21325, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.11235, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.08535, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0819, 0.0, 0.0),
    ]
}

/// **UR5** (CB3), 6 revolute joints, standard DH, identity tool.
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics (UR5
/// kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.089159 | 0 | π/2 |
/// | 2 | 0 | −0.425 | 0 |
/// | 3 | 0 | −0.39225 | 0 |
/// | 4 | 0.10915 | 0 | π/2 |
/// | 5 | 0.09465 | 0 | −π/2 |
/// | 6 | 0.0823 | 0 | 0 |
///
/// Limits and speeds: UR5 (CB3) technical specification sheet dated 06/2023,
/// <https://www.universal-robots.com/media/1828033/ur5_tech_spec_web_en.pdf> — every joint ±360° (±2π rad)
/// at 180°/s (π rad/s). Effort: unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.425 − 0.39225, −(0.10915 + 0.0823), 0.089159 − 0.09465) = (−0.81725, −0.19145, −0.005491)` m; at the
/// home pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.19145, 1.001059)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.33035, −0.10915, −0.385391)` m. Cross-checked entry by entry
/// against the independent second transcription: every `d`, `a`, limit and the known answer agree.
pub fn ur5() -> Robot {
    build(&ur5_rows(), [PI; 6])
}

fn ur5_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.089159, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.425, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.39225, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.10915, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.09465, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0823, 0.0, 0.0).with_limits(-TAU, TAU),
    ]
}

/// **UR10** (CB3), 6 revolute joints, standard DH, identity tool.
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics (UR10
/// kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.1273 | 0 | π/2 |
/// | 2 | 0 | −0.612 | 0 |
/// | 3 | 0 | −0.5723 | 0 |
/// | 4 | 0.163941 | 0 | π/2 |
/// | 5 | 0.1157 | 0 | −π/2 |
/// | 6 | 0.0922 | 0 | 0 |
///
/// Limits and speeds: UR10 (CB3) Technical Specifications sheet, item no. 110110, EN 09/2016,
/// <https://www.universal-robots.com/media/50895/ur10_en.pdf> — every joint ±360° (±2π rad); base and shoulder
/// 120°/s (2π/3 rad/s); elbow and the three wrists 180°/s (π rad/s). Effort: unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.612 − 0.5723, −(0.163941 + 0.0922), 0.1273 − 0.1157) = (−1.1843, −0.256141, 0.0116)` m; at the
/// home pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.256141, 1.4273)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.4963, −0.163941, −0.5372)` m. The second sourcing states the
/// same `d`, `a` columns and `q = 0` position.
pub fn ur10() -> Robot {
    build(&ur10_rows(), [DEG_120_PER_S, DEG_120_PER_S, PI, PI, PI, PI])
}

fn ur10_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.1273, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.612, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.5723, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.163941, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.1157, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0922, 0.0, 0.0).with_limits(-TAU, TAU),
    ]
}

// ----------------------------------------------------------------------------------------------------------
// e-Series
// ----------------------------------------------------------------------------------------------------------

/// **UR3e**, 6 revolute joints, standard DH, identity tool.
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics (UR3e
/// kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.15185 | 0 | π/2 |
/// | 2 | 0 | −0.24355 | 0 |
/// | 3 | 0 | −0.2132 | 0 |
/// | 4 | 0.13105 | 0 | π/2 |
/// | 5 | 0.08535 | 0 | −π/2 |
/// | 6 | 0.0921 | 0 | 0 |
///
/// Limits and speeds: e-Series collective technical specification 11/2023,
/// <https://www.universal-robots.com/media/1829346/11_2023_collective_data-sheet.pdf> — base, shoulder, elbow
/// ±360° (±2π rad) at 180°/s (π rad/s); wrist 1 and 2 ±360° at 360°/s (2π rad/s); wrist 3 "Infinite" at 360°/s,
/// so its limit is unset. Effort: unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.24355 − 0.2132, −(0.13105 + 0.0921), 0.15185 − 0.08535) = (−0.45675, −0.22315, 0.0665)` m; at the
/// home pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.22315, 0.69395)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.1582, −0.13105, −0.15345)` m. Cross-checked entry by entry
/// against the independent second transcription: every `d`, `a`, limit and the known answer agree (that
/// transcription records no speeds for the UR3e).
pub fn ur3e() -> Robot {
    build(&ur3e_rows(), [PI, PI, PI, TAU, TAU, TAU])
}

fn ur3e_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.15185, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.24355, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.2132, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.13105, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.08535, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0921, 0.0, 0.0),
    ]
}

/// **UR5e**, 6 revolute joints, standard DH, identity tool. The article labels this table "UR5e/UR7e".
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics
/// (UR5e/UR7e kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.1625 | 0 | π/2 |
/// | 2 | 0 | −0.425 | 0 |
/// | 3 | 0 | −0.3922 | 0 |
/// | 4 | 0.1333 | 0 | π/2 |
/// | 5 | 0.0997 | 0 | −π/2 |
/// | 6 | 0.0996 | 0 | 0 |
///
/// Limits and speeds: e-Series collective technical specification 11/2023,
/// <https://www.universal-robots.com/media/1829346/11_2023_collective_data-sheet.pdf> — every joint ±360°
/// (±2π rad) at 180°/s (π rad/s). The PolyScope 5.19 manual adds "Unlimited rotation of tool flange"; the
/// datasheet's ±360° is what is attached to wrist 3. Effort: unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.425 − 0.3922, −(0.1333 + 0.0996), 0.1625 − 0.0997) = (−0.8172, −0.2329, 0.0628)` m; at the home
/// pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.2329, 1.0794)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.3253, −0.1333, −0.3293)` m. Cross-checked entry by entry against
/// the independent second transcription: every `d`, `a`, limit, speed and the known answer agree.
pub fn ur5e() -> Robot {
    build(&ur5e_rows(), [PI; 6])
}

fn ur5e_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.1625, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.425, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.3922, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.1333, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0997, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0996, 0.0, 0.0).with_limits(-TAU, TAU),
    ]
}

/// **UR10e**, 6 revolute joints, standard DH, identity tool. The article labels this table "UR10e/UR12e".
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics
/// (UR10e/UR12e kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.1807 | 0 | π/2 |
/// | 2 | 0 | −0.6127 | 0 |
/// | 3 | 0 | −0.57155 | 0 |
/// | 4 | 0.17415 | 0 | π/2 |
/// | 5 | 0.11985 | 0 | −π/2 |
/// | 6 | 0.11655 | 0 | 0 |
///
/// Limits and speeds: e-Series collective technical specification 11/2023,
/// <https://www.universal-robots.com/media/1829346/11_2023_collective_data-sheet.pdf> — every joint ±360°
/// (±2π rad); base and shoulder 120°/s (2π/3 rad/s); elbow and the three wrists 180°/s (π rad/s). Effort:
/// unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.6127 − 0.57155, −(0.17415 + 0.11655), 0.1807 − 0.11985) = (−1.18425, −0.2907, 0.06085)` m; at the
/// home pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.2907, 1.4848)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.49285, −0.17415, −0.5074)` m. Cross-checked entry by entry
/// against the independent second transcription: every `d`, `a`, limit, speed and the known answer agree.
pub fn ur10e() -> Robot {
    build(&ur10e_rows(), [DEG_120_PER_S, DEG_120_PER_S, PI, PI, PI, PI])
}

fn ur10e_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.1807, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.6127, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.57155, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.17415, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.11985, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.11655, 0.0, 0.0).with_limits(-TAU, TAU),
    ]
}

/// **UR16e**, 6 revolute joints, standard DH, identity tool.
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics (UR16e
/// kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.1807 | 0 | π/2 |
/// | 2 | 0 | −0.4784 | 0 |
/// | 3 | 0 | −0.36 | 0 |
/// | 4 | 0.17415 | 0 | π/2 |
/// | 5 | 0.11985 | 0 | −π/2 |
/// | 6 | 0.11655 | 0 | 0 |
///
/// The UR16e shares every `d` with the UR10e and differs only in `a₂`, `a₃`.
///
/// Limits and speeds: e-Series collective technical specification 11/2023,
/// <https://www.universal-robots.com/media/1829346/11_2023_collective_data-sheet.pdf> — every joint ±360°
/// (±2π rad); base and shoulder 120°/s (2π/3 rad/s); elbow and the three wrists 180°/s (π rad/s). Effort:
/// unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.4784 − 0.36, −(0.17415 + 0.11655), 0.1807 − 0.11985) = (−0.8384, −0.2907, 0.06085)` m; at the home
/// pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.2907, 1.13895)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.35855, −0.17415, −0.29585)` m. Cross-checked entry by entry against
/// the independent second transcription: every `d`, `a` and the known answer agree; that transcription records
/// no speeds and leaves the wrist-3 range unset where the datasheet's ±360° is attached here.
pub fn ur16e() -> Robot {
    build(&ur16e_rows(), [DEG_120_PER_S, DEG_120_PER_S, PI, PI, PI, PI])
}

fn ur16e_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.1807, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.4784, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.36, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.17415, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.11985, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.11655, 0.0, 0.0).with_limits(-TAU, TAU),
    ]
}

/// **UR20**, 6 revolute joints, standard DH, identity tool.
///
/// Source: *Universal Robots support article: DH Parameters for calculations of kinematics and dynamics (UR20
/// kinematics + dynamics tables)*,
/// <https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/>.
/// Confidence: published primary source. Convention: [`DhConvention::Standard`].
///
/// | row | d (m) | a (m) | α (rad) |
/// |---|---|---|---|
/// | 1 | 0.2363 | 0 | π/2 |
/// | 2 | 0 | −0.862 | 0 |
/// | 3 | 0 | −0.7287 | 0 |
/// | 4 | 0.201 | 0 | π/2 |
/// | 5 | 0.1593 | 0 | −π/2 |
/// | 6 | 0.1543 | 0 | 0 |
///
/// Limits and speeds: UR20 technical sheet,
/// <https://www.universal-robots.com/manuals/EN/DataSheets/UR20_techsheet_pdf_online/UR20_techsheet_en.pdf>,
/// and the e-Series collective specification 11/2023 — every joint ±360° (±2π rad); base and shoulder 120°/s
/// (2π/3 rad/s); elbow 150°/s (5π/6 rad/s); wrists 1, 2, 3 210°/s (7π/6 rad/s). Effort: unset, not published.
///
/// Known answer, **computed by hand from the table** and not stated by the source: at `q = 0`,
/// `p = (−0.862 − 0.7287, −(0.201 + 0.1543), 0.2363 − 0.1593) = (−1.5907, −0.3553, 0.077)` m; at the home
/// pose `q = (0, −π/2, 0, −π/2, 0, 0)`, `p = (0, −0.3553, 1.9863)` m; at the bent pose
/// `q = (0, 0, π/2, 0, π/2, 0)`, `p = (a₂ + d₅, −d₄, d₁ + a₃ − d₆) = (−0.7027, −0.201, −0.6467)` m. No second transcription of this table
/// was available for cross-checking.
pub fn ur20() -> Robot {
    build(&ur20_rows(), [DEG_120_PER_S, DEG_120_PER_S, DEG_150_PER_S, DEG_210_PER_S, DEG_210_PER_S, DEG_210_PER_S])
}

fn ur20_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 0.2363, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.862, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.0, -0.7287, 0.0).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.201, 0.0, FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.1593, 0.0, -FRAC_PI_2).with_limits(-TAU, TAU),
        DhRow::revolute(0.0, 0.1543, 0.0, 0.0).with_limits(-TAU, TAU),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    /// The teach-pendant home pose the specification's second hand check uses.
    const HOME: [f64; 6] = [0.0, -FRAC_PI_2, 0.0, -FRAC_PI_2, 0.0, 0.0];
    /// The bent pose that separates `a₂` from `a₃` and `d₄` from `d₆`; see the module documentation.
    const BENT: [f64; 6] = [0.0, 0.0, FRAC_PI_2, 0.0, FRAC_PI_2, 0.0];
    /// A generic, non-singular configuration for the second-order check.
    const GENERIC: [f64; 6] = [0.3, -0.5, 0.12, 0.7, -0.4, 0.9];

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    /// Known answer: `fk(0)`, `fk(HOME)` and `fk(BENT)` against the hand-computed positions, and the `q = 0`
    /// rotation against the hand-computed `[[1,0,0],[0,0,−1],[0,1,0]]`. Non-vacuity first: the three poses
    /// must pairwise differ by more than a decimetre, so the test is looking at a moving quantity and not a
    /// constant.
    fn assert_known_answer(name: &str, r: &Robot, want0: Vector3<f64>, want_home: Vector3<f64>, want_bent: Vector3<f64>) {
        let p0 = r.fk(&[0.0; 6]);
        let ph = r.fk(&HOME);
        let pb = r.fk(&BENT);
        for (a, b) in [(want0, want_home), (want0, want_bent), (want_home, want_bent)] {
            assert!((a - b).norm() > 0.1, "{name}: fixture poses must differ, {a:?} vs {b:?}");
        }
        assert!((p0.translation.vector - ph.translation.vector).norm() > 0.1, "{name}: fk must move between q=0 and HOME");
        assert!((p0.translation.vector - pb.translation.vector).norm() > 0.1, "{name}: fk must move between q=0 and BENT");
        assert!(close(&p0.translation.vector, &want0, 1e-9), "{name} at q=0: {:?} vs {want0:?}", p0.translation.vector);
        assert!(close(&ph.translation.vector, &want_home, 1e-9), "{name} at HOME: {:?} vs {want_home:?}", ph.translation.vector);
        assert!(close(&pb.translation.vector, &want_bent, 1e-9), "{name} at BENT: {:?} vs {want_bent:?}", pb.translation.vector);
        let want_r = Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        let got_r = p0.rotation.to_rotation_matrix().into_inner();
        assert!((got_r - want_r).norm() < 1e-9, "{name} rotation at q=0: {got_r} vs {want_r}");
    }

    /// The analytic Hessian against central differences of the Jacobian, the pattern of
    /// `a_dh_arm_passes_the_hessian_finite_difference_check` in `ferromotion_core::dh`.
    fn assert_hessian_matches_finite_differences(name: &str, r: &Robot) {
        let q = GENERIC;
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
        assert!(worst < 1e-6 * scale, "{name}: Hessian vs FD, worst {worst:e} on a scale of {scale:e}");
    }

    /// The same rows read as Modified DH must build a different arm: the `q = 0` pose moves by more than a
    /// millimetre. Non-vacuity: the Standard build still hits the known answer, so the displacement is the
    /// convention's and not a broken fixture's.
    fn modified_swap_distance(name: &str, rows: &[DhRow; 6], want0: Vector3<f64>) -> f64 {
        let standard = Robot::from_dh(rows, DhConvention::Standard, Iso::identity()).unwrap();
        let modified = Robot::from_dh(rows, DhConvention::Modified, Iso::identity()).unwrap();
        let ps = standard.fk(&[0.0; 6]).translation.vector;
        let pm = modified.fk(&[0.0; 6]).translation.vector;
        assert!(close(&ps, &want0, 1e-9), "{name}: the Standard build must hit the known answer, got {ps:?}");
        (pm - ps).norm()
    }

    /// Datasheet ranges on joints 1–5, the stated wrist-3 range, the stated speeds, and no effort anywhere.
    fn assert_datasheet_limits(name: &str, r: &Robot, wrist3: Option<(f64, f64)>, speeds: [f64; 6]) {
        for (i, j) in r.joints.iter().enumerate() {
            let want = if i == 5 { wrist3 } else { Some((-TAU, TAU)) };
            assert_eq!(j.limits, want, "{name} joint {} limits", i + 1);
            let v = j.max_velocity.expect("every UR joint carries a datasheet speed");
            assert!((v - speeds[i]).abs() < 1e-12, "{name} joint {} speed {v} vs {}", i + 1, speeds[i]);
            assert!(j.effort.is_none(), "{name} joint {}: no UR document states an effort", i + 1);
        }
        assert!(speeds.iter().all(|v| *v > 0.0), "{name}: the speed fixture must be positive");
    }

    /// One module of five tests per model. `$q0` and `$home` are the specification's hand-computed known
    /// answers (closed forms `(a₂+a₃, −(d₄+d₆), d₁−d₅)` and `(0, −(d₄+d₆), d₁−a₂−a₃+d₅)`) and `$bent` the
    /// module's `(a₂+d₅, −d₄, d₁+a₃−d₆)`, typed as numbers here so the test does not recompute them from the
    /// rows it is checking.
    macro_rules! ur_model_tests {
        ($name:ident, $ctor:path, $rows:path, $q0:expr, $home:expr, $bent:expr, $wrist3:expr, $speeds:expr) => {
            mod $name {
                use super::*;

                #[test]
                fn known_answer_at_q_zero_home_and_bent_computed_by_hand_from_the_table() {
                    let r = $ctor();
                    assert_known_answer(stringify!($name), &r, Vector3::from($q0), Vector3::from($home), Vector3::from($bent));
                }

                #[test]
                fn has_six_dof() {
                    assert_eq!($ctor().dof(), 6);
                }

                #[test]
                fn passes_the_hessian_finite_difference_check() {
                    assert_hessian_matches_finite_differences(stringify!($name), &$ctor());
                }

                #[test]
                fn read_as_modified_dh_moves_the_known_answer_by_more_than_a_millimetre() {
                    let dist = modified_swap_distance(stringify!($name), &$rows(), Vector3::from($q0));
                    assert!(dist > 1e-3, "{}: convention swap moved q=0 by only {dist:e} m", stringify!($name));
                }

                #[test]
                fn carries_the_datasheet_limits_and_speeds_and_no_effort() {
                    assert_datasheet_limits(stringify!($name), &$ctor(), $wrist3, $speeds);
                }
            }
        };
    }

    ur_model_tests!(ur3, crate::ur::ur3, crate::ur::ur3_rows, [-0.4569, -0.19425, 0.06655], [0.0, -0.19425, 0.69415], [-0.1583, -0.11235, -0.14325], None, [PI, PI, PI, TAU, TAU, TAU]);
    ur_model_tests!(ur5, crate::ur::ur5, crate::ur::ur5_rows, [-0.81725, -0.19145, -0.005491], [0.0, -0.19145, 1.001059], [-0.33035, -0.10915, -0.385391], Some((-TAU, TAU)), [PI; 6]);
    ur_model_tests!(ur10, crate::ur::ur10, crate::ur::ur10_rows, [-1.1843, -0.256141, 0.0116], [0.0, -0.256141, 1.4273], [-0.4963, -0.163941, -0.5372], Some((-TAU, TAU)), [DEG_120_PER_S, DEG_120_PER_S, PI, PI, PI, PI]);
    ur_model_tests!(ur3e, crate::ur::ur3e, crate::ur::ur3e_rows, [-0.45675, -0.22315, 0.0665], [0.0, -0.22315, 0.69395], [-0.1582, -0.13105, -0.15345], None, [PI, PI, PI, TAU, TAU, TAU]);
    ur_model_tests!(ur5e, crate::ur::ur5e, crate::ur::ur5e_rows, [-0.8172, -0.2329, 0.0628], [0.0, -0.2329, 1.0794], [-0.3253, -0.1333, -0.3293], Some((-TAU, TAU)), [PI; 6]);
    ur_model_tests!(ur10e, crate::ur::ur10e, crate::ur::ur10e_rows, [-1.18425, -0.2907, 0.06085], [0.0, -0.2907, 1.4848], [-0.49285, -0.17415, -0.5074], Some((-TAU, TAU)), [DEG_120_PER_S, DEG_120_PER_S, PI, PI, PI, PI]);
    ur_model_tests!(ur16e, crate::ur::ur16e, crate::ur::ur16e_rows, [-0.8384, -0.2907, 0.06085], [0.0, -0.2907, 1.13895], [-0.35855, -0.17415, -0.29585], Some((-TAU, TAU)), [DEG_120_PER_S, DEG_120_PER_S, PI, PI, PI, PI]);
    ur_model_tests!(ur20, crate::ur::ur20, crate::ur::ur20_rows, [-1.5907, -0.3553, 0.077], [0.0, -0.3553, 1.9863], [-0.7027, -0.201, -0.6467], Some((-TAU, TAU)), [DEG_120_PER_S, DEG_120_PER_S, DEG_150_PER_S, DEG_210_PER_S, DEG_210_PER_S, DEG_210_PER_S]);

    /// The eight tables are eight different arms. A copy-paste of one table into another constructor would
    /// pass every per-model test whose numbers were also copied; this pins that the `q = 0` poses are pairwise
    /// distinct by more than a centimetre (UR10e and UR16e share every `d` and differ only in `a₂ + a₃`).
    #[test]
    fn the_eight_models_are_pairwise_distinct_arms() {
        let arms: [(&str, Robot); 8] = [
            ("ur3", ur3()),
            ("ur5", ur5()),
            ("ur10", ur10()),
            ("ur3e", ur3e()),
            ("ur5e", ur5e()),
            ("ur10e", ur10e()),
            ("ur16e", ur16e()),
            ("ur20", ur20()),
        ];
        let poses: Vec<Vector3<f64>> = arms.iter().map(|(_, r)| r.fk(&[0.0; 6]).translation.vector).collect();
        for i in 0..poses.len() {
            for j in (i + 1)..poses.len() {
                let d = (poses[i] - poses[j]).norm();
                assert!(d > 1e-2, "{} and {} share a q=0 pose to {d:e} m", arms[i].0, arms[j].0);
            }
        }
    }

    /// The datasheet degree-to-radian constants, pinned to the radian values the specification records
    /// (`±360°` is `std::f64::consts::TAU` itself and needs no pin).
    #[test]
    fn datasheet_speed_constants_match_the_specification_radians() {
        assert!((DEG_120_PER_S - 2.0943951023931953).abs() < 1e-15);
        assert!((DEG_150_PER_S - 2.6179938779914944).abs() < 1e-15);
        assert!((DEG_210_PER_S - 3.6651914291880923).abs() < 1e-15);
    }
}
