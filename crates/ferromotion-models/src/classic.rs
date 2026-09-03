//! **The classic arms of the robotics literature**, built from the tables their sources print: the
//! Unimation PUMA 560 in both Denavit–Hartenberg conventions, the Stanford arm, and the two- and
//! three-link planar teaching arms.
//!
//! Each constructor cites its primary source (title and URL), the convention that source uses, every
//! unit conversion applied, and the known-answer pose its tests hold it to. Where a source gives no joint
//! limit, effort or velocity, the corresponding field is left `None` and the doc says so.
//!
//! The two PUMA constructors describe the same physical arm from two different tables. Their forward
//! kinematics are asserted to agree at several joint configurations under the joint relabelling
//! measured in the tests (`q_modified = q_standard + (π, π, 0, 0, 0, 0)`, tool frames identical), which
//! is the cross-convention oracle for both tables. A third, independent sourcing of the PUMA 560 table
//! (Armstrong, Khatib & Burdick 1986, Table A1) is compared in the tests and the measured differences are
//! stated in [`puma560_modified`]'s documentation.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use nalgebra::{Translation3, UnitQuaternion};
use std::f64::consts::FRAC_PI_2;

/// Metres per inch, exact by definition (international inch, 1959).
const METRES_PER_INCH: f64 = 0.0254;

/// A pure translation of `d` along the last DH frame's `z`, the shape every flange offset here takes.
fn tool_z(d: f64) -> Iso {
    Iso::from_parts(Translation3::new(0.0, 0.0, d), UnitQuaternion::identity())
}

/// Build from a constant table. Every table in this module is finite and non-empty, so
/// [`Robot::from_dh`] cannot return `None` for it; the `expect` states that invariant.
fn build(rows: &[DhRow], convention: DhConvention, tool: Iso) -> Robot {
    Robot::from_dh(rows, convention, tool).expect("a constant, finite, non-empty DH table")
}

/// Attach per-joint effort (N·m) and, when the source states one, maximum velocity (rad/s).
fn with_actuation(mut robot: Robot, effort: &[f64], max_velocity: Option<&[f64]>) -> Robot {
    for (j, &e) in robot.joints.iter_mut().zip(effort) {
        *j = j.clone().with_effort(e);
    }
    if let Some(v) = max_velocity {
        for (j, &v) in robot.joints.iter_mut().zip(v) {
            *j = j.clone().with_max_velocity(v);
        }
    }
    robot
}

// PUMA 560 consensus link lengths, metres (Corke 1996 Table 2.1, mm ÷ 1000).
/// Upper-arm length `a₂`: 431.8 mm.
const PUMA_A2: f64 = 0.4318;
/// Elbow offset `a₃`: 20.3 mm.
const PUMA_A3: f64 = 0.0203;
/// Shoulder offset `d₃`: 125.4 mm.
const PUMA_D3: f64 = 0.1254;
/// Forearm length `d₄`: 431.8 mm.
const PUMA_D4: f64 = 0.4318;
/// Wrist centre to mounting-flange surface `d₆`: 56.25 mm.
const PUMA_D6: f64 = 0.05625;

/// The rows of [`puma560`], shared with its tests so the convention-swap test reads the very table the
/// constructor builds.
fn puma560_rows() -> [DhRow; 6] {
    let deg = f64::to_radians;
    [
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2).with_limits(deg(-180.0), deg(180.0)),
        DhRow::revolute(0.0, 0.0, PUMA_A2, 0.0).with_limits(deg(-170.0), deg(165.0)),
        DhRow::revolute(0.0, PUMA_D3, PUMA_A3, -FRAC_PI_2).with_limits(deg(-160.0), deg(150.0)),
        DhRow::revolute(0.0, PUMA_D4, 0.0, FRAC_PI_2).with_limits(deg(-180.0), deg(180.0)),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(deg(-10.0), deg(100.0)),
        DhRow::revolute(0.0, PUMA_D6, 0.0, 0.0).with_limits(deg(-180.0), deg(180.0)),
    ]
}

/// **PUMA 560 maximum joint torques, N·m**, per joint: B. Armstrong, O. Khatib, J. Burdick, "The explicit
/// dynamic model and inertial parameters of the PUMA 560 arm", Proc. IEEE ICRA 1986, Table 7 "Maximum
/// Torque (N-m)", as quoted in the specification for both PUMA entries.
const PUMA_EFFORT_NM: [f64; 6] = [97.6, 186.4, 89.4, 24.2, 20.1, 21.3];

/// **PUMA 560 joint velocity limits, rad/s at the joint**, DERIVED (not a manufacturer figure): Corke 1996
/// Table 2.18 measured motor-referenced back-EMF/voltage-saturation limits (120, 163, 129, 406, 366, 440
/// rad/s) divided by the magnitudes of the Table 2.9 gear ratios (62.6111, 107.815, 53.7063, 76.03636,
/// 71.923, 76.686; Corke prints `G₁` and `G₃` with a negative sign, which encodes joint direction and
/// does not belong in a rate limit). Cross-check: Corke's own Table 2.21 "load referenced θ̇" column
/// prints 1.92, 1.51, 2.40, 5.34, 5.09, 5.74 rad/s (Tables 2.1, 2.9, 2.18 and 2.21 were each re-read
/// from the author-hosted PDF text while verifying this module), which are these values rounded to two
/// decimals.
const PUMA_VELOCITY_RAD_S: [f64; 6] = [1.917, 1.512, 2.402, 5.340, 5.089, 5.738];

/// **Unimation PUMA 560, standard Denavit–Hartenberg** (Paul & Zhang frame assignments, consensus lengths).
///
/// **Source** — "P. I. Corke, Visual Control of Robots: High-Performance Visual Servoing, Research
/// Studies Press / John Wiley, 1996 (Mechatronics series), Table 2.1 'Kinematic parameters and joint
/// limits for the Puma 560' (p. 12), Tables 2.17-2.18 (pp. 47-49); author-hosted PDF of the out-of-print
/// book", <https://petercorke.com/bluebook/book.pdf>. Confidence: **published primary source**. Table 2.1
/// was re-read from that PDF while writing this constructor and matches the specification verbatim.
///
/// **Convention** — [`DhConvention::Standard`]: `T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`.
///
/// **Table, verbatim from Corke Table 2.1 (degrees, mm)** and the conversion applied here (deg × π/180,
/// mm ÷ 1000):
///
/// | joint | α    | a     | d     | θmin | θmax |
/// |---|---|---|---|---|---|
/// | 1 |  90 |     0 |     0 | −180 | 180 |
/// | 2 |   0 | 431.8 |     0 | −170 | 165 |
/// | 3 | −90 |  20.3 | 125.4 | −160 | 150 |
/// | 4 |  90 |     0 | 431.8 | −180 | 180 |
/// | 5 | −90 |     0 |     0 |  −10 | 100 |
/// | 6 |   0 |     0 | 56.25 | −180 | 180 |
///
/// `d₆ = 56.25 mm` is the wrist centre to the mounting-flange surface (the book attributes it to Lee),
/// so frame 6 of this table is **on the flange**; no further tool offset is applied. Zero pose: upper
/// arm horizontal along `+x`, forearm vertical up; the READY pose `q = (0, 90°, −90°, 0, 0, 0)` is fully
/// extended and upright.
///
/// **Effort** — `PUMA_EFFORT_NM` (Armstrong, Khatib & Burdick 1986 Table 7). **Velocity** —
/// `PUMA_VELOCITY_RAD_S`, derived from Corke's measured motor saturation velocities and gear ratios;
/// no manufacturer velocity specification was located by the specification's review.
///
/// **Known answer (computed by hand from the table, not printed by the source)** — at `q = 0` the frame-6
/// origin is `(a₂ + a₃, −d₃, d₄ + d₆) = (0.4521, −0.1254, 0.48805) m` with identity rotation; at READY
/// `(0.0203, −0.1254, 0.91985) m`, identity rotation. Verified to `1e-9 m` in this module's tests, and
/// against [`puma560_modified`] (a different table of the same arm) at several configurations.
pub fn puma560() -> Robot {
    with_actuation(build(&puma560_rows(), DhConvention::Standard, Iso::identity()), &PUMA_EFFORT_NM, Some(&PUMA_VELOCITY_RAD_S))
}

/// The rows of [`puma560_modified`], shared with its tests.
fn puma560_modified_rows() -> [DhRow; 6] {
    let deg = f64::to_radians;
    [
        DhRow::revolute(0.0, 0.0, 0.0, 0.0).with_limits(deg(-170.0), deg(170.0)),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(deg(-225.0), deg(45.0)),
        DhRow::revolute(0.0, PUMA_D3, PUMA_A2, 0.0).with_limits(deg(-250.0), deg(75.0)),
        DhRow::revolute(0.0, PUMA_D4, PUMA_A3, -FRAC_PI_2).with_limits(deg(-135.0), deg(135.0)),
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2).with_limits(deg(-100.0), deg(100.0)),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2).with_limits(deg(-180.0), deg(180.0)),
    ]
}

/// **Unimation PUMA 560, modified (Craig) Denavit–Hartenberg**, consensus lengths, flange as tool offset.
///
/// **Source** — "J. J. Craig, Introduction to Robotics: Mechanics and Control, 3rd ed., Pearson Prentice
/// Hall, 2005, Section 3.7 'The PUMA 560', Figs. 3.18-3.21 (pp. 78-80) and Exercise 4.5 (p. 129); numeric
/// lengths from Corke 1996 Table 2.1 (see previous entry)",
/// J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 3rd ed., Pearson Prentice Hall, 2005 (ISBN 0-13-123629-6). Confidence: **derived
/// from published geometry** — Craig's Fig. 3.21 gives `a₂, a₃, d₃, d₄` as symbols; the numbers are the
/// Corke 1996 consensus values ([`puma560`]).
///
/// **Convention** — [`DhConvention::Modified`]: `T_i = Rx(α_{i−1})·Tx(a_{i−1})·Rz(θ_i)·Tz(d_i)`; row `i`
/// carries `α_{i−1}, a_{i−1}, d_i`.
///
/// **Table, verbatim from Craig Fig. 3.21** `(α_{i−1}, a_{i−1}, d_i, θ_i)`: `(0, 0, 0, θ₁)`,
/// `(−90°, 0, 0, θ₂)`, `(0, a₂, d₃, θ₃)`, `(−90°, a₃, d₄, θ₄)`, `(90°, 0, 0, θ₅)`, `(−90°, 0, 0, θ₆)`,
/// with `a₂ = 0.4318`, `a₃ = 0.0203`, `d₃ = 0.1254`, `d₄ = 0.4318 m`. Craig's frame {6} is at the wrist
/// centre; the flange is `d₆ = 56.25 mm` further along `z₆` (Corke Table 2.1) and is applied here as the
/// `tool` argument, so [`Robot::fk`] reports the **flange**, as [`puma560`] does. Zero pose: upper arm
/// along `+x`, forearm hanging down along `−z` (Craig Fig. 3.18).
///
/// **Limits** — Craig Exercise 4.5 (degrees, converted × π/180): θ₁ [−170, 170], θ₂ [−225, 45],
/// θ₃ [−250, 75], θ₄ [−135, 135], θ₅ [−100, 100], θ₆ [−180, 180]. These are in **Craig's zero
/// convention** and are not interchangeable with Corke's. **Effort** — `PUMA_EFFORT_NM`. **Velocity**
/// — not stated in Craig; left `None`.
///
/// **Known answer (computed by hand from the table, not printed by the source)** — at `q = 0` the wrist
/// centre is `(a₂ + a₃, d₃, −d₄) = (0.4521, 0.1254, −0.4318) m`, identity rotation of frame {6} relative
/// to Craig's frame {0} conventions as built here; the flange (what this constructor returns) is
/// `(0.4521, 0.1254, −0.48805) m`. The specification confirms the wrist-centre value against Craig's
/// closed-form position equations (3.14) at `q = 0`.
///
/// **Cross-convention oracle (measured)** — for every `q`, `puma560().fk(q) == puma560_modified().fk(q +
/// (π, π, 0, 0, 0, 0))` to `1e-9 m` in position and `1e-9` in rotation: the two tables are the same axes
/// and lengths, with the base yaw and the shoulder zero each differing by half a turn. The relabelling
/// was found by an exhaustive search over per-joint sign flips and quarter-turn offsets.
///
/// **Third sourcing, compared** — Armstrong, Khatib & Burdick 1986 Table A1 (modified DH, wrist-centre
/// frame) gives `α_{i−1} = (0, −90, 0, 90, −90, 90)°`, `a_{i−1} = (0, 0, 0.4318, −0.0203, 0, 0) m`,
/// `d_i = (0, 0.2435, −0.0934, 0.4331, 0, 0) m`. Same `a₂`; same `|a₃|` with the opposite sign and the
/// opposite `α₃` sign (their zero pose has the forearm **up**); shoulder offset `d₂ + d₃ = 0.1501 m`
/// against `0.1254 m` here (**24.7 mm apart**); forearm `d₄ = 0.4331 m` against `0.4318 m` (**1.3 mm
/// apart**). Craig's own Exercise 4.5 quotes, in inches, `a₂ = 17.0, a₃ = 0.8, d₃ = 4.9, d₄ = 17.0`,
/// i.e. `0.4318, 0.02032, 0.12446, 0.4318 m` — 0.02 mm and 0.94 mm from the consensus `a₃` and `d₃`.
/// The tests pin the AKB-versus-consensus wrist-centre difference at `(0, 0.0247, 0.0013) m`.
pub fn puma560_modified() -> Robot {
    with_actuation(build(&puma560_modified_rows(), DhConvention::Modified, tool_z(PUMA_D6)), &PUMA_EFFORT_NM, None)
}

/// The rows of [`stanford`], shared with its tests.
fn stanford_rows() -> [DhRow; 6] {
    [
        DhRow::revolute(0.0, 16.24 * METRES_PER_INCH, 0.0, -FRAC_PI_2),
        DhRow::revolute(0.0, 6.05 * METRES_PER_INCH, 0.0, FRAC_PI_2),
        DhRow::prismatic(-FRAC_PI_2, 0.0, 0.0, 0.0),
        DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2),
        DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2),
        DhRow::revolute(0.0, 10.35 * METRES_PER_INCH, 0.0, 0.0),
    ]
}

/// **The Stanford arm (Scheinman), standard Denavit–Hartenberg**, from Paul's 1972 report.
///
/// **Source** — "R. P. Paul, 'Modelling, Trajectory Calculation and Servoing of a Computer Controlled
/// Arm', Stanford AI Lab Memo AIM-177 / STAN-CS-72-311, November 1972 (NTIS AD-785 071), Section 2.1
/// Kinematics, Figure 2.3 'Arm Coordinate Systems' (p. 8), Table 2.1 and the A/T matrices on pp. 9-11",
/// <https://archive.org/download/DTIC_AD0785071/DTIC_AD0785071.pdf>. Confidence: **published primary
/// source**.
///
/// **Convention** — [`DhConvention::Standard`]: Paul's A-matrix (Eq. 2.2) is
/// `Rot(z,θ)·Trans(0,0,s)·Trans(a,0,0)·Rot(x,α)`.
///
/// **Table, verbatim from Figure 2.3 (degrees, inches; all `a = 0`)** — joint 1: `α = −90, s = 16.24,
/// θ₁`; joint 2: `α = 90, s = 6.05, θ₂`; joint 3: `α = 0, s = s₃ (variable), θ = −90 fixed`; joint 4:
/// `α = −90, s = 0, θ₄`; joint 5: `α = 90, s = 0, θ₅`; joint 6: `α = 0, s = 10.35, θ₆`. Conversion applied:
/// inches × 0.0254 → `16.24 in = 0.412496 m`, `6.05 in = 0.15367 m`, `10.35 in = 0.26289 m`; degrees ×
/// π/180. Joint 3 is prismatic with the fixed `θ₃ = −90°` as its row's theta offset and `d₃ = 0` as the
/// zero of the variable, so `q₃` is the extension `s₃` in metres. Frame 6's origin is "a point centrally
/// located between the finger tips" (p. 11); no tool offset.
///
/// **Limits, effort, velocity** — the report says every joint but joint 6 has a partial range of motion,
/// but the specification's review located no numeric limit table, and no effort or velocity figures; all
/// are left `None`.
///
/// **Known answer (source-stated)** — Paul's worked example (Table 2.1, p. 9) at `θ = (−95.7°, −112.4°,
/// s₃ = 22.16 in, −38.2°, 80.4°, 68.9°)` prints `T6` with position `(−.00, 19.78, 1.30)` inches and
/// rotation `[[−.63, −.00, −.78], [.00, 1.00, −.00], [.78, −.00, −.63]]` (two decimals). Recomputing here
/// from the source's inch and degree values gives `(−0.000139115, 0.502340752, 0.033096690) m` =
/// `(−0.0055, 19.7772, 1.3030) in`; the tests hold the source's printed position and rotation to one
/// printed unit (0.01 in, 0.01) — measured worst component 0.0055 in on `x` — and the specification's
/// six-decimal hand value `(−0.000139, 0.502341, 0.033097) m` to `1e-6 m` (measured residual
/// `4.1e-7 m`). Note: the specification's `q_rad_or_m` array for this pose encodes `−95.6958°` and
/// `−112.3983°` rather than the source's `−95.7°` and `−112.4°` (up to `7.4e-5 rad` off), which moves the
/// tip by `4.1e-5 m`; this constructor's tests use the source's degree values. Secondary hand check at
/// `q = 0`: `(0, 0.15367, 0.675386) m`, rotation `[[0, 1, 0], [−1, 0, 0], [0, 0, 1]]`.
pub fn stanford() -> Robot {
    build(&stanford_rows(), DhConvention::Standard, Iso::identity())
}

/// The rows of [`two_link_planar`], shared with its tests.
fn two_link_planar_rows() -> [DhRow; 2] {
    [DhRow::revolute(0.0, 0.0, 1.0, 0.0), DhRow::revolute(0.0, 0.0, 1.0, 0.0)]
}

/// **Two-link planar arm (RR)**, standard Denavit–Hartenberg, `l₁ = l₂ = 1 m`.
///
/// **Source** — "J. J. Craig, Introduction to Robotics: Mechanics and Control, 3rd ed., Pearson Prentice
/// Hall, 2005, Example 5.3, Figs. 5.8-5.9 and link transforms (5.49) (pp. 146-147)",
/// J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 3rd ed., Pearson Prentice Hall, 2005 (ISBN 0-13-123629-6). Confidence: **derived
/// from published geometry**.
///
/// **The metre values are a library choice, not sourced.** Craig gives the link lengths only as the symbols
/// `l₁, l₂`; every primary source the specification's review found does the same. `l₁ = l₂ = 1.0 m` is a
/// choice made for this library and is the reason for the "derived" confidence.
///
/// **Convention** — Craig writes the chain in modified DH (`⁰T₁ = Rz(θ₁)`, `¹T₂ = Tx(l₁)·Rz(θ₂)`,
/// `²T₃ = Tx(l₂)`, frame {3} at the tip). As [`DhConvention::Standard`] the same chain is `a₁ = l₁`,
/// `a₂ = l₂`, all `d = 0`, `α = 0`, frame 2 at the tip, which is what this constructor builds:
/// `tip = (l₁c₁ + l₂c₁₂, l₁s₁ + l₂s₁₂, 0)`.
///
/// **Limits, effort, velocity** — none in the source; all `None`.
///
/// **Known answer (computed by hand from the table, not printed by the source)** — `q = (30°, 60°)` →
/// `(cos 30° + cos 90°, sin 30° + sin 90°, 0) = (0.866025, 1.5, 0) m`; also `q = 0 → (2, 0, 0)` and
/// `q = (90°, 0) → (0, 2, 0)`.
pub fn two_link_planar() -> Robot {
    build(&two_link_planar_rows(), DhConvention::Standard, Iso::identity())
}

/// The rows of [`three_link_planar`], shared with its tests.
fn three_link_planar_rows() -> [DhRow; 3] {
    [DhRow::revolute(0.0, 0.0, 1.0, 0.0), DhRow::revolute(0.0, 0.0, 1.0, 0.0), DhRow::revolute(0.0, 0.0, 0.0, 0.0)]
}

/// **Three-link planar arm (RRR)**, standard Denavit–Hartenberg, `L₁ = L₂ = 1 m`, frame 3 on the joint-3
/// axis.
///
/// **Source** — "J. J. Craig, Introduction to Robotics: Mechanics and Control, 3rd ed., Pearson Prentice
/// Hall, 2005, Example 3.3, Figs. 3.6-3.8 'Link parameters of the three-link planar manipulator' (pp.
/// 69-71)", J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 3rd ed., Pearson Prentice Hall, 2005 (ISBN 0-13-123629-6). Confidence:
/// **derived from published geometry**.
///
/// **The metre values are a library choice, not sourced.** Craig's Fig. 3.8 gives the lengths as the
/// symbols `L₁, L₂` (his Exercise 4.1 uses unitless `15.0, 10.0, 3.0` for a workspace sketch); the unit
/// lengths here are a choice made for this library and the reason for the "derived" confidence.
///
/// **Convention** — Craig Fig. 3.8 (modified DH, `α_{i−1}, a_{i−1}, d_i, θ_i`): `(0, 0, 0, θ₁)`,
/// `(0, L₁, 0, θ₂)`, `(0, L₂, 0, θ₃)`; his frame {3} lies on the joint-3 axis, so the last link `L₃` is a
/// separate end-effector offset he does not put in the table. As [`DhConvention::Standard`] the same chain
/// is `a₁ = L₁`, `a₂ = L₂`, `a₃ = 0`, all `d = 0`, `α = 0`, which is what this constructor builds; to reach
/// a fingertip, compose the returned robot's pose with `(L₃, 0, 0)` along `x₃` yourself. Consequently
/// `q₃` rotates the end frame without moving its origin.
///
/// **Limits, effort, velocity** — none in the source; all `None`.
///
/// **Known answer (computed by hand from the table, not printed by the source)** — `q = (30°, 30°, 30°)`
/// → `(cos 30° + cos 60°, sin 30° + sin 60°, 0) = (1.366025, 1.366025, 0) m`; `q = (30°, 30°, 0)` gives
/// the same point; `q = 0 → (2, 0, 0)`; the end-frame yaw is `θ₁ + θ₂ + θ₃`.
pub fn three_link_planar() -> Robot {
    build(&three_link_planar_rows(), DhConvention::Standard, Iso::identity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    use std::f64::consts::PI;

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    fn position(r: &Robot, q: &[f64]) -> Vector3<f64> {
        r.fk(q).translation.vector
    }

    /// The analytic Hessian against central differences of the Jacobian, the check every `Robot` in the
    /// workspace is held to (pattern of `a_dh_arm_passes_the_hessian_finite_difference_check` in
    /// `ferromotion_core::dh`).
    fn hessian_matches_finite_differences(r: &Robot, q: &[f64]) {
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

    /// Build the constructor's own rows under both conventions and return how far the pose at `q`
    /// moves between them. Also asserts the `convention` build reproduces the constructor's pose, so the
    /// rows under test are the rows the constructor ships.
    fn swap_distance(rows: &[DhRow], convention: DhConvention, tool: Iso, q: &[f64]) -> f64 {
        let other = match convention {
            DhConvention::Standard => DhConvention::Modified,
            DhConvention::Modified => DhConvention::Standard,
        };
        let a = build(rows, convention, tool);
        let b = build(rows, other, tool);
        (position(&a, q) - position(&b, q)).norm()
    }

    fn swap_distance_of(model: &Robot, rows: &[DhRow], convention: DhConvention, tool: Iso, q: &[f64]) -> f64 {
        let rebuilt = build(rows, convention, tool);
        assert!(close(&position(model, q), &position(&rebuilt, q), 1e-12), "the rows under test must be the constructor's rows");
        swap_distance(rows, convention, tool, q)
    }

    // ---------------------------------------------------------------- PUMA 560, standard DH


    /// **Hand computation from Corke Table 2.1** (not printed by the source): at `q = 0` the flange is at
    /// `(a₂ + a₃, −d₃, d₄ + d₆)` with identity rotation; at READY `(0, π/2, −π/2, 0, 0, 0)` the arm is
    /// fully extended and upright, `(a₃, −d₃, a₂ + d₄ + d₆)`, which is how the book describes that pose.
    #[test]
    fn puma560_known_answer_computed_by_hand_from_corke_table_2_1() {
        let r = puma560();
        assert_eq!(r.dof(), 6);
        let home = position(&r, &[0.0; 6]);
        let ready = position(&r, &[0.0, FRAC_PI_2, -FRAC_PI_2, 0.0, 0.0, 0.0]);
        // non-vacuous: the two poses are 0.43 m apart, so a constructor that ignored q would fail below
        assert!((home - ready).norm() > 0.4, "home and READY must differ: {home:?} vs {ready:?}");
        assert!(close(&home, &Vector3::new(0.4521, -0.1254, 0.48805), 1e-9), "home: {home:?}");
        assert!(close(&ready, &Vector3::new(0.0203, -0.1254, 0.91985), 1e-9), "READY: {ready:?}");
        let rot_home = r.fk(&[0.0; 6]).rotation.to_rotation_matrix();
        let rot_ready = r.fk(&[0.0, FRAC_PI_2, -FRAC_PI_2, 0.0, 0.0, 0.0]).rotation.to_rotation_matrix();
        assert!((rot_home.matrix() - Matrix3::identity()).norm() < 1e-12, "home rotation: {rot_home:?}");
        assert!((rot_ready.matrix() - Matrix3::identity()).norm() < 1e-12, "READY rotation: {rot_ready:?}");
    }

    #[test]
    fn puma560_carries_corke_limits_and_the_stated_actuation() {
        let r = puma560();
        let deg = f64::to_radians;
        let want_limits = [(-180.0, 180.0), (-170.0, 165.0), (-160.0, 150.0), (-180.0, 180.0), (-10.0, 100.0), (-180.0, 180.0)];
        for (i, (j, (lo, hi))) in r.joints.iter().zip(want_limits).enumerate() {
            assert_eq!(j.limits, Some((deg(lo), deg(hi))), "joint {} limits", i + 1);
            assert_eq!(j.effort, Some(PUMA_EFFORT_NM[i]), "joint {} effort", i + 1);
            assert_eq!(j.max_velocity, Some(PUMA_VELOCITY_RAD_S[i]), "joint {} velocity", i + 1);
        }
        // the derived velocities are Corke's own Table 2.21 load-referenced column to its two decimals
        for (v, printed) in PUMA_VELOCITY_RAD_S.iter().zip([1.92, 1.51, 2.40, 5.34, 5.09, 5.74]) {
            assert!((v - printed).abs() <= 0.005 + 1e-12, "{v} vs Corke Table 2.21 {printed}");
        }
    }

    #[test]
    fn puma560_passes_the_hessian_finite_difference_check() {
        hessian_matches_finite_differences(&puma560(), &[0.3, -0.5, 0.7, 0.2, -0.9, 0.4]);
    }

    /// Reading Corke's standard table as modified DH builds a different arm: measured 0.433 m at `q = 0`.
    #[test]
    fn puma560_convention_argument_is_load_bearing() {
        let d = swap_distance_of(&puma560(), &puma560_rows(), DhConvention::Standard, Iso::identity(), &[0.0; 6]);
        assert!(d > 1e-3, "convention swap moved the known-answer pose by only {d:e} m");
        assert!((d - 0.4333150).abs() < 1e-6, "measured swap distance changed: {d}");
    }

    // ---------------------------------------------------------------- PUMA 560, modified DH


    /// **Hand computation from Craig Fig. 3.21 with the consensus lengths** (not printed by the source):
    /// wrist centre `(a₂ + a₃, d₃, −d₄)` at `q = 0`, the flange `d₆` further down `−z`.
    #[test]
    fn puma560_modified_known_answer_computed_by_hand_from_craig_fig_3_21() {
        let r = puma560_modified();
        assert_eq!(r.dof(), 6);
        let wrist = build(&puma560_modified_rows(), DhConvention::Modified, Iso::identity());
        let pw = position(&wrist, &[0.0; 6]);
        let pf = position(&r, &[0.0; 6]);
        // non-vacuous: bending the shoulder by a right angle moves the flange by more than 0.4 m
        let bent = position(&r, &[0.0, -FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]);
        assert!((pf - bent).norm() > 0.4, "home and bent must differ: {pf:?} vs {bent:?}");
        assert!(close(&pw, &Vector3::new(0.4521, 0.1254, -0.4318), 1e-9), "wrist centre: {pw:?}");
        assert!(close(&pf, &Vector3::new(0.4521, 0.1254, -0.48805), 1e-9), "flange: {pf:?}");
        // Craig's closed-form (3.14) at a second pose, q₂ = −π/2 (upper arm raised): c₂ = 0, s₂ = −1,
        // c₂₃ = 0, s₂₃ = −1 → px = −d₃·s₁ + c₁·(a₃·0 − d₄·(−1)) = d₄, py = d₃, pz = −a₃·(−1) − a₂·(−1) − 0
        let raised = position(&wrist, &[0.0, -FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]);
        assert!(close(&raised, &Vector3::new(PUMA_D4, PUMA_D3, PUMA_A3 + PUMA_A2), 1e-9), "Craig (3.14) at q₂ = −π/2: {raised:?}");
    }

    #[test]
    fn puma560_modified_carries_craig_limits_efforts_and_no_velocity() {
        let r = puma560_modified();
        let deg = f64::to_radians;
        let want = [(-170.0, 170.0), (-225.0, 45.0), (-250.0, 75.0), (-135.0, 135.0), (-100.0, 100.0), (-180.0, 180.0)];
        for (i, (j, (lo, hi))) in r.joints.iter().zip(want).enumerate() {
            assert_eq!(j.limits, Some((deg(lo), deg(hi))), "joint {} limits", i + 1);
            assert_eq!(j.effort, Some(PUMA_EFFORT_NM[i]), "joint {} effort", i + 1);
            assert_eq!(j.max_velocity, None, "Craig states no velocity for joint {}", i + 1);
        }
    }

    #[test]
    fn puma560_modified_passes_the_hessian_finite_difference_check() {
        hessian_matches_finite_differences(&puma560_modified(), &[0.3, -0.5, 0.7, 0.2, -0.9, 0.4]);
    }

    /// Reading Craig's modified table as standard DH builds a different arm: measured 0.611 m at `q = 0`.
    #[test]
    fn puma560_modified_convention_argument_is_load_bearing() {
        let d = swap_distance_of(&puma560_modified(), &puma560_modified_rows(), DhConvention::Modified, tool_z(PUMA_D6), &[0.0; 6]);
        assert!(d > 1e-3, "convention swap moved the known-answer pose by only {d:e} m");
        assert!((d - 0.6106574).abs() < 1e-6, "measured swap distance changed: {d}");
    }

    /// **The cross-convention oracle.** Two tables of one arm must agree everywhere, not just at home.
    /// The relabelling `q_modified = q_standard + (π, π, 0, 0, 0, 0)` with identical tool frames was found
    /// by an exhaustive search over per-joint sign flips and quarter-turn offsets; with it the full
    /// poses coincide to floating-point precision at home, READY and four generic configurations.
    #[test]
    fn both_puma560_tables_describe_the_same_arm() {
        let s = puma560();
        let m = puma560_modified();
        let configs: [[f64; 6]; 6] = [
            [0.0; 6],
            [0.0, FRAC_PI_2, -FRAC_PI_2, 0.0, 0.0, 0.0],
            [0.3, -0.5, 0.7, 0.2, -0.9, 0.4],
            [-1.2, 0.8, -2.1, 1.5, 0.6, -2.8],
            [2.5, -1.9, 0.1, -0.7, 1.3, 0.9],
            [0.05, 1.4, -0.3, 2.9, -1.1, -0.2],
        ];
        let mut spread = 0.0f64;
        for q in configs {
            let mut qm = q;
            qm[0] += PI;
            qm[1] += PI;
            let ts = s.fk(&q);
            let tm = m.fk(&qm);
            spread = spread.max((ts.translation.vector - position(&s, &[0.0; 6])).norm());
            let dp = (ts.translation.vector - tm.translation.vector).norm();
            let dr = (ts.rotation.to_rotation_matrix().matrix() - tm.rotation.to_rotation_matrix().matrix()).norm();
            assert!(dp < 1e-9, "position disagrees at {q:?}: {dp:e} m");
            assert!(dr < 1e-9, "rotation disagrees at {q:?}: {dr:e}");
        }
        // non-vacuous: the configurations sweep the flange over more than half a metre
        assert!(spread > 0.5, "configurations should span the workspace, spread {spread}");
        // and the relabelling is load-bearing: without it the two tables disagree by decimetres
        let raw = (position(&s, &configs[2]) - position(&m, &configs[2])).norm();
        assert!(raw > 0.1, "without the (π, π) relabelling the tables should disagree, got {raw}");
    }

    /// **Third sourcing, compared.** Armstrong, Khatib & Burdick 1986 Table A1 is an independent
    /// modified-DH table of the same arm at the wrist centre. Its zero pose has the forearm up, which is
    /// the Craig table at `q₃ = π`; the residual is then purely the measured length differences:
    /// shoulder offset `0.1501 − 0.1254 = 0.0247 m`, forearm `0.4331 − 0.4318 = 0.0013 m`.
    #[test]
    fn armstrong_khatib_burdick_table_a1_differs_from_the_consensus_by_the_stated_lengths() {
        let akb = build(
            &[
                DhRow::revolute(0.0, 0.0, 0.0, 0.0),
                DhRow::revolute(0.0, 0.2435, 0.0, -FRAC_PI_2),
                DhRow::revolute(0.0, -0.0934, 0.4318, 0.0),
                DhRow::revolute(0.0, 0.4331, -0.0203, FRAC_PI_2),
                DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2),
                DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2),
            ],
            DhConvention::Modified,
            Iso::identity(),
        );
        let p_akb = position(&akb, &[0.0; 6]);
        assert!(close(&p_akb, &Vector3::new(0.4115, 0.1501, 0.4331), 1e-9), "AKB wrist centre by hand: {p_akb:?}");
        let craig_wrist = build(&puma560_modified_rows(), DhConvention::Modified, Iso::identity());
        let p_craig = position(&craig_wrist, &[0.0, 0.0, PI, 0.0, 0.0, 0.0]);
        let diff = p_akb - p_craig;
        assert!(diff.norm() > 0.02, "the sourcings differ by centimetres, not nothing: {diff:?}");
        assert!(close(&diff, &Vector3::new(0.0, 0.0247, 0.0013), 1e-9), "AKB − consensus: {diff:?}");
    }

    // ---------------------------------------------------------------- Stanford arm


    /// Paul's worked configuration, from the source's degree and inch values.
    fn paul_example_q() -> [f64; 6] {
        let deg = f64::to_radians;
        [deg(-95.7), deg(-112.4), 22.16 * METRES_PER_INCH, deg(-38.2), deg(80.4), deg(68.9)]
    }

    /// **Source-stated known answer**: Paul AIM-177 pp. 9-11 print `T6` for this configuration to two
    /// decimal inches. Held to one printed unit (0.01 in / 0.01), and the specification's six-decimal
    /// hand recomputation to `1e-6 m`.
    #[test]
    fn stanford_known_answer_is_pauls_printed_t6() {
        let r = stanford();
        assert_eq!(r.dof(), 6);
        let q = paul_example_q();
        let t = r.fk(&q);
        let p = t.translation.vector;
        // non-vacuous: this pose is far from home
        let home = position(&r, &[0.0; 6]);
        assert!((p - home).norm() > 0.5, "example pose must differ from home: {p:?} vs {home:?}");
        // the source, in inches, to one printed unit per component
        let inches = p / METRES_PER_INCH;
        let printed = Vector3::new(-0.00, 19.78, 1.30);
        let worst = (inches - printed).abs().max();
        assert!(worst <= 0.01, "position vs Paul's printed T6 (in): {inches:?} vs {printed:?}, worst {worst}");
        let rot = t.rotation.to_rotation_matrix();
        let printed_rot = Matrix3::new(-0.63, -0.00, -0.78, 0.00, 1.00, -0.00, 0.78, -0.00, -0.63);
        let worst_r = (rot.matrix() - printed_rot).abs().max();
        assert!(worst_r <= 0.01, "rotation vs Paul's printed T6: {rot:?}, worst {worst_r}");
        // the specification's hand recomputation, to its six printed decimals
        assert!(close(&p, &Vector3::new(-0.000139, 0.502341, 0.033097), 1e-6), "spec hand value: {p:?}");
        // and this module's own full-precision recomputation, so a drift below 1e-6 is still caught
        assert!(close(&p, &Vector3::new(-0.000139115187, 0.502340752, 0.0330966901), 1e-9), "full precision: {p:?}");
    }

    /// Secondary hand check at `q = 0`: `(0, s₂, s₁ + s₆)` with rotation `[[0,1,0],[−1,0,0],[0,0,1]]`, and
    /// the prismatic joint extends the tip linearly along its axis.
    #[test]
    fn stanford_home_pose_computed_by_hand() {
        let r = stanford();
        let p = position(&r, &[0.0; 6]);
        assert!(close(&p, &Vector3::new(0.0, 0.15367, 0.675386), 1e-9), "home: {p:?}");
        let rot = r.fk(&[0.0; 6]).rotation.to_rotation_matrix();
        let want = Matrix3::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert!((rot.matrix() - want).norm() < 1e-12, "home rotation: {rot:?}");
        let p1 = position(&r, &[0.0, 0.0, 0.1, 0.0, 0.0, 0.0]);
        let p2 = position(&r, &[0.0, 0.0, 0.2, 0.0, 0.0, 0.0]);
        assert!(close(&(p2 - p1), &(p1 - p), 1e-12), "prismatic extension must be linear in q₃");
        assert!((p1 - p).norm() > 0.09, "extension must move the tip");
        for (i, j) in r.joints.iter().enumerate() {
            assert_eq!(j.limits, None, "no limit is stated for joint {}", i + 1);
            assert_eq!(j.effort, None, "no effort is stated for joint {}", i + 1);
            assert_eq!(j.max_velocity, None, "no velocity is stated for joint {}", i + 1);
        }
    }

    #[test]
    fn stanford_passes_the_hessian_finite_difference_check() {
        hessian_matches_finite_differences(&stanford(), &[0.3, -0.5, 0.12, 0.7, -0.4, 0.9]);
    }

    /// Reading Paul's standard table as modified DH builds a different arm: measured 0.9459 m at the
    /// example pose.
    #[test]
    fn stanford_convention_argument_is_load_bearing() {
        let d = swap_distance_of(&stanford(), &stanford_rows(), DhConvention::Standard, Iso::identity(), &paul_example_q());
        assert!(d > 1e-3, "convention swap moved the known-answer pose by only {d:e} m");
        assert!((d - 0.9459408).abs() < 1e-6, "measured swap distance changed: {d}");
    }

    // ---------------------------------------------------------------- two-link planar


    /// **Hand computation** (the source gives symbolic lengths): `(c₁ + c₁₂, s₁ + s₁₂, 0)` at three poses.
    #[test]
    fn two_link_planar_known_answer_computed_by_hand() {
        let r = two_link_planar();
        assert_eq!(r.dof(), 2);
        let deg = f64::to_radians;
        let p = position(&r, &[deg(30.0), deg(60.0)]);
        let home = position(&r, &[0.0, 0.0]);
        assert!((p - home).norm() > 1.0, "poses must differ: {p:?} vs {home:?}");
        assert!(close(&p, &Vector3::new(0.8660254037844387, 1.5, 0.0), 1e-9), "(30°, 60°): {p:?}");
        assert!(close(&home, &Vector3::new(2.0, 0.0, 0.0), 1e-9), "home: {home:?}");
        let up = position(&r, &[FRAC_PI_2, 0.0]);
        assert!(close(&up, &Vector3::new(0.0, 2.0, 0.0), 1e-9), "(90°, 0): {up:?}");
        for j in &r.joints {
            assert!(j.limits.is_none() && j.effort.is_none() && j.max_velocity.is_none(), "nothing is sourced");
        }
    }

    #[test]
    fn two_link_planar_passes_the_hessian_finite_difference_check() {
        hessian_matches_finite_differences(&two_link_planar(), &[0.3, -0.5]);
    }

    /// Under modified DH the same rows put both links before their joints: tip `(1 + c₁, s₁, 0)`,
    /// measured 1.414 m from the standard-DH tip at the known-answer pose.
    #[test]
    fn two_link_planar_convention_argument_is_load_bearing() {
        let deg = f64::to_radians;
        let d = swap_distance_of(&two_link_planar(), &two_link_planar_rows(), DhConvention::Standard, Iso::identity(), &[deg(30.0), deg(60.0)]);
        assert!(d > 1e-3, "convention swap moved the known-answer pose by only {d:e} m");
        assert!((d - std::f64::consts::SQRT_2).abs() < 1e-9, "measured swap distance changed: {d}");
    }

    // ---------------------------------------------------------------- three-link planar


    /// **Hand computation** (the source gives symbolic lengths): frame 3 sits on the joint-3 axis, so
    /// `q₃` turns it without moving it and the yaw is `θ₁ + θ₂ + θ₃`.
    #[test]
    fn three_link_planar_known_answer_computed_by_hand() {
        let r = three_link_planar();
        assert_eq!(r.dof(), 3);
        let deg = f64::to_radians;
        let p = position(&r, &[deg(30.0); 3]);
        let home = position(&r, &[0.0; 3]);
        assert!((p - home).norm() > 1.0, "poses must differ: {p:?} vs {home:?}");
        assert!(close(&p, &Vector3::new(1.3660254037844386, 1.3660254037844386, 0.0), 1e-9), "(30°, 30°, 30°): {p:?}");
        assert!(close(&home, &Vector3::new(2.0, 0.0, 0.0), 1e-9), "home: {home:?}");
        let p3 = position(&r, &[deg(30.0), deg(30.0), 0.0]);
        assert!(close(&p3, &p, 1e-12), "joint 3 must not move the frame-3 origin: {p3:?}");
        let yaw = r.fk(&[deg(30.0); 3]).rotation.euler_angles().2;
        assert!((yaw - deg(90.0)).abs() < 1e-12, "yaw should be θ₁ + θ₂ + θ₃ = 90°, got {yaw}");
        let yaw3 = r.fk(&[deg(30.0), deg(30.0), 0.0]).rotation.euler_angles().2;
        assert!((yaw3 - deg(60.0)).abs() < 1e-12, "joint 3 changes the yaw even though it does not move the origin");
    }

    #[test]
    fn three_link_planar_passes_the_hessian_finite_difference_check() {
        hessian_matches_finite_differences(&three_link_planar(), &[0.3, -0.5, 0.8]);
    }

    /// Under modified DH the third row's `a = 0` drops one link out of the position: tip `(1 + c₁, s₁, 0)`,
    /// measured 1.0 m from the standard-DH tip at the known-answer pose.
    #[test]
    fn three_link_planar_convention_argument_is_load_bearing() {
        let deg = f64::to_radians;
        let d = swap_distance_of(&three_link_planar(), &three_link_planar_rows(), DhConvention::Standard, Iso::identity(), &[deg(30.0); 3]);
        assert!(d > 1e-3, "convention swap moved the known-answer pose by only {d:e} m");
        assert!((d - 1.0).abs() < 1e-9, "measured swap distance changed: {d}");
    }
}
