//! **Kinova arms from their published Denavit–Hartenberg tables**: Gen3 7 DoF, Gen3 6 DoF, Gen3 lite
//! and no JACO: see below.
//!
//! Every table here is the manufacturer's own, read in the **standard** (classical) convention the
//! documents state, `T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`. The Gen3 user guide prints link lengths
//! in millimetres, often as sums (e.g. `-(156.4 + 128.4)`); the specification this module was written
//! from carries them in metres and the values are used as given. The guide's `π` and `π/2` entries
//! appear in the specification as the seven-digit `3.1415927` / `1.5707963`; this module substitutes
//! the exact `PI` / `FRAC_PI_2`, which moves the Gen3 7 DoF home pose by `7.1e-8` m (measured: the
//! seven-digit constants leave a `7.1e-8` m residual against the known answer, the exact ones `1.9e-16`).
//! Joint limits are in radians as the specification converts them from the guides' degree tables;
//! efforts in N·m and rates in rad/s are the guides' "soft limit" values, verbatim.
//!
//! **Verified against**: for each arm, forward kinematics at the specification's known-answer
//! configuration (a hand computation from the table, since none of the guides prints an end-effector
//! position), the analytic kinematic Hessian against central differences of the Jacobian, and a
//! convention-swap control — the same rows read as modified DH move the home pose by 0.5–1.5 m, so
//! the convention argument is load-bearing for every table here.
//!
//! **What none of the sources state**: the JACO document gives no joint limits in DH units, no
//! efforts and no rates, so that model carries none. The Gen3 7 DoF and 6 DoF guides mark joints
//! 1, 3, 5, 7 (7 DoF) and 1, 4, 6 (6 DoF) as continuous; those joints carry no limits here.

//! **JACO (Gen1) is deliberately not shipped.** The only DH table located for it is a PDF stamped
//! "Confidential & Proprietary ... property of Kinova", obtained from a third-party repository. Numbers
//! from a document that says it may not be reproduced are not a source this crate can cite, whatever
//! their accuracy, so the constructor was removed before the first release rather than shipped with a
//! caveat. It returns if Kinova publishes the table.

use crate::{DhConvention, DhRow, Robot};
use ferromotion_core::Iso;
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use std::f64::consts::{FRAC_PI_2, PI};

/// `Rx(t)`: the fixed base transform the Gen3 guides print as their "row 0" (`α = π`, no joint).
fn rx(t: f64) -> Iso {
    Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), t))
}

/// [`Robot::from_dh_based`] with the tool at identity, since every table here ends at the interface
/// plate. A fixed row is not a joint, so the Gen3 guides' row 0 (`Rz(0)·Tz(0)·Tx(0)·Rx(π) = Rx(π)`) goes
/// in as `base`; the other two tables pass identity.
fn build(base: Iso, rows: &[DhRow], convention: DhConvention) -> Robot {
    Robot::from_dh_based(base, rows, convention, Iso::identity()).expect("a finite, non-empty DH table")
}

// ---------------------------------------------------------------------------------------------
// Gen3 7 DoF
// ---------------------------------------------------------------------------------------------

/// Joints 1–7 of the Gen3 7 DoF table. Row 0 (fixed, `α = π`) is the base transform in [`build`].
fn gen3_7dof_rows() -> [DhRow; 7] {
    [
        // effort N·m (guide Table 44 soft limits) and rate rad/s (general mode) on every row
        DhRow::revolute(0.0, -0.2848, 0.0, FRAC_PI_2).with_effort(39.0).with_max_velocity(1.39), // continuous
        DhRow::revolute(PI, -0.0118, 0.0, FRAC_PI_2).with_limits(-2.2497, 2.2497).with_effort(39.0).with_max_velocity(1.39),
        DhRow::revolute(PI, -0.4208, 0.0, FRAC_PI_2).with_effort(39.0).with_max_velocity(1.39), // continuous
        DhRow::revolute(PI, -0.0128, 0.0, FRAC_PI_2).with_limits(-2.5796, 2.5796).with_effort(39.0).with_max_velocity(1.39),
        DhRow::revolute(PI, -0.3143, 0.0, FRAC_PI_2).with_effort(9.0).with_max_velocity(1.22), // continuous
        DhRow::revolute(PI, 0.0, 0.0, FRAC_PI_2).with_limits(-2.0996, 2.0996).with_effort(9.0).with_max_velocity(1.22),
        DhRow::revolute(PI, -0.1674, 0.0, PI).with_effort(9.0).with_max_velocity(1.22), // continuous; ends at the interface plate
    ]
}

fn gen3_7dof_in(convention: DhConvention) -> Robot {
    build(rx(PI), &gen3_7dof_rows(), convention)
}

/// **Kinova Gen3, 7 DoF, spherical wrist**, from its classical DH table.
///
/// **Source** (verbatim from the specification): *KINOVA Gen3 Ultra lightweight robot User Guide, R07
/// (2022): Table 94 '7 DoF spherical Classical DH parameters' p.198; Tables 39-44 joint limits
/// pp.97-98; Tables 98-105 inertial parameters pp.204-207* —
/// <https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf>. Convention: standard DH.
/// Confidence: **published primary source**.
///
/// **Table** (θ offset rad, d m, a m, α rad), after the fixed row 0 `(0, 0, 0, π)` that becomes the base
/// transform: `(0, −0.2848, 0, π/2)`, `(π, −0.0118, 0, π/2)`, `(π, −0.4208, 0, π/2)`,
/// `(π, −0.0128, 0, π/2)`, `(π, −0.3143, 0, π/2)`, `(π, 0, 0, π/2)`, `(π, −0.1674, 0, π)`. The guide
/// writes the `d` values as sums of link lengths in mm, e.g. `−(156.4 + 128.4)`; the specification
/// carries them in metres and that is what is used. The last row ends at the tool interface plate, so
/// the tool offset is zero. **Units**: `π` and `π/2` are the exact constants rather than the
/// specification's seven-digit decimals (see the module doc for what that changes).
///
/// **Limits** (rad, from the guide's degree tables as converted by the specification): joints 2, 4, 6
/// are `±2.2497`, `±2.5796`, `±2.0996`; joints 1, 3, 5, 7 are continuous and carry `None`.
/// **Effort** (soft limits, Table 44): 39 N·m on joints 1–4, 9 N·m on joints 5–7. **Rate**: 1.39 rad/s
/// on joints 1–4, 1.22 rad/s on joints 5–7 (general mode; the guide's admittance/force-mode figure of
/// 0.8727 rad/s is not encoded).
///
/// **Known answer** (computed by hand from the table, not printed by the guide): at `q = 0` the
/// interface frame is at `(0, −0.0246, 1.1873)` m — the specification's derivation is
/// `z = 0.1564 + 0.1284 + 0.0054 + 0.0064 + 0.2104 + 0.2104 + 0.0064 + 0.0064 + 0.2084 + 0.1059 + 0.1059 + 0.0615`
/// and `y = −(0.0054 + 0.0064) − (0.0064 + 0.0064)`: the arm straight up with small lateral offsets.
/// Verified here to `1e-9` m (measured residual `1.9e-16` m).
pub fn gen3_7dof() -> Robot {
    gen3_7dof_in(DhConvention::Standard)
}

// ---------------------------------------------------------------------------------------------
// Gen3 6 DoF
// ---------------------------------------------------------------------------------------------

/// Joints 1–6 of the Gen3 6 DoF table. Row 0 (fixed, `α = π`) is the base transform in [`build`].
fn gen3_6dof_rows() -> [DhRow; 6] {
    [
        // effort N·m (guide soft limits) and rate rad/s (general mode) on every row
        DhRow::revolute(0.0, -0.28481, 0.0, FRAC_PI_2).with_effort(39.0).with_max_velocity(1.39), // continuous
        DhRow::revolute(-FRAC_PI_2, -0.00538, 0.41, PI).with_limits(-2.2497, 2.2497).with_effort(39.0).with_max_velocity(1.39),
        DhRow::revolute(-FRAC_PI_2, -0.00638, 0.0, FRAC_PI_2).with_limits(-2.5796, 2.5796).with_effort(39.0).with_max_velocity(1.39),
        DhRow::revolute(PI, -0.31436, 0.0, FRAC_PI_2).with_effort(9.0).with_max_velocity(1.22), // continuous
        DhRow::revolute(PI, 0.0, 0.0, FRAC_PI_2).with_limits(-2.0996, 2.0996).with_effort(9.0).with_max_velocity(1.22),
        DhRow::revolute(PI, -0.16746, 0.0, PI).with_effort(9.0).with_max_velocity(1.22), // continuous; ends at the interface plate
    ]
}

fn gen3_6dof_in(convention: DhConvention) -> Robot {
    build(rx(PI), &gen3_6dof_rows(), convention)
}

/// **Kinova Gen3, 6 DoF, spherical wrist**, from its classical DH table.
///
/// **Source** (verbatim from the specification): *KINOVA Gen3 Ultra lightweight robot User Guide, R07
/// (2022): Table 95 '6 DoF spherical Classical DH parameters' p.200 and Figure 89; Tables 45-50 joint
/// limits pp.98-99; inertial parameters of the 6 DoF robot pp.207ff* —
/// <https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf>. Convention: standard DH.
/// Confidence: **published primary source**.
///
/// **Transcription caution carried from the specification**: in the printed Table 95 the column
/// headers read `(α, a, d, θ)` but the numbers under them are in the order `(a [mm], d [mm], α, θ)`;
/// row 2 prints `410.0, −5.38, π, q₂ − π/2` and Figure 89 confirms 410 mm is the link length. The
/// table here follows that reading.
///
/// **Table** (θ offset rad, d m, a m, α rad), after the fixed row 0 `(0, 0, 0, π)` that becomes the base
/// transform: `(0, −0.28481, 0, π/2)`, `(−π/2, −0.00538, 0.41, π)`, `(−π/2, −0.00638, 0, π/2)`,
/// `(π, −0.31436, 0, π/2)`, `(π, 0, 0, π/2)`, `(π, −0.16746, 0, π)`. **Units**: guide mm → metres by the
/// specification; exact `π` / `π/2` constants here.
///
/// **Limits** (rad): joints 2, 3, 5 are `±2.2497`, `±2.5796`, `±2.0996`; joints 1, 4, 6 are continuous
/// and carry `None`. **Effort** (soft limits): 39 N·m joints 1–3, 9 N·m joints 4–6. **Rate**: 1.39 rad/s
/// joints 1–3, 1.22 rad/s joints 4–6.
///
/// **Known answer** (computed by hand from the table, not stated by the guide): at `q = 0` the
/// interface frame is at `(0, 0.001, 1.17663)` m — the specification's derivation is
/// `z = 156.43 + 128.38 + 410 + 208.43 + 105.93 + 105.93 + 61.53 = 1176.63` mm and
/// `y = −5.38 + 6.38 = 1.0` mm. Verified here to `1e-9` m (measured residual `6.6e-16` m).
pub fn gen3_6dof() -> Robot {
    gen3_6dof_in(DhConvention::Standard)
}

// ---------------------------------------------------------------------------------------------
// Gen3 lite
// ---------------------------------------------------------------------------------------------

/// The Gen3 lite table, joints 1–6. No fixed row 0 in this guide.
fn gen3_lite_rows() -> [DhRow; 6] {
    [
        // 9 N·m soft limit on every joint; 1.0 rad/s on joints 1–5, 1.57 rad/s on joint 6
        DhRow::revolute(0.0, 0.2433, 0.0, FRAC_PI_2).with_limits(-2.6896, 2.6896).with_effort(9.0).with_max_velocity(1.0),
        DhRow::revolute(FRAC_PI_2, 0.03, 0.28, PI).with_limits(-2.6198, 2.6198).with_effort(9.0).with_max_velocity(1.0),
        DhRow::revolute(FRAC_PI_2, 0.02, 0.0, FRAC_PI_2).with_limits(-2.6198, 2.6198).with_effort(9.0).with_max_velocity(1.0),
        DhRow::revolute(FRAC_PI_2, 0.245, 0.0, FRAC_PI_2).with_limits(-2.6002, 2.6002).with_effort(9.0).with_max_velocity(1.0),
        DhRow::revolute(PI, 0.057, 0.0, FRAC_PI_2).with_limits(-2.5302, 2.5307).with_effort(9.0).with_max_velocity(1.0),
        DhRow::revolute(FRAC_PI_2, 0.235, 0.0, 0.0).with_limits(-2.6002, 2.6002).with_effort(9.0).with_max_velocity(1.57),
    ]
}

fn gen3_lite_in(convention: DhConvention) -> Robot {
    build(Iso::identity(), &gen3_lite_rows(), convention)
}

/// **Kinova Gen3 lite**, from its classical DH table.
///
/// **Source** (verbatim from the specification): *KINOVA Gen3 lite robot User guide: Table 50
/// 'Classical DH parameters' p.135, Figures 63-64; Tables 25-28 joint limits pp.71-72; Tables 52-58
/// inertial parameters pp.138ff* —
/// <https://static.generation-robots.com/media/Kinova-lite-fiche-technique.pdf>. Convention:
/// standard DH. Confidence: **published primary source**.
///
/// **Table** (θ offset rad, d m, a m, α rad): `(0, 0.2433, 0, π/2)`, `(π/2, 0.03, 0.28, π)`,
/// `(π/2, 0.02, 0, π/2)`, `(π/2, 0.245, 0, π/2)`, `(π, 0.057, 0, π/2)`, `(π/2, 0.235, 0, 0)`. Table 50
/// prints `d` as sums in mm, e.g. `(128.3 + 115.0)`; the specification carries metres. The last row ends
/// at the tool interface, so the tool offset is zero. **Units**: guide mm → metres by the specification;
/// exact `π` / `π/2` constants here.
///
/// **Limits** (rad; the guide's Table 25 gives degrees: ±154.1, ±150.1, ±150.1, ±148.98,
/// −144.97/+145.0, ±148.98, converted by the specification): `±2.6896`, `±2.6198`, `±2.6198`,
/// `±2.6002`, `−2.5302/+2.5307`, `±2.6002`. **Effort**: 9 N·m soft limit on every joint. **Rate**:
/// 1.0 rad/s on joints 1–5, 1.57 rad/s on joint 6.
///
/// **Known answer** (computed by hand from the table, not stated by the guide): at `q = 0` the tool
/// interface is at `(0.057, −0.01, 1.0033)` m — the specification's derivation is
/// `z = 243.3 + 280 + 245 + 235 = 1003.3` mm, `x = 28.5 + 28.5 = 57` mm (wrist offset), and
/// `y = 30 − 20 = 10` mm with the sign from the frame chain. Verified here to `1e-9` m (measured
/// residual `3.3e-16` m).
pub fn gen3_lite() -> Robot {
    gen3_lite_in(DhConvention::Standard)
}

// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    fn tip(r: &Robot, q: &[f64]) -> Vector3<f64> {
        r.fk(q).translation.vector
    }

    /// The known-answer pattern every model shares. Non-vacuity first: the tip must move when a joint
    /// moves (a robot whose FK ignored `q` would sit at any fixed answer for free), and the answer must
    /// not be the origin; only then the tolerance assertion against the hand-computed pose.
    fn known_answer(name: &str, r: &Robot, q: &[f64], want: Vector3<f64>, tol: f64) {
        let mut moved = q.to_vec();
        moved[1] += 0.5;
        let (p, p2) = (tip(r, q), tip(r, &moved));
        assert!((p - p2).norm() > 0.05, "{name}: the tip must move when joint 2 moves, {p:?} vs {p2:?}");
        assert!(want.norm() > 0.1, "{name}: a fixture at the origin would not test the link lengths");
        assert!(close(&p, &want, tol), "{name}: FK at {q:?} = {p:?}, hand answer {want:?}, error {:e}", (p - want).norm());
    }

    /// The central-difference Hessian check, copied from
    /// `ferromotion_core::dh::tests::a_dh_arm_passes_the_hessian_finite_difference_check`.
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
        assert!(worst < 1e-6 * scale, "{name}: Hessian vs FD: worst {worst:e} on a scale of {scale:e}");
    }

    /// The same rows read under the other convention must build a *different* arm. If it did not, the
    /// convention argument would be decorative for this table. The threshold is 1 mm; the measured
    /// moves are 0.5–1.5 m (see each test).
    fn convention_swap(name: &str, standard: &Robot, modified: &Robot, q: &[f64]) -> f64 {
        let (s, m) = (tip(standard, q), tip(modified, q));
        assert!(s.norm() > 0.1 && m.norm() > 0.1, "{name}: both builds must place the tip somewhere");
        let moved = (s - m).norm();
        assert!(moved > 1e-3, "{name}: reading the table as modified DH moved the tip only {moved:e} m");
        moved
    }

    // ---- Gen3 7 DoF ---------------------------------------------------------------------------

    const Q7: [f64; 7] = [0.0; 7];

    /// Hand-computed from Table 94 (the guide prints no EE position): `(0, −0.0246, 1.1873)` m.
    #[test]
    fn gen3_7dof_matches_the_known_answer_computed_by_hand_from_the_table() {
        known_answer("gen3_7dof", &gen3_7dof(), &Q7, Vector3::new(0.0, -0.0246, 1.1873), 1e-9);
    }

    #[test]
    fn gen3_7dof_has_seven_joints_and_the_stated_limits_and_ratings() {
        let r = gen3_7dof();
        assert_eq!(r.dof(), 7);
        let limits: Vec<_> = r.joints.iter().map(|j| j.limits).collect();
        assert_eq!(limits, vec![None, Some((-2.2497, 2.2497)), None, Some((-2.5796, 2.5796)), None, Some((-2.0996, 2.0996)), None], "joints 1, 3, 5, 7 are continuous in the guide");
        let effort: Vec<_> = r.joints.iter().map(|j| j.effort).collect();
        assert_eq!(effort, vec![Some(39.0), Some(39.0), Some(39.0), Some(39.0), Some(9.0), Some(9.0), Some(9.0)]);
        let rate: Vec<_> = r.joints.iter().map(|j| j.max_velocity).collect();
        assert_eq!(rate, vec![Some(1.39), Some(1.39), Some(1.39), Some(1.39), Some(1.22), Some(1.22), Some(1.22)]);
    }

    #[test]
    fn gen3_7dof_passes_the_hessian_finite_difference_check() {
        hessian_check("gen3_7dof", &gen3_7dof(), &[0.3, -0.5, 0.12, 0.7, -0.4, 0.9, -0.2]);
    }

    /// Measured move under the swap: 1.44 m.
    #[test]
    fn gen3_7dof_read_as_modified_dh_is_a_different_arm() {
        let moved = convention_swap("gen3_7dof", &gen3_7dof(), &gen3_7dof_in(DhConvention::Modified), &Q7);
        assert!(moved > 1.0, "measured 1.44 m; got {moved}");
    }

    /// The fixed row 0 is load-bearing: without `Rx(π)` the guide's negative `d` values would put the
    /// arm below the base. This pins that the base transform is applied, and applied once.
    #[test]
    fn gen3_7dof_base_row_flips_the_arm_upright() {
        let without = build(Iso::identity(), &gen3_7dof_rows(), DhConvention::Standard);
        let p = tip(&without, &Q7);
        assert!(close(&p, &Vector3::new(0.0, 0.0246, -1.1873), 1e-9), "without row 0 the arm hangs down: {p:?}");
    }

    // ---- Gen3 6 DoF ---------------------------------------------------------------------------

    const Q6: [f64; 6] = [0.0; 6];

    /// Hand-computed from Table 95 (not stated by the guide): `(0, 0.001, 1.17663)` m.
    #[test]
    fn gen3_6dof_matches_the_known_answer_computed_by_hand_from_the_table() {
        known_answer("gen3_6dof", &gen3_6dof(), &Q6, Vector3::new(0.0, 0.001, 1.17663), 1e-9);
    }

    #[test]
    fn gen3_6dof_has_six_joints_and_the_stated_limits_and_ratings() {
        let r = gen3_6dof();
        assert_eq!(r.dof(), 6);
        let limits: Vec<_> = r.joints.iter().map(|j| j.limits).collect();
        assert_eq!(limits, vec![None, Some((-2.2497, 2.2497)), Some((-2.5796, 2.5796)), None, Some((-2.0996, 2.0996)), None], "joints 1, 4, 6 are continuous in the guide");
        let effort: Vec<_> = r.joints.iter().map(|j| j.effort).collect();
        assert_eq!(effort, vec![Some(39.0), Some(39.0), Some(39.0), Some(9.0), Some(9.0), Some(9.0)]);
        let rate: Vec<_> = r.joints.iter().map(|j| j.max_velocity).collect();
        assert_eq!(rate, vec![Some(1.39), Some(1.39), Some(1.39), Some(1.22), Some(1.22), Some(1.22)]);
    }

    #[test]
    fn gen3_6dof_passes_the_hessian_finite_difference_check() {
        hessian_check("gen3_6dof", &gen3_6dof(), &[0.3, -0.5, 0.12, 0.7, -0.4, 0.9]);
    }

    /// Measured move under the swap: 1.54 m.
    #[test]
    fn gen3_6dof_read_as_modified_dh_is_a_different_arm() {
        let moved = convention_swap("gen3_6dof", &gen3_6dof(), &gen3_6dof_in(DhConvention::Modified), &Q6);
        assert!(moved > 1.0, "measured 1.54 m; got {moved}");
    }

    // ---- Gen3 lite ----------------------------------------------------------------------------

    /// Hand-computed from Table 50 (not stated by the guide): `(0.057, −0.01, 1.0033)` m.
    #[test]
    fn gen3_lite_matches_the_known_answer_computed_by_hand_from_the_table() {
        known_answer("gen3_lite", &gen3_lite(), &Q6, Vector3::new(0.057, -0.01, 1.0033), 1e-9);
    }

    #[test]
    fn gen3_lite_has_six_joints_and_the_stated_limits_and_ratings() {
        let r = gen3_lite();
        assert_eq!(r.dof(), 6);
        let limits: Vec<_> = r.joints.iter().map(|j| j.limits).collect();
        assert_eq!(limits, vec![Some((-2.6896, 2.6896)), Some((-2.6198, 2.6198)), Some((-2.6198, 2.6198)), Some((-2.6002, 2.6002)), Some((-2.5302, 2.5307)), Some((-2.6002, 2.6002))]);
        assert!(r.joints.iter().all(|j| j.effort == Some(9.0)), "9 N·m soft limit on every joint");
        let rate: Vec<_> = r.joints.iter().map(|j| j.max_velocity).collect();
        assert_eq!(rate, vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(1.57)]);
    }

    #[test]
    fn gen3_lite_passes_the_hessian_finite_difference_check() {
        hessian_check("gen3_lite", &gen3_lite(), &[0.3, -0.5, 0.12, 0.7, -0.4, 0.9]);
    }

    /// Measured move under the swap: 1.27 m.
    #[test]
    fn gen3_lite_read_as_modified_dh_is_a_different_arm() {
        let moved = convention_swap("gen3_lite", &gen3_lite(), &gen3_lite_in(DhConvention::Modified), &Q6);
        assert!(moved > 1.0, "measured 1.27 m; got {moved}");
    }



    /// **Orientation at q = 0 is the identity, by hand.** The six constant rotations of the Gen3 lite
    /// table, `Rz(θ₀)·Rx(α)` per row with θ₀ ∈ {0, π/2, π/2, π/2, π, π/2} and α ∈ {π/2, π, π/2, π/2, π/2, 0},
    /// multiply to `I`, computed independently of this crate. Every other kinova test pins position only,
    /// and a wrong twist on the LAST row moves no position at all: it only rotates the tool frame. This is
    /// the pin that sees it.
    #[test]
    fn gen3_lite_home_orientation_is_the_identity_computed_by_hand() {
        let r = gen3_lite().fk(&[0.0; 6]).rotation.to_rotation_matrix();
        let worst = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| (r[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs()).fold(0.0f64, f64::max);
        assert!(worst < 1e-9, "home rotation should be I, worst entry error {worst:e}: {r}");
    }

}
