//! **Denavit–Hartenberg construction**, so a robot can be built from the table its datasheet or paper
//! publishes rather than from a URDF someone had to write first.
//!
//! Two conventions are in use and they are not interchangeable. A **standard** (Spong / Paul) row is
//! `T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)`, with `a_i, α_i` describing the link *after* joint `i`. A
//! **modified** (Craig) row is `T_i = Rx(α_{i−1})·Tx(a_{i−1})·Rz(θ_i)·Tz(d_i)`, with `a, α` describing
//! the link *before* it. Manufacturers use both — Universal Robots publishes standard, Franka publishes
//! modified — and reading one as the other builds a different arm that still looks plausible, which is
//! why [`DhConvention`] is a required argument and not a default.
//!
//! # How a row becomes a [`Joint`]
//!
//! This crate represents a joint as a fixed `origin` transform followed by a motion about the joint
//! frame's own axis. Each DH row splits into a constant part *before* the variable, the variable itself
//! (always about or along `z`), and a constant part *after* it:
//!
//! | convention | joint | before | variable | after |
//! |---|---|---|---|---|
//! | standard | revolute, `θ = θ₀ + q` | `Rz(θ₀)` | `Rz(q)` | `Tz(d)·Tx(a)·Rx(α)` |
//! | standard | prismatic, `d = d₀ + q` | `Rz(θ)·Tz(d₀)` | `Tz(q)` | `Tx(a)·Rx(α)` |
//! | modified | revolute | `Rx(α)·Tx(a)·Rz(θ₀)` | `Rz(q)` | `Tz(d)` |
//! | modified | prismatic | `Rx(α)·Tx(a)·Rz(θ)·Tz(d₀)` | `Tz(q)` | identity |
//!
//! Joint `i`'s `origin` is then the previous row's *after* part times this row's *before* part, its axis
//! is `z`, and the last row's *after* part folds into the tool offset. The chain product is unchanged;
//! only where the constant transforms are attached moves. Both conventions are verified against
//! hand-computed forward kinematics, against each other on the same physical arm, and against the
//! finite-difference Jacobian and Hessian checks that every [`Robot`] in this crate is held to.

use crate::{Iso, Joint, JointKind, Robot};
use nalgebra::{Translation3, UnitQuaternion, Vector3};

/// Which Denavit–Hartenberg convention a table is written in. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DhConvention {
    /// `T_i = Rz(θ_i)·Tz(d_i)·Tx(a_i)·Rx(α_i)` — Spong, Paul, Universal Robots' published tables.
    Standard,
    /// `T_i = Rx(α_{i−1})·Tx(a_{i−1})·Rz(θ_i)·Tz(d_i)` — Craig; Franka's published tables.
    Modified,
}

/// One row of a Denavit–Hartenberg table.
///
/// `theta` and `d` are the row's constant values; for a revolute joint `theta` is the **offset** added
/// to the joint variable, and for a prismatic joint `d` is. Angles in radians, lengths in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DhRow {
    pub kind: JointKind,
    pub theta: f64,
    pub d: f64,
    pub a: f64,
    pub alpha: f64,
    /// Optional `(lower, upper)` joint limit, in radians or metres.
    pub limits: Option<(f64, f64)>,
    /// Optional actuator effort the source states, N·m or N. Left `None` when the source gives none,
    /// rather than guessed; a model that needs one must then be handed it explicitly.
    pub effort: Option<f64>,
    /// Optional joint-rate limit the source states, rad/s or m/s. Same rule as `effort`.
    pub max_velocity: Option<f64>,
}

impl DhRow {
    /// A revolute row with `theta` as its offset.
    pub fn revolute(theta: f64, d: f64, a: f64, alpha: f64) -> DhRow {
        DhRow { kind: JointKind::Revolute, theta, d, a, alpha, limits: None, effort: None, max_velocity: None }
    }

    /// A prismatic row with `d` as its offset.
    pub fn prismatic(theta: f64, d: f64, a: f64, alpha: f64) -> DhRow {
        DhRow { kind: JointKind::Prismatic, theta, d, a, alpha, limits: None, effort: None, max_velocity: None }
    }

    /// Attach a `(lower, upper)` limit.
    pub fn with_limits(mut self, lower: f64, upper: f64) -> DhRow {
        self.limits = Some((lower, upper));
        self
    }

    /// Attach the effort the source states (N·m or N); non-positive or non-finite is treated as unstated.
    pub fn with_effort(mut self, effort: f64) -> DhRow {
        self.effort = (effort.is_finite() && effort > 0.0).then_some(effort);
        self
    }

    /// Attach the rate limit the source states (rad/s or m/s); non-positive or non-finite is unstated.
    pub fn with_max_velocity(mut self, v: f64) -> DhRow {
        self.max_velocity = (v.is_finite() && v > 0.0).then_some(v);
        self
    }
}

fn rz(t: f64) -> Iso {
    Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Vector3::z_axis(), t))
}
fn rx(t: f64) -> Iso {
    Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), t))
}
fn tz(d: f64) -> Iso {
    Iso::from_parts(Translation3::new(0.0, 0.0, d), UnitQuaternion::identity())
}
fn tx(a: f64) -> Iso {
    Iso::from_parts(Translation3::new(a, 0.0, 0.0), UnitQuaternion::identity())
}

/// The constant parts of a row around its variable: `(before, after)`.
fn split(row: &DhRow, convention: DhConvention) -> (Iso, Iso) {
    match (convention, row.kind) {
        (DhConvention::Standard, JointKind::Revolute) => (rz(row.theta), tz(row.d) * tx(row.a) * rx(row.alpha)),
        (DhConvention::Standard, JointKind::Prismatic) => (rz(row.theta) * tz(row.d), tx(row.a) * rx(row.alpha)),
        (DhConvention::Modified, JointKind::Revolute) => (rx(row.alpha) * tx(row.a) * rz(row.theta), tz(row.d)),
        (DhConvention::Modified, JointKind::Prismatic) => (rx(row.alpha) * tx(row.a) * rz(row.theta) * tz(row.d), Iso::identity()),
    }
}

impl Robot {
    /// Build a serial arm from a Denavit–Hartenberg table.
    ///
    /// `tool` is an extra fixed transform from the last DH frame to the end effector — the flange
    /// offset a datasheet gives separately — or identity. Returns `None` for an empty table or any
    /// non-finite entry, since a NaN in a DH row would otherwise build a robot that reports NaN poses
    /// without ever failing.
    pub fn from_dh(rows: &[DhRow], convention: DhConvention, tool: Iso) -> Option<Robot> {
        if rows.is_empty() {
            return None;
        }
        if rows.iter().any(|r| ![r.theta, r.d, r.a, r.alpha].iter().all(|v| v.is_finite())) {
            return None;
        }
        let mut joints = Vec::with_capacity(rows.len());
        let mut carry = Iso::identity(); // the previous row's `after` part
        for row in rows {
            let (before, after) = split(row, convention);
            let origin = carry * before;
            let mut j = match row.kind {
                JointKind::Revolute => Joint::revolute(origin, Vector3::z()),
                JointKind::Prismatic => Joint::prismatic(origin, Vector3::z()),
            };
            if let Some((lo, hi)) = row.limits {
                j = j.with_limits(lo, hi);
            }
            if let Some(e) = row.effort {
                j = j.with_effort(e);
            }
            if let Some(v) = row.max_velocity {
                j = j.with_max_velocity(v);
            }
            joints.push(j);
            carry = after;
        }
        Some(Robot { joints, ee_offset: carry * tool })
    }

    /// [`Self::from_dh`] with a fixed `base` transform ahead of the first row.
    ///
    /// Published tables often begin with a transform that carries no joint: a Kinova Gen3 table opens
    /// with a fixed `α = π` row, Baxter's arm frame sits on a shoulder rise and a 45° yaw, Universal
    /// Robots relate their DH base to the controller's `Base` by a half turn. A DH row cannot express
    /// that and a model would otherwise fold it into the first joint's origin by hand. This is the one
    /// place to put it: `fk = base · (chain) · tool`, verified as exactly that.
    pub fn from_dh_based(base: Iso, rows: &[DhRow], convention: DhConvention, tool: Iso) -> Option<Robot> {
        let mut r = Robot::from_dh(rows, convention, tool)?;
        r.joints[0].origin = base * r.joints[0].origin;
        Some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: &Vector3<f64>, b: &Vector3<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    /// **The 2R planar arm, by hand.** `x = a₁c₁ + a₂c₁₂`, `y = a₁s₁ + a₂s₁₂`, in both conventions.
    ///
    /// Standard DH puts each link length on its own row. Modified DH puts it on the *next* row and the
    /// last one in the tool, so the same arm is a different table — and if the split in [`split`] were
    /// wrong for either convention, one of these would fail against the hand formula.
    #[test]
    fn a_planar_2r_arm_matches_the_textbook_forward_kinematics_in_both_conventions() {
        let (a1, a2) = (0.7, 0.4);
        let standard = Robot::from_dh(&[DhRow::revolute(0.0, 0.0, a1, 0.0), DhRow::revolute(0.0, 0.0, a2, 0.0)], DhConvention::Standard, Iso::identity()).unwrap();
        let modified = Robot::from_dh(&[DhRow::revolute(0.0, 0.0, 0.0, 0.0), DhRow::revolute(0.0, 0.0, a1, 0.0)], DhConvention::Modified, tx(a2)).unwrap();
        for (q1, q2) in [(0.0, 0.0), (0.3, -0.5), (1.2, 0.8), (-2.0, 2.5), (FRAC_PI_2, FRAC_PI_2)] {
            let want = Vector3::new(a1 * q1.cos() + a2 * (q1 + q2).cos(), a1 * q1.sin() + a2 * (q1 + q2).sin(), 0.0);
            let s = standard.fk(&[q1, q2]).translation.vector;
            let m = modified.fk(&[q1, q2]).translation.vector;
            assert!(close(&s, &want, 1e-12), "standard DH at ({q1}, {q2}): {s:?} vs {want:?}");
            assert!(close(&m, &want, 1e-12), "modified DH at ({q1}, {q2}): {m:?} vs {want:?}");
        }
    }

    /// **A spatial arm with twist, by hand.** Three revolute joints, the first about `z`, then an
    /// `α = π/2` twist so the next two swing in a vertical plane — the shoulder of every anthropomorphic
    /// arm. At `q = 0` the tip is at `(a₂ + a₃, 0, d₁)`; at `q₂ = π/2` the upper arm points straight up.
    #[test]
    fn a_spatial_rrr_arm_puts_its_tip_where_the_geometry_says() {
        let (d1, a2, a3) = (0.5, 0.6, 0.4);
        let r = Robot::from_dh(
            &[DhRow::revolute(0.0, d1, 0.0, FRAC_PI_2), DhRow::revolute(0.0, 0.0, a2, 0.0), DhRow::revolute(0.0, 0.0, a3, 0.0)],
            DhConvention::Standard,
            Iso::identity(),
        )
        .unwrap();
        let p0 = r.fk(&[0.0, 0.0, 0.0]).translation.vector;
        assert!(close(&p0, &Vector3::new(a2 + a3, 0.0, d1), 1e-12), "home: {p0:?}");
        // upper arm vertical: after the α = π/2 twist, rotating joint 2 by π/2 lifts the arm along +z
        let p1 = r.fk(&[0.0, FRAC_PI_2, 0.0]).translation.vector;
        assert!(close(&p1, &Vector3::new(0.0, 0.0, d1 + a2 + a3), 1e-12), "vertical: {p1:?}");
        // base yaw of π/2 swings the whole home pose onto the y axis
        let p2 = r.fk(&[FRAC_PI_2, 0.0, 0.0]).translation.vector;
        assert!(close(&p2, &Vector3::new(0.0, a2 + a3, d1), 1e-12), "yawed: {p2:?}");
    }

    /// **A prismatic joint in each convention.** An RP arm: yaw about `z`, then extend along the
    /// twisted axis. Prismatic rows put the variable on `d`, and the two conventions attach `a, α` on
    /// different sides of it, so this catches a split that only handled the revolute case.
    #[test]
    fn a_prismatic_joint_extends_along_its_axis_in_both_conventions() {
        let d0 = 0.2;
        // standard: yaw, then a prismatic row whose z has been twisted to world +y by α = -π/2 on row 1
        let standard = Robot::from_dh(&[DhRow::revolute(0.0, 0.0, 0.0, -FRAC_PI_2), DhRow::prismatic(0.0, d0, 0.0, 0.0)], DhConvention::Standard, Iso::identity()).unwrap();
        let p = standard.fk(&[0.0, 0.3]).translation.vector;
        // Rx(-π/2) sends z to -y... check: Rx(-π/2)·ẑ = (0, sin(π/2), cos(π/2))?? compute honestly below
        let z_after_twist = rx(-FRAC_PI_2).rotation * Vector3::z();
        let want = z_after_twist * (d0 + 0.3);
        assert!(close(&p, &want, 1e-12), "standard RP: {p:?} vs {want:?}");
        // and extension is linear in q, which is the whole point of a prismatic joint
        let p2 = standard.fk(&[0.0, 0.6]).translation.vector;
        assert!(close(&(p2 - p), &(z_after_twist * 0.3), 1e-12));

        // modified: the twist lives on the prismatic row itself (α_{i−1}), same physical arm
        let modified = Robot::from_dh(&[DhRow::revolute(0.0, 0.0, 0.0, 0.0), DhRow::prismatic(0.0, d0, 0.0, -FRAC_PI_2)], DhConvention::Modified, Iso::identity()).unwrap();
        let m = modified.fk(&[0.0, 0.3]).translation.vector;
        assert!(close(&m, &want, 1e-12), "modified RP: {m:?} vs {want:?}");
        // yaw the base and the extension direction yaws with it
        let y = standard.fk(&[PI, 0.3]).translation.vector;
        assert!(close(&y, &(rz(PI).rotation * want), 1e-12), "yawed RP: {y:?}");
    }

    /// **Theta and d offsets are offsets**, added to the variable, not replacing it. A revolute row with
    /// `θ₀ = π/2` at `q = 0` must equal the same row with `θ₀ = 0` at `q = π/2`; same for `d₀` on a
    /// prismatic row. Getting this backwards makes every "home pose" in a datasheet come out rotated.
    #[test]
    fn offsets_add_to_the_joint_variable() {
        let with_offset = Robot::from_dh(&[DhRow::revolute(FRAC_PI_2, 0.0, 0.5, 0.0)], DhConvention::Standard, Iso::identity()).unwrap();
        let no_offset = Robot::from_dh(&[DhRow::revolute(0.0, 0.0, 0.5, 0.0)], DhConvention::Standard, Iso::identity()).unwrap();
        let a = with_offset.fk(&[0.3]).translation.vector;
        let b = no_offset.fk(&[0.3 + FRAC_PI_2]).translation.vector;
        assert!(close(&a, &b, 1e-12), "theta offset: {a:?} vs {b:?}");
        let pw = Robot::from_dh(&[DhRow::prismatic(0.0, 0.25, 0.0, 0.0)], DhConvention::Modified, Iso::identity()).unwrap();
        let pn = Robot::from_dh(&[DhRow::prismatic(0.0, 0.0, 0.0, 0.0)], DhConvention::Modified, Iso::identity()).unwrap();
        let c = pw.fk(&[0.1]).translation.vector;
        let d = pn.fk(&[0.35]).translation.vector;
        assert!(close(&c, &d, 1e-12), "d offset: {c:?} vs {d:?}");
    }

    /// A DH-built arm is a [`Robot`] like any other, so it must satisfy the same second-order
    /// kinematics checks: the analytic Hessian against central differences of the Jacobian.
    #[test]
    fn a_dh_arm_passes_the_hessian_finite_difference_check() {
        let r = Robot::from_dh(
            &[
                DhRow::revolute(0.0, 0.4, 0.0, FRAC_PI_2),
                DhRow::revolute(0.0, 0.0, 0.45, 0.0),
                DhRow::prismatic(FRAC_PI_2, 0.1, 0.0, FRAC_PI_2),
                DhRow::revolute(0.0, 0.3, 0.0, -FRAC_PI_2),
                DhRow::revolute(0.0, 0.0, 0.0, FRAC_PI_2),
            ],
            DhConvention::Standard,
            tx(0.08),
        )
        .unwrap();
        let q = [0.3, -0.5, 0.12, 0.7, -0.4];
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
        assert!(scale > 0.1, "the Jacobian should vary, scale {scale:e}");
        assert!(worst < 1e-6 * scale, "Hessian vs FD on a DH arm: worst {worst:e} on a scale of {scale:e}");
    }

    /// **Modified DH with a nonzero twist AND a nonzero theta offset on revolute rows, by hand.**
    ///
    /// Every other Modified fixture here has `θ₀ = 0` and `α = 0` on its revolute rows, and with both
    /// zero the order of `Rz(θ₀)` against `Rx(α)·Tx(a)` is invisible: `Rz(0)` and `Rx(0)` are identity.
    /// A mutation that put the z-group before the x-group in the Modified split passed all of them.
    /// This row has `α₁ = π/2` and `θ₀ = π/2`, which do not commute, so the order is now load-bearing.
    ///
    /// `T = Rz(q₁)·Tz(d₁)·Rx(π/2)·Tx(a₁)·Rz(q₂ + π/2)·Tx(a₂)`. At `q = 0`: `Rz(π/2)` turns the tool
    /// offset `(a₂,0,0)` into `(0,a₂,0)`; `Tx(a₁)` gives `(a₁,a₂,0)`; `Rx(π/2)` sends `y → z`, giving
    /// `(a₁,0,a₂)`; `Tz(d₁)` lifts it to `(a₁, 0, d₁ + a₂)`.
    #[test]
    fn modified_dh_with_twist_and_theta_offset_matches_the_hand_chain() {
        let (d1, a1, a2) = (0.3, 0.5, 0.2);
        let r = Robot::from_dh(
            &[DhRow::revolute(0.0, d1, 0.0, 0.0), DhRow::revolute(FRAC_PI_2, 0.0, a1, FRAC_PI_2)],
            DhConvention::Modified,
            tx(a2),
        )
        .unwrap();
        let p = r.fk(&[0.0, 0.0]).translation.vector;
        assert!(close(&p, &Vector3::new(a1, 0.0, d1 + a2), 1e-12), "home: {p:?}");
        // undo the offset with the joint: q₂ = −π/2 puts the tool back along x, tip at (a₁ + a₂, 0, d₁)
        let p2 = r.fk(&[0.0, -FRAC_PI_2]).translation.vector;
        assert!(close(&p2, &Vector3::new(a1 + a2, 0.0, d1), 1e-12), "offset undone: {p2:?}");
        // and the base yaw carries the whole thing round
        let p3 = r.fk(&[FRAC_PI_2, 0.0]).translation.vector;
        assert!(close(&p3, &Vector3::new(0.0, a1, d1 + a2), 1e-12), "yawed: {p3:?}");
    }

    /// A base transform composes on the left of the whole chain and nowhere else.
    #[test]
    fn a_base_transform_prepends_to_the_chain() {
        let rows = [DhRow::revolute(0.0, 0.3, 0.0, FRAC_PI_2), DhRow::revolute(0.0, 0.0, 0.5, 0.0)];
        let base = Iso::from_parts(Translation3::new(0.1, -0.2, 0.05), UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.7));
        let plain = Robot::from_dh(&rows, DhConvention::Standard, tx(0.1)).unwrap();
        let based = Robot::from_dh_based(base, &rows, DhConvention::Standard, tx(0.1)).unwrap();
        for q in [[0.0, 0.0], [0.4, -0.9], [-1.3, 2.1]] {
            let want = base * plain.fk(&q);
            let got = based.fk(&q);
            assert!((want.translation.vector - got.translation.vector).norm() < 1e-12, "at {q:?}: {got:?} vs {want:?}");
            assert!(want.rotation.angle_to(&got.rotation) < 1e-12);
        }
        // and it is not a no-op: the base actually moves the arm
        assert!((based.fk(&[0.0, 0.0]).translation.vector - plain.fk(&[0.0, 0.0]).translation.vector).norm() > 0.1);
    }

    /// Effort and velocity ride the row into the joint, and an unstated one stays unstated.
    #[test]
    fn effort_and_velocity_carry_into_the_joint_or_stay_none() {
        let r = Robot::from_dh(
            &[DhRow::revolute(0.0, 0.0, 0.4, 0.0).with_effort(87.0).with_max_velocity(2.175), DhRow::revolute(0.0, 0.0, 0.3, 0.0)],
            DhConvention::Standard,
            Iso::identity(),
        )
        .unwrap();
        assert_eq!(r.joints[0].effort, Some(87.0));
        assert_eq!(r.joints[0].max_velocity, Some(2.175));
        assert_eq!(r.joints[1].effort, None, "a source that states no effort must leave it None, not invent one");
        assert_eq!(r.joints[1].max_velocity, None);
        // a zero or NaN effort is "unstated", the URDF convention, not a limit of nothing
        let z = DhRow::revolute(0.0, 0.0, 0.4, 0.0).with_effort(0.0).with_max_velocity(f64::NAN);
        assert_eq!((z.effort, z.max_velocity), (None, None));
    }

    #[test]
    fn degenerate_tables_are_refused() {
        assert!(Robot::from_dh(&[], DhConvention::Standard, Iso::identity()).is_none(), "an empty table is not a robot");
        assert!(Robot::from_dh(&[DhRow::revolute(f64::NAN, 0.0, 0.1, 0.0)], DhConvention::Standard, Iso::identity()).is_none(), "a NaN row would build a robot that reports NaN poses without ever failing");
        let ok = Robot::from_dh(&[DhRow::revolute(0.0, 0.0, 0.1, 0.0).with_limits(-1.0, 1.0)], DhConvention::Standard, Iso::identity()).unwrap();
        assert_eq!(ok.joints[0].limits, Some((-1.0, 1.0)), "limits must carry through");
    }
}
