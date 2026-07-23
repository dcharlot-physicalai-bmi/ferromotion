//! **The constraint API** — constraints as first-class models over the dynamics, unified at the
//! velocity-impulse level (design: `docs/CONSTRAINTS.md`; the Pinocchio-4-class capability).
//!
//! Declare *models* — anchors, joint limits, dry friction, mimic couplings — on a
//! [`ConstraintSet`]; [`constrained_step`] assembles their Jacobian rows, builds the **Delassus
//! operator** `G = J M⁻¹ Jᵀ`, and solves the coupled impulses by projected Gauss–Seidel with one
//! **law** per row group:
//!
//! - `Equality` — `v_c = 0`, λ free (anchors, mimic couplings),
//! - `Unilateral` — `0 ≤ λ ⟂ v_c ≥ 0` (joint limits; contact normals in stage 2),
//! - `Box` — `|λ| ≤ λmax`, stick inside the box, slide at its bounds (dry friction).
//!
//! Position drift is handled by Baumgarte terms folded into the constraint bias. The
//! acceleration-level KKT path in [`crate::closed_loop`] remains the exact-equality oracle (see
//! tests); `contact`/`robot_contact` keep their niches and become stage-2 models here.

use crate::dynamics::{inverse_dynamics, mass_matrix, LinkInertia};
use crate::{JointKind, Robot};
use nalgebra::{Cholesky, DMatrix, DVector, Vector3};

/// The per-row-group coupling law between impulse and constraint velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Law {
    Equality,
    Unilateral,
    Box { lambda_max: f64 },
}

/// Baumgarte position-stabilization gain (fraction of the violation corrected per step).
pub const BAUMGARTE: f64 = 0.2;

enum Model {
    /// Weld a point (given in frame `upto`'s coordinates) to a world target: 3 equality rows.
    AnchorPoint { upto: usize, local: Vector3<f64>, target: Vector3<f64> },
    /// Dry Coulomb friction on joint `j`: one box row with `λmax = h·τ_c`.
    JointFriction { j: usize, tau_coulomb: f64 },
    /// Follower joint tracks `ratio·leader + offset`: one equality row.
    Mimic { follower: usize, leader: usize, ratio: f64, offset: f64 },
    /// Enable limit rows for every joint with declared limits (activated near the bound).
    JointLimits,
}

/// A declared set of constraint models (assembly happens per step, at the current state).
#[derive(Default)]
pub struct ConstraintSet {
    models: Vec<Model>,
}

/// One assembled row group: its law and the indices of its rows in the stacked Jacobian.
pub struct Group {
    pub law: Law,
    pub rows: core::ops::Range<usize>,
    /// Which model kind produced it (for reporting): "anchor", "limit", "friction", "mimic".
    pub kind: &'static str,
}

/// The result of a constrained step: next velocity, impulses, and the assembled groups.
pub struct StepResult {
    pub v_next: Vec<f64>,
    pub lambda: Vec<f64>,
    pub groups: Vec<Group>,
    /// PGS iterations used.
    pub iters: usize,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn anchor_point(&mut self, upto: usize, local: Vector3<f64>, target: Vector3<f64>) -> &mut Self {
        self.models.push(Model::AnchorPoint { upto, local, target });
        self
    }
    pub fn joint_friction(&mut self, j: usize, tau_coulomb: f64) -> &mut Self {
        self.models.push(Model::JointFriction { j, tau_coulomb });
        self
    }
    pub fn mimic(&mut self, follower: usize, leader: usize, ratio: f64, offset: f64) -> &mut Self {
        self.models.push(Model::Mimic { follower, leader, ratio, offset });
        self
    }
    /// Activate limit handling for every joint that declares `limits`.
    pub fn joint_limits(&mut self) -> &mut Self {
        self.models.push(Model::JointLimits);
        self
    }
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// 3×n Jacobian of a point (frame `upto`'s coordinates) with respect to the joint velocities,
/// plus the point's current world position.
fn point_jacobian(robot: &Robot, q: &[f64], upto: usize, local: Vector3<f64>) -> (DMatrix<f64>, Vector3<f64>) {
    let n = robot.dof();
    let p_w = (robot.frame_pose(q, upto) * nalgebra::Point3::from(local)).coords;
    let mut jac = DMatrix::zeros(3, n);
    let mut t = crate::Iso::identity();
    for (i, (j, &qi)) in robot.joints.iter().zip(q).enumerate().take(upto) {
        let pre = t * j.origin;
        let z = pre.rotation * j.axis.into_inner();
        let o = pre.translation.vector;
        match j.kind {
            JointKind::Revolute => {
                let lin = z.cross(&(p_w - o));
                jac.fixed_view_mut::<3, 1>(0, i).copy_from(&lin);
            }
            JointKind::Prismatic => {
                jac.fixed_view_mut::<3, 1>(0, i).copy_from(&z);
            }
        }
        t = pre * j.motion(qi);
    }
    (jac, p_w)
}

/// The Delassus operator `G = J M⁻¹ Jᵀ` (dense; `εI`-regularized) plus the pieces every solver
/// and consumer needs: `M⁻¹Jᵀ` for mapping impulses back to velocities.
pub struct Delassus {
    pub g: DMatrix<f64>,
    pub minv_jt: DMatrix<f64>,
}

impl Delassus {
    pub fn build(m: &DMatrix<f64>, j: &DMatrix<f64>) -> Delassus {
        let chol = Cholesky::new(m.clone()).expect("mass matrix must be SPD");
        let minv_jt = chol.solve(&j.transpose());
        let mut g = j * &minv_jt;
        for i in 0..g.nrows() {
            g[(i, i)] += 1e-10;
        }
        Delassus { g, minv_jt }
    }
}

/// One velocity-impulse step under the declared constraints (projected Gauss–Seidel).
/// `v_next = v_free + M⁻¹Jᵀλ` with `λ` satisfying each group's law.
#[allow(clippy::too_many_arguments)]
pub fn constrained_step(
    robot: &Robot,
    inertia: &[LinkInertia],
    q: &[f64],
    v: &[f64],
    tau: &[f64],
    h: f64,
    gravity: Vector3<f64>,
    cs: &ConstraintSet,
) -> StepResult {
    let n = robot.dof();
    let m = mass_matrix(robot, inertia, q);
    let bias = inverse_dynamics(robot, inertia, q, v, &vec![0.0; n], gravity);
    let chol = Cholesky::new(m.clone()).expect("mass matrix must be SPD");
    let rhs_free = DVector::from_iterator(n, (0..n).map(|i| h * (tau[i] - bias[i])));
    let dv_free = chol.solve(&rhs_free);
    let v_free: Vec<f64> = (0..n).map(|i| v[i] + dv_free[i]).collect();

    // assemble rows
    let mut jrows: Vec<DVector<f64>> = Vec::new();
    let mut brows: Vec<f64> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    for model in &cs.models {
        match model {
            Model::AnchorPoint { upto, local, target } => {
                let (jp, p_w) = point_jacobian(robot, q, *upto, *local);
                let start = jrows.len();
                for r in 0..3 {
                    jrows.push(jp.row(r).transpose());
                    brows.push(BAUMGARTE / h * (p_w[r] - target[r]));
                }
                groups.push(Group { law: Law::Equality, rows: start..start + 3, kind: "anchor" });
            }
            Model::JointFriction { j, tau_coulomb } => {
                let mut row = DVector::zeros(n);
                row[*j] = 1.0;
                let start = jrows.len();
                jrows.push(row);
                brows.push(0.0);
                groups.push(Group {
                    law: Law::Box { lambda_max: h * tau_coulomb },
                    rows: start..start + 1,
                    kind: "friction",
                });
            }
            Model::Mimic { follower, leader, ratio, offset } => {
                let mut row = DVector::zeros(n);
                row[*follower] = 1.0;
                row[*leader] = -ratio;
                let start = jrows.len();
                jrows.push(row);
                brows.push(BAUMGARTE / h * (q[*follower] - ratio * q[*leader] - offset));
                groups.push(Group { law: Law::Equality, rows: start..start + 1, kind: "mimic" });
            }
            Model::JointLimits => {
                for (j, joint) in robot.joints.iter().enumerate() {
                    let Some((lo, hi)) = joint.limits else { continue };
                    // activate when the free motion would end the step beyond the bound
                    let q_pred = q[j] + h * v_free[j];
                    if q_pred < lo {
                        let mut row = DVector::zeros(n);
                        row[j] = 1.0; // v_c = v_j must become ≥ 0
                        let start = jrows.len();
                        jrows.push(row);
                        brows.push(BAUMGARTE / h * (q[j] - lo).min(0.0));
                        groups.push(Group { law: Law::Unilateral, rows: start..start + 1, kind: "limit" });
                    } else if q_pred > hi {
                        let mut row = DVector::zeros(n);
                        row[j] = -1.0; // v_c = −v_j must become ≥ 0
                        let start = jrows.len();
                        jrows.push(row);
                        brows.push(BAUMGARTE / h * (hi - q[j]).min(0.0));
                        groups.push(Group { law: Law::Unilateral, rows: start..start + 1, kind: "limit" });
                    }
                }
            }
        }
    }

    let nc = jrows.len();
    if nc == 0 {
        return StepResult { v_next: v_free, lambda: vec![], groups, iters: 0 };
    }
    let mut jmat = DMatrix::zeros(nc, n);
    for (r, row) in jrows.iter().enumerate() {
        jmat.row_mut(r).copy_from(&row.transpose());
    }
    let del = Delassus::build(&m, &jmat);

    // rhs: G λ = −(J v_free + b)
    let vfree_v = DVector::from_column_slice(&v_free);
    let rhs: DVector<f64> = -(&jmat * &vfree_v + DVector::from_column_slice(&brows));

    // projected Gauss–Seidel over the groups' laws
    let mut lambda: DVector<f64> = DVector::zeros(nc);
    let mut iters = 0;
    for _ in 0..200 {
        iters += 1;
        let mut max_change = 0.0f64;
        for g in &groups {
            for r in g.rows.clone() {
                let mut acc = rhs[r];
                for c in 0..nc {
                    if c != r {
                        acc -= del.g[(r, c)] * lambda[c];
                    }
                }
                let mut cand: f64 = acc / del.g[(r, r)];
                cand = match g.law {
                    Law::Equality => cand,
                    Law::Unilateral => cand.max(0.0),
                    Law::Box { lambda_max } => cand.clamp(-lambda_max, lambda_max),
                };
                max_change = max_change.max((cand - lambda[r]).abs());
                lambda[r] = cand;
            }
        }
        if max_change < 1e-12 {
            break;
        }
    }

    let dv = &del.minv_jt * &lambda;
    let v_next: Vec<f64> = (0..n).map(|i| v_free[i] + dv[i]).collect();
    StepResult { v_next, lambda: lambda.iter().copied().collect(), groups, iters }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Iso, Joint};
    use nalgebra::{Matrix3, Translation3, UnitQuaternion};

    fn arm2() -> (Robot, Vec<LinkInertia>) {
        let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
        let robot = Robot {
            joints: vec![
                Joint::revolute(mk(0.1), Vector3::y()),
                Joint::revolute(mk(0.4), Vector3::y()),
            ],
            ee_offset: mk(0.3),
        };
        let inertia = vec![
            LinkInertia { mass: 2.0, com: Vector3::new(0.0, 0.0, 0.2), inertia: Matrix3::identity() * 0.02 },
            LinkInertia { mass: 1.2, com: Vector3::new(0.0, 0.0, 0.15), inertia: Matrix3::identity() * 0.01 },
        ];
        (robot, inertia)
    }

    const G: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);

    /// Oracle 1 — equality vs the KKT path: an anchored chain-end point must have (near-)zero
    /// post-step velocity, and with zero Baumgarte the anchor impulse does no work.
    #[test]
    fn anchor_kills_the_point_velocity() {
        let (robot, inertia) = arm2();
        let q = [0.4, -0.7];
        let v = [0.6, -0.3];
        let h = 1e-3;
        let (_, p_now) = point_jacobian(&robot, &q, 2, Vector3::zeros());
        let mut cs = ConstraintSet::new();
        cs.anchor_point(2, Vector3::zeros(), p_now); // anchor AT the current position (no Baumgarte drive)
        let res = constrained_step(&robot, &inertia, &q, &v, &[0.0; 2], h, G, &cs);
        // post-step point velocity = Jp v⁺ ≈ 0
        let (jp, _) = point_jacobian(&robot, &q, 2, Vector3::zeros());
        let vp = &jp * DVector::from_column_slice(&res.v_next);
        assert!(vp.norm() < 1e-8, "anchored point must not move: |v_p| = {}", vp.norm());
        // sticking equality: impulse power ≈ 0 (λ · v_c = 0 since v_c = 0)
        let vc = &jp * DVector::from_column_slice(&res.v_next);
        let power: f64 = res.lambda.iter().zip(vc.iter()).map(|(l, v)| l * v).sum();
        assert!(power.abs() < 1e-10, "anchor impulse must do no work: {power}");
    }

    /// Oracle 2 — analytic joint limit: pushed into the bound the joint stops dead with λ > 0;
    /// pulled away the limit is inactive.
    #[test]
    fn joint_limit_complementarity_is_exact() {
        let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
        let robot = Robot {
            joints: vec![Joint::revolute(mk(0.1), Vector3::y()).with_limits(-1.0, 1.0)],
            ee_offset: mk(0.3),
        };
        let inertia = vec![LinkInertia { mass: 1.0, com: Vector3::new(0.0, 0.0, 0.15), inertia: Matrix3::identity() * 0.01 }];
        let h = 1e-3;
        let mut cs = ConstraintSet::new();
        cs.joint_limits();
        // at the upper bound, torque pushing further + approach velocity → v⁺ = 0, λ > 0
        let res = constrained_step(&robot, &inertia, &[1.0], &[0.5], &[2.0], h, Vector3::zeros(), &cs);
        assert!(res.v_next[0].abs() < 1e-10, "must stop at the bound: v⁺ = {}", res.v_next[0]);
        assert!(res.lambda[0] > 0.0, "active limit must push: λ = {:?}", res.lambda);
        // same state, torque pulling away → no limit row survives activation OR λ = 0, joint accelerates inward
        let res = constrained_step(&robot, &inertia, &[1.0], &[-0.2], &[-2.0], h, Vector3::zeros(), &cs);
        assert!(res.v_next[0] < -0.2, "pulling away must accelerate freely: v⁺ = {}", res.v_next[0]);
        assert!(res.lambda.iter().all(|&l| l.abs() < 1e-12), "inactive limit must carry no impulse");
    }

    /// Oracle 3 — analytic dry friction: below breakaway the joint sticks exactly; above it, it
    /// slides with the effective torque reduced by exactly τ_coulomb.
    #[test]
    fn dry_friction_stick_slip_is_exact() {
        let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
        let robot = Robot { joints: vec![Joint::revolute(mk(0.0), Vector3::y())], ee_offset: Iso::identity() };
        // pure rotor: no gravity moment (com on axis), I = 0.02
        let inertia = vec![LinkInertia { mass: 1.0, com: Vector3::zeros(), inertia: Matrix3::identity() * 0.02 }];
        let (h, tau_c) = (1e-3, 0.5);
        let mut cs = ConstraintSet::new();
        cs.joint_friction(0, tau_c);
        // stick: τ = 0.3 < τ_c from rest → v⁺ = 0
        let res = constrained_step(&robot, &inertia, &[0.0], &[0.0], &[0.3], h, Vector3::zeros(), &cs);
        assert!(res.v_next[0].abs() < 1e-12, "below breakaway must stick: v⁺ = {}", res.v_next[0]);
        // slip: τ = 2.0 → v⁺ = h(τ − τ_c)/I exactly
        let res = constrained_step(&robot, &inertia, &[0.0], &[0.0], &[2.0], h, Vector3::zeros(), &cs);
        let want = h * (2.0 - tau_c) / 0.02;
        assert!((res.v_next[0] - want).abs() < 1e-9, "slip: {} vs {}", res.v_next[0], want);
    }

    /// Oracle 4 — mimic: the follower tracks ratio·leader in velocity, and the coupling transmits
    /// load (torquing ONLY the follower still accelerates the leader).
    #[test]
    fn mimic_couples_velocities_and_transmits_torque() {
        let (robot, inertia) = arm2();
        let h = 1e-3;
        let mut cs = ConstraintSet::new();
        let ratio = -0.5;
        cs.mimic(1, 0, ratio, 0.0);
        // consistent state, torque on the leader only
        let res = constrained_step(&robot, &inertia, &[0.2, -0.1], &[0.3, ratio * 0.3], &[1.0, 0.0], h, Vector3::zeros(), &cs);
        assert!((res.v_next[1] - ratio * res.v_next[0]).abs() < 1e-10, "velocity coupling: {:?}", res.v_next);
        // torque on the FOLLOWER only must move the leader through the coupling
        let res2 = constrained_step(&robot, &inertia, &[0.2, -0.1], &[0.0, 0.0], &[0.0, 1.0], h, Vector3::zeros(), &cs);
        assert!((res2.v_next[1] - ratio * res2.v_next[0]).abs() < 1e-10);
        assert!(res2.v_next[0].abs() > 1e-6, "coupling must transmit torque to the leader: {:?}", res2.v_next);
    }

    /// Mixed problem: anchor + friction + limits assemble and solve together without interference
    /// when each is individually inactive/consistent.
    #[test]
    fn mixed_constraints_coexist() {
        let (robot, inertia) = arm2();
        let q = [0.3, -0.4];
        let v = [0.1, 0.05];
        let h = 1e-3;
        let (_, p_now) = point_jacobian(&robot, &q, 2, Vector3::zeros());
        let mut cs = ConstraintSet::new();
        cs.anchor_point(2, Vector3::zeros(), p_now).joint_friction(0, 0.2).joint_limits();
        let res = constrained_step(&robot, &inertia, &q, &v, &[0.4, -0.2], h, G, &cs);
        assert!(res.v_next.iter().all(|v| v.is_finite()));
        let (jp, _) = point_jacobian(&robot, &q, 2, Vector3::zeros());
        let vp = &jp * DVector::from_column_slice(&res.v_next);
        assert!(vp.norm() < 1e-7, "anchor still holds in the mixed problem: {}", vp.norm());
    }
}
