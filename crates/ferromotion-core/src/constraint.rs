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
    /// Coulomb friction cone: rows are `[normal, tangent₁, tangent₂]`, `‖λ_t‖ ≤ μ·λ_n`, `λ_n ≥ 0`.
    Cone { mu: f64 },
}

/// Which impulse solver runs the projections (both consume the same laws and Delassus).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Solver {
    /// Projected Gauss–Seidel — the robust sequential workhorse (cone via normal-then-disk).
    #[default]
    Pgs,
    /// ADMM with a factored `(G + ρI)` and exact set projections (second-order cone for friction)
    /// — the Pinocchio-4 / SAP-family choice for stiff, coupled problems.
    Admm,
}

/// Baumgarte position-stabilization gain (fraction of the violation corrected per step).
pub const BAUMGARTE: f64 = 0.2;

#[derive(Debug)]
enum Model {
    /// Weld a point (given in frame `upto`'s coordinates) to a world target: 3 equality rows.
    AnchorPoint { upto: usize, local: Vector3<f64>, target: Vector3<f64> },
    /// Dry Coulomb friction on joint `j`: one box row with `λmax = h·τ_c`.
    JointFriction { j: usize, tau_coulomb: f64 },
    /// Follower joint tracks `ratio·leader + offset`: one equality row.
    Mimic { follower: usize, leader: usize, ratio: f64, offset: f64 },
    /// Enable limit rows for every joint with declared limits (activated near the bound).
    JointLimits,
    /// Point-vs-plane frictional contact: a point of frame `upto` against the plane through
    /// `plane_p` with unit `normal`; Coulomb coefficient `mu`. Activated when the free motion
    /// would penetrate. 3 cone rows.
    PointContact { upto: usize, local: Vector3<f64>, plane_p: Vector3<f64>, normal: Vector3<f64>, mu: f64 },
    /// Collision-driven contacts: every declared link sphere against a world plane. Each sphere
    /// whose free motion would touch emits one cone group (sphere contact ≡ point contact at the
    /// center against the plane shifted out by the radius).
    SpheresVsPlane { plane_p: Vector3<f64>, normal: Vector3<f64>, mu: f64 },
}

/// A collision sphere decorating the kinematic chain (the cuRobo-style link-sphere model the
/// `collision` costs also use): center `offset` in frame `upto`'s coordinates, radius `radius`.
#[derive(Clone, Copy, Debug)]
pub struct LinkSphere {
    pub upto: usize,
    pub offset: Vector3<f64>,
    pub radius: f64,
}

/// A declared set of constraint models (assembly happens per step, at the current state).
#[derive(Default, Debug)]
pub struct ConstraintSet {
    models: Vec<Model>,
    spheres: Vec<LinkSphere>,
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
    /// Frictional point-vs-plane contact (see [`Law::Cone`]); `normal` need not be unit.
    pub fn point_contact(&mut self, upto: usize, local: Vector3<f64>, plane_p: Vector3<f64>, normal: Vector3<f64>, mu: f64) -> &mut Self {
        self.models.push(Model::PointContact { upto, local, plane_p, normal: normal.normalize(), mu });
        self
    }
    /// Declare a collision sphere on the chain (used by [`Self::spheres_vs_plane`]).
    pub fn link_sphere(&mut self, upto: usize, offset: Vector3<f64>, radius: f64) -> &mut Self {
        self.spheres.push(LinkSphere { upto, offset, radius });
        self
    }
    /// Collision-driven contact generation: every declared link sphere against a world plane —
    /// touching spheres emit frictional cone contacts automatically, step by step.
    pub fn spheres_vs_plane(&mut self, plane_p: Vector3<f64>, normal: Vector3<f64>, mu: f64) -> &mut Self {
        self.models.push(Model::SpheresVsPlane { plane_p, normal: normal.normalize(), mu });
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

/// Emit one frictional plane-contact cone group for a chain point, if its free motion would touch.
#[allow(clippy::too_many_arguments)]
fn emit_plane_contact(
    robot: &Robot,
    q: &[f64],
    v_free: &[f64],
    h: f64,
    upto: usize,
    local: Vector3<f64>,
    plane_p: Vector3<f64>,
    normal: Vector3<f64>,
    mu: f64,
    jrows: &mut Vec<DVector<f64>>,
    brows: &mut Vec<f64>,
    groups: &mut Vec<Group>,
) {
    let (jp, p_w) = point_jacobian(robot, q, upto, local);
    let gap = normal.dot(&(p_w - plane_p));
    let vfree_vec = DVector::from_column_slice(v_free);
    let vn_free = normal.transpose() * (&jp * &vfree_vec);
    if gap + h * vn_free[0] > 0.0 {
        return; // free motion separates — inactive this step
    }
    let t1 = if normal.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    let t1 = (t1 - normal * normal.dot(&t1)).normalize();
    let t2 = normal.cross(&t1);
    let start = jrows.len();
    for dir in [&normal, &t1, &t2] {
        jrows.push((dir.transpose() * &jp).transpose());
    }
    brows.push(BAUMGARTE / h * gap.min(0.0));
    brows.push(0.0);
    brows.push(0.0);
    groups.push(Group { law: Law::Cone { mu }, rows: start..start + 3, kind: "contact" });
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
    constrained_step_with(robot, inertia, q, v, tau, h, gravity, cs, Solver::Pgs)
}

/// [`constrained_step`] with an explicit [`Solver`] choice.
#[allow(clippy::too_many_arguments)]
pub fn constrained_step_with(
    robot: &Robot,
    inertia: &[LinkInertia],
    q: &[f64],
    v: &[f64],
    tau: &[f64],
    h: f64,
    gravity: Vector3<f64>,
    cs: &ConstraintSet,
    solver: Solver,
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
            Model::PointContact { upto, local, plane_p, normal, mu } => {
                emit_plane_contact(robot, q, &v_free, h, *upto, *local, *plane_p, *normal, *mu, &mut jrows, &mut brows, &mut groups);
            }
            Model::SpheresVsPlane { plane_p, normal, mu } => {
                for sp in &cs.spheres {
                    // sphere-vs-plane ≡ its center vs the plane shifted out by the radius
                    let shifted = plane_p + normal * sp.radius;
                    emit_plane_contact(robot, q, &v_free, h, sp.upto, sp.offset, shifted, *normal, *mu, &mut jrows, &mut brows, &mut groups);
                }
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

    let (lambda, iters) = match solver {
        Solver::Pgs => solve_pgs(&del, &rhs, &groups),
        Solver::Admm => solve_admm(&del, &rhs, &groups),
    };
    let dv = &del.minv_jt * &lambda;
    let v_next: Vec<f64> = (0..n).map(|i| v_free[i] + dv[i]).collect();
    StepResult { v_next, lambda: lambda.iter().copied().collect(), groups, iters }
}

/// Projected Gauss–Seidel over the groups' laws.
fn solve_pgs(del: &Delassus, rhs: &DVector<f64>, groups: &[Group]) -> (DVector<f64>, usize) {
    let nc = rhs.len();
    let mut lambda: DVector<f64> = DVector::zeros(nc);
    let mut iters = 0;
    for _ in 0..200 {
        iters += 1;
        let mut max_change = 0.0f64;
        for g in groups {
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
                    // cone rows: normal row is unilateral; tangent rows update freely here and are
                    // projected onto the friction disk of radius μ·λ_n after the row sweep below
                    Law::Cone { .. } => {
                        if r == g.rows.start {
                            cand.max(0.0)
                        } else {
                            cand
                        }
                    }
                };
                max_change = max_change.max((cand - lambda[r]).abs());
                lambda[r] = cand;
            }
            if let Law::Cone { mu } = g.law {
                let (n0, t1, t2) = (g.rows.start, g.rows.start + 1, g.rows.start + 2);
                let cap = mu * lambda[n0];
                let tn = (lambda[t1] * lambda[t1] + lambda[t2] * lambda[t2]).sqrt();
                if tn > cap {
                    let sc = if tn > 0.0 { cap / tn } else { 0.0 };
                    lambda[t1] *= sc;
                    lambda[t2] *= sc;
                    max_change = max_change.max(tn - cap);
                }
            }
        }
        if max_change < 1e-12 {
            break;
        }
    }
    (lambda, iters)
}

/// ADMM over the same laws: factor `(G + ρI)` once, then iterate
/// `λ ← (G+ρI)⁻¹(rhs + ρ(z − u)); z ← Π_K(λ + u); u ← u + λ − z` with exact set projections.
fn solve_admm(del: &Delassus, rhs: &DVector<f64>, groups: &[Group]) -> (DVector<f64>, usize) {
    let nc = rhs.len();
    let rho = (0..nc).map(|i| del.g[(i, i)]).sum::<f64>() / nc as f64;
    let mut greg = del.g.clone();
    for i in 0..nc {
        greg[(i, i)] += rho;
    }
    let chol = Cholesky::new(greg).expect("G + ρI is SPD");
    let mut z: DVector<f64> = DVector::zeros(nc);
    let mut u: DVector<f64> = DVector::zeros(nc);
    let mut lambda: DVector<f64> = DVector::zeros(nc);
    let mut iters = 0;
    for _ in 0..400 {
        iters += 1;
        lambda = chol.solve(&(rhs + (&z - &u) * rho));
        let mut z_new = &lambda + &u;
        project_laws(&mut z_new, groups);
        let r_prim = (&lambda - &z_new).norm();
        let r_dual = (&z_new - &z).norm() * rho; // dual residual — primal alone stalls at
        u += &lambda - &z_new; // unconverged fixed points whenever the projection is inactive
        z = z_new;
        if r_prim < 1e-12 && r_dual < 1e-12 {
            break;
        }
    }
    (z, iters)
}

/// Exact projection of a candidate impulse vector onto every group's admissible set.
fn project_laws(x: &mut DVector<f64>, groups: &[Group]) {
    for g in groups {
        match g.law {
            Law::Equality => {}
            Law::Unilateral => {
                for r in g.rows.clone() {
                    x[r] = x[r].max(0.0);
                }
            }
            Law::Box { lambda_max } => {
                for r in g.rows.clone() {
                    x[r] = x[r].clamp(-lambda_max, lambda_max);
                }
            }
            Law::Cone { mu } => {
                // CP-consistent projection: normal first (unilateral), then tangents capped at
                // μ·λn — the maximum-dissipation complementarity PGS also targets. The exact
                // SOC projection would make ADMM solve the plain convex cone-QP instead, whose
                // known artifact is pulling EXTRA normal force during sliding (observed here as
                // the Coulomb block braking harder than μ·m·g); both solvers must agree on the
                // physical CP, so both use the sequential projection.
                let (n0, t1, t2) = (g.rows.start, g.rows.start + 1, g.rows.start + 2);
                x[n0] = x[n0].max(0.0);
                let cap = mu * x[n0];
                let tn = (x[t1] * x[t1] + x[t2] * x[t2]).sqrt();
                if tn > cap {
                    let sc = if tn > 0.0 { cap / tn } else { 0.0 };
                    x[t1] *= sc;
                    x[t2] *= sc;
                }
            }
        }
    }
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

    /// Oracle — the classic Coulomb block, fully analytic: a block on the ground pushed
    /// horizontally. Below μ·m·g it sticks (both axes dead); above, it slides with the friction
    /// force exactly μ·m·g, and the tangential impulse sits ON the cone opposing the slip.
    #[test]
    fn coulomb_block_stick_slip_both_solvers() {
        let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
        // block: vertical slide (z) then horizontal slide (x); all mass on the moving block
        let robot = Robot {
            joints: vec![Joint::prismatic(mk(0.0), Vector3::z()), Joint::prismatic(mk(0.0), Vector3::x())],
            ee_offset: Iso::identity(),
        };
        let m_blk = 2.0;
        let inertia = vec![
            LinkInertia::zero(),
            LinkInertia { mass: m_blk, com: Vector3::zeros(), inertia: Matrix3::identity() * 1e-6 },
        ];
        let (h, mu, g) = (1e-3, 0.4, 9.81);
        let mut cs = ConstraintSet::new();
        cs.point_contact(2, Vector3::zeros(), Vector3::zeros(), Vector3::z(), mu);
        for solver in [Solver::Pgs, Solver::Admm] {
            // stick: F = 0.5·μ·m·g
            let f = 0.5 * mu * m_blk * g;
            let res = constrained_step_with(&robot, &inertia, &[0.0, 0.0], &[0.0, 0.0], &[0.0, f], h, G, &cs, solver);
            assert!(res.v_next[0].abs() < 1e-9 && res.v_next[1].abs() < 1e-9, "{solver:?} must stick: {:?}", res.v_next);
            // normal impulse supports the weight exactly
            assert!((res.lambda[0] - h * m_blk * g).abs() < 1e-9, "{solver:?} N: {}", res.lambda[0]);
            // slide: F = 2·μ·m·g → v_x⁺ = h(F − μmg)/m
            let f = 2.0 * mu * m_blk * g;
            let res = constrained_step_with(&robot, &inertia, &[0.0, 0.0], &[0.0, 0.0], &[0.0, f], h, G, &cs, solver);
            let want = h * (f - mu * m_blk * g) / m_blk;
            assert!((res.v_next[1] - want).abs() < 1e-8, "{solver:?} slide: {} vs {want}", res.v_next[1]);
            assert!(res.v_next[0].abs() < 1e-9, "{solver:?} stays on the ground");
            // the tangential impulse sits ON the cone, opposing the push
            let tn = (res.lambda[1] * res.lambda[1] + res.lambda[2] * res.lambda[2]).sqrt();
            assert!((tn - mu * res.lambda[0]).abs() < 1e-9, "{solver:?} on-cone: |λt| {tn} vs μλn {}", mu * res.lambda[0]);
        }
    }

    /// PGS and ADMM must agree on mixed problems (anchor + friction + limit + contact together).
    #[test]
    fn pgs_and_admm_agree_on_mixed_problems() {
        let (robot, inertia) = arm2();
        let q = [0.5, -0.9];
        let v = [0.4, -0.6];
        let h = 1e-3;
        let (_, p_now) = point_jacobian(&robot, &q, 1, Vector3::zeros());
        let mut cs = ConstraintSet::new();
        cs.anchor_point(1, Vector3::zeros(), p_now)
            .joint_friction(1, 0.3)
            .point_contact(2, Vector3::zeros(), Vector3::new(0.0, 0.0, -2.0), Vector3::z(), 0.5);
        let a = constrained_step_with(&robot, &inertia, &q, &v, &[0.7, -0.4], h, G, &cs, Solver::Pgs);
        let b = constrained_step_with(&robot, &inertia, &q, &v, &[0.7, -0.4], h, G, &cs, Solver::Admm);
        for (x, y) in a.v_next.iter().zip(&b.v_next) {
            assert!((x - y).abs() < 1e-6, "solver disagreement: {:?} vs {:?}", a.v_next, b.v_next);
        }
    }

    /// Oracle 7 — the four-bar: a parallelogram linkage as a 3-joint spanning tree + a cut-joint
    /// anchor, against the acceleration-level KKT path (`closed_loop::PlanarLoop`). The two
    /// formulations differ by the J̇q̇ term inside the step, so the honest assertion is FIRST-ORDER
    /// agreement that tightens linearly as h shrinks.
    #[test]
    fn four_bar_cut_joint_matches_the_kkt_path_to_first_order() {
        use crate::closed_loop::{Pin, PlanarLoop};
        let mk = |x: f64| Iso::from_parts(Translation3::new(x, 0.0, 0.0), UnitQuaternion::identity());
        // parallelogram in the x-y plane: L1 = L3 = 1, coupler L2 = ground d = 1.5, joints about z
        let robot = Robot {
            joints: vec![
                Joint::revolute(mk(0.0), Vector3::z()),
                Joint::revolute(mk(1.0), Vector3::z()),
                Joint::revolute(mk(1.5), Vector3::z()),
            ],
            ee_offset: mk(1.0),
        };
        let inertia = vec![
            LinkInertia { mass: 1.0, com: Vector3::new(0.5, 0.0, 0.0), inertia: Matrix3::identity() * 0.02 },
            LinkInertia { mass: 1.5, com: Vector3::new(0.75, 0.0, 0.0), inertia: Matrix3::identity() * 0.04 },
            LinkInertia { mass: 1.0, com: Vector3::new(0.5, 0.0, 0.0), inertia: Matrix3::identity() * 0.02 },
        ];
        // consistent parallelogram state: q = (θ, −θ, θ+π), q̇ = (w, −w, w)
        let (th, w) = (0.6, 0.8);
        let q = [th, -th, th + std::f64::consts::PI];
        let v = [w, -w, w];
        let tau = [0.4, -0.2, 0.1];
        let grav = Vector3::new(0.0, -9.81, 0.0);
        let closure_target = Vector3::new(1.5, 0.0, 0.0);

        // KKT reference (no Baumgarte: the state is exactly consistent)
        let kkt = PlanarLoop {
            robot: &robot,
            inertia: &inertia,
            pins: vec![Pin { frame: 3, offset: Vector3::new(1.0, 0.0, 0.0), target: [1.5, 0.0] }],
            omega: 0.0,
        };
        let mut prev_err = f64::INFINITY;
        for h in [1e-3, 1e-4] {
            let (qdd, _lam) = kkt.forward_dynamics(&q, &v, &tau, grav);
            let v_kkt: Vec<f64> = (0..3).map(|i| v[i] + h * qdd[i]).collect();
            let mut cs = ConstraintSet::new();
            cs.anchor_point(3, Vector3::new(1.0, 0.0, 0.0), closure_target);
            let res = constrained_step(&robot, &inertia, &q, &v, &tau, h, grav, &cs);
            let err = res
                .v_next
                .iter()
                .zip(&v_kkt)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(err < 5.0 * h, "h={h}: |Δv| = {err} must be O(h)");
            assert!(err < prev_err * 0.5, "error must shrink with h: {err} vs {prev_err}");
            prev_err = err;
        }
    }

    /// Collision-driven contacts: a "dumbbell" (two spheres on one link) resting on the ground —
    /// both contacts activate, together they support the weight exactly, and a sphere lifted off
    /// the plane emits nothing.
    #[test]
    fn sphere_contacts_generate_and_support_the_weight() {
        let mk = |z: f64| Iso::from_parts(Translation3::new(0.0, 0.0, z), UnitQuaternion::identity());
        // one vertical slide joint carrying a body with two ground spheres + one lifted sphere
        let robot = Robot { joints: vec![Joint::prismatic(mk(0.0), Vector3::z())], ee_offset: Iso::identity() };
        let m_body = 3.0;
        let inertia = vec![LinkInertia { mass: m_body, com: Vector3::zeros(), inertia: Matrix3::identity() * 0.01 }];
        let (h, r) = (1e-3, 0.05);
        let mut cs = ConstraintSet::new();
        cs.link_sphere(1, Vector3::new(0.2, 0.0, r), r)   // touching (center at height r)
            .link_sphere(1, Vector3::new(-0.2, 0.0, r), r) // touching
            .link_sphere(1, Vector3::new(0.0, 0.0, 0.8), r) // well above the ground
            .spheres_vs_plane(Vector3::zeros(), Vector3::z(), 0.6);
        let res = constrained_step(&robot, &inertia, &[0.0], &[0.0], &[0.0], h, G, &cs);
        // exactly two contact groups assembled
        let contacts = res.groups.iter().filter(|g| g.kind == "contact").count();
        assert_eq!(contacts, 2, "two touching spheres, one lifted: {contacts} contacts");
        // the body rests: v⁺ = 0, and the normal impulses sum to the weight impulse
        assert!(res.v_next[0].abs() < 1e-10, "resting: v⁺ = {}", res.v_next[0]);
        let n_sum: f64 = res
            .groups
            .iter()
            .filter(|g| g.kind == "contact")
            .map(|g| res.lambda[g.rows.start])
            .sum();
        assert!((n_sum - h * m_body * 9.81).abs() < 1e-9, "N sum {} vs weight impulse {}", n_sum, h * m_body * 9.81);
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
