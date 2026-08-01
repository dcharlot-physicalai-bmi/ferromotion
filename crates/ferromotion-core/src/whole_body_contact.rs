//! **Whole-body multi-contact** — one hard, frictional, non-penetrating contact solve over an entire
//! free-floating kinematic tree, resolving *all* simultaneous contacts together (both feet, both hands,
//! self-contact) in a single interior-point step. This is the physics whole-body loco-manipulation
//! needs: a humanoid that walks while it reaches touches the ground and an object at once, and every
//! contact impulse must be consistent with every other through the shared body inertia.
//!
//! The step assembles the floating-base mass matrix `H` ([`tree_floating_mass_matrix`]) and the free
//! (contact-less) next velocity `v_free` from the tree ABA ([`tree_floating_forward_dynamics`]), maps
//! each contact point into generalized coordinates through its Jacobian `Jc` (so the normal row is
//! `n̂ᵀJc` and each friction facet is `t̂ᵀJc`), and feeds the whole set to the differentiable
//! Stewart-Trinkle interior-point solver ([`solve_frictional_ipm`]) — the same trusted core the serial
//! [`RobotContactSim`](crate::RobotContactSim) uses, now over the base+joints together. The result is a
//! post-contact generalized velocity, integrated on `SE(3)` for the base and in joint space for the
//! limbs. Unlike the penalty model in [`tree_floating_contact_step`](crate::tree_floating_contact_step)
//! this is hard non-penetration + a true friction cone. Pure `nalgebra` → WASM-clean.

use crate::{solve_frictional_ipm, tree_floating_forward_dynamics, tree_floating_mass_matrix, Joint, JointKind, LinkInertia, StFrictionContact};
use nalgebra::{DMatrix, DVector, Isometry3, Point3, Translation3, UnitQuaternion, Vector3, Vector6};

/// A contact point in the whole-body model: it rides at `offset` in the frame of tree body `body`
/// (`None` = the floating base itself, so a torso or belly can strike the ground), with friction
/// coefficient `mu`, and collides with the ground plane `z = floor_z`.
#[derive(Clone, Copy, Debug)]
pub struct WholeBodyContactPoint {
    pub body: Option<usize>,
    pub offset: Vector3<f64>,
    pub mu: f64,
}

impl WholeBodyContactPoint {
    /// A contact point on tree body `body`.
    pub fn on(body: usize, offset: Vector3<f64>, mu: f64) -> Self {
        WholeBodyContactPoint { body: Some(body), offset, mu }
    }
    /// A contact point on the floating base (torso).
    pub fn base(offset: Vector3<f64>, mu: f64) -> Self {
        WholeBodyContactPoint { body: None, offset, mu }
    }
}

/// World poses of every tree body, composed along the parent chain (topological order) from the base
/// pose. The companion to [`whole_body_contact_jacobian`], which takes these poses, and the way to
/// place barrier/clearance checks on any point of a floating-base robot.
pub fn whole_body_forward_kinematics(joints: &[Joint], parent: &[isize], base: Isometry3<f64>, q: &[f64]) -> Vec<Isometry3<f64>> {
    forward_kinematics(joints, parent, base, q)
}

/// Base→body world poses for every tree body, composed along the parent chain (topological order).
fn forward_kinematics(joints: &[Joint], parent: &[isize], base: Isometry3<f64>, q: &[f64]) -> Vec<Isometry3<f64>> {
    let n = joints.len();
    let mut w = vec![Isometry3::identity(); n];
    for i in 0..n {
        let local = joints[i].transform(q[i]);
        w[i] = if parent[i] < 0 { base * local } else { w[parent[i] as usize] * local };
    }
    w
}

/// The `3×(6+n)` contact Jacobian mapping the generalized velocity `[v₀ (base twist, base frame); q̇]`
/// to the **world** linear velocity of a point at `offset` on body `body`. `world[i]` are the body world
/// poses from [`forward_kinematics`], `base` the base world pose. Base columns come from the base twist
/// (`ṗ = v_origin + ω × r`), joint columns from the geometric Jacobian of the ancestor joints.
pub fn whole_body_contact_jacobian(joints: &[Joint], parent: &[isize], world: &[Isometry3<f64>], base: Isometry3<f64>, body: Option<usize>, offset: Vector3<f64>) -> DMatrix<f64> {
    let n = joints.len();
    let mut j = DMatrix::zeros(3, 6 + n);
    // contact point in world: on a tree body, or on the floating base itself
    let p_w = match body {
        Some(b) => (world[b] * Point3::from(offset)).coords,
        None => (base * Point3::from(offset)).coords,
    };
    let r_wb = base.rotation.to_rotation_matrix();
    let o_b = base.translation.vector; // base origin in world

    // base angular columns 0..3: ω = R_wb·e_k → (R_wb e_k) × (p_w − o_b)
    for k in 0..3 {
        let axis_w = r_wb * Vector3::ith(k, 1.0);
        let col = axis_w.cross(&(p_w - o_b));
        j.fixed_view_mut::<3, 1>(0, k).copy_from(&col);
    }
    // base linear columns 3..6: velocity of the base origin, R_wb·e_k
    for k in 0..3 {
        let col = r_wb * Vector3::ith(k, 1.0);
        j.fixed_view_mut::<3, 1>(0, 3 + k).copy_from(&col);
    }
    // joint columns: only ancestors of `body` (walk the parent chain, `body` included) contribute. A
    // base-attached point has no ancestors, so joint motion never moves it.
    let mut jj = match body {
        Some(b) => b as isize,
        None => -1,
    };
    while jj >= 0 {
        let idx = jj as usize;
        let w_j = world[idx];
        let axis_w = w_j.rotation.to_rotation_matrix() * joints[idx].axis.into_inner();
        let o_j = w_j.translation.vector;
        let col = match joints[idx].kind {
            JointKind::Revolute => axis_w.cross(&(p_w - o_j)),
            JointKind::Prismatic => axis_w,
        };
        j.fixed_view_mut::<3, 1>(0, 6 + idx).copy_from(&col);
        jj = parent[idx];
    }
    j
}

/// One whole-body hard-contact step. `base` is the base world pose, `v0` its spatial velocity in the
/// base frame (`[ω; v]`), `q`/`qd`/`tau` the joint state and torques. Every point in `contacts` collides
/// with the plane `z = floor_z`; all active contacts are resolved together by one interior-point solve
/// with central-path smoothing `kappa`. Returns `(base, v0, q, qd)` advanced by `dt`.
#[allow(clippy::too_many_arguments)]
pub fn whole_body_contact_step(
    joints: &[Joint],
    inertia: &[LinkInertia],
    parent: &[isize],
    base_inertia: &LinkInertia,
    base: Isometry3<f64>,
    v0: Vector6<f64>,
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    contacts: &[WholeBodyContactPoint],
    floor_z: f64,
    dt: f64,
    kappa: f64,
    gravity: Vector3<f64>,
) -> (Isometry3<f64>, Vector6<f64>, Vec<f64>, Vec<f64>) {
    let n = joints.len();
    let zero = vec![Vector6::zeros(); n];

    // free (contact-less) next velocity from the tree ABA
    let (a0, qdd) = tree_floating_forward_dynamics(joints, inertia, parent, base_inertia, v0, q, qd, tau, Vector6::zeros(), &zero, gravity);
    let mut v_free = DVector::zeros(6 + n);
    for r in 0..6 {
        v_free[r] = v0[r] + dt * a0[r];
    }
    for i in 0..n {
        v_free[6 + i] = qd[i] + dt * qdd[i];
    }

    // assemble the active contact set in generalized coordinates
    let world = forward_kinematics(joints, parent, base, q);
    let mut cset: Vec<StFrictionContact> = Vec::new();
    for c in contacts {
        let p_w = match c.body {
            Some(b) => (world[b] * Point3::from(c.offset)).coords,
            None => (base * Point3::from(c.offset)).coords,
        };
        let phi = p_w.z - floor_z; // signed gap to the floor (normal +z)
        let jc = whole_body_contact_jacobian(joints, parent, &world, base, c.body, c.offset);
        let row = |r: usize| DVector::from_iterator(6 + n, jc.row(r).iter().copied());
        cset.push(StFrictionContact {
            jn: row(2),                                  // world +z normal
            jt: vec![row(0), -row(0), row(1), -row(1)], // ±x, ±y friction pyramid
            phi,
            mu: c.mu,
        });
    }

    // resolve all contacts together (or coast if none)
    let v_next = if cset.is_empty() {
        v_free
    } else {
        let h = tree_floating_mass_matrix(joints, inertia, parent, base_inertia, q);
        solve_frictional_ipm(&h, &v_free, &cset, dt, kappa).v_next
    };

    // integrate: SE(3) for the base (body-frame twist), joint space for the limbs
    let v0n = Vector6::from_iterator(v_next.iter().take(6).copied());
    let w = v0n.fixed_rows::<3>(0).into_owned();
    let vlin = v0n.fixed_rows::<3>(3).into_owned();
    let step = Isometry3::from_parts(Translation3::from(dt * vlin), UnitQuaternion::from_scaled_axis(dt * w));
    let base_n = base * step;
    let mut qn = q.to_vec();
    let mut qdn = qd.to_vec();
    for i in 0..n {
        qdn[i] = v_next[6 + i];
        qn[i] += dt * qdn[i];
    }
    (base_n, v0n, qn, qdn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadruped;
    use nalgebra::Matrix3;

    fn quad_base() -> LinkInertia {
        LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) }
    }

    // The analytic contact Jacobian matches a finite-difference of the forward kinematics: perturb each
    // generalized coordinate (base twist and joints) and the contact point moves by the matching column.
    #[test]
    fn contact_jacobian_matches_finite_difference() {
        let (joints, _inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base = Isometry3::from_parts(Translation3::new(0.1, -0.05, 0.6), UnitQuaternion::from_euler_angles(0.05, -0.08, 0.12));
        let q: Vec<f64> = (0..n).map(|i| 0.2 * ((i as f64) * 0.7).sin()).collect();
        let (body, off, _mu) = contacts[0];
        let world = forward_kinematics(&joints, &parent, base, &q);
        let jc = whole_body_contact_jacobian(&joints, &parent, &world, base, Some(body), off);
        let p0 = (world[body] * Point3::from(off)).coords;
        let eps = 1e-6;
        let mut worst = 0.0f64;
        // base angular columns
        for k in 0..3 {
            let tw = UnitQuaternion::from_scaled_axis(Vector3::ith(k, eps));
            let bp = base * Isometry3::from_parts(Translation3::identity(), tw);
            let wp = forward_kinematics(&joints, &parent, bp, &q);
            let fd = ((wp[body] * Point3::from(off)).coords - p0) / eps;
            worst = worst.max((fd - jc.fixed_view::<3, 1>(0, k)).amax());
        }
        // base linear columns
        for k in 0..3 {
            let bp = base * Isometry3::from_parts(Translation3::from(Vector3::ith(k, eps)), UnitQuaternion::identity());
            let wp = forward_kinematics(&joints, &parent, bp, &q);
            let fd = ((wp[body] * Point3::from(off)).coords - p0) / eps;
            worst = worst.max((fd - jc.fixed_view::<3, 1>(0, 3 + k)).amax());
        }
        // joint columns
        for c in 0..n {
            let mut qp = q.clone();
            qp[c] += eps;
            let wp = forward_kinematics(&joints, &parent, base, &qp);
            let fd = ((wp[body] * Point3::from(off)).coords - p0) / eps;
            worst = worst.max((fd - jc.fixed_view::<3, 1>(0, 6 + c)).amax());
        }
        eprintln!("whole-body contact Jacobian vs finite difference: worst |Δ| {worst:.3e}");
        assert!(worst < 1e-5, "contact Jacobian disagrees with FK finite-difference: {worst}");
    }

    // A quadruped dropped onto the floor settles into a stable stance under HARD interior-point contact:
    // the feet do not sink through the floor (non-penetration, far tighter than a penalty spring), the
    // body stays upright, friction stops it sliding, and it comes to rest. The whole-body invariant.
    #[test]
    fn quadruped_settles_under_hard_contact() {
        let (joints, inertia, parent, foot_list) = quadruped();
        let n = joints.len();
        let contacts: Vec<WholeBodyContactPoint> = foot_list.iter().map(|&(body, offset, mu)| WholeBodyContactPoint::on(body, offset, mu)).collect();
        let base_inertia = quad_base();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor, dt, kappa) = (0.0, 2e-4, 1e-6);
        // legs straight down (q=0) reach 0.6 m; start above so the body DROPS and the hard contact
        // catches it on impact (feet begin 2 cm off the floor, ~0.6 m/s at touchdown)
        let mut base = Isometry3::translation(0.0, 0.0, 0.62);
        let mut v0 = Vector6::zeros();
        let mut q = vec![0.0; n];
        let mut qd = vec![0.0; n];
        let tau = vec![0.0; n];

        let mut worst_pen = 0.0f64;
        for _ in 0..4000 {
            let (b, v, qn, qdn) = whole_body_contact_step(&joints, &inertia, &parent, &base_inertia, base, v0, &q, &qd, &tau, &contacts, floor, dt, kappa, g);
            base = b;
            v0 = v;
            q = qn;
            qd = qdn;
            let world = forward_kinematics(&joints, &parent, base, &q);
            for c in &contacts {
                let z = match c.body { Some(b) => (world[b] * Point3::from(c.offset)).coords.z, None => (base * Point3::from(c.offset)).coords.z };
                worst_pen = worst_pen.min(z - floor);
            }
        }
        let up = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
        let base_speed = v0.norm();
        let horiz = base.translation.vector.xy().norm();
        eprintln!("whole-body drop: base z {:.4}, up {:.4}, worst foot penetration {:.4} mm, base speed {:.4}, horiz drift {:.4} m", base.translation.z, up, worst_pen * 1000.0, base_speed, horiz);
        assert!(base.translation.vector.iter().all(|v| v.is_finite()), "sim blew up");
        assert!(up > 0.95, "quadruped toppled: up {up}");
        assert!(base.translation.z > 0.55 && base.translation.z < 0.61, "did not settle near stance height: z {}", base.translation.z);
        assert!(worst_pen > -1.5e-3, "feet sank through the floor under hard contact: {worst_pen} m (penalty allows ~8 mm; hard should be sub-mm)");
        assert!(base_speed < 0.1, "did not settle to rest: {base_speed}");
        assert!(horiz < 0.02, "friction should hold it in place, drifted {horiz} m");
    }
}
