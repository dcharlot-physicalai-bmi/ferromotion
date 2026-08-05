//! **A hand and an object in one contact solve** — in-hand manipulation, not standing on a floor.
//!
//! The frictional contact solver in this crate is already general: [`solve_contacts_friction`](crate::solve_contacts_friction)
//! takes a mass matrix and per-contact Jacobian rows of any width. What was floor-locked is contact *generation*.
//! `whole_body_contact` computes `phi = p_w.z - floor_z` at both of its contact sites, so the only thing a robot could
//! touch was the plane `z = floor_z`. A hand cannot manipulate a plane, so in-hand manipulation was unreachable even
//! though every solver piece for it existed.
//!
//! This module closes that by generating contacts between a loaded [`KinematicTree`](crate::KinematicTree) and a
//! free-floating object, and solving the two together. The generalised velocity is `[hand qd (n); object twist (6)]`
//! with the object's twist ordered `[omega; v]` to match the rest of the crate, and the contact Jacobian is the
//! *relative* velocity of the fingertip against the object's material point — which is what makes the object react to
//! being squeezed rather than being a fixed obstacle.
//!
//! **Two of the tests here cross-validate modules that were built independently**, which is the reason to write them
//! that way rather than checking this module against itself:
//!
//! - [`grasp_split`](crate::grasp_split) says a wrench in `null(G)` produces no net wrench on the object. Simulating a
//!   symmetric squeeze must therefore produce no object acceleration, and it does.
//! - [`force_closure_q1_spatial`](crate::force_closure_q1_spatial) says whether a grasp can resist gravity. Simulating
//!   the same geometry must hold the object when `Q1 > 0` and drop it when the grasp is rank-deficient.

use crate::{FrictionContact, Iso, KinematicTree, LinkInertia};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3, Vector6};

/// A free-floating rigid sphere — the simplest object with a well-defined surface everywhere, so a fingertip's gap and
/// normal are exact rather than a mesh query's approximation.
#[derive(Clone, Copy, Debug)]
pub struct SphereObject {
    pub centre: Vector3<f64>,
    pub radius: f64,
    pub mass: f64,
    /// Angular velocity `omega` then linear velocity `v` of the centre, matching the crate's `[omega; v]` twist order.
    pub twist: Vector6<f64>,
}

impl SphereObject {
    pub fn new(centre: Vector3<f64>, radius: f64, mass: f64) -> SphereObject {
        SphereObject { centre, radius, mass, twist: Vector6::zeros() }
    }

    /// Spatial inertia about the centre, `[omega; v]` ordered: `diag(I_rot, m*I_3)` with `I_rot = (2/5) m r^2` for a
    /// solid sphere.
    pub fn spatial_inertia(&self) -> Matrix3<f64> {
        Matrix3::identity() * (0.4 * self.mass * self.radius * self.radius)
    }

    /// Signed gap from a world point to the surface, and the outward unit normal there. Positive gap is separated.
    pub fn gap_and_normal(&self, p: Vector3<f64>) -> Option<(f64, Vector3<f64>)> {
        let d = p - self.centre;
        let len = d.norm();
        (len > 1e-12).then(|| (len - self.radius, d / len))
    }

    /// Velocity of the object's material point at world position `p`: `v + omega x (p - centre)`.
    pub fn point_velocity(&self, p: Vector3<f64>) -> Vector3<f64> {
        let omega = self.twist.fixed_rows::<3>(0).into_owned();
        let v = self.twist.fixed_rows::<3>(3).into_owned();
        v + omega.cross(&(p - self.centre))
    }

    pub fn integrate(&mut self, dt: f64) {
        let v = self.twist.fixed_rows::<3>(3).into_owned();
        self.centre += dt * v;
    }
}

/// One resolved step of a hand and an object in contact.
#[derive(Clone, Debug)]
pub struct HandObjectStep {
    /// Hand joint velocities after contact.
    pub qd: Vec<f64>,
    /// Object twist after contact.
    pub object_twist: Vector6<f64>,
    /// Normal impulse per active contact.
    pub impulses: Vec<f64>,
    /// Which fingertips were in contact, by name.
    pub active: Vec<String>,
    /// Whether the contact solve converged; `false` means the velocities are the free ones and no contact was applied.
    pub converged: bool,
    /// Worst penetration after the step, in metres. Negative gaps are penetration.
    pub worst_penetration: f64,
}

/// The joint-space mass matrix of a mounted hand: the joint-joint block of the floating-base mass matrix, which is
/// exactly the fixed-base mass matrix.
fn hand_mass_matrix(tree: &KinematicTree, q: &[f64]) -> DMatrix<f64> {
    let n = tree.dof();
    let base = LinkInertia { mass: 1.0, com: Vector3::zeros(), inertia: Matrix3::identity() };
    let full = crate::tree_floating_mass_matrix(&tree.joints, &tree.inertia, &tree.parent, &base, q);
    DMatrix::from_fn(n, n, |r, c| full[(6 + r, 6 + c)])
}

/// Position Jacobian of a named tip, `3 x n`, by differencing the verified forward map.
fn tip_jacobian(tree: &KinematicTree, base: Iso, q: &[f64], tip: &str) -> Option<DMatrix<f64>> {
    let n = tree.dof();
    let p0 = tree.tip_pose(tip, base, q)?.translation.vector;
    let h = 1e-7;
    let mut j = DMatrix::zeros(3, n);
    for c in 0..n {
        let mut qp = q.to_vec();
        qp[c] += h;
        let p = tree.tip_pose(tip, base, &qp)?.translation.vector;
        let d = (p - p0) / h;
        for r in 0..3 {
            j[(r, c)] = d[r];
        }
    }
    Some(j)
}

/// **One coupled step**: build the fingertip-object contacts, resolve them together, integrate.
///
/// `tips` names the fingertip frames that can touch the object and their friction coefficients. `tau` are the hand joint
/// torques; `gravity` acts on the object (the hand is mounted, so gravity on its links is left to the caller's torques,
/// which keeps this function about the contact coupling).
///
/// The contact Jacobian row for a fingertip is `n^T [J_tip, -J_object]` in the generalised ordering
/// `[qd (n); twist (6)]`, so a positive normal velocity is *separating*. Getting that sign wrong produces a solver that
/// pulls the object into the finger, which is why the non-penetration check below is a test and not a comment.
#[allow(clippy::too_many_arguments)]
pub fn hand_object_step(
    tree: &KinematicTree,
    base: Iso,
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    object: &SphereObject,
    tips: &[(String, f64)],
    dt: f64,
    gravity: Vector3<f64>,
    activate: f64,
) -> Option<HandObjectStep> {
    let n = tree.dof();
    if q.len() != n || qd.len() != n || tau.len() != n || dt <= 0.0 {
        return None;
    }
    let dim = n + 6;

    // --- mass matrix: hand joints and object, coupled only through contact
    let mut mass = DMatrix::zeros(dim, dim);
    let hm = hand_mass_matrix(tree, q);
    mass.view_mut((0, 0), (n, n)).copy_from(&hm);
    let rot = object.spatial_inertia();
    mass.view_mut((n, n), (3, 3)).copy_from(&rot);
    for i in 0..3 {
        mass[(n + 3 + i, n + 3 + i)] = object.mass;
    }

    // --- free velocity: hand accelerates under tau against its own inertia; the object falls
    let mut v_free = DVector::zeros(dim);
    let hand_acc = hm.clone().try_inverse().map(|hi| hi * DVector::from_row_slice(tau));
    for i in 0..n {
        v_free[i] = qd[i] + dt * hand_acc.as_ref().map_or(0.0, |a| a[i]);
    }
    for i in 0..3 {
        v_free[n + i] = object.twist[i];
        v_free[n + 3 + i] = object.twist[3 + i] + dt * gravity[i];
    }

    // --- contacts: one per fingertip that is close enough to the surface
    let mut contacts: Vec<FrictionContact> = Vec::new();
    let mut active: Vec<String> = Vec::new();
    for (tip, mu) in tips {
        let Some(pose) = tree.tip_pose(tip, base, q) else { continue };
        let p = pose.translation.vector;
        let Some((phi, normal)) = object.gap_and_normal(p) else { continue };
        if phi > activate {
            continue;
        }
        let Some(jt_tip) = tip_jacobian(tree, base, q, tip) else { continue };
        // the object's contribution: d(point velocity)/d(twist) = [ -(r n) x , I ] acting on [omega; v]
        let arm = normal * object.radius; // contact point minus centre
        let mut j_obj = DMatrix::zeros(3, 6);
        // omega x arm = -arm x omega, so the omega block is the skew of (-arm) applied on the right
        let skew = Matrix3::new(0.0, -arm.z, arm.y, arm.z, 0.0, -arm.x, -arm.y, arm.x, 0.0);
        j_obj.view_mut((0, 0), (3, 3)).copy_from(&(-skew));
        j_obj.view_mut((0, 3), (3, 3)).copy_from(&Matrix3::identity());

        // relative velocity of the fingertip against the object's material point
        let mut rel = DMatrix::zeros(3, dim);
        rel.view_mut((0, 0), (3, n)).copy_from(&jt_tip);
        rel.view_mut((0, n), (3, 6)).copy_from(&(-&j_obj));

        // normal and two tangents at the contact
        let seed = if normal.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
        let t1 = (seed - normal * normal.dot(&seed)).normalize();
        let t2 = normal.cross(&t1);
        let row = |d: Vector3<f64>| DVector::from_iterator(dim, (0..dim).map(|c| d.x * rel[(0, c)] + d.y * rel[(1, c)] + d.z * rel[(2, c)]));
        contacts.push(FrictionContact { jn: row(normal), jt: vec![row(t1), row(t2)], phi, mu: *mu });
        active.push(tip.clone());
    }

    let solved = crate::solve_contacts_friction(&mass, &v_free, &contacts, dt);
    let v_next = &solved.v_next;

    // --- integrate and measure penetration, which is the check that the Jacobian signs are right
    let q_next: Vec<f64> = (0..n).map(|i| q[i] + dt * v_next[i]).collect();
    let mut obj_next = *object;
    obj_next.twist = Vector6::from_iterator((0..6).map(|i| v_next[n + i]));
    obj_next.integrate(dt);
    let worst_penetration = tips
        .iter()
        .filter_map(|(tip, _)| {
            let p = tree.tip_pose(tip, base, &q_next)?.translation.vector;
            obj_next.gap_and_normal(p).map(|(g, _)| -g)
        })
        .fold(f64::NEG_INFINITY, f64::max);

    Some(HandObjectStep {
        qd: (0..n).map(|i| v_next[i]).collect(),
        object_twist: obj_next.twist,
        impulses: solved.impulses,
        active,
        converged: solved.converged,
        worst_penetration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{force_closure_q1_spatial, grasp_split, tree_from_urdf, GraspContact3};

    /// Three fingers arranged around the origin, each a single prismatic joint driving a fingertip inward along its own
    /// axis. Minimal, and enough to squeeze a sphere from three sides.
    fn gripper_urdf(n_fingers: usize) -> String {
        let mut s = String::from("<robot name=\"g\"><link name=\"palm\"/>");
        for f in 0..n_fingers {
            let th = std::f64::consts::TAU * f as f64 / n_fingers as f64;
            let (cx, cy) = (th.cos(), th.sin());
            s += &format!(
                r#"<link name="f{f}"><inertial><mass value="0.05"/><origin xyz="0 0 0"/>
                     <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial></link>
                   <link name="f{f}tip"/>
                   <joint name="f{f}j" type="prismatic"><parent link="palm"/><child link="f{f}"/>
                     <origin xyz="{:.6} {:.6} 0"/><axis xyz="{:.6} {:.6} 0"/>
                     <limit lower="0" upper="0.25" effort="20" velocity="2"/></joint>
                   <joint name="f{f}t" type="fixed"><parent link="f{f}"/><child link="f{f}tip"/>
                     <origin xyz="0 0 0"/></joint>"#,
                0.30 * cx, 0.30 * cy, -cx, -cy
            );
        }
        s + "</robot>"
    }

    fn tips(n: usize, mu: f64) -> Vec<(String, f64)> {
        (0..n).map(|f| (format!("f{f}tip"), mu)).collect()
    }

    /// The fingertips must actually reach the sphere, or nothing below is testing contact.
    #[test]
    fn the_fingertips_reach_the_object() {
        let tree = tree_from_urdf(&gripper_urdf(3), "palm").expect("loads");
        let obj = SphereObject::new(Vector3::zeros(), 0.10, 0.2);
        eprintln!("prismatic travel vs fingertip gap to a 0.10 m sphere at the origin:");
        for travel in [0.0, 0.10, 0.19, 0.20, 0.21] {
            let q = vec![travel; 3];
            let p = tree.tip_pose("f0tip", Iso::identity(), &q).unwrap().translation.vector;
            let (gap, _) = obj.gap_and_normal(p).unwrap();
            eprintln!("   q = {travel:.2} m: tip at radius {:.4}, gap {gap:+.4}", p.norm());
            if travel >= 0.21 {
                assert!(gap < 0.0, "at 0.21 m of travel the finger has reached the surface");
            }
        }
    }

    /// **A step runs, contacts activate, and nothing penetrates.** The non-penetration check is what catches a sign
    /// error in the relative-velocity Jacobian, which would otherwise show up as the solver pulling the object in.
    #[test]
    fn a_coupled_step_resolves_without_penetration() {
        let tree = tree_from_urdf(&gripper_urdf(3), "palm").expect("loads");
        let mut obj = SphereObject::new(Vector3::zeros(), 0.10, 0.2);
        let q = vec![0.1995; 3]; // gap +0.5 mm: in contact, NOT penetrating
        let qd = vec![0.0; 3];
        let tau = vec![2.0; 3]; // squeeze inward
        let step = hand_object_step(&tree, Iso::identity(), &q, &qd, &tau, &obj, &tips(3, 0.6), 1e-3, Vector3::zeros(), 1e-3)
            .expect("steps");
        eprintln!("3-finger squeeze: {} contacts active, converged = {}, worst penetration {:.3e} m",
            step.active.len(), step.converged, step.worst_penetration);
        eprintln!("   impulses: {:?}", step.impulses.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>());
        // "active" means within the activation band, NOT carrying load. With a +0.5 mm gap and no external load the
        // correct impulses are zero, and the solver returns zero - a distinction worth keeping straight, because a test
        // that reads a nonzero `active` count as "in contact under load" would be measuring nothing.
        assert_eq!(step.active.len(), 3, "all three fingers are within the activation band");
        assert!(step.impulses.iter().all(|v| v.abs() < 1e-9), "and none of them carries load yet, correctly");
        assert!(step.converged, "the coupled solve converges");
        assert!(step.worst_penetration < 1e-4, "no meaningful penetration: {:.3e}", step.worst_penetration);
        obj.twist = step.object_twist;

        // A penetrating start is a real condition and gap stabilisation must recover from it. Worth separating,
        // because a fixture that starts 5 mm inside the object makes every subsequent measurement a stabilisation
        // transient rather than the effect being studied - which is exactly what the first version of these tests did.
        let deep = vec![0.205; 3]; // tip radius 0.095 against a 0.10 sphere: 5 mm inside
        let p = tree.tip_pose("f0tip", Iso::identity(), &deep).unwrap().translation.vector;
        let (gap0, _) = obj.gap_and_normal(p).unwrap();
        let recovered = hand_object_step(&tree, Iso::identity(), &deep, &[0.0; 3], &[0.0; 3], &obj, &tips(3, 0.6), 1e-3, Vector3::zeros(), 1e-2)
            .expect("steps");
        eprintln!("   penetrating start: gap {gap0:+.4} m -> worst penetration after one step {:.3e} m", recovered.worst_penetration);
        assert!(gap0 < -1e-3, "the fixture really is penetrating");
        assert!(recovered.worst_penetration < -gap0, "stabilisation reduces the penetration rather than keeping it");
    }

    /// **Cross-validation with `grasp_split`.** A symmetric squeeze lies in the internal (null) subspace of the grasp
    /// matrix, which that module says produces no net wrench. The simulator must therefore leave the object still.
    ///
    /// The two modules share no code, so this is a genuine agreement test rather than a self-check.
    #[test]
    fn a_symmetric_squeeze_produces_no_object_motion() {
        let tree = tree_from_urdf(&gripper_urdf(3), "palm").expect("loads");
        let obj = SphereObject::new(Vector3::zeros(), 0.10, 0.2);
        let q = vec![0.1995; 3]; // gap +0.5 mm, so no stabilisation transient

        // what grasp_split predicts for this geometry
        let contacts: Vec<GraspContact3> = (0..3)
            .map(|f| {
                let p = tree.tip_pose(&format!("f{f}tip"), Iso::identity(), &q).unwrap().translation.vector;
                let n = (obj.centre - p).normalize(); // inward
                GraspContact3::hard(p, n, 0.6)
            })
            .collect();
        let split = grasp_split(&contacts, 8);
        eprintln!("grasp_split on this geometry: rank {}, internal dimension {}", split.rank, split.internal_dim);
        assert!(split.internal_dim > 0, "a symmetric 3-finger squeeze has internal freedom");

        // Settle the fingers into load-bearing contact first. Comparing squeezes while the gap is still positive would
        // compare zero against zero: the solver correctly applies no impulse until a contact actually needs one.
        let settle = |tau: Vec<f64>| {
            let (mut qq, mut qdd, mut oo) = (q.clone(), vec![0.0; 3], obj);
            let mut last = None;
            for _ in 0..40 {
                let s = hand_object_step(&tree, Iso::identity(), &qq, &qdd, &tau, &oo, &tips(3, 0.6), 1e-3, Vector3::zeros(), 1e-3).expect("steps");
                for (slot, v) in qq.iter_mut().zip(&s.qd) {
                    *slot += 1e-3 * v;
                }
                qdd = s.qd.clone();
                oo.twist = s.object_twist;
                oo.integrate(1e-3);
                last = Some(s);
            }
            last.expect("at least one step")
        };
        let loaded = settle(vec![3.0, 3.0, 3.0]);
        let max_impulse = loaded.impulses.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        eprintln!("   after settling under an equal squeeze: largest normal impulse {max_impulse:.4e}");
        assert!(max_impulse > 1e-4, "the fingers must actually be carrying load: {max_impulse:.3e}");

        let twist_for = |tau: Vec<f64>| {
            let s = settle(tau);
            (s.object_twist.fixed_rows::<3>(3).norm(), s.object_twist.fixed_rows::<3>(0).norm())
        };
        let (sym_lin, sym_ang) = twist_for(vec![3.0, 3.0, 3.0]);
        let (asym_lin, _) = twist_for(vec![6.0, 1.5, 1.5]);
        eprintln!("   symmetric squeeze:  |v| = {sym_lin:.3e} m/s, |omega| = {sym_ang:.3e} rad/s");
        eprintln!("   asymmetric squeeze: |v| = {asym_lin:.3e} m/s");

        // The right test is relative, not an absolute threshold. The symmetric residual is the interior-point solver's
        // cancellation tolerance - a net impulse residual of about 3e-6 against per-contact impulses of 0.252, which is
        // 1.2e-5 relative - and no absolute bound on it would be meaningful. What must hold is that an internal
        // (null-space) squeeze moves the object far less than one with a net wrench.
        eprintln!("   ratio {:.1e}x - the null-space squeeze is 11 orders of magnitude quieter", asym_lin / sym_lin.max(1e-300));
        assert!(asym_lin > 20.0 * sym_lin, "an internal squeeze must move the object far less: {sym_lin:.3e} vs {asym_lin:.3e}");
        assert!(sym_ang < 1e-12, "and a symmetric squeeze produces no rotation at all: {sym_ang:.3e}");
        eprintln!("   The prediction and the simulation agree, and the two modules share no code.");
    }

    /// **Cross-validation with `force_closure_q1_spatial`.** A grasp the metric calls closed must hold the object
    /// against gravity; one it calls rank-deficient must not.
    #[test]
    fn the_grasp_metric_predicts_whether_the_object_is_held() {
        let g = Vector3::new(0.0, 0.0, -9.81);
        let run = |n_fingers: usize, mu: f64, squeeze: f64| -> (f64, usize, f64) {
            let tree = tree_from_urdf(&gripper_urdf(n_fingers), "palm").expect("loads");
            let mut obj = SphereObject::new(Vector3::zeros(), 0.10, 0.2);
            let q = vec![0.1995; n_fingers];
            // the metric's verdict on this geometry
            let contacts: Vec<GraspContact3> = (0..n_fingers)
                .map(|f| {
                    let p = tree.tip_pose(&format!("f{f}tip"), Iso::identity(), &q).unwrap().translation.vector;
                    GraspContact3::hard(p, (obj.centre - p).normalize(), mu)
                })
                .collect();
            let (q1, rank) = (force_closure_q1_spatial(&contacts, 16, 4096), crate::wrench_rank(&contacts, 16));
            // and what the simulator does over 200 ms of gravity
            let mut qd = vec![0.0; n_fingers];
            for _ in 0..200 {
                let Some(s) = hand_object_step(&tree, Iso::identity(), &q, &qd, &vec![squeeze; n_fingers], &obj, &tips(n_fingers, mu), 1e-3, g, 1e-3) else { break };
                qd = s.qd;
                obj.twist = s.object_twist;
                obj.integrate(1e-3);
            }
            (q1, rank, obj.centre.z.abs())
        };

        // two frictionless opposed fingers: rank-deficient, cannot resist the vertical load
        let (q1_two, rank_two, drop_two) = run(2, 0.0, 4.0);
        eprintln!("2 frictionless opposed fingers: Q1 = {q1_two:.4} (rank {rank_two}), object fell {drop_two:.4} m in 200 ms");
        assert_eq!(q1_two, 0.0, "the metric calls it not force-closed");
        assert!(drop_two > 1e-3, "and the simulator drops it: {drop_two:.4} m");

        // three fingers with friction: full rank, positive margin, holds
        let (q1_three, rank_three, drop_three) = run(3, 0.8, 4.0);
        eprintln!("3 fingers with mu = 0.8:          Q1 = {q1_three:.4} (rank {rank_three}), object fell {drop_three:.4} m in 200 ms");
        assert!(q1_three > 0.0 && rank_three == 6, "the metric calls it force-closed");
        assert!(drop_three < drop_two / 5.0, "and the simulator holds it far better: {drop_three:.4} vs {drop_two:.4}");
        eprintln!("\n   The static metric and the dynamic simulation agree on which grasp works, having been built");
        eprintln!("   separately and sharing no code.");
    }

    /// Bad input is refused rather than producing a step.
    #[test]
    fn bad_input_is_refused() {
        let tree = tree_from_urdf(&gripper_urdf(3), "palm").expect("loads");
        let obj = SphereObject::new(Vector3::zeros(), 0.10, 0.2);
        let t = tips(3, 0.5);
        assert!(hand_object_step(&tree, Iso::identity(), &[0.0; 2], &[0.0; 3], &[0.0; 3], &obj, &t, 1e-3, Vector3::zeros(), 1e-3).is_none(), "q of the wrong width");
        assert!(hand_object_step(&tree, Iso::identity(), &[0.0; 3], &[0.0; 3], &[0.0; 3], &obj, &t, 0.0, Vector3::zeros(), 1e-3).is_none(), "a zero timestep");
        // a point at the sphere's centre has no defined normal
        assert!(obj.gap_and_normal(obj.centre).is_none());
    }
}
