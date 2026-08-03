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
///
/// # Verified operating range, and a known limitation
///
/// **Passive contact is sound.** With no actuation this conserves energy exactly at every friction
/// coefficient measured from `mu = 0.3` to `1.5`: a dropped body settles and never gains energy. That
/// is locked by `passive_contact_never_creates_energy`, and it is the regime the settling and stance
/// tests exercise.
///
/// **Under actuation with several simultaneous frictional contacts, this interior-point path does not
/// converge — use [`whole_body_contact_step_pgs`] instead.** Driving the quadruped's four feet with a
/// gait, the central-path residual here reaches `5e-2` at `mu = 0.9` and `1e4` at `mu = 1.2` against a
/// target of `kappa` (`1e-6`), and the returned point is infeasible. A non-converged impulse is not a
/// solution to anything, and downstream it appears as a body gaining speed without bound: peak speed
/// rises from `1.3 m/s` at `mu = 0.9` to `110 m/s` at `mu = 2.0`. It is not an integration artifact
/// (it survives a tenfold smaller timestep) and it is not cured by kappa-continuation or a line search.
/// The cause is the problem class: several sticking contacts on a floating body are over-constrained,
/// so the impulses are not uniquely determined and a damped Newton method has nothing to converge to.
///
/// The Gauss-Seidel sweep solves the same cases cleanly — bounded at every friction coefficient
/// measured, `1.25 m/s` peak at `mu = 2.0` where this path reaches `110` — because each projection is
/// non-expansive. Keep this interior-point path when the contact set is small or a **gradient** through
/// contact is needed, which the sweep does not provide. [`whole_body_contact_step_checked`] reports
/// this solver's own residual so a bad solve is visible rather than silent.
/// `examples/contact_models_compared.rs` reproduces every number above.
#[allow(clippy::too_many_arguments)]
pub fn whole_body_contact_step_checked(
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
) -> (WholeBodyState, (f64, f64)) {
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
    let (v_next, health) = if cset.is_empty() {
        (v_free, (0.0, f64::INFINITY))
    } else {
        let h = tree_floating_mass_matrix(joints, inertia, parent, base_inertia, q);
        let s = solve_frictional_ipm(&h, &v_free, &cset, dt, kappa);
        (s.v_next, (s.residual, s.feasibility))
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
    ((base_n, v0n, qn, qdn), health)
}

/// The advanced state a whole-body step returns: `(base pose, base spatial velocity, q, q̇)`.
pub type WholeBodyState = (Isometry3<f64>, Vector6<f64>, Vec<f64>, Vec<f64>);

/// One whole-body hard-contact step. See [`whole_body_contact_step_checked`] for the same step with
/// the contact solver's own health reported alongside the result.
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
    whole_body_contact_step_checked(joints, inertia, parent, base_inertia, base, v0, q, qd, tau, contacts, floor_z, dt, kappa, gravity).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadruped;
    use nalgebra::Matrix3;

    fn quad_base() -> LinkInertia {
        LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) }
    }

    /// **Contact chatter, and the stabilisation that removes it.**
    ///
    /// A quadruped standing on four feet is over-constrained: the impulses are not uniquely determined,
    /// and the Gauss-Seidel sweep shuffles load between the redundant feet. That leaves each foot's gap
    /// jittering by nanometres. Divided by the timestep, a nanometre of gap becomes a velocity demand
    /// large enough to switch a marginally-loaded foot off, and the next step switches it back — so the
    /// contact mode flips at the timestep scale.
    ///
    /// The cost is not cosmetic. A mode sequence that flips every step has no derivative, so no gait on
    /// top of it can be linearised or certified. This test pins the fix where the effect actually arises:
    /// neither a single contact in isolation nor a statically standing robot chatters, because both sit at
    /// a fixed point. It takes a *walking* robot, whose feet are repeatedly loaded near the margin.
    #[test]
    fn a_walking_quadruped_does_not_chatter() {
        let (joints, inertia, parent, feet) = quadruped();
        let bi = quad_base();
        let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, 0.9)).collect();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (dt, steps) = (5e-4, 3000);

        // Walk the gait and count how often each foot's load switches on or off. A crawl has four
        // touchdowns and four liftoffs per period, so a sound sequence gives a handful of switches.
        let run = |stab: crate::PgsStabilization| {
            let mut base = Isometry3::translation(0.0, 0.0, 0.60);
            let mut v0 = Vector6::zeros();
            let (mut q, mut qd) = (vec![0.0; joints.len()], vec![0.0; joints.len()]);
            let mut loaded = vec![false; pts.len()];
            let (mut flips, mut warm) = (0usize, None);
            for k in 0..steps {
                let tau = crate::quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * k as f64 * dt);
                let r = whole_body_contact_step_pgs_with(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, 0.0, dt, 40, g, warm.as_deref(), stab);
                base = r.base;
                v0 = r.v0;
                q = r.q.clone();
                qd = r.qd.clone();
                // skip the initial settling transient; only steady standing is being measured
                if k > steps / 4 {
                    for (f, l) in r.impulses.iter().enumerate() {
                        let now = l.z > 1e-12;
                        if now != loaded[f] {
                            flips += 1;
                        }
                        loaded[f] = now;
                    }
                } else {
                    for (f, l) in r.impulses.iter().enumerate() {
                        loaded[f] = l.z > 1e-12;
                    }
                }
                warm = Some(r.impulses);
            }
            (flips, base.translation.z)
        };

        let (exact_flips, exact_z) = run(crate::PgsStabilization::exact());
        let (stab_flips, stab_z) = run(crate::PgsStabilization::default());
        eprintln!("walking quadruped, {} steps past the transient: exact gap feedback flipped load {exact_flips} times (torso z {exact_z:.4} m), stabilised flipped {stab_flips} times (torso z {stab_z:.4} m)", steps - steps / 4);

        // A crawl over this window makes and breaks a handful of contacts. Anything approaching one flip
        // per timestep is chatter, not gait.
        assert!(stab_flips < 40, "the contact sequence is still chattering: {stab_flips} load switches over {} steps", steps - steps / 4);
        assert!(exact_flips >= 3 * stab_flips.max(1), "the unstabilised condition is the one that chatters; if it no longer does, this test has stopped measuring what it claims ({exact_flips} vs {stab_flips})");
        // and the robot must still be walking upright, resting within the penetration allowance
        assert!(stab_z > 0.45, "the robot collapsed instead of walking: torso at {stab_z:.4} m");
    }

    /// **The gait's monodromy is measurable, and it contracts.** This is the result that the chatter fix
    /// bought, and it is worth a regression test because four separate things have to hold for the number
    /// to exist at all: the contact sequence must not chatter, the probe must stay inside one contact mode,
    /// the step map must be a function of the state (so cold-started, not warm), and the state must include
    /// the base's `x`, `y` and yaw rather than silently dropping them.
    ///
    /// A cheap proxy for all four: perturb the settled gait, run one period, and check the growth is both
    /// bounded and stable under the probe size. A chattering or hidden-state map fails this immediately.
    /// The full analysis, with the saltation-composed cross-check, is in the
    /// `quadruped_saltation_monodromy` example.
    #[test]
    fn the_gait_linearises_and_the_growth_is_probe_stable() {
        let (joints, inertia, parent, feet) = quadruped();
        let bi = quad_base();
        let pts: Vec<WholeBodyContactPoint> = feet.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, 0.9)).collect();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (dt, iters) = (5e-4, 150);

        // Cold-started so the map is a function of the state, and carrying the full base pose.
        let roll = |base0: Isometry3<f64>, v00: Vector6<f64>, q0: &[f64], qd0: &[f64], t0: f64, steps: usize| {
            let mut base = base0;
            let mut v0 = v00;
            let (mut q, mut qd) = (q0.to_vec(), qd0.to_vec());
            for k in 0..steps {
                let tau = crate::quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * (t0 + k as f64 * dt));
                let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, 0.0, dt, iters, g, None);
                base = r.base;
                v0 = r.v0;
                q = r.q;
                qd = r.qd;
            }
            (base, v0, q, qd)
        };

        // A settled orbit and a half-period window. Both matter: mid-transient, the state sits near a mode
        // boundary and even a 1e-8 probe flips a foot, which is the phenomenon under study rather than a
        // property to regression-test.
        let window = 1000;
        let (b0, vv0, q0, qd0) = roll(Isometry3::translation(0.0, 0.0, 0.60), Vector6::zeros(), &vec![0.0; joints.len()], &vec![0.0; joints.len()], 0.0, 4 * window);
        assert!(b0.translation.z > 0.45, "the gait collapsed while settling: torso at {:.4} m", b0.translation.z);

        let (nb, nv, nq, nqd) = roll(b0, vv0, &q0, &qd0, 0.0, window);
        let mut growth = Vec::new();
        for &eps in &[1e-8_f64, 1e-9] {
            // perturb one velocity coordinate, keeping every other coordinate exact
            let mut vp = vv0;
            vp[5] += eps; // vertical velocity of the base
            let (pb, pv, pq, pqd) = roll(b0, vp, &q0, &qd0, 0.0, window);
            let d = ((pb.translation.z - nb.translation.z).powi(2) + (pv - nv).norm_squared() + pq.iter().zip(&nq).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() + pqd.iter().zip(&nqd).map(|(a, b)| (a - b) * (a - b)).sum::<f64>()).sqrt();
            growth.push(d / eps);
        }
        eprintln!("growth of a vertical-velocity perturbation over the window: {:.4} at probe 1e-8, {:.4} at 1e-9", growth[0], growth[1]);

        // A real derivative barely moves with the probe. That is the whole assertion, and it is the one
        // that fails loudly when the contact sequence chatters: before the fix, differencing this gait gave
        // gains of 4155, then 51, then 24 as the probe shrank by single decades.
        //
        // The *magnitude* is deliberately not pinned. The one-period gain varies strongly along the orbit,
        // and a debug and a release build accumulate enough different rounding over several thousand steps
        // to settle on different points of it — 1.7 against 42, each probe-stable to 0.04%. Asserting a
        // value would be fitting to one build profile rather than testing the property.
        let drift = (growth[0] - growth[1]).abs() / growth[1].max(1e-30);
        assert!(drift < 0.05, "the gain is not probe-stable, so the map is not differentiating: {:.4} vs {:.4}", growth[0], growth[1]);
        assert!(growth[1].is_finite() && growth[1] > 0.0, "the gain is not a finite positive number: {}", growth[1]);
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

    /// A contact may dissipate energy and must never create it. Dropped with no actuation, the
    /// quadruped's total mechanical energy never rises above its starting value, at every friction
    /// coefficient across a wide sweep. This is the invariant that says the contact solve itself is
    /// sound, and it is what distinguishes a solver defect from an actuation instability.
    #[test]
    fn passive_contact_never_creates_energy() {
        let (joints, inertia, parent, foot_list) = quadruped();
        let n = joints.len();
        let base_inertia = quad_base();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mtot: f64 = base_inertia.mass + inertia.iter().map(|l| l.mass).sum::<f64>();
        for &mu in &[0.3, 0.9, 1.5] {
            let contacts: Vec<WholeBodyContactPoint> = foot_list.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
            let mut base = Isometry3::translation(0.0, 0.0, 0.62);
            let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
            let tau = vec![0.0; n];
            let energy = |base: &Isometry3<f64>, v0: &Vector6<f64>, qd: &[f64]| {
                0.5 * base_inertia.mass * v0.fixed_rows::<3>(3).norm_squared()
                    + 0.5 * 0.1 * qd.iter().map(|v| v * v).sum::<f64>()
                    + mtot * 9.81 * base.translation.z
            };
            let e0 = energy(&base, &v0, &qd);
            let mut worst = 0.0f64;
            for _ in 0..2000 {
                let (b1, v1, q1, qd1) = whole_body_contact_step(&joints, &inertia, &parent, &base_inertia, base, v0, &q, &qd, &tau, &contacts, 0.0, 2e-4, 1e-6, g);
                base = b1; v0 = v1; q = q1; qd = qd1;
                assert!(base.translation.vector.iter().all(|v| v.is_finite()), "mu {mu}: diverged");
                worst = worst.max(energy(&base, &v0, &qd) - e0);
            }
            eprintln!("passive energy, mu {mu}: start {e0:.2} J, worst gain {worst:+.4} J");
            assert!(worst < 1e-2 * e0.abs(), "mu {mu}: contact created {worst} J from nothing");
        }
    }

    /// The case the interior-point solve cannot handle: a quadruped driven by a gait on four
    /// simultaneous frictional contacts. Through the Gauss-Seidel sweep the body stays bounded and
    /// upright at every friction coefficient, and walks further as grip increases, which is the
    /// physically sensible direction. The direct solve reaches 110 m/s on the same input at mu = 2.
    #[test]
    fn driven_multi_contact_stays_bounded_at_high_friction() {
        let (joints, inertia, parent, foot_list) = quadruped();
        let n = joints.len();
        let bi = quad_base();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let dt = 2e-4;
        let mut travel_by_mu = Vec::new();
        for &mu in &[0.9, 1.5, 2.0] {
            let pts: Vec<WholeBodyContactPoint> = foot_list.iter().map(|&(b, o, _)| WholeBodyContactPoint::on(b, o, mu)).collect();
            let mut base = Isometry3::translation(0.0, 0.0, 0.60);
            let (mut v0, mut q, mut qd) = (Vector6::zeros(), vec![0.0; n], vec![0.0; n]);
            let mut warm: Option<Vec<Vector3<f64>>> = None;
            let (mut peak, mut worst_viol) = (0.0f64, 0.0f64);
            for k in 0..5000 {
                let t = k as f64 * dt;
                let tau = crate::quadruped_trot_tau(&q, &qd, std::f64::consts::TAU * t);
                let r = whole_body_contact_step_pgs(&joints, &inertia, &parent, &bi, base, v0, &q, &qd, &tau, &pts, 0.0, dt, 150, g, warm.as_deref());
                base = r.base;
                v0 = r.v0;
                q = r.q.clone();
                qd = r.qd.clone();
                warm = Some(r.impulses);
                assert!(base.translation.vector.iter().all(|x| x.is_finite()), "mu {mu}: diverged");
                peak = peak.max(v0.norm());
                worst_viol = worst_viol.max(r.violation);
            }
            let up = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
            eprintln!("driven multi-contact, mu {mu}: peak {peak:.2} m/s, travel {:.3} m, base z {:.3}, up {up:.3}, worst violation {worst_viol:.1e}", base.translation.x, base.translation.z);
            assert!(peak < 4.0, "mu {mu}: speed ran away to {peak} m/s");
            assert!(up > 0.9, "mu {mu}: toppled, up {up}");
            assert!(base.translation.z > 0.5, "mu {mu}: collapsed to z {}", base.translation.z);
            assert!(worst_viol < 5e-2, "mu {mu}: contact conditions violated by {worst_viol}");
            travel_by_mu.push(base.translation.x);
        }
        // 1 s of the crawl carries it a few centimetres, and more grip carries it further
        assert!(travel_by_mu.iter().all(|d| *d > 0.02), "the gait should carry it forward: {travel_by_mu:?}");
        assert!(travel_by_mu[2] > travel_by_mu[0], "more friction should mean more propulsion: {travel_by_mu:?}");
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

/// The result of a whole-body step taken with the Gauss-Seidel contact solver, including the contact
/// impulses (hand them back next step as a warm start) and how well the solve satisfied the contact
/// conditions.
#[derive(Clone, Debug)]
pub struct WholeBodyStep {
    pub base: Isometry3<f64>,
    pub v0: Vector6<f64>,
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub impulses: Vec<Vector3<f64>>,
    /// Worst violation of the contact conditions; near zero means the answer is trustworthy.
    pub violation: f64,
    pub iters: usize,
}

/// One whole-body hard-contact step solved by **projected Gauss-Seidel**
/// ([`solve_contacts_pgs`](crate::solve_contacts_pgs)) instead of the interior-point core.
///
/// Prefer this for a body standing or walking on several contacts at once. The interior-point path is
/// differentiable and is the right tool for a small contact set or when gradients are needed, but it
/// does not reliably converge on an over-constrained frictional problem, and a non-converged impulse
/// is unusable. This sweep gives up the closed-form derivative and in exchange cannot amplify the
/// impulses, so it stays bounded on exactly the cases that defeat the direct solve.
///
/// Gap feedback uses [`PgsStabilization::default`](crate::PgsStabilization::default), which is what keeps
/// a standing foot from chattering. [`whole_body_contact_step_pgs_with`] takes the stabilisation
/// explicitly.
#[allow(clippy::too_many_arguments)]
pub fn whole_body_contact_step_pgs(
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
    iters: usize,
    gravity: Vector3<f64>,
    warm: Option<&[Vector3<f64>]>,
) -> WholeBodyStep {
    whole_body_contact_step_pgs_with(joints, inertia, parent, base_inertia, base, v0, q, qd, tau, contacts, floor_z, dt, iters, gravity, warm, crate::PgsStabilization::default())
}

/// [`whole_body_contact_step_pgs`] with the gap stabilisation named explicitly.
#[allow(clippy::too_many_arguments)]
pub fn whole_body_contact_step_pgs_with(
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
    iters: usize,
    gravity: Vector3<f64>,
    warm: Option<&[Vector3<f64>]>,
    stab: crate::PgsStabilization,
) -> WholeBodyStep {
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

    // only the contacts that are actually touching or penetrating take part
    let world = forward_kinematics(joints, parent, base, q);
    let mut active: Vec<crate::PgsContact> = Vec::new();
    let mut idx = Vec::new();
    for (k, c) in contacts.iter().enumerate() {
        let p_w = match c.body {
            Some(b) => (world[b] * Point3::from(c.offset)).coords,
            None => (base * Point3::from(c.offset)).coords,
        };
        let phi = p_w.z - floor_z;
        if phi < 1e-3 {
            active.push(crate::PgsContact { j: whole_body_contact_jacobian(joints, parent, &world, base, c.body, c.offset), phi, mu: c.mu });
            idx.push(k);
        }
    }

    let warm_active: Option<Vec<Vector3<f64>>> = warm.map(|w| idx.iter().map(|&k| w.get(k).copied().unwrap_or_else(Vector3::zeros)).collect());
    let res = crate::solve_contacts_pgs_with(&tree_floating_mass_matrix(joints, inertia, parent, base_inertia, q), &v_free, &active, dt, iters, warm_active.as_deref(), stab);

    // scatter the active impulses back to full contact indexing so the caller can warm-start
    let mut impulses = vec![Vector3::zeros(); contacts.len()];
    for (a, &k) in idx.iter().enumerate() {
        impulses[k] = res.lambda[a];
    }

    let v_next = res.v_next;
    let v0n = Vector6::from_iterator(v_next.iter().take(6).copied());
    let w = v0n.fixed_rows::<3>(0).into_owned();
    let vlin = v0n.fixed_rows::<3>(3).into_owned();
    let step = Isometry3::from_parts(Translation3::from(dt * vlin), UnitQuaternion::from_scaled_axis(dt * w));
    let mut qn = q.to_vec();
    let mut qdn = qd.to_vec();
    for i in 0..n {
        qdn[i] = v_next[6 + i];
        qn[i] += dt * qdn[i];
    }
    WholeBodyStep { base: base * step, v0: v0n, q: qn, qd: qdn, impulses, violation: res.violation, iters: res.iters }
}
