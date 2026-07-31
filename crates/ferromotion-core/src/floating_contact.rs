//! **Floating-base contact + SE(3) integration** — the step that turns the floating-base dynamics
//! ([`floating_base_forward_dynamics_ext`](crate::floating_base_forward_dynamics_ext)) into a
//! simulator a legged body can locomote in. Feet (contact points on links) meet a ground plane; the
//! penalty force at each foot becomes an external spatial wrench on that link, ABA propagates the
//! reaction to the free base, and the base pose is advanced on SE(3). This is the physics layer under
//! learned floating-base locomotion — built and validated on the CPU (drop-and-settle) so it is the
//! trusted reference the GPU port checks against. Pure `nalgebra` → WASM-clean.

use crate::aba::{motion_subspace, motion_transform};
use crate::{floating_base_forward_dynamics_ext, tree_floating_forward_dynamics, Joint, LinkInertia, Robot};
use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector2, Vector3, Vector6};

/// A contact point on the robot: `(frame index 0..=dof, offset in that frame, friction μ)`.
pub type FootContact = (usize, Vector3<f64>, f64);

/// One floating-base contact step (semi-implicit Euler). `base` = `world_from_base` pose, `v0` = base
/// spatial velocity `[ω; v]` in the base frame. Returns the advanced `(base pose, v0, q, qd)`.
#[allow(clippy::too_many_arguments)]
pub fn floating_contact_step(
    robot: &Robot,
    inertia: &[LinkInertia],
    base_inertia: &LinkInertia,
    base: Isometry3<f64>,
    v0: Vector6<f64>,
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    contacts: &[FootContact],
    floor_z: f64,
    kn: f64,
    kd: f64,
    dt: f64,
    g: Vector3<f64>,
) -> (Isometry3<f64>, Vector6<f64>, Vec<f64>, Vec<f64>) {
    let n = robot.dof();

    // Per-frame spatial velocity (in each frame's own coordinates), seeded by the base velocity —
    // the same outward recursion the ABA uses, needed here to get each foot's velocity.
    let mut vframes = vec![Vector6::zeros(); n];
    let mut vpar = v0;
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        let x = motion_transform(*a.rotation.to_rotation_matrix().matrix(), a.translation.vector);
        let si = motion_subspace(robot.joints[i].kind, robot.joints[i].axis.into_inner());
        vframes[i] = x * vpar + si * qd[i];
        vpar = vframes[i];
    }

    // Contact → external spatial wrenches.
    let mut f_ext_base = Vector6::zeros();
    let mut f_ext = vec![Vector6::zeros(); n];
    for &(frame, offset, mu) in contacts {
        // world pose of the contact frame, and the contact point
        let wf = base * robot.frame_pose(q, frame); // world_from_frame
        let p_foot = (wf * Point3::from(offset)).coords;
        let phi = p_foot.z - floor_z;
        if phi >= 0.0 {
            continue;
        }
        // contact-point velocity in world: R_wf·(ω_link × offset + v_link), link vel in frame coords
        let v_link = if frame == 0 { v0 } else { vframes[frame - 1] };
        let (wl, vl) = (v_link.fixed_rows::<3>(0).into_owned(), v_link.fixed_rows::<3>(3).into_owned());
        let r_wf = *wf.rotation.to_rotation_matrix().matrix();
        let v_cp = r_wf * (wl.cross(&offset) + vl);
        // spring–dashpot normal (push only) + regularized-Coulomb friction, in world
        let fnrm = (-kn * phi - kd * v_cp.z).max(0.0);
        let vt = Vector2::new(v_cp.x, v_cp.y);
        let ft = -mu * fnrm * vt / (vt.norm() + 1e-4);
        let f_world = Vector3::new(ft.x, ft.y, fnrm);
        // as a spatial force in the contact frame: [offset × f_local ; f_local]
        let f_local = r_wf.transpose() * f_world;
        let mut w = Vector6::zeros();
        w.fixed_rows_mut::<3>(0).copy_from(&offset.cross(&f_local));
        w.fixed_rows_mut::<3>(3).copy_from(&f_local);
        if frame == 0 {
            f_ext_base += w;
        } else {
            f_ext[frame - 1] += w;
        }
    }

    let (a0, qdd) = floating_base_forward_dynamics_ext(robot, inertia, base_inertia, v0, q, qd, tau, f_ext_base, &f_ext, g);

    // integrate (semi-implicit): base spatial velocity, then joints, then the base pose on SE(3)
    let v0n = v0 + dt * a0;
    let mut qn = q.to_vec();
    let mut qdn = qd.to_vec();
    for i in 0..n {
        qdn[i] += dt * qdd[i];
        qn[i] += dt * qdn[i];
    }
    let w = v0n.fixed_rows::<3>(0).into_owned();
    let vlin = v0n.fixed_rows::<3>(3).into_owned();
    let step = Isometry3::from_parts(Translation3::from(dt * vlin), UnitQuaternion::from_scaled_axis(dt * w));
    let basen = base * step; // body-frame twist integration
    (basen, v0n, qn, qdn)
}

/// One floating-base contact step for a **kinematic tree** (a quadruped/biped) — the multi-leg
/// generalization of [`floating_contact_step`]. `parent[i]` is body `i`'s parent (`-1` = base);
/// `contacts` are `(body index, offset in that body's frame, μ)`. Uses
/// [`tree_floating_forward_dynamics`](crate::tree_floating_forward_dynamics) and returns the advanced
/// `(base pose, v0, q, qd)`.
#[allow(clippy::too_many_arguments)]
pub fn tree_floating_contact_step(
    joints: &[Joint],
    inertia: &[LinkInertia],
    parent: &[isize],
    base_inertia: &LinkInertia,
    base: Isometry3<f64>,
    v0: Vector6<f64>,
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    contacts: &[FootContact],
    floor_z: f64,
    kn: f64,
    kd: f64,
    dt: f64,
    g: Vector3<f64>,
) -> (Isometry3<f64>, Vector6<f64>, Vec<f64>, Vec<f64>) {
    let n = joints.len();
    // tree forward kinematics: base→body pose `tt[i]` and each body's spatial velocity `vf[i]`
    let mut tt = vec![Isometry3::identity(); n];
    let mut vf = vec![Vector6::zeros(); n];
    for i in 0..n {
        let tf = joints[i].transform(q[i]);
        let tpar = if parent[i] < 0 { Isometry3::identity() } else { tt[parent[i] as usize] };
        tt[i] = tpar * tf;
        let x = motion_transform(*tf.rotation.to_rotation_matrix().matrix(), tf.translation.vector);
        let si = motion_subspace(joints[i].kind, joints[i].axis.into_inner());
        let vpar = if parent[i] < 0 { v0 } else { vf[parent[i] as usize] };
        vf[i] = x * vpar + si * qd[i];
    }
    let mut f_ext = vec![Vector6::zeros(); n];
    for &(body, offset, mu) in contacts {
        let wf = base * tt[body];
        let p = (wf * Point3::from(offset)).coords;
        let phi = p.z - floor_z;
        if phi >= 0.0 {
            continue;
        }
        let (wl, vl) = (vf[body].fixed_rows::<3>(0).into_owned(), vf[body].fixed_rows::<3>(3).into_owned());
        let r_wf = *wf.rotation.to_rotation_matrix().matrix();
        let v_cp = r_wf * (wl.cross(&offset) + vl);
        let fnrm = (-kn * phi - kd * v_cp.z).max(0.0);
        let vt = Vector2::new(v_cp.x, v_cp.y);
        let ft = -mu * fnrm * vt / (vt.norm() + 1e-4);
        let f_local = r_wf.transpose() * Vector3::new(ft.x, ft.y, fnrm);
        let mut w = Vector6::zeros();
        w.fixed_rows_mut::<3>(0).copy_from(&offset.cross(&f_local));
        w.fixed_rows_mut::<3>(3).copy_from(&f_local);
        f_ext[body] += w;
    }
    let (a0, qdd) = tree_floating_forward_dynamics(joints, inertia, parent, base_inertia, v0, q, qd, tau, Vector6::zeros(), &f_ext, g);
    let v0n = v0 + dt * a0;
    let mut qn = q.to_vec();
    let mut qdn = qd.to_vec();
    for i in 0..n {
        qdn[i] += dt * qdd[i];
        qn[i] += dt * qdn[i];
    }
    let w = v0n.fixed_rows::<3>(0).into_owned();
    let vlin = v0n.fixed_rows::<3>(3).into_owned();
    let step = Isometry3::from_parts(Translation3::from(dt * vlin), UnitQuaternion::from_scaled_axis(dt * w));
    (base * step, v0n, qn, qdn)
}

/// A 4-legged robot (a torso with four 2-joint legs at its corners): `(joints, inertia, parent,
/// foot contacts)`. Legs point straight down (`q = 0`); each foot is the shank tip.
pub fn quadruped() -> (Vec<Joint>, Vec<LinkInertia>, Vec<isize>, Vec<FootContact>) {
    let thigh = LinkInertia { mass: 0.8, com: Vector3::new(0.0, 0.0, -0.15), inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.006, 0.006, 0.002)) };
    let shank = LinkInertia { mass: 0.4, com: Vector3::new(0.0, 0.0, -0.15), inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.004, 0.004, 0.0015)) };
    let corners = [(0.15, 0.1), (0.15, -0.1), (-0.15, 0.1), (-0.15, -0.1)];
    let (mut joints, mut inertia, mut parent, mut contacts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &(cx, cy) in &corners {
        let hip = joints.len();
        joints.push(Joint::revolute(Isometry3::translation(cx, cy, 0.0), Vector3::y()));
        inertia.push(thigh.clone());
        parent.push(-1); // hip attaches to the base
        let knee = joints.len();
        joints.push(Joint::revolute(Isometry3::translation(0.0, 0.0, -0.3), Vector3::y()));
        inertia.push(shank.clone());
        parent.push(hip as isize);
        contacts.push((knee, Vector3::new(0.0, 0.0, -0.3), 0.9)); // foot = shank tip
    }
    (joints, inertia, parent, contacts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_full;

    // an arm whose links extend up/out (nothing dangles below the base), so the only floor contact
    // is the four corner feet on the base itself — a stable stance that settles cleanly.
    const UPARM: &str = r#"<robot name="up">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="0.6"/><inertia ixx="0.01" iyy="0.01" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="0.4"/><inertia ixx="0.006" iyy="0.006" izz="0.003" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="tip"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.15" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="20" velocity="5"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="20" velocity="5"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tip"/><origin xyz="0 0 0.2" rpy="0 0 0"/></joint></robot>"#;

    /// A heavy floating base on four corner feet, released just above the ground, settles into a
    /// stable stance: bounded foot penetration, comes to rest, base stays near its resting height,
    /// no NaN. Validates the contact + SE(3) integration physics — the trusted CPU reference.
    #[test]
    fn floating_base_drops_and_settles() {
        let (robot, inertia) = from_urdf_full(UPARM, "base", "tip").unwrap();
        let n = robot.dof();
        let base_inertia = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.06, 0.06, 0.08)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor_z, kn, kd) = (0.0, 2.0e4, 150.0);
        let dt = 2e-4;
        // four feet at the base corners, 0.06 m below the base origin
        let hx = 0.12;
        let contacts = vec![
            (0, Vector3::new(hx, hx, -0.06), 0.9),
            (0, Vector3::new(-hx, hx, -0.06), 0.9),
            (0, Vector3::new(hx, -hx, -0.06), 0.9),
            (0, Vector3::new(-hx, -hx, -0.06), 0.9),
        ];

        let mut base = Isometry3::translation(0.0, 0.0, 0.10); // feet ~0.04 above the floor → small drop
        let mut v0 = Vector6::zeros();
        let mut q = vec![0.2, -0.3];
        let mut qd = vec![0.0; n];
        let tau = vec![0.0; n];

        let mut min_pen = 0.0f64;
        for _ in 0..6000 {
            let (b, v, qn, qdn) = floating_contact_step(&robot, &inertia, &base_inertia, base, v0, &q, &qd, &tau, &contacts, floor_z, kn, kd, dt, g);
            base = b; v0 = v; q = qn; qd = qdn;
            for &(fr, off, _) in &contacts {
                let p = (base * robot.frame_pose(&q, fr) * Point3::from(off)).coords;
                min_pen = min_pen.min(p.z - floor_z);
            }
        }
        let base_speed = v0.norm();
        let joint_speed = qd.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        eprintln!("floating base settle: base z {:.4}, worst foot penetration {:.4} m, base speed {:.4}, joint speed {:.4}", base.translation.z, min_pen, base_speed, joint_speed);
        assert!(base.translation.vector.iter().all(|v| v.is_finite()) && v0.iter().all(|v| v.is_finite()), "sim blew up (NaN/inf)");
        assert!(min_pen > -0.03, "feet sank through the floor: {min_pen} m");
        assert!(base.translation.z > 0.03 && base.translation.z < 0.12, "base did not rest near its stance height: z {}", base.translation.z);
        // the base settles on its feet; the unactuated frictionless arm keeps swinging (a passive
        // pendulum never comes to rest), so require the base at rest and the joints merely bounded.
        assert!(base_speed < 0.15, "base did not settle to rest: {base_speed}");
        assert!(joint_speed < 10.0, "joint dynamics unstable (should be a bounded pendulum swing): {joint_speed}");
    }

    /// A QUADRUPED (torso + four 2-joint legs) dropped onto the ground settles into a stable
    /// standing stance — base upright and near stance height, feet not sinking, coming to rest.
    /// A single leg toppled; four legs are statically stable. Validates the tree contact step.
    #[test]
    fn quadruped_stands_stably() {
        let (joints, inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base_inertia = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor_z, kn, kd, dt) = (0.0, 1.5e4, 120.0, 2e-4);
        // legs straight down (q=0) reach 0.6 m; start the base at 0.62 so the feet just touch
        let mut base = Isometry3::translation(0.0, 0.0, 0.62);
        let mut v0 = Vector6::zeros();
        let mut q = vec![0.0; n];
        let mut qd = vec![0.0; n];
        let tau = vec![0.0; n];

        let mut min_pen = 0.0f64;
        for _ in 0..6000 {
            let (b, v, qn, qdn) = tree_floating_contact_step(&joints, &inertia, &parent, &base_inertia, base, v0, &q, &qd, &tau, &contacts, floor_z, kn, kd, dt, g);
            base = b; v0 = v; q = qn; qd = qdn;
            for &(body, off, _) in &contacts {
                let ft = base * frame_from_tree(&joints, &parent, &q, body) * Point3::from(off);
                min_pen = min_pen.min(ft.coords.z - floor_z);
            }
        }
        let base_speed = v0.norm();
        let up = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
        eprintln!("quadruped stance: base z {:.4}, up-alignment {:.4}, worst foot penetration {:.4} m, base speed {:.4}", base.translation.z, up, min_pen, base_speed);
        assert!(base.translation.vector.iter().all(|v| v.is_finite()), "sim blew up");
        assert!(up > 0.98, "torso did not stay upright: up {up}");
        assert!(base.translation.z > 0.5 && base.translation.z < 0.63, "base not at stance height: {}", base.translation.z);
        assert!(min_pen > -0.03, "feet sank through the floor: {min_pen}");
        assert!(base_speed < 0.1, "quadruped did not settle: {base_speed}");
    }

    /// base→body pose in a tree (compose along the parent chain).
    fn frame_from_tree(joints: &[Joint], parent: &[isize], q: &[f64], body: usize) -> Isometry3<f64> {
        let mut chain = vec![body];
        let mut cur = parent[body];
        while cur >= 0 {
            chain.push(cur as usize);
            cur = parent[cur as usize];
        }
        let mut t = Isometry3::identity();
        for &i in chain.iter().rev() {
            t *= joints[i].transform(q[i]);
        }
        t
    }
}
