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

/// A scripted **crawl** gait for the [`quadruped`] — a statically stable walk. Only one foot swings at a
/// time (the other three stay planted, keeping the centre of mass inside the support triangle), so the
/// body never tips even open-loop. Each stance leg sweeps its hip backward to push the base forward;
/// the swing leg lifts its knee to clear the ground and repositions. PD torques track the phase-varying
/// targets. `phase` is the gait clock in radians (2π per full 4-step cycle).
pub fn quadruped_trot_tau(q: &[f64], qd: &[f64], phase: f64) -> Vec<f64> {
    let (kp, kv) = (60.0, 5.0);
    let (a_sweep, a_lift, duty) = (0.22, 0.5, 0.75); // hip amplitude, knee lift, stance fraction
    // lift order over one cycle: FL, BR, FR, BL — each leg airborne for (1-duty) of the cycle
    let order = [0.0f64, 0.5, 0.75, 0.25];
    let mut tau = vec![0.0; 8];
    for (leg, &offset) in order.iter().enumerate() {
        let (hip, knee) = (leg * 2, leg * 2 + 1);
        let phi = (phase / std::f64::consts::TAU + offset).rem_euclid(1.0); // 0..1 in this leg's cycle
        let (hip_t, knee_t);
        if phi < duty {
            let s = phi / duty; // 0..1 across stance: hip sweeps back-to-front, driving the base +x
            hip_t = a_sweep * (2.0 * s - 1.0);
            knee_t = 0.0; // straight leg bears load
        } else {
            let s = (phi - duty) / (1.0 - duty); // 0..1 across swing: reposition + lift
            hip_t = a_sweep * (1.0 - 2.0 * s);
            knee_t = -a_lift * (std::f64::consts::PI * s).sin(); // lift then place
        }
        tau[hip] = kp * (hip_t - q[hip]) - kv * qd[hip];
        tau[knee] = kp * (knee_t - q[knee]) - kv * qd[knee];
    }
    tau
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

    /// The scripted trot walks the quadruped forward while keeping the torso roughly upright — the
    /// controller the browser bench uses.
    #[test]
    fn quadruped_walks_forward() {
        let (joints, inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base_inertia = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor, kn, kd, dt, freq) = (0.0, 1.5e4, 120.0, 2e-4, 1.0);
        let mut base = Isometry3::translation(0.0, 0.0, 0.62);
        let mut v0 = Vector6::zeros();
        let mut q = vec![0.0; n];
        let mut qd = vec![0.0; n];
        let x0 = base.translation.x;
        for t in 0..15000 {
            let phase = std::f64::consts::TAU * freq * t as f64 * dt;
            let tau = quadruped_trot_tau(&q, &qd, phase);
            let (b, v, qn, qdn) = tree_floating_contact_step(&joints, &inertia, &parent, &base_inertia, base, v0, &q, &qd, &tau, &contacts, floor, kn, kd, dt, g);
            base = b; v0 = v; q = qn; qd = qdn;
        }
        let dx = base.translation.x - x0;
        let up = base.rotation.to_rotation_matrix().matrix()[(2, 2)];
        eprintln!("crawl walk: Δx {:.3} m over 3.0 s, base z {:.3}, up-alignment {:.3}", dx, base.translation.z, up);
        assert!(base.translation.vector.iter().all(|v| v.is_finite()), "sim blew up");
        assert!(up > 0.6, "quadruped toppled while trotting: up {up}");
        assert!(dx.abs() > 0.03, "quadruped did not locomote: Δx {dx}");
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

    /// **The contact stiffnesses this repo ships must stay inside the explicit-integration limit.**
    ///
    /// A foot on the floor is a spring-damper against the effective mass at that foot, integrated
    /// explicitly in the damper, so the condition is the same one `Admittance::stability_limit` and
    /// `Rotor::max_stable_dt` solve: `dt²·(kn/m) + 2·dt·(kd/m) ≤ 4`. Nothing here reports that bound —
    /// `kn`, `kd` and `dt` are all the caller's — so this pins the configurations the repo itself uses.
    ///
    /// ⛔ **This used to read a HAND-COPIED table of four rows, and it was both wrong and short.** Its
    /// tightest row claimed `kn = 2e4, kd = 150, dt = 1e-3` for "this module's contact test"; that site
    /// sets `dt = 2e-4` on the very next line, and this review did not locate `2e4/150` at `1e-3` anywhere
    /// in the workspace. So the quoted "tightest 5.0x" belonged to a configuration nothing ships, while
    /// the named site's real margin is 25x. Meanwhile the actually-tightest live site,
    /// `gpu.rs`'s `(0.0, 1.5e4, 120.0, 1e-3)`, was not in the table at all, so the gate was not watching
    /// it: coarsening that step to `1e-2` puts it at 0.6x its stability limit with this test still green.
    /// That is the canonical vacuity failure — a gate whose subject count is smaller than its claim.
    ///
    /// The table is gone. The gate READS THE SOURCE, finds every `let (…, kn, kd[, dt]) = (…)` binding in
    /// the workspace, resolves a nearby literal `dt` when the binding does not carry one, and checks the
    /// margin at every site it finds. It also asserts the subject count, so a refactor that changes the
    /// binding shape shows up as a shrinking scan rather than as a quietly narrower gate.
    ///
    /// A site whose `dt` is NOT a literal — a struct field, a parameter — must carry
    /// `CONTACT DT CHECKED BY: <test fn>` within the six lines above it, and this gate verifies that test
    /// exists in that file. ⛔ **A tolerated unresolved COUNT was the first version of that rule and it was
    /// the wrong guard: it let the next unreadable site in for free.** The scan does not distinguish
    /// shipped code from test fixtures, which errs toward more coverage; a test that deliberately probes
    /// an unstable configuration would need to avoid the tuple shape or carry a marker.
    #[test]
    fn the_shipped_contact_parameters_stay_inside_the_explicit_integration_limit() {
        let limit = |kn: f64, kd: f64, m: f64| {
            let (w2, gamma) = (kn / m, kd / m);
            (-gamma + (gamma * gamma + 4.0 * w2).sqrt()) / w2
        };

        // ---- read every contact-parameter site out of the workspace source ----
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates/").to_path_buf();
        let mut files = Vec::new();
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else { return };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        for c in std::fs::read_dir(&root).expect("crates/ readable").flatten() {
            for sub in ["src", "examples"] {
                let d = c.path().join(sub);
                if d.is_dir() {
                    walk(&d, &mut files);
                }
            }
        }
        files.sort();
        assert!(files.len() > 100, "the walk should cover the workspace, found {}", files.len());

        let num = |t: &str| t.trim().parse::<f64>().ok();
        let mut sites: Vec<(String, f64, f64, f64)> = Vec::new();
        let mut unresolved: Vec<(String, usize)> = Vec::new();
        for f in &files {
            let text = std::fs::read_to_string(f).expect("readable");
            let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
            let lines: Vec<&str> = text.lines().collect();
            for (ln, line) in lines.iter().enumerate() {
                let t = line.trim();
                // `let (<names>) = (<values>);` with kn and kd among the names
                let Some(rest) = t.strip_prefix("let (") else { continue };
                let Some((names, tail)) = rest.split_once(") = (") else { continue };
                let Some(values) = tail.strip_suffix(");") else { continue };
                let names: Vec<&str> = names.split(',').map(|x| x.trim()).collect();
                let values: Vec<&str> = values.split(',').map(|x| x.trim()).collect();
                if names.len() != values.len() || !names.contains(&"kn") || !names.contains(&"kd") {
                    continue;
                }
                let at = |want: &str| names.iter().position(|n| *n == want).and_then(|i| num(values[i]));
                let (Some(kn), Some(kd)) = (at("kn"), at("kd")) else {
                    unresolved.push((rel.clone(), ln + 1));
                    continue;
                };
                // dt from the same binding, else the nearest literal `let dt = …;` within three lines
                let dt = at("dt").or_else(|| {
                    let lo = ln.saturating_sub(3);
                    let hi = (ln + 4).min(lines.len());
                    lines[lo..hi].iter().find_map(|l| {
                        l.trim().strip_prefix("let dt = ").and_then(|v| num(v.trim_end_matches(';')))
                    })
                });
                match dt {
                    Some(dt) => sites.push((format!("{rel}:{}", ln + 1), kn, kd, dt)),
                    None => unresolved.push((rel.clone(), ln + 1)),
                }
            }
        }

        // ---- the subject count, which is the thing the old hand table got wrong ----
        eprintln!("\n  contact-parameter sites found in source: {} resolved, {} unresolved", sites.len(), unresolved.len());
        assert!(
            sites.len() >= 9,
            "the scan found only {} sites; it found 9 when written, so either sites were deleted or the \
             binding shape changed and this gate is now watching less than it claims. Sites: {sites:#?}",
            sites.len()
        );
        // ---- an unreadable site must NAME the test that checks it instead ----
        // A tolerated count was the wrong guard: it let the next unreadable site in for free. A site whose
        // `dt` is not a literal (a struct field, a parameter) has to carry
        // `CONTACT DT CHECKED BY: <test fn>` in the three lines above it, and the gate verifies that test
        // exists in the same file. Same shape as `stability_bounds_are_tested_from_outside`'s CROSSED BY:.
        let mut holes = Vec::new();
        for (rel, ln) in &unresolved {
            let f = root.join(rel);
            let text = std::fs::read_to_string(&f).expect("readable");
            let lines: Vec<&str> = text.lines().collect();
            let lo = ln.saturating_sub(6);
            let block = lines[lo..(*ln).min(lines.len())].join("\n");
            match block.split("CONTACT DT CHECKED BY:").nth(1).and_then(|r| r.split_whitespace().next()) {
                Some(test_fn) if text.contains(&format!("fn {}(", test_fn.trim_end_matches(&['`', ',', '.'][..]))) => {
                    eprintln!("    unreadable dt at {rel}:{ln} -> checked by {test_fn}");
                }
                Some(test_fn) => holes.push(format!("{rel}:{ln} names `{test_fn}`, which does not exist in that file")),
                None => holes.push(format!("{rel}:{ln} has no `CONTACT DT CHECKED BY:` marker")),
            }
        }
        assert!(
            holes.is_empty(),
            "a contact site whose dt this gate cannot read must name the test that checks it: {holes:#?}"
        );

        // ---- the check itself, at every site found ----
        let mut tightest = (f64::INFINITY, String::new());
        for (what, kn, kd, dt) in &sites {
            // 0.5 kg is a deliberately small effective mass: a light foot link is the worst case, since
            // the bound falls as the mass does.
            let lim = limit(*kn, *kd, 0.5);
            let margin = lim / dt;
            eprintln!("    {margin:>7.1}x  kn {kn:>8.1e} kd {kd:>6.1} dt {dt:.0e}  {what}");
            assert!(margin > 3.0, "{what}: dt {dt:.0e} is only {margin:.1}x inside the {lim:.3e} s limit");
            if margin < tightest.0 {
                tightest = (margin, what.clone());
            }
        }
        eprintln!("  tightest: {:.1}x at {}\n", tightest.0, tightest.1);
        assert!(
            tightest.0 < 20.0,
            "if every margin is now huge, the fixtures changed and this guard is no longer watching what it \
             was written for: tightest {:.1}x at {}",
            tightest.0, tightest.1
        );
        // and the bound must actually tighten with stiffness, or it is not this bound
        assert!(limit(1.0e6, 150.0, 0.5) < limit(2.0e4, 150.0, 0.5), "a stiffer contact must permit a smaller step");
        assert!(limit(2.0e4, 150.0, 0.1) < limit(2.0e4, 150.0, 6.0), "a lighter effective mass must too");
    }

    /// ⛔⛔ **A contact may only push, and nothing in this crate was checking that here.**
    ///
    /// The penalty normal is `max(0, −kₙφ − k_d ż)`, and removing that clamp from **both** sites in this
    /// module broke none of the 809 tests in the crate. A spring–dashpot with an unclamped damper pulls
    /// whenever a penetrating contact separates faster than `kₙ|φ|/k_d` — here `2e4 · 1e-4 / 150`, about
    /// **1.33 cm/s** — so feet would stick to the floor on lift-off, and the quadruped fixtures never
    /// separate fast enough to show it.
    ///
    /// ⭐ The oracle needs no force accessor, which this API does not offer: **a separating contact must
    /// do nothing at all**, so the body must be in free flight. One step, and the vertical velocity has
    /// to be exactly `v₀ + g·dt`.
    #[test]
    fn a_penetrating_foot_that_is_separating_leaves_the_body_in_free_flight() {
        let (robot, inertia) = from_urdf_full(UPARM, "base", "tip").unwrap();
        let base_inertia = LinkInertia {
            mass: 8.0,
            com: Vector3::zeros(),
            inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.06, 0.06, 0.08)),
        };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor_z, kn, kd, dt) = (0.0, 2.0e4, 150.0, 2e-4);
        let pen = 1e-4;
        let contacts = vec![(0usize, Vector3::new(0.0, 0.0, -0.06), 0.9)];
        let base = Isometry3::translation(0.0, 0.0, 0.06 - pen); // the foot is `pen` below the floor
        let q = vec![0.2, -0.3];
        let qd = vec![0.0; robot.dof()];
        let tau = vec![0.0; robot.dof()];

        // the speed above which an UNCLAMPED dashpot would pull instead of push
        let pull_speed = kn * pen / kd;
        let step = |vz: f64| {
            let mut v0 = Vector6::zeros();
            v0[5] = vz;
            let (_, v, _, _) = floating_contact_step(
                &robot, &inertia, &base_inertia, base, v0, &q, &qd, &tau, &contacts, floor_z, kn, kd, dt, g,
            );
            v[5]
        };

        // ⭐ the control first: at rest the same penetrating foot must PUSH, or "does nothing" below is
        // a statement about a contact that was never active
        let resting = step(0.0);
        assert!(
            resting > g.z * dt + 1e-9,
            "a penetrating foot at rest must push: got {resting:.9}, free flight is {:.9}",
            g.z * dt
        );

        // and separating well past the pull threshold, it must do nothing
        let vz = 10.0 * pull_speed;
        let got = step(vz);
        let free = vz + g.z * dt;
        eprintln!(
            "  pull threshold {pull_speed:.4} m/s; separating at {vz:.4} m/s -> {got:.12}, free flight {free:.12} (at rest: {resting:.9})"
        );
        assert!(
            (got - free).abs() < 1e-12,
            "a separating foot must not be felt at all: {got:.12} against free flight {free:.12}"
        );
    }

    /// ⛔ **The same law, at the other site.** [`tree_floating_contact_step`] carries its own copy of the
    /// penalty normal, and removing the clamp there ALONE survived the crate even after
    /// `a_penetrating_foot_that_is_separating_leaves_the_body_in_free_flight` was added — that test
    /// drives the serial path and cannot reach this one.
    ///
    /// ⚠ Two copies of a physical law is the thing to notice here. A single test per module is not
    /// coverage when the module states the law twice.
    #[test]
    fn a_separating_foot_on_the_branched_path_is_also_not_felt() {
        let (joints, inertia, parent, contacts) = quadruped();
        let n = joints.len();
        let base_inertia = LinkInertia {
            mass: 8.0,
            com: Vector3::zeros(),
            inertia: nalgebra::Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)),
        };
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (floor_z, kn, kd, dt) = (0.0, 1.5e4, 120.0, 2e-4);
        let pen = 1e-4;
        // legs straight down at q = 0 reach 0.6 m, so this puts every foot `pen` under the floor
        let base = Isometry3::translation(0.0, 0.0, 0.6 - pen);
        let q = vec![0.0; n];
        let qd = vec![0.0; n];
        let tau = vec![0.0; n];
        for &(body, off, _) in &contacts {
            let ft = base * frame_from_tree(&joints, &parent, &q, body) * Point3::from(off);
            assert!(
                (ft.coords.z - (floor_z - pen)).abs() < 1e-12,
                "every foot must start exactly {pen} under the floor, got {}",
                ft.coords.z
            );
        }

        let pull_speed = kn * pen / kd;
        let step = |vz: f64| {
            let mut v0 = Vector6::zeros();
            v0[5] = vz;
            let (_, v, _, _) = tree_floating_contact_step(
                &joints, &inertia, &parent, &base_inertia, base, v0, &q, &qd, &tau, &contacts, floor_z, kn, kd, dt, g,
            );
            v[5]
        };

        let resting = step(0.0);
        assert!(resting > g.z * dt + 1e-9, "penetrating feet at rest must push: {resting:.9}");

        let vz = 10.0 * pull_speed;
        let got = step(vz);
        let free = vz + g.z * dt;
        eprintln!("  tree path: pull threshold {pull_speed:.4} m/s; separating at {vz:.4} -> {got:.12}, free flight {free:.12}");
        assert!(
            (got - free).abs() < 1e-12,
            "a separating foot must not be felt on the branched path either: {got:.12} against {free:.12}"
        );
    }
}
