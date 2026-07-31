//! **Branched-tree floating-base dynamics** — the generalization of the serial floating-base ABA
//! ([`floating_base_forward_dynamics`](crate::floating_base_forward_dynamics)) to a *kinematic tree*:
//! a free 6-DoF base with any number of limbs branching off it (a quadruped's four legs, a biped's
//! two legs + two arms). Each joint carries a parent index `λ(i)` (`-1` = the base) instead of an
//! implicit predecessor; the three ABA recursions traverse the tree instead of a chain. When the tree
//! *is* a chain (`λ(i) = i−1`) this reduces exactly to the serial version — the check it is verified
//! against. This is the gate to real legged locomotion. Featherstone Ch. 7. Pure `nalgebra` → WASM-clean.

use crate::aba::{crf, crm, gravity_wrench, motion_subspace, motion_transform, spatial_inertia};
use crate::{Joint, LinkInertia};
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};

/// Floating-base forward dynamics for a kinematic tree. `joints[i]` connects body `i` to its parent
/// `parent[i]` (`-1` = the free base); joints must be in topological order (`parent[i] < i`).
/// `inertia[i]` is body `i`'s inertia, `base_inertia` the base's. `v0` is the base spatial velocity in
/// the base frame; `f_ext_base` / `f_ext[i]` are external spatial wrenches (`[torque; force]`) on the
/// base / each body in that body's frame. Returns `(a0, q̈)` — the base spatial acceleration and joint
/// accelerations.
#[allow(clippy::too_many_arguments)]
pub fn tree_floating_forward_dynamics(
    joints: &[Joint],
    inertia: &[LinkInertia],
    parent: &[isize],
    base_inertia: &LinkInertia,
    v0: Vector6<f64>,
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    f_ext_base: Vector6<f64>,
    f_ext: &[Vector6<f64>],
    gravity: Vector3<f64>,
) -> (Vector6<f64>, Vec<f64>) {
    let n = joints.len();
    let (mut xm, mut s) = (vec![Matrix6::zeros(); n], vec![Vector6::zeros(); n]);
    let (mut v, mut c) = (vec![Vector6::zeros(); n], vec![Vector6::zeros(); n]);
    let (mut ia, mut pa) = (Vec::with_capacity(n), vec![Vector6::zeros(); n]);
    let mut r_frames = vec![Matrix3::<f64>::identity(); n]; // base → frame i rotation

    // Pass 1 (outward, topological): transforms, velocities, articulated-inertia seeds, biases.
    for i in 0..n {
        let a = joints[i].transform(q[i]);
        let r = *a.rotation.to_rotation_matrix().matrix(); // child → parent
        let x = motion_transform(r, a.translation.vector);
        let si = motion_subspace(joints[i].kind, joints[i].axis.into_inner());
        let v_parent = if parent[i] < 0 { v0 } else { v[parent[i] as usize] };
        v[i] = x * v_parent + si * qd[i];
        c[i] = crm(v[i]) * (si * qd[i]);
        let ii = spatial_inertia(&inertia[i]);
        let r_bparent = if parent[i] < 0 { Matrix3::identity() } else { r_frames[parent[i] as usize] };
        let r_bi = r.transpose() * r_bparent; // base → frame i
        pa[i] = crf(v[i]) * (ii * v[i]) - gravity_wrench(&ii, gravity, &r_bi) - f_ext[i];
        ia.push(ii);
        xm[i] = x;
        s[i] = si;
        r_frames[i] = r_bi;
    }

    let ib = spatial_inertia(base_inertia);
    let mut ia_base = ib;
    let mut pa_base = crf(v0) * (ib * v0) - gravity_wrench(&ib, gravity, &Matrix3::identity()) - f_ext_base;

    // Pass 2 (inward): fold each articulated inertia / bias into its parent (or the base).
    let (mut u, mut d, mut uu) = (vec![Vector6::zeros(); n], vec![0.0; n], vec![0.0; n]);
    for i in (0..n).rev() {
        u[i] = ia[i] * s[i];
        d[i] = s[i].dot(&u[i]);
        uu[i] = tau[i] - s[i].dot(&pa[i]);
        let ia_bar = ia[i] - u[i] * u[i].transpose() / d[i];
        let pa_bar = pa[i] + ia_bar * c[i] + u[i] * (uu[i] / d[i]);
        let xt = xm[i].transpose();
        if parent[i] < 0 {
            ia_base += xt * ia_bar * xm[i];
            pa_base += xt * pa_bar;
        } else {
            let p = parent[i] as usize;
            ia[p] += xt * ia_bar * xm[i];
            pa[p] += xt * pa_bar;
        }
    }

    let a0 = -ia_base.try_inverse().expect("base articulated inertia invertible") * pa_base;

    // Pass 3 (outward): joint accelerations.
    let mut qdd = vec![0.0; n];
    let mut a = vec![Vector6::zeros(); n];
    for i in 0..n {
        let a_parent = if parent[i] < 0 { a0 } else { a[parent[i] as usize] };
        let a_prime = xm[i] * a_parent + c[i];
        qdd[i] = (uu[i] - u[i].dot(&a_prime)) / d[i];
        a[i] = a_prime + s[i] * qdd[i];
    }
    (a0, qdd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{floating_base_forward_dynamics, from_urdf_full};

    // a serial 3-link arm with inertials
    const ARM: &str = r#"<robot name="a3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0 0 0.1" rpy="0 0 0"/><mass value="1.5"/><inertia ixx="0.02" iyy="0.02" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.15 0 0" rpy="0 0 0"/><mass value="1.0"/><inertia ixx="0.01" iyy="0.03" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.1 0 0" rpy="0 0 0"/><mass value="0.6"/><inertia ixx="0.005" iyy="0.012" izz="0.012" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.2 0 0" rpy="0 0 0"/></joint></robot>"#;

    fn base_body() -> LinkInertia {
        LinkInertia { mass: 5.0, com: Vector3::new(0.0, 0.0, 0.05), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.05)) }
    }

    /// When the tree is a chain (`λ(i) = i−1`), the branched ABA must equal the serial floating-base
    /// ABA exactly (to f64) — verified against the trusted oracle over a random state.
    #[test]
    fn tree_reduces_to_serial_chain() {
        let (robot, inertia) = from_urdf_full(ARM, "base", "tool").unwrap();
        let n = robot.dof();
        let base = base_body();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let parent: Vec<isize> = (0..n).map(|i| i as isize - 1).collect(); // serial chain
        let zero = vec![Vector6::zeros(); n];

        let mut s = 0x77u64;
        let mut rng = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let v0 = Vector6::from_iterator((0..6).map(|_| 0.5 * rng()));
        let q: Vec<f64> = (0..n).map(|_| 1.3 * rng()).collect();
        let qd: Vec<f64> = (0..n).map(|_| 0.8 * rng()).collect();
        let tau: Vec<f64> = (0..n).map(|_| 1.5 * rng()).collect();

        let (a0_t, qdd_t) = tree_floating_forward_dynamics(&robot.joints, &inertia, &parent, &base, v0, &q, &qd, &tau, Vector6::zeros(), &zero, g);
        let (a0_s, qdd_s) = floating_base_forward_dynamics(&robot, &inertia, &base, v0, &q, &qd, &tau, g);

        let da0 = (a0_t - a0_s).amax();
        let dqdd = qdd_t.iter().zip(&qdd_s).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("tree vs serial (chain): |Δa0| {da0:.3e}, |Δqdd| {dqdd:.3e}");
        assert!(da0 < 1e-12 && dqdd < 1e-12, "branched ABA did not reduce to serial: {da0}, {dqdd}");
    }

    /// A genuinely BRANCHED free body (base with two identical legs) under gravity, at rest, no torque:
    /// the whole thing free-falls, so the base linear acceleration equals gravity and the joints do not
    /// accelerate (a uniform field costs no internal motion). A physical invariant for the tree path.
    #[test]
    fn branched_free_body_falls_at_g() {
        // two 2-link legs off the base: joints 0,1 (leg A) and 2,3 (leg B)
        let (arm, ai) = from_urdf_full(ARM, "base", "tool").unwrap();
        let leg: Vec<Joint> = arm.joints[0..2].to_vec();
        let leg_in: Vec<LinkInertia> = ai[0..2].to_vec();
        let joints: Vec<Joint> = leg.iter().chain(leg.iter()).cloned().collect(); // 4 joints
        let inertia: Vec<LinkInertia> = leg_in.iter().chain(leg_in.iter()).cloned().collect();
        let parent: Vec<isize> = vec![-1, 0, -1, 2]; // legA: base→0→1 ; legB: base→2→3
        let base = base_body();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let n = 4;
        let zero = vec![Vector6::zeros(); n];

        let (a0, qdd) = tree_floating_forward_dynamics(&joints, &inertia, &parent, &base, Vector6::zeros(), &vec![0.3, -0.5, 0.3, -0.5], &vec![0.0; n], &vec![0.0; n], Vector6::zeros(), &zero, g);
        let lin = a0.fixed_rows::<3>(3).into_owned();
        let worst_qdd = qdd.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        eprintln!("branched free-fall: base lin accel {lin:?} (want {g:?}), worst qdd {worst_qdd:.3e}");
        assert!((lin - g).amax() < 1e-9, "branched free body did not free-fall at g: {lin:?}");
        assert!(worst_qdd < 1e-9, "branched free body should have zero joint acceleration: {worst_qdd}");
    }
}
