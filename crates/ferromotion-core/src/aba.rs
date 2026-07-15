//! Featherstone's **Articulated-Body Algorithm** — O(n) forward dynamics, a clean-room implementation
//! from *Rigid Body Dynamics Algorithms* (Featherstone, Ch. 7) in spatial (6D Plücker) notation.
//! Where [`crate::forward_dynamics`] forms and inverts the mass matrix (O(n³)), ABA computes joint
//! accelerations `q̈` from torques directly in three recursions — the standard efficient method, and
//! the foundation for floating-base and large-DoF systems. Verified bit-for-bit against the
//! mass-matrix solve. Pure `nalgebra` → WASM-clean.

use crate::{JointKind, LinkInertia, Robot};
use nalgebra::{Matrix3, Matrix6, Vector3, Vector6};
#[cfg(test)]
use {crate::Iso, nalgebra::Translation3, nalgebra::UnitQuaternion};

/// 3×3 skew-symmetric (cross-product) matrix of `v`.
pub(crate) fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Build a 6×6 from four 3×3 blocks (spatial ordering `[angular; linear]`).
pub(crate) fn block6(tl: Matrix3<f64>, tr: Matrix3<f64>, bl: Matrix3<f64>, br: Matrix3<f64>) -> Matrix6<f64> {
    let mut m = Matrix6::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&tl);
    m.fixed_view_mut::<3, 3>(0, 3).copy_from(&tr);
    m.fixed_view_mut::<3, 3>(3, 0).copy_from(&bl);
    m.fixed_view_mut::<3, 3>(3, 3).copy_from(&br);
    m
}

/// Spatial motion transform `ⁱX_{parent}`: `R` is the child→parent rotation, `p` the child origin in
/// the parent frame. Motion transforms as `v_child = X·v_parent`; forces as `f_parent = Xᵀ·f_child`.
pub(crate) fn motion_transform(r: Matrix3<f64>, p: Vector3<f64>) -> Matrix6<f64> {
    let e = r.transpose(); // parent → child
    block6(e, Matrix3::zeros(), -e * skew(p), e)
}

/// Spatial inertia (6×6) of a link about its frame origin, from mass, COM `c`, and COM-inertia `ic`.
fn spatial_inertia(li: &LinkInertia) -> Matrix6<f64> {
    let (m, c) = (li.mass, li.com);
    let cx = skew(c);
    block6(li.inertia - m * cx * cx, m * cx, -m * cx, m * Matrix3::identity())
}

/// Spatial motion cross product `v ×` (acts on motion vectors).
pub(crate) fn crm(v: Vector6<f64>) -> Matrix6<f64> {
    let (w, vl) = (v.fixed_rows::<3>(0).into_owned(), v.fixed_rows::<3>(3).into_owned());
    block6(skew(w), Matrix3::zeros(), skew(vl), skew(w))
}

/// Spatial force cross product `v ×*` (acts on force vectors) `= −(v×)ᵀ`.
pub(crate) fn crf(v: Vector6<f64>) -> Matrix6<f64> {
    -crm(v).transpose()
}

/// Joint motion subspace `S` for joint `i` (revolute → `[axis; 0]`, prismatic → `[0; axis]`).
pub(crate) fn motion_subspace(kind: JointKind, axis: Vector3<f64>) -> Vector6<f64> {
    let mut s = Vector6::zeros();
    match kind {
        JointKind::Revolute => s.fixed_rows_mut::<3>(0).copy_from(&axis),
        JointKind::Prismatic => s.fixed_rows_mut::<3>(3).copy_from(&axis),
    }
    s
}

/// Forward dynamics `q̈` via the Articulated-Body Algorithm — same signature and result as
/// [`crate::forward_dynamics`], but O(n).
pub fn forward_dynamics_aba(robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64], tau: &[f64], gravity: Vector3<f64>) -> Vec<f64> {
    let n = robot.dof();
    // Base spatial acceleration folds in gravity (linear part = −gravity), as RNEA does.
    let mut a0 = Vector6::zeros();
    a0.fixed_rows_mut::<3>(3).copy_from(&(-gravity));

    let (mut xm, mut s) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut v, mut c) = (vec![Vector6::zeros(); n], vec![Vector6::zeros(); n]);
    let (mut ia, mut pa) = (Vec::with_capacity(n), vec![Vector6::zeros(); n]);

    // Pass 1 (outward): transforms, velocities, velocity-product terms, articulated-inertia seeds.
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        let r = *a.rotation.to_rotation_matrix().matrix();
        let x = motion_transform(r, a.translation.vector);
        let si = motion_subspace(robot.joints[i].kind, robot.joints[i].axis.into_inner());
        let v_parent = if i == 0 { Vector6::zeros() } else { v[i - 1] };
        v[i] = x * v_parent + si * qd[i];
        c[i] = crm(v[i]) * (si * qd[i]);
        ia.push(spatial_inertia(&inertia[i]));
        pa[i] = crf(v[i]) * (spatial_inertia(&inertia[i]) * v[i]);
        xm.push(x);
        s.push(si);
    }

    // Pass 2 (inward): articulated inertias/bias, propagated to the parent.
    let (mut u, mut d, mut uu) = (vec![Vector6::zeros(); n], vec![0.0; n], vec![0.0; n]);
    for i in (0..n).rev() {
        u[i] = ia[i] * s[i];
        d[i] = s[i].dot(&u[i]);
        uu[i] = tau[i] - s[i].dot(&pa[i]);
        if i > 0 {
            let ia_bar = ia[i] - u[i] * u[i].transpose() / d[i];
            let pa_bar = pa[i] + ia_bar * c[i] + u[i] * (uu[i] / d[i]);
            let xt = xm[i].transpose();
            ia[i - 1] += xt * ia_bar * xm[i];
            pa[i - 1] += xt * pa_bar;
        }
    }

    // Pass 3 (outward): accelerations and joint accelerations.
    let mut qdd = vec![0.0; n];
    let mut a = vec![Vector6::zeros(); n];
    for i in 0..n {
        let a_parent = if i == 0 { a0 } else { a[i - 1] };
        let a_prime = xm[i] * a_parent + c[i];
        qdd[i] = (uu[i] - u[i].dot(&a_prime)) / d[i];
        a[i] = a_prime + s[i] * qdd[i];
    }
    qdd
}

/// Gravity as a spatial force on a body: the wrench that would produce the gravitational spatial
/// acceleration `[0; g_frame]`, i.e. `I·a_g`. `r_to_frame` rotates a base-frame vector into this frame.
fn gravity_wrench( inertia_sp: &Matrix6<f64>, gravity: Vector3<f64>, r_to_frame: &Matrix3<f64>) -> Vector6<f64> {
    let mut a_g = Vector6::zeros();
    a_g.fixed_rows_mut::<3>(3).copy_from(&(r_to_frame * gravity));
    inertia_sp * a_g
}

/// **Floating-base forward dynamics** via ABA (Featherstone Ch. 9): a free 6-DoF root plus the serial
/// chain. Given the base spatial velocity `v0` (in the base frame), joint state, and torques, returns
/// `(a0, q̈)` — the base spatial acceleration and joint accelerations. Gravity enters as a per-body
/// external wrench (a uniform field), so it works for the free base too. This is the gate to
/// humanoid/quadruped dynamics, where the base is unactuated and floats.
pub fn floating_base_forward_dynamics(robot: &Robot, inertia: &[LinkInertia], base_inertia: &LinkInertia, v0: Vector6<f64>, q: &[f64], qd: &[f64], tau: &[f64], gravity: Vector3<f64>) -> (Vector6<f64>, Vec<f64>) {
    let n = robot.dof();
    let (mut xm, mut s) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut v, mut c) = (vec![Vector6::zeros(); n], vec![Vector6::zeros(); n]);
    let (mut ia, mut pa) = (Vec::with_capacity(n), vec![Vector6::zeros(); n]);

    // Base body seeds (base frame == identity rotation to itself, so gravity is expressed directly).
    let ib = spatial_inertia(base_inertia);
    let mut ia_base = ib;
    let mut pa_base = crf(v0) * (ib * v0) - gravity_wrench(&ib, gravity, &Matrix3::identity());

    // Pass 1 (outward). `r_bi` = rotation base→frame i, to express gravity in each frame.
    let mut r_parent = Matrix3::identity(); // base→parent
    let mut r_frames = Vec::with_capacity(n);
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        let r = *a.rotation.to_rotation_matrix().matrix(); // child→parent
        let x = motion_transform(r, a.translation.vector);
        let si = motion_subspace(robot.joints[i].kind, robot.joints[i].axis.into_inner());
        let v_parent = if i == 0 { v0 } else { v[i - 1] };
        v[i] = x * v_parent + si * qd[i];
        c[i] = crm(v[i]) * (si * qd[i]);
        let ii = spatial_inertia(&inertia[i]);
        let r_bi = r.transpose() * r_parent; // base→frame i
        pa[i] = crf(v[i]) * (ii * v[i]) - gravity_wrench(&ii, gravity, &r_bi);
        ia.push(ii);
        xm.push(x);
        s.push(si);
        r_frames.push(r_bi);
        r_parent = r_bi;
    }

    // Pass 2 (inward): fold each articulated inertia/bias into its parent (the base for link 0).
    let (mut u, mut d, mut uu) = (vec![Vector6::zeros(); n], vec![0.0; n], vec![0.0; n]);
    for i in (0..n).rev() {
        u[i] = ia[i] * s[i];
        d[i] = s[i].dot(&u[i]);
        uu[i] = tau[i] - s[i].dot(&pa[i]);
        let ia_bar = ia[i] - u[i] * u[i].transpose() / d[i];
        let pa_bar = pa[i] + ia_bar * c[i] + u[i] * (uu[i] / d[i]);
        let xt = xm[i].transpose();
        if i > 0 {
            ia[i - 1] += xt * ia_bar * xm[i];
            pa[i - 1] += xt * pa_bar;
        } else {
            ia_base += xt * ia_bar * xm[0];
            pa_base += xt * pa_bar;
        }
    }

    // Base equation: I₀ᴬ·a₀ + p₀ᴬ = 0  (no external wrench on the free base) → a₀ = −(I₀ᴬ)⁻¹ p₀ᴬ.
    let a0 = -ia_base.try_inverse().expect("base articulated inertia invertible") * pa_base;

    // Pass 3 (outward): joint accelerations.
    let mut qdd = vec![0.0; n];
    let mut a = vec![Vector6::zeros(); n];
    for i in 0..n {
        let a_parent = if i == 0 { a0 } else { a[i - 1] };
        let a_prime = xm[i] * a_parent + c[i];
        qdd[i] = (uu[i] - u[i].dot(&a_prime)) / d[i];
        a[i] = a_prime + s[i] * qdd[i];
    }
    (a0, qdd)
}

/// Total spatial momentum of a floating-base system, expressed in the world frame — for checking
/// conservation. `t0` is `world_from_base`.
#[cfg(test)]
fn world_momentum(robot: &Robot, inertia: &[LinkInertia], base_inertia: &LinkInertia, t0: &Iso, v0: Vector6<f64>, q: &[f64], qd: &[f64]) -> Vector6<f64> {
    let n = robot.dof();
    let mut h_base = spatial_inertia(base_inertia) * v0; // base frame
    let mut x0i = Matrix6::identity(); // base→frame i motion transform
    let mut v_parent = v0;
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        let r = *a.rotation.to_rotation_matrix().matrix();
        let x = motion_transform(r, a.translation.vector);
        let si = motion_subspace(robot.joints[i].kind, robot.joints[i].axis.into_inner());
        let vi = x * v_parent + si * qd[i];
        x0i = x * x0i;
        h_base += x0i.transpose() * (spatial_inertia(&inertia[i]) * vi); // map momentum i→base
        v_parent = vi;
    }
    // Base→world: forces map by (bXw)ᵀ, bXw = motion_transform(base→world rot, base origin in world).
    let bxw = motion_transform(*t0.rotation.to_rotation_matrix().matrix(), t0.translation.vector);
    bxw.transpose() * h_base
}

/// SE(3) exponential of a body-frame twist `xi·dt` (`[ω; v]`), for integrating the base pose.
#[cfg(test)]
fn exp6(xi: Vector6<f64>, dt: f64) -> Iso {
    let w = xi.fixed_rows::<3>(0).into_owned() * dt;
    let vv = xi.fixed_rows::<3>(3).into_owned() * dt;
    let theta = w.norm();
    let rot = if theta < 1e-12 { UnitQuaternion::identity() } else { UnitQuaternion::from_scaled_axis(w) };
    let trans = if theta < 1e-9 {
        vv
    } else {
        let wx = skew(w);
        let a = (1.0 - theta.cos()) / (theta * theta);
        let b = (theta - theta.sin()) / theta.powi(3);
        (Matrix3::identity() + a * wx + b * wx * wx) * vv
    };
    Iso::from_parts(Translation3::from(trans), rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{forward_dynamics, from_urdf_full};

    fn base_body() -> LinkInertia {
        LinkInertia { mass: 3.0, com: Vector3::new(0.05, 0.0, 0.0), inertia: Matrix3::from_diagonal(&Vector3::new(0.05, 0.06, 0.07)) }
    }

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    // A 3-DoF chain mixing revolute axes + a prismatic joint (out-of-plane) to exercise the algebra.
    const CHAIN3: &str = r#"<robot name="c3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.2 0.1 0" rpy="0 0 0"/><mass value="1.2"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0" iyy="0.03" iyz="0" izz="0.025"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.15 0 0.05" rpy="0 0 0"/><mass value="0.8"/>
        <inertia ixx="0.01" ixy="0" ixz="0.001" iyy="0.012" iyz="0" izz="0.011"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.1 0 0" rpy="0 0 0"/><mass value="0.5"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.4 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/><origin xyz="0.3 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/><limit lower="-0.2" upper="0.2" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.2 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    #[test]
    fn aba_matches_the_mass_matrix_solve() {
        let g = Vector3::new(0.0, 0.0, -9.81);
        for (urdf, q, qd, tau) in [
            (ARM2, vec![0.3, -0.7], vec![0.5, -0.2], vec![0.4, 0.1]),
            (CHAIN3, vec![0.4, -0.6, 0.05], vec![-0.3, 0.7, 0.2], vec![0.2, -0.5, 0.3]),
        ] {
            let (robot, inertia) = from_urdf_full(urdf, "base", "tool").unwrap();
            let ref_qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            let aba = forward_dynamics_aba(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..robot.dof() {
                assert!((aba[i] - ref_qdd[i]).abs() < 1e-9, "ABA[{i}]={} vs oracle {}", aba[i], ref_qdd[i]);
            }
        }
    }

    #[test]
    fn floating_base_free_falls_at_g() {
        // At rest with gravity, a free-floating system's base accelerates at exactly g (no rotation).
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (a0, qdd) = floating_base_forward_dynamics(&robot, &inertia, &base_body(), Vector6::zeros(), &[0.2, -0.4], &[0.0, 0.0], &[0.0, 0.0], g);
        assert!(a0.fixed_rows::<3>(0).norm() < 1e-9, "base should not angularly accelerate: {}", a0.fixed_rows::<3>(0).norm());
        assert!((a0.fixed_rows::<3>(3).into_owned() - g).norm() < 1e-9, "base linear accel {:?} ≠ g", a0.fixed_rows::<3>(3));
        // Joints, rigidly free-falling with the base, need no relative acceleration.
        assert!(qdd.iter().all(|&x| x.abs() < 1e-9), "joints should not accelerate in free fall: {qdd:?}");
    }

    #[test]
    fn floating_base_conserves_momentum() {
        // Gravity off, zero torque, internal motion: total linear + angular momentum is conserved.
        let (robot, inertia) = from_urdf_full(CHAIN3, "base", "tool").unwrap();
        let base = base_body();
        let g = Vector3::zeros();
        let mut t0 = Iso::identity();
        let mut v0 = Vector6::from_row_slice(&[0.3, -0.2, 0.4, 0.1, 0.2, -0.1]);
        let (mut q, mut qd) = (vec![0.4, -0.6, 0.05], vec![-0.5, 0.8, 0.3]);

        let h0 = world_momentum(&robot, &inertia, &base, &t0, v0, &q, &qd);
        let (dt, steps) = (1e-4, 200);
        for _ in 0..steps {
            let (a0, qdd) = floating_base_forward_dynamics(&robot, &inertia, &base, v0, &q, &qd, &[0.0, 0.0, 0.0], g);
            v0 += a0 * dt; // semi-implicit: advance velocity, then pose with the new velocity
            for k in 0..robot.dof() {
                qd[k] += qdd[k] * dt;
                q[k] += qd[k] * dt;
            }
            t0 *= exp6(v0, dt);
        }
        let hf = world_momentum(&robot, &inertia, &base, &t0, v0, &q, &qd);
        let drift = (hf - h0).norm() / h0.norm();
        eprintln!("momentum drift over {steps} steps: {drift:.2e}  (h0={:.4}, hf={:.4})", h0.norm(), hf.norm());
        assert!(drift < 1e-3, "floating-base momentum not conserved: relative drift {drift:.2e}");
    }
}
