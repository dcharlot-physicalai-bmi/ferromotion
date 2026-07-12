//! Rigid-body dynamics — a clean-room port of Pinocchio's core algorithms for our serial chain.
//! Recursive Newton-Euler (RNEA) gives inverse dynamics `τ = ID(q, q̇, q̈, g)`; the gravity vector
//! `G(q)` and the joint-space inertia (mass) matrix `M(q)` fall out as special cases. Per-link
//! inertias come from the URDF (`from_urdf_full`). Pure `nalgebra` → WASM-clean.

use crate::Robot;
use nalgebra::{DMatrix, DVector, Matrix3, Point3, Vector3};

/// Inertial parameters of a link, expressed in that link's (its joint's output) frame.
#[derive(Clone, Debug)]
pub struct LinkInertia {
    pub mass: f64,
    /// Center of mass in the link frame.
    pub com: Vector3<f64>,
    /// Inertia tensor about the COM, in link-frame orientation.
    pub inertia: Matrix3<f64>,
}

impl LinkInertia {
    pub fn zero() -> Self {
        Self { mass: 0.0, com: Vector3::zeros(), inertia: Matrix3::zeros() }
    }
}

/// Re-express an inertia given in a child frame into a parent frame via `tf` (parent_from_child).
pub(crate) fn transform_inertia(li: &LinkInertia, tf: &crate::Iso) -> LinkInertia {
    let rm = *tf.rotation.to_rotation_matrix().matrix();
    let com = (tf * Point3::from(li.com)).coords;
    LinkInertia { mass: li.mass, com, inertia: rm * li.inertia * rm.transpose() }
}

/// Composite of two inertias expressed in the same frame (parallel-axis about the combined COM).
pub(crate) fn combine_inertia(a: &LinkInertia, b: &LinkInertia) -> LinkInertia {
    let mass = a.mass + b.mass;
    if mass <= 0.0 {
        return LinkInertia::zero();
    }
    let com = (a.com * a.mass + b.com * b.mass) / mass;
    let paxis = |d: Vector3<f64>| Matrix3::identity() * d.dot(&d) - d * d.transpose();
    let inertia =
        a.inertia + a.mass * paxis(a.com - com) + b.inertia + b.mass * paxis(b.com - com);
    LinkInertia { mass, com, inertia }
}

/// Inverse dynamics via Recursive Newton-Euler: joint torques for a desired motion under `gravity`
/// (e.g. `Vector3::new(0,0,-9.81)`). `inertia[i]` is link `i`'s inertia (from `from_urdf_full`).
pub fn inverse_dynamics(
    robot: &Robot,
    inertia: &[LinkInertia],
    q: &[f64],
    qd: &[f64],
    qdd: &[f64],
    gravity: Vector3<f64>,
) -> Vec<f64> {
    let n = robot.dof();
    // Per-joint relative transform A_i (frame i → i-1): rotation, translation, axis.
    let mut rr = Vec::with_capacity(n); // frame i → i-1 rotation
    let mut pp = Vec::with_capacity(n); // origin of frame i in frame i-1
    let mut zz = Vec::with_capacity(n); // joint axis in frame i
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        rr.push(*a.rotation.to_rotation_matrix().matrix());
        pp.push(a.translation.vector);
        zz.push(robot.joints[i].axis.into_inner());
    }

    // Outward recursion: link velocities/accelerations and Newton-Euler forces, all in frame i.
    let (mut omega, mut omegad, mut vd) =
        (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
    let (mut ff, mut nn) = (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
    let (mut pw, mut pwd, mut pvd) = (Vector3::zeros(), Vector3::zeros(), -gravity);
    for i in 0..n {
        let rt = rr[i].transpose(); // frame i-1 → i
        let z = zz[i];
        let base = rt * (pvd + pwd.cross(&pp[i]) + pw.cross(&pw.cross(&pp[i])));
        match robot.joints[i].kind {
            crate::JointKind::Revolute => {
                omega[i] = rt * pw + qd[i] * z;
                omegad[i] = rt * pwd + (rt * pw).cross(&(qd[i] * z)) + qdd[i] * z;
                vd[i] = base;
            }
            crate::JointKind::Prismatic => {
                omega[i] = rt * pw;
                omegad[i] = rt * pwd;
                vd[i] = base + 2.0 * omega[i].cross(&(qd[i] * z)) + qdd[i] * z;
            }
        }
        let li = &inertia[i];
        let vdc = vd[i] + omegad[i].cross(&li.com) + omega[i].cross(&omega[i].cross(&li.com));
        ff[i] = li.mass * vdc;
        nn[i] = li.inertia * omegad[i] + omega[i].cross(&(li.inertia * omega[i]));
        pw = omega[i];
        pwd = omegad[i];
        pvd = vd[i];
    }

    // Inward recursion: propagate forces/moments, read off joint torques.
    let mut tau = vec![0.0; n];
    let (mut f_next, mut n_next) = (Vector3::zeros(), Vector3::zeros());
    for i in (0..n).rev() {
        let (rr_next, p_next) =
            if i + 1 < n { (rr[i + 1], pp[i + 1]) } else { (Matrix3::identity(), Vector3::zeros()) };
        let f_i = rr_next * f_next + ff[i];
        let n_i = nn[i] + rr_next * n_next + inertia[i].com.cross(&ff[i]) + p_next.cross(&(rr_next * f_next));
        tau[i] = match robot.joints[i].kind {
            crate::JointKind::Revolute => n_i.dot(&zz[i]),
            crate::JointKind::Prismatic => f_i.dot(&zz[i]),
        };
        f_next = f_i;
        n_next = n_i;
    }
    tau
}

/// Generalized gravity torques `G(q)` (RNEA with zero velocity and acceleration).
pub fn gravity_vector(robot: &Robot, inertia: &[LinkInertia], q: &[f64], gravity: Vector3<f64>) -> Vec<f64> {
    let z = vec![0.0; robot.dof()];
    inverse_dynamics(robot, inertia, q, &z, &z, gravity)
}

/// Joint-space inertia (mass) matrix `M(q)`: column `j` is RNEA with `q̈ = eⱼ`, no gravity/velocity.
pub fn mass_matrix(robot: &Robot, inertia: &[LinkInertia], q: &[f64]) -> DMatrix<f64> {
    let n = robot.dof();
    let z = vec![0.0; n];
    let mut m = DMatrix::zeros(n, n);
    for j in 0..n {
        let mut qdd = vec![0.0; n];
        qdd[j] = 1.0;
        let col = inverse_dynamics(robot, inertia, q, &z, &qdd, Vector3::zeros());
        for i in 0..n {
            m[(i, j)] = col[i];
        }
    }
    m
}

/// Forward dynamics: joint accelerations under applied torques,
/// `q̈ = M(q)⁻¹ (τ − C(q,q̇)q̇ − G(q))`. The bias `C·q̇ + G` is RNEA with `q̈ = 0`. Enables
/// simulation (integrate q̈ forward) and closed-loop controller testing.
pub fn forward_dynamics(
    robot: &Robot,
    inertia: &[LinkInertia],
    q: &[f64],
    qd: &[f64],
    tau: &[f64],
    gravity: Vector3<f64>,
) -> Vec<f64> {
    let n = robot.dof();
    let m = mass_matrix(robot, inertia, q);
    let bias = inverse_dynamics(robot, inertia, q, qd, &vec![0.0; n], gravity);
    let rhs = DVector::from_iterator(n, (0..n).map(|i| tau[i] - bias[i]));
    match m.cholesky() {
        Some(ch) => ch.solve(&rhs).as_slice().to_vec(),
        None => vec![0.0; n],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_full;

    const PENDULUM: &str = r#"<robot name="pend">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="2.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l1"/><child link="tool"/><origin xyz="1 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    #[test]
    fn pendulum_gravity_and_inertia() {
        let (robot, inertia) = from_urdf_full(PENDULUM, "base", "tool").unwrap();
        assert_eq!(robot.dof(), 1);
        // Gravity torque holding a horizontal 2 kg link with COM 0.5 m out: m·g·d = 2·9.81·0.5.
        let g = gravity_vector(&robot, &inertia, &[0.0], Vector3::new(0.0, 0.0, -9.81));
        assert!((g[0].abs() - 9.81).abs() < 1e-4, "gravity torque {}", g[0]);
        // Inertia about the joint axis: I_com,yy + m·d² = 0.01 + 2·0.25 = 0.51.
        let m = mass_matrix(&robot, &inertia, &[0.0]);
        assert!((m[(0, 0)] - 0.51).abs() < 1e-6, "M[0,0] = {}", m[(0, 0)]);
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

    #[test]
    fn forward_dynamics_matches_released_pendulum() {
        let (robot, inertia) = from_urdf_full(PENDULUM, "base", "tool").unwrap();
        // Released from horizontal, zero torque: q̈ = -G/M = -9.81/0.51 ≈ 19.23 rad/s² (magnitude).
        let qdd = forward_dynamics(&robot, &inertia, &[0.0], &[0.0], &[0.0], Vector3::new(0.0, 0.0, -9.81));
        assert!((qdd[0].abs() - 9.81 / 0.51).abs() < 0.05, "released accel {}", qdd[0]);
    }

    #[test]
    fn mass_matrix_is_symmetric_and_positive_definite() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let m = mass_matrix(&robot, &inertia, &[0.3, -0.7]);
        assert!((m.clone() - m.transpose()).norm() < 1e-9, "M not symmetric");
        assert!(m.clone().cholesky().is_some(), "M not positive-definite");
        // Coriolis/gravity-free consistency: τ = M·q̈ exactly when q̇ = 0, g = 0.
        let qdd = [0.4, -0.2];
        let tau = inverse_dynamics(&robot, &inertia, &[0.3, -0.7], &[0.0, 0.0], &qdd, Vector3::zeros());
        let m_qdd = &m * nalgebra::DVector::from_row_slice(&qdd);
        for i in 0..2 {
            assert!((tau[i] - m_qdd[i]).abs() < 1e-9, "τ ≠ M·q̈ at row {i}");
        }
    }
}
