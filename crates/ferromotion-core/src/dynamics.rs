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
///
/// Two actuator terms are added to the rigid-body result when the model states them:
/// [`crate::Joint::armature`] (reflected rotor inertia, `+J_a·q̈`) and [`crate::Joint::damping`] (viscous, `+b·q̇`). Both
/// default to unstated and contribute exactly zero, so a model without them is bit-identical to before they
/// existed. Adding them **here** rather than at each call site is what keeps
/// `M(q)·q̈ + bias == inverse_dynamics(q, q̇, q̈)` true: [`mass_matrix`] is built from this function with
/// `q̈ = eⱼ`, so armature lands on its diagonal automatically, and [`forward_dynamics`] inherits both.
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
        // The rotor accelerates with the joint and the joint damping opposes its rate. Neither is part of
        // the link chain, so neither appears in the recursion above; both are diagonal at the joint.
        if let Some(j_a) = robot.joints[i].armature {
            tau[i] += j_a * qdd[i];
        }
        if let Some(b) = robot.joints[i].damping {
            tau[i] += b * qd[i];
        }
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
///
/// Includes [`crate::Joint::armature`] on the diagonal, because it is built from [`inverse_dynamics`] and that is
/// where the term is applied. Velocity is zero here, so [`crate::Joint::damping`] cannot contribute.
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

    /// A two-link arm for the actuator-term tests.
    const TWO_LINK: &str = r#"<robot name="two">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.2 0 0"/><mass value="0.8"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="8" velocity="4"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="5" velocity="4"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.4 0 0"/></joint>
    </robot>"#;

    /// **An unstated actuator term must cost exactly nothing.** `with_armature(0.0)` stores `None`, not
    /// `Some(0.0)`, so the `if let` in the recursion does not execute and no float operation happens — the
    /// result is bit-identical, not merely close. Asserted with `==` deliberately: a tolerance here would let a
    /// real regression through, and this is the one place where exact equality is the actual claim.
    #[test]
    fn an_unstated_actuator_term_changes_nothing_at_all() {
        let (plain, inertia) = from_urdf_full(TWO_LINK, "base", "tool").unwrap();
        assert_eq!(
            plain.joints[0].armature, None,
            "loader must not invent an armature"
        );
        assert_eq!(
            plain.joints[0].damping, None,
            "loader must not invent a damping"
        );

        let mut zeroed = plain.clone();
        for j in zeroed.joints.iter_mut() {
            *j = j.clone().with_armature(0.0).with_damping(-1.0); // non-positive is "unstated", not "zero rotor"
        }
        assert_eq!(zeroed.joints[0].armature, None);
        assert_eq!(zeroed.joints[0].damping, None);

        let (q, qd, qdd) = ([0.4, -0.7], [1.1, -0.3], [2.0, 0.5]);
        let g = Vector3::new(0.0, 0.0, -9.81);
        let a = inverse_dynamics(&plain, &inertia, &q, &qd, &qdd, g);
        let b = inverse_dynamics(&zeroed, &inertia, &q, &qd, &qdd, g);
        assert_eq!(
            a, b,
            "an unstated term must not perturb the rigid-body result"
        );
        assert_eq!(
            mass_matrix(&plain, &inertia, &q),
            mass_matrix(&zeroed, &inertia, &q)
        );
    }

    /// **The identity that makes the placement correct.** Armature and damping are applied inside
    /// `inverse_dynamics`, so `M(q)·q̈ + bias(q,q̇) == inverse_dynamics(q,q̇,q̈)` must still hold exactly.
    /// Putting armature in `mass_matrix` instead — the obvious shortcut — breaks this, because the bias would
    /// then be missing the damping and the two paths would disagree wherever `q̇ ≠ 0`.
    #[test]
    fn the_mass_matrix_and_bias_still_reconstruct_inverse_dynamics() {
        let (mut robot, inertia) = from_urdf_full(TWO_LINK, "base", "tool").unwrap();
        robot.joints[0] = robot.joints[0]
            .clone()
            .with_armature(0.011)
            .with_damping(0.64);
        robot.joints[1] = robot.joints[1]
            .clone()
            .with_armature(0.004)
            .with_damping(0.21);
        let g = Vector3::new(0.0, 0.0, -9.81);
        // Several states, because a term that vanishes at q̇ = 0 would pass a single lazy sample.
        for &(q, qd, qdd) in &[
            ([0.0, 0.0], [0.0, 0.0], [1.0, 0.0]),
            ([0.4, -0.7], [1.1, -0.3], [2.0, 0.5]),
            ([-1.2, 0.9], [-2.5, 3.0], [-0.7, 1.8]),
        ] {
            let direct = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            let m = mass_matrix(&robot, &inertia, &q);
            let bias = inverse_dynamics(&robot, &inertia, &q, &qd, &[0.0, 0.0], g);
            for i in 0..2 {
                let recon = m[(i, 0)] * qdd[0] + m[(i, 1)] * qdd[1] + bias[i];
                assert!(
                    (direct[i] - recon).abs() < 1e-12,
                    "row {i}: {direct:?} vs reconstruction {recon}"
                );
            }
        }
    }

    /// **Armature goes on the diagonal and only there.** A rotor spins about its own joint axis; it must not
    /// couple joint 0 to joint 1. Measured as the difference between the matrix with and without it.
    #[test]
    fn armature_is_diagonal() {
        let (plain, inertia) = from_urdf_full(TWO_LINK, "base", "tool").unwrap();
        let mut geared = plain.clone();
        geared.joints[0] = geared.joints[0].clone().with_armature(0.011);
        geared.joints[1] = geared.joints[1].clone().with_armature(0.004);
        let q = [0.3, -1.1];
        let (a, b) = (
            mass_matrix(&plain, &inertia, &q),
            mass_matrix(&geared, &inertia, &q),
        );
        assert!(
            (b[(0, 0)] - a[(0, 0)] - 0.011).abs() < 1e-14,
            "joint 0 diagonal"
        );
        assert!(
            (b[(1, 1)] - a[(1, 1)] - 0.004).abs() < 1e-14,
            "joint 1 diagonal"
        );
        assert_eq!(b[(0, 1)], a[(0, 1)], "armature must not create coupling");
        assert_eq!(b[(1, 0)], a[(1, 0)], "armature must not create coupling");
    }

    /// **The SO-101's wrist is the case that motivated all of this.** Its link inertia is small enough that a
    /// geared servo's reflected rotor dominates it outright, so a URDF-only model of this arm is missing the
    /// larger of the two terms. This asserts the ratio the `so101_reach_rl` bench depends on.
    #[test]
    fn a_geared_rotor_can_dominate_the_link_it_drives() {
        let so101 = include_str!("../examples/so101.urdf");
        let (mut robot, inertia) = from_urdf_full(so101, "base_link", "gripper_link").unwrap();
        let n = robot.dof();
        let rigid = mass_matrix(&robot, &inertia, &vec![0.0; n]);
        let wrist = rigid[(n - 1, n - 1)];
        assert!(
            wrist < 1e-4,
            "wrist link inertia should be tiny, measured {wrist:.3e}"
        );

        let reflected = 345.0f64.powi(2) * 1e-7; // N^2 J_rotor for the STS3215
        assert!(
            reflected / wrist > 100.0,
            "reflected/link ratio {:.0}",
            reflected / wrist
        );

        // With it applied, the smallest eigenvalue of M rises by roughly the armature, which is what makes the
        // plant integrable at a usable step size.
        for j in robot.joints.iter_mut() {
            *j = j.clone().with_armature(reflected);
        }
        let geared = mass_matrix(&robot, &inertia, &vec![0.0; n]);
        let before = rigid
            .symmetric_eigenvalues()
            .iter()
            .fold(f64::INFINITY, |a: f64, &b| a.min(b));
        let after = geared
            .symmetric_eigenvalues()
            .iter()
            .fold(f64::INFINITY, |a: f64, &b| a.min(b));
        assert!(
            after > before * 10.0,
            "smallest eigenvalue {before:.3e} -> {after:.3e}"
        );
    }

    /// **Damping must remove energy, and the loss must come from damping rather than the integrator.**
    ///
    /// A passive term with the wrong sign is invisible to a shape or finiteness check and shows up only as a
    /// system that quietly gains energy — the failure mode the SMA cooling branch in this workspace hid behind
    /// endpoint tests. The undamped run is the control: semi-implicit Euler does not conserve energy exactly,
    /// so "energy fell" on its own proves nothing. The bound is the measured ratio between the two, not a
    /// number chosen in advance. Both thresholds here were guessed wrong before being measured: a halving in
    /// 0.2 s (measured 0.73, because `I/b` is about a second) and then a 100x separation (measured 17.9x).
    /// Measured over 2 s: the undamped run conserves energy to `1.1e-4` relative, the damped run keeps
    /// `5.48e-2` of `9.80e-1`. The bounds below are those numbers with margin.
    #[test]
    fn damping_dissipates() {
        let zero = Vector3::zeros(); // no gravity, so total energy is kinetic alone
        let dt = 1e-4;
        let steps = 20_000; // 2 s, several times the I/b time constant

        let run = |damping: Option<f64>| -> (f64, f64, bool) {
            let (mut robot, inertia) = from_urdf_full(TWO_LINK, "base", "tool").unwrap();
            if let Some(b) = damping {
                for j in robot.joints.iter_mut() {
                    *j = j.clone().with_damping(b);
                }
            }
            let (mut q, mut qd) = (vec![0.2, -0.4], vec![2.0, -1.5]);
            let ke = |robot: &Robot, q: &[f64], qd: &[f64]| {
                let m = mass_matrix(robot, &inertia, q);
                0.5 * (0..2)
                    .map(|i| (0..2).map(|j| qd[i] * m[(i, j)] * qd[j]).sum::<f64>())
                    .sum::<f64>()
            };
            let start = ke(&robot, &q, &qd);
            let mut prev = start;
            let mut monotone = true;
            for _ in 0..steps {
                let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &[0.0, 0.0], zero);
                for i in 0..2 {
                    qd[i] += qdd[i] * dt;
                    q[i] += qd[i] * dt;
                }
                let now = ke(&robot, &q, &qd);
                if now > prev + 1e-9 {
                    monotone = false;
                }
                prev = now;
            }
            (start, prev, monotone)
        };

        let (start_d, end_d, monotone) = run(Some(0.5));
        let (start_u, end_u, _) = run(None);
        assert!(monotone, "a damped free system must never gain energy");
        assert!(
            (start_d - start_u).abs() < 1e-12,
            "both runs must start from the same energy"
        );
        // The integrator itself is near-conservative here, so any real loss is attributable to damping.
        assert!(
            (end_u - start_u).abs() / start_u < 0.01,
            "undamped drift {:.2e} is too large for this to be a clean control",
            (end_u - start_u).abs() / start_u
        );
        assert!(
            end_d < end_u / 10.0,
            "damped left {end_d:.3e} of {start_d:.3e}; undamped left {end_u:.3e}, so the loss is not damping"
        );
    }
}
