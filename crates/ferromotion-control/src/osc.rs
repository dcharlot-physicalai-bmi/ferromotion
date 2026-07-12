//! Operational-space control (Khatib): task-space torque control with the operational-space
//! inertia `Λ = (J·M⁻¹·Jᵀ)⁻¹`, gravity/Coriolis compensation, and a dynamically-consistent
//! null-space posture term so redundant DoF stay well-behaved without disturbing the task.

use nalgebra::{DMatrix, DVector, Vector3};
use ferromotion_core::{inverse_dynamics, mass_matrix, LinkInertia, Robot};

/// Operational-space (task-space) controller regulating the tool position to a target.
#[derive(Clone, Debug)]
pub struct OperationalSpace {
    pub kp: f64,
    pub kd: f64,
    pub kp_posture: f64,
    pub kd_posture: f64,
    /// Damping on `Λ`'s inversion (keeps it well-posed at/near kinematic singularities).
    pub damping: f64,
    /// Null-space posture target.
    pub rest: Vec<f64>,
}

impl OperationalSpace {
    pub fn torque(
        &self,
        robot: &Robot,
        inertia: &[LinkInertia],
        q: &[f64],
        qd: &[f64],
        x_des: Vector3<f64>,
        gravity: Vector3<f64>,
    ) -> Vec<f64> {
        let n = robot.dof();
        let tip = robot.fk(q).translation.vector;
        let jp = robot.point_jacobian(q, n, &tip); // 3×n
        let m = mass_matrix(robot, inertia, q);
        let minv = m.try_inverse().expect("mass matrix invertible");

        // Operational-space inertia (damped) and the dynamically-consistent inverse.
        let jt = jp.transpose();
        let lambda = (&jp * &minv * &jt + DMatrix::identity(3, 3) * self.damping)
            .try_inverse()
            .expect("Λ invertible with damping");
        let jbar = &minv * &jt * &lambda; // n×3

        // Task-space PD → wrench.
        let qd_v = DVector::from_row_slice(qd);
        let xdot = &jp * &qd_v;
        let e = x_des - tip;
        let acc = DVector::from_row_slice(&[
            self.kp * e.x - self.kd * xdot[0],
            self.kp * e.y - self.kd * xdot[1],
            self.kp * e.z - self.kd * xdot[2],
        ]);
        let tau_task = &jt * (&lambda * acc);

        // Null-space posture, projected to not fight the task: Nᵀ = I − Jᵀ·J̄ᵀ.
        let nt = DMatrix::identity(n, n) - &jt * jbar.transpose();
        let tau0 = DVector::from_iterator(
            n,
            (0..n).map(|i| self.kp_posture * (self.rest[i] - q[i]) - self.kd_posture * qd[i]),
        );
        let tau_null = &nt * tau0;

        // Gravity + Coriolis compensation.
        let b = inverse_dynamics(robot, inertia, q, qd, &vec![0.0; n], gravity);
        (0..n).map(|i| tau_task[i] + tau_null[i] + b[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferromotion_core::{forward_dynamics, from_urdf_full};

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
    fn osc_regulates_tool_to_target() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let ctrl = OperationalSpace {
            kp: 200.0,
            kd: 28.0,
            kp_posture: 1.0,
            kd_posture: 2.0,
            damping: 1e-4,
            rest: vec![0.0, 0.0],
        };
        let x_des = Vector3::new(0.7, 0.5, 0.0);
        let (mut q, mut qd, dt) = (vec![0.3, -0.4], vec![0.0, 0.0], 1e-3);
        for _ in 0..6000 {
            let tau = ctrl.torque(&robot, &inertia, &q, &qd, x_des, g);
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let tip = robot.fk(&q).translation.vector;
        assert!((tip - x_des).norm() < 5e-3, "tool at {tip:?}, target {x_des:?}");
    }
}
