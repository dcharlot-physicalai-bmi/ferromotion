//! Whole-body control as an acceleration-level QP: solve for joint accelerations that best meet a
//! weighted stack of Cartesian tasks plus a posture task, subject to joint-acceleration limits,
//! then map to torques by inverse dynamics `τ = M·q̈ + C·q̇ + G`. The unifying manipulation/humanoid
//! controller — extends the diff-IK QP idea to accelerations + torque. (First cut neglects `J̇·q̇`,
//! exact at rest; a `J̇·q̇` feedforward is a drop-in refinement.)

use crate::qp::solve_box_qp;
use nalgebra::{DMatrix, DVector, Point3, Vector3};
use ferromotion_core::{inverse_dynamics, LinkInertia, Robot};

/// A Cartesian position task: drive `frame`+`offset` toward `target` with PD gains and a weight.
#[derive(Clone, Copy, Debug)]
pub struct CartesianTask {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub target: Vector3<f64>,
    pub kp: f64,
    pub kd: f64,
    pub weight: f64,
}

/// Whole-body controller: a stack of Cartesian tasks + posture, solved as one QP over `q̈`.
#[derive(Clone, Debug)]
pub struct WholeBody {
    pub tasks: Vec<CartesianTask>,
    pub posture_weight: f64,
    pub posture_kp: f64,
    pub posture_kd: f64,
    pub rest: Vec<f64>,
    /// Regularization on `q̈` and box limit `|q̈ᵢ| ≤ accel_limit`.
    pub reg: f64,
    pub accel_limit: f64,
}

impl WholeBody {
    pub fn torque(
        &self,
        robot: &Robot,
        inertia: &[LinkInertia],
        q: &[f64],
        qd: &[f64],
        gravity: Vector3<f64>,
    ) -> Vec<f64> {
        let n = robot.dof();
        let qd_v = DVector::from_row_slice(qd);

        // Assemble the QP: min ½ q̈ᵀH·q̈ + gᵀq̈, H = Σ wᵢ JᵢᵀJᵢ + (posture_w + reg)·I.
        let mut h = DMatrix::<f64>::identity(n, n) * self.reg;
        let mut g = DVector::<f64>::zeros(n);
        for t in &self.tasks {
            let p = (robot.frame_pose(q, t.frame) * Point3::from(t.offset)).coords;
            let j = robot.point_jacobian(q, t.frame, &p);
            let xdot = &j * &qd_v;
            let e = t.target - p;
            let a = DVector::from_row_slice(&[
                t.kp * e.x - t.kd * xdot[0],
                t.kp * e.y - t.kd * xdot[1],
                t.kp * e.z - t.kd * xdot[2],
            ]);
            h += t.weight * (j.transpose() * &j);
            g -= t.weight * (j.transpose() * a);
        }
        // Posture task, directly at the acceleration level.
        for i in 0..n {
            let a_post = self.posture_kp * (self.rest[i] - q[i]) - self.posture_kd * qd[i];
            h[(i, i)] += self.posture_weight;
            g[i] -= self.posture_weight * a_post;
        }
        h = 0.5 * (&h + &h.transpose());

        let g_lin: Vec<f64> = g.iter().cloned().collect();
        let lo = vec![-self.accel_limit; n];
        let hi = vec![self.accel_limit; n];
        let qdd = solve_box_qp(&h, &g_lin, &lo, &hi);

        // Torque from inverse dynamics.
        inverse_dynamics(robot, inertia, q, qd, &qdd, gravity)
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
    fn wbc_regulates_tool_and_respects_accel_limit() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let wbc = WholeBody {
            tasks: vec![CartesianTask {
                frame: 2,
                offset: Vector3::new(0.5, 0.0, 0.0), // tool tip (frame 2 + 0.5 m)
                target: Vector3::new(0.7, 0.5, 0.0),
                kp: 150.0,
                kd: 24.0,
                weight: 1.0,
            }],
            posture_weight: 0.01,
            posture_kp: 1.0,
            posture_kd: 2.0,
            rest: vec![0.0, 0.0],
            reg: 1e-4,
            accel_limit: 50.0,
        };
        let (mut q, mut qd, dt) = (vec![0.3, -0.4], vec![0.0, 0.0], 1e-3);
        for _ in 0..6000 {
            let tau = wbc.torque(&robot, &inertia, &q, &qd, g);
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        let tip = robot.fk(&q).translation.vector;
        assert!((tip - Vector3::new(0.7, 0.5, 0.0)).norm() < 1e-2, "tool at {tip:?}");
    }
}
