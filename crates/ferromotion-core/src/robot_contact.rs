//! Articulated-multibody contact — the differentiable frictional contact step applied to a robot
//! chain (full-Dojo at articulated scale). A contact point on a link maps to joint space through the
//! point Jacobian: the joint-space normal row is `Jₚᵀ·n̂` and each friction facet is `Jₚᵀ·t̂`. Feeding
//! the joint-space mass matrix `M(q)` and these rows into the interior-point frictional solver
//! ([`crate::solve_frictional_ipm`]) gives a differentiable, non-penetrating contact step for the
//! whole arm: `q̇⁺` (and, via the solver, its gradient). Pure `nalgebra` → WASM-clean.

use crate::{inverse_dynamics, mass_matrix, solve_frictional_ipm, LinkInertia, Robot, StFrictionContact};
use nalgebra::{DMatrix, DVector, Point3, Vector3};

/// A ground-plane contact simulator for a serial robot: contact points (link frame + offset +
/// friction) collide with a horizontal floor at `floor_z`.
#[derive(Clone, Debug)]
pub struct RobotContactSim<'a> {
    pub robot: &'a Robot,
    pub inertia: &'a [LinkInertia],
    /// Contact points as `(frame index, offset in that frame, friction μ)`.
    pub contacts: Vec<(usize, Vector3<f64>, f64)>,
    pub floor_z: f64,
    pub kappa: f64,
}

impl RobotContactSim<'_> {
    /// World position of contact point `i` at configuration `q`.
    pub fn contact_world(&self, q: &[f64], i: usize) -> Vector3<f64> {
        let (frame, offset, _) = self.contacts[i];
        (self.robot.frame_pose(q, frame) * Point3::from(offset)).coords
    }

    /// Advance one step under applied joint torques `tau` and gravity `g`, resolving floor contact.
    pub fn step(&self, q: &[f64], qd: &[f64], tau: &[f64], dt: f64, g: Vector3<f64>) -> (Vec<f64>, Vec<f64>) {
        let n = self.robot.dof();
        let mm = mass_matrix(self.robot, self.inertia, q);
        // Free joint velocity after gravity/Coriolis + applied torque: q̇_free = q̇ + dt·M⁻¹(τ − bias).
        let bias = inverse_dynamics(self.robot, self.inertia, q, qd, &vec![0.0; n], g);
        let rhs = DVector::from_iterator(n, (0..n).map(|i| tau[i] - bias[i]));
        let qdd_free = mm.clone().cholesky().expect("M PD").solve(&rhs);
        let v_free = DVector::from_row_slice(qd) + qdd_free * dt;

        // Build joint-space contact rows for points near/below the floor.
        let mut cs = Vec::new();
        for (i, &(frame, offset, mu)) in self.contacts.iter().enumerate() {
            let world = self.contact_world(q, i);
            let phi = world.z - self.floor_z;
            if phi < 0.05 {
                let jp = self.robot.point_jacobian(q, frame, &world); // 3×n
                let _ = offset;
                let jrow = |axis: Vector3<f64>| jp.transpose() * axis; // n-vector = Jₚᵀ·axis
                let (jx, jy) = (jrow(Vector3::x()), jrow(Vector3::y()));
                cs.push(StFrictionContact {
                    jn: jrow(Vector3::z()),
                    jt: vec![jx.clone(), -&jx, jy.clone(), -&jy],
                    phi,
                    mu,
                });
            }
        }

        let qd_next = if cs.is_empty() {
            v_free
        } else {
            solve_frictional_ipm(&mm, &v_free, &cs, dt, self.kappa).v_next
        };
        let q_next: Vec<f64> = (0..n).map(|i| q[i] + qd_next[i] * dt).collect();
        (q_next, qd_next.as_slice().to_vec())
    }

    /// Like [`step`], but also returns the control gradient `∂q̇⁺/∂τ` (dof×dof) through the contact.
    /// Chain rule: `q̇_free = q̇ + dt·M⁻¹(τ − bias)` ⇒ `∂q̇_free/∂τ = dt·M⁻¹`, then the frictional
    /// solver supplies `∂q̇⁺/∂q̇_free`, so `∂q̇⁺/∂τ = (∂q̇⁺/∂q̇_free)·dt·M⁻¹`. Differentiable even
    /// through stick↔slip and contact make/break, because the interior-point step is smoothed.
    pub fn step_diff(&self, q: &[f64], qd: &[f64], tau: &[f64], dt: f64, g: Vector3<f64>) -> (Vec<f64>, Vec<f64>, DMatrix<f64>) {
        let n = self.robot.dof();
        let mm = mass_matrix(self.robot, self.inertia, q);
        let minv = mm.clone().try_inverse().expect("M invertible");
        let bias = inverse_dynamics(self.robot, self.inertia, q, qd, &vec![0.0; n], g);
        let rhs = DVector::from_iterator(n, (0..n).map(|i| tau[i] - bias[i]));
        let v_free = DVector::from_row_slice(qd) + &minv * &rhs * dt;

        let mut cs = Vec::new();
        for (i, &(frame, _offset, mu)) in self.contacts.iter().enumerate() {
            let world = self.contact_world(q, i);
            let phi = world.z - self.floor_z;
            if phi < 0.05 {
                let jp = self.robot.point_jacobian(q, frame, &world);
                let jrow = |axis: Vector3<f64>| jp.transpose() * axis;
                let (jx, jy) = (jrow(Vector3::x()), jrow(Vector3::y()));
                cs.push(StFrictionContact { jn: jrow(Vector3::z()), jt: vec![jx.clone(), -&jx, jy.clone(), -&jy], phi, mu });
            }
        }

        let dvfree_dtau = &minv * dt; // ∂q̇_free/∂τ
        let (qd_next, dqdnext_dtau) = if cs.is_empty() {
            (v_free.clone(), dvfree_dtau)
        } else {
            let step = solve_frictional_ipm(&mm, &v_free, &cs, dt, self.kappa);
            (step.v_next, &step.dvnext_dvfree * &dvfree_dtau)
        };
        let q_next: Vec<f64> = (0..n).map(|i| q[i] + qd_next[i] * dt).collect();
        (q_next, qd_next.as_slice().to_vec(), dqdnext_dtau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_full;

    // A 2-link arm in the vertical x–z plane (joints about +y), so gravity swings it down.
    const VARM: &str = r#"<robot name="varm">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="50" velocity="10"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="50" velocity="10"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="1 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    #[test]
    fn arm_falls_onto_the_floor_without_penetrating() {
        let (robot, inertia) = from_urdf_full(VARM, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        // Contact at the end-effector (frame = dof, offset = tool tip in l2 frame).
        let sim = RobotContactSim {
            robot: &robot,
            inertia: &inertia,
            contacts: vec![(2, Vector3::new(1.0, 0.0, 0.0), 0.5)],
            floor_z: -0.8, // the EE would reach z=-2 hanging straight down → contact must hold it
            kappa: 1e-5,
        };

        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 0.002);
        let mut min_phi = f64::INFINITY;
        for _ in 0..3000 {
            // Light joint damping so the swing dissipates and it settles against the floor.
            let tau = [-0.6 * qd[0], -0.6 * qd[1]];
            let (qn, qdn) = sim.step(&q, &qd, &tau, dt, g);
            q = qn;
            qd = qdn;
            min_phi = min_phi.min(sim.contact_world(&q, 0).z - sim.floor_z);
        }
        let ee_z = sim.contact_world(&q, 0).z;
        let speed = (qd[0] * qd[0] + qd[1] * qd[1]).sqrt();
        assert!(min_phi > -0.02, "EE penetrated the floor: min gap = {min_phi}");
        assert!((ee_z - (-0.8)).abs() < 0.05, "EE should rest on the floor: z = {ee_z}");
        assert!(speed < 0.1, "arm did not settle: |q̇| = {speed}");
    }

    #[test]
    fn contact_control_gradient_matches_finite_difference() {
        // ∂q̇⁺/∂τ through the articulated contact — the control gradient a policy would backprop.
        let (robot, inertia) = from_urdf_full(VARM, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let sim = RobotContactSim {
            robot: &robot,
            inertia: &inertia,
            contacts: vec![(2, Vector3::new(1.0, 0.0, 0.0), 0.5)],
            floor_z: -0.8,
            kappa: 1e-3, // smoothed so the gradient is well-defined at the contact boundary
        };
        // A configuration with the EE pressed against the floor (contact active).
        let (q, qd, dt) = ([1.9, -1.6], [0.1, -0.2], 0.005);
        assert!(sim.contact_world(&q, 0).z - sim.floor_z < 0.05, "test config must be in contact");
        let tau = [0.5, -0.3];
        let (_, _, grad) = sim.step_diff(&q, &qd, &tau, dt, g);
        assert!(grad.iter().all(|v| v.is_finite()), "gradient not finite");

        let eps = 1e-6;
        for col in 0..2 {
            let mut tp = tau;
            tp[col] += eps;
            let (_, qdp, _) = sim.step_diff(&q, &qd, &tp, dt, g);
            let (_, qd0, _) = sim.step_diff(&q, &qd, &tau, dt, g);
            for r in 0..2 {
                let fd = (qdp[r] - qd0[r]) / eps;
                assert!((grad[(r, col)] - fd).abs() < 5e-3, "∂q̇⁺/∂τ[{r},{col}]: {} vs fd {fd}", grad[(r, col)]);
            }
        }
    }
}
