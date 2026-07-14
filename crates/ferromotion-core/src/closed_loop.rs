//! Closed-loop / parallel mechanisms — dynamics of linkages that are **not** serial chains (4-bars,
//! parallel ankles, delta/Stewart platforms). The standard construction: model the mechanism as a
//! spanning tree (our [`Robot`]) plus **loop-closure holonomic constraints** `c(q) = 0`, and solve the
//! constrained forward dynamics as the KKT/Lagrange-multiplier saddle system
//!
//! ```text
//!   [ M   −Jᵀ ] [ q̈ ]   [ τ − (C q̇ + G) ]
//!   [ J    0  ] [ λ ] = [       γ        ] ,   γ = −J̇q̇ − 2ω(Jq̇) − ω²c   (Baumgarte stabilization)
//! ```
//!
//! where `J = ∂c/∂q` is the loop Jacobian and `λ` the constraint forces. Baumgarte terms keep the
//! loop from drifting open during integration. This lets the serial-chain core represent the closed
//! kinematic loops real robots have (parallel legs/ankles). Pure `nalgebra` → WASM-clean.

use crate::{inverse_dynamics, mass_matrix, LinkInertia, Robot};
use nalgebra::{DMatrix, DVector, Point3, Vector3};

/// A pin constraint closing a loop: the world position of point `offset` on frame `frame` is held at
/// `target` (in the x–y plane — the natural setting for planar linkages).
#[derive(Clone, Debug)]
pub struct Pin {
    pub frame: usize,
    pub offset: Vector3<f64>,
    pub target: [f64; 2],
}

/// A planar closed-loop mechanism: a serial spanning tree plus pin constraints closing the loops.
pub struct PlanarLoop<'a> {
    pub robot: &'a Robot,
    pub inertia: &'a [LinkInertia],
    pub pins: Vec<Pin>,
    /// Baumgarte stabilization frequency (rad/s); larger = stiffer loop closure.
    pub omega: f64,
}

impl PlanarLoop<'_> {
    fn nc(&self) -> usize {
        2 * self.pins.len()
    }

    fn world_xy(&self, q: &[f64], pin: &Pin) -> Vector3<f64> {
        (self.robot.frame_pose(q, pin.frame) * Point3::from(pin.offset)).coords
    }

    /// Loop-closure residual `c(q)` (stacked x,y per pin).
    pub fn constraint(&self, q: &[f64]) -> DVector<f64> {
        let mut c = DVector::zeros(self.nc());
        for (i, pin) in self.pins.iter().enumerate() {
            let w = self.world_xy(q, pin);
            c[2 * i] = w.x - pin.target[0];
            c[2 * i + 1] = w.y - pin.target[1];
        }
        c
    }

    /// Loop Jacobian `J = ∂c/∂q` (x,y rows of each pin point's position Jacobian).
    pub fn jacobian(&self, q: &[f64]) -> DMatrix<f64> {
        let n = self.robot.dof();
        let mut j = DMatrix::zeros(self.nc(), n);
        for (i, pin) in self.pins.iter().enumerate() {
            let w = self.world_xy(q, pin);
            let jp = self.robot.point_jacobian(q, pin.frame, &w); // 3×n
            j.row_mut(2 * i).copy_from(&jp.row(0));
            j.row_mut(2 * i + 1).copy_from(&jp.row(1));
        }
        j
    }

    /// The constraint velocity-product term `J̇q̇`, by central finite difference of `J` along `q̇`.
    fn jdot_qd(&self, q: &[f64], qd: &[f64]) -> DVector<f64> {
        let n = self.robot.dof();
        let h = 1e-6;
        let qp: Vec<f64> = (0..n).map(|i| q[i] + h * qd[i]).collect();
        let qm: Vec<f64> = (0..n).map(|i| q[i] - h * qd[i]).collect();
        let jdot = (self.jacobian(&qp) - self.jacobian(&qm)) / (2.0 * h);
        jdot * DVector::from_row_slice(qd)
    }

    /// Constrained forward dynamics: joint accelerations `q̈` and constraint forces `λ` for the closed
    /// loop under applied torques `tau` and gravity.
    pub fn forward_dynamics(&self, q: &[f64], qd: &[f64], tau: &[f64], gravity: Vector3<f64>) -> (Vec<f64>, Vec<f64>) {
        let n = self.robot.dof();
        let nc = self.nc();
        let m = mass_matrix(self.robot, self.inertia, q);
        let bias = inverse_dynamics(self.robot, self.inertia, q, qd, &vec![0.0; n], gravity);
        let j = self.jacobian(q);
        let qd_v = DVector::from_row_slice(qd);
        // Baumgarte-stabilized acceleration-level constraint target γ.
        let gamma = -self.jdot_qd(q, qd) - 2.0 * self.omega * (&j * &qd_v) - self.omega * self.omega * self.constraint(q);

        // Assemble and solve the (n+nc) KKT saddle system.
        let dim = n + nc;
        let mut kkt = DMatrix::zeros(dim, dim);
        kkt.view_mut((0, 0), (n, n)).copy_from(&m);
        kkt.view_mut((0, n), (n, nc)).copy_from(&(-j.transpose()));
        kkt.view_mut((n, 0), (nc, n)).copy_from(&j);
        let mut rhs = DVector::zeros(dim);
        for i in 0..n {
            rhs[i] = tau[i] - bias[i];
        }
        for i in 0..nc {
            rhs[n + i] = gamma[i];
        }
        let sol = kkt.lu().solve(&rhs).expect("KKT system solvable");
        let qdd: Vec<f64> = (0..n).map(|i| sol[i]).collect();
        let lambda: Vec<f64> = (0..nc).map(|i| sol[n + i]).collect();
        (qdd, lambda)
    }

    /// Kinetic energy `½ q̇ᵀ M(q) q̇` (for conservation checks).
    pub fn kinetic_energy(&self, q: &[f64], qd: &[f64]) -> f64 {
        let m = mass_matrix(self.robot, self.inertia, q);
        let v = DVector::from_row_slice(qd);
        0.5 * v.dot(&(&m * &v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_urdf_full;

    // A 3-revolute planar chain (all joints about +z, links in the x–y plane). Pinning its tip to a
    // fixed ground point closes it into a 1-DoF four-bar linkage (3 joints − 2 constraints = 1 DoF).
    const CHAIN3: &str = r#"<robot name="c3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.08"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.08"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.5 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.08"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="20" velocity="10"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="20" velocity="10"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="1 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="20" velocity="10"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="1 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    fn four_bar<'a>(robot: &'a Robot, inertia: &'a [LinkInertia], q0: &[f64]) -> PlanarLoop<'a> {
        // Pin the tip to wherever it starts → an initially-consistent closed loop through q0.
        let tip = (robot.fk(q0)).translation.vector;
        PlanarLoop { robot, inertia, pins: vec![Pin { frame: 3, offset: Vector3::new(1.0, 0.0, 0.0), target: [tip.x, tip.y] }], omega: 30.0 }
    }

    #[test]
    fn four_bar_conserves_energy_and_holds_the_loop_closed() {
        let (robot, inertia) = from_urdf_full(CHAIN3, "base", "tool").unwrap();
        let q0 = [0.6, 0.9, -0.7];
        let loop_sys = four_bar(&robot, &inertia, &q0);
        assert!(loop_sys.constraint(&q0).norm() < 1e-12, "initial config not on the loop");

        // Consistent 1-DoF initial velocity: the null space of J (2×3) is the single mechanism DoF.
        let j = loop_sys.jacobian(&q0);
        let (r0, r1) = (j.row(0).transpose(), j.row(1).transpose());
        let null = Vector3::new(r0[0], r0[1], r0[2]).cross(&Vector3::new(r1[0], r1[1], r1[2])).normalize();
        let qd0 = [null[0] * 1.5, null[1] * 1.5, null[2] * 1.5];
        assert!((loop_sys.jacobian(&q0) * DVector::from_row_slice(&qd0)).norm() < 1e-9, "qd0 not in the null space");

        // Passive (no gravity, no torque): energy is conserved and the loop stays closed.
        let (mut q, mut qd, dt, g) = (q0.to_vec(), qd0.to_vec(), 5e-4, Vector3::zeros());
        let e0 = loop_sys.kinetic_energy(&q, &qd);
        let (mut worst_c, mut moved) = (0.0f64, 0.0f64);
        for _ in 0..2000 {
            let (qdd, _) = loop_sys.forward_dynamics(&q, &qd, &[0.0, 0.0, 0.0], g);
            for i in 0..3 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
            worst_c = worst_c.max(loop_sys.constraint(&q).norm());
            moved = moved.max((q[0] - q0[0]).abs());
        }
        let e1 = loop_sys.kinetic_energy(&q, &qd);
        let drift = (e1 - e0).abs() / e0;
        eprintln!("four-bar: E0={e0:.5} E1={e1:.5} drift={drift:.2e} worst|c|={worst_c:.2e} moved={moved:.3}");
        assert!(worst_c < 1e-3, "loop drifted open: worst |c| = {worst_c:.2e}");
        assert!(moved > 0.2, "mechanism barely moved (moved={moved}); test not exercising motion");
        assert!(drift < 2e-2, "energy not conserved: relative drift {drift:.2e}");
    }

    #[test]
    fn four_bar_stays_closed_under_gravity_and_torque() {
        let (robot, inertia) = from_urdf_full(CHAIN3, "base", "tool").unwrap();
        let q0 = [0.5, 1.0, -0.8];
        let loop_sys = four_bar(&robot, &inertia, &q0);
        let (mut q, mut qd, dt, g) = (q0.to_vec(), vec![0.0, 0.0, 0.0], 5e-4, Vector3::new(0.0, -9.81, 0.0));
        let mut worst_c = 0.0f64;
        for k in 0..3000 {
            // A little crank torque on top of gravity, to drive the loop around.
            let tau = [if k < 1500 { 3.0 } else { -3.0 }, 0.0, 0.0];
            let (qdd, _lambda) = loop_sys.forward_dynamics(&q, &qd, &tau, g);
            for i in 0..3 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
            worst_c = worst_c.max(loop_sys.constraint(&q).norm());
        }
        assert!(worst_c < 1e-2, "loop opened under load: worst |c| = {worst_c:.2e}");
    }
}
