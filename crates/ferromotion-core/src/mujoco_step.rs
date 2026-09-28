//! **MuJoCo's `mj_step` integrators in MuJoCo's own arithmetic** — `mj_EulerSkip` (semi-implicit, with the
//! implicit treatment of dof damping), `mj_advance` and `mj_integratePos` from `engine_forward.c` and
//! `engine_support.c` (3.13.0), after the native forward pass of [`crate::mujoco_smooth`],
//! [`crate::mujoco_efc`] and [`crate::mujoco_solver`].
//!
//! ⛔ Contracted as clang contracts MuJoCo's C on arm64 (see [`crate::mujoco_ccd`]).

use crate::mujoco_kinematics::{axis_angle2quat, mul_quat, normalize4, KinJointKind};
use crate::mujoco_smooth::{SmoothData, SmoothModel};
use crate::mujoco_solver::Solution;

/// The state `mj_step` carries from one step to the next, `qacc_warmstart` included: MuJoCo's solver
/// starts from it, so it is state as much as `qpos` is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MjNativeState {
    pub time: f64,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub act: Vec<f64>,
    pub qacc_warmstart: Vec<f64>,
    /// set on a returned state when MuJoCo's auto-reset fired during the step (`mj_checkPos`, `mj_checkVel`
    /// or `mj_checkAcc`): `mj_resetData` also zeroes the control, so the caller must too from here on
    pub reset: bool,
}

impl SmoothModel {
    /// `mj_EulerSkip` then `mj_advance`: the next state from this one's forward pass.
    pub(crate) fn euler(&self, sd: &SmoothData, sol: &Solution, st: &MjNativeState, eulerdamp: bool) -> MjNativeState {
        let nv = self.dof_bodyid.len();
        let damped = eulerdamp && self.dof_damping.iter().any(|&d| d > 0.0);
        let qacc = if !damped {
            sol.qacc.clone()
        } else {
            // qH = M + h·diag(damping), factorised in place
            let mut qh = sd.m.clone();
            for i in 0..nv {
                let adr = self.m_rowadr[i] + self.m_rownnz[i] - 1;
                qh[adr] = self.timestep.mul_add(self.dof_damping[i], qh[adr]);
            }
            let (qld, diag_inv) = self.factor_i(&qh);
            let qfrc: Vec<f64> = (0..nv).map(|i| sd.qfrc_smooth[i] + sol.qfrc_constraint[i]).collect();
            self.solve_ld_pub(&qld, &diag_inv, &qfrc)
        };
        self.advance(sd, st, &qacc, &sol.qacc)
    }

    /// `mj_advance`: activations, then `qvel += h·qacc`, then `mj_integratePos`, then time; `qacc_solver`
    /// becomes the next warmstart.
    pub(crate) fn advance(&self, sd: &SmoothData, st: &MjNativeState, qacc: &[f64], qacc_solver: &[f64]) -> MjNativeState {
        let h = self.timestep;
        let mut next = st.clone();
        for a in &self.actuators {
            if let Some(adr) = a.actadr {
                next.act[adr] = self.next_activation(a, st.act[adr], sd.act_dot[adr]);
            }
        }
        for (v, a) in next.qvel.iter_mut().zip(qacc) {
            *v = a.mul_add(h, *v);
        }
        self.integrate_pos(&mut next.qpos, &next.qvel, h);
        next.time += h;
        next.qacc_warmstart = qacc_solver.to_vec();
        next
    }

    /// `mj_integratePos`
    pub(crate) fn integrate_pos(&self, qpos: &mut [f64], qvel: &[f64], dt: f64) {
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let (mut padr, mut vadr) = (self.jnt_qposadr[j], self.jnt_dofadr[j]);
            match kind {
                KinJointKind::Hinge | KinJointKind::Slide => qpos[padr] = dt.mul_add(qvel[vadr], qpos[padr]),
                KinJointKind::Free | KinJointKind::Ball => {
                    if kind == KinJointKind::Free {
                        for i in 0..3 {
                            qpos[padr + i] = dt.mul_add(qvel[vadr + i], qpos[padr + i]);
                        }
                        padr += 3;
                        vadr += 3;
                    }
                    // `mju_quatIntegrate`
                    let mut axis = [qvel[vadr], qvel[vadr + 1], qvel[vadr + 2]];
                    let norm = axis[2].mul_add(axis[2], axis[0].mul_add(axis[0], axis[1] * axis[1])).sqrt();
                    if norm < 1e-15 {
                        axis = [1.0, 0.0, 0.0];
                    } else {
                        let inv = 1.0 / norm;
                        axis = axis.map(|x| x * inv);
                    }
                    let qrot = axis_angle2quat(&axis, dt * norm);
                    let mut q = [qpos[padr], qpos[padr + 1], qpos[padr + 2], qpos[padr + 3]];
                    normalize4(&mut q);
                    let q = mul_quat(&q, &qrot);
                    qpos[padr..padr + 4].copy_from_slice(&q);
                }
            }
        }
    }
}
