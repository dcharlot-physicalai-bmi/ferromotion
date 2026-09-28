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

/// `mji_mulMatVec3`, contracted.
fn mat_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [m[2].mul_add(v[2], m[0].mul_add(v[0], m[1] * v[1])), m[5].mul_add(v[2], m[3].mul_add(v[0], m[4] * v[1])), m[8].mul_add(v[2], m[6].mul_add(v[0], m[7] * v[1]))]
}

/// `mji_mulMatMat3`, contracted.
fn mat_mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    std::array::from_fn(|k| {
        let (r, c) = (k / 3, k % 3);
        a[3 * r + 2].mul_add(b[6 + c], a[3 * r].mul_add(b[c], a[3 * r + 1] * b[3 + c]))
    })
}

/// `mji_mulMatTMat3`, contracted.
fn mat_t_mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    std::array::from_fn(|k| {
        let (r, c) = (k / 3, k % 3);
        a[6 + r].mul_add(b[6 + c], a[r].mul_add(b[c], a[3 + r] * b[3 + c]))
    })
}

/// `mji_cross`, contracted.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1].mul_add(b[2], -(a[2] * b[1])), a[2].mul_add(b[0], -(a[0] * b[2])), a[0].mul_add(b[1], -(a[1] * b[0]))]
}

/// `freeBias_vel_blocks`: the two 3×3 blocks of a free body's gyroscopic derivative.
fn free_bias_vel_blocks(mass: f64, r: &[f64; 9], xi: &[f64; 9], inertia: &[f64; 3], s: &[f64; 3], qvel_rot: &[f64; 3]) -> ([f64; 9], [f64; 9]) {
    let w = mat_vec3(r, qvel_rot);
    let mut xi_i = [0.0; 9];
    for i in 0..3 {
        for k in 0..3 {
            xi_i[3 * i + k] = xi[3 * i + k] * inertia[k];
        }
    }
    let row = |a: usize, b: usize| xi_i[3 * a + 2].mul_add(xi[3 * b + 2], xi_i[3 * a].mul_add(xi[3 * b], xi_i[3 * a + 1] * xi[3 * b + 1]));
    let mut iw = [0.0; 9];
    iw[0] = row(0, 0);
    iw[4] = row(1, 1);
    iw[8] = row(2, 2);
    iw[1] = row(0, 1);
    iw[3] = iw[1];
    iw[2] = row(0, 2);
    iw[6] = iw[2];
    iw[5] = row(1, 2);
    iw[7] = iw[5];
    let ws = cross3(&w, s);
    let iww = mat_vec3(&iw, &w);
    let w_dot_s = w[2].mul_add(s[2], w[0].mul_add(s[0], w[1] * s[1]));
    let k = [
        s[0].mul_add(w[0], -w_dot_s),
        s[0].mul_add(w[1], -ws[2]),
        s[0].mul_add(w[2], ws[1]),
        s[1].mul_add(w[0], ws[2]),
        s[1].mul_add(w[1], -w_dot_s),
        s[1].mul_add(w[2], -ws[0]),
        s[2].mul_add(w[0], -ws[1]),
        s[2].mul_add(w[1], ws[0]),
        s[2].mul_add(w[2], -w_dot_s),
    ];
    let lin = mat_mat3(&k, r);
    let mut c = [0.0; 9];
    for col in 0..3 {
        let sxk0 = s[1].mul_add(k[6 + col], -(s[2] * k[3 + col]));
        let sxk1 = s[2].mul_add(k[col], -(s[0] * k[6 + col]));
        let sxk2 = s[0].mul_add(k[3 + col], -(s[1] * k[col]));
        let wxi0 = w[1].mul_add(iw[6 + col], -(w[2] * iw[3 + col]));
        let wxi1 = w[2].mul_add(iw[col], -(w[0] * iw[6 + col]));
        let wxi2 = w[0].mul_add(iw[3 + col], -(w[1] * iw[col]));
        let t0 = if col == 1 { iww[2] } else if col == 2 { -iww[1] } else { 0.0 };
        let t1 = if col == 0 { -iww[2] } else if col == 2 { iww[0] } else { 0.0 };
        let t2 = if col == 0 { iww[1] } else if col == 1 { -iww[0] } else { 0.0 };
        c[col] = (-mass).mul_add(sxk0, wxi0) + t0;
        c[3 + col] = (-mass).mul_add(sxk1, wxi1) + t1;
        c[6 + col] = (-mass).mul_add(sxk2, wxi2) + t2;
    }
    let tmp = mat_t_mat3(r, &c);
    (lin, mat_mat3(&tmp, r))
}

/// `mju_factorLU6`: in place with partial pivoting; `false` when singular.
fn factor_lu6(a: &mut [f64; 36], pivot: &mut [usize; 6]) -> bool {
    for k in 0..6 {
        pivot[k] = k;
        let mut maxval = a[k * 6 + k].abs();
        let mut maxrow = k;
        for i in k + 1..6 {
            let v = a[i * 6 + k].abs();
            if v > maxval {
                maxval = v;
                maxrow = i;
            }
        }
        if maxval < 1e-15 {
            return false;
        }
        if maxrow != k {
            pivot[k] = maxrow;
            for j in 0..6 {
                a.swap(k * 6 + j, maxrow * 6 + j);
            }
        }
        let diaginv = 1.0 / a[k * 6 + k];
        for i in k + 1..6 {
            a[i * 6 + k] *= diaginv;
            let aik = a[i * 6 + k];
            for j in k + 1..6 {
                a[i * 6 + j] = (-aik).mul_add(a[k * 6 + j], a[i * 6 + j]);
            }
        }
    }
    true
}

/// `mju_solveLU6`
fn solve_lu6(lu: &[f64; 36], b: &[f64], pivot: &[usize; 6]) -> [f64; 6] {
    let mut x = [b[0], b[1], b[2], b[3], b[4], b[5]];
    for i in 0..6 {
        if pivot[i] != i {
            x.swap(i, pivot[i]);
        }
        for j in 0..i {
            x[i] = (-lu[i * 6 + j]).mul_add(x[j], x[i]);
        }
    }
    for i in (0..6).rev() {
        for j in i + 1..6 {
            x[i] = (-lu[i * 6 + j]).mul_add(x[j], x[i]);
        }
        x[i] /= lu[i * 6 + i];
    }
    x
}

impl SmoothModel {
    /// Whether body `b` is a standalone free body (`mj_isFreeBody`): one free joint, a six-dof tree, no
    /// children (its subtree mass is its own).
    fn is_free_body(&self, b: usize) -> bool {
        let bj = &self.kin.bodies[b].joints;
        if bj.len() != 1 || bj[0].kind != KinJointKind::Free {
            return false;
        }
        let adr = self.body_dofadr[b];
        let mut n = 0;
        let mut i = adr;
        // the tree's dof count: dofs from `adr` until the next tree root
        while i < self.dof_bodyid.len() && (i == adr || self.dof_parentid[i] != -1) {
            n += 1;
            i += 1;
        }
        n == 6 && self.body_subtreemass[b] == self.body_mass[b]
    }

    /// **`mj_implicitSkip` for `implicitfast`**: `qH = M − h·qDeriv` (actuator and dof-damping derivatives,
    /// no bias), factorised; then each standalone free body's 6×6 block solved locally with its gyroscopic
    /// derivative (`mjd_freeMhat`, `mju_factorLU6`); then `mj_advance`.
    pub(crate) fn implicit_fast(&self, sd: &SmoothData, sol: &Solution, st: &MjNativeState, ctrl: &[f64]) -> Result<MjNativeState, String> {
        use crate::mujoco_actuator::{ActBias, ActDyn, ActGain};
        let nv = self.dof_bodyid.len();
        let h = self.timestep;
        let qfrc: Vec<f64> = (0..nv).map(|i| sd.qfrc_smooth[i] + sol.qfrc_constraint[i]).collect();
        // qDeriv: only its diagonal can be non-zero for the transmissions and passive forces carried here
        let mut qderiv = vec![0.0; nv];
        for (i, a) in self.actuators.iter().enumerate() {
            if let Some([lo, hi]) = a.act.forcerange {
                let f = sd.actuator_force[i];
                if f <= lo || f >= hi {
                    continue;
                }
            }
            let mut bias_vel = if a.act.bias == ActBias::Affine { a.act.biasprm[2] } else { 0.0 };
            let gain_vel = match a.act.gain {
                ActGain::Affine => a.act.gainprm[2],
                ActGain::Muscle => return Err("the implicitfast derivative of a muscle gain is not ported".into()),
                ActGain::Fixed => 0.0,
            };
            if gain_vel != 0.0 {
                let input = match (a.act.dynamics, a.actadr) {
                    (ActDyn::None, _) | (_, None) => {
                        let c = ctrl.get(i).copied().unwrap_or(0.0);
                        match a.act.ctrlrange {
                            Some([lo, hi]) => c.clamp(lo, hi),
                            None => c,
                        }
                    }
                    (_, Some(adr)) => {
                        let x = st.act[adr];
                        if a.act.actearly { self.next_activation(a, x, sd.act_dot[adr]) } else { x }
                    }
                };
                bias_vel = gain_vel.mul_add(input, bias_vel);
            }
            if bias_vel != 0.0 {
                let g = a.act.gear;
                qderiv[a.dofadr] = (g * bias_vel).mul_add(g, qderiv[a.dofadr]);
            }
        }
        for i in 0..nv {
            qderiv[i] -= self.dof_damping[i];
        }
        // qH = M + qDeriv·(−h) on M's pattern (off the diagonal qDeriv is zero)
        let mut qh: Vec<f64> = sd.m.clone();
        for i in 0..nv {
            for adr in self.m_rowadr[i]..self.m_rowadr[i] + self.m_rownnz[i] {
                let q = if self.m_colind[adr] == i { qderiv[i] } else { 0.0 };
                qh[adr] = q.mul_add(-h, sd.m[adr]);
            }
        }
        // standalone free bodies: their rows back to M
        let free: Vec<usize> = (0..self.jnt_type.len()).filter(|&j| self.jnt_type[j] == KinJointKind::Free && self.is_free_body(self.jnt_bodyid[j])).collect();
        for &j in &free {
            let adr = self.jnt_dofadr[j];
            for r in 0..6 {
                let (ra, rn) = (self.m_rowadr[adr + r], self.m_rownnz[adr + r]);
                qh[ra..ra + rn].copy_from_slice(&sd.m[ra..ra + rn]);
            }
        }
        let (qld, diag_inv) = self.factor_i(&qh);
        let mut qacc = self.solve_ld_pub(&qld, &diag_inv, &qfrc);
        // `mjd_freeMhat` and the local solve
        for &j in &free {
            let b = self.jnt_bodyid[j];
            let adr = self.jnt_dofadr[j];
            let mut a = [0.0; 36];
            for r in 0..6 {
                for k in self.m_rowadr[adr + r]..self.m_rowadr[adr + r] + self.m_rownnz[adr + r] {
                    let c = self.m_colind[k] - adr;
                    a[6 * r + c] = sd.m[k];
                    a[6 * c + r] = sd.m[k];
                }
            }
            // A −= h·qDeriv over the block (only its diagonal is non-zero)
            for r in 0..6 {
                for c in 0..6 {
                    let q = if r == c { qderiv[adr + r] } else { 0.0 };
                    a[6 * r + c] = (-h).mul_add(q, a[6 * r + c]);
                }
            }
            let k = &sd.kin;
            let s = [k.xipos[b][0] - k.xpos[b][0], k.xipos[b][1] - k.xpos[b][1], k.xipos[b][2] - k.xpos[b][2]];
            let mass = self.body_mass[b];
            let (lin, rot) = free_bias_vel_blocks(mass, &k.xmat[b], &k.ximat[b], &self.body_inertia[b], &s, &[st.qvel[adr + 3], st.qvel[adr + 4], st.qvel[adr + 5]]);
            let h_mass = -h * mass;
            for r in 0..3 {
                for c in 0..3 {
                    a[6 * r + 3 + c] = h_mass.mul_add(lin[3 * r + c], a[6 * r + 3 + c]);
                    a[6 * (3 + r) + 3 + c] = h.mul_add(rot[3 * r + c], a[6 * (3 + r) + 3 + c]);
                }
            }
            let mut pivot = [0usize; 6];
            if factor_lu6(&mut a, &mut pivot) {
                let x = solve_lu6(&a, &qfrc[adr..adr + 6], &pivot);
                qacc[adr..adr + 6].copy_from_slice(&x);
            }
        }
        Ok(self.advance(sd, st, &qacc, &sol.qacc))
    }
}
