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

/// `5491d70f` `bortzCoef`. Horner form `k[i] + xx·res` below `|x| = 1.75`.
fn bortz_coef(xx: f64) -> f64 {
    const K: [f64; 9] = [
        0.083333333333333384,
        0.0013888888888859755,
        3.3068783094433201e-05,
        8.2671949071520734e-07,
        2.0876900642039142e-08,
        5.2828494303844588e-10,
        1.3455485951057888e-11,
        3.1615515500778219e-13,
        1.230313397310169e-14,
    ];
    if xx < 1.75 * 1.75 {
        let mut res = K[8];
        for i in (0..8).rev() {
            res = K[i] + xx * res;
        }
        return res;
    }
    let x = xx.sqrt();
    (1.0 - 0.5 * x / (0.5 * x).tan()) / xx
}

/// `5491d70f` `mju_derivExpmap`. `res` may alias `vel`.
fn deriv_expmap(expmap: &[f64; 3], vel: &[f64; 3]) -> [f64; 3] {
    let mut r = *expmap;
    let mut xx = (r[0] * r[0] + r[1] * r[1]) + r[2] * r[2];
    const PI: f64 = std::f64::consts::PI;
    if xx > PI * PI {
        let x = xx.sqrt();
        let scale = (x - 2.0 * PI * (x / (2.0 * PI) + 0.5).floor()) / x;
        r = [r[0] * scale, r[1] * scale, r[2] * scale];
        xx = (r[0] * r[0] + r[1] * r[1]) + r[2] * r[2];
    }
    let c = bortz_coef(xx);
    let rxv = cross3(&r, vel);
    let rxrxv = cross3(&r, &rxv);
    [
        (vel[0] + 0.5 * rxv[0]) + c * rxrxv[0],
        (vel[1] + 0.5 * rxv[1]) + c * rxrxv[1],
        (vel[2] + 0.5 * rxv[2]) + c * rxrxv[2],
    ]
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
            let separate = crate::fp_contract::contract_off_line();
            for i in 0..nv {
                let adr = self.m_rowadr[i] + self.m_rownnz[i] - 1;
                // `qH[diag] += h * damp`. `64156abe` does `(h * damp) + qH`.
                let damp = self.dof_damping[i];
                qh[adr] = if separate { self.timestep * damp + qh[adr] } else { self.timestep.mul_add(damp, qh[adr]) };
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
        self.advance_with(st, &sd.act_dot, qacc, None, qacc_solver)
    }

    /// `mj_advance` with the rates given: `act_dot` for the activations, `qacc` for the velocities, and — when
    /// given, as `mj_RungeKutta` gives it — the velocity the positions are integrated with in place of the
    /// new `qvel`.
    pub(crate) fn advance_with(&self, st: &MjNativeState, act_dot: &[f64], qacc: &[f64], qvel_pos: Option<&[f64]>, qacc_solver: &[f64]) -> MjNativeState {
        let h = self.timestep;
        let mut next = st.clone();
        // every slot of every stateful actuator (a `mujoco.pid` plugin keeps up to two), in order
        for a in &self.actuators {
            if let Some(adr) = a.actadr {
                for j in adr..adr + a.act.actnum() {
                    next.act[j] = self.next_activation(a, st.act[j], act_dot[j]);
                }
            }
        }
        // `mju_addToScl(qvel, qacc, h)`. Wide prefix `qvel + h·qacc` from n = 6;
        // the tail, and every lane below that, is `h·qacc + qvel`.
        let separate = crate::fp_contract::contract_off_line();
        if separate {
            crate::mujoco_solver::add_to_scl(&mut next.qvel, qacc, h, 6);
        } else {
            for (v, a) in next.qvel.iter_mut().zip(qacc) {
                *v = a.mul_add(h, *v);
            }
        }
        let v = qvel_pos.map_or_else(|| next.qvel.clone(), <[f64]>::to_vec);
        self.integrate_pos(&mut next.qpos, &v, h);
        next.time += h;
        next.qacc_warmstart = qacc_solver.to_vec();
        next
    }

    /// `5491d70f` `mju_derivExpmap`: `dexp⁻¹₋rot(vel)`. Below `|rot| = 1.75` the Bortz
    /// coefficient is the Horner polynomial `k[i] + xx·res`.
    pub(crate) fn rk_correct_vel(&self, vel: &mut [f64], dqvel: &[f64], h: f64) {
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let mut vadr = self.jnt_dofadr[j];
            if kind == crate::mujoco_kinematics::KinJointKind::Free {
                vadr += 3;
            } else if kind != crate::mujoco_kinematics::KinJointKind::Ball {
                continue;
            }
            let rot = [h * dqvel[vadr], h * dqvel[vadr + 1], h * dqvel[vadr + 2]];
            let ang = [vel[vadr], vel[vadr + 1], vel[vadr + 2]];
            let out = deriv_expmap(&rot, &ang);
            vel[vadr] = out[0];
            vel[vadr + 1] = out[1];
            vel[vadr + 2] = out[2];
        }
    }

    /// `mj_integratePos`
    pub(crate) fn integrate_pos(&self, qpos: &mut [f64], qvel: &[f64], dt: f64) {
        let separate = crate::fp_contract::contract_off_line();
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let (mut padr, mut vadr) = (self.jnt_qposadr[j], self.jnt_dofadr[j]);
            match kind {
                // `mj_integratePos`: `qpos += dt * qvel`. The published path fuses that.
                KinJointKind::Hinge | KinJointKind::Slide => {
                    // `64156abe`: `(dt * qvel) + qpos`.
                    qpos[padr] = if separate { dt * qvel[vadr] + qpos[padr] } else { dt.mul_add(qvel[vadr], qpos[padr]) };
                }
                KinJointKind::Free | KinJointKind::Ball => {
                    if kind == KinJointKind::Free {
                        for i in 0..3 {
                            qpos[padr + i] = if separate { dt * qvel[vadr + i] + qpos[padr + i] } else { dt.mul_add(qvel[vadr + i], qpos[padr + i]) };
                        }
                        padr += 3;
                        vadr += 3;
                    }
                    if separate {
                        quat_integrate_off(&mut qpos[padr..padr + 4], &[qvel[vadr], qvel[vadr + 1], qvel[vadr + 2]], dt);
                    } else {
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
}

/// `mju_quatIntegrate` as `64156abe` inlines it. The axis norm is `(y y + x x) + z z`.
/// The multiply is `mju_mulQuat` except lane 1, which adds the cross term on the left:
/// `(p23 + (p10 + p01)) - p32`. `5491d70f` and `4849b2dd` pair each lane the other way,
/// inside `mju_quatIntegrate` rather than through `mju_mulQuat`. There is no second
/// normalize after the multiply.
fn quat_integrate_off(q: &mut [f64], vel: &[f64], scale: f64) {
    let norm = ((vel[1] * vel[1] + vel[0] * vel[0]) + vel[2] * vel[2]).sqrt();
    let axis = if norm < 1e-15 {
        [1.0, 0.0, 0.0]
    } else {
        let inv = 1.0 / norm;
        [vel[0] * inv, vel[1] * inv, vel[2] * inv]
    };
    let qrot = axis_angle2quat(&axis, scale * norm);
    let mut a = [q[0], q[1], q[2], q[3]];
    normalize4(&mut a);
    let b = qrot;
    let (p00, p11, p22, p33) = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
    let (p01, p10, p23, p32) = (a[0] * b[1], a[1] * b[0], a[2] * b[3], a[3] * b[2]);
    let (p02, p13, p20, p31) = (a[0] * b[2], a[1] * b[3], a[2] * b[0], a[3] * b[1]);
    let (p03, p12, p21, p30) = (a[0] * b[3], a[1] * b[2], a[2] * b[1], a[3] * b[0]);
    // `5491d70f` and `4849b2dd` inline the product in `mju_quatIntegrate` and pair the
    // terms the other way. `64156abe` keeps the left-associated differences below.
    let out = if crate::fp_contract::oracle_4_0_schedules() {
        [(-p22 - p33) + (p00 - p11), (p23 - p32) + (p10 + p01), (p31 + p20) + (p02 - p13), (p30 - p21) + (p12 + p03)]
    } else {
        [((p00 - p11) - p22) - p33, (p23 + (p10 + p01)) - p32, ((p02 - p13) + p20) + p31, ((p12 + p03) - p21) + p30]
    };
    q.copy_from_slice(&out);
}

fn off() -> bool {
    crate::fp_contract::contract_off()
}

/// 4.0.0 and 4.0.1 compile these helpers with no FMA. 3.13 stays on [`off`].
fn unfused() -> bool {
    off() || crate::fp_contract::oracle_4_0_schedules()
}

/// `a·b` as `mji_mulMatVec3` writes it: `(a0 b0 + a1 b1) + a2 b2`.
fn dot3_row(a0: f64, b0: f64, a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    if unfused() { (a0 * b0 + a1 * b1) + a2 * b2 } else { a2.mul_add(b2, a0.mul_add(b0, a1 * b1)) }
}

/// `mji_mulMatVec3`.
fn mat_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [dot3_row(m[0], v[0], m[1], v[1], m[2], v[2]), dot3_row(m[3], v[0], m[4], v[1], m[5], v[2]), dot3_row(m[6], v[0], m[7], v[1], m[8], v[2])]
}

/// `mji_mulMatMat3`.
fn mat_mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    std::array::from_fn(|k| {
        let (r, c) = (k / 3, k % 3);
        dot3_row(a[3 * r], b[c], a[3 * r + 1], b[3 + c], a[3 * r + 2], b[6 + c])
    })
}

/// `mji_mulMatTMat3`.
fn mat_t_mat3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    std::array::from_fn(|k| {
        let (r, c) = (k / 3, k % 3);
        dot3_row(a[r], b[c], a[3 + r], b[3 + c], a[6 + r], b[6 + c])
    })
}

/// `mji_cross`: `a1 b2 − a2 b1`.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    if unfused() {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    } else {
        [a[1].mul_add(b[2], -(a[2] * b[1])), a[2].mul_add(b[0], -(a[0] * b[2])), a[0].mul_add(b[1], -(a[1] * b[0]))]
    }
}

/// `freeBias_vel_blocks`: the two 3×3 blocks of a free body's gyroscopic derivative.
fn free_bias_vel_blocks(mass: f64, r: &[f64; 9], xi: &[f64; 9], inertia: &[f64; 3], s: &[f64; 3], qvel_rot: &[f64; 3]) -> ([f64; 9], [f64; 9]) {
    let mut xi_i = [0.0; 9];
    for i in 0..3 {
        for k in 0..3 {
            xi_i[3 * i + k] = xi[3 * i + k] * inertia[k];
        }
    }
    let row = |a: usize, b: usize| {
        dot3_row(xi_i[3 * a], xi[3 * b], xi_i[3 * a + 1], xi[3 * b + 1], xi_i[3 * a + 2], xi[3 * b + 2])
    };
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
    free_bias_from_iw(mass, r, &iw, s, qvel_rot)
}

/// `freeBias_vel_blocks` once `Iw` is already the world inertia about the CoM.
fn free_bias_from_iw(mass: f64, r: &[f64; 9], iw: &[f64; 9], s: &[f64; 3], qvel_rot: &[f64; 3]) -> ([f64; 9], [f64; 9]) {
    let w = mat_vec3(r, qvel_rot);
    let ws = cross3(&w, s);
    let iww = mat_vec3(&iw, &w);
    let w_dot_s = dot3_row(w[0], s[0], w[1], s[1], w[2], s[2]);
    let k = if unfused() {
        [s[0] * w[0] - w_dot_s, s[0] * w[1] - ws[2], s[0] * w[2] + ws[1], s[1] * w[0] + ws[2], s[1] * w[1] - w_dot_s, s[1] * w[2] - ws[0], s[2] * w[0] - ws[1], s[2] * w[1] + ws[0], s[2] * w[2] - w_dot_s]
    } else {
        [
            s[0].mul_add(w[0], -w_dot_s),
            s[0].mul_add(w[1], -ws[2]),
            s[0].mul_add(w[2], ws[1]),
            s[1].mul_add(w[0], ws[2]),
            s[1].mul_add(w[1], -w_dot_s),
            s[1].mul_add(w[2], -ws[0]),
            s[2].mul_add(w[0], -ws[1]),
            s[2].mul_add(w[1], ws[0]),
            s[2].mul_add(w[2], -w_dot_s),
        ]
    };
    let lin = mat_mat3(&k, r);
    let mut c = [0.0; 9];
    for col in 0..3 {
        let (sxk0, sxk1, sxk2, wxi0, wxi1, wxi2) = if unfused() {
            (s[1] * k[6 + col] - s[2] * k[3 + col], s[2] * k[col] - s[0] * k[6 + col], s[0] * k[3 + col] - s[1] * k[col], w[1] * iw[6 + col] - w[2] * iw[3 + col], w[2] * iw[col] - w[0] * iw[6 + col], w[0] * iw[3 + col] - w[1] * iw[col])
        } else {
            (
                s[1].mul_add(k[6 + col], -(s[2] * k[3 + col])),
                s[2].mul_add(k[col], -(s[0] * k[6 + col])),
                s[0].mul_add(k[3 + col], -(s[1] * k[col])),
                w[1].mul_add(iw[6 + col], -(w[2] * iw[3 + col])),
                w[2].mul_add(iw[col], -(w[0] * iw[6 + col])),
                w[0].mul_add(iw[3 + col], -(w[1] * iw[col])),
            )
        };
        let t0 = if col == 1 { iww[2] } else if col == 2 { -iww[1] } else { 0.0 };
        let t1 = if col == 0 { -iww[2] } else if col == 2 { iww[0] } else { 0.0 };
        let t2 = if col == 0 { iww[1] } else if col == 1 { -iww[0] } else { 0.0 };
        // `(-mass)·(s×K) + (w×Iw) + t`.
        let acc = |sxk, wxi, t| if unfused() { ((-mass) * sxk + wxi) + t } else { (-mass).mul_add(sxk, wxi) + t };
        c[col] = acc(sxk0, wxi0, t0);
        c[3 + col] = acc(sxk1, wxi1, t1);
        c[6 + col] = acc(sxk2, wxi2, t2);
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
                // `a − aik·pivot`. `64156abe` subtracts the product.
                a[i * 6 + j] = if unfused() { a[i * 6 + j] - aik * a[k * 6 + j] } else { (-aik).mul_add(a[k * 6 + j], a[i * 6 + j]) };
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
            x[i] = if unfused() { x[i] - lu[i * 6 + j] * x[j] } else { (-lu[i * 6 + j]).mul_add(x[j], x[i]) };
        }
    }
    for i in (0..6).rev() {
        for j in i + 1..6 {
            x[i] = if unfused() { x[i] - lu[i * 6 + j] * x[j] } else { (-lu[i * 6 + j]).mul_add(x[j], x[i]) };
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
        // 3.13 also requires the subtree mass to be the body's own. 4.0.0 and 4.0.1
        // take any awake free joint whose tree is six dofs.
        n == 6 && (crate::fp_contract::oracle_4_0_schedules() || self.body_subtreemass[b] == self.body_mass[b])
    }

    /// **`mj_implicitSkip` for `implicitfast`**: `qH = M − h·qDeriv` (actuator and dof-damping derivatives,
    /// no bias), factorised; then each standalone free body's 6×6 block solved locally with its gyroscopic
    /// derivative (`mjd_freeMhat`, `mju_factorLU6`); then `mj_advance`.
    pub(crate) fn implicit_fast(&self, sd: &SmoothData, sol: &Solution, st: &MjNativeState, ctrl: &[f64]) -> Result<MjNativeState, String> {
        use crate::mujoco_actuator::{ActBias, ActDyn, ActGain};
        let nv = self.dof_bodyid.len();
        let h = self.timestep;
        let qfrc: Vec<f64> = (0..nv).map(|i| sd.qfrc_smooth[i] + sol.qfrc_constraint[i]).collect();
        // qDeriv on the entries `mapD2M` gathers — M's lower pattern, by M's address (the diagonal is each
        // row's last). A joint transmission and a damper touch only the diagonal; a fixed tendon's
        // `moment'·b·moment` reaches every pair of its dofs, and of those only the ones M stores survive.
        let mut qderiv = vec![0.0; self.m_colind.len()];
        let separate = crate::fp_contract::contract_off_line();
        let diag = |i: usize| self.m_rowadr[i] + self.m_rownnz[i] - 1;
        let at = |r: usize, c: usize| (self.m_rowadr[r]..self.m_rowadr[r] + self.m_rownnz[r]).find(|&k| self.m_colind[k] == c);
        for (i, a) in self.actuators.iter().enumerate() {
            if let Some([lo, hi]) = a.act.forcerange {
                let f = sd.actuator_force[i];
                if f <= lo || f >= hi {
                    continue;
                }
            }
            // `mjd_actuatorDerivSkip` in `5491d70f` and `4849b2dd`: a joint transmission whose
            // summed force sits on `jnt_actfrcrange` contributes no velocity derivative. `64156abe`
            // has no such check, so this stays on the 4.x profiles.
            if crate::fp_contract::oracle_4_0_schedules() && a.tendon.is_none() && a.site.is_none() && a.body.is_none() {
                let j = self.dof_jntid[a.dofadr];
                if let Some([lo, hi]) = self.jnt_actfrcrange[j] {
                    let f = sd.qfrc_actuator[a.dofadr];
                    if f <= lo || f >= hi {
                        continue;
                    }
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
            // `addJTBJSparse`: row r += moment·(moment_r·b), at the columns row r stores
            if bias_vel != 0.0 {
                let (cols, vals) = &sd.moment[i];
                for (&r, &mr) in cols.iter().zip(vals) {
                    let scl = mr * bias_vel;
                    for (&c, &mc) in cols.iter().zip(vals) {
                        if let Some(k) = at(r, c) {
                            qderiv[k] = if separate { qderiv[k] + scl * mc } else { scl.mul_add(mc, qderiv[k]) };
                        }
                    }
                }
            }
        }
        for i in 0..nv {
            qderiv[diag(i)] -= self.dof_damping[i];
        }
        // tendon dampers: `addJTBJSparse` of `ten_J` with `B = −damping`
        for (t, jrow) in self.tendons.iter().zip(&sd.ten_j) {
            let Some(t) = t else { continue };
            let b = -t.damping;
            if b == 0.0 {
                continue;
            }
            for (&r, &jr) in t.colind.iter().zip(jrow) {
                let scl = jr * b;
                for (&c, &jc) in t.colind.iter().zip(jrow) {
                    if let Some(k) = at(r, c) {
                        qderiv[k] = if separate { qderiv[k] + scl * jc } else { scl.mul_add(jc, qderiv[k]) };
                    }
                }
            }
        }
        // qH = M + qDeriv·(−h) on M's pattern. Published path fuses that.
        // `mju_addScl` in `5491d70f` / `4849b2dd` is `(qDeriv * -h) + M`. The 3.13 contract-off
        // library adds `M` first.
        let qh: Vec<f64> = qderiv.iter().zip(&sd.m).map(|(q, m)| {
            if crate::fp_contract::oracle_4_0_schedules() {
                *q * -h + *m
            } else if separate {
                *m + *q * -h
            } else {
                q.mul_add(-h, *m)
            }
        }).collect();
        let mut qh = qh;
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
                    let q = if r == c { qderiv[diag(adr + r)] } else { 0.0 };
                    // `A −= h·qDeriv`. 4.0.0 and 4.0.1 `mjd_freeMhat` is not this 3.13 schedule.
                    a[6 * r + c] = if unfused() { a[6 * r + c] - h * q } else { (-h).mul_add(q, a[6 * r + c]) };
                }
            }
            let k = &sd.kin;
            if crate::fp_contract::oracle_4_0_schedules() {
                // `mjd_freeBias_vel` in 4849b2dd and 5491d70f: mass is `crb[9]`, `Iw` is the
                // composite about the subtree com, `s = subtree_com − xpos`. `mjd_freeMhat`
                // then does `mju_addToScl(A, B, h, 36)`, which is `A + h·B` on every lane.
                let com = sd.subtree_com[b];
                let s = [com[0] - k.xpos[b][0], com[1] - k.xpos[b][1], com[2] - k.xpos[b][2]];
                let crb = sd.crb[b];
                let mass = crb[9];
                let iw = [crb[0], crb[3], crb[4], crb[3], crb[1], crb[5], crb[4], crb[5], crb[2]];
                let (lin, rot) = free_bias_from_iw(mass, &k.xmat[b], &iw, &s, &[st.qvel[adr + 3], st.qvel[adr + 4], st.qvel[adr + 5]]);
                let mut bias = [0.0; 36];
                for r in 0..3 {
                    for c in 0..3 {
                        bias[6 * r + 3 + c] = -mass * lin[3 * r + c];
                        bias[6 * (r + 3) + 3 + c] = rot[3 * r + c];
                    }
                }
                crate::mujoco_solver::add_to_scl(&mut a, &bias, h, 6);
            } else {
                let s = [k.xipos[b][0] - k.xpos[b][0], k.xipos[b][1] - k.xpos[b][1], k.xipos[b][2] - k.xpos[b][2]];
                let mass = self.body_mass[b];
                let (lin, rot) = free_bias_vel_blocks(mass, &k.xmat[b], &k.ximat[b], &self.body_inertia[b], &s, &[st.qvel[adr + 3], st.qvel[adr + 4], st.qvel[adr + 5]]);
                let h_mass = -h * mass;
                for r in 0..3 {
                    for c in 0..3 {
                        // `A += h_mass·lin` and `A += h·rot`. This is the 3.13 inlined form.
                        if off() {
                            a[6 * r + 3 + c] += h_mass * lin[3 * r + c];
                            a[6 * (3 + r) + 3 + c] += h * rot[3 * r + c];
                        } else {
                            a[6 * r + 3 + c] = h_mass.mul_add(lin[3 * r + c], a[6 * r + 3 + c]);
                            a[6 * (3 + r) + 3 + c] = h.mul_add(rot[3 * r + c], a[6 * (3 + r) + 3 + c]);
                        }
                    }
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
