//! **MuJoCo's smooth dynamics in MuJoCo's own coordinates and arithmetic** — `mj_comPos`, `mj_crb`,
//! `mj_factorM`, `mj_comVel` and `mj_rne` from `engine_core_smooth.c` (3.13.0), on the compiled model's own
//! arrays: the composite inertias `cinert` about each tree's centre of mass, the motion dofs `cdof`, the
//! mass matrix in MuJoCo's sparse layout (`M_rownnz`, `M_rowadr`, `M_colind`) and its `L'DL` factor.
//!
//! The rest of this crate computes the same quantities in its own coordinates (a free base as Euler angles,
//! jointless bodies welded into the link they ride on), which agree with MuJoCo's to round-off and no
//! further. That is not enough to follow MuJoCo's trajectory: its step is not continuous everywhere
//! (see `examples/menagerie_step.rs`), so a state one ulp off can land somewhere else. This module is the
//! first stage of computing MuJoCo's state to the bit.
//!
//! ⛔ Contracted as clang contracts MuJoCo's C on arm64 (see [`crate::mujoco_ccd`]): the left product of
//! `a*b + c*d` is fused, each later product of a chain is fused into the running sum, `s += a*b` is
//! `fma(a, b, s)`, and `x - p*q` is `fma(-p, q, x)`.

use crate::mujoco_kinematics::{KinJointKind, KinState, MjKinematics, SameFrame};

/// `mjMINVAL`
const MJ_MINVAL: f64 = 1e-15;

/// `mju_mulInertVec`: a 6-D motion vector through a com-based inertia (`i` = the 10 numbers of `cinert`).
fn mul_inert_vec(i: &[f64; 10], v: &[f64; 6]) -> [f64; 6] {
    [
        i[7].mul_add(v[5], (-i[8]).mul_add(v[4], i[4].mul_add(v[2], i[0].mul_add(v[0], i[3] * v[1])))),
        (-i[6]).mul_add(v[5], i[8].mul_add(v[3], i[5].mul_add(v[2], i[3].mul_add(v[0], i[1] * v[1])))),
        i[6].mul_add(v[4], (-i[7]).mul_add(v[3], i[2].mul_add(v[2], i[4].mul_add(v[0], i[5] * v[1])))),
        i[9].mul_add(v[3], i[8].mul_add(v[1], -(i[7] * v[2]))),
        i[9].mul_add(v[4], i[6].mul_add(v[2], -(i[8] * v[0]))),
        i[9].mul_add(v[5], i[7].mul_add(v[0], -(i[6] * v[1]))),
    ]
}

/// `mji_dot6`: `((v0·w0 + v2·w2) + (v1·w1 + v3·w3)) + (v4·w4 + v5·w5)`, each pair contracted.
fn dot6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    (a[0].mul_add(b[0], a[2] * b[2]) + a[1].mul_add(b[1], a[3] * b[3])) + a[4].mul_add(b[4], a[5] * b[5])
}

/// `mju_dot` at `n = 6`: four running sums, then the two left over.
fn dot_n6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let (r0, r1, r2, r3) = (a[0].mul_add(b[0], 0.0), a[1].mul_add(b[1], 0.0), a[2].mul_add(b[2], 0.0), a[3].mul_add(b[3], 0.0));
    ((r0 + r2) + (r1 + r3)) + a[4].mul_add(b[4], a[5] * b[5])
}

/// `mji_cross`, contracted.
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1].mul_add(b[2], -(a[2] * b[1])), a[2].mul_add(b[0], -(a[0] * b[2])), a[0].mul_add(b[1], -(a[1] * b[0]))]
}

/// `mji_crossMotion`: `vel × v` for motion vectors.
fn cross_motion(vel: &[f64; 6], v: &[f64; 6]) -> [f64; 6] {
    let mut r = [
        (-vel[2]).mul_add(v[1], vel[1] * v[2]),
        vel[2].mul_add(v[0], -(vel[0] * v[2])),
        (-vel[1]).mul_add(v[0], vel[0] * v[1]),
        (-vel[2]).mul_add(v[4], vel[1] * v[5]),
        vel[2].mul_add(v[3], -(vel[0] * v[5])),
        (-vel[1]).mul_add(v[3], vel[0] * v[4]),
    ];
    r[3] += (-vel[5]).mul_add(v[1], vel[4] * v[2]);
    r[4] += vel[5].mul_add(v[0], -(vel[3] * v[2]));
    r[5] += (-vel[4]).mul_add(v[0], vel[3] * v[1]);
    r
}

/// `mji_crossForce`: `vel ×* f` for force vectors.
fn cross_force(vel: &[f64; 6], f: &[f64; 6]) -> [f64; 6] {
    let mut r = [
        (-vel[2]).mul_add(f[1], vel[1] * f[2]),
        vel[2].mul_add(f[0], -(vel[0] * f[2])),
        (-vel[1]).mul_add(f[0], vel[0] * f[1]),
        (-vel[2]).mul_add(f[4], vel[1] * f[5]),
        vel[2].mul_add(f[3], -(vel[0] * f[5])),
        (-vel[1]).mul_add(f[3], vel[0] * f[4]),
    ];
    r[0] += (-vel[5]).mul_add(f[4], vel[4] * f[5]);
    r[1] += vel[5].mul_add(f[3], -(vel[3] * f[5]));
    r[2] += (-vel[4]).mul_add(f[3], vel[3] * f[4]);
    r
}

/// `mju_inertCom`: a body's principal inertia `inert` in frame `mat`, mass `mass`, moved by `dif` to a
/// com-based 10-vector (the rotational 3×3 as `xx yy zz xy xz yz`, then `mass·dif`, then `mass`).
fn inert_com(inert: &[f64; 3], mat: &[f64; 9], dif: &[f64; 3], mass: f64) -> [f64; 10] {
    let tmp = [mat[0] * inert[0], mat[3] * inert[0], mat[6] * inert[0], mat[1] * inert[1], mat[4] * inert[1], mat[7] * inert[1], mat[2] * inert[2], mat[5] * inert[2], mat[8] * inert[2]];
    let mut r = [0.0; 10];
    r[0] = mat[2].mul_add(tmp[6], mat[0].mul_add(tmp[0], mat[1] * tmp[3]));
    r[1] = mat[5].mul_add(tmp[7], mat[3].mul_add(tmp[1], mat[4] * tmp[4]));
    r[2] = mat[8].mul_add(tmp[8], mat[6].mul_add(tmp[2], mat[7] * tmp[5]));
    r[3] = mat[2].mul_add(tmp[7], mat[0].mul_add(tmp[1], mat[1] * tmp[4]));
    r[4] = mat[2].mul_add(tmp[8], mat[0].mul_add(tmp[2], mat[1] * tmp[5]));
    r[5] = mat[5].mul_add(tmp[8], mat[3].mul_add(tmp[2], mat[4] * tmp[5]));
    r[0] = mass.mul_add(dif[1].mul_add(dif[1], dif[2] * dif[2]), r[0]);
    r[1] = mass.mul_add(dif[0].mul_add(dif[0], dif[2] * dif[2]), r[1]);
    r[2] = mass.mul_add(dif[0].mul_add(dif[0], dif[1] * dif[1]), r[2]);
    r[3] = (-(mass * dif[0])).mul_add(dif[1], r[3]);
    r[4] = (-(mass * dif[0])).mul_add(dif[2], r[4]);
    r[5] = (-(mass * dif[1])).mul_add(dif[2], r[5]);
    r[6] = mass * dif[0];
    r[7] = mass * dif[1];
    r[8] = mass * dif[2];
    r[9] = mass;
    r
}

/// `mju_dofCom`: a hinge's axis about `offset` (a slide has none).
fn dof_com(axis: &[f64; 3], offset: Option<&[f64; 3]>) -> [f64; 6] {
    match offset {
        Some(o) => {
            let c = cross(axis, o);
            [axis[0], axis[1], axis[2], c[0], c[1], c[2]]
        }
        None => [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]],
    }
}

/// `mju_mulDofVec`: `Σ dofₖ·vecₖ` — one dof scaled, several accumulated from zero (`mju_mulMatTVec`, which
/// skips a zero coefficient).
fn mul_dof_vec(dof: &[[f64; 6]], vec: &[f64]) -> [f64; 6] {
    match dof.len() {
        0 => [0.0; 6],
        1 => dof[0].map(|x| x * vec[0]),
        _ => {
            let mut r = [0.0; 6];
            for (d, &v) in dof.iter().zip(vec) {
                if v != 0.0 {
                    for k in 0..6 {
                        r[k] = d[k].mul_add(v, r[k]);
                    }
                }
            }
            r
        }
    }
}

/// The compiled model's arrays the smooth dynamics read, bodies, joints and dofs in MuJoCo's order.
#[derive(Clone, Debug, Default)]
pub(crate) struct SmoothModel {
    pub(crate) kin: MjKinematics,
    pub(crate) body_rootid: Vec<usize>,
    pub(crate) body_jntadr: Vec<usize>,
    pub(crate) body_dofadr: Vec<usize>,
    pub(crate) body_dofnum: Vec<usize>,
    pub(crate) body_simple: Vec<i32>,
    pub(crate) body_mass: Vec<f64>,
    pub(crate) body_subtreemass: Vec<f64>,
    pub(crate) body_inertia: Vec<[f64; 3]>,
    pub(crate) jnt_type: Vec<KinJointKind>,
    pub(crate) jnt_dofadr: Vec<usize>,
    pub(crate) jnt_bodyid: Vec<usize>,
    pub(crate) dof_bodyid: Vec<usize>,
    pub(crate) dof_jntid: Vec<usize>,
    pub(crate) dof_parentid: Vec<i32>,
    pub(crate) dof_simplenum: Vec<i32>,
    pub(crate) dof_armature: Vec<f64>,
    pub(crate) dof_m0: Vec<f64>,
    pub(crate) m_rownnz: Vec<usize>,
    pub(crate) m_rowadr: Vec<usize>,
    pub(crate) m_colind: Vec<usize>,
    pub(crate) gravity: [f64; 3],
}

/// What the smooth stages leave in `mjData`.
#[derive(Clone, Debug)]
pub(crate) struct SmoothData {
    pub(crate) kin: KinState,
    pub(crate) subtree_com: Vec<[f64; 3]>,
    pub(crate) cinert: Vec<[f64; 10]>,
    pub(crate) cdof: Vec<[f64; 6]>,
    pub(crate) crb: Vec<[f64; 10]>,
    /// the mass matrix, `M_rowadr`/`M_colind` layout
    pub(crate) m: Vec<f64>,
    pub(crate) qld: Vec<f64>,
    pub(crate) qld_diag_inv: Vec<f64>,
    pub(crate) cvel: Vec<[f64; 6]>,
    pub(crate) cdof_dot: Vec<[f64; 6]>,
    pub(crate) qfrc_bias: Vec<f64>,
}

impl SmoothModel {
    /// The compiler's arrays from the kinematic model, each body's `body_mass` and principal `body_inertia`,
    /// each joint's armature (joints in MuJoCo's order), `qpos0` and `<option gravity>`. `dof_M0` is what
    /// `mj_setConst` makes of them: the composite inertia at `qpos0`.
    pub(crate) fn new(kin: &MjKinematics, body_mass: Vec<f64>, body_inertia: Vec<[f64; 3]>, jnt_armature: &[f64], qpos0: &[f64], gravity: [f64; 3]) -> SmoothModel {
        let nbody = kin.bodies.len();
        let mut m = SmoothModel { kin: kin.clone(), body_mass, body_inertia, gravity, ..Default::default() };
        let parent: Vec<usize> = kin.bodies.iter().map(|b| b.parent).collect();
        let width = |k: KinJointKind| match k {
            KinJointKind::Free => 6,
            KinJointKind::Ball => 3,
            _ => 1,
        };
        // rootid, joint and dof numbering, dof_parentid through each body's last dof
        let mut lastdof = vec![-1i32; nbody];
        for (i, b) in kin.bodies.iter().enumerate() {
            m.body_rootid.push(if i == 0 || parent[i] == 0 { i } else { m.body_rootid[parent[i]] });
            m.body_jntadr.push(m.jnt_type.len());
            m.body_dofadr.push(m.dof_bodyid.len());
            let mut last = if i == 0 { -1 } else { lastdof[parent[i]] };
            for j in &b.joints {
                let jid = m.jnt_type.len();
                m.jnt_type.push(j.kind);
                m.jnt_dofadr.push(m.dof_bodyid.len());
                m.jnt_bodyid.push(i);
                for _ in 0..width(j.kind) {
                    m.dof_parentid.push(last);
                    last = m.dof_bodyid.len() as i32;
                    m.dof_bodyid.push(i);
                    m.dof_jntid.push(jid);
                    m.dof_armature.push(jnt_armature.get(jid).copied().unwrap_or(0.0));
                }
            }
            lastdof[i] = last;
            m.body_dofnum.push(m.dof_bodyid.len() - m.body_dofadr[i]);
        }
        // `setFixed`: subtree masses, children into parents from the last body back
        m.body_subtreemass = m.body_mass.clone();
        for i in (1..nbody).rev() {
            let s = m.body_subtreemass[i];
            m.body_subtreemass[parent[i]] += s;
        }
        // `body_simple`, as `mjCModel::CopyTree` decides it, then `dof_simplenum`
        let null_vec = |v: &[f64; 3]| v.iter().all(|x| x.abs() < 1e-6);
        let mut simple = vec![0i32; nbody];
        for i in 0..nbody {
            let b = &kin.bodies[i];
            let p = parent[i];
            simple[i] = (b.sameframe == SameFrame::Body && (m.body_rootid[i] == i || (parent[p] == 0 && m.body_dofnum[p] == 0))) as i32;
            if i > 0 && p > 0 {
                simple[p] = 0;
            }
            let mut rotfound = false;
            for j in &b.joints {
                let aligned = j.axis.iter().filter(|a| a.abs() > 1e-14).count() == 1;
                if rotfound || !null_vec(&j.pos) || (matches!(j.kind, KinJointKind::Hinge | KinJointKind::Slide) && !aligned) {
                    simple[i] = 0;
                }
                if matches!(j.kind, KinJointKind::Ball | KinJointKind::Hinge) {
                    rotfound = true;
                }
            }
            if simple[i] != 0 && m.body_dofnum[i] != 0 {
                simple[i] = if b.joints.iter().all(|j| j.kind == KinJointKind::Slide) { 2 } else { 1 };
            }
        }
        m.body_simple = simple;
        let nv = m.dof_bodyid.len();
        m.dof_simplenum = vec![0; nv];
        let mut count = 0;
        for i in (0..nv).rev() {
            count = if m.body_simple[m.dof_bodyid[i]] != 0 { count + 1 } else { 0 };
            m.dof_simplenum[i] = count;
        }
        // `M`: a simple dof's row is its diagonal; any other's is its ancestors, ascending, then itself
        for i in 0..nv {
            m.m_rowadr.push(m.m_colind.len());
            let mut row = vec![i];
            if m.dof_simplenum[i] == 0 {
                let mut j = m.dof_parentid[i];
                while j >= 0 {
                    row.push(j as usize);
                    j = m.dof_parentid[j as usize];
                }
            }
            row.reverse();
            m.m_rownnz.push(row.len());
            m.m_colind.extend(row);
        }
        // `mj_setM0`: at qpos0, the composite inertia seen by each dof, plus its armature
        let kin0 = kin.state(qpos0);
        let (cinert, cdof, _) = m.com_pos(&kin0);
        let crb = m.composite(&cinert);
        m.dof_m0 = (0..nv).map(|i| m.dof_armature[i] + dot_n6(&cdof[i], &mul_inert_vec(&crb[m.dof_bodyid[i]], &cdof[i]))).collect();
        m
    }

    /// `mj_comPos`: `cinert`, `cdof` and `subtree_com`.
    fn com_pos(&self, k: &KinState) -> (Vec<[f64; 10]>, Vec<[f64; 6]>, Vec<[f64; 3]>) {
        let nbody = self.kin.bodies.len();
        let mut sub: Vec<[f64; 3]> = (0..nbody).map(|i| k.xipos[i].map(|x| x * self.body_mass[i])).collect();
        for i in (1..nbody).rev() {
            let p = self.kin.bodies[i].parent;
            let c = sub[i];
            for a in 0..3 {
                sub[p][a] += c[a];
            }
        }
        for i in 0..nbody {
            if self.body_subtreemass[i] < MJ_MINVAL {
                sub[i] = k.xipos[i];
            } else {
                let inv = 1.0 / self.body_subtreemass[i];
                sub[i] = sub[i].map(|x| x * inv);
            }
        }
        let mut cinert = vec![[0.0; 10]; nbody];
        for i in 1..nbody {
            let r = &sub[self.body_rootid[i]];
            let off = [k.xipos[i][0] - r[0], k.xipos[i][1] - r[1], k.xipos[i][2] - r[2]];
            cinert[i] = inert_com(&self.body_inertia[i], &k.ximat[i], &off, self.body_mass[i]);
        }
        let mut cdof = vec![[0.0; 6]; self.dof_bodyid.len()];
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let i = self.jnt_bodyid[j];
            let da = self.jnt_dofadr[j];
            let r = &sub[self.body_rootid[i]];
            let a = &k.xanchor[j];
            let off = [r[0] - a[0], r[1] - a[1], r[2] - a[2]];
            let rot = |cdof: &mut Vec<[f64; 6]>, first: usize| {
                for c in 0..3 {
                    let axis = [k.xmat[i][c], k.xmat[i][c + 3], k.xmat[i][c + 6]];
                    cdof[first + c] = dof_com(&axis, Some(&off));
                }
            };
            match kind {
                KinJointKind::Free => {
                    for c in 0..3 {
                        cdof[da + c] = [0.0; 6];
                        cdof[da + c][3 + c] = 1.0;
                    }
                    rot(&mut cdof, da + 3);
                }
                KinJointKind::Ball => rot(&mut cdof, da),
                KinJointKind::Slide => cdof[da] = dof_com(&k.xaxis[j], None),
                KinJointKind::Hinge => cdof[da] = dof_com(&k.xaxis[j], Some(&off)),
            }
        }
        (cinert, cdof, sub)
    }

    /// The first half of `mj_crb`: each body's composite inertia, children into parents from the last back.
    fn composite(&self, cinert: &[[f64; 10]]) -> Vec<[f64; 10]> {
        let mut crb = cinert.to_vec();
        for i in (1..crb.len()).rev() {
            let p = self.kin.bodies[i].parent;
            if p > 0 {
                let c = crb[i];
                for a in 0..10 {
                    crb[p][a] += c[a];
                }
            }
        }
        crb
    }

    /// **`mj_kinematics` → `mj_comPos` → `mj_crb` → `mj_factorM` → `mj_comVel` → `mj_rne`** (without
    /// acceleration) at `qpos`, `qvel`.
    pub(crate) fn forward(&self, qpos: &[f64], qvel: &[f64]) -> SmoothData {
        let nbody = self.kin.bodies.len();
        let nv = self.dof_bodyid.len();
        let kin = self.kin.state(qpos);
        let (cinert, cdof, subtree_com) = self.com_pos(&kin);
        // `mj_crb`
        let crb = self.composite(&cinert);
        let mut m = vec![0.0; self.m_colind.len()];
        for i in 0..nv {
            let adr = self.m_rowadr[i];
            if self.dof_simplenum[i] != 0 {
                m[adr] = self.dof_m0[i];
                continue;
            }
            let mut madr = adr + self.m_rownnz[i] - 1;
            m[madr] = self.dof_armature[i];
            let buf = mul_inert_vec(&crb[self.dof_bodyid[i]], &cdof[i]);
            let mut j = i as i32;
            while j >= 0 {
                m[madr] += dot6(&cdof[j as usize], &buf);
                madr = madr.wrapping_sub(1);
                j = self.dof_parentid[j as usize];
            }
        }
        // `mj_factorI`: L'DL in place, rows from the last back
        let mut qld = m.clone();
        let mut qld_diag_inv = vec![0.0; nv];
        for k in (0..nv).rev() {
            let start = self.m_rowadr[k];
            let diag = self.m_rownnz[k] - 1;
            let end = start + diag;
            if qld[end] < MJ_MINVAL {
                qld[end] = MJ_MINVAL;
            }
            let inv_d = 1.0 / qld[end];
            qld_diag_inv[k] = inv_d;
            for adr in (start..end).rev() {
                let i = self.m_colind[adr];
                let scl = -qld[adr] * inv_d;
                let (ri, n) = (self.m_rowadr[i], self.m_rownnz[i]);
                for t in 0..n {
                    qld[ri + t] = qld[start + t].mul_add(scl, qld[ri + t]);
                }
            }
            for t in 0..diag {
                qld[start + t] *= inv_d;
            }
        }
        // `mj_comVel`
        let mut cvel = vec![[0.0; 6]; nbody];
        let mut cdof_dot = vec![[0.0; 6]; nv];
        for i in 1..nbody {
            let mut v = cvel[self.kin.bodies[i].parent];
            let (bda, n) = (self.body_dofadr[i], self.body_dofnum[i]);
            let mut j = 0;
            while j < n {
                let kind = self.jnt_type[self.dof_jntid[bda + j]];
                if kind == KinJointKind::Free {
                    let t = mul_dof_vec(&cdof[bda..bda + 3], &qvel[bda..bda + 3]);
                    for a in 0..6 {
                        v[a] += t[a];
                    }
                    for c in 0..3 {
                        cdof_dot[bda + c] = [0.0; 6];
                    }
                    j += 3;
                }
                if matches!(kind, KinJointKind::Free | KinJointKind::Ball) {
                    for c in 0..3 {
                        cdof_dot[bda + j + c] = cross_motion(&v, &cdof[bda + j + c]);
                    }
                    let t = mul_dof_vec(&cdof[bda + j..bda + j + 3], &qvel[bda + j..bda + j + 3]);
                    for a in 0..6 {
                        v[a] += t[a];
                    }
                    j += 3;
                } else {
                    cdof_dot[bda + j] = cross_motion(&v, &cdof[bda + j]);
                    let t = mul_dof_vec(&cdof[bda + j..bda + j + 1], &qvel[bda + j..bda + j + 1]);
                    for a in 0..6 {
                        v[a] += t[a];
                    }
                    j += 1;
                }
            }
            cvel[i] = v;
        }
        // `mj_rne` without acceleration
        let mut cacc = vec![[0.0; 6]; nbody];
        cacc[0] = [0.0, 0.0, 0.0, -self.gravity[0], -self.gravity[1], -self.gravity[2]];
        let mut cfrc = vec![[0.0; 6]; nbody];
        for i in 1..nbody {
            let (bda, n) = (self.body_dofadr[i], self.body_dofnum[i]);
            let t = mul_dof_vec(&cdof_dot[bda..bda + n], &qvel[bda..bda + n]);
            let p = cacc[self.kin.bodies[i].parent];
            cacc[i] = std::array::from_fn(|a| p[a] + t[a]);
            let mut f = mul_inert_vec(&cinert[i], &cacc[i]);
            let t = mul_inert_vec(&cinert[i], &cvel[i]);
            let t1 = cross_force(&cvel[i], &t);
            for a in 0..6 {
                f[a] += t1[a];
            }
            cfrc[i] = f;
        }
        cfrc[0] = [0.0; 6];
        for i in (1..nbody).rev() {
            let p = self.kin.bodies[i].parent;
            if p > 0 {
                let c = cfrc[i];
                for a in 0..6 {
                    cfrc[p][a] += c[a];
                }
            }
        }
        let qfrc_bias = (0..nv).map(|i| dot6(&cdof[i], &cfrc[self.dof_bodyid[i]])).collect();
        SmoothData { kin, subtree_com, cinert, cdof, crb, m, qld, qld_diag_inv, cvel, cdof_dot, qfrc_bias }
    }
}

impl SmoothData {
    /// Every array under its `mjData` name, flattened as MuJoCo stores it.
    pub(crate) fn arrays(&self) -> Vec<(&'static str, Vec<f64>)> {
        fn flat<const N: usize>(v: &[[f64; N]]) -> Vec<f64> {
            v.iter().flatten().copied().collect()
        }
        vec![
            ("xpos", flat(&self.kin.xpos)),
            ("xquat", flat(&self.kin.xquat)),
            ("xmat", flat(&self.kin.xmat)),
            ("xipos", flat(&self.kin.xipos)),
            ("ximat", flat(&self.kin.ximat)),
            ("xanchor", flat(&self.kin.xanchor)),
            ("xaxis", flat(&self.kin.xaxis)),
            ("subtree_com", flat(&self.subtree_com)),
            ("cinert", flat(&self.cinert)),
            ("cdof", flat(&self.cdof)),
            ("crb", flat(&self.crb)),
            ("M", self.m.clone()),
            ("qLD", self.qld.clone()),
            ("qLDiagInv", self.qld_diag_inv.clone()),
            ("cvel", flat(&self.cvel)),
            ("cdof_dot", flat(&self.cdof_dot)),
            ("qfrc_bias", self.qfrc_bias.clone()),
        ]
    }
}

impl SmoothModel {
    /// Every compiled array under its `mjModel` name (integers as floats).
    pub(crate) fn arrays(&self) -> Vec<(&'static str, Vec<f64>)> {
        let f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let g = |v: &[i32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        vec![
            ("body_parentid", f(&self.kin.bodies.iter().map(|b| b.parent).collect::<Vec<_>>())),
            ("body_rootid", f(&self.body_rootid)),
            // -1 on a body with none, as MuJoCo stores it
            ("body_jntadr", self.body_jntadr.iter().zip(&self.kin.bodies).map(|(&a, b)| if b.joints.is_empty() { -1.0 } else { a as f64 }).collect()),
            ("body_dofadr", self.body_dofadr.iter().zip(&self.body_dofnum).map(|(&a, &n)| if n == 0 { -1.0 } else { a as f64 }).collect()),
            ("body_dofnum", f(&self.body_dofnum)),
            ("body_simple", g(&self.body_simple)),
            ("body_mass", self.body_mass.clone()),
            ("body_subtreemass", self.body_subtreemass.clone()),
            ("body_inertia", self.body_inertia.iter().flatten().copied().collect()),
            ("jnt_dofadr", f(&self.jnt_dofadr)),
            ("jnt_bodyid", f(&self.jnt_bodyid)),
            ("dof_bodyid", f(&self.dof_bodyid)),
            ("dof_jntid", f(&self.dof_jntid)),
            ("dof_parentid", g(&self.dof_parentid)),
            ("dof_simplenum", g(&self.dof_simplenum)),
            ("dof_M0", self.dof_m0.clone()),
            ("dof_armature", self.dof_armature.clone()),
            ("M_rownnz", f(&self.m_rownnz)),
            ("M_rowadr", f(&self.m_rowadr)),
            ("M_colind", f(&self.m_colind)),
            ("gravity", self.gravity.to_vec()),
        ]
    }
}
