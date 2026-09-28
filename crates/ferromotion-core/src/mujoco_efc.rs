//! **MuJoCo's constraint rows in MuJoCo's own arithmetic** — `mj_makeConstraint` (dof friction, joint limits,
//! contacts), `mj_diagApprox`, `mj_makeImpedance` and `mj_referenceConstraint` from
//! `engine_core_constraint.c` (3.13.0), on the native smooth state of [`crate::mujoco_smooth`], with the
//! compiler's `body_invweight0`/`dof_invweight0` (`set0` in `engine_setconst.c`).
//!
//! Rows are kept as MuJoCo keeps them: dense (every dof) below 60 dofs, sparse (the merged dof chain of the
//! two bodies, common ancestors skipped) from 60 on — and the two are not interchangeable to the bit,
//! because `efc_vel = J·qvel` runs `mju_dot`'s four running sums over whichever entries the row holds.
//!
//! ⛔ Contracted as clang contracts MuJoCo's C on arm64 (see [`crate::mujoco_ccd`]).

use crate::mujoco_kinematics::KinJointKind;
use crate::mujoco_smooth::{SmoothData, SmoothModel};

const MJ_MINVAL: f64 = 1e-15;
const MJ_MINIMP: f64 = 0.0001;
const MJ_MAXIMP: f64 = 0.9999;

/// `mjtConstraint`
pub(crate) const EQUALITY: i32 = 0;
pub(crate) const FRICTION_DOF: i32 = 1;
pub(crate) const FRICTION_TENDON: i32 = 2;
pub(crate) const LIMIT_JOINT: i32 = 3;
pub(crate) const LIMIT_TENDON: i32 = 4;
pub(crate) const CONTACT_FRICTIONLESS: i32 = 5;
pub(crate) const CONTACT_PYRAMIDAL: i32 = 6;
pub(crate) const CONTACT_ELLIPTIC: i32 = 7;

/// **`mju_dot` as the arm64 wheel computes it**: four running sums over the groups of four, then the
/// remainder.
///
/// ⛔⛔ Not uniformly contracted. clang vectorises the four sums eight groups (32 entries) at a time and
/// in that block multiplies and adds SEPARATELY (`fmul.2d`, then scalar `fadd`s in order); the groups left
/// over go through a fused `fmla` loop, and the tail is fused as the source reads. So a dot of 32 or more
/// entries rounds its first `32·⌊n/32⌋` products on their own — read off `_mju_dot` in
/// `libmujoco.3.13.0.dylib`, and the reason `body_invweight0` was one ulp off on every free-floating
/// model while every smaller dot agreed.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let (mut r0, mut r1, mut r2, mut r3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let groups = n / 4;
    let unfused = if n >= 32 { groups / 8 * 8 } else { 0 };
    for g in 0..groups {
        let i = 4 * g;
        if g < unfused {
            r0 += a[i] * b[i];
            r1 += a[i + 1] * b[i + 1];
            r2 += a[i + 2] * b[i + 2];
            r3 += a[i + 3] * b[i + 3];
        } else {
            r0 = a[i].mul_add(b[i], r0);
            r1 = a[i + 1].mul_add(b[i + 1], r1);
            r2 = a[i + 2].mul_add(b[i + 2], r2);
            r3 = a[i + 3].mul_add(b[i + 3], r3);
        }
    }
    let i = 4 * groups;
    let mut r = (r0 + r2) + (r1 + r3);
    match n - i {
        3 => r += a[i + 2].mul_add(b[i + 2], a[i].mul_add(b[i], a[i + 1] * b[i + 1])),
        2 => r += a[i].mul_add(b[i], a[i + 1] * b[i + 1]),
        1 => r = a[i].mul_add(b[i], r),
        _ => {}
    }
    r
}

/// `mju_dot` of a vector WITH ITSELF where clang inlined it (`mju_norm`, the Cholesky diagonal): there the
/// vectoriser keeps every group fused (`fmla.2d` only, no unfused 32-entry block) — read off the inlined
/// copies in `_mju_cholFactor` and `_mj_solPrimal`.
pub(crate) fn dot_self(a: &[f64]) -> f64 {
    let n = a.len();
    let (mut r0, mut r1, mut r2, mut r3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let groups = n / 4;
    for g in 0..groups {
        let i = 4 * g;
        r0 = a[i].mul_add(a[i], r0);
        r1 = a[i + 1].mul_add(a[i + 1], r1);
        r2 = a[i + 2].mul_add(a[i + 2], r2);
        r3 = a[i + 3].mul_add(a[i + 3], r3);
    }
    let i = 4 * groups;
    let mut r = (r0 + r2) + (r1 + r3);
    match n - i {
        3 => r += a[i + 2].mul_add(a[i + 2], a[i].mul_add(a[i], a[i + 1] * a[i + 1])),
        2 => r += a[i].mul_add(a[i], a[i + 1] * a[i + 1]),
        1 => r = a[i].mul_add(a[i], r),
        _ => {}
    }
    r
}

/// One contact as `mjContact` holds it after `mj_setContact`.
#[derive(Clone, Debug)]
pub(crate) struct NativeContact {
    pub(crate) geom: [usize; 2],
    pub(crate) dist: f64,
    pub(crate) pos: [f64; 3],
    /// rows: normal, first tangent, second tangent
    pub(crate) frame: [f64; 9],
    pub(crate) dim: usize,
    pub(crate) includemargin: f64,
    pub(crate) friction: [f64; 5],
    pub(crate) solref: [f64; 2],
    pub(crate) solreffriction: [f64; 2],
    pub(crate) solimp: [f64; 5],
    pub(crate) exclude: bool,
}

/// One active equality constraint as the rows read it.
#[derive(Clone, Debug)]
pub(crate) enum NativeEq {
    /// `mjEQ_JOINT`: `q1 − ref1 = poly(q2 − ref2)`, each joint as `(dofadr, qposadr, qpos0)`.
    Joint { j1: (usize, usize, f64), j2: Option<(usize, usize, f64)>, poly: [f64; 5] },
    /// `mjEQ_CONNECT` on bodies: `eq_data[0..3]` in body1's frame, `eq_data[3..6]` in body2's (from `set0`)
    ConnectBodies { b1: usize, b2: usize, data1: [f64; 3], data2: [f64; 3] },
    /// `mjEQ_CONNECT` on sites
    ConnectSites { s1: crate::mujoco_kinematics::KinGeom, s2: crate::mujoco_kinematics::KinGeom },
}

impl NativeEq {
    /// the two bodies a connect acts on
    pub(crate) fn connect_bodies(&self) -> Option<(usize, usize)> {
        match self {
            NativeEq::ConnectBodies { b1, b2, .. } => Some((*b1, *b2)),
            NativeEq::ConnectSites { s1, s2 } => Some((s1.body, s2.body)),
            NativeEq::Joint { .. } => None,
        }
    }
}

/// `mji_mulMatVec3` / `mju_mulMatVec3`, contracted.
pub(crate) fn mat_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [m[2].mul_add(v[2], m[0].mul_add(v[0], m[1] * v[1])), m[5].mul_add(v[2], m[3].mul_add(v[0], m[4] * v[1])), m[8].mul_add(v[2], m[6].mul_add(v[0], m[7] * v[1]))]
}

/// `mju_mulMatTVec3`, contracted.
pub(crate) fn mat_t_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [m[6].mul_add(v[2], m[0].mul_add(v[0], m[3] * v[1])), m[7].mul_add(v[2], m[1].mul_add(v[0], m[4] * v[1])), m[8].mul_add(v[2], m[2].mul_add(v[0], m[5] * v[1]))]
}

/// `set0`'s `eq_data[3..6]` for a body connect: the anchor as a world point at `qpos0`, in body2's frame.
pub(crate) fn connect_data2(k: &crate::mujoco_kinematics::KinState, b1: usize, b2: usize, anchor: &[f64; 3]) -> [f64; 3] {
    let p = mat_vec3(&k.xmat[b1], anchor);
    let pos = [p[0] + k.xpos[b1][0] - k.xpos[b2][0], p[1] + k.xpos[b1][1] - k.xpos[b2][1], p[2] + k.xpos[b1][2] - k.xpos[b2][2]];
    mat_t_vec3(&k.xmat[b2], &pos)
}

/// `mj_equalityAnchors` for a connect: the two world points and the two bodies.
fn connect_anchors(eq: &NativeEq, k: &crate::mujoco_kinematics::KinState) -> ([f64; 3], [f64; 3], usize, usize) {
    match eq {
        NativeEq::ConnectBodies { b1, b2, data1, data2 } => {
            let p1 = mat_vec3(&k.xmat[*b1], data1);
            let p2 = mat_vec3(&k.xmat[*b2], data2);
            ([p1[0] + k.xpos[*b1][0], p1[1] + k.xpos[*b1][1], p1[2] + k.xpos[*b1][2]], [p2[0] + k.xpos[*b2][0], p2[1] + k.xpos[*b2][1], p2[2] + k.xpos[*b2][2]], *b1, *b2)
        }
        NativeEq::ConnectSites { s1, s2 } => {
            let at = |s: &crate::mujoco_kinematics::KinGeom| -> [f64; 3] {
                use crate::mujoco_kinematics::SameFrame;
                match s.sameframe {
                    SameFrame::Body => k.xpos[s.body],
                    SameFrame::Inertia => k.xipos[s.body],
                    _ => {
                        let v = mat_vec3(&k.xmat[s.body], &s.pos);
                        [v[0] + k.xpos[s.body][0], v[1] + k.xpos[s.body][1], v[2] + k.xpos[s.body][2]]
                    }
                }
            };
            (at(s1), at(s2), s1.body, s2.body)
        }
        NativeEq::Joint { .. } => unreachable!("not a connect"),
    }
}

/// What the constraint rows need from the compiled model beyond the smooth dynamics.
#[derive(Clone, Debug, Default)]
pub(crate) struct EfcModel {
    /// the active equalities, in order, with their `eq_solref`/`eq_solimp`
    pub(crate) eq: Vec<(NativeEq, [f64; 2], [f64; 5])>,
    pub(crate) jnt_limited: Vec<bool>,
    pub(crate) jnt_range: Vec<[f64; 2]>,
    pub(crate) jnt_margin: Vec<f64>,
    pub(crate) jnt_solref: Vec<[f64; 2]>,
    pub(crate) jnt_solimp: Vec<[f64; 5]>,
    pub(crate) dof_frictionloss: Vec<f64>,
    pub(crate) dof_solref: Vec<[f64; 2]>,
    pub(crate) dof_solimp: Vec<[f64; 5]>,
    pub(crate) elliptic: bool,
    pub(crate) impratio: f64,
    /// `body_invweight0`, `dof_invweight0`
    pub(crate) body_invweight0: Vec<[f64; 2]>,
    pub(crate) dof_invweight0: Vec<f64>,
    /// per tendon: `tendon_range` when `tendon_limited`, `tendon_margin`, the limit's `solref`/`solimp`,
    /// `tendon_frictionloss` and the friction's `solref`/`solimp`, and `tendon_invweight0`
    pub(crate) ten_range: Vec<Option<[f64; 2]>>,
    pub(crate) ten_margin: Vec<f64>,
    pub(crate) ten_solref_lim: Vec<[f64; 2]>,
    pub(crate) ten_solimp_lim: Vec<[f64; 5]>,
    pub(crate) ten_frictionloss: Vec<f64>,
    pub(crate) ten_solref_fri: Vec<[f64; 2]>,
    pub(crate) ten_solimp_fri: Vec<[f64; 5]>,
    pub(crate) tendon_invweight0: Vec<f64>,
    /// `stat.meaninertia` (`mj_setConst`): the mean of `M`'s diagonal at `qpos0`, summed in dof order
    pub(crate) meaninertia: f64,
}

/// The `efc_*` arrays. Each row of `J` is `(columns, values)`: every dof when dense, the chain when sparse.
#[derive(Clone, Debug, Default)]
pub(crate) struct Efc {
    pub(crate) typ: Vec<i32>,
    pub(crate) id: Vec<usize>,
    pub(crate) j: Vec<(Vec<usize>, Vec<f64>)>,
    pub(crate) pos: Vec<f64>,
    pub(crate) margin: Vec<f64>,
    pub(crate) frictionloss: Vec<f64>,
    pub(crate) diag_a: Vec<f64>,
    pub(crate) r: Vec<f64>,
    pub(crate) d: Vec<f64>,
    pub(crate) kbip: Vec<[f64; 4]>,
    pub(crate) vel: Vec<f64>,
    pub(crate) aref: Vec<f64>,
    /// `contact.mu`, the regularised cone's friction, per contact (0 where frictionless or excluded)
    pub(crate) contact_mu: Vec<f64>,
    /// the contacts instantiated, in order, and each one's first row (`efc_address`)
    pub(crate) contact_address: Vec<Option<usize>>,
}

impl Efc {
    fn push(&mut self, cols: Vec<usize>, vals: Vec<f64>, pos: f64, margin: f64, frictionloss: f64, typ: i32, id: usize) {
        self.j.push((cols, vals));
        self.pos.push(pos);
        self.margin.push(margin);
        self.frictionloss.push(frictionloss);
        self.typ.push(typ);
        self.id.push(id);
    }

    /// `efc_J` as a dense `nefc × nv` array, row-major.
    pub(crate) fn dense_j(&self, nv: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.j.len() * nv];
        for (r, (cols, vals)) in self.j.iter().enumerate() {
            for (&c, &v) in cols.iter().zip(vals) {
                out[r * nv + c] = v;
            }
        }
        out
    }
}

impl SmoothModel {
    /// The weld body a body's dofs are found through (`body_weldid`).
    fn weld(&self, mut b: usize) -> usize {
        while b != 0 && self.body_dofnum[b] == 0 {
            b = self.kin.bodies[b].parent;
        }
        b
    }

    /// `mj_jac`, dense: the translation (and, asked for, rotation) Jacobian of `point` on `body`.
    pub(crate) fn jac(&self, sd: &SmoothData, point: &[f64; 3], body: usize, rot: bool) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        self.jac_at(&sd.cdof, &sd.subtree_com, point, body, rot)
    }

    /// [`SmoothModel::jac`] from `cdof` and `subtree_com` directly, for the stages that run before the
    /// smooth data is assembled (`mj_transmission`, `mj_passive`).
    pub(crate) fn jac_at(&self, cdof: &[[f64; 6]], subtree_com: &[[f64; 3]], point: &[f64; 3], body: usize, rot: bool) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let nv = self.dof_bodyid.len();
        let r = &subtree_com[self.body_rootid[body]];
        let offset = [point[0] - r[0], point[1] - r[1], point[2] - r[2]];
        let mut jp = vec![[0.0; 3]; nv];
        let mut jr = if rot { vec![[0.0; 3]; nv] } else { Vec::new() };
        let b = self.weld(body);
        if self.body_dofnum[b] == 0 {
            return (jp, jr);
        }
        let mut i = (self.body_dofadr[b] + self.body_dofnum[b] - 1) as i32;
        while i >= 0 {
            let c = &cdof[i as usize];
            if rot {
                jr[i as usize] = [c[0], c[1], c[2]];
            }
            let t = cross(&[c[0], c[1], c[2]], &offset);
            jp[i as usize] = [c[3] + t[0], c[4] + t[1], c[5] + t[2]];
            i = self.dof_parentid[i as usize];
        }
        (jp, jr)
    }

    /// `mj_mergeChain` with the common ancestors skipped.
    fn merge_chain(&self, b1: usize, b2: usize) -> Vec<usize> {
        let (b1, b2) = (self.weld(b1), self.weld(b2));
        if self.body_dofnum[b1] == 0 && self.body_dofnum[b2] == 0 {
            return Vec::new();
        }
        let mut da1 = self.body_dofadr[b1] as i32 + self.body_dofnum[b1] as i32 - 1;
        let mut da2 = self.body_dofadr[b2] as i32 + self.body_dofnum[b2] as i32 - 1;
        let mut chain = Vec::new();
        while da1 >= 0 || da2 >= 0 {
            let da = da1.max(da2);
            if da1 == da && da2 == da {
                break;
            }
            chain.push(da as usize);
            if da1 == da {
                da1 = self.dof_parentid[da1 as usize];
            }
            if da2 == da {
                da2 = self.dof_parentid[da2 as usize];
            }
        }
        chain.reverse();
        chain
    }

    /// `mj_jacSparse` over `chain`, common dofs skipped: rows `(translation, rotation)` per chain entry.
    fn jac_sparse(&self, sd: &SmoothData, point: &[f64; 3], body: usize, chain: &[usize]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let nn = chain.len();
        let (mut jp, mut jr) = (vec![[0.0; 3]; nn], vec![[0.0; 3]; nn]);
        let r = &sd.subtree_com[self.body_rootid[body]];
        let offset = [point[0] - r[0], point[1] - r[1], point[2] - r[2]];
        let b = self.weld(body);
        if self.body_dofnum[b] == 0 {
            return (jp, jr);
        }
        let mut da = (self.body_dofadr[b] + self.body_dofnum[b] - 1) as i32;
        let mut ci = nn as i32 - 1;
        while da >= 0 {
            while ci >= 0 && chain[ci as usize] as i32 > da {
                ci -= 1;
            }
            if ci < 0 || chain[ci as usize] as i32 != da {
                da = self.dof_parentid[da as usize];
                continue;
            }
            let c = &sd.cdof[da as usize];
            jr[ci as usize] = [c[0], c[1], c[2]];
            let t = cross(&[c[0], c[1], c[2]], &offset);
            jp[ci as usize] = [c[3] + t[0], c[4] + t[1], c[5] + t[2]];
            da = self.dof_parentid[da as usize];
        }
        (jp, jr)
    }

    /// `mj_jacDifPair` at one point: `(chain, translation difference, rotation difference)`, each difference
    /// as three rows over the chain.
    fn jac_dif_pair(&self, sd: &SmoothData, b1: usize, b2: usize, pos: &[f64; 3], rot: bool, sparse: bool) -> (Vec<usize>, [Vec<f64>; 3], [Vec<f64>; 3]) {
        let nv = self.dof_bodyid.len();
        if nv == 0 {
            return (Vec::new(), Default::default(), Default::default());
        }
        let simple = self.body_simple[b1] != 0 && self.body_simple[b2] != 0;
        if !sparse {
            let (p1, r1) = self.jac(sd, pos, b1, rot);
            let (p2, r2) = self.jac(sd, pos, b2, rot);
            let dp: [Vec<f64>; 3] = std::array::from_fn(|k| (0..nv).map(|c| p2[c][k] - p1[c][k]).collect());
            let dr: [Vec<f64>; 3] = if rot { std::array::from_fn(|k| (0..nv).map(|c| r2[c][k] - r1[c][k]).collect()) } else { Default::default() };
            return ((0..nv).collect(), dp, dr);
        }
        if simple {
            // `mj_mergeChainSimple`: the lower body's dofs, then the other's
            let (lo, hi) = if b1 > b2 { (b2, b1) } else { (b1, b2) };
            let (n1, n2) = (self.body_dofnum[lo], self.body_dofnum[hi]);
            if n1 == 0 && n2 == 0 {
                return (Vec::new(), Default::default(), Default::default());
            }
            let chain: Vec<usize> = (0..n1).map(|i| self.body_dofadr[lo] + i).chain((0..n2).map(|i| self.body_dofadr[hi] + i)).collect();
            let nn = chain.len();
            let mut dp: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nn]);
            let mut dr: [Vec<f64>; 3] = if rot { std::array::from_fn(|_| vec![0.0; nn]) } else { Default::default() };
            // `mj_jacSparseSimple`: the first body negated, the second as is
            for (body, second, start) in [(b1, false, if b1 < b2 { 0 } else { self.body_dofnum[b2] }), (b2, true, if b2 < b1 { 0 } else { self.body_dofnum[b1] })] {
                if self.body_dofnum[body] == 0 {
                    continue;
                }
                let r = &sd.subtree_com[self.body_rootid[body]];
                let offset = [pos[0] - r[0], pos[1] - r[1], pos[2] - r[2]];
                for (ci, da) in (self.body_dofadr[body]..self.body_dofadr[body] + self.body_dofnum[body]).enumerate() {
                    let c = &sd.cdof[da];
                    let t = cross(&[c[0], c[1], c[2]], &offset);
                    let v = [c[3] + t[0], c[4] + t[1], c[5] + t[2]];
                    for k in 0..3 {
                        dp[k][start + ci] = if second { v[k] } else { -v[k] };
                        if rot {
                            dr[k][start + ci] = if second { c[k] } else { -c[k] };
                        }
                    }
                }
            }
            return (chain, dp, dr);
        }
        let chain = self.merge_chain(b1, b2);
        if chain.is_empty() {
            return (chain, Default::default(), Default::default());
        }
        let (p1, r1) = self.jac_sparse(sd, pos, b1, &chain);
        let (p2, r2) = self.jac_sparse(sd, pos, b2, &chain);
        let nn = chain.len();
        let dp: [Vec<f64>; 3] = std::array::from_fn(|k| (0..nn).map(|c| p2[c][k] - p1[c][k]).collect());
        let dr: [Vec<f64>; 3] = if rot { std::array::from_fn(|k| (0..nn).map(|c| r2[c][k] - r1[c][k]).collect()) } else { Default::default() };
        (chain, dp, dr)
    }

    /// `mj_jacDifPair` with the common dofs KEPT (`flg_skipcommon = 0`, as the equalities call it), at two
    /// points: `J(b2 at pos2) − J(b1 at pos1)`.
    fn jac_dif_pair_common(&self, sd: &SmoothData, b1: usize, b2: usize, pos1: &[f64; 3], pos2: &[f64; 3], sparse: bool) -> (Vec<usize>, [Vec<f64>; 3], [Vec<f64>; 3]) {
        let nv = self.dof_bodyid.len();
        assert!(!sparse, "a sparse connect is refused before it gets here");
        let (p1, _) = self.jac(sd, pos1, b1, false);
        let (p2, _) = self.jac(sd, pos2, b2, false);
        let dp: [Vec<f64>; 3] = std::array::from_fn(|k| (0..nv).map(|c| p2[c][k] - p1[c][k]).collect());
        ((0..nv).collect(), dp, Default::default())
    }

    /// `mj_jacDot`, translation only, dense: the time derivative of `point`'s Jacobian on `body`.
    fn jac_dot(&self, sd: &SmoothData, point: &[f64; 3], body: usize) -> [Vec<f64>; 3] {
        let nv = self.dof_bodyid.len();
        let mut jp: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; nv]);
        let com = &sd.subtree_com[self.body_rootid[body]];
        let offset = [point[0] - com[0], point[1] - com[1], point[2] - com[2]];
        // `mju_transformSpatial` of the body's cvel to the point (motion)
        let cv = &sd.cvel[body];
        let c = cross(&offset, &[cv[0], cv[1], cv[2]]);
        let plin = [cv[3] - c[0], cv[4] - c[1], cv[5] - c[2]];
        let b = self.weld(body);
        if self.body_dofnum[b] == 0 {
            return jp;
        }
        let mut i = (self.body_dofadr[b] + self.body_dofnum[b] - 1) as i32;
        while i >= 0 {
            let iu = i as usize;
            let mut cdd = sd.cdof_dot[iu];
            let j = self.dof_jntid[iu];
            let is_quat = self.jnt_type[j] == KinJointKind::Ball || (self.jnt_type[j] == KinJointKind::Free && iu >= self.jnt_dofadr[j] + 3);
            if is_quat {
                cdd = cross_motion(&sd.cvel[self.dof_bodyid[iu]], &sd.cdof[iu]);
            }
            let cd = &sd.cdof[iu];
            let t1 = cross(&[cdd[0], cdd[1], cdd[2]], &offset);
            let t2 = cross(&[cd[0], cd[1], cd[2]], &plin);
            for k in 0..3 {
                jp[k][iu] += cdd[3 + k] + t1[k] + t2[k];
            }
            i = self.dof_parentid[iu];
        }
        jp
    }

    /// **`set0`**'s `body_invweight0` and `dof_invweight0`, at `qpos0`.
    pub(crate) fn invweight0(&self, qpos0: &[f64]) -> (Vec<[f64; 2]>, Vec<f64>) {
        let nv = self.dof_bodyid.len();
        let nbody = self.kin.bodies.len();
        let sd = self.forward(qpos0, &vec![0.0; nv], &[], &[]);
        let mut body = vec![[0.0; 2]; nbody];
        let solve_a = |rows: &[Vec<f64>]| -> Vec<Vec<f64>> {
            let tmp: Vec<Vec<f64>> = rows.iter().map(|r| self.solve_m(&sd, r)).collect();
            rows.iter().map(|ri| tmp.iter().map(|tj| dot(ri, tj)).collect()).collect()
        };
        for i in 1..nbody {
            if self.body_dofnum[self.weld(i)] == 0 {
                continue;
            }
            if self.body_simple[i] == 2 {
                body[i] = [1.0 / MJ_MINVAL.max(self.body_mass[i]), 0.0];
                continue;
            }
            if nv == 0 {
                continue;
            }
            let (jp, jr) = self.jac(&sd, &sd.kin.xipos[i], i, true);
            let rows: Vec<Vec<f64>> = (0..3).map(|k| jp.iter().map(|v| v[k]).collect()).chain((0..3).map(|k| jr.iter().map(|v| v[k]).collect())).collect();
            let a = solve_a(&rows);
            body[i] = [(a[0][0] + a[1][1] + a[2][2]) / 3.0, (a[3][3] + a[4][4] + a[5][5]) / 3.0];
        }
        let mut dof = vec![0.0; nv];
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let id = self.jnt_dofadr[j];
            let b = self.jnt_bodyid[j];
            if self.body_simple[b] == 2 {
                dof[id] = 1.0 / MJ_MINVAL.max(self.body_mass[b]);
                continue;
            }
            let dnum = match kind {
                KinJointKind::Free => 6,
                KinJointKind::Ball => 3,
                _ => 1,
            };
            let rows: Vec<Vec<f64>> = (0..dnum).map(|k| (0..nv).map(|c| if c == id + k { 1.0 } else { 0.0 }).collect()).collect();
            let a = solve_a(&rows);
            match dnum {
                6 => {
                    let (t, r) = ((a[0][0] + a[1][1] + a[2][2]) / 3.0, (a[3][3] + a[4][4] + a[5][5]) / 3.0);
                    dof[id..id + 3].fill(t);
                    dof[id + 3..id + 6].fill(r);
                }
                3 => dof[id..id + 3].fill((a[0][0] + a[1][1] + a[2][2]) / 3.0),
                _ => dof[id] = a[0][0],
            }
        }
        (body, dof)
    }

    /// **`mj_setConst`**'s `stat.meaninertia`: `M`'s diagonal at `qpos0`, summed in dof order, over `nv`.
    pub(crate) fn meaninertia(&self, qpos0: &[f64]) -> f64 {
        let nv = self.dof_bodyid.len();
        if nv == 0 {
            return 1.0;
        }
        let sd = self.forward(qpos0, &vec![0.0; nv], &[], &[]);
        let mut s = 0.0;
        for i in 0..nv {
            s += sd.m[self.m_rowadr[i] + self.m_rownnz[i] - 1];
        }
        s / nv as f64
    }

    /// **`set0`**'s `tendon_invweight0`: `J·M⁻¹·Jᵀ` of each fixed tendon at `qpos0` (zero for a spatial one,
    /// which is not ported).
    pub(crate) fn tendon_invweight0(&self, qpos0: &[f64]) -> Vec<f64> {
        let nv = self.dof_bodyid.len();
        if nv == 0 {
            return vec![0.0; self.tendons.len()];
        }
        let sd = self.forward(qpos0, &vec![0.0; nv], &[], &[]);
        self.tendons
            .iter()
            .zip(&sd.ten_j)
            .map(|(t, j)| {
                let Some(t) = t else { return 0.0 };
                let mut tmp = vec![0.0; nv];
                for (&c, &v) in t.colind.iter().zip(j) {
                    tmp[c] = v;
                }
                let x = self.solve_m(&sd, &tmp);
                dot(&tmp, &x)
            })
            .collect()
    }

    /// **`mj_makeConstraint` → `mj_diagApprox` → `mj_makeImpedance` → `mj_referenceConstraint`**, for
    /// equalities (joint, connect), dof and fixed-tendon friction, joint and fixed-tendon limits, and contacts.
    pub(crate) fn constraints(&self, em: &EfcModel, sd: &SmoothData, qpos: &[f64], qvel: &[f64], contacts: &[NativeContact]) -> Efc {
        let nv = self.dof_bodyid.len();
        let sparse = nv >= 60;
        let mut e = Efc::default();
        let dense_row = |col: usize, v: f64| -> (Vec<usize>, Vec<f64>) {
            if sparse {
                (vec![col], vec![v])
            } else {
                let mut row = vec![0.0; nv];
                row[col] = v;
                ((0..nv).collect(), row)
            }
        };
        // equalities
        for (k, (eq, _, _)) in em.eq.iter().enumerate() {
            match eq {
                NativeEq::Joint { j1, j2, poly: d } => {
                    let (dof1, qadr1, ref1) = *j1;
                    let pos0 = qpos[qadr1];
                    let (cols, vals, cpos) = match j2 {
                        Some((dof2, qadr2, ref2)) => {
                            let dif = qpos[*qadr2] - ref2;
                            let p = d[1].mul_add(dif, d[2] * dif * dif);
                            let p = (d[3] * dif * dif).mul_add(dif, p);
                            let p = (d[4] * dif * dif * dif).mul_add(dif, p);
                            let cpos = pos0 - ref1 - d[0] - p;
                            let deriv = (2.0 * d[2]).mul_add(dif, d[1]);
                            let deriv = (3.0 * d[3] * dif).mul_add(dif, deriv);
                            let deriv = (4.0 * d[4] * dif * dif).mul_add(dif, deriv);
                            if sparse {
                                // `mju_combineSparse`: the two sorted chains merged
                                let mut entries = [(dof1, 1.0f64), (*dof2, -deriv)];
                                entries.sort_by_key(|e| e.0);
                                (entries.iter().map(|e| e.0).collect(), entries.iter().map(|e| e.1).collect(), cpos)
                            } else {
                                let mut row = vec![0.0f64; nv];
                                row[dof1] = 1.0;
                                let mut j2row = vec![0.0f64; nv];
                                j2row[*dof2] = 1.0;
                                for c in 0..nv {
                                    row[c] = j2row[c].mul_add(-deriv, row[c]);
                                }
                                ((0..nv).collect(), row, cpos)
                            }
                        }
                        None => {
                            let (c, v) = dense_row(dof1, 1.0);
                            (c, v, pos0 - ref1 - d[0])
                        }
                    };
                    // a dense row that is all zeros adds nothing
                    if !sparse && vals.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    e.push(cols, vals, cpos, 0.0, 0.0, EQUALITY, k);
                }
                NativeEq::ConnectBodies { .. } | NativeEq::ConnectSites { .. } => {
                    let (p0, p1, b0, b1) = connect_anchors(eq, &sd.kin);
                    let cpos = [p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]];
                    // `mj_jacDifPair(body1, body0, p1, p0)`, common dofs kept: J(body0 at p0) − J(body1 at p1)
                    let (chain, dp, _) = self.jac_dif_pair_common(sd, b1, b0, &p1, &p0, sparse);
                    if !sparse && dp.iter().all(|r| r.iter().all(|&x| x == 0.0)) {
                        continue;
                    }
                    for r in 0..3 {
                        e.push(chain.clone(), dp[r].clone(), cpos[r], 0.0, 0.0, EQUALITY, k);
                    }
                }
            }
        }
        // dof friction
        for i in 0..nv {
            if em.dof_frictionloss[i] == 0.0 {
                continue;
            }
            let (c, v) = dense_row(i, 1.0);
            e.push(c, v, 0.0, 0.0, em.dof_frictionloss[i], FRICTION_DOF, i);
        }
        // tendon friction: the row is `ten_J`, made dense when the model is (`mju_sparse2dense`)
        let ten_row = |t: usize, scl: Option<f64>| -> Option<(Vec<usize>, Vec<f64>)> {
            let nt = self.tendons.get(t)?.as_ref()?;
            let j = &sd.ten_j[t];
            if sparse {
                Some((nt.colind.clone(), j.iter().map(|v| scl.map_or(*v, |s| v * s)).collect()))
            } else {
                let mut row = vec![0.0f64; nv];
                for (&c, &v) in nt.colind.iter().zip(j) {
                    row[c] = v;
                }
                // `mju_scl` over the whole dense row: a zero times −1 is −0
                if let Some(s) = scl {
                    row.iter_mut().for_each(|x| *x *= s);
                }
                Some(((0..nv).collect(), row))
            }
        };
        for t in 0..em.ten_frictionloss.len() {
            if em.ten_frictionloss[t] > 0.0
                && let Some((c, v)) = ten_row(t, None)
            {
                if !sparse && v.iter().all(|&x| x == 0.0) {
                    continue;
                }
                e.push(c, v, 0.0, 0.0, em.ten_frictionloss[t], FRICTION_TENDON, t);
            }
        }
        // joint limits
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            if !em.jnt_limited[j] {
                continue;
            }
            let margin = em.jnt_margin[j];
            match kind {
                KinJointKind::Slide | KinJointKind::Hinge => {
                    let value = qpos[self.jnt_qposadr[j]];
                    for side in [-1.0f64, 1.0] {
                        let dist = side * (em.jnt_range[j][if side < 0.0 { 0 } else { 1 }] - value);
                        if dist < margin {
                            let (c, v) = dense_row(self.jnt_dofadr[j], -side);
                            e.push(c, v, dist, margin, 0.0, LIMIT_JOINT, j);
                        }
                    }
                }
                // ball-joint limits are not ported yet; the caller lists a model with one as unsupported
                _ => {}
            }
        }
        // tendon limits
        for (t, range) in em.ten_range.iter().enumerate() {
            let Some(range) = range else { continue };
            let (value, margin) = (sd.ten_length[t], em.ten_margin[t]);
            for side in [-1.0f64, 1.0] {
                let dist = side * (range[if side < 0.0 { 0 } else { 1 }] - value);
                if dist < margin
                    && let Some((c, v)) = ten_row(t, Some(-side))
                {
                    if !sparse && v.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    e.push(c, v, dist, margin, 0.0, LIMIT_TENDON, t);
                }
            }
        }
        // contacts
        e.contact_address = vec![None; contacts.len()];
        e.contact_mu = vec![0.0; contacts.len()];
        for (ci, con) in contacts.iter().enumerate() {
            if con.exclude {
                continue;
            }
            let dim = con.dim;
            let (b1, b2) = (self.kin.geoms[con.geom[0]].body, self.kin.geoms[con.geom[1]].body);
            let (chain, dp, dr) = self.jac_dif_pair(sd, b1, b2, &con.pos, dim > 3, sparse);
            if chain.is_empty() {
                continue;
            }
            let nn = chain.len();
            // rotate to the contact frame: `mju_mulMatMat`, a zero frame entry skipped
            let rotate = |rows: usize, first_frame_row: usize, dif: &[Vec<f64>; 3]| -> Vec<Vec<f64>> {
                (0..rows)
                    .map(|i| {
                        let mut out = vec![0.0; nn];
                        for k in 0..3 {
                            let f = con.frame[(first_frame_row + i) * 3 + k];
                            if f != 0.0 {
                                for (o, d) in out.iter_mut().zip(&dif[k]) {
                                    *o = d.mul_add(f, *o);
                                }
                            }
                        }
                        out
                    })
                    .collect()
            };
            let mut jac = rotate(if dim > 1 { 3 } else { 1 }, 0, &dp);
            if dim > 3 {
                jac.extend(rotate(dim - 3, 0, &dr));
            }
            e.contact_address[ci] = Some(e.j.len());
            if dim == 1 {
                e.push(chain.clone(), jac[0].clone(), con.dist, con.includemargin, 0.0, CONTACT_FRICTIONLESS, ci);
            } else if !em.elliptic {
                for k in 1..dim {
                    let fr = con.friction[k - 1];
                    let plus: Vec<f64> = jac[0].iter().zip(&jac[k]).map(|(a, b)| b.mul_add(fr, *a)).collect();
                    let minus: Vec<f64> = jac[0].iter().zip(&jac[k]).map(|(a, b)| b.mul_add(-fr, *a)).collect();
                    e.push(chain.clone(), plus, con.dist, con.includemargin, 0.0, CONTACT_PYRAMIDAL, ci);
                    e.push(chain.clone(), minus, con.dist, con.includemargin, 0.0, CONTACT_PYRAMIDAL, ci);
                }
            } else {
                for (k, row) in jac.into_iter().enumerate() {
                    let (p, mg) = if k == 0 { (con.dist, con.includemargin) } else { (0.0, 0.0) };
                    e.push(chain.clone(), row, p, mg, 0.0, CONTACT_ELLIPTIC, ci);
                }
            }
        }
        let nefc = e.j.len();
        // `mj_diagApprox`
        e.diag_a = vec![0.0; nefc];
        let mut i = 0;
        while i < nefc {
            let id = e.id[i];
            match e.typ[i] {
                EQUALITY => match &em.eq[id].0 {
                    NativeEq::Joint { j1, j2, .. } => {
                        let mut a = em.dof_invweight0[j1.0];
                        if let Some(j2) = j2 {
                            a += em.dof_invweight0[j2.0];
                        }
                        e.diag_a[i] = a;
                    }
                    eq => {
                        let (b1, b2) = eq.connect_bodies().expect("a connect");
                        e.diag_a[i] = em.body_invweight0[b1][0] + em.body_invweight0[b2][0];
                    }
                },
                FRICTION_DOF => e.diag_a[i] = em.dof_invweight0[id],
                LIMIT_JOINT => e.diag_a[i] = em.dof_invweight0[self.jnt_dofadr[id]],
                FRICTION_TENDON | LIMIT_TENDON => e.diag_a[i] = em.tendon_invweight0[id],
                t => {
                    let con = &contacts[id];
                    let (mut tran, mut rot) = (0.0f64, 0.0f64);
                    for side in 0..2 {
                        let b = self.kin.geoms[con.geom[side]].body;
                        tran = em.body_invweight0[b][0].mul_add(1.0, tran);
                        rot = em.body_invweight0[b][1].mul_add(1.0, rot);
                    }
                    if t == CONTACT_FRICTIONLESS {
                        e.diag_a[i] = tran;
                    } else if t == CONTACT_ELLIPTIC {
                        for j in 0..con.dim {
                            e.diag_a[i + j] = if j < 3 { tran } else { rot };
                        }
                        i += con.dim - 1;
                    } else {
                        for j in 0..con.dim - 1 {
                            let fri = con.friction[j];
                            let v = (fri * fri).mul_add(if j < 2 { tran } else { rot }, tran);
                            e.diag_a[i + 2 * j] = v;
                            e.diag_a[i + 2 * j + 1] = v;
                        }
                        i += 2 * con.dim - 3;
                    }
                }
            }
            i += 1;
        }
        // `mj_makeImpedance`
        let h = self.timestep;
        e.r = vec![0.0; nefc];
        e.kbip = vec![[0.0; 4]; nefc];
        let mut i = 0;
        while i < nefc {
            let id = e.id[i];
            let (mut solref, mut solreffriction, mut solimp) = match e.typ[i] {
                EQUALITY => (em.eq[id].1, [0.0; 2], em.eq[id].2),
                LIMIT_JOINT => (em.jnt_solref[id], [0.0; 2], em.jnt_solimp[id]),
                FRICTION_DOF => (em.dof_solref[id], [0.0; 2], em.dof_solimp[id]),
                LIMIT_TENDON => (em.ten_solref_lim[id], [0.0; 2], em.ten_solimp_lim[id]),
                FRICTION_TENDON => (em.ten_solref_fri[id], [0.0; 2], em.ten_solimp_fri[id]),
                _ => (contacts[id].solref, contacts[id].solreffriction, contacts[id].solimp),
            };
            if (solref[0] > 0.0) ^ (solref[1] > 0.0) {
                solref = [0.02, 1.0];
            }
            if solref[0] > 0.0 {
                solref[0] = max(solref[0], 2.0 * h);
            }
            if (solreffriction[0] > 0.0) ^ (solreffriction[1] > 0.0) {
                solreffriction = [0.0; 2];
            }
            if solreffriction[0] > 0.0 {
                solreffriction[0] = max(solreffriction[0], 2.0 * h);
            }
            solimp[0] = min(MJ_MAXIMP, max(MJ_MINIMP, solimp[0]));
            solimp[1] = min(MJ_MAXIMP, max(MJ_MINIMP, solimp[1]));
            solimp[2] = max(0.0, solimp[2]);
            solimp[3] = min(MJ_MAXIMP, max(MJ_MINIMP, solimp[3]));
            solimp[4] = max(1.0, solimp[4]);
            let (pos, dim) = match e.typ[i] {
                // a connect's three rows share one impedance, at the residual's length (`mju_norm`)
                EQUALITY if em.eq[id].0.connect_bodies().is_some() => {
                    let p = &e.pos[i..i + 3];
                    ((0.0 + p[2].mul_add(p[2], p[0].mul_add(p[0], p[1] * p[1]))).sqrt(), 3)
                }
                CONTACT_ELLIPTIC => (e.pos[i], contacts[id].dim),
                CONTACT_PYRAMIDAL => (e.pos[i], 2 * (contacts[id].dim - 1)),
                _ => (e.pos[i], 1),
            };
            let (imp, imp_p) = impedance(&solimp, pos, e.margin[i]);
            for j in 0..dim {
                e.r[i + j] = max(MJ_MINVAL, (1.0 - imp) * e.diag_a[i + j] / imp);
                let tp = e.typ[i + j];
                let elliptic_friction = tp == CONTACT_ELLIPTIC && j > 0;
                let rf = if elliptic_friction && (solreffriction[0] != 0.0 || solreffriction[1] != 0.0) { solreffriction } else { solref };
                let k = &mut e.kbip[i + j];
                k[0] = if tp == FRICTION_DOF || tp == FRICTION_TENDON || elliptic_friction {
                    0.0
                } else if rf[0] > 0.0 {
                    1.0 / max(MJ_MINVAL, solimp[1] * solimp[1] * rf[0] * rf[0] * rf[1] * rf[1])
                } else {
                    -rf[0] / max(MJ_MINVAL, solimp[1] * solimp[1])
                };
                k[1] = if rf[1] > 0.0 { 2.0 / max(MJ_MINVAL, solimp[1] * rf[0]) } else { -rf[1] / max(MJ_MINVAL, solimp[1]) };
                k[2] = imp;
                k[3] = imp_p;
            }
            i += dim;
        }
        // frictional contacts: the friction rows' R, and the cone's mu
        let mut i = 0;
        while i < nefc {
            let t = e.typ[i];
            if t == CONTACT_PYRAMIDAL || t == CONTACT_ELLIPTIC {
                let id = e.id[i];
                let con = &contacts[id];
                let dim = con.dim;
                e.r[i + 1] = e.r[i] / max(MJ_MINVAL, em.impratio);
                let mu = con.friction[0] * (e.r[i + 1] / e.r[i]).sqrt();
                e.contact_mu[id] = mu;
                if t == CONTACT_ELLIPTIC {
                    for j in 1..dim - 1 {
                        e.r[i + j + 1] = e.r[i + 1] * con.friction[0] * con.friction[0] / (con.friction[j] * con.friction[j]);
                    }
                    i += dim;
                } else {
                    let rpy = 2.0 * mu * mu * e.r[i];
                    for j in 0..2 * (dim - 1) {
                        e.r[i + j] = rpy;
                    }
                    i += 2 * (dim - 1);
                }
                continue;
            }
            i += 1;
        }
        e.d = e.r.iter().map(|r| 1.0 / r).collect();
        for i in 0..nefc {
            e.diag_a[i] = e.r[i] * e.kbip[i][2] / (1.0 - e.kbip[i][2]);
        }
        // `mj_referenceConstraint`
        e.vel = e
            .j
            .iter()
            .map(|(cols, vals)| {
                if sparse {
                    crate::mujoco_smooth::dot_sparse(vals, qvel, cols)
                } else {
                    dot(vals, qvel)
                }
            })
            .collect();
        e.aref = (0..nefc)
            .map(|i| {
                let k = &e.kbip[i];
                let x = 0.0f64.mul_add(e.vel[i], e.pos[i] - e.margin[i]);
                (-k[1]).mul_add(e.vel[i], -(k[0] * k[2] * x))
            })
            .collect();
        // `mj_Jdotv`: a connect's reference loses its J̇·v
        let mut row = 0;
        while row < nefc && e.typ[row] == EQUALITY {
            let eq = &em.eq[e.id[row]].0;
            if eq.connect_bodies().is_some() {
                let (p1, p2, b1, b2) = connect_anchors(eq, &sd.kin);
                let jd1 = self.jac_dot(sd, &p1, b1);
                let jd2 = self.jac_dot(sd, &p2, b2);
                for k in 0..3 {
                    let v1 = dot(&jd1[k], qvel);
                    let v2 = dot(&jd2[k], qvel);
                    e.aref[row + k] -= v1 - v2;
                }
                row += 3;
            } else {
                row += 1;
            }
        }
        e
    }
}

/// `mji_crossMotion`
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

/// `mju_max` / `mju_min`: the first argument on a tie
fn max(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}
fn min(a: f64, b: f64) -> f64 {
    if a <= b { a } else { b }
}

/// `mji_cross`, contracted.
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1].mul_add(b[2], -(a[2] * b[1])), a[2].mul_add(b[0], -(a[0] * b[2])), a[0].mul_add(b[1], -(a[1] * b[0]))]
}

/// `power`: a quick return for the exponents 1 and 2.
fn power(a: f64, b: f64) -> f64 {
    if b == 1.0 {
        a
    } else if b == 2.0 {
        a * a
    } else {
        a.powf(b)
    }
}

/// `getimpedance`: the impedance and its derivative at `pos` against `margin`.
fn impedance(s: &[f64; 5], pos: f64, margin: f64) -> (f64, f64) {
    if s[0] == s[1] || s[2] <= MJ_MINVAL {
        return (0.5 * (s[0] + s[1]), 0.0);
    }
    let mut x = (pos - margin) / s[2];
    let mut sgn = 1.0;
    if x < 0.0 {
        x = -x;
        sgn = -1.0;
    }
    if x >= 1.0 || x <= 0.0 {
        return (if x >= 1.0 { s[1] } else { s[0] }, 0.0);
    }
    let (y, yp);
    if s[4] == 1.0 {
        y = x;
        yp = 1.0;
    } else if x <= s[3] {
        let a = 1.0 / power(s[3], s[4] - 1.0);
        y = a * power(x, s[4]);
        yp = s[4] * a * power(x, s[4] - 1.0);
    } else {
        let b = 1.0 / power(1.0 - s[3], s[4] - 1.0);
        y = (-b).mul_add(power(1.0 - x, s[4]), 1.0);
        yp = s[4] * b * power(1.0 - x, s[4] - 1.0);
    }
    (y.mul_add(s[1] - s[0], s[0]), yp * sgn * (s[1] - s[0]) / s[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mju_dot` as MuJoCo 3.13.0's arm64 wheel returns it (`mujoco.mju_dot`), below and above the 32 entries
    /// where clang's vectorised block stops fusing.
    #[test]
    fn dot_is_mujocos_on_both_sides_of_the_vectorised_block() {
        for (n, want) in [(8, 0xc0010f1dbea8b758u64), (31, 0xc018a647313fe0cb), (32, 0xc01658bf258bf259), (45, 0xc00a3a83a83a83a9), (70, 0x4020af8af8af8af9)] {
            let a: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 3.0 - 1.7).collect();
            let b: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 / 7.0 + 0.3).collect();
            assert_eq!(dot(&a, &b).to_bits(), want, "n = {n}");
        }
    }
}
