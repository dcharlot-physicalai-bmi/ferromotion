//! **MuJoCo's constraint solver in MuJoCo's own arithmetic** — `mj_fwdConstraint` with its warmstart,
//! `mj_island` (the constraints split into independent groups of kinematic trees) and the primal Newton
//! solver `mj_solPrimal` run island by island, from `engine_forward.c`, `engine_island.c`,
//! `engine_solver.c` and the dense Cholesky kernels of `engine_util_solve.c` (3.13.0).
//!
//! Carried: the dense Jacobian (fewer than 60 dofs) and pyramidal cones — the configuration most of
//! Menagerie runs. The island decomposition is not an optimisation to skip: each island is solved in its
//! own local coordinates, so every dot product runs over the island's dofs only, and `mju_dot`'s running
//! sums round differently with the length.
//!
//! ⛔ Contracted as clang contracts MuJoCo's C on arm64 (see [`crate::mujoco_ccd`]); dot products follow
//! [`crate::mujoco_efc::dot`], whose vectorised block is not fused.

use crate::mujoco_efc::{dot, dot_self, Efc, NativeContact, CONTACT_ELLIPTIC, EQUALITY, FRICTION_DOF, FRICTION_TENDON, LIMIT_JOINT, LIMIT_TENDON};
use crate::mujoco_smooth::{SmoothData, SmoothModel};

const MJ_MINVAL: f64 = 1e-15;

/// `mjtConstraintState`
pub(crate) const SATISFIED: i32 = 0;
pub(crate) const QUADRATIC: i32 = 1;
pub(crate) const LINEARNEG: i32 = 2;
pub(crate) const LINEARPOS: i32 = 3;

/// `<option>`'s solver settings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SolverOptions {
    pub(crate) iterations: usize,
    pub(crate) tolerance: f64,
    pub(crate) ls_iterations: usize,
    pub(crate) ls_tolerance: f64,
    /// `noslip_iterations` (0: no post-pass), `noslip_tolerance`, and `stat.meaninertia`, which scales the
    /// post-pass's improvement
    pub(crate) noslip_iterations: usize,
    pub(crate) noslip_tolerance: f64,
    pub(crate) meaninertia: f64,
}

/// What `mj_fwdConstraint` leaves in `mjData`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Solution {
    pub(crate) qacc: Vec<f64>,
    pub(crate) efc_b: Vec<f64>,
    pub(crate) efc_force: Vec<f64>,
    pub(crate) efc_state: Vec<i32>,
    pub(crate) qfrc_constraint: Vec<f64>,
    pub(crate) nisland: usize,
    pub(crate) solver_niter: Vec<usize>,
}

fn max(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}

/// `mju_norm`, as inlined into the solver: a self-dot, fused throughout
fn norm(v: &[f64]) -> f64 {
    dot_self(v).sqrt()
}

/// A contact as the cone code reads it: `dim`, the regularised cone's `mu`, and the five frictions.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConeInfo {
    pub(crate) dim: usize,
    pub(crate) mu: f64,
    pub(crate) friction: [f64; 5],
}

pub(crate) const CONE: i32 = 4;

/// `mj_constraintUpdate_impl`: forces, states and the constraint cost; with `cone_h`, each cone-state
/// contact's local Hessian (`contact.H`, `dim×dim`) by contact id.
#[allow(clippy::too_many_arguments)]
fn constraint_update(ne: usize, nf: usize, d: &[f64], r: &[f64], floss: &[f64], jar: &[f64], typ: &[i32], id: &[usize], cones: &[ConeInfo], state: &mut [i32], force: &mut [f64], mut cone_h: Option<&mut [[f64; 36]]>) -> f64 {
    let nefc = jar.len();
    let mut s = 0.0f64;
    if nefc == 0 {
        return 0.0;
    }
    for i in 0..nefc {
        force[i] = -d[i] * jar[i];
    }
    let mut i = 0;
    while i < nefc {
        if i < ne {
            s = (0.5 * d[i] * jar[i]).mul_add(jar[i], s);
            state[i] = QUADRATIC;
            i += 1;
            continue;
        }
        if i < ne + nf {
            if jar[i] <= -r[i] * floss[i] {
                s += (-0.5 * r[i] * floss[i]).mul_add(floss[i], -(floss[i] * jar[i]));
                force[i] = floss[i];
                state[i] = LINEARNEG;
            } else if jar[i] >= r[i] * floss[i] {
                s += (-0.5 * r[i] * floss[i]).mul_add(floss[i], floss[i] * jar[i]);
                force[i] = -floss[i];
                state[i] = LINEARPOS;
            } else {
                s = (0.5 * d[i] * jar[i]).mul_add(jar[i], s);
                state[i] = QUADRATIC;
            }
            i += 1;
            continue;
        }
        if typ[i] != CONTACT_ELLIPTIC {
            if jar[i] >= 0.0 {
                force[i] = 0.0;
                state[i] = SATISFIED;
            } else {
                s = (0.5 * d[i] * jar[i]).mul_add(jar[i], s);
                state[i] = QUADRATIC;
            }
            i += 1;
            continue;
        }
        // elliptic cone
        let con = &cones[id[i]];
        let (mu, fr, dim) = (con.mu, &con.friction, con.dim);
        let mut u = [0.0f64; 6];
        u[0] = jar[i] * mu;
        for j in 1..dim {
            u[j] = jar[i + j] * fr[j - 1];
        }
        let n = u[0];
        let t = dot_self(&u[1..dim]).sqrt();
        if n >= mu * t || (t <= 0.0 && n >= 0.0) {
            for j in 0..dim {
                force[i + j] = 0.0;
            }
            state[i] = SATISFIED;
        } else if mu.mul_add(n, t) <= 0.0 || (t <= 0.0 && n < 0.0) {
            for j in 0..dim {
                s = (0.5 * d[i + j] * jar[i + j]).mul_add(jar[i + j], s);
            }
            state[i] = QUADRATIC;
        } else {
            let dm = d[i] / (mu * mu * (1.0 + mu * mu));
            let nmt = (-mu).mul_add(t, n);
            s = (0.5 * dm * nmt).mul_add(nmt, s);
            force[i] = -dm * nmt * mu;
            for j in 1..dim {
                force[i + j] = -force[i] / t * u[j] * fr[j - 1];
            }
            state[i] = CONE;
            if let Some(hs) = cone_h.as_deref_mut() {
                let h = &mut hs[id[i]];
                let scl = -mu / t;
                h[0] = 1.0;
                for j in 1..dim {
                    h[j] = scl * u[j];
                }
                let scl = mu * n / (t * t * t);
                for k in 1..dim {
                    for j in k..dim {
                        h[k * dim + j] = scl * u[j] * u[k];
                    }
                }
                let scl = mu.mul_add(mu, -(mu * n / t));
                for j in 1..dim {
                    h[j * (dim + 1)] += scl;
                }
                for k in 0..dim {
                    let scl = dm * if k == 0 { mu } else { fr[k - 1] };
                    for j in k..dim {
                        h[k * dim + j] *= scl * if j == 0 { mu } else { fr[j - 1] };
                    }
                }
                for k in 0..dim {
                    for j in k + 1..dim {
                        h[j * dim + k] = h[k * dim + j];
                    }
                }
            }
        }
        for j in 1..dim {
            state[i + j] = state[i];
        }
        i += dim;
    }
    s
}

/// `mju_mulSymVecSparse`: a symmetric matrix stored as its lower triangle, by a vector.
fn mul_sym_vec(m: &[f64], rownnz: &[usize], rowadr: &[usize], colind: &[usize], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut res = vec![0.0; n];
    for i in 0..n {
        let adr = rowadr[i];
        let diag = rownnz[i] - 1;
        res[i] = m[adr + diag] * v[i];
        for k in (0..diag).rev() {
            let j = colind[adr + k];
            let val = m[adr + k];
            res[i] = val.mul_add(v[j], res[i]);
            res[j] = val.mul_add(v[i], res[j]);
        }
    }
    res
}

/// `mju_mulMatTVec`: `mat'·vec`, rows with a zero coefficient skipped.
fn mul_mat_t_vec(mat: &[f64], vec: &[f64], nr: usize, nc: usize) -> Vec<f64> {
    let mut res = vec![0.0; nc];
    for r in 0..nr {
        let t = vec[r];
        if t != 0.0 {
            for c in 0..nc {
                res[c] = mat[r * nc + c].mul_add(t, res[c]);
            }
        }
    }
    res
}

/// `mju_cholFactor` on the lower triangle of a dense `n×n`; returns the rank.
pub(crate) fn chol_factor(mat: &mut [f64], n: usize, mindiag: f64) -> usize {
    let mut rank = n;
    for j in 0..n {
        let mut tmp = mat[j * (n + 1)];
        if j > 0 {
            tmp -= dot_self(&mat[j * n..j * n + j]);
        }
        let deficient = tmp < mindiag;
        if deficient {
            tmp = mindiag;
            rank -= 1;
        }
        mat[j * (n + 1)] = tmp.sqrt();
        if deficient {
            for i in j + 1..n {
                mat[i * n + j] = 0.0;
            }
        } else {
            let inv = 1.0 / mat[j * (n + 1)];
            for i in j + 1..n {
                let d = dot(&mat[i * n..i * n + j], &mat[j * n..j * n + j]);
                mat[i * n + j] = (mat[i * n + j] - d) * inv;
            }
        }
    }
    rank
}

/// `mju_cholSolve`
pub(crate) fn chol_solve(mat: &[f64], vec: &[f64], n: usize) -> Vec<f64> {
    let mut res = vec.to_vec();
    for i in 0..n {
        if i > 0 {
            let d = dot(&mat[i * n..i * n + i], &res[..i]);
            res[i] -= d;
        }
        res[i] /= mat[i * (n + 1)];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            res[i] = (-mat[j * n + i]).mul_add(res[j], res[i]);
        }
        res[i] /= mat[i * (n + 1)];
    }
    res
}

/// `mju_cholUpdate`: `L·L' ± x·x'`; returns the rank.
fn chol_update(mat: &mut [f64], x: &mut [f64], n: usize, plus: bool) -> usize {
    let mut rank = n;
    for k in 0..n {
        if x[k] != 0.0 {
            let lkk = mat[k * (n + 1)];
            let xx = if plus { x[k] * x[k] } else { -(x[k] * x[k]) };
            let mut tmp = lkk.mul_add(lkk, xx);
            if tmp < MJ_MINVAL {
                tmp = MJ_MINVAL;
                rank -= 1;
            }
            let r = tmp.sqrt();
            let c = r / lkk;
            let cinv = 1.0 / c;
            let s = x[k] / lkk;
            mat[k * (n + 1)] = r;
            for i in k + 1..n {
                let v = mat[i * n + k];
                mat[i * n + k] = if plus { s.mul_add(x[i], v) } else { (-s).mul_add(x[i], v) } * cinv;
            }
            for i in k + 1..n {
                x[i] = c.mul_add(x[i], -(s * mat[i * n + k]));
            }
        }
    }
    rank
}

/// One linesearch point.
#[derive(Clone, Copy, Debug, Default)]
struct Pnt {
    alpha: f64,
    cost: f64,
    deriv: [f64; 2],
}

/// `mjPrimalContext` for one island, dense, pyramidal.
struct Ctx {
    nv: usize,
    ne: usize,
    nf: usize,
    nefc: usize,
    // island-local inertia (lower-triangle sparse, `M`'s layout)
    m_rownnz: Vec<usize>,
    m_rowadr: Vec<usize>,
    m_colind: Vec<usize>,
    m: Vec<f64>,
    qld: Vec<f64>,
    qld_diag_inv: Vec<f64>,
    // dense island Jacobian, nefc × nv
    j: Vec<f64>,
    efc_d: Vec<f64>,
    efc_r: Vec<f64>,
    floss: Vec<f64>,
    aref: Vec<f64>,
    typ: Vec<i32>,
    force: Vec<f64>,
    state: Vec<i32>,
    qfrc_smooth: Vec<f64>,
    qacc_smooth: Vec<f64>,
    qfrc_constraint: Vec<f64>,
    qacc: Vec<f64>,
    jaref: Vec<f64>,
    jv: Vec<f64>,
    ma: Vec<f64>,
    mv: Vec<f64>,
    grad: Vec<f64>,
    mgrad: Vec<f64>,
    search: Vec<f64>,
    /// `3·nefc` wide: an elliptic contact's first row carries nine entries, over its own rows' slots
    quad: Vec<f64>,
    id: Vec<usize>,
    cones: Vec<ConeInfo>,
    cone_h: Vec<[f64; 36]>,
    ncone: usize,
    lcone: Vec<f64>,
    dvec: Vec<f64>,
    l: Vec<f64>,
    /// 60 or more dofs: the island Jacobian row by row (island-local columns), its transpose, `M`'s lower
    /// rows, and the sparse reverse Cholesky factor of `H`, its pattern fixed at the first factorization
    sparse: bool,
    jrows: crate::mujoco_sparse::Rows,
    jt: crate::mujoco_sparse::Rows,
    m_rows: crate::mujoco_sparse::Rows,
    chol: Option<crate::mujoco_sparse::Chol>,
    cost: f64,
    quad_gauss: [f64; 3],
    scale: f64,
    ls_iter: usize,
}

impl Ctx {
    /// `mju_mulMatVec` on the island Jacobian.
    fn j_mul(&self, v: &[f64]) -> Vec<f64> {
        if self.sparse {
            return self.jrows.iter().map(|(c, x)| crate::mujoco_smooth::dot_sparse(x, v, c)).collect();
        }
        (0..self.nefc).map(|r| dot(&self.j[r * self.nv..(r + 1) * self.nv], v)).collect()
    }

    /// `Mgrad = H \ grad` on whichever factor is current.
    fn h_solve(&self, v: &[f64]) -> Vec<f64> {
        match &self.chol {
            Some(c) if self.sparse => c.solve(v),
            _ => chol_solve(if self.ncone > 0 { &self.lcone } else { &self.l }, v, self.nv),
        }
    }

    /// `PrimalUpdateConstraint`
    fn update_constraint(&mut self) {
        let elliptic = self.typ.contains(&CONTACT_ELLIPTIC);
        self.cost = constraint_update(self.ne, self.nf, &self.efc_d, &self.efc_r, &self.floss, &self.jaref, &self.typ, &self.id, &self.cones, &mut self.state, &mut self.force, if elliptic { Some(&mut self.cone_h) } else { None });
        self.ncone = self.state.iter().filter(|&&x| x == CONE).count();
        self.qfrc_constraint = if self.sparse {
            // `mju_mulMatVecSparse(JT, force)`
            self.jt.iter().map(|(c, x)| crate::mujoco_smooth::dot_sparse(x, &self.force, c)).collect()
        } else {
            mul_mat_t_vec(&self.j, &self.force, self.nefc, self.nv)
        };
        let mut gauss = 0.0f64;
        for i in 0..self.nv {
            gauss = (0.5 * (self.ma[i] - self.qfrc_smooth[i])).mul_add(self.qacc[i] - self.qacc_smooth[i], gauss);
        }
        self.quad_gauss[0] = gauss;
        self.cost += gauss;
    }

    /// `PrimalUpdateGrad`
    fn update_grad(&mut self) {
        for i in 0..self.nv {
            self.grad[i] = self.ma[i] - self.qfrc_smooth[i] - self.qfrc_constraint[i];
        }
    }

    /// `mj_solveLD` on the island factor of `M`.
    fn solve_m(&self, y: &[f64]) -> Vec<f64> {
        let nv = self.nv;
        let mut x = y.to_vec();
        for i in (0..nv).rev() {
            if self.m_rownnz[i] == 1 {
                continue;
            }
            let xi = x[i];
            if xi != 0.0 {
                let (start, end) = (self.m_rowadr[i], self.m_rowadr[i] + self.m_rownnz[i] - 1);
                for adr in start..end {
                    let c = self.m_colind[adr];
                    x[c] = (-self.qld[adr]).mul_add(xi, x[c]);
                }
            }
        }
        for i in 0..nv {
            x[i] *= self.qld_diag_inv[i];
        }
        for i in 0..nv {
            if self.m_rownnz[i] == 1 {
                continue;
            }
            let d = self.m_rownnz[i] - 1;
            let adr = self.m_rowadr[i];
            x[i] -= crate::mujoco_smooth::dot_sparse(&self.qld[adr..adr + d], &x, &self.m_colind[adr..adr + d]);
        }
        x
    }

    /// `MakeHessian`/`FactorizeHessian` (dense, recomputed): `L = chol(M + J'·D·J)`.
    fn factorize(&mut self) {
        let (nv, nefc) = (self.nv, self.nefc);
        for i in 0..nefc {
            self.dvec[i] = if self.state[i] == QUADRATIC { self.efc_d[i] } else { 0.0 };
        }
        if self.sparse {
            let h = crate::mujoco_sparse::hessian(&self.jrows, &self.jt, &self.dvec, &self.m_rows);
            let mut chol = self.chol.take().unwrap_or_else(|| crate::mujoco_sparse::Chol::symbolic(&h));
            chol.numeric(&h, MJ_MINVAL);
            self.chol = Some(chol);
            return;
        }
        // `mju_sqrMatTD_impl`, lower triangle
        let mut l = vec![0.0; nv * nv];
        for jr in 0..nefc {
            if self.dvec[jr] != 0.0 {
                for i in 0..nv {
                    let t = self.j[jr * nv + i];
                    if t != 0.0 {
                        let scl = t * self.dvec[jr];
                        for k in 0..=i {
                            l[i * nv + k] = self.j[jr * nv + k].mul_add(scl, l[i * nv + k]);
                        }
                    }
                }
            }
        }
        // `mju_addToSymSparse`, lower
        for i in 0..nv {
            for adr in self.m_rowadr[i]..self.m_rowadr[i] + self.m_rownnz[i] {
                l[i * nv + self.m_colind[adr]] += self.m[adr];
            }
        }
        chol_factor(&mut l, nv, MJ_MINVAL);
        self.l = l;
        if self.ncone > 0 {
            self.hessian_cone();
        }
    }

    /// `HessianConeUpdate`, dense: `Lcone` = `L` updated by each cone contact's `L'·J` rows.
    fn hessian_cone(&mut self) {
        let nv = self.nv;
        self.lcone = self.l.clone();
        let mut i = 0;
        while i < self.nefc {
            if self.state[i] == CONE {
                let dim = self.cones[self.id[i]].dim;
                let mut local = [0.0f64; 36];
                let h = &self.cone_h[self.id[i]];
                // `mju_copy(local, con->H, dim*dim)`, then factor on a dim×dim
                let mut lm = vec![0.0f64; dim * dim];
                lm.copy_from_slice(&h[..dim * dim]);
                chol_factor(&mut lm, dim, MJ_MINVAL);
                local[..dim * dim].copy_from_slice(&lm);
                let mut ltj = vec![0.0f64; dim * nv];
                for r in 0..dim {
                    for c in 0..=r {
                        let scl = local[r * dim + c];
                        for k in 0..nv {
                            ltj[c * nv + k] = self.j[(i + r) * nv + k].mul_add(scl, ltj[c * nv + k]);
                        }
                    }
                }
                for r in 0..dim {
                    let mut row = ltj[r * nv..(r + 1) * nv].to_vec();
                    chol_update(&mut self.lcone, &mut row, nv, true);
                }
                i += dim;
                continue;
            }
            i += 1;
        }
    }

    /// `HessianIncremental`
    fn hessian_incremental(&mut self, oldstate: &[i32]) {
        let nv = self.nv;
        for i in 0..self.nefc {
            let plus = if oldstate[i] != QUADRATIC && self.state[i] == QUADRATIC {
                true
            } else if oldstate[i] == QUADRATIC && self.state[i] != QUADRATIC {
                false
            } else {
                continue;
            };
            let sq = self.efc_d[i].sqrt();
            let rank = if self.sparse {
                let (cols, vals) = &self.jrows[i];
                let upd: Vec<f64> = vals.iter().map(|x| x * sq).collect();
                self.chol.as_mut().map_or(nv, |c| c.update(&upd, cols, plus))
            } else {
                let mut upd: Vec<f64> = self.j[i * nv..(i + 1) * nv].iter().map(|x| x * sq).collect();
                chol_update(&mut self.l, &mut upd, nv, plus)
            };
            if rank < nv {
                self.factorize();
                return;
            }
        }
        if self.ncone > 0 {
            self.hessian_cone();
        }
    }

    /// `PrimalPrepare`
    fn prepare(&mut self) {
        let v = &self.search;
        self.quad_gauss[1] = dot(v, &self.ma) - dot(&self.qfrc_smooth, v);
        self.quad_gauss[2] = 0.5 * dot(v, &self.mv);
        let mut i = 0;
        while i < self.nefc {
            let (jv, jaref, d) = (&self.jv, &self.jaref, &self.efc_d);
            let dj0 = d[i] * jaref[i];
            let q = &mut self.quad[3 * i..];
            q[0] = jaref[i] * dj0;
            q[1] = jv[i] * dj0;
            q[2] = jv[i] * d[i] * jv[i];
            let mut adv = 1;
            if self.typ[i] == CONTACT_ELLIPTIC {
                let con = self.cones[self.id[i]];
                let (dim, mu, fr) = (con.dim, con.mu, con.friction);
                for j in 1..dim {
                    let djj = d[i + j] * jaref[i + j];
                    q[0] = jaref[i + j].mul_add(djj, q[0]);
                    q[1] = jv[i + j].mul_add(djj, q[1]);
                    q[2] = (jv[i + j] * d[i + j]).mul_add(jv[i + j], q[2]);
                }
                let (mut uu, mut uv, mut vv) = (0.0f64, 0.0f64, 0.0f64);
                let (mut u, mut vv6) = ([0.0f64; 6], [0.0f64; 6]);
                u[0] = jaref[i] * mu;
                vv6[0] = jv[i] * mu;
                for j in 1..dim {
                    u[j] = jaref[i + j] * fr[j - 1];
                    vv6[j] = jv[i + j] * fr[j - 1];
                }
                for j in 1..dim {
                    uu = u[j].mul_add(u[j], uu);
                    uv = u[j].mul_add(vv6[j], uv);
                    vv = vv6[j].mul_add(vv6[j], vv);
                }
                q[3] = u[0];
                q[4] = vv6[0];
                q[5] = uu;
                q[6] = uv;
                q[7] = vv;
                q[8] = d[i] / ((mu * mu) * (1.0 + (mu * mu)));
                adv = dim;
            }
            q[0] *= 0.5;
            q[2] *= 0.5;
            i += adv;
        }
    }

    /// `PrimalEval`: `cost(alpha) - cost(0)` and its two derivatives.
    fn eval(&mut self, alpha: f64) -> Pnt {
        let (ne, nf) = (self.ne, self.nf);
        let mut cost = 0.0f64;
        let mut deriv = [0.0f64; 2];
        let mut qt = [0.0, self.quad_gauss[1], self.quad_gauss[2]];
        let mut i = 0;
        while i < self.nefc {
            if i < ne {
                qt[1] += self.quad[3 * i + 1];
                qt[2] += self.quad[3 * i + 2];
                i += 1;
                continue;
            }
            if i < ne + nf {
                let start = self.jaref[i];
                let dir = self.jv[i];
                let x = alpha.mul_add(dir, start);
                let f = self.floss[i];
                let d = self.efc_d[i];
                let rf = self.efc_r[i] * f;
                cost += friction_cost_dif(start, x, f, rf, d);
                if -rf < x && x < rf {
                    deriv[0] = (d * x).mul_add(dir, deriv[0]);
                    deriv[1] = (d * dir).mul_add(dir, deriv[1]);
                } else if x <= -rf {
                    deriv[0] = (-f).mul_add(dir, deriv[0]);
                } else {
                    deriv[0] = f.mul_add(dir, deriv[0]);
                }
                i += 1;
                continue;
            }
            if self.typ[i] == CONTACT_ELLIPTIC {
                let con = self.cones[self.id[i]];
                let q = &self.quad[3 * i..3 * i + 9];
                let mu = con.mu;
                let (u0, v0, uu, uv, vv, dm) = (q[3], q[4], q[5], q[6], q[7], q[8]);
                cost += elliptic_cost_dif(q, alpha, mu, dm);
                let n = alpha.mul_add(v0, u0);
                let tsqr = alpha.mul_add(2.0f64.mul_add(uv, alpha * vv), uu);
                if tsqr <= 0.0 {
                    if n < 0.0 {
                        deriv[0] += (2.0 * alpha).mul_add(q[2], q[1]);
                        deriv[1] = 2.0f64.mul_add(q[2], deriv[1]);
                    }
                } else {
                    let t = tsqr.sqrt();
                    if n >= mu * t {
                    } else if mu.mul_add(n, t) <= 0.0 {
                        deriv[0] += (2.0 * alpha).mul_add(q[2], q[1]);
                        deriv[1] = 2.0f64.mul_add(q[2], deriv[1]);
                    } else {
                        let n1 = v0;
                        let t1 = alpha.mul_add(vv, uv) / t;
                        let t2 = vv / t - alpha.mul_add(vv, uv) * t1 / (t * t);
                        let a = (-mu).mul_add(t1, n1);
                        let b = (-mu).mul_add(t, n);
                        deriv[0] = (dm * b).mul_add(a, deriv[0]);
                        deriv[1] = dm.mul_add(a.mul_add(a, b * (-mu * t2)), deriv[1]);
                    }
                }
                i += con.dim;
                continue;
            }
            let start = self.jaref[i];
            let x = alpha.mul_add(self.jv[i], start);
            let cost0 = if start < 0.0 { self.quad[3 * i] } else { 0.0 };
            if x < 0.0 {
                qt[0] += self.quad[3 * i] - cost0;
                qt[1] += self.quad[3 * i + 1];
                qt[2] += self.quad[3 * i + 2];
            } else {
                cost -= cost0;
            }
            i += 1;
        }
        cost += (alpha * alpha).mul_add(qt[2], alpha * qt[1]) + qt[0];
        deriv[0] += (2.0 * alpha).mul_add(qt[2], qt[1]);
        deriv[1] = 2.0f64.mul_add(qt[2], deriv[1]);
        if deriv[1] <= 0.0 {
            deriv[1] = MJ_MINVAL;
        }
        self.ls_iter += 1;
        Pnt { alpha, cost, deriv }
    }

    /// `updateBracket`
    fn update_bracket(&mut self, p: &mut Pnt, candidates: &[Pnt; 3], pnext: &mut Pnt) -> u8 {
        let mut flag = 0;
        for c in candidates {
            if p.deriv[0] < 0.0 && c.deriv[0] < 0.0 && p.deriv[0] < c.deriv[0] {
                *p = *c;
                flag = 1;
            } else if p.deriv[0] > 0.0 && c.deriv[0] > 0.0 && p.deriv[0] > c.deriv[0] {
                *p = *c;
                flag = 2;
            }
        }
        if flag != 0 {
            *pnext = self.eval(p.alpha - p.deriv[0] / p.deriv[1]);
        }
        flag
    }

    /// `PrimalSearch`: returns `(alpha, improvement)`.
    fn search(&mut self, tolerance: f64, ls_iterations: usize) -> (f64, f64) {
        self.ls_iter = 0;
        let snorm = norm(&self.search);
        if snorm < MJ_MINVAL {
            return (0.0, 0.0);
        }
        let gtol = tolerance * snorm / self.scale;
        self.mv = mul_sym_vec(&self.m, &self.m_rownnz, &self.m_rowadr, &self.m_colind, &self.search);
        self.jv = self.j_mul(&self.search);
        self.prepare();
        let p0 = self.eval(0.0);
        let mut p1 = self.eval(p0.alpha - p0.deriv[0] / p0.deriv[1]);
        if p1.deriv[0].abs() < gtol && (p1.alpha == 0.0 || p1.cost < 0.0) {
            return (p1.alpha, -p1.cost);
        }
        let dir = if p1.deriv[0] < 0.0 { 1.0 } else { -1.0 };
        // one-sided search
        let mut p2 = p0;
        while p1.deriv[0] * dir <= -gtol && self.ls_iter < ls_iterations {
            p2 = p1;
            let a = p1.alpha - p1.deriv[0] / p1.deriv[1];
            p1 = self.eval(a);
            if p1.deriv[0].abs() < gtol && p1.cost < 0.0 {
                return (p1.alpha, -p1.cost);
            }
        }
        if self.ls_iter >= ls_iterations {
            return (p1.alpha, -p1.cost);
        }
        // bracketed search
        let mut p2next = p1;
        let mut p1next = self.eval(p1.alpha - p1.deriv[0] / p1.deriv[1]);
        while self.ls_iter < ls_iterations {
            let pmid = self.eval(0.5 * (p1.alpha + p2.alpha));
            let candidates = [p1next, p2next, pmid];
            let mut best: Option<usize> = None;
            let mut bestcost = 0.0;
            for (i, c) in candidates.iter().enumerate() {
                if c.deriv[0].abs() < gtol && (best.is_none() || c.cost < bestcost) {
                    bestcost = c.cost;
                    best = Some(i);
                }
            }
            if let Some(b) = best {
                return (candidates[b].alpha, -candidates[b].cost);
            }
            let b1 = self.update_bracket(&mut p1, &candidates, &mut p1next);
            let b2 = self.update_bracket(&mut p2, &candidates, &mut p2next);
            if b1 == 0 && b2 == 0 {
                return (pmid.alpha, -pmid.cost);
            }
        }
        if p1.cost <= p2.cost && p1.cost < 0.0 {
            (p1.alpha, -p1.cost)
        } else if p2.cost <= p1.cost && p2.cost < 0.0 {
            (p2.alpha, -p2.cost)
        } else {
            (0.0, 0.0)
        }
    }

    /// `mj_solPrimal` with Newton steps; returns the iteration count.
    fn solve_newton(&mut self, opt: &SolverOptions) -> usize {
        let nv = self.nv;
        self.ma = mul_sym_vec(&self.m, &self.m_rownnz, &self.m_rowadr, &self.m_colind, &self.qacc);
        self.jaref = self.j_mul(&self.qacc);
        for i in 0..self.nefc {
            self.jaref[i] -= self.aref[i];
        }
        self.update_constraint();
        self.update_grad();
        let mut inertia = 0.0f64;
        for i in 0..nv {
            inertia += self.m[self.m_rowadr[i] + self.m_rownnz[i] - 1];
        }
        self.scale = 1.0 / inertia;
        let scale = self.scale;
        self.mgrad = self.solve_m(&self.grad);
        let flg_gap = max(0.0, 0.5 * scale * dot(&self.grad, &self.mgrad)) < opt.tolerance;
        let flg_gradient = scale * norm(&self.grad) < opt.tolerance;
        let mut done = flg_gap && flg_gradient;
        if !done {
            self.factorize();
            self.mgrad = self.h_solve(&self.grad);
            done = flg_gradient && max(0.0, 0.5 * scale * dot(&self.grad, &self.mgrad)) < opt.tolerance;
        }
        if !done {
            self.search = self.mgrad.iter().map(|x| x * -1.0).collect();
        }
        let mut iter = 0;
        while !done && iter < opt.iterations {
            let (alpha, ls_improvement) = self.search(opt.tolerance * opt.ls_tolerance, opt.ls_iterations);
            if alpha == 0.0 {
                break;
            }
            for i in 0..nv {
                self.qacc[i] = self.search[i].mul_add(alpha, self.qacc[i]);
                self.ma[i] = self.mv[i].mul_add(alpha, self.ma[i]);
            }
            for i in 0..self.nefc {
                self.jaref[i] = self.jv[i].mul_add(alpha, self.jaref[i]);
            }
            let oldstate = self.state.clone();
            self.update_constraint();
            self.hessian_incremental(&oldstate);
            self.update_grad();
            self.mgrad = self.h_solve(&self.grad);
            let improvement = scale * ls_improvement;
            let gradient = scale * norm(&self.grad);
            let decrement = max(0.0, 0.5 * scale * dot(&self.grad, &self.mgrad));
            iter += 1;
            if (improvement > 0.0 && improvement < opt.tolerance) || gradient < opt.tolerance || decrement < opt.tolerance {
                break;
            }
            self.search = self.mgrad.iter().map(|x| x * -1.0).collect();
        }
        iter
    }
}

/// `ellipticCostDif`: `cost(alpha) − cost(0)` of one elliptic cone.
fn elliptic_cost_dif(q: &[f64], alpha: f64, mu: f64, dm: f64) -> f64 {
    let (u0, v0, uu, uv, vv) = (q[3], q[4], q[5], q[6], q[7]);
    let (zone0, t0) = if uu <= 0.0 {
        (if u0 < 0.0 { 2 } else { 1 }, 0.0)
    } else {
        let t0 = uu.sqrt();
        (if u0 >= mu * t0 { 1 } else if mu.mul_add(u0, t0) <= 0.0 { 2 } else { 3 }, t0)
    };
    let n = alpha.mul_add(v0, u0);
    let tsqr = alpha.mul_add(2.0f64.mul_add(uv, alpha * vv), uu);
    let (zone, t) = if tsqr <= 0.0 {
        (if n < 0.0 { 2 } else { 1 }, 0.0)
    } else {
        let t = tsqr.sqrt();
        (if n >= mu * t { 1 } else if mu.mul_add(n, t) <= 0.0 { 2 } else { 3 }, t)
    };
    match (zone0, zone) {
        (1, 1) => 0.0,
        (2, 2) => (alpha * alpha).mul_add(q[2], alpha * q[1]),
        (3, 3) => {
            let tsqr_delta = alpha * 2.0f64.mul_add(uv, alpha * vv);
            let t_delta = tsqr_delta / (t + t0);
            let r_delta = alpha.mul_add(v0, -(mu * t_delta));
            let r0 = (-mu).mul_add(t0, u0);
            0.5 * dm * r_delta * 2.0f64.mul_add(r0, r_delta)
        }
        (3, 2) => {
            let dq = alpha * alpha.mul_add(q[2], q[1]);
            let b0 = mu.mul_add(u0, t0);
            dq + 0.5 * dm * b0 * b0
        }
        (2, 3) => {
            let dq = alpha * alpha.mul_add(q[2], q[1]);
            let b = mu.mul_add(n, t);
            dq - 0.5 * dm * b * b
        }
        (1, 2) => (alpha * alpha).mul_add(q[2], alpha * q[1]) + q[0],
        (1, 3) => {
            let r = (-mu).mul_add(t, n);
            0.5 * dm * r * r
        }
        (3, 1) => {
            let r0 = (-mu).mul_add(t0, u0);
            -0.5 * dm * r0 * r0
        }
        (2, 1) => -q[0],
        _ => 0.0,
    }
}

/// `frictionCost`
fn friction_cost(x: f64, f: f64, rf: f64, d: f64) -> f64 {
    if -rf < x && x < rf {
        0.5 * d * x * x
    } else if x <= -rf {
        f * (-0.5f64).mul_add(rf, -x)
    } else {
        f * (-0.5f64).mul_add(rf, x)
    }
}

/// `frictionCostDif`
fn friction_cost_dif(start: f64, x: f64, f: f64, rf: f64, d: f64) -> f64 {
    let zone = |v: f64| if -rf < v && v < rf { 0 } else if v <= -rf { -1 } else { 1 };
    match (zone(start), zone(x)) {
        (0, 0) => 0.5 * d * (x - start) * (x + start),
        (-1, -1) => f * (start - x),
        (1, 1) => f * (x - start),
        _ => friction_cost(x, f, rf, d) - friction_cost(start, f, rf, d),
    }
}

impl SmoothModel {
    /// `dof_treeid`, `tree_dofadr`, `tree_dofnum` and `body_treeid`, as the compiler numbers the trees.
    pub(crate) fn trees(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i32>) {
        let nv = self.dof_bodyid.len();
        let mut dof_treeid = vec![0; nv];
        let (mut adr, mut num): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for i in 0..nv {
            if self.dof_parentid[i] == -1 {
                adr.push(i);
                num.push(0);
            }
            dof_treeid[i] = adr.len() - 1;
            *num.last_mut().expect("a tree") += 1;
        }
        let body_treeid = (0..self.kin.bodies.len())
            .map(|b| {
                let mut w = b;
                while w != 0 && self.body_dofnum[w] == 0 {
                    w = self.kin.bodies[w].parent;
                }
                if self.body_dofnum[w] == 0 { -1 } else { dof_treeid[self.body_dofadr[w]] as i32 }
            })
            .collect();
        (dof_treeid, adr, num, body_treeid)
    }
}

impl SmoothModel {
    /// **`mj_fwdConstraint`**: `efc_b`, the warmstart (the better of `qacc_warmstart` and `qacc_smooth`),
    /// `mj_island`, then the Newton solver island by island. Dense Jacobian and pyramidal cones only; the
    /// caller checks.
    pub(crate) fn fwd_constraint(&self, sd: &SmoothData, e: &Efc, contacts: &[NativeContact], qacc_warmstart: &[f64], opt: &SolverOptions) -> Solution {
        let em_connect = |id: usize| self.efc.eq.get(id).and_then(|q| q.0.connect_bodies());
        let nv = self.dof_bodyid.len();
        let nefc = e.typ.len();
        let mut out = Solution { qfrc_constraint: vec![0.0; nv], ..Default::default() };
        if nefc == 0 {
            out.qacc = sd.qacc_smooth.clone();
            return out;
        }
        let sparse = nv >= 60;
        let jd = e.dense_j(nv);
        let cones: Vec<ConeInfo> = contacts.iter().zip(&e.contact_mu).map(|(c, &mu)| ConeInfo { dim: c.dim, mu, friction: c.friction }).collect();
        // `mj_mulJacVec`: a dense row's `mju_dot`, or a sparse row's `mju_dotSparse` over its chain
        let jmul = |v: &[f64]| -> Vec<f64> {
            if sparse {
                e.j.iter().map(|(c, x)| crate::mujoco_smooth::dot_sparse(x, v, c)).collect()
            } else {
                (0..nefc).map(|r| dot(&jd[r * nv..(r + 1) * nv], v)).collect()
            }
        };
        let ne = e.typ.iter().filter(|&&t| t == EQUALITY).count();
        let nf = e.typ.iter().filter(|&&t| t == FRICTION_DOF || t == FRICTION_TENDON).count();
        // efc_b = J·qacc_smooth − aref
        out.efc_b = jmul(&sd.qacc_smooth);
        for i in 0..nefc {
            out.efc_b[i] -= e.aref[i];
        }
        // warmstart
        let mut state = vec![0i32; nefc];
        let mut force = vec![0.0; nefc];
        let mut qacc = qacc_warmstart.to_vec();
        let mut jar = jmul(qacc_warmstart);
        for i in 0..nefc {
            jar[i] -= e.aref[i];
        }
        let mut cost_ws = constraint_update(ne, nf, &e.d, &e.r, &e.frictionloss, &jar, &e.typ, &e.id, &cones, &mut state, &mut force, None);
        let da: Vec<f64> = (0..nv).map(|i| qacc_warmstart[i] - sd.qacc_smooth[i]).collect();
        let mda = mul_sym_vec(&sd.m, &self.m_rownnz, &self.m_rowadr, &self.m_colind, &da);
        cost_ws = 0.5f64.mul_add(dot(&da, &mda), cost_ws);
        let cost_smooth = constraint_update(ne, nf, &e.d, &e.r, &e.frictionloss, &out.efc_b, &e.typ, &e.id, &cones, &mut state, &mut force, None);
        out.qfrc_constraint = if sparse {
            // `mju_mulMatTVecSparse`
            let mut q = vec![0.0f64; nv];
            for ((cols, vals), &f) in e.j.iter().zip(&force) {
                if f != 0.0 {
                    for (&c, &v) in cols.iter().zip(vals) {
                        q[c] = v.mul_add(f, q[c]);
                    }
                }
            }
            q
        } else {
            mul_mat_t_vec(&jd, &force, nefc, nv)
        };
        if cost_ws > cost_smooth {
            qacc = sd.qacc_smooth.clone();
        }
        // `mj_island`
        let (dof_treeid, _tree_dofadr, tree_dofnum, body_treeid) = self.trees();
        let ntree = tree_dofnum.len();
        let mut parent = vec![-1i32; ntree];
        let mut efc_tree = vec![0usize; nefc];
        for i in 0..nefc {
            if i > 0 && e.typ[i] == e.typ[i - 1] && e.id[i] == e.id[i - 1] {
                efc_tree[i] = efc_tree[i - 1];
                continue;
            }
            let (t1, t2) = match e.typ[i] {
                // a connect: the trees of its two bodies
                EQUALITY if em_connect(e.id[i]).is_some() => {
                    let (b1, b2) = em_connect(e.id[i]).expect("a connect");
                    (body_treeid[b1], body_treeid[b2])
                }
                // generic scan (a joint equality, a tendon's rows): the trees of the row's non-zero columns,
                // in dof order, one per tree
                EQUALITY | FRICTION_TENDON | LIMIT_TENDON => {
                    let mut trees: Vec<i32> = Vec::new();
                    if sparse {
                        // every structural entry of the chain, zero or not
                        for &c in &e.j[i].0 {
                            let t = dof_treeid[c] as i32;
                            if trees.last() != Some(&t) {
                                trees.push(t);
                            }
                        }
                    } else {
                        let row = &jd[i * nv..(i + 1) * nv];
                        let mut j = 0;
                        while j < nv {
                            if row[j] != 0.0 {
                                let t = dof_treeid[j] as i32;
                                if trees.last() != Some(&t) {
                                    trees.push(t);
                                }
                                j = _tree_dofadr[t as usize] + tree_dofnum[t as usize];
                                continue;
                            }
                            j += 1;
                        }
                    }
                    efc_tree[i] = trees[0] as usize;
                    if trees.len() == 1 {
                        dsu_merge(&mut parent, trees[0], -1);
                    } else {
                        for w in trees.windows(2) {
                            dsu_merge(&mut parent, w[0], w[1]);
                        }
                    }
                    continue;
                }
                FRICTION_DOF => (dof_treeid[e.id[i]] as i32, -2),
                LIMIT_JOINT => (dof_treeid[self.jnt_dofadr[e.id[i]]] as i32, -2),
                _ => {
                    let con = &contacts[e.id[i]];
                    (body_treeid[self.kin.geoms[con.geom[0]].body], body_treeid[self.kin.geoms[con.geom[1]].body])
                }
            };
            efc_tree[i] = if t1 >= 0 { t1 } else { t2 } as usize;
            if t2 == -2 {
                dsu_merge(&mut parent, t1, -1);
            } else {
                dsu_merge(&mut parent, t1, t2);
            }
        }
        // `mj_dsuAssign`
        let mut tree_island = vec![-1i32; ntree];
        let mut nisland = 0;
        for t in 0..ntree {
            if parent[t] == -1 {
                continue;
            }
            if parent[t] == t as i32 {
                tree_island[t] = nisland;
                nisland += 1;
            } else {
                parent[t] = parent[parent[t] as usize];
                tree_island[t] = tree_island[parent[t] as usize];
            }
        }
        out.nisland = nisland as usize;
        let island_dofs: Vec<Vec<usize>> = (0..nisland).map(|isl| (0..nv).filter(|&d| tree_island[dof_treeid[d]] == isl).collect()).collect();
        let island_efcs: Vec<Vec<usize>> = (0..nisland).map(|isl| (0..nefc).filter(|&c| tree_island[efc_tree[c]] == isl).collect()).collect();
        // unconstrained dofs take qacc_smooth
        if nisland > 0 {
            for d in 0..nv {
                if tree_island[dof_treeid[d]] < 0 {
                    qacc[d] = sd.qacc_smooth[d];
                }
            }
        }
        out.efc_state = state.clone();
        out.efc_force = force.clone();
        // per island
        for isl in 0..nisland as usize {
            let (dofs, efcs) = (&island_dofs[isl], &island_efcs[isl]);
            let (inv, ie) = (dofs.len(), efcs.len());
            let local: std::collections::HashMap<usize, usize> = dofs.iter().enumerate().map(|(k, &d)| (d, k)).collect();
            let (mut m_rownnz, mut m_rowadr, mut m_colind, mut m, mut qld) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &d in dofs {
                let (adr, nnz) = (self.m_rowadr[d], self.m_rownnz[d]);
                m_rowadr.push(m_colind.len());
                m_rownnz.push(nnz);
                for k in adr..adr + nnz {
                    m_colind.push(local[&self.m_colind[k]]);
                    m.push(sd.m[k]);
                    qld.push(sd.qld[k]);
                }
            }
            let mut j = Vec::new();
            let mut jrows: crate::mujoco_sparse::Rows = Vec::new();
            if sparse {
                // `mju_blockSparse`: the rows' chains in island-local columns
                jrows = efcs.iter().map(|&c| (e.j[c].0.iter().map(|d| local[d]).collect(), e.j[c].1.clone())).collect();
            } else {
                j = vec![0.0; ie * inv];
                for (r, &c) in efcs.iter().enumerate() {
                    for (k, &d) in dofs.iter().enumerate() {
                        j[r * inv + k] = jd[c * nv + d];
                    }
                }
            }
            let jt = if sparse { crate::mujoco_sparse::transpose(&jrows, inv) } else { Vec::new() };
            let m_rows: crate::mujoco_sparse::Rows = if sparse {
                (0..inv).map(|k| {
                    let (a, n) = (m_rowadr[k], m_rownnz[k]);
                    (m_colind[a..a + n].to_vec(), m[a..a + n].to_vec())
                }).collect()
            } else {
                Vec::new()
            };
            let gather = |v: &[f64]| dofs.iter().map(|&d| v[d]).collect::<Vec<f64>>();
            let gather_e = |v: &[f64]| efcs.iter().map(|&c| v[c]).collect::<Vec<f64>>();
            let mut ctx = Ctx {
                nv: inv,
                ne: efcs.iter().filter(|&&c| e.typ[c] == EQUALITY).count(),
                nf: efcs.iter().filter(|&&c| e.typ[c] == FRICTION_DOF || e.typ[c] == FRICTION_TENDON).count(),
                nefc: ie,
                m_rownnz,
                m_rowadr,
                m_colind,
                m,
                qld,
                qld_diag_inv: gather(&sd.qld_diag_inv),
                j,
                efc_d: gather_e(&e.d),
                efc_r: gather_e(&e.r),
                floss: gather_e(&e.frictionloss),
                aref: gather_e(&e.aref),
                typ: efcs.iter().map(|&c| e.typ[c]).collect(),
                force: gather_e(&force),
                state: efcs.iter().map(|&c| state[c]).collect(),
                qfrc_smooth: gather(&sd.qfrc_smooth),
                qacc_smooth: gather(&sd.qacc_smooth),
                qfrc_constraint: gather(&out.qfrc_constraint),
                qacc: gather(&qacc),
                jaref: vec![0.0; ie],
                jv: vec![0.0; ie],
                ma: vec![0.0; inv],
                mv: vec![0.0; inv],
                grad: vec![0.0; inv],
                mgrad: vec![0.0; inv],
                search: vec![0.0; inv],
                quad: vec![0.0; 3 * ie + 9],
                id: efcs.iter().map(|&c| e.id[c]).collect(),
                cones: cones.clone(),
                cone_h: vec![[0.0; 36]; cones.len()],
                ncone: 0,
                lcone: Vec::new(),
                dvec: vec![0.0; ie],
                l: Vec::new(),
                sparse,
                jrows,
                jt,
                m_rows,
                chol: None,
                cost: 0.0,
                quad_gauss: [0.0; 3],
                scale: 0.0,
                ls_iter: 0,
            };
            let iters = ctx.solve_newton(opt);
            out.solver_niter.push(iters);
            for (k, &d) in dofs.iter().enumerate() {
                qacc[d] = ctx.qacc[k];
                out.qfrc_constraint[d] = ctx.qfrc_constraint[k];
            }
            for (r, &c) in efcs.iter().enumerate() {
                out.efc_force[c] = ctx.force[r];
            }
        }
        out.qacc = qacc;
        // the noslip post-pass, island by island on the global forces, then `dualFinish`
        if opt.noslip_iterations > 0 {
            let ar = self.efc_ar(sd, &jd, &e.r);
            let ar_cols = sparse.then(|| self.ar_pattern(e));
            let dual = crate::mujoco_noslip::Dual { ar: &ar, ar_cols, efc_b: &out.efc_b, e, contacts };
            let scale = 1.0 / (opt.meaninertia * nv.max(1) as f64);
            let (mut force, mut state) = (out.efc_force.clone(), out.efc_state.clone());
            for (isl, efcs) in island_efcs.iter().enumerate() {
                let ne = efcs.iter().filter(|&&c| e.typ[c] == EQUALITY).count();
                let nf = efcs.iter().filter(|&&c| e.typ[c] == FRICTION_DOF || e.typ[c] == FRICTION_TENDON).count();
                out.solver_niter[isl] += crate::mujoco_noslip::sol_noslip(&dual, efcs, ne, nf, &mut force, &mut state, scale, opt.noslip_iterations, opt.noslip_tolerance);
            }
            out.qfrc_constraint = if sparse {
                let mut q = vec![0.0f64; nv];
                for ((cols, vals), &f) in e.j.iter().zip(&force) {
                    if f != 0.0 {
                        for (&c, &v) in cols.iter().zip(vals) {
                            q[c] = v.mul_add(f, q[c]);
                        }
                    }
                }
                q
            } else {
                mul_mat_t_vec(&jd, &force, nefc, nv)
            };
            let x = self.solve_ld_pub(&sd.qld, &sd.qld_diag_inv, &out.qfrc_constraint);
            out.qacc = (0..nv).map(|i| x[i] + sd.qacc_smooth[i]).collect();
            out.efc_force = force;
            out.efc_state = state;
        }
        out
    }
}

/// `mj_dsuRoot`, with path compression.
fn dsu_root(parent: &mut [i32], tree: usize) -> usize {
    let mut root = tree;
    while parent[root] != root as i32 {
        root = parent[root] as usize;
    }
    let mut t = tree;
    while parent[t] != t as i32 {
        let next = parent[t] as usize;
        parent[t] = root as i32;
        t = next;
    }
    root
}

/// `mj_dsuMerge`: activate and union two trees; `-1` is the static world.
fn dsu_merge(parent: &mut [i32], tree1: i32, tree2: i32) {
    let (mut t1, mut t2) = (tree1, tree2);
    if t1 == -1 {
        t1 = t2;
    }
    if t2 == -1 {
        t2 = t1;
    }
    let (t1, t2) = (t1 as usize, t2 as usize);
    if parent[t1] == -1 {
        parent[t1] = t1 as i32;
    }
    if parent[t2] == -1 {
        parent[t2] = t2 as i32;
    }
    if parent[t1] == parent[t2] {
        return;
    }
    let (r1, r2) = (dsu_root(parent, t1), dsu_root(parent, t2));
    if r1 < r2 {
        parent[r2] = r1 as i32;
    } else if r2 < r1 {
        parent[r1] = r2 as i32;
    }
}
