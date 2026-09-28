//! **MuJoCo's noslip post-pass in MuJoCo's own arithmetic** — `mj_makeYNumeric` and `mj_makeARNumeric`
//! (dense), `solNoSlip` and `dualFinish` from `engine_solver.c`, as MuJoCo 3.13.0 runs them on arm64.
//!
//! After the Newton solve, `<option noslip_iterations>` re-solves the FRICTION forces alone by projected
//! Gauss–Seidel on the dual matrix `AR = J·M⁻¹·Jᵀ + R`, with the regulariser `R` taken back off the blocks
//! it solves, so that a contact that should stick does not creep at the rate the soft constraint allows.
//! Every normal force stays as the primal solve left it. The forces then go back to joint space
//! (`qacc = M⁻¹·Jᵀ·f + qacc_smooth`), and each row's state is re-read with the DUAL solver's conventions.
//!
//! Contraction follows clang's `-ffp-contract=on` as read off `_solNoSlip`, `_mju_QCQP2` and `_mju_QCQP3` in
//! `libmujoco.3.13.0.dylib`: a product added to anything is fused, the left product of a sum of two.

use crate::mujoco_efc::{dot, dot_self, Efc, NativeContact, CONTACT_ELLIPTIC, CONTACT_PYRAMIDAL};
use crate::mujoco_smooth::{SmoothData, SmoothModel};
use crate::mujoco_solver::{chol_factor, chol_solve, CONE, LINEARNEG, LINEARPOS, QUADRATIC, SATISFIED};

const MJ_MINVAL: f64 = 1e-15;

impl SmoothModel {
    /// **`AR = Y·Yᵀ + diag(R)`, `Y = J·M^{-1/2}`**, dense `nefc×nefc`, row-major: `mj_solveM2` backsubstitutes
    /// each row of `J` through `L'` (a `x −= L·xᵢ` sweep from the last dof, simple dofs skipped) and scales by
    /// `1/√D`; `mju_sqrMatTD` accumulates the lower triangle over the dofs in order, a zero entry skipped, and
    /// mirrors it.
    pub(crate) fn efc_ar(&self, sd: &SmoothData, jd: &[f64], r: &[f64]) -> Vec<f64> {
        let nv = self.dof_bodyid.len();
        let nefc = r.len();
        let sqrt_inv_d: Vec<f64> = (0..nv).map(|i| 1.0 / sd.qld[self.m_rowadr[i] + self.m_rownnz[i] - 1].sqrt()).collect();
        let mut y = jd.to_vec();
        for i in (1..nv).rev() {
            if self.dof_simplenum[i] != 0 {
                continue;
            }
            let start = self.m_rowadr[i];
            let end = start + self.m_rownnz[i] - 1;
            for off in (0..nefc * nv).step_by(nv.max(1)) {
                let xi = y[off + i];
                if xi != 0.0 {
                    for adr in start..end {
                        let c = off + self.m_colind[adr];
                        y[c] = (-sd.qld[adr]).mul_add(xi, y[c]);
                    }
                }
            }
        }
        for (i, s) in sqrt_inv_d.iter().enumerate() {
            for off in (0..nefc * nv).step_by(nv.max(1)) {
                y[off + i] *= s;
            }
        }
        let mut ar = vec![0.0; nefc * nefc];
        for i in 0..nefc {
            for j in 0..nv {
                let t = y[i * nv + j];
                if t != 0.0 {
                    for k in 0..=i {
                        ar[i * nefc + k] = y[k * nv + j].mul_add(t, ar[i * nefc + k]);
                    }
                }
            }
        }
        for i in 0..nefc {
            for k in i + 1..nefc {
                ar[i * nefc + k] = ar[k * nefc + i];
            }
            ar[i * (nefc + 1)] += r[i];
        }
        ar
    }
}

/// What `solNoSlip` reads besides the forces it writes.
pub(crate) struct Dual<'a> {
    pub(crate) ar: &'a [f64],
    pub(crate) efc_b: &'a [f64],
    pub(crate) e: &'a Efc,
    pub(crate) contacts: &'a [NativeContact],
}

impl Dual<'_> {
    fn nefc(&self) -> usize {
        self.efc_b.len()
    }

    /// `residual` with `flg_subR`: `b + AR·f − R·f` for rows `i..i+dim`.
    fn residual(&self, force: &[f64], i: usize, dim: usize) -> [f64; 5] {
        let n = self.nefc();
        let mut res = [0.0; 5];
        for j in 0..dim {
            res[j] = self.efc_b[i + j] + dot(&self.ar[(i + j) * n..(i + j + 1) * n], force);
        }
        for j in 0..dim {
            res[j] = (-self.e.r[i + j]).mul_add(force[i + j], res[j]);
        }
        res
    }

    /// `extractBlock` with `flg_subR`: the `n×n` diagonal block of `AR` at `start`, `R` taken off its
    /// diagonal and clamped at `1e-10`.
    fn block(&self, start: usize, n: usize) -> [f64; 25] {
        let nefc = self.nefc();
        let mut ac = [0.0; 25];
        for j in 0..n {
            ac[j * n..(j + 1) * n].copy_from_slice(&self.ar[start + (start + j) * nefc..start + (start + j) * nefc + n]);
        }
        for j in 0..n {
            ac[j * (n + 1)] -= self.e.r[start + j];
            ac[j * (n + 1)] = max(1e-10, ac[j * (n + 1)]);
        }
        ac
    }

    /// `dualState`: equality QUADRATIC; friction by its force against the loss (the SIGNS opposite to the
    /// primal's); limits and contacts by the force, an elliptic cone by its zone in the regular cone.
    fn state(&self, force: &[f64], state: &mut [i32], ne: usize, nf: usize, efclist: &[usize]) {
        let floss = &self.e.frictionloss;
        for &i in &efclist[..ne] {
            state[i] = QUADRATIC;
        }
        for &i in &efclist[ne..ne + nf] {
            state[i] = if force[i] <= -floss[i] {
                LINEARPOS
            } else if force[i] >= floss[i] {
                LINEARNEG
            } else {
                QUADRATIC
            };
        }
        let mut c = ne + nf;
        while c < efclist.len() {
            let i = efclist[c];
            if self.e.typ[i] != CONTACT_ELLIPTIC {
                state[i] = if force[i] <= 0.0 { SATISFIED } else { QUADRATIC };
                c += 1;
                continue;
            }
            let id = self.e.id[i];
            let con = &self.contacts[id];
            let (dim, mu) = (con.dim, self.e.contact_mu[id]);
            let mut f = [0.0; 6];
            f[0] = force[i] / mu;
            for j in 1..dim {
                f[j] = force[i + j] / con.friction[j - 1];
            }
            let (n, t) = (f[0], dot_self(&f[1..dim]).sqrt());
            let result = if mu * n >= t {
                SATISFIED
            } else if mu.mul_add(t, n) <= 0.0 {
                QUADRATIC
            } else {
                CONE
            };
            state[i..i + dim].fill(result);
            c += dim;
        }
    }
}

fn max(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}

/// `costChange`: the cost the block update changed; a rise above `1e-10` restores the old forces.
fn cost_change(a: &[f64], force: &mut [f64], old: &[f64], res: &[f64], dim: usize) -> f64 {
    let mut change = if dim == 1 {
        let delta = force[0] - old[0];
        (0.5 * delta * delta).mul_add(a[0], delta * res[0])
    } else {
        let delta: Vec<f64> = (0..dim).map(|k| force[k] - old[k]).collect();
        // `mju_mulVecMatVec`
        let mut vmv = 0.0f64;
        for i in 0..dim {
            vmv = delta[i].mul_add(dot(&a[i * dim..(i + 1) * dim], &delta), vmv);
        }
        0.5f64.mul_add(vmv, dot(&delta, &res[..dim]))
    };
    if change > 1e-10 {
        force[..dim].copy_from_slice(&old[..dim]);
        change = 0.0;
    }
    change
}

/// `mju_QCQP2`: `min ½xᵀAx + xᵀb` subject to `Σ (xᵢ/dᵢ)² ≤ r²`; the solution, and whether the constraint was
/// active.
fn qcqp2(ain: &[f64], bin: &[f64], d: &[f64], r: f64) -> ([f64; 5], bool) {
    let (b1, b2) = (bin[0] * d[0], bin[1] * d[1]);
    let (a11, a22, a12) = (ain[0] * d[0] * d[0], ain[3] * d[1] * d[1], ain[1] * d[0] * d[1]);
    let (mut la, mut v1, mut v2) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..20 {
        let det = (a11 + la).mul_add(a22 + la, -(a12 * a12));
        if det < 1e-10 {
            return ([0.0; 5], false);
        }
        let detinv = 1.0 / det;
        let (p11, p22, p12) = ((a22 + la) * detinv, (a11 + la) * detinv, -a12 * detinv);
        v1 = (-p11).mul_add(b1, -(p12 * b2));
        v2 = (-p12).mul_add(b1, -(p22 * b2));
        let val = (-r).mul_add(r, v1.mul_add(v1, v2 * v2));
        if val < 1e-10 {
            break;
        }
        let deriv = -2.0 * (p22 * v2).mul_add(v2, (p11 * v1).mul_add(v1, 2.0 * p12 * v1 * v2));
        let delta = -val / deriv;
        if delta < 1e-10 {
            break;
        }
        la += delta;
    }
    ([v1 * d[0], v2 * d[1], 0.0, 0.0, 0.0], la != 0.0)
}

/// `mju_QCQP3`
fn qcqp3(ain: &[f64], bin: &[f64], d: &[f64], r: f64) -> ([f64; 5], bool) {
    let (b1, b2, b3) = (bin[0] * d[0], bin[1] * d[1], bin[2] * d[2]);
    let (a11, a22, a33) = (ain[0] * d[0] * d[0], ain[4] * d[1] * d[1], ain[8] * d[2] * d[2]);
    let (a12, a13, a23) = (ain[1] * d[0] * d[1], ain[2] * d[0] * d[2], ain[5] * d[1] * d[2]);
    let (mut la, mut v1, mut v2, mut v3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for _ in 0..20 {
        let mut p11 = (a22 + la).mul_add(a33 + la, -(a23 * a23));
        let mut p22 = (a11 + la).mul_add(a33 + la, -(a13 * a13));
        let mut p33 = (a11 + la).mul_add(a22 + la, -(a12 * a12));
        let mut p12 = a13.mul_add(a23, -(a12 * (a33 + la)));
        let mut p13 = a12.mul_add(a23, -(a13 * (a22 + la)));
        let mut p23 = a12.mul_add(a13, -(a23 * (a11 + la)));
        let det = a13.mul_add(p13, (a11 + la).mul_add(p11, a12 * p12));
        if det < 1e-10 {
            return ([0.0; 5], false);
        }
        let detinv = 1.0 / det;
        p11 *= detinv;
        p22 *= detinv;
        p33 *= detinv;
        p12 *= detinv;
        p13 *= detinv;
        p23 *= detinv;
        v1 = (-p13).mul_add(b3, (-p11).mul_add(b1, -(p12 * b2)));
        v2 = (-p23).mul_add(b3, (-p12).mul_add(b1, -(p22 * b2)));
        v3 = (-p33).mul_add(b3, (-p13).mul_add(b1, -(p23 * b2)));
        let val = (-r).mul_add(r, v3.mul_add(v3, v1.mul_add(v1, v2 * v2)));
        if val < 1e-10 {
            break;
        }
        let s = (p33 * v3).mul_add(v3, (p11 * v1).mul_add(v1, p22 * v2 * v2));
        let t = (p23 * v2).mul_add(v3, (p12 * v1).mul_add(v2, p13 * v1 * v3));
        let deriv = (-2.0f64).mul_add(s, -4.0 * t);
        let delta = -val / deriv;
        if delta < 1e-10 {
            break;
        }
        la += delta;
    }
    ([v1 * d[0], v2 * d[1], v3 * d[2], 0.0, 0.0], la != 0.0)
}

/// `mju_QCQP` for `n` up to 5: Newton on the multiplier with a Cholesky solve of `A + λ`.
fn qcqp_n(ain: &[f64], bin: &[f64], d: &[f64], r: f64, n: usize) -> ([f64; 5], bool) {
    let mut a = [0.0; 25];
    let mut b = [0.0; 5];
    for i in 0..n {
        b[i] = bin[i] * d[i];
        for j in 0..n {
            a[j + i * n] = ain[j + i * n] * d[i] * d[j];
        }
    }
    let mut la = 0.0f64;
    let mut res = vec![0.0; n];
    for _ in 0..20 {
        let mut ala = a[..n * n].to_vec();
        for i in 0..n {
            ala[i * (n + 1)] += la;
        }
        if chol_factor(&mut ala, n, 1e-10) < n {
            return ([0.0; 5], false);
        }
        res = chol_solve(&ala, &b[..n], n);
        for x in res.iter_mut() {
            *x *= -1.0;
        }
        let val = (-r).mul_add(r, dot(&res, &res));
        if val < 1e-10 {
            break;
        }
        let tmp = chol_solve(&ala, &res, n);
        let deriv = -2.0 * dot(&res, &tmp);
        let delta = -val / deriv;
        if delta < 1e-10 {
            break;
        }
        la += delta;
    }
    let mut out = [0.0; 5];
    for i in 0..n {
        out[i] = res[i] * d[i];
    }
    (out, la != 0.0)
}

/// `solveQCQP`: the tangential forces of an elliptic contact at `i`, put on the ellipsoid when the
/// constraint is active (`projectEllipsoid` with `feasible = 0`).
fn solve_qcqp(force: &mut [f64], i: usize, dim: usize, ac: &[f64], bc: &[f64], mu: &[f64; 5]) {
    let (mut v, active) = match dim {
        3 => qcqp2(ac, bc, mu, force[i]),
        4 => qcqp3(ac, bc, mu, force[i]),
        _ => qcqp_n(ac, bc, mu, force[i], dim - 1),
    };
    if active {
        let mut s = 0.0f64;
        for j in 0..dim - 1 {
            s += v[j] * v[j] / (mu[j] * mu[j]);
        }
        let normal2 = force[i] * force[i];
        let scl = (normal2 / max(MJ_MINVAL, s)).sqrt();
        for x in v[..dim - 1].iter_mut() {
            *x *= scl;
        }
    }
    force[i + 1..i + dim].copy_from_slice(&v[..dim - 1]);
}

/// **`solNoSlip`** over one island's rows (`efclist`: `ne` equalities, `nf` frictions, then limits and
/// contacts, in MuJoCo's order), on the global `efc_force` and `efc_state`. Returns the iterations run,
/// which MuJoCo adds to the island's `solver_niter`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sol_noslip(p: &Dual, efclist: &[usize], ne: usize, nf: usize, force: &mut [f64], state: &mut [i32], scale: f64, maxiter: usize, tolerance: f64) -> usize {
    let nefc = p.nefc();
    let floss = &p.e.frictionloss;
    // `ARdiaginv` with `flg_subR`
    let arinv: Vec<f64> = efclist.iter().map(|&i| 1.0 / max(MJ_MINVAL, p.ar[i * (nefc + 1)] - p.e.r[i])).collect();
    p.state(force, state, ne, nf, efclist);
    let mut iter = 0;
    while iter < maxiter {
        let mut improvement = 0.0f64;
        if iter == 0 {
            for &i in efclist {
                improvement = (0.5 * force[i] * force[i]).mul_add(p.e.r[i], improvement);
            }
        }
        // dry friction
        for c in ne..ne + nf {
            let i = efclist[c];
            let res = p.residual(force, i, 1);
            let old = force[i];
            force[i] = (-res[0]).mul_add(arinv[c], force[i]);
            if force[i] < -floss[i] {
                force[i] = -floss[i];
            } else if force[i] > floss[i] {
                force[i] = floss[i];
            }
            let delta = force[i] - old;
            improvement -= delta.mul_add(res[0], 0.5 * delta * delta / arinv[c]);
        }
        // contact friction
        let mut c = ne + nf;
        while c < efclist.len() {
            let i = efclist[c];
            let typ = p.e.typ[i];
            if typ == CONTACT_PYRAMIDAL {
                let con = &p.contacts[p.e.id[i]];
                let dim = con.dim;
                let mut j = i;
                while j < i + 2 * (dim - 1) {
                    let res = p.residual(force, j, 2);
                    let old = [force[j], force[j + 1]];
                    let ac = p.block(j, 2);
                    let mut bc = [res[0], res[1]];
                    for k in 0..2 {
                        bc[k] -= dot(&ac[k * 2..k * 2 + 2], &old);
                    }
                    let mid = 0.5 * (force[j] + force[j + 1]);
                    let k1 = ac[0] + ac[3] - ac[1] - ac[2];
                    let k0 = mid.mul_add(ac[0] - ac[3], bc[0]) - bc[1];
                    if k1 < MJ_MINVAL {
                        force[j] = mid;
                        force[j + 1] = mid;
                    } else {
                        let y = -k0 / k1;
                        if y < -mid {
                            force[j] = 0.0;
                            force[j + 1] = 2.0 * mid;
                        } else if y > mid {
                            force[j] = 2.0 * mid;
                            force[j + 1] = 0.0;
                        } else {
                            force[j] = mid + y;
                            force[j + 1] = mid - y;
                        }
                    }
                    improvement -= cost_change(&ac, &mut force[j..j + 2], &old, &res, 2);
                    j += 2;
                }
                c += 2 * (dim - 1);
                continue;
            }
            if typ == CONTACT_ELLIPTIC {
                let con = &p.contacts[p.e.id[i]];
                let dim = con.dim;
                let res = p.residual(force, i + 1, dim - 1);
                let old: Vec<f64> = force[i + 1..i + dim].to_vec();
                let n = dim - 1;
                let ac = p.block(i + 1, n);
                let mut bc = [0.0; 5];
                bc[..n].copy_from_slice(&res[..n]);
                for j in 0..n {
                    bc[j] -= dot(&ac[j * n..(j + 1) * n], &old);
                }
                if force[i] < MJ_MINVAL {
                    force[i + 1..i + dim].fill(0.0);
                } else {
                    solve_qcqp(force, i, dim, &ac[..n * n], &bc[..n], &con.friction);
                }
                improvement -= cost_change(&ac[..n * n], &mut force[i + 1..i + dim], &old, &res, n);
                c += dim;
                continue;
            }
            c += 1;
        }
        p.state(force, state, ne, nf, efclist);
        improvement *= scale;
        iter += 1;
        if improvement < tolerance {
            break;
        }
    }
    iter
}
