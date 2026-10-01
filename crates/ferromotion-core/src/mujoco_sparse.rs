//! **MuJoCo's sparse Newton machinery in MuJoCo's own arithmetic** — what `mj_solPrimal` runs in place of
//! the dense Hessian once a model has 60 or more dofs: `H = M + J'·D·J` from `mju_sqrMatTDSparseSymbolic`/
//! `…Numeric` and `mju_addToMatSparse`, and its reverse Cholesky factor `H = L'·L` from
//! `mju_cholFactorSymbolic`/`…Numeric`, solved by `mju_cholSolveSparse` and updated rank-one by
//! `mju_cholUpdateSparse` (`engine_util_sparse.c`, `engine_util_solve.c`, MuJoCo 3.13.0).
//!
//! Rows are `(columns, values)`, columns ascending. The supernode batching MuJoCo uses to speed these up
//! changes no arithmetic — every entry is accumulated over the same constraints in the same order — so it
//! is not reproduced.

use crate::mujoco_smooth::dot_sparse;

/// A sparse matrix, row by row.
pub(crate) type Rows = Vec<(Vec<usize>, Vec<f64>)>;

/// `mju_transposeSparse`: each column's rows, ascending, with the values.
pub(crate) fn transpose(rows: &Rows, nc: usize) -> Rows {
    let mut t: Rows = vec![(Vec::new(), Vec::new()); nc];
    for (r, (cols, vals)) in rows.iter().enumerate() {
        for (&c, &v) in cols.iter().zip(vals) {
            t[c].0.push(r);
            t[c].1.push(v);
        }
    }
    t
}

/// **`H = J'·D·J` (lower triangle) `+ M`**: the pattern of `mju_sqrMatTDSparseSymbolic` (row `r` holds every
/// `c < r` some constraint couples to `r`, and `r` itself when a constraint touches it), the values of
/// `…Numeric` (each row accumulated over its constraints in order, `buffer[c] += (D_k·J_kr)·J_kc`, a zero
/// scale skipped), then `M`'s lower triangle merged in (`mju_combineSparse`, `1·h + 1·m`).
pub(crate) fn hessian(j: &Rows, jt: &Rows, d: &[f64], m: &Rows) -> Rows {
    let nv = jt.len();
    let mut h: Rows = Vec::with_capacity(nv);
    let mut buffer = vec![0.0f64; nv];
    let mut mark = vec![false; nv];
    for r in 0..nv {
        // pattern
        let mut cols: Vec<usize> = Vec::new();
        for &k in &jt[r].0 {
            for &c in &j[k].0 {
                if c < r && !mark[c] {
                    mark[c] = true;
                    cols.push(c);
                }
            }
        }
        for &c in &cols {
            mark[c] = false;
        }
        cols.sort_unstable();
        if !jt[r].0.is_empty() {
            cols.push(r);
        }
        // values
        for (&k, &vt) in jt[r].0.iter().zip(&jt[r].1) {
            let scale = d[k] * vt;
            if scale == 0.0 {
                continue;
            }
            for (&c, &v) in j[k].0.iter().zip(&j[k].1) {
                if c > r {
                    break;
                }
                buffer[c] = scale.mul_add(v, buffer[c]);
            }
        }
        let vals: Vec<f64> = cols.iter().map(|&c| std::mem::take(&mut buffer[c])).collect();
        // `mju_combineSparse(H_r, M_r, 1, 1)`
        let (mc, mv) = &m[r];
        if cols == *mc {
            let v = vals.iter().zip(mv).map(|(h, m)| h.mul_add(1.0, m * 1.0)).collect();
            h.push((cols, v));
            continue;
        }
        let (mut oc, mut ov) = (Vec::new(), Vec::new());
        let (mut a, mut b) = (0, 0);
        while a < cols.len() || b < mc.len() {
            if b >= mc.len() || (a < cols.len() && cols[a] < mc[b]) {
                oc.push(cols[a]);
                ov.push(1.0 * vals[a]);
                a += 1;
            } else if a >= cols.len() || mc[b] < cols[a] {
                oc.push(mc[b]);
                ov.push(1.0 * mv[b]);
                b += 1;
            } else {
                oc.push(cols[a]);
                ov.push(1.0f64.mul_add(vals[a], 1.0 * mv[b]));
                a += 1;
                b += 1;
            }
        }
        h.push((oc, ov));
    }
    h
}

/// A reverse Cholesky factor `H = L'·L` on the pattern `mju_cholFactorSymbolic` gives it: each row's
/// columns ascending with its diagonal last, and each column's rows (`LT`) in the order the elimination tree
/// was walked, with where each entry sits in `L`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Chol {
    pub(crate) l_colind: Vec<Vec<usize>>,
    pub(crate) l: Vec<Vec<f64>>,
    lt: Vec<Vec<(usize, usize)>>,
    /// the cone-fold gate's inputs: each row's nonzeros summed along its reverse elimination-tree path, and
    /// the factor's squared row counts summed
    pub(crate) pathcost: Vec<u64>,
    pub(crate) lflops: u64,
}

impl Chol {
    /// `mju_cholFactorSymbolic` from `H`'s lower-triangle rows (its upper triangle read through the
    /// transpose).
    pub(crate) fn symbolic(h: &Rows) -> Chol {
        let n = h.len();
        // the upper triangle: row r of H' lists the rows i > r with H[i][r]
        let mut upper: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, (cols, _)) in h.iter().enumerate() {
            for &c in cols {
                upper[c].push(i);
            }
        }
        let mut parent = vec![-1i64; n];
        let mut flag = vec![0usize; n];
        // columns of each L row, collected in decreasing column order
        let mut rev_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        // LT column r: (row i, index of r in row i counted from the END of i's off-diagonals)
        let mut lt_raw: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for r in (0..n).rev() {
            parent[r] = -1;
            flag[r] = r;
            for &i0 in &upper[r] {
                if i0 <= r {
                    continue;
                }
                let mut i = i0;
                while flag[i] != r {
                    if parent[i] == -1 {
                        parent[i] = r as i64;
                    }
                    lt_raw[r].push((i, rev_cols[i].len()));
                    rev_cols[i].push(r);
                    flag[i] = r;
                    i = parent[i] as usize;
                }
            }
        }
        let mut l_colind: Vec<Vec<usize>> = Vec::with_capacity(n);
        for (r, rc) in rev_cols.iter().enumerate() {
            let mut cols: Vec<usize> = rc.iter().rev().copied().collect();
            cols.push(r);
            l_colind.push(cols);
        }
        // the diagonal leads each LT column, then the rows as the walk found them
        let lt = (0..n)
            .map(|r| {
                let mut col = vec![(r, l_colind[r].len() - 1)];
                for &(i, from_end) in &lt_raw[r] {
                    // written at `cursor[i]`, counting down from the slot before the diagonal
                    col.push((i, l_colind[i].len() - 2 - from_end));
                }
                col
            })
            .collect();
        let l = l_colind.iter().map(|c| vec![0.0; c.len()]).collect();
        let mut pathcost = vec![0u64; n];
        let mut lflops = 0u64;
        for r in 0..n {
            let nnz = l_colind[r].len();
            let parent = if nnz >= 2 { Some(l_colind[r][nnz - 2]) } else { None };
            pathcost[r] = nnz as u64 + parent.map_or(0, |p| pathcost[p]);
            lflops += (nnz * nnz) as u64;
        }
        Chol { l_colind, l, lt, pathcost, lflops }
    }

    /// `mju_cholFactorNumeric`: rows from the last back, each one's `H` row less `L[c][r]·L[c][..=r]` for the
    /// rows `c` below it, diagonal floored at `mindiag`; returns the rank.
    pub(crate) fn numeric(&mut self, h: &Rows, mindiag: f64) -> usize {
        let n = h.len();
        let mut rank = n;
        let mut dense = vec![0.0f64; n];
        for r in (0..n).rev() {
            for (&c, &v) in h[r].0.iter().zip(&h[r].1) {
                dense[c] = v;
            }
            for k in 1..self.lt[r].len() {
                let (c, idx) = self.lt[r][k];
                let lcr = self.l[c][idx];
                for i in 0..=idx {
                    let col = self.l_colind[c][i];
                    dense[col] = (-lcr).mul_add(self.l[c][i], dense[col]);
                }
            }
            let mut diag = dense[r];
            if diag < mindiag {
                diag = mindiag;
                rank -= 1;
            }
            let lrr = diag.sqrt();
            let inv = 1.0 / lrr;
            let nnz = self.l_colind[r].len();
            for i in 0..nnz - 1 {
                self.l[r][i] = dense[self.l_colind[r][i]] * inv;
            }
            self.l[r][nnz - 1] = lrr;
            for &c in &self.l_colind[r] {
                dense[c] = 0.0;
            }
        }
        rank
    }

    /// `mju_cholSolveSparse`: `L'·L·x = vec`.
    pub(crate) fn solve(&self, vec: &[f64]) -> Vec<f64> {
        let n = vec.len();
        let mut res = vec.to_vec();
        for i in (0..n).rev() {
            if res[i] != 0.0 {
                let nnz = self.l[i].len();
                res[i] /= self.l[i][nnz - 1];
                let tmp = res[i];
                for j in 0..nnz - 1 {
                    let c = self.l_colind[i][j];
                    res[c] = (-self.l[i][j]).mul_add(tmp, res[c]);
                }
            }
        }
        for i in 0..n {
            let nnz = self.l[i].len();
            if nnz > 1 {
                res[i] -= dot_sparse(&self.l[i][..nnz - 1], &res, &self.l_colind[i][..nnz - 1]);
            }
            res[i] /= self.l[i][nnz - 1];
        }
        res
    }

    /// `mju_cholUpdateSparse`: `L'·L ± x·x'` for `x` on `x_ind` (ascending), by Givens rotations from
    /// its last row back; returns the rank.
    pub(crate) fn update(&mut self, x: &[f64], x_ind: &[usize], plus: bool) -> usize {
        let n = self.l.len();
        if x.is_empty() {
            return n;
        }
        let start = x_ind[x.len() - 1];
        let mut dense = vec![0.0f64; start + 1];
        for (&c, &v) in x_ind.iter().zip(x) {
            dense[c] = v;
        }
        let mut rank = n;
        for row in (0..=start).rev() {
            if dense[row] == 0.0 {
                continue;
            }
            let nnz = self.l[row].len() - 1;
            let diag = self.l[row][nnz];
            let xr = dense[row];
            let mut tmp = diag.mul_add(diag, if plus { xr * xr } else { -xr * xr });
            if tmp < 1e-15 {
                tmp = 1e-15;
                rank -= 1;
            }
            let r = tmp.sqrt();
            self.l[row][nnz] = r;
            let c = diag / r;
            let s = -xr / r;
            let ss = if plus { -s } else { s };
            for i in 0..nnz {
                let j = self.l_colind[row][i];
                let (dj, mv) = (dense[j], self.l[row][i]);
                self.l[row][i] = c.mul_add(mv, ss * dj);
                dense[j] = s.mul_add(mv, c * dj);
            }
        }
        rank
    }
}
