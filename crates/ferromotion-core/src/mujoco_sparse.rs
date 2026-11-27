//! **MuJoCo's sparse Newton machinery in MuJoCo's own arithmetic** — what `mj_solPrimal` runs in place of
//! the dense Hessian once a model has 60 or more dofs: `H = M + J'·D·J` from `mju_sqrMatTDSparseSymbolic`/
//! `…Numeric` and `mju_addToMatSparse`, and its reverse Cholesky factor `H = L'·L` from
//! `mju_cholFactorSymbolic`/`…Numeric`, solved by `mju_cholSolveSparse` and updated rank-one by
//! `mju_cholUpdateSparse` (`engine_util_sparse.c`, `engine_util_solve.c`, MuJoCo 3.13.0).
//!
//! Rows are `(columns, values)`, columns ascending. `64156abe` factors and solves one row at a time.
//! `5491d70f` and `4849b2dd` merge up to three rows into a supernode (`L_super`) and factor and solve
//! those rows together; that changes the rounding, so the 4.x path does it.

use crate::mujoco_smooth::dot_sparse;

fn oracle_contract_off() -> bool {
    crate::fp_contract::contract_off_line()
}

/// Forward dot inside `mju_cholSolveSparse` (`64156abe`).
///
/// For four or more entries, pairs accumulate as `(p0, p1)` and `(p2, p3)`, then
/// `high + low` of their sum. The tail is `acc + product`. Shorter dots are only that tail.
fn chol_solve_dot(l: &[f64], res: &[f64], ind: &[usize]) -> f64 {
    let n = l.len();
    if n < 4 {
        let mut acc = 0.0;
        for i in 0..n {
            acc += l[i] * res[ind[i]];
        }
        return acc;
    }
    let mut a01 = [0.0, 0.0];
    let mut a23 = [0.0, 0.0];
    let prefix = n & !3;
    let mut i = 0;
    while i < prefix {
        a01[0] += l[i] * res[ind[i]];
        a01[1] += l[i + 1] * res[ind[i + 1]];
        a23[0] += l[i + 2] * res[ind[i + 2]];
        a23[1] += l[i + 3] * res[ind[i + 3]];
        i += 4;
    }
    let low = a01[0] + a23[0];
    let high = a01[1] + a23[1];
    let mut acc = high + low;
    while i < n {
        acc += l[i] * res[ind[i]];
        i += 1;
    }
    acc
}

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
            // `64156abe` `mju_sqrMatTDSparseNumeric`: `scale = vt * D`, then
            // `(J * scale) + acc` (`mulsd` with `J` as the destination, `addsd` of the accumulator).
            // 4.0.0 and 4.0.1 `mju_sqrMatTDSparseNumeric` is not this 3.13 schedule.
            // `64156abe`: `scale = vt * D`, then `(J * scale) + acc`.
            // `5491d70f`: `scale = D * vt`, then `acc + (scale * J)`.
            let four = crate::fp_contract::oracle_4_0_schedules();
            let scale = if crate::fp_contract::contract_off() { vt * d[k] } else { d[k] * vt };
            if scale == 0.0 {
                continue;
            }
            for (&c, &v) in j[k].0.iter().zip(&j[k].1) {
                if c > r {
                    break;
                }
                buffer[c] = if crate::fp_contract::contract_off() {
                    (v * scale) + buffer[c]
                } else if four {
                    buffer[c] + scale * v
                } else {
                    scale.mul_add(v, buffer[c])
                };
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
    /// `5491d70f` `L_super`: rows merged downward from this row, at most 3. `64156abe` leaves this at 1.
    l_super: Vec<usize>,
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
        // nodes of LT[r+1] also reached while walking row r (`5491d70f` overlap)
        let mut overlap_next = vec![0usize; n];
        for r in (0..n).rev() {
            parent[r] = -1;
            flag[r] = r;
            let mut overlap = 0usize;
            for &i0 in &upper[r] {
                if i0 <= r {
                    continue;
                }
                let mut i = i0;
                while flag[i] != r {
                    if r + 1 < n && flag[i] == r + 1 {
                        overlap += 1;
                    }
                    if parent[i] == -1 {
                        parent[i] = r as i64;
                    }
                    lt_raw[r].push((i, rev_cols[i].len()));
                    rev_cols[i].push(r);
                    flag[i] = r;
                    i = parent[i] as usize;
                }
            }
            overlap_next[r] = overlap;
        }
        let mut l_colind: Vec<Vec<usize>> = Vec::with_capacity(n);
        for (r, rc) in rev_cols.iter().enumerate() {
            let mut cols: Vec<usize> = rc.iter().rev().copied().collect();
            cols.push(r);
            l_colind.push(cols);
        }
        // the diagonal leads each LT column, then the rows as the walk found them
        let lt: Vec<Vec<(usize, usize)>> = (0..n)
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
            let parent_col = if nnz >= 2 { Some(l_colind[r][nnz - 2]) } else { None };
            pathcost[r] = nnz as u64 + parent_col.map_or(0, |p| pathcost[p]);
            lflops += (nnz * nnz) as u64;
        }
        // `L_super[r+1] = 1` when row r+1 merges into r. Then each top row's size is
        // `1 + merge + (merge && the row below also merges)`, at most 3.
        let mut l_super = vec![0usize; n];
        for r in 0..n.saturating_sub(1) {
            let child = r + 1;
            let merges = parent[child] == r as i64
                && l_colind[r].len() + 1 == l_colind[child].len()
                && lt[r].len() == lt[child].len() + 1
                && overlap_next[r] == lt[child].len();
            l_super[child] = usize::from(merges);
        }
        for r in (0..n).rev() {
            let merge = l_super[r];
            let extra = merge == 1 && r > 0 && l_super[r - 1] == 1;
            l_super[r] = 1 + merge + usize::from(extra);
        }
        Chol { l_colind, l, lt, l_super, pathcost, lflops }
    }

    /// `mju_cholFactorNumeric`: rows from the last back, each one's `H` row less `L[c][r]·L[c][..=r]` for the
    /// rows `c` below it, diagonal floored at `mindiag`; returns the rank.
    /// `5491d70f` factors supernodes of 2 or 3 rows in one step.
    pub(crate) fn numeric(&mut self, h: &Rows, mindiag: f64) -> usize {
        if crate::fp_contract::oracle_4_0_schedules() {
            return self.numeric_super(h, mindiag);
        }
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
                    // `64156abe`: `mulsd` then `subsd`, `dense - L·Lrow`.
                    dense[col] = if oracle_contract_off() {
                        dense[col] - lcr * self.l[c][i]
                    } else {
                        (-lcr).mul_add(self.l[c][i], dense[col])
                    };
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
    /// `5491d70f` fuses a 2- or 3-row supernode into one backward and one forward step.
    pub(crate) fn solve(&self, vec: &[f64]) -> Vec<f64> {
        if crate::fp_contract::oracle_4_0_schedules() {
            return self.solve_super(vec);
        }
        let n = vec.len();
        let mut res = vec.to_vec();
        for i in (0..n).rev() {
            if res[i] != 0.0 {
                let nnz = self.l[i].len();
                res[i] /= self.l[i][nnz - 1];
                let tmp = res[i];
                for j in 0..nnz - 1 {
                    let c = self.l_colind[i][j];
                    // `64156abe`: `res[c] - L·tmp`.
                    res[c] = if oracle_contract_off() {
                        res[c] - self.l[i][j] * tmp
                    } else {
                        (-self.l[i][j]).mul_add(tmp, res[c])
                    };
                }
            }
        }
        for i in 0..n {
            let nnz = self.l[i].len();
            if nnz > 1 {
                let dot = if oracle_contract_off() {
                    chol_solve_dot(&self.l[i][..nnz - 1], &res, &self.l_colind[i][..nnz - 1])
                } else {
                    dot_sparse(&self.l[i][..nnz - 1], &res, &self.l_colind[i][..nnz - 1])
                };
                res[i] -= dot;
            }
            res[i] /= self.l[i][nnz - 1];
        }
        res
    }

    /// `5491d70f` `mju_cholFactorNumeric`: leaf rows scale `H` directly; every other row is a supernode
    /// of 1, 2, or 3. Descendant updates are subtracted before the supernode is scaled, and scaling row
    /// `r` immediately eliminates it from rows `r-1` and `r-2`.
    fn numeric_super(&mut self, h: &Rows, mindiag: f64) -> usize {
        let n = h.len();
        let mut rank = n;
        let mut dense = vec![0.0f64; n];
        let mut r_i = n as isize - 1;
        while r_i >= 0 {
            let r = r_i as usize;
            let l_nnz = self.l_colind[r].len();
            let lt_nnz = self.lt[r].len();
            let h_nofill = h[r].0.len() >= l_nnz && h[r].0[l_nnz - 1] == r;
            if lt_nnz == 1 && h_nofill {
                let mut diag = h[r].1[l_nnz - 1];
                if !(diag >= mindiag) {
                    diag = mindiag;
                    rank -= 1;
                    for i in 0..l_nnz - 1 {
                        self.l[r][i] = 0.0;
                    }
                } else {
                    let inv = 1.0 / diag.sqrt();
                    for i in 0..l_nnz - 1 {
                        self.l[r][i] = h[r].1[i] * inv;
                    }
                }
                self.l[r][l_nnz - 1] = diag.sqrt();
                r_i -= 1;
                continue;
            }
            let ns = self.l_super.get(r).copied().unwrap_or(1).clamp(1, 3).min(r + 1);
            rank -= self.factor_supernode(h, r, ns, mindiag, &mut dense);
            r_i -= ns as isize;
        }
        rank
    }

    /// One `cholFactorSupernode` call. `r` is the highest row of the supernode.
    fn factor_supernode(&mut self, h: &Rows, r: usize, ns: usize, mindiag: f64, dense: &mut [f64]) -> usize {
        let l_nnz = self.l_colind[r].len();
        let cols = self.l_colind[r].clone();
        let lt = self.lt[r].clone();
        let mut rows: Vec<Vec<f64>> = (0..ns).map(|s| vec![0.0; l_nnz - s]).collect();
        let mut used_dense = false;
        for s in 0..ns {
            let row = r - s;
            let nnz_s = l_nnz - s;
            let (hc, hv) = (&h[row].0, &h[row].1);
            if hc.len() >= nnz_s && hc[nnz_s - 1] == row {
                rows[s][..nnz_s].copy_from_slice(&hv[..nnz_s]);
            } else if ns == 1 {
                for i in 0..nnz_s {
                    rows[s][i] = 0.0;
                }
                for (&c, &v) in hc.iter().zip(hv) {
                    if c > row {
                        break;
                    }
                    dense[c] = v;
                }
                used_dense = true;
            } else {
                for (&c, &v) in hc.iter().zip(hv) {
                    if c > row {
                        break;
                    }
                    dense[c] = v;
                }
                for i in 0..nnz_s {
                    let c = cols[i];
                    rows[s][i] = dense[c];
                    dense[c] = 0.0;
                }
            }
        }
        let max_batch = if ns == 3 { 3 } else { 4 };
        let mut k = 1usize;
        while k < lt.len() {
            let (c0, idx0) = lt[k];
            let num_cols = idx0 + 1;
            let col0 = self.l_colind[c0][0];
            let is_suffix = cols[l_nnz - num_cols] == col0;
            let mut batch = 1usize;
            let mut cs = [c0, 0, 0, 0];
            let mut idxs = [idx0, 0, 0, 0];
            if is_suffix || ns == 1 {
                let mut prev_c = c0;
                while k + batch < lt.len() && batch < max_batch {
                    let (next_c, next_idx) = lt[k + batch];
                    if next_idx + 1 != num_cols {
                        break;
                    }
                    if is_suffix {
                        if self.l_colind[next_c][0] != col0 {
                            break;
                        }
                    } else {
                        let (hi, lo) = if next_c > prev_c { (next_c, prev_c) } else { (prev_c, next_c) };
                        let lo_nnz = self.l_colind[lo].len();
                        if lo_nnz >= self.l_colind[hi].len() || self.l_colind[hi][lo_nnz - 1] != lo {
                            break;
                        }
                        prev_c = next_c;
                    }
                    cs[batch] = next_c;
                    idxs[batch] = next_idx;
                    batch += 1;
                }
            }
            if is_suffix {
                let off = l_nnz - num_cols;
                let nlead = num_cols - (ns - 1);
                let mut scale = [[0.0f64; 4]; 3];
                for t in 0..ns {
                    for b in 0..batch {
                        scale[t][b] = self.l[cs[b]][idxs[b] - t];
                    }
                }
                for i in 0..nlead {
                    let mut v = [0.0f64; 4];
                    for b in 0..batch {
                        v[b] = self.l[cs[b]][i];
                    }
                    for b in 0..batch {
                        rows[0][off + i] -= scale[0][b] * v[b];
                    }
                    if ns >= 2 {
                        for b in 0..batch {
                            rows[1][off + i] -= scale[1][b] * v[b];
                        }
                    }
                    if ns >= 3 {
                        for b in 0..batch {
                            rows[2][off + i] -= scale[2][b] * v[b];
                        }
                    }
                }
                for u in 0..ns.saturating_sub(1) {
                    let idx = nlead + u;
                    let mut v = [0.0f64; 4];
                    for b in 0..batch {
                        v[b] = self.l[cs[b]][idx];
                    }
                    for b in 0..batch {
                        rows[0][off + idx] -= scale[0][b] * v[b];
                    }
                    if u == 0 && ns >= 3 {
                        for b in 0..batch {
                            rows[1][off + idx] -= scale[1][b] * v[b];
                        }
                    }
                }
            } else {
                used_dense = true;
                let mut scale = [0.0f64; 4];
                for b in 0..batch {
                    scale[b] = self.l[cs[b]][idxs[b]];
                }
                let colptr = self.l_colind[c0].clone();
                for i in 0..num_cols {
                    let col = colptr[i];
                    let mut acc = dense[col] - scale[0] * self.l[cs[0]][i];
                    if batch >= 2 {
                        acc -= scale[1] * self.l[cs[1]][i];
                    }
                    if batch >= 3 {
                        acc -= scale[2] * self.l[cs[2]][i];
                    }
                    if batch >= 4 {
                        acc -= scale[3] * self.l[cs[3]][i];
                    }
                    dense[col] = acc;
                }
            }
            k += batch;
        }
        if used_dense {
            for s in 0..ns {
                if s > 0 {
                    for j in 1..lt.len() {
                        let (c0, idx0) = lt[j];
                        let num_cols = idx0 + 1;
                        if cols[l_nnz - num_cols] != self.l_colind[c0][0] {
                            let scale = self.l[c0][idx0 - s];
                            for i in 0..num_cols - s {
                                let col = self.l_colind[c0][i];
                                dense[col] -= scale * self.l[c0][i];
                            }
                        }
                    }
                }
                let nnz_s = l_nnz - s;
                for i in 0..nnz_s {
                    let c = cols[i];
                    rows[s][i] += dense[c];
                    dense[c] = 0.0;
                }
            }
        }
        let mut rank_dec = 0usize;
        let n_off = l_nnz - ns;
        for s in 0..ns {
            let nnz_s = l_nnz - s;
            let mut diag = rows[s][nnz_s - 1];
            let deficient = !(diag >= mindiag);
            if deficient {
                diag = mindiag;
                rank_dec += 1;
            }
            let lrr = diag.sqrt();
            if deficient {
                for i in 0..nnz_s - 1 {
                    rows[s][i] = 0.0;
                }
            } else {
                let inv = 1.0 / lrr;
                let mut s_st = [0.0f64; 3];
                for t in (s + 1)..ns {
                    rows[s][l_nnz - 1 - t] *= inv;
                    s_st[t] = rows[s][l_nnz - 1 - t];
                }
                for i in 0..n_off {
                    rows[s][i] *= inv;
                    let l_si = rows[s][i];
                    if s + 1 < ns {
                        rows[s + 1][i] -= s_st[s + 1] * l_si;
                    }
                    if s + 2 < ns {
                        rows[s + 2][i] -= s_st[s + 2] * l_si;
                    }
                }
                for t in (s + 1)..ns {
                    for u in t..ns {
                        rows[t][l_nnz - 1 - u] -= s_st[t] * s_st[u];
                    }
                }
            }
            rows[s][nnz_s - 1] = lrr;
        }
        for s in 0..ns {
            let nnz_s = l_nnz - s;
            self.l[r - s][..nnz_s].copy_from_slice(&rows[s][..nnz_s]);
        }
        rank_dec
    }

    /// `5491d70f` `mju_cholSolveSparse`: a detected 2- or 3-row supernode is one substitution.
    fn solve_super(&self, vec: &[f64]) -> Vec<f64> {
        let n = vec.len();
        let mut res = vec.to_vec();
        let mut i = n as isize - 1;
        while i >= 0 {
            let iu = i as usize;
            let nnz = self.l[iu].len();
            if iu > 0 && nnz >= 2 && self.l[iu - 1].len() == nnz - 1 && self.l_colind[iu][nnz - 2] == iu - 1 {
                let row0 = self.l[iu].clone();
                let row1 = self.l[iu - 1].clone();
                let ci = self.l_colind[iu].clone();
                if iu > 1 && nnz >= 3 && self.l[iu - 2].len() == nnz - 2 && ci[nnz - 3] == iu - 2 {
                    let nob = nnz - 3;
                    let row2 = self.l[iu - 2].clone();
                    let x0 = {
                        res[iu] /= row0[nob + 2];
                        res[iu]
                    };
                    let x1 = {
                        res[iu - 1] = (res[iu - 1] - row0[nob + 1] * x0) / row1[nob + 1];
                        res[iu - 1]
                    };
                    let x2 = {
                        res[iu - 2] = ((res[iu - 2] - row0[nob] * x0) - row1[nob] * x1) / row2[nob];
                        res[iu - 2]
                    };
                    if x0 != 0.0 || x1 != 0.0 || x2 != 0.0 {
                        for j in 0..nob {
                            let c = ci[j];
                            res[c] = ((res[c] - row0[j] * x0) - row1[j] * x1) - row2[j] * x2;
                        }
                    }
                    i -= 3;
                    continue;
                }
                let nob = nnz - 2;
                let x0 = {
                    res[iu] /= row0[nob + 1];
                    res[iu]
                };
                let x1 = {
                    res[iu - 1] = (res[iu - 1] - row0[nob] * x0) / row1[nob];
                    res[iu - 1]
                };
                if x0 != 0.0 || x1 != 0.0 {
                    for j in 0..nob {
                        let c = ci[j];
                        res[c] = (res[c] - row0[j] * x0) - row1[j] * x1;
                    }
                }
                i -= 2;
                continue;
            }
            if res[iu] != 0.0 {
                let tmp = {
                    res[iu] /= self.l[iu][nnz - 1];
                    res[iu]
                };
                for j in 0..nnz - 1 {
                    let c = self.l_colind[iu][j];
                    res[c] -= self.l[iu][j] * tmp;
                }
            }
            i -= 1;
        }
        let mut i = 0usize;
        while i < n {
            let nnz = self.l[i].len();
            if i + 1 < n && self.l[i + 1].len() == nnz + 1 && self.l_colind[i + 1][nnz - 1] == i {
                let nob = nnz - 1;
                let row0 = &self.l[i];
                let row1 = &self.l[i + 1];
                let ci = &self.l_colind[i];
                if i + 2 < n && self.l[i + 2].len() == nnz + 2 && self.l_colind[i + 2][nnz - 1] == i {
                    let row2 = &self.l[i + 2];
                    let mut s0 = 0.0;
                    let mut s1 = 0.0;
                    let mut s2 = 0.0;
                    for j in 0..nob {
                        let y = res[ci[j]];
                        s0 += row0[j] * y;
                        s1 += row1[j] * y;
                        s2 += row2[j] * y;
                    }
                    res[i] = (res[i] - s0) / row0[nob];
                    res[i + 1] = ((res[i + 1] - s1) - row1[nob] * res[i]) / row1[nob + 1];
                    res[i + 2] = (((res[i + 2] - s2) - row2[nob] * res[i]) - row2[nob + 1] * res[i + 1]) / row2[nob + 2];
                    i += 3;
                    continue;
                }
                let mut s0 = 0.0;
                let mut s1 = 0.0;
                for j in 0..nob {
                    let y = res[ci[j]];
                    s0 += row0[j] * y;
                    s1 += row1[j] * y;
                }
                res[i] = (res[i] - s0) / row0[nob];
                res[i + 1] = ((res[i + 1] - s1) - row1[nob] * res[i]) / row1[nob + 1];
                i += 2;
                continue;
            }
            if nnz > 1 {
                let dot = chol_solve_dot(&self.l[i][..nnz - 1], &res, &self.l_colind[i][..nnz - 1]);
                res[i] -= dot;
            }
            res[i] /= self.l[i][nnz - 1];
            i += 1;
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
            let xx = xr * xr;
            let mut tmp = if oracle_contract_off() {
                let ll = diag * diag;
                // plus: `x x + diag diag`. minus: `(-x x) + diag diag`.
                if plus { xx + ll } else { -xx + ll }
            } else {
                diag.mul_add(diag, if plus { xx } else { -xx })
            };
            if tmp < 1e-15 {
                tmp = 1e-15;
                rank -= 1;
            }
            let r = tmp.sqrt();
            self.l[row][nnz] = r;
            // `(c, s) = (diag, -x) / (r, r)`, then `s_signed = -s` on an update.
            let c = diag / r;
            let s = -xr / r;
            let ss = if plus { -s } else { s };
            for i in 0..nnz {
                let j = self.l_colind[row][i];
                let (dj, mv) = (dense[j], self.l[row][i]);
                if oracle_contract_off() {
                    self.l[row][i] = ss * dj + c * mv;
                    dense[j] = s * mv + c * dj;
                } else {
                    self.l[row][i] = c.mul_add(mv, ss * dj);
                    dense[j] = s.mul_add(mv, c * dj);
                }
            }
        }
        rank
    }
}
