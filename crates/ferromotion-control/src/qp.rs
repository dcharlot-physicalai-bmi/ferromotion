//! Shared box-constrained QP backend (`clarabel`, pure Rust → WASM-clean), used by MPC and WBC.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::DMatrix;

/// Upper-triangular CSC of a dense symmetric matrix (clarabel wants `P` upper-triangular).
fn csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.ncols();
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..=j {
            rowval.push(i);
            nzval.push(p[(i, j)]);
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// `[I; −I]` (2n×n) in CSC — the constraint matrix for elementwise box bounds.
fn csc_box(n: usize) -> CscMatrix<f64> {
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        rowval.push(j);
        nzval.push(1.0);
        rowval.push(n + j);
        nzval.push(-1.0);
        colptr.push(rowval.len());
    }
    CscMatrix::new(2 * n, n, colptr, rowval, nzval)
}

/// Dense `m×n` matrix to CSC (column-major; rows within a column are pushed in order → sorted).
fn csc_dense(a: &DMatrix<f64>) -> CscMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..m {
            let v = a[(i, j)];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

/// Solve `min ½ xᵀH·x + gᵀx` subject to general linear inequalities `A·x ≤ b`. `H` symmetric PSD.
pub(crate) fn solve_qp(h: &DMatrix<f64>, g: &[f64], a: &DMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let p_csc = csc_upper(h);
    let a_csc = csc_dense(a);
    let cones = [SupportedConeT::NonnegativeConeT(a.nrows())];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p_csc, g, &a_csc, b, &cones, settings).unwrap();
    solver.solve();
    solver.solution.x.clone()
}

/// Solve `min ½ xᵀH·x + gᵀx` subject to `lo ≤ x ≤ hi` (elementwise). `H` must be symmetric PSD.
pub(crate) fn solve_box_qp(h: &DMatrix<f64>, g: &[f64], lo: &[f64], hi: &[f64]) -> Vec<f64> {
    let n = h.ncols();
    let p_csc = csc_upper(h);
    let a_csc = csc_box(n);
    let mut b = hi.to_vec();
    b.extend(lo.iter().map(|v| -v)); // [I;−I]·x ≤ [hi; −lo]
    let cones = [SupportedConeT::NonnegativeConeT(2 * n)];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p_csc, g, &a_csc, &b, &cones, settings).unwrap();
    solver.solve();
    solver.solution.x.clone()
}
