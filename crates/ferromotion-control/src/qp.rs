//! Shared box-constrained QP backend (`clarabel`, pure Rust → WASM-clean), used by MPC and WBC.
//!
//! **Both solves return `Result`.** They previously returned `solver.solution.x` regardless of the solver's status,
//! which means an infeasible or non-converged problem handed back its last interior-point iterate as though it were the
//! optimum. That is not a degraded answer, it is a plausible wrong one: the same defect measured in
//! [`cbf`](crate::cbf), where an infeasible barrier row returned a control violating it by `10.0` with nothing
//! reported. Every caller in this crate now applies a named fault reaction instead.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
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

/// Why a QP solve produced no usable answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QpFailure {
    /// The solver's own terminal status, verbatim.
    pub status: String,
}

/// Accept only a status that means the returned point is a solution.
fn accept(status: SolverStatus, x: &[f64]) -> Result<Vec<f64>, QpFailure> {
    match status {
        SolverStatus::Solved | SolverStatus::AlmostSolved if x.iter().all(|v| v.is_finite()) => Ok(x.to_vec()),
        other => Err(QpFailure { status: format!("{other:?}") }),
    }
}

/// Solve `min ½ xᵀH·x + gᵀx` subject to general linear inequalities `A·x ≤ b`. `H` symmetric PSD.
pub(crate) fn solve_qp(h: &DMatrix<f64>, g: &[f64], a: &DMatrix<f64>, b: &[f64]) -> Result<Vec<f64>, QpFailure> {
    let p_csc = csc_upper(h);
    let a_csc = csc_dense(a);
    let cones = [SupportedConeT::NonnegativeConeT(a.nrows())];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().map_err(|e| QpFailure { status: format!("settings: {e:?}") })?;
    let mut solver = DefaultSolver::new(&p_csc, g, &a_csc, b, &cones, settings).map_err(|e| QpFailure { status: format!("setup: {e:?}") })?;
    solver.solve();
    accept(solver.solution.status, &solver.solution.x)
}

/// Solve `min ½ xᵀH·x + gᵀx` subject to `lo ≤ x ≤ hi` (elementwise). `H` must be symmetric PSD.
pub(crate) fn solve_box_qp(h: &DMatrix<f64>, g: &[f64], lo: &[f64], hi: &[f64]) -> Result<Vec<f64>, QpFailure> {
    let n = h.ncols();
    let p_csc = csc_upper(h);
    let a_csc = csc_box(n);
    let mut b = hi.to_vec();
    b.extend(lo.iter().map(|v| -v)); // [I;−I]·x ≤ [hi; −lo]
    let cones = [SupportedConeT::NonnegativeConeT(2 * n)];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().map_err(|e| QpFailure { status: format!("settings: {e:?}") })?;
    let mut solver = DefaultSolver::new(&p_csc, g, &a_csc, &b, &cones, settings).map_err(|e| QpFailure { status: format!("setup: {e:?}") })?;
    solver.solve();
    accept(solver.solution.status, &solver.solution.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An infeasible box must be refused, not answered. `hi < lo` has no solution.
    #[test]
    fn an_infeasible_box_is_refused() {
        let h = DMatrix::identity(2, 2);
        let err = solve_box_qp(&h, &[0.0, 0.0], &[1.0, 1.0], &[-1.0, -1.0]).expect_err("hi < lo has no solution");
        eprintln!("infeasible box -> {err:?}");
        assert!(!err.status.is_empty());
    }

    /// Opposing halfspaces must be refused by the general solve.
    #[test]
    fn opposing_halfspaces_are_refused() {
        let h = DMatrix::identity(1, 1);
        let a = DMatrix::from_row_slice(2, 1, &[1.0, -1.0]);
        let err = solve_qp(&h, &[0.0], &a, &[-1.0, -1.0]).expect_err("u <= -1 and -u <= -1 cannot both hold");
        eprintln!("opposing halfspaces -> {err:?}");
        assert!(!err.status.is_empty());
    }

    /// A well-posed problem still returns the right answer, so the status gate costs nothing.
    #[test]
    fn a_feasible_problem_is_unaffected() {
        let h = DMatrix::identity(2, 2);
        // min 1/2|x|^2 - [1,2]'x subject to -0.5 <= x <= 0.5 -> both clamp at the upper bound
        let x = solve_box_qp(&h, &[-1.0, -2.0], &[-0.5, -0.5], &[0.5, 0.5]).expect("feasible");
        eprintln!("clamped solution: {x:?}");
        assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6);
    }
}
