//! **Differentiable QP** (Amos & Kolter, *OptNet*, ICML 2017) — differentiate the *solution* of an
//! equality-constrained quadratic program with respect to its data, by the implicit function theorem on
//! the KKT conditions. This is the substrate that turns a QP/iLQR/MPC solver into a **trainable layer**:
//! MPC and iLQR reduce to a QP, so `∂(optimal action)/∂(cost, dynamics)` follows from the QP's KKT adjoint
//! — one extra linear solve, independent of how many solver iterations produced the forward pass.
//!
//! For `min_z ½ zᵀQ z + qᵀz  s.t.  A z = b`, the KKT system is `[[Q, Aᵀ],[A, 0]]·[z; ν] = [−q; b]`.
//! Given a downstream loss's cotangent `∂L/∂z*`, the **adjoint** solves the same (symmetric) KKT matrix
//! against `[−∂L/∂z*; 0]`, and the parameter gradients drop out in closed form. Verified against finite
//! differences. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// The solution of the equality-constrained QP: the primal `z` and the multipliers `nu`.
#[derive(Clone, Debug)]
pub struct QpSolution {
    pub z: DVector<f64>,
    pub nu: DVector<f64>,
}

/// Gradients of a scalar loss w.r.t. the QP data (from one adjoint solve).
#[derive(Clone, Debug)]
pub struct QpGrads {
    pub d_q_mat: DMatrix<f64>, // ∂L/∂Q (symmetric)
    pub d_q_vec: DVector<f64>, // ∂L/∂q
    pub d_a: DMatrix<f64>,     // ∂L/∂A
    pub d_b: DVector<f64>,     // ∂L/∂b
}

fn kkt(q: &DMatrix<f64>, a: &DMatrix<f64>) -> DMatrix<f64> {
    let (n, m) = (q.nrows(), a.nrows());
    let mut k = DMatrix::zeros(n + m, n + m);
    k.view_mut((0, 0), (n, n)).copy_from(q);
    k.view_mut((0, n), (n, m)).copy_from(&a.transpose());
    k.view_mut((n, 0), (m, n)).copy_from(a);
    k
}

/// Solve `min_z ½ zᵀQ z + qᵀz  s.t.  A z = b` via its KKT system.
pub fn solve_eq_qp(q: &DMatrix<f64>, qv: &DVector<f64>, a: &DMatrix<f64>, b: &DVector<f64>) -> QpSolution {
    let (n, m) = (q.nrows(), a.nrows());
    let k = kkt(q, a);
    let mut rhs = DVector::zeros(n + m);
    rhs.rows_mut(0, n).copy_from(&(-qv));
    rhs.rows_mut(n, m).copy_from(b);
    let sol = k.lu().solve(&rhs).expect("KKT system is nonsingular for a well-posed QP");
    QpSolution { z: sol.rows(0, n).into_owned(), nu: sol.rows(n, m).into_owned() }
}

/// Backpropagate a loss cotangent `dl_dz = ∂L/∂z*` through the QP: returns `∂L/∂{Q, q, A, b}` via the KKT
/// adjoint (OptNet). One linear solve on the same KKT matrix.
pub fn diff_eq_qp(q: &DMatrix<f64>, a: &DMatrix<f64>, sol: &QpSolution, dl_dz: &DVector<f64>) -> QpGrads {
    let (n, m) = (q.nrows(), a.nrows());
    let k = kkt(q, a);
    // adjoint: [[Q,Aᵀ],[A,0]]·[dz; dnu] = [−dl_dz; 0]
    let mut rhs = DVector::zeros(n + m);
    rhs.rows_mut(0, n).copy_from(&(-dl_dz));
    let adj = k.lu().solve(&rhs).expect("KKT adjoint solve");
    let dz = adj.rows(0, n).into_owned();
    let dnu = adj.rows(n, m).into_owned();

    // OptNet closed forms:
    let d_q_vec = dz.clone();
    let d_b = -&dnu;
    let dq_raw = &dz * sol.z.transpose();
    let d_q_mat = (&dq_raw + dq_raw.transpose()) * 0.5;
    let d_a = &dnu * sol.z.transpose() + &sol.nu * dz.transpose();
    QpGrads { d_q_mat, d_q_vec, d_a, d_b }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    // A small, well-posed equality-constrained QP.
    fn problem() -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>) {
        let q = DMatrix::from_row_slice(3, 3, &[2.0, 0.2, 0.0, 0.2, 3.0, -0.4, 0.0, -0.4, 1.5]);
        let qv = dv(&[-1.0, 0.5, 0.3]);
        let a = DMatrix::from_row_slice(1, 3, &[1.0, 1.0, 1.0]); // sum(z) = 2
        let b = dv(&[2.0]);
        (q, qv, a, b)
    }

    #[test]
    fn the_qp_solution_satisfies_the_kkt_conditions() {
        let (q, qv, a, b) = problem();
        let s = solve_eq_qp(&q, &qv, &a, &b);
        // stationarity: Qz + q + Aᵀν = 0 ; feasibility: Az = b
        let stat = &q * &s.z + &qv + a.transpose() * &s.nu;
        assert!(stat.abs().max() < 1e-10, "stationarity residual {}", stat.abs().max());
        assert!((&a * &s.z - &b).abs().max() < 1e-10, "constraint residual");
    }

    #[test]
    fn the_gradients_match_finite_differences() {
        // THE ORACLE. For a linear loss L = cᵀz*, the OptNet gradients w.r.t. q, b, Q, A must match central
        // finite differences of a re-solve.
        let (q, qv, a, b) = problem();
        let c = dv(&[0.7, -0.4, 1.1]); // ∂L/∂z*
        let s = solve_eq_qp(&q, &qv, &a, &b);
        let g = diff_eq_qp(&q, &a, &s, &c);
        let loss = |qq: &DMatrix<f64>, qvv: &DVector<f64>, aa: &DMatrix<f64>, bb: &DVector<f64>| c.dot(&solve_eq_qp(qq, qvv, aa, bb).z);
        let eps = 1e-6;

        // ∂L/∂q
        for i in 0..3 {
            let (mut qp, mut qm) = (qv.clone(), qv.clone());
            qp[i] += eps;
            qm[i] -= eps;
            let fd = (loss(&q, &qp, &a, &b) - loss(&q, &qm, &a, &b)) / (2.0 * eps);
            assert!((g.d_q_vec[i] - fd).abs() < 1e-6, "∂L/∂q[{i}] {} vs fd {fd}", g.d_q_vec[i]);
        }
        // ∂L/∂b
        {
            let (mut bp, mut bm) = (b.clone(), b.clone());
            bp[0] += eps;
            bm[0] -= eps;
            let fd = (loss(&q, &qv, &a, &bp) - loss(&q, &qv, &a, &bm)) / (2.0 * eps);
            assert!((g.d_b[0] - fd).abs() < 1e-6, "∂L/∂b {} vs fd {fd}", g.d_b[0]);
        }
        // ∂L/∂Q. For i≠j a symmetric perturbation moves both Q_ij and Q_ji (compare to d_Q[i,j]+d_Q[j,i]);
        // for i==j that would be the SAME entry twice, so perturb it once.
        for i in 0..3 {
            for j in i..3 {
                let mut qp = q.clone();
                let mut qm = q.clone();
                qp[(i, j)] += eps;
                qm[(i, j)] -= eps;
                let analytic = if i == j {
                    g.d_q_mat[(i, j)]
                } else {
                    qp[(j, i)] += eps;
                    qm[(j, i)] -= eps;
                    g.d_q_mat[(i, j)] + g.d_q_mat[(j, i)]
                };
                let fd = (loss(&qp, &qv, &a, &b) - loss(&qm, &qv, &a, &b)) / (2.0 * eps);
                assert!((analytic - fd).abs() < 1e-5, "∂L/∂Q[{i},{j}] {analytic} vs fd {fd}");
            }
        }
        // ∂L/∂A
        for j in 0..3 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[(0, j)] += eps;
            am[(0, j)] -= eps;
            let fd = (loss(&q, &qv, &ap, &b) - loss(&q, &qv, &am, &b)) / (2.0 * eps);
            assert!((g.d_a[(0, j)] - fd).abs() < 1e-5, "∂L/∂A[0,{j}] {} vs fd {fd}", g.d_a[(0, j)]);
        }
    }

    #[test]
    fn unconstrained_reduces_to_the_normal_equations_gradient() {
        // With no constraints, z* = −Q⁻¹q and ∂z*/∂q = −Q⁻¹, so for L = cᵀz*, ∂L/∂q = −Q⁻¹c.
        let q = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.0]);
        let qv = dv(&[0.3, -0.7]);
        let a = DMatrix::zeros(0, 2);
        let b = DVector::zeros(0);
        let s = solve_eq_qp(&q, &qv, &a, &b);
        assert!((&s.z + q.clone().try_inverse().unwrap() * &qv).abs().max() < 1e-10, "z* = −Q⁻¹q");
        let c = dv(&[1.0, 2.0]);
        let g = diff_eq_qp(&q, &a, &s, &c);
        let expect = -(q.try_inverse().unwrap() * &c);
        assert!((&g.d_q_vec - &expect).abs().max() < 1e-10, "∂L/∂q should be −Q⁻¹c");
    }
}
