//! **Lyapunov / LMI certificates** — the wasm-clean stability-certification substrate. The frontier's
//! certified-control methods (control contraction metrics on their LTI reduction, LQR-tree regions of
//! attraction) all bottom out on a quadratic Lyapunov certificate: a `P ≻ 0` proving `AᵀP + PA ≺ 0`
//! (continuous) or `AᵀPA − P ≺ 0` (discrete). For the quadratic case this needs **no SDP solver** — the
//! Lyapunov *equation* `AᵀP + PA = −Q` has a unique solution via a Kronecker linear solve, and checking
//! that solution is positive-definite *is* the stability certificate (Lyapunov's theorem). That keeps the
//! whole thing pure `nalgebra`, WASM-clean — no BLAS/LAPACK.
//!
//! (General polynomial SOS and inequality-LMI feasibility over the PSD cone need a semidefinite solver;
//! clarabel's PSD cone requires BLAS and so is *not* wasm-clean — a pure-Rust PSD interior point is the
//! planned follow-up. The quadratic Lyapunov certificate below covers the LTI reductions the roadmap's
//! CCM / LQR-tree items depend on.)

use nalgebra::{DMatrix, DVector};

/// Column-major `vec(M)` (stack the columns).
fn vecm(m: &DMatrix<f64>) -> DVector<f64> {
    DVector::from_column_slice(m.as_slice())
}

/// Inverse of [`vecm`]: reshape a length-`n²` vector into an `n×n` matrix (column-major).
fn unvec(v: &DVector<f64>, n: usize) -> DMatrix<f64> {
    DMatrix::from_column_slice(n, n, v.as_slice())
}

fn is_pd(p: &DMatrix<f64>, tol: f64) -> bool {
    p.clone().symmetric_eigen().eigenvalues.iter().all(|&e| e > tol)
}

/// Solve the **continuous Lyapunov equation** `AᵀP + PA = −Q` for symmetric `P` (unique when `A` and `−A`
/// share no eigenvalues), via the Kronecker linear system `(I⊗Aᵀ + Aᵀ⊗I) vec(P) = −vec(Q)`.
pub fn solve_lyapunov(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let at = a.transpose();
    let id = DMatrix::<f64>::identity(n, n);
    let k = id.kronecker(&at) + at.kronecker(&id); // I⊗Aᵀ + Aᵀ⊗I
    let rhs = -vecm(q);
    let p = k.lu().solve(&rhs)?;
    let mut p = unvec(&p, n);
    p = (&p + p.transpose()) * 0.5; // symmetrize away round-off
    Some(p)
}

/// A **continuous Lyapunov certificate**: return `P ≻ 0` with `AᵀP + PA = −I` if `A` is Hurwitz (all
/// eigenvalues in the open left half-plane), else `None`. The positive-definiteness of `P` is the proof.
pub fn lyapunov(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let p = solve_lyapunov(a, &DMatrix::identity(n, n))?;
    if is_pd(&p, 1e-9) { Some(p) } else { None }
}

/// True iff `A` is Hurwitz (continuous-time stable), certified by [`lyapunov`].
pub fn is_hurwitz(a: &DMatrix<f64>) -> bool {
    lyapunov(a).is_some()
}

/// Solve the **discrete Lyapunov (Stein) equation** `AᵀPA − P = −Q` via `(Aᵀ⊗Aᵀ − I) vec(P) = −vec(Q)`.
pub fn solve_lyapunov_discrete(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let at = a.transpose();
    let k = at.kronecker(&at) - DMatrix::<f64>::identity(n * n, n * n);
    let rhs = -vecm(q);
    let p = k.lu().solve(&rhs)?;
    let mut p = unvec(&p, n);
    p = (&p + p.transpose()) * 0.5;
    Some(p)
}

/// A **discrete Lyapunov certificate**: `P ≻ 0` with `AᵀPA − P = −I` iff `A` is Schur-stable (spectral
/// radius < 1), else `None`.
pub fn lyapunov_discrete(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = a.nrows();
    let p = solve_lyapunov_discrete(a, &DMatrix::identity(n, n))?;
    if is_pd(&p, 1e-9) { Some(p) } else { None }
}

/// True iff `A` is Schur-stable (discrete-time stable), certified by [`lyapunov_discrete`].
pub fn is_schur(a: &DMatrix<f64>) -> bool {
    lyapunov_discrete(a).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_continuous_certificate_proves_a_stable_system() {
        // THE INVARIANT. For a Hurwitz A, P ≻ 0 and AᵀP + PA = −I exactly.
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, -2.0]);
        let p = lyapunov(&a).expect("Hurwitz ⇒ certificate");
        assert!(is_pd(&p, 1e-9), "P must be positive-definite");
        let res = &a.transpose() * &p + &p * &a + DMatrix::<f64>::identity(2, 2);
        assert!(res.abs().max() < 1e-9, "AᵀP+PA should equal −I: residual {}", res.abs().max());
    }

    #[test]
    fn the_continuous_certificate_rejects_an_unstable_system() {
        // A has an eigenvalue in the right half-plane ⇒ the Lyapunov solution is not PD.
        let a = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, -2.0]);
        assert!(lyapunov(&a).is_none(), "unstable ⇒ no certificate");
        assert!(!is_hurwitz(&a));
    }

    #[test]
    fn it_certifies_a_marginally_stable_spiral() {
        // a stable focus (complex eigenvalues with negative real part)
        let a = DMatrix::from_row_slice(2, 2, &[-0.2, 1.0, -1.0, -0.2]);
        let p = lyapunov(&a).expect("stable focus ⇒ certificate");
        assert!(is_pd(&p, 1e-9));
        assert!(is_hurwitz(&a));
    }

    #[test]
    fn the_discrete_certificate_matches_schur_stability() {
        // spectral radius < 1 ⇒ certificate with AᵀPA − P = −I; ≥ 1 ⇒ none.
        let stable = DMatrix::from_row_slice(2, 2, &[0.5, 0.2, 0.0, -0.3]);
        let p = lyapunov_discrete(&stable).expect("Schur ⇒ certificate");
        assert!(is_pd(&p, 1e-9));
        let res = &stable.transpose() * &p * &stable - &p + DMatrix::<f64>::identity(2, 2);
        assert!(res.abs().max() < 1e-9, "AᵀPA−P should equal −I: {}", res.abs().max());
        assert!(is_schur(&stable));

        let unstable = DMatrix::from_row_slice(2, 2, &[1.1, 0.0, 0.0, 0.4]); // ρ = 1.1
        assert!(lyapunov_discrete(&unstable).is_none(), "spectral radius ≥ 1 ⇒ no certificate");
        assert!(!is_schur(&unstable));
    }

    #[test]
    fn the_lyapunov_solution_is_symmetric_and_solves_the_equation() {
        let a = DMatrix::from_row_slice(3, 3, &[-2.0, 1.0, 0.0, 0.0, -1.0, 0.5, 0.0, 0.0, -3.0]);
        let q = DMatrix::from_row_slice(3, 3, &[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]);
        let p = solve_lyapunov(&a, &q).unwrap();
        assert!((&p - p.transpose()).abs().max() < 1e-10, "P symmetric");
        let res = &a.transpose() * &p + &p * &a + &q;
        assert!(res.abs().max() < 1e-9, "AᵀP+PA+Q should vanish: {}", res.abs().max());
    }
}
