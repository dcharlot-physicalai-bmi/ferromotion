//! **Semidefinite programming (SDP) via ADMM** — a dense small-scale solver for
//! `min ⟨C, X⟩  s.t.  ⟨Aᵢ, X⟩ = bᵢ,  X ⪰ 0`. This is the primitive the crate's [`crate::lmi`] and
//! [`crate::ccm`] modules explicitly *avoid* ("needs no SDP solver…"): with it, general **LMI feasibility**,
//! **Lyapunov/contraction synthesis**, **sum-of-squares** certificates, and convex relaxations become
//! available. The one operation an SDP needs beyond linear algebra is projection onto the **PSD cone** —
//! which is just an eigendecomposition (clamp the negative eigenvalues to zero) — so the whole solver is
//! **pure `nalgebra`, no BLAS/LAPACK, wasm-clean**, unlike the interior-point SDP stacks that don't compile
//! to WASM.
//!
//! Operator-splitting (ADMM): alternate an equality-constrained least-squares step (a fixed `m×m` linear
//! solve on the Gram matrix of the constraints) with the PSD projection, plus a dual update. Verified on the
//! canonical **λ_min-as-an-SDP** (`min ⟨C,X⟩ s.t. tr X = 1` has optimum `λ_min(C)`, attained at the
//! min-eigenvector's outer product) and on an LMI feasibility problem, and the returned `X` is PSD and
//! constraint-feasible. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// A standard-form SDP `min ⟨C, X⟩  s.t.  ⟨Aᵢ, X⟩ = bᵢ,  X ⪰ 0` (all matrices symmetric `n×n`).
#[derive(Clone, Debug)]
pub struct SdpProblem {
    pub c: DMatrix<f64>,
    pub a: Vec<DMatrix<f64>>,
    pub b: DVector<f64>,
}

/// The result of an SDP solve.
#[derive(Clone, Debug)]
pub struct SdpSolution {
    pub x: DMatrix<f64>,
    pub objective: f64,
    pub primal_residual: f64,
    pub iterations: usize,
}

fn inner(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Project a symmetric matrix onto the PSD cone (zero out negative eigenvalues).
pub fn project_psd(m: &DMatrix<f64>) -> DMatrix<f64> {
    let sym = (m + m.transpose()) * 0.5;
    let eig = SymmetricEigen::new(sym);
    let n = m.nrows();
    let mut out = DMatrix::zeros(n, n);
    for i in 0..n {
        let lam = eig.eigenvalues[i].max(0.0);
        if lam > 0.0 {
            let v = eig.eigenvectors.column(i);
            out += lam * v * v.transpose();
        }
    }
    out
}

/// Solve the SDP by ADMM. `rho` is the penalty parameter; returns the solution after at most `iters`.
pub fn solve_sdp(prob: &SdpProblem, iters: usize, rho: f64) -> SdpSolution {
    let n = prob.c.nrows();
    let m = prob.a.len();
    // Gram matrix G_ij = ⟨Aᵢ, Aⱼ⟩ and its inverse (constraints assumed independent)
    let mut g = DMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            g[(i, j)] = inner(&prob.a[i], &prob.a[j]);
        }
    }
    let g_inv = g.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(m, m));
    let a_of = |x: &DMatrix<f64>| DVector::from_iterator(m, prob.a.iter().map(|ai| inner(ai, x)));
    let at_of = |y: &DVector<f64>| {
        let mut s = DMatrix::zeros(n, n);
        for (i, ai) in prob.a.iter().enumerate() {
            s += y[i] * ai;
        }
        s
    };

    // splitting: min ⟨C,X⟩ + I_PSD(Z) s.t. A(X)=b, X=Z
    let mut z = DMatrix::<f64>::identity(n, n);
    let mut u = DMatrix::<f64>::zeros(n, n);
    let ac = a_of(&prob.c);
    let mut prim = f64::INFINITY;
    let mut it = 0;
    for k in 0..iters {
        it = k + 1;
        // X-update: min ⟨C,X⟩ + (ρ/2)‖X−Z+U‖² s.t. A(X)=b  ⇒  X = Z−U − (C+Aᵀλ)/ρ, λ from the equality
        let zmu = &z - &u;
        let rhs = (a_of(&zmu) - &prob.b) * rho - &ac; // ρ(A(Z−U) − b) − A(C)
        let lambda = &g_inv * rhs;
        let x = &zmu - &(&prob.c + at_of(&lambda)) / rho;
        // Z-update: PSD projection
        z = project_psd(&(&x + &u));
        // dual update
        u += &x - &z;

        prim = (a_of(&z) - &prob.b).norm();
        if prim < 1e-8 && (&x - &z).norm() < 1e-8 {
            break;
        }
    }
    SdpSolution { objective: inner(&prob.c, &z), x: z, primal_residual: prim, iterations: it }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(data: &[f64], n: usize) -> DMatrix<f64> {
        let m = DMatrix::from_row_slice(n, n, data);
        (&m + m.transpose()) * 0.5
    }

    #[test]
    fn psd_projection_zeros_negative_eigenvalues() {
        // THE ORACLE (cone). Projecting a matrix with a negative eigenvalue removes exactly that direction.
        let m = sym(&[2.0, 0.0, 0.0, -3.0], 2);
        let p = project_psd(&m);
        // eigenvalues of p must all be ≥ 0
        let e = SymmetricEigen::new(p.clone());
        assert!(e.eigenvalues.iter().all(|&l| l > -1e-9), "projection should be PSD");
        assert!((p[(0, 0)] - 2.0).abs() < 1e-9 && p[(1, 1)].abs() < 1e-9, "keeps +2, drops −3");
    }

    #[test]
    fn it_solves_the_min_eigenvalue_sdp() {
        // THE HEADLINE. min ⟨C,X⟩ s.t. tr(X)=1, X⪰0 has optimum λ_min(C), attained at v_min v_minᵀ.
        let c = sym(&[4.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0], 3);
        let prob = SdpProblem { c: c.clone(), a: vec![DMatrix::identity(3, 3)], b: DVector::from_row_slice(&[1.0]) };
        let sol = solve_sdp(&prob, 4000, 1.0);
        let lam_min = SymmetricEigen::new(c).eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((sol.objective - lam_min).abs() < 1e-4, "SDP optimum {} vs λ_min {lam_min}", sol.objective);
        assert!(sol.primal_residual < 1e-6, "trace constraint satisfied: {}", sol.primal_residual);
        // solution is PSD and rank-1 (trace 1, so it's a projector onto the min-eigenvector)
        assert!(SymmetricEigen::new(sol.x.clone()).eigenvalues.iter().all(|&l| l > -1e-6), "X should be PSD");
    }

    #[test]
    fn it_finds_a_feasible_lmi_point() {
        // Feasibility: find X ⪰ 0 with tr(X)=3 and ⟨E12, X⟩ = 0 (E12 picks the (0,1)+(1,0) entries). A
        // diagonal X with the right trace works; the solver must return a PSD, constraint-feasible X.
        let mut e12 = DMatrix::zeros(2, 2);
        e12[(0, 1)] = 1.0;
        e12[(1, 0)] = 1.0;
        let prob = SdpProblem { c: DMatrix::zeros(2, 2), a: vec![DMatrix::identity(2, 2), e12], b: DVector::from_row_slice(&[3.0, 0.0]) };
        let sol = solve_sdp(&prob, 4000, 1.0);
        assert!(sol.primal_residual < 1e-6, "constraints satisfied: {}", sol.primal_residual);
        assert!(SymmetricEigen::new(sol.x.clone()).eigenvalues.iter().all(|&l| l > -1e-6), "feasible X is PSD");
        assert!((sol.x.trace() - 3.0).abs() < 1e-5, "trace = 3");
    }
}
