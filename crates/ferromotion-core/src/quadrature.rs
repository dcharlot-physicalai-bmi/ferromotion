//! **Gauss–Legendre quadrature** — high-accuracy definite integration `∫ₐᵇ f(x) dx`. An `n`-point
//! Gauss–Legendre rule samples the integrand at the roots of the degree-`n` Legendre polynomial and is
//! **exact for polynomials up to degree `2n − 1`** — dramatically more accurate per evaluation than a
//! uniform (trapezoid/Simpson) rule for smooth integrands. Its uses in physical AI: trajectory cost / effort
//! integrals in optimal control, work and impulse computations, moments of inertia over a body, and finite-
//! element assembly. This is distinct from the crate's [`crate::integrate`] (adaptive DOPRI5, which solves
//! *ODEs*); this evaluates *definite integrals* of a known function.
//!
//! The nodes and weights are computed for any order by the **Golub–Welsch** algorithm — the eigenvalues of
//! the symmetric tridiagonal Jacobi matrix are the nodes, and the weights come from the first component of
//! each eigenvector — so no hard-coded tables. Verified: an `n`-point rule integrates polynomials up to
//! degree `2n−1` to machine precision, reproduces closed-form integrals of transcendental functions, and
//! converges as `n` grows. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, SymmetricEigen};

/// Gauss–Legendre nodes and weights on `[−1, 1]` for an `n`-point rule (Golub–Welsch).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }
    // symmetric tridiagonal Jacobi matrix: diagonal 0, off-diagonal βₖ = k/√(4k²−1)
    let mut j = DMatrix::zeros(n, n);
    for k in 1..n {
        let beta = k as f64 / (4.0 * (k * k) as f64 - 1.0).sqrt();
        j[(k, k - 1)] = beta;
        j[(k - 1, k)] = beta;
    }
    let eig = SymmetricEigen::new(j);
    // nodes = eigenvalues; weights = 2·(first component of the normalized eigenvector)²
    let mut nw: Vec<(f64, f64)> = (0..n).map(|i| (eig.eigenvalues[i], 2.0 * eig.eigenvectors[(0, i)].powi(2))).collect();
    nw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (nw.iter().map(|x| x.0).collect(), nw.iter().map(|x| x.1).collect())
}

/// Integrate `f` over `[a, b]` with an `n`-point Gauss–Legendre rule.
pub fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = gauss_legendre(n);
    let (half, mid) = ((b - a) / 2.0, (a + b) / 2.0);
    half * nodes.iter().zip(&weights).map(|(&x, &w)| w * f(half * x + mid)).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_n_point_rule_is_exact_up_to_degree_2n_minus_1() {
        // THE ORACLE. A 3-point rule integrates any polynomial of degree ≤ 5 exactly. ∫₋₁¹ xᵏ dx = 0 for odd
        // k, 2/(k+1) for even k.
        for k in 0..=5u32 {
            let exact = if k % 2 == 1 { 0.0 } else { 2.0 / (k as f64 + 1.0) };
            let got = integrate(|x| x.powi(k as i32), -1.0, 1.0, 3);
            assert!((got - exact).abs() < 1e-12, "∫x^{k}: {got} vs {exact}");
        }
        // but degree 6 is NOT exact for the 3-point rule (2n−1 = 5)
        let got6 = integrate(|x| x.powi(6), -1.0, 1.0, 3);
        assert!((got6 - 2.0 / 7.0).abs() > 1e-6, "degree 6 should not be exact for a 3-point rule");
    }

    #[test]
    fn it_matches_closed_form_transcendental_integrals() {
        // ∫₀^π sin x dx = 2 ; ∫₀¹ eˣ dx = e − 1.
        assert!((integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 8) - 2.0).abs() < 1e-10, "∫sin");
        assert!((integrate(|x| x.exp(), 0.0, 1.0, 8) - (std::f64::consts::E - 1.0)).abs() < 1e-10, "∫eˣ");
    }

    #[test]
    fn accuracy_improves_with_more_points() {
        // A harder integrand: ∫₀¹ 1/(1+25x²) dx = atan(5)/5. More points ⇒ smaller error.
        let exact = (5.0_f64).atan() / 5.0;
        let e4 = (integrate(|x| 1.0 / (1.0 + 25.0 * x * x), 0.0, 1.0, 4) - exact).abs();
        let e12 = (integrate(|x| 1.0 / (1.0 + 25.0 * x * x), 0.0, 1.0, 12) - exact).abs();
        assert!(e12 < e4, "more points should be more accurate: {e12} vs {e4}");
    }

    #[test]
    fn the_weights_sum_to_the_interval_length() {
        // On [−1,1] the weights sum to 2 (they integrate the constant 1).
        for n in [2, 5, 9] {
            let (_, w) = gauss_legendre(n);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-12, "weights for n={n} should sum to 2");
        }
    }
}
