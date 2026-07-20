//! **Polynomial root-finding** via the **companion matrix**. The roots of a polynomial
//! `a₀ + a₁x + … + aₙxⁿ` are exactly the eigenvalues of its companion (Frobenius) matrix — so a robust
//! general eigensolver gives all `n` roots (real *and* complex) at once, with none of the deflation /
//! starting-guess fragility of iterative root-finders. This is a small but foundational numerical primitive:
//! closed-form inverse kinematics reduce to polynomials, time-optimal trajectory timing and time-of-arrival
//! solves hit quartics, and characteristic/stability equations need their roots.
//!
//! Verified: it recovers the exact roots of polynomials built from known real and complex-conjugate roots,
//! and agrees with the closed-form quadratic formula. Pure `nalgebra` → WASM-clean.

use nalgebra::{Complex, DMatrix};

/// All roots of the polynomial with coefficients `coeffs` in **ascending** order (`coeffs[i]` multiplies
/// `xⁱ`), as complex numbers. Leading (and trailing) zero coefficients are handled. Returns an empty vector
/// for a constant polynomial.
pub fn roots(coeffs: &[f64]) -> Vec<Complex<f64>> {
    // strip leading zeros from the top (highest-degree) end
    let mut c = coeffs.to_vec();
    while c.len() > 1 && c.last().unwrap().abs() < 1e-14 {
        c.pop();
    }
    if c.len() <= 1 {
        return Vec::new(); // constant ⇒ no roots
    }
    // factor out roots at zero (trailing low-order zeros) to keep the companion well-conditioned
    let mut zero_roots = 0;
    while c.len() > 1 && c[0].abs() < 1e-14 {
        c.remove(0);
        zero_roots += 1;
    }
    let n = c.len() - 1; // degree of the reduced polynomial
    let lead = *c.last().unwrap();
    // monic normalized coefficients c₀..c_{n-1} (of the reduced polynomial)
    let mono: Vec<f64> = c[..n].iter().map(|v| v / lead).collect();

    let mut out = Vec::new();
    if n >= 1 {
        // companion matrix (roots = eigenvalues): subdiagonal ones, last column = −monic coeffs
        let mut comp = DMatrix::<f64>::zeros(n, n);
        for i in 1..n {
            comp[(i, i - 1)] = 1.0;
        }
        for (i, &m) in mono.iter().enumerate() {
            comp[(i, n - 1)] = -m;
        }
        for ev in comp.complex_eigenvalues().iter() {
            out.push(*ev);
        }
    }
    for _ in 0..zero_roots {
        out.push(Complex::new(0.0, 0.0));
    }
    out
}

/// The real roots only (imaginary part within `tol`), sorted ascending.
pub fn real_roots(coeffs: &[f64], tol: f64) -> Vec<f64> {
    let mut r: Vec<f64> = roots(coeffs).into_iter().filter(|z| z.im.abs() < tol).map(|z| z.re).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    // build ascending coefficients of ∏(x − rᵢ) for real roots
    fn from_real_roots(roots: &[f64]) -> Vec<f64> {
        let mut c = vec![1.0]; // start with the constant polynomial 1
        for &r in roots {
            // multiply by (x − r): new_c[i] = c[i-1] − r·c[i]
            let mut nc = vec![0.0; c.len() + 1];
            for (i, &ci) in c.iter().enumerate() {
                nc[i] -= r * ci;
                nc[i + 1] += ci;
            }
            c = nc;
        }
        c
    }

    #[test]
    fn it_recovers_known_real_roots() {
        // THE ORACLE. Roots of (x−1)(x−2)(x−3).
        let coeffs = from_real_roots(&[1.0, 2.0, 3.0]);
        let r = real_roots(&coeffs, 1e-8);
        assert_eq!(r.len(), 3, "should find 3 real roots");
        for (got, exp) in r.iter().zip([1.0, 2.0, 3.0]) {
            assert!((got - exp).abs() < 1e-9, "root {got} vs {exp}");
        }
    }

    #[test]
    fn it_recovers_complex_conjugate_roots() {
        // (x² + 1)(x − 2) = x³ − 2x² + x − 2 ⇒ roots {2, i, −i}.
        let coeffs = [-2.0, 1.0, -2.0, 1.0]; // ascending
        let all = roots(&coeffs);
        assert_eq!(all.len(), 3);
        // one real root at 2
        assert!(all.iter().any(|z| (z.re - 2.0).abs() < 1e-9 && z.im.abs() < 1e-9), "real root 2");
        // a conjugate pair at ±i
        assert!(all.iter().any(|z| z.re.abs() < 1e-9 && (z.im - 1.0).abs() < 1e-9), "root +i");
        assert!(all.iter().any(|z| z.re.abs() < 1e-9 && (z.im + 1.0).abs() < 1e-9), "root −i");
    }

    #[test]
    fn it_agrees_with_the_quadratic_formula() {
        // 2x² − 4x − 6 = 0 ⇒ x = (4 ± √(16+48))/4 = {3, −1}.
        let r = real_roots(&[-6.0, -4.0, 2.0], 1e-9);
        assert!((r[0] + 1.0).abs() < 1e-9 && (r[1] - 3.0).abs() < 1e-9, "quadratic roots {r:?}");
    }

    #[test]
    fn it_handles_roots_at_zero_and_leading_zeros() {
        // x³ − x = x(x−1)(x+1) ⇒ roots {−1, 0, 1}; with a spurious leading zero coefficient.
        let r = real_roots(&[0.0, -1.0, 0.0, 1.0, 0.0], 1e-9); // (0)x⁴ + x³ + 0x² − x + 0
        assert_eq!(r.len(), 3, "three real roots, got {r:?}");
        assert!((r[0] + 1.0).abs() < 1e-9 && r[1].abs() < 1e-9 && (r[2] - 1.0).abs() < 1e-9, "{r:?}");
    }
}
