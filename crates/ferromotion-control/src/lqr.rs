//! Linear-quadratic regulator — the optimal-control reference point.

use nalgebra::{DMatrix, DVector};

/// Discrete-time infinite-horizon LQR gain `K` (so `u = −K·x` minimizes `Σ xᵀQx + uᵀRu` subject
/// to `xₖ₊₁ = A·xₖ + B·uₖ`). Solves the discrete algebraic Riccati equation by iteration.
pub fn dlqr(a: &DMatrix<f64>, b: &DMatrix<f64>, q: &DMatrix<f64>, r: &DMatrix<f64>) -> DMatrix<f64> {
    let mut p = q.clone();
    for _ in 0..5000 {
        let btp = b.transpose() * &p; // m×n
        let Some(inv) = (r + &btp * b).try_inverse() else { break };
        let atp = a.transpose() * &p; // n×n
        let k = &inv * (&btp * a); // m×n
        let p_next = &atp * a - (&atp * b) * &k + q;
        let converged = (&p_next - &p).norm() < 1e-12;
        p = p_next;
        if converged {
            break;
        }
    }
    let btp = b.transpose() * &p;
    (r + &btp * b).try_inverse().expect("R + BᵀPB singular") * (&btp * a)
}

/// LQR state-feedback controller: `u = −K·x`.
#[derive(Clone, Debug)]
pub struct Lqr {
    pub k: DMatrix<f64>,
}

impl Lqr {
    pub fn discrete(a: &DMatrix<f64>, b: &DMatrix<f64>, q: &DMatrix<f64>, r: &DMatrix<f64>) -> Self {
        Self { k: dlqr(a, b, q, r) }
    }

    pub fn control(&self, x: &[f64]) -> Vec<f64> {
        (&self.k * DVector::from_row_slice(x)).iter().map(|v| -v).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lqr_stabilizes_a_double_integrator() {
        let dt = 0.05;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0]));
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let lqr = Lqr::discrete(&a, &b, &q, &r);

        // Regulate from a displaced state back to the origin.
        let mut x = [1.0, 0.0];
        for _ in 0..400 {
            let u = lqr.control(&x)[0];
            let xn0 = x[0] + dt * x[1] + 0.5 * dt * dt * u;
            let xn1 = x[1] + dt * u;
            x = [xn0, xn1];
        }
        assert!(x[0].abs() < 1e-2 && x[1].abs() < 1e-2, "state did not regulate: {x:?}");
    }
}
