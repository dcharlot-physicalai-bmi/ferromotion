//! Linear model-predictive control — the workhorse of modern control. Condensed formulation
//! (states eliminated), solved each step as a QP over the input sequence with box input bounds
//! via `clarabel` (pure Rust → WASM-clean). Receding horizon: solve, apply the first input, repeat.

use crate::qp::solve_box_qp;
use nalgebra::{DMatrix, DVector};

/// Discrete linear MPC: `min Σₖ xₖᵀQxₖ + uₖᵀRuₖ + x_Nᵀ Q_f x_N` s.t. `xₖ₊₁ = A·xₖ + B·uₖ`,
/// `u_min ≤ uₖ ≤ u_max`.
#[derive(Clone, Debug)]
pub struct LinearMpc {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub qf: DMatrix<f64>,
    pub horizon: usize,
    pub u_min: f64,
    pub u_max: f64,
}

impl LinearMpc {
    /// Optimal first input for the current state `x0` (apply it, then re-solve next step).
    pub fn control(&self, x0: &[f64]) -> Vec<f64> {
        let n = self.a.nrows();
        let m = self.b.ncols();
        let nn = self.horizon;

        // Prediction matrices: X = Sx·x0 + Su·U, stacking x₁…x_N.
        let mut apow = vec![DMatrix::<f64>::identity(n, n)];
        for _ in 0..nn {
            let next = &self.a * apow.last().unwrap();
            apow.push(next);
        }
        let mut sx = DMatrix::zeros(nn * n, n);
        let mut su = DMatrix::zeros(nn * n, nn * m);
        for k in 0..nn {
            sx.view_mut((k * n, 0), (n, n)).copy_from(&apow[k + 1]);
            for j in 0..=k {
                let blk = &apow[k - j] * &self.b;
                su.view_mut((k * n, j * m), (n, m)).copy_from(&blk);
            }
        }

        // Block-diagonal cost weights over the horizon (terminal Q_f on x_N).
        let mut qbar = DMatrix::zeros(nn * n, nn * n);
        let mut rbar = DMatrix::zeros(nn * m, nn * m);
        for k in 0..nn {
            let blk = if k == nn - 1 { &self.qf } else { &self.q };
            qbar.view_mut((k * n, k * n), (n, n)).copy_from(blk);
            rbar.view_mut((k * m, k * m), (m, m)).copy_from(&self.r);
        }

        // Condensed QP: minimize ½ Uᵀ H U + fᵀ U.
        let sut_q = su.transpose() * &qbar;
        let h = &sut_q * &su + &rbar;
        let h = 0.5 * (&h + &h.transpose()); // symmetrize for numerical safety
        let f = &sut_q * &sx * DVector::from_row_slice(x0);

        let dim = nn * m;
        let q_lin: Vec<f64> = f.iter().cloned().collect();
        let u = solve_box_qp(&h, &q_lin, &vec![self.u_min; dim], &vec![self.u_max; dim]);
        u[0..m].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpc_regulates_within_input_bounds() {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let mpc = LinearMpc {
            a: a.clone(),
            b: b.clone(),
            q: DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0])),
            r: DMatrix::from_row_slice(1, 1, &[0.1]),
            qf: DMatrix::from_diagonal(&DVector::from_row_slice(&[10.0, 10.0])),
            horizon: 20,
            u_min: -1.0,
            u_max: 1.0,
        };

        let mut x = DVector::from_row_slice(&[1.0, 0.0]);
        let mut max_u: f64 = 0.0;
        for _ in 0..120 {
            let u = mpc.control(x.as_slice())[0];
            max_u = max_u.max(u.abs());
            x = &a * &x + &b * u;
        }
        assert!(max_u <= 1.0 + 1e-6, "input bound violated: {max_u}");
        assert!(x.norm() < 1e-2, "state did not regulate: {x:?}");
    }
}
