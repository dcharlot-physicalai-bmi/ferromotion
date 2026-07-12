//! TinyMPC — a tiny embedded MPC (Nguyen et al., ICRA 2024). Box-constrained linear MPC solved by
//! ADMM where the LQR backward pass is **cached** (infinite-horizon gain + cost-to-go precomputed
//! once), so each ADMM iteration is a cheap Riccati sweep + a clamp. Allocation-light and
//! `no_std`-friendly in spirit — the MPC for the sim-to-real / microcontroller path. Pure Rust.

use nalgebra::{DMatrix, DVector};

/// Discrete linear MPC with box input constraints, solved by cached-Riccati ADMM.
#[derive(Clone, Debug)]
pub struct TinyMpc {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub horizon: usize,
    pub u_min: f64,
    pub u_max: f64,
    pub rho: f64,
    pub admm_iters: usize,
}

impl TinyMpc {
    /// Optimal (bounded) first input for the current state, receding-horizon.
    pub fn control(&self, x0: &[f64]) -> Vec<f64> {
        let n = self.a.nrows();
        let m = self.b.ncols();
        let nn = self.horizon;

        // Cache: infinite-horizon LQR for the ADMM-augmented input cost R̃ = R + ρI.
        let rtil = &self.r + DMatrix::identity(m, m) * self.rho;
        let mut p = self.q.clone();
        for _ in 0..2000 {
            let btp = self.b.transpose() * &p;
            let Some(sinv) = (&rtil + &btp * &self.b).try_inverse() else { break };
            let atp = self.a.transpose() * &p;
            let k = &sinv * (&btp * &self.a);
            let pn = &atp * &self.a - (&atp * &self.b) * &k + &self.q;
            let done = (&pn - &p).norm() < 1e-10;
            p = pn;
            if done {
                break;
            }
        }
        let btp = self.b.transpose() * &p;
        let quu_inv = (&rtil + &btp * &self.b).try_inverse().expect("Quu invertible");
        let kinf = &quu_inv * (&btp * &self.a); // m×n
        let ambk_t = (&self.a - &self.b * &kinf).transpose(); // (A − B·K)ᵀ, n×n
        let x0v = DVector::from_row_slice(x0);

        // ADMM state (input slack z, dual y).
        let mut z = vec![DVector::<f64>::zeros(m); nn];
        let mut y = vec![DVector::<f64>::zeros(m); nn];
        let mut u = vec![DVector::<f64>::zeros(m); nn];

        for _ in 0..self.admm_iters {
            // Backward pass: feedforward d and linear cost-to-go p_lin from the ADMM terms.
            let mut d = vec![DVector::<f64>::zeros(m); nn];
            let mut p_lin = vec![DVector::<f64>::zeros(n); nn + 1];
            for k in (0..nn).rev() {
                let r_k = (&z[k] - &y[k]) * -self.rho; // input linear term
                d[k] = &quu_inv * (self.b.transpose() * &p_lin[k + 1] + &r_k);
                p_lin[k] = &ambk_t * &p_lin[k + 1] - kinf.transpose() * &r_k;
            }
            // Forward rollout with u = −K·x − d.
            let mut x = x0v.clone();
            for k in 0..nn {
                u[k] = -(&kinf * &x) - &d[k];
                x = &self.a * &x + &self.b * &u[k];
            }
            // Slack projection onto the box + dual update.
            for k in 0..nn {
                let mut znew = &u[k] + &y[k];
                for i in 0..m {
                    znew[i] = znew[i].clamp(self.u_min, self.u_max);
                }
                y[k] += &u[k] - &znew;
                z[k] = znew;
            }
        }
        // z[0] is feasible by construction (projected onto the box).
        z[0].as_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regulates_a_double_integrator_within_bounds() {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let mpc = TinyMpc {
            a: a.clone(),
            b: b.clone(),
            q: DMatrix::from_diagonal(&DVector::from_row_slice(&[10.0, 1.0])),
            r: DMatrix::from_row_slice(1, 1, &[1.0]),
            horizon: 20,
            u_min: -1.0,
            u_max: 1.0,
            rho: 5.0,
            admm_iters: 80,
        };

        let mut x = DVector::from_row_slice(&[1.0, 0.0]);
        let mut max_u: f64 = 0.0;
        for _ in 0..100 {
            let u = mpc.control(x.as_slice())[0];
            max_u = max_u.max(u.abs());
            x = &a * &x + &b * u;
        }
        assert!(max_u <= 1.0 + 1e-6, "input bound violated: {max_u}");
        assert!(x.norm() < 1e-2, "state did not regulate: {x:?}");
    }
}
