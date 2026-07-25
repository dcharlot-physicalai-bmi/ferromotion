//! **Moving-Horizon / full-information Estimation** — optimization-based state estimation, the
//! complement to the recursive filters. Instead of a single Bayesian update, MHE solves a batch
//! least-squares (MAP) problem over a window of measurements: minimize the arrival cost plus the
//! dynamics-residual and measurement-residual energies. Its strength is constraints and nonlinearity
//! over a window; here the verifiable core is the **full-information linear** case, which is provably
//! identical to the Kalman filter (batch MAP = recursive KF for a linear-Gaussian model) — two
//! independent algorithms, one answer. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A linear-Gaussian model `x_{k+1} = A x_k + w`, `y_k = C x_k + v`, `w∼𝒩(0,Q)`, `v∼𝒩(0,R)`, with a
/// Gaussian prior `x_0 ∼ 𝒩(x̄₀, P₀)`.
pub struct LinearModel {
    pub a: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub q: DMatrix<f64>,
    pub r: DMatrix<f64>,
    pub x0: DVector<f64>,
    pub p0: DMatrix<f64>,
}

/// Full-information batch MHE: the MAP estimate of the whole trajectory `x₀…x_T` given measurements
/// `y₁…y_T`. Stacks the arrival, dynamics, and measurement residuals into one weighted least-squares
/// system and solves the normal equations; returns the smoothed states `x₀…x_T`.
#[allow(clippy::needless_range_loop)] // block-row index addresses several stacked matrices
pub fn batch_estimate(m: &LinearModel, ys: &[DVector<f64>]) -> Vec<DVector<f64>> {
    let n = m.a.nrows(); // state dim
    let t = ys.len(); // measurements y_1..y_T
    let nv = (t + 1) * n; // variables x_0..x_T

    // whitening factors L such that LᵀL = W⁻¹ (so ‖r‖²_{W⁻¹} = ‖L r‖²)
    let whiten = |w: &DMatrix<f64>| -> DMatrix<f64> {
        // W⁻¹ = (W⁻¹); use Cholesky of W⁻¹
        let winv = w.clone().try_inverse().expect("cov invertible");
        winv.cholesky().expect("cov PD").l().transpose() // upper L with LᵀL = W⁻¹
    };
    let la = whiten(&m.q);
    let lc = whiten(&m.r);
    let lp = whiten(&m.p0);

    // count rows: arrival(n) + dynamics(t·n) + measurement(t·c_rows)
    let cr = m.c.nrows();
    let rows = n + t * n + t * cr;
    let mut big = DMatrix::<f64>::zeros(rows, nv);
    let mut rhs = DVector::<f64>::zeros(rows);
    let mut row = 0;
    let blk = |big: &mut DMatrix<f64>, r: usize, c: usize, m: &DMatrix<f64>| {
        big.view_mut((r, c), (m.nrows(), m.ncols())).copy_from(m);
    };

    // arrival: L_p (x_0 − x̄₀)
    blk(&mut big, row, 0, &lp);
    rhs.rows_mut(row, n).copy_from(&(&lp * &m.x0));
    row += n;
    // dynamics: L_a (x_{i+1} − A x_i) = 0
    for i in 0..t {
        let neg_a = &la * &m.a;
        blk(&mut big, row, i * n, &(-&neg_a));
        blk(&mut big, row, (i + 1) * n, &la);
        row += n;
    }
    // measurements: L_c (y_i − C x_i)
    for i in 0..t {
        let lcc = &lc * &m.c;
        blk(&mut big, row, (i + 1) * n, &lcc);
        rhs.rows_mut(row, cr).copy_from(&(&lc * &ys[i]));
        row += cr;
    }

    // normal equations (AᵀA) x = Aᵀ b
    let ata = big.transpose() * &big;
    let atb = big.transpose() * &rhs;
    let x = ata.cholesky().expect("normal system PD").solve(&atb);
    (0..=t).map(|i| x.rows(i * n, n).into_owned()).collect()
}

#[cfg(test)]
mod verification {
    use super::*;

    /// Full-information linear MHE equals the Kalman filter at the final time — the same estimate by
    /// two independent routes (recursive Bayesian update vs one batch least-squares).
    #[test]
    fn mhe_matches_kalman_filter_linear_gaussian() {
        // a constant-velocity model: x = [pos, vel]
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]); // observe position
        let q = DMatrix::from_row_slice(2, 2, &[1e-3, 0.0, 0.0, 1e-2]);
        let r = DMatrix::from_row_slice(1, 1, &[0.05]);
        let x0 = DVector::from_vec(vec![0.0, 1.0]);
        let p0 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let model = LinearModel { a: a.clone(), c: c.clone(), q: q.clone(), r: r.clone(), x0: x0.clone(), p0: p0.clone() };

        // synthetic measurements (fixed)
        let ys: Vec<DVector<f64>> = [0.12, 0.19, 0.31, 0.38, 0.52, 0.58, 0.71, 0.79]
            .iter()
            .map(|&z| DVector::from_vec(vec![z]))
            .collect();

        // batch MHE
        let smoothed = batch_estimate(&model, &ys);
        let mhe_final = smoothed.last().unwrap().clone();

        // recursive Kalman filter over the same measurements
        let mut xh = x0.clone();
        let mut p = p0.clone();
        for y in &ys {
            // predict
            xh = &a * &xh;
            p = &a * &p * a.transpose() + &q;
            // update
            let s = &c * &p * c.transpose() + &r;
            let k = &p * c.transpose() * s.try_inverse().unwrap();
            let innov = y - &c * &xh;
            xh += &k * innov;
            p = (DMatrix::identity(2, 2) - &k * &c) * p;
        }
        let err = (&mhe_final - &xh).amax();
        eprintln!("MHE final {:?} vs KF final {:?}: max diff {err:.2e}", mhe_final.as_slice(), xh.as_slice());
        assert!(err < 1e-9, "batch MHE ≠ Kalman filter: {err}");
    }
}
