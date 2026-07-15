//! **Reduced-order modal deformables** — linear modal analysis for real-time deformable simulation
//! (the classical subspace method behind interactive cloth/soft-body, and the base that neural
//! subspace / "low-rank Koopman deformable" methods build on).
//!
//! A deformable linearized about its rest shape obeys `M ẍ + K x = f` (mass matrix `M`, stiffness
//! `K = ∂²E/∂x²`). The generalized eigenproblem `K φ = ω² M φ` yields **mode shapes** `φ_i` that are
//! `M`- and `K`-orthogonal, so in the modal coordinates `x = Σ q_i φ_i` the dynamics **decouple** into
//! independent oscillators `q̈_i + ω_i² q_i = φ_iᵀ f`. Keeping only the few lowest-frequency modes
//! gives a tiny reduced model that captures the dominant motion — a full `n`-DoF simulation collapses
//! to `r` scalar oscillators. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A modal reduced-order model: a basis of mode shapes, their frequencies, and the projector onto
/// modal coordinates.
#[derive(Clone, Debug)]
pub struct ModalModel {
    /// Mode shapes as columns (`n × r`), `M`-orthonormal.
    pub basis: DMatrix<f64>,
    /// Natural frequencies `ω_i` (`r`), ascending.
    pub freq: DVector<f64>,
    /// Projector to modal coordinates `q = Uᵀ M x` (`r × n`).
    pub project: DMatrix<f64>,
}

/// Modal analysis: the `r` lowest-frequency modes of `M ẍ + K x = 0` (`M` SPD, `K` symmetric PSD).
pub fn modal_analysis(m: &DMatrix<f64>, k: &DMatrix<f64>, r: usize) -> ModalModel {
    let n = m.nrows();
    // Transform K φ = ω² M φ to a standard symmetric eigenproblem via M = L Lᵀ.
    let l = m.clone().cholesky().expect("M must be SPD").l();
    let linv = l.try_inverse().expect("L invertible");
    let a = &linv * k * linv.transpose();
    let a = (&a + a.transpose()) * 0.5; // symmetrize against round-off
    let eig = a.symmetric_eigen();

    // Sort eigenpairs by ascending eigenvalue and keep the r smallest.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| eig.eigenvalues[i].partial_cmp(&eig.eigenvalues[j]).unwrap());

    let (mut basis, mut freq) = (DMatrix::zeros(n, r), DVector::zeros(r));
    for (c, &i) in idx.iter().take(r).enumerate() {
        // φ = L⁻ᵀ ψ  (then φᵀ M φ = ψᵀψ = 1 automatically).
        let phi = linv.transpose() * eig.eigenvectors.column(i);
        basis.set_column(c, &phi);
        freq[c] = eig.eigenvalues[i].max(0.0).sqrt();
    }
    let project = basis.transpose() * m;
    ModalModel { basis, freq, project }
}

impl ModalModel {
    /// Project a full displacement to modal coordinates.
    pub fn reduce(&self, x: &DVector<f64>) -> DVector<f64> {
        &self.project * x
    }

    /// Reconstruct a full displacement from modal coordinates.
    pub fn reconstruct(&self, q: &DVector<f64>) -> DVector<f64> {
        &self.basis * q
    }

    /// Advance the decoupled modal oscillators one semi-implicit step under a full force `f`
    /// (`q̈_i = −ω_i² q_i + φ_iᵀ f`), updating modal position/velocity in place.
    pub fn step(&self, q: &mut DVector<f64>, qd: &mut DVector<f64>, f: &DVector<f64>, dt: f64) {
        let modal_f = self.basis.transpose() * f;
        for i in 0..self.freq.len() {
            let acc = -self.freq[i] * self.freq[i] * q[i] + modal_f[i];
            qd[i] += acc * dt;
            q[i] += qd[i] * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed-fixed 1D mass-spring chain of `n` unit masses, unit springs.
    fn chain(n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
        let m = DMatrix::identity(n, n);
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 2.0;
            if i > 0 {
                k[(i, i - 1)] = -1.0;
            }
            if i + 1 < n {
                k[(i, i + 1)] = -1.0;
            }
        }
        (m, k)
    }

    #[test]
    fn modes_solve_the_generalized_eigenproblem_and_match_the_analytic_chain() {
        let n = 10;
        let (m, k) = chain(n);
        let model = modal_analysis(&m, &k, n);
        for c in 0..n {
            let phi = model.basis.column(c).into_owned();
            let w2 = model.freq[c] * model.freq[c];
            // K φ = ω² M φ.
            assert!((&k * &phi - w2 * (&m * &phi)).norm() < 1e-9, "eigenpair {c} residual");
            // Analytic fixed-fixed chain frequency ω_j = 2·sin(jπ / 2(n+1)).
            let w_true = 2.0 * ((c + 1) as f64 * std::f64::consts::PI / (2.0 * (n + 1) as f64)).sin();
            assert!((model.freq[c] - w_true).abs() < 1e-9, "ω[{c}] = {} vs analytic {w_true}", model.freq[c]);
        }
        // M-orthonormal basis: UᵀMU = I.
        let gram = model.basis.transpose() * &m * &model.basis;
        assert!((gram - DMatrix::identity(n, n)).norm() < 1e-9, "basis not M-orthonormal");
    }

    #[test]
    fn full_basis_reproduces_the_full_simulation_and_a_few_modes_capture_low_frequency_motion() {
        let n = 12;
        let (m, k) = chain(n);
        let dt = 0.01;
        // Excite mode 2 (a smooth, low-frequency shape).
        let full_model = modal_analysis(&m, &k, n);
        let x0 = full_model.basis.column(1).into_owned() * 0.5;

        // Reference: full linearized dynamics M ẍ = −K x, same semi-implicit integrator.
        let minv = m.clone().try_inverse().unwrap();
        let mut xf = x0.clone();
        let mut vf = DVector::zeros(n);

        // Full-rank modal model (r = n) — should reproduce the full sim exactly.
        let mut qn = full_model.reduce(&x0);
        let mut qdn = DVector::zeros(n);

        // Reduced model with just 3 modes.
        let red = modal_analysis(&m, &k, 3);
        let mut qr = red.reduce(&x0);
        let mut qdr = DVector::zeros(3);

        let (mut err_full, mut err_red) = (0.0f64, 0.0f64);
        let zero = DVector::zeros(n);
        for _ in 0..400 {
            // full
            let acc = &minv * (-(&k * &xf));
            vf += acc * dt;
            xf += &vf * dt;
            // full-rank modal
            full_model.step(&mut qn, &mut qdn, &zero, dt);
            err_full = err_full.max((full_model.reconstruct(&qn) - &xf).norm());
            // 3-mode reduced
            red.step(&mut qr, &mut qdr, &zero, dt);
            err_red = err_red.max((red.reconstruct(&qr) - &xf).norm());
        }
        // The full modal basis reproduces the full simulation to machine precision.
        assert!(err_full < 1e-9, "full modal ≠ full sim: {err_full:.2e}");
        // Three modes capture this low-frequency excitation very well.
        assert!(err_red < 1e-6, "3-mode ROM did not capture the low mode: {err_red:.2e}");
    }
}
