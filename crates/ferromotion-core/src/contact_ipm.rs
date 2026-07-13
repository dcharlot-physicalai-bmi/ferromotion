//! Fully-differentiable frictional contact step, via the interior-point core. Normal + pyramidal
//! (Stewart-Trinkle) friction is a linear complementarity problem `0 ≤ z ⟂ (M_lcp·z + q) ≥ 0` with a
//! P-matrix `M_lcp` (not symmetric — friction couples through a cone slack). Solving it on the central
//! path ([`crate::ipm`]) makes the *entire* frictional step differentiable: the post-contact velocity
//! and its gradient `∂v⁺/∂v_free` fall out via the implicit function theorem, smooth even through
//! stick↔slip transitions. This is the Dojo assembly — contact + friction + smooth gradients in one
//! step. Pure `nalgebra` → WASM-clean.

use crate::ipm::solve_lcp_diff;
use nalgebra::{DMatrix, DVector};

/// A frictional contact with one tangent direction (planar): normal row `jn`, tangent row `jt`,
/// signed gap `phi`, friction coefficient `mu`.
#[derive(Clone, Debug)]
pub struct StFrictionContact {
    pub jn: DVector<f64>,
    pub jt: DVector<f64>,
    pub phi: f64,
    pub mu: f64,
}

/// Result of a differentiable frictional contact solve.
#[derive(Clone, Debug)]
pub struct FrictionalStep {
    pub v_next: DVector<f64>,
    /// `∂v⁺/∂v_free` (dof×dof) — backprop through the frictional contact.
    pub dvnext_dvfree: DMatrix<f64>,
}

/// Solve one frictional contact step and its gradient. Per contact the LCP variables are
/// `[λₙ, β⁺, β⁻, s]` (normal impulse, ± friction magnitudes, cone slack); `kappa` is the central-path
/// smoothing (→0 = hard contact, >0 = smooth gradients).
pub fn solve_frictional_ipm(m: &DMatrix<f64>, v_free: &DVector<f64>, contacts: &[StFrictionContact], dt: f64, kappa: f64) -> FrictionalStep {
    let nv = v_free.len();
    let k = contacts.len();
    let nz = 4 * k;
    let minv = m.clone().try_inverse().expect("mass matrix invertible");

    // Impulse map B (nv×nz): only λₙ, β⁺, β⁻ apply impulse (jn, +jt, −jt); s applies none.
    let mut b = DMatrix::zeros(nv, nz);
    // Direct z-coupling E and constant q0.
    let mut e = DMatrix::zeros(nz, nz);
    let mut q0 = DVector::zeros(nz);
    for (i, c) in contacts.iter().enumerate() {
        let (ln, bp, bm, s) = (4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
        b.set_column(ln, &c.jn);
        b.set_column(bp, &c.jt);
        b.set_column(bm, &(-&c.jt));
        q0[ln] = c.phi / dt; // Baumgarte gap stabilization on the normal row
        e[(bp, s)] = 1.0; // w_{β⁺} += s
        e[(bm, s)] = 1.0; // w_{β⁻} += s
        e[(s, ln)] = c.mu; // w_s = μ·λₙ − β⁺ − β⁻ (friction cone)
        e[(s, bp)] = -1.0;
        e[(s, bm)] = -1.0;
    }

    // w = M_lcp·z + q, with M_lcp = Bᵀ M⁻¹ B + E and q = Bᵀ v_free + q0.
    let bt = b.transpose();
    let m_lcp = &bt * &minv * &b + &e;
    let q = &bt * v_free + &q0;

    let (z, dz_dq) = solve_lcp_diff(&m_lcp, q.as_slice(), kappa);
    let v_next = v_free + &minv * &b * &z;
    // ∂v⁺/∂v_free = I + M⁻¹ B (∂z/∂q) Bᵀ   (since ∂q/∂v_free = Bᵀ).
    let dvnext_dvfree = DMatrix::identity(nv, nv) + &minv * &b * &dz_dq * &bt;
    FrictionalStep { v_next, dvnext_dvfree }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_contact() -> (DMatrix<f64>, StFrictionContact) {
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[2.0, 2.0])); // (x, z), mass 2
        let c = StFrictionContact {
            jn: DVector::from_row_slice(&[0.0, 1.0]),
            jt: DVector::from_row_slice(&[1.0, 0.0]),
            phi: 0.0,
            mu: 0.5,
        };
        (m, c)
    }

    #[test]
    fn ipm_friction_decelerates_a_sliding_block() {
        let (m, c) = block_contact();
        let (g, dt) = (9.81, 0.01);
        let (mut vx, mut vz) = (2.0, 0.0);
        let mut min_vx = vx;
        let mut worst_cone: f64 = 0.0;
        for _ in 0..80 {
            let v_free = DVector::from_row_slice(&[vx, vz - g * dt]);
            let step = solve_frictional_ipm(&m, &v_free, std::slice::from_ref(&c), dt, 1e-6);
            vx = step.v_next[0];
            vz = step.v_next[1];
            min_vx = min_vx.min(vx);
        }
        assert!(vx.abs() < 0.05, "block did not stop: vx = {vx}");
        assert!(min_vx > -1e-2, "friction reversed the motion: min vx = {min_vx}");
        let _ = worst_cone;
    }

    #[test]
    fn frictional_step_gradient_matches_finite_difference() {
        let (m, c) = block_contact();
        let dt = 0.01;
        let v_free = DVector::from_row_slice(&[0.3, -0.1]); // sliding + pressing into the floor
        let kappa = 1e-3;
        let step = solve_frictional_ipm(&m, &v_free, std::slice::from_ref(&c), dt, kappa);
        assert!(step.dvnext_dvfree.iter().all(|v| v.is_finite()), "gradient not finite");
        let eps = 1e-6;
        for col in 0..2 {
            let mut vp = v_free.clone();
            vp[col] += eps;
            let fd = (solve_frictional_ipm(&m, &vp, std::slice::from_ref(&c), dt, kappa).v_next - &step.v_next) / eps;
            for row in 0..2 {
                assert!(
                    (step.dvnext_dvfree[(row, col)] - fd[row]).abs() < 5e-3,
                    "∂v⁺/∂v_free[{row},{col}]: analytic {} vs fd {}",
                    step.dvnext_dvfree[(row, col)],
                    fd[row]
                );
            }
        }
    }
}
