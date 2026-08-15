//! Fully-differentiable frictional contact step, via the interior-point core — in 2D or 3D. Normal +
//! pyramidal (Stewart-Trinkle) friction with an arbitrary set of tangent facets is a linear
//! complementarity problem `0 ≤ z ⟂ (M_lcp·z + q) ≥ 0` with a P-matrix `M_lcp` (not symmetric —
//! friction couples through a cone slack). Solving it on the central path ([`crate::ipm`]) makes the
//! *entire* frictional step differentiable: the post-contact velocity and `∂v⁺/∂v_free` fall out via
//! the implicit function theorem, smooth even through stick↔slip. This is Dojo's mechanism, applied
//! per contact with `d` friction facets. Pure `nalgebra` → WASM-clean.

use crate::ipm::solve_lcp_diff;
use nalgebra::{DMatrix, DVector};

/// A frictional contact: normal row `jn`, a set of tangent friction-facet directions `jt` (each a
/// row; e.g. `[+t, −t]` for planar, `[+x, −x, +y, −y]` for a 3D pyramid), signed gap `phi`, and
/// friction coefficient `mu`.
#[derive(Clone, Debug)]
pub struct StFrictionContact {
    pub jn: DVector<f64>,
    pub jt: Vec<DVector<f64>>,
    pub phi: f64,
    pub mu: f64,
}

/// Result of a differentiable frictional contact solve.
#[derive(Clone, Debug)]
pub struct FrictionalStep {
    pub v_next: DVector<f64>,
    /// `∂v⁺/∂v_free` (dof×dof) — backprop through the frictional contact.
    pub dvnext_dvfree: DMatrix<f64>,
    /// How well the central-path condition `z∘w = κ` was actually met: `max|zᵢwᵢ − κ|`. A solve that
    /// quietly failed to converge returns an impulse that is not a solution to anything, and in a
    /// simulation loop that shows up as a body gaining energy. Report this next to the answer.
    pub residual: f64,
    /// Worst violation of `z ≥ 0, w ≥ 0`; negative means the returned point is infeasible.
    pub feasibility: f64,
}

/// Solve one frictional contact step and its gradient. Per contact the LCP variables are
/// `[λₙ, β₁…β_d, s]` (normal impulse, per-facet friction magnitudes, cone slack). `kappa` is the
/// central-path smoothing (→0 = hard contact, >0 = smooth gradients).
pub fn solve_frictional_ipm(m: &DMatrix<f64>, v_free: &DVector<f64>, contacts: &[StFrictionContact], dt: f64, kappa: f64) -> FrictionalStep {
    let nv = v_free.len();
    let minv = m.clone().try_inverse().expect("mass matrix invertible");

    // Per-contact variable block: [λₙ, β₁…β_d, s]; record each block's start index.
    let mut starts = Vec::with_capacity(contacts.len());
    let mut nz = 0;
    for c in contacts {
        starts.push(nz);
        nz += 2 + c.jt.len();
    }

    let mut b = DMatrix::zeros(nv, nz); // impulse map (columns that apply impulse)
    let mut e = DMatrix::zeros(nz, nz); // direct z-coupling
    let mut q0 = DVector::zeros(nz);
    for (i, c) in contacts.iter().enumerate() {
        let d = c.jt.len();
        let ln = starts[i];
        let s_idx = ln + 1 + d;
        b.set_column(ln, &c.jn);
        q0[ln] = c.phi / dt;
        for (kf, dir) in c.jt.iter().enumerate() {
            let bk = ln + 1 + kf;
            b.set_column(bk, dir);
            e[(bk, s_idx)] = 1.0; // w_{β_k} += s
            e[(s_idx, bk)] = -1.0; // w_s −= β_k
        }
        e[(s_idx, ln)] = c.mu; // w_s = μ·λₙ − Σ β_k
    }

    // w = M_lcp·z + q, with M_lcp = Bᵀ M⁻¹ B + E and q = Bᵀ v_free + q0.
    let bt = b.transpose();
    let m_lcp = &bt * &minv * &b + &e;
    let q = &bt * v_free + &q0;

    let (z, dz_dq) = solve_lcp_diff(&m_lcp, q.as_slice(), kappa);
    let v_next = v_free + &minv * &b * &z;
    let dvnext_dvfree = DMatrix::identity(nv, nv) + &minv * &b * &dz_dq * &bt;
    let w = &m_lcp * &z + &q;
    // **Both reductions propagate `NaN` on purpose (2026-08-14).** `f64::max` and `f64::min` return the
    // *non-`NaN`* operand by IEEE-754 `maxNum`/`minNum`, so a solve that has gone entirely non-finite used to
    // report `residual = 0.0` — the central-path condition met exactly, per this struct's own doc — and
    // `feasibility = +INFINITY`, i.e. maximally feasible, while `v_next` was all-`NaN`. The two numbers whose
    // stated job is to tell a caller the solve failed were the two that could not. Reachable from
    // `dt = 0.0` (φ/dt is non-finite) or a non-finite φ, neither of which is guarded upstream.
    let residual = z
        .iter()
        .zip(w.iter())
        .fold(0.0f64, |m, (zi, wi)| {
            let r = (zi * wi - kappa).abs();
            if m.is_nan() || r.is_nan() { f64::NAN } else { m.max(r) }
        });
    let feasibility = z
        .iter()
        .chain(w.iter())
        .fold(f64::INFINITY, |m, &v| if m.is_nan() || v.is_nan() { f64::NAN } else { m.min(v) });
    FrictionalStep { v_next, dvnext_dvfree, residual, feasibility }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_health_readouts_report_a_failed_solve_instead_of_a_perfect_one() {
        // `residual` and `feasibility` exist to tell a caller the solve did not work. With `f64::max` /
        // `f64::min` they were the two quantities that could not: IEEE maxNum/minNum return the non-NaN
        // operand, so an all-NaN solve read as residual 0.0 (central-path condition met exactly) and
        // feasibility +INFINITY (maximally feasible) beside an all-NaN `v_next`.
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[2.0, 2.0]));
        // A non-finite gap. Nothing upstream guards φ, and φ/dt is how it enters the LCP right-hand side.
        let c = StFrictionContact {
            jn: DVector::from_row_slice(&[0.0, 1.0]),
            jt: vec![DVector::from_row_slice(&[1.0, 0.0]), DVector::from_row_slice(&[-1.0, 0.0])],
            phi: f64::NAN,
            mu: 0.5,
        };
        let v_free = DVector::from_row_slice(&[2.0, -0.0981]);
        let s = solve_frictional_ipm(&m, &v_free, std::slice::from_ref(&c), 0.01, 1e-6);
        assert!(s.v_next.iter().any(|v| v.is_nan()), "this fixture is supposed to produce a broken solve");
        assert!(
            !s.residual.is_finite(),
            "a non-finite solve reported residual {} — the old fold returned exactly 0.0, which reads as \
             `converged` against the doc's own contract",
            s.residual
        );
        assert!(
            !s.feasibility.is_finite() || s.feasibility <= 0.0,
            "a non-finite solve reported feasibility {} — the old fold returned +INFINITY, i.e. maximally \
             feasible",
            s.feasibility
        );
    }

    #[test]
    fn ipm_friction_decelerates_a_planar_sliding_block() {
        // 2-DoF (x, z), mass 2, normal +z, friction facets ±x.
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[2.0, 2.0]));
        let c = StFrictionContact {
            jn: DVector::from_row_slice(&[0.0, 1.0]),
            jt: vec![DVector::from_row_slice(&[1.0, 0.0]), DVector::from_row_slice(&[-1.0, 0.0])],
            phi: 0.0,
            mu: 0.5,
        };
        let (g, dt) = (9.81, 0.01);
        let (mut vx, mut vz, mut min_vx) = (2.0, 0.0, 2.0f64);
        for _ in 0..80 {
            let v_free = DVector::from_row_slice(&[vx, vz - g * dt]);
            let s = solve_frictional_ipm(&m, &v_free, std::slice::from_ref(&c), dt, 1e-6);
            vx = s.v_next[0];
            vz = s.v_next[1];
            min_vx = min_vx.min(vx);
        }
        assert!(vx.abs() < 0.05 && min_vx > -1e-2, "block did not cleanly stop: vx = {vx}, min = {min_vx}");
    }

    #[test]
    fn ipm_friction_stops_a_3d_block_sliding_diagonally() {
        // 3-DoF (x, y, z), mass 1, normal +z, 4-facet pyramid ±x, ±y. Initial diagonal slide.
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0, 1.0]));
        let row = |a: [f64; 3]| DVector::from_row_slice(&a);
        let c = StFrictionContact {
            jn: row([0.0, 0.0, 1.0]),
            jt: vec![row([1.0, 0.0, 0.0]), row([-1.0, 0.0, 0.0]), row([0.0, 1.0, 0.0]), row([0.0, -1.0, 0.0])],
            phi: 0.0,
            mu: 0.6,
        };
        let (g, dt) = (9.81, 0.01);
        let mut v = DVector::from_row_slice(&[1.5, 1.0, 0.0]);
        for _ in 0..100 {
            let v_free = DVector::from_row_slice(&[v[0], v[1], v[2] - g * dt]);
            v = solve_frictional_ipm(&m, &v_free, std::slice::from_ref(&c), dt, 1e-6).v_next;
        }
        assert!(v[0].abs() < 0.06 && v[1].abs() < 0.06, "3D block did not stop: v = {v:?}");
    }

    #[test]
    fn frictional_step_gradient_matches_finite_difference_3d() {
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 1.0, 1.0]));
        let row = |a: [f64; 3]| DVector::from_row_slice(&a);
        let c = StFrictionContact {
            jn: row([0.0, 0.0, 1.0]),
            jt: vec![row([1.0, 0.0, 0.0]), row([-1.0, 0.0, 0.0]), row([0.0, 1.0, 0.0]), row([0.0, -1.0, 0.0])],
            phi: 0.0,
            mu: 0.6,
        };
        let (dt, kappa) = (0.01, 1e-3);
        let v_free = DVector::from_row_slice(&[0.3, 0.2, -0.1]);
        let s = solve_frictional_ipm(&m, &v_free, std::slice::from_ref(&c), dt, kappa);
        assert!(s.dvnext_dvfree.iter().all(|v| v.is_finite()), "gradient not finite");
        let eps = 1e-6;
        for col in 0..3 {
            let mut vp = v_free.clone();
            vp[col] += eps;
            let fd = (solve_frictional_ipm(&m, &vp, std::slice::from_ref(&c), dt, kappa).v_next - &s.v_next) / eps;
            for r in 0..3 {
                assert!((s.dvnext_dvfree[(r, col)] - fd[r]).abs() < 5e-3, "∂v⁺/∂v_free[{r},{col}] mismatch");
            }
        }
    }
}
