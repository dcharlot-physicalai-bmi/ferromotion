//! Control Barrier Function (CBF) safety filter: minimally correct a nominal control so the closed
//! loop stays inside a safe set. Each barrier `h_i(x) ≥ 0` is folded to an affine-in-control row
//! `a_iᵀu ≤ b_i` that enforces `ḣ_i + α·h_i ≥ 0` (relative-degree-1) or the higher-order analogue
//! (HOCBF). The filter solves `min ½‖u − u_nom‖²` s.t. those rows (plus optional actuator box
//! bounds), so a safe `u_nom` passes through untouched and an unsafe one is projected to the
//! nearest safe command. The single-halfspace case is closed-form; the general case is a small
//! dense QP through `clarabel` (mirrors `crate::qp`, WASM-clean, no extra deps).

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::DMatrix;

/// One affine-in-control CBF constraint: `aᵀu ≤ b`. Build it from a barrier so that satisfying the
/// row implies `ḣ + α·h ≥ 0` (see [`CbfConstraint::relative_degree1`] / [`CbfConstraint::hocbf2`]).
#[derive(Clone, Debug)]
pub struct CbfConstraint {
    /// Row of the control coefficient, `−L_g h` for a relative-degree-1 barrier.
    pub a: Vec<f64>,
    /// Upper bound, `L_f h + α·h` for a relative-degree-1 barrier.
    pub b: f64,
}

impl CbfConstraint {
    /// Relative-degree-1 barrier with control-affine dynamics `ẋ = f + g·u`:
    /// `ḣ = ∂h·f + (∂h·g)·u ≥ −α h`  ⟺  `−(∂h·g)ᵀ u ≤ ∂h·f + α h`.
    ///
    /// `lgh[j] = ∂h·g[:,j]` (one per control), `lfh = ∂h·f`, `h` and `alpha ≥ 0`.
    pub fn relative_degree1(lgh: &[f64], lfh: f64, h: f64, alpha: f64) -> Self {
        Self { a: lgh.iter().map(|v| -v).collect(), b: lfh + alpha * h }
    }

    /// Second-order (exponential) HOCBF for a relative-degree-2 barrier `h`, using
    /// `ψ0 = h`, `ψ1 = ḣ + α₁ h`, and enforcing `ψ̇1 + α₂ ψ1 ≥ 0`. The caller supplies the
    /// Lie derivatives of `ψ1`: `ψ̇1 = lf_psi1 + lg_psi1ᵀ u`, plus `psi1` itself. Both
    /// `α₁, α₂ ≥ 0`. Yields `−lg_psi1ᵀ u ≤ lf_psi1 + α₂ ψ1`.
    pub fn hocbf2(lg_psi1: &[f64], lf_psi1: f64, psi1: f64, alpha2: f64) -> Self {
        Self { a: lg_psi1.iter().map(|v| -v).collect(), b: lf_psi1 + alpha2 * psi1 }
    }
}

/// Minimal-intervention CBF safety filter with optional actuator box bounds.
#[derive(Clone, Debug, Default)]
pub struct CbfFilter {
    /// Per-control lower bound `u ≥ u_min` (elementwise), if the actuators are limited.
    pub u_min: Option<Vec<f64>>,
    /// Per-control upper bound `u ≤ u_max` (elementwise), if the actuators are limited.
    pub u_max: Option<Vec<f64>>,
}

impl CbfFilter {
    /// Unconstrained-box filter (only the CBF rows apply).
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter with symmetric torque/force limits `|u_i| ≤ limit`.
    pub fn with_symmetric_limits(limit: &[f64]) -> Self {
        Self {
            u_min: Some(limit.iter().map(|v| -v).collect()),
            u_max: Some(limit.to_vec()),
        }
    }

    /// Return the safe control `u* = argmin ½‖u − u_nom‖²` subject to every `a_iᵀu ≤ b_i` and the
    /// box bounds. If `u_nom` is already safe it is returned unchanged.
    pub fn filter(&self, u_nom: &[f64], constraints: &[CbfConstraint]) -> Vec<f64> {
        let n = u_nom.len();
        let boxed = self.u_min.is_some() || self.u_max.is_some();

        // Fast path: a single halfspace and no box bounds → orthogonal projection onto {aᵀu ≤ b}.
        if constraints.len() == 1 && !boxed {
            return project_halfspace(u_nom, &constraints[0]);
        }
        if constraints.is_empty() && !boxed {
            return u_nom.to_vec();
        }

        // General case: assemble the inequality rows [CBF; +I (u≤hi); −I (−u≤−lo)] and solve the QP
        // `min ½uᵀu − u_nomᵀu  s.t.  A u ≤ b`.
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(constraints.len() + 2 * n);
        let mut b: Vec<f64> = Vec::with_capacity(constraints.len() + 2 * n);
        for c in constraints {
            debug_assert_eq!(c.a.len(), n, "constraint row width must match u_nom");
            rows.push(c.a.clone());
            b.push(c.b);
        }
        if let Some(hi) = &self.u_max {
            for i in 0..n {
                let mut r = vec![0.0; n];
                r[i] = 1.0;
                rows.push(r);
                b.push(hi[i]);
            }
        }
        if let Some(lo) = &self.u_min {
            for i in 0..n {
                let mut r = vec![0.0; n];
                r[i] = -1.0;
                rows.push(r);
                b.push(-lo[i]);
            }
        }
        let a_mat = DMatrix::from_fn(rows.len(), n, |i, j| rows[i][j]);
        solve_ineq_qp(n, u_nom, &a_mat, &b)
    }
}

/// Orthogonal projection of `u_nom` onto the halfspace `{u : aᵀu ≤ b}` (the single-constraint QP
/// solved in closed form). Returns `u_nom` when already feasible.
fn project_halfspace(u_nom: &[f64], c: &CbfConstraint) -> Vec<f64> {
    let dot: f64 = c.a.iter().zip(u_nom).map(|(ai, ui)| ai * ui).sum();
    let slack = dot - c.b;
    let a2: f64 = c.a.iter().map(|v| v * v).sum();
    if slack <= 0.0 || a2 <= 0.0 {
        return u_nom.to_vec();
    }
    let scale = slack / a2;
    u_nom.iter().zip(&c.a).map(|(ui, ai)| ui - scale * ai).collect()
}

/// Upper-triangular CSC of the identity (`P = I` for the `½‖u‖²` objective).
fn csc_identity(n: usize) -> CscMatrix<f64> {
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        rowval.push(j);
        nzval.push(1.0);
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Column-compressed sparse form of a dense `m×n` matrix (keeps only nonzeros).
fn csc_dense(a: &DMatrix<f64>) -> CscMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..m {
            let v = a[(i, j)];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

/// Solve `min ½uᵀu − u_nomᵀu  s.t.  A u ≤ b` (nonnegative cone on the slack). `A` is `m×n`.
fn solve_ineq_qp(n: usize, u_nom: &[f64], a: &DMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let p_csc = csc_identity(n);
    let q: Vec<f64> = u_nom.iter().map(|v| -v).collect();
    let a_csc = csc_dense(a);
    let cones = [SupportedConeT::NonnegativeConeT(a.nrows())];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p_csc, &q, &a_csc, b, &cones, settings).unwrap();
    solver.solve();
    solver.solution.x.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Double integrator `ẋ = v, v̇ = u` driven RIGHT by a nominal controller into a wall at
    /// `x_max`. A relative-degree-2 barrier `h = x_max − x` (exponential HOCBF) must keep `x`
    /// inside the safe set for all time while the arm still approaches the wall.
    #[test]
    fn double_integrator_never_crosses_the_wall() {
        let x_max = 1.0;
        let x_target = 2.0; // beyond the wall → u_nom always pushes toward violation
        let k = 5.0;
        let (a1, a2) = (4.0, 4.0); // HOCBF class-K gains
        let filter = CbfFilter::new();

        let (mut x, mut v, dt) = (0.0_f64, 0.0_f64, 1e-3);
        let mut max_x = f64::MIN;
        let mut constrained_ever = false;

        for _ in 0..3000 {
            // Nominal: proportional drive toward a target past the wall (always u_nom > 0 here).
            let u_nom = -k * (x - x_target);

            // HOCBF for h = x_max − x, relative degree 2:
            //   ḣ  = −v
            //   ψ1 = ḣ + a1·h = −v + a1·(x_max − x)
            //   ψ̇1 = −u − a1·v   ⇒   lf = −a1·v, lg = −1
            // constraint  ψ̇1 + a2·ψ1 ≥ 0  ⇒  u ≤ −(a1+a2)v + a1·a2·(x_max − x)
            let h = x_max - x;
            let psi1 = -v + a1 * h;
            let c = CbfConstraint::hocbf2(&[-1.0], -a1 * v, psi1, a2);

            let u = filter.filter(&[u_nom], &c_slice(&c))[0];
            if u < u_nom - 1e-9 {
                constrained_ever = true;
            }

            // Semi-implicit Euler.
            v += u * dt;
            x += v * dt;
            if x > max_x {
                max_x = x;
            }
        }

        // Core invariance property: the safe set is never left (tiny discretization tolerance).
        assert!(max_x <= x_max + 5e-3, "safe set violated: max_x = {max_x}");
        // The filter actually had to intervene (otherwise the test proves nothing).
        assert!(constrained_ever, "filter never engaged — test is vacuous");
        // ...and the system still approaches the wall rather than stalling far away.
        assert!(x > x_max - 0.05, "did not approach the wall: final x = {x}");
    }

    /// A safe nominal control must pass through untouched (minimal intervention).
    #[test]
    fn safe_nominal_is_unchanged() {
        let filter = CbfFilter::new();
        // h = x_max − x with x well inside and moving left → u_nom pulling further left is safe.
        let c = CbfConstraint::hocbf2(&[-1.0], -4.0 * (-0.2), -(-0.2) + 4.0 * 0.9, 4.0);
        let u_nom = [-3.0];
        let u = filter.filter(&u_nom, &c_slice(&c));
        assert!((u[0] - u_nom[0]).abs() < 1e-9, "safe control changed: {u:?}");
    }

    /// Closed-form single-halfspace projection must agree with the general QP path.
    #[test]
    fn projection_matches_qp() {
        // Force the QP path by adding wide box bounds that never bind.
        let boxed = CbfFilter::with_symmetric_limits(&[1e6, 1e6]);
        let plain = CbfFilter::new();
        let c = CbfConstraint { a: vec![1.0, 2.0], b: 0.5 };
        let u_nom = [3.0, -1.0];
        let via_proj = plain.filter(&u_nom, &c_slice(&c));
        let via_qp = boxed.filter(&u_nom, &c_slice(&c));
        for i in 0..2 {
            assert!((via_proj[i] - via_qp[i]).abs() < 1e-5, "mismatch: {via_proj:?} vs {via_qp:?}");
        }
        // And the projected point sits on the active halfspace boundary aᵀu = b.
        let dot: f64 = c.a.iter().zip(&via_proj).map(|(a, u)| a * u).sum();
        assert!((dot - c.b).abs() < 1e-9, "projection not on boundary: aᵀu = {dot}");
    }

    /// With actuator limits, the box bound can dominate the CBF row; the result stays feasible.
    #[test]
    fn box_bounds_are_respected() {
        let filter = CbfFilter::with_symmetric_limits(&[2.0]);
        // CBF alone would allow u ≤ 5, but |u| ≤ 2 clamps the (feasible) optimum to 2.
        let c = CbfConstraint { a: vec![1.0], b: 5.0 };
        let u = filter.filter(&[10.0], &c_slice(&c));
        assert!(u[0] <= 2.0 + 1e-6 && u[0] >= 2.0 - 1e-3, "box not enforced: {u:?}");
    }

    fn c_slice(c: &CbfConstraint) -> Vec<CbfConstraint> {
        vec![c.clone()]
    }
}
