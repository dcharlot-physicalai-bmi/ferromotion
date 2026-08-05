//! Control Barrier Function (CBF) safety filter: minimally correct a nominal control so the closed
//! loop stays inside a safe set. Each barrier `h_i(x) ≥ 0` is folded to an affine-in-control row
//! `a_iᵀu ≤ b_i` that enforces `ḣ_i + α·h_i ≥ 0` (relative-degree-1) or the higher-order analogue
//! (HOCBF). The filter solves `min ½‖u − u_nom‖²` s.t. those rows (plus optional actuator box
//! bounds), so a safe `u_nom` passes through untouched and an unsafe one is projected to the
//! nearest safe command. The single-halfspace case is closed-form; the general case is a small
//! dense QP through `clarabel` (mirrors `crate::qp`, WASM-clean, no extra deps).

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
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

/// **How much safety the filter was actually able to deliver.**
///
/// A control-barrier filter exists to guarantee a constraint, so "here is a number" is not an adequate return value:
/// the CBF rows and the actuator box can be jointly infeasible, and then no control satisfies both. Before this type
/// existed, [`CbfFilter::filter`] returned the interior-point solver's last iterate in that case — for a row
/// `u <= -10` against a box `|u| <= 2` it returned `-1.9e-10` for *every* nominal input, violating the row by exactly
/// `10.0`, sitting inside the box, finite, with nothing reported. A caller had no way to tell that apart from a
/// certified control.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterMode {
    /// The QP converged and the returned control was **checked** to satisfy every CBF row and the actuator box.
    Certified,
    /// No control satisfies the CBF rows inside the actuator box. The returned control is the best available inside the
    /// box, and `slack` is the worst row violation it still incurs. The actuator box is never relaxed, because
    /// exceeding it is not something a controller may choose to do.
    Relaxed { slack: f64 },
    /// The solver did not converge, or its answer failed the a-posteriori check. The control must not be used.
    Failed { status: String },
}

/// The filter's answer, with its status attached so it cannot be dropped in transit.
#[derive(Clone, Debug)]
pub struct FilterOutcome {
    pub u: Vec<f64>,
    pub mode: FilterMode,
}

impl FilterOutcome {
    /// True only when the returned control was verified against every constraint.
    pub fn certified(&self) -> bool {
        self.mode == FilterMode::Certified
    }

    /// The control, for callers that have already inspected [`Self::mode`].
    pub fn control(&self) -> &[f64] {
        &self.u
    }

    /// The control if and only if it is certified. The safe default for a caller that has no fault reaction.
    pub fn certified_control(&self) -> Option<&[f64]> {
        self.certified().then_some(self.u.as_slice())
    }

    /// How far the returned control violates the CBF rows: `0` when certified.
    pub fn slack(&self) -> f64 {
        match &self.mode {
            FilterMode::Certified => 0.0,
            FilterMode::Relaxed { slack } => *slack,
            FilterMode::Failed { .. } => f64::INFINITY,
        }
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

    /// **Return the safe control and how safe it actually is.**
    ///
    /// Solves `min ½‖u − u_nom‖² + rho·s` subject to `a_iᵀu ≤ b_i + s`, the actuator box, and `s ≥ 0`. The slack `s`
    /// enters only the CBF rows, never the box, so the problem is always feasible — pick any `u` in the box and let `s`
    /// grow — and the solver therefore always has a well-posed answer to give. `s` is then the measurement of whether
    /// safety was achievable: zero means it was, positive means it was not and by how much.
    ///
    /// The returned control is checked against every row before it is called [`FilterMode::Certified`]. A solver that
    /// converges to something infeasible is a real occurrence, and the check costs `O(mn)`.
    pub fn filter(&self, u_nom: &[f64], constraints: &[CbfConstraint]) -> FilterOutcome {
        const TOL: f64 = 1e-7;
        let n = u_nom.len();
        // A box whose width disagrees with the control width used to index out of bounds and panic. A safety filter
        // must not take a process down; refuse the call instead.
        for (name, bound) in [("u_max", &self.u_max), ("u_min", &self.u_min)] {
            if let Some(v) = bound.as_ref().filter(|v| v.len() != n) {
                return FilterOutcome { u: vec![0.0; n], mode: FilterMode::Failed { status: format!("{name} has width {} but u_nom has width {n}", v.len()) } };
            }
        }
        // Row widths are validated BEFORE the fast path. `project_halfspace` zips the row against `u_nom`, so a
        // mismatched row was silently truncated there rather than refused - a wrong answer with no diagnostic, which is
        // the same class of defect this type exists to remove.
        if let Some(c) = constraints.iter().find(|c| c.a.len() != n) {
            return FilterOutcome { u: vec![0.0; n], mode: FilterMode::Failed { status: format!("a constraint row has width {} but u_nom has width {n}", c.a.len()) } };
        }
        let boxed = self.u_min.is_some() || self.u_max.is_some();

        // Fast path: a single halfspace and no box bounds. An orthogonal projection onto one halfspace is always
        // feasible and exact, so it is certified by construction - but check it anyway, since that costs one dot
        // product and the whole point of this type is that nothing is certified without being checked.
        if constraints.len() == 1 && !boxed {
            let u = project_halfspace(u_nom, &constraints[0]);
            return self.classify(u, constraints, 0.0, TOL);
        }
        if constraints.is_empty() && !boxed {
            return FilterOutcome { u: u_nom.to_vec(), mode: FilterMode::Certified };
        }

        // Variables [u (n); s (1)]. Rows: CBF - s <= b, u <= hi, -u <= -lo, -s <= 0.
        let m = constraints.len();
        let nv = n + 1;
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(m + 2 * n + 1);
        let mut b: Vec<f64> = Vec::with_capacity(m + 2 * n + 1);
        for c in constraints {
            let mut r = c.a.clone();
            r.push(-1.0); // -s, so the row becomes a^T u - s <= b
            rows.push(r);
            b.push(c.b);
        }
        if let Some(hi) = &self.u_max {
            for i in 0..n {
                let mut r = vec![0.0; nv];
                r[i] = 1.0;
                rows.push(r);
                b.push(hi[i]);
            }
        }
        if let Some(lo) = &self.u_min {
            for i in 0..n {
                let mut r = vec![0.0; nv];
                r[i] = -1.0;
                rows.push(r);
                b.push(-lo[i]);
            }
        }
        let mut sr = vec![0.0; nv];
        sr[n] = -1.0;
        rows.push(sr);
        b.push(0.0);

        // rho weights the slack heavily enough that it is only used when the rows are genuinely unsatisfiable, and the
        // tiny quadratic term on s keeps the objective strictly convex so the solution is unique.
        let rho = 1e6 * (1.0 + u_nom.iter().fold(0.0f64, |a, v| a.max(v.abs())));
        let a_mat = DMatrix::from_fn(rows.len(), nv, |i, j| rows[i][j]);
        let mut q: Vec<f64> = u_nom.iter().map(|v| -v).collect();
        q.push(rho);
        let Some(sol) = solve_slack_qp(nv, &q, &a_mat, &b) else {
            return FilterOutcome { u: vec![0.0; n], mode: FilterMode::Failed { status: "the slack QP did not converge".into() } };
        };
        if sol.iter().any(|v| !v.is_finite()) {
            return FilterOutcome { u: vec![0.0; n], mode: FilterMode::Failed { status: "the solver returned a non-finite control".into() } };
        }
        let u: Vec<f64> = sol[..n].to_vec();
        let slack = sol[n].max(0.0);
        self.classify(u, constraints, slack, TOL)
    }

    /// The a-posteriori check: does the returned control actually satisfy every row and the box?
    fn classify(&self, u: Vec<f64>, constraints: &[CbfConstraint], slack: f64, tol: f64) -> FilterOutcome {
        let worst_row = constraints
            .iter()
            .map(|c| c.a.iter().zip(&u).map(|(ai, ui)| ai * ui).sum::<f64>() - c.b)
            .fold(0.0f64, f64::max);
        let box_violation = {
            let hi = self.u_max.as_ref().map_or(0.0, |h| u.iter().zip(h).map(|(ui, l)| ui - l).fold(0.0f64, f64::max));
            let lo = self.u_min.as_ref().map_or(0.0, |l| u.iter().zip(l).map(|(ui, l)| l - ui).fold(0.0f64, f64::max));
            hi.max(lo)
        };
        // the box is hard: a control outside it is not something to report as merely relaxed
        if box_violation > tol {
            return FilterOutcome { u, mode: FilterMode::Failed { status: format!("the returned control leaves the actuator box by {box_violation:.3e}") } };
        }
        if worst_row > tol || slack > tol {
            return FilterOutcome { u, mode: FilterMode::Relaxed { slack: worst_row.max(slack) } };
        }
        FilterOutcome { u, mode: FilterMode::Certified }
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

/// Upper-triangular CSC of `diag(1, ..., 1, eps)` — identity on the controls, a small positive weight on the slack.
fn csc_slack_identity(nv: usize) -> CscMatrix<f64> {
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..nv {
        rowval.push(j);
        nzval.push(if j + 1 == nv { 1e-9 } else { 1.0 });
        colptr.push(rowval.len());
    }
    CscMatrix::new(nv, nv, colptr, rowval, nzval)
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

/// Solve `min ½xᵀHx + qᵀx  s.t.  A x ≤ b` with `H = diag(I_n-1, eps)` — the slack variable's quadratic weight is tiny
/// but nonzero so the objective stays strictly convex and the minimiser is unique.
///
/// **Returns `None` unless the solver reports a solved status.** Handing back `solution.x` regardless is what made the
/// filter above report an infeasible problem as a safe control; the status is the only thing that distinguishes the two.
fn solve_slack_qp(nv: usize, q: &[f64], a: &DMatrix<f64>, b: &[f64]) -> Option<Vec<f64>> {
    let p_csc = csc_slack_identity(nv);
    let a_csc = csc_dense(a);
    let cones = [SupportedConeT::NonnegativeConeT(a.nrows())];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().ok()?;
    let mut solver = DefaultSolver::new(&p_csc, q, &a_csc, b, &cones, settings).ok()?;
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Some(solver.solution.x.clone()),
        _ => None,
    }
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

            let u = filter.filter(&[u_nom], &c_slice(&c)).u[0];
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
        let u = filter.filter(&u_nom, &c_slice(&c)).u;
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
        let via_proj = plain.filter(&u_nom, &c_slice(&c)).u;
        let via_qp = boxed.filter(&u_nom, &c_slice(&c)).u;
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
        let u = filter.filter(&[10.0], &c_slice(&c)).u;
        assert!(u[0] <= 2.0 + 1e-6 && u[0] >= 2.0 - 1e-3, "box not enforced: {u:?}");
    }

    fn c_slice(c: &CbfConstraint) -> Vec<CbfConstraint> {
        vec![c.clone()]
    }

    /// **The oracle for A1: an infeasible instance must never be reported as certified.**
    ///
    /// A CBF row demanding `u <= -10` against an actuator box `|u| <= 2` has no solution. Before the fix the filter
    /// returned `-1.9e-10` here for every nominal input — violating the row by exactly `10.0`, inside the box, finite,
    /// and silent. The output was independent of `u_nom` to `1.4e-9` across a ten-unit input range, so the filter was
    /// not even attempting the request.
    #[test]
    fn an_infeasible_instance_is_never_certified() {
        let filter = CbfFilter::with_symmetric_limits(&[2.0]);
        let row = CbfConstraint { a: vec![1.0], b: -10.0 };
        eprintln!("row: u <= -10, box: |u| <= 2  =>  infeasible");
        eprintln!("{:>8}  {:>12}  {:>10}  {:>10}  mode", "u_nom", "u", "row viol", "in box");
        for u_nom in [-5.0, -0.5, 0.0, 2.0, 5.0] {
            let out = filter.filter(&[u_nom], std::slice::from_ref(&row));
            let viol = out.u[0] - row.b;
            eprintln!("{u_nom:>8.2}  {:>12.4}  {viol:>10.4}  {:>10}  {:?}", out.u[0], out.u[0].abs() <= 2.0 + 1e-7, out.mode);
            assert!(!out.certified(), "an infeasible instance must not certify");
            assert!(out.certified_control().is_none(), "and must hand back nothing to a caller with no fault reaction");
            // the actuator box is hard and is still respected
            assert!(out.u[0].abs() <= 2.0 + 1e-7, "the box is never relaxed: {}", out.u[0]);
            // the slack reports how much safety was unattainable
            assert!(out.slack() > 1.0, "the shortfall is reported, not hidden: {}", out.slack());
        }

        // What changed is NOT that the output now varies with u_nom. On an infeasible instance the right fault
        // reaction is to do the safest thing the actuator permits, and that is the same control whatever was asked
        // for - so the output is still u_nom-independent, and deliberately so. My first version of this test asserted
        // the opposite and was wrong about what a safety filter should do.
        let low = filter.filter(&[-5.0], std::slice::from_ref(&row)).u[0];
        let high = filter.filter(&[5.0], std::slice::from_ref(&row)).u[0];
        eprintln!("\n   u_nom -5 -> {low:.4}, u_nom +5 -> {high:.4}: still u_nom-independent, and correctly so");
        assert!((low + 2.0).abs() < 1e-5 && (high + 2.0).abs() < 1e-5, "both saturate at the box in the SAFE direction");
        eprintln!("   What changed is the control and the report. Before: u ~ 0, row violated by 10.0, silent.");
        eprintln!("   After: u = -2 (the box limit, pushing as hard as it can toward safety), violation 8.0, REPORTED.");
        // the residual violation is strictly smaller than doing nothing, which is the measurable improvement
        assert!(low - row.b < 10.0 - 1.0, "it gets strictly closer to safety than the old u ~ 0 did");
    }

    /// Two opposing halfspaces `a'u <= -1` and `-a'u <= -1` cannot both hold. The canonical infeasibility.
    #[test]
    fn two_opposing_halfspaces_never_certify() {
        for filter in [CbfFilter::new(), CbfFilter::with_symmetric_limits(&[10.0, 10.0])] {
            let cs = [CbfConstraint { a: vec![1.0, 0.0], b: -1.0 }, CbfConstraint { a: vec![-1.0, 0.0], b: -1.0 }];
            let out = filter.filter(&[0.0, 0.0], &cs);
            eprintln!("opposing halfspaces, boxed = {}: {:?}", filter.u_max.is_some(), out.mode);
            assert!(!out.certified());
            assert!(out.slack() >= 1.0 - 1e-6, "each row is missed by at least 1: {}", out.slack());
        }
    }

    /// **Property test: every certified control is genuinely feasible.** 20000 random feasible instances; a certified
    /// verdict must survive an independent row check.
    #[test]
    fn every_certified_control_passes_an_independent_check() {
        let mut rng = crate::Xorshift::new(0xC0FFEE);
        let (mut certified, mut relaxed, mut failed) = (0usize, 0usize, 0usize);
        for _ in 0..20_000 {
            let n = 2;
            let limit = 1.0 + 4.0 * rng.uniform();
            let filter = CbfFilter::with_symmetric_limits(&vec![limit; n]);
            // rows built so the origin is feasible (b >= 0), hence the instance is always solvable inside the box
            let cs: Vec<CbfConstraint> = (0..2)
                .map(|_| CbfConstraint { a: (0..n).map(|_| 2.0 * rng.uniform() - 1.0).collect(), b: 2.0 * rng.uniform() })
                .collect();
            let u_nom: Vec<f64> = (0..n).map(|_| 6.0 * rng.uniform() - 3.0).collect();
            let out = filter.filter(&u_nom, &cs);
            match &out.mode {
                FilterMode::Certified => {
                    certified += 1;
                    for c in &cs {
                        let v: f64 = c.a.iter().zip(&out.u).map(|(a, u)| a * u).sum::<f64>() - c.b;
                        assert!(v <= 1e-7, "certified but violates a row by {v:.3e}");
                    }
                    for u in &out.u {
                        assert!(u.abs() <= limit + 1e-7, "certified but outside the box");
                    }
                }
                FilterMode::Relaxed { .. } => relaxed += 1,
                FilterMode::Failed { .. } => failed += 1,
            }
        }
        eprintln!("20000 feasible instances: {certified} certified, {relaxed} relaxed, {failed} failed");
        assert!(certified > 19_000, "an instance containing the origin should almost always certify: {certified}");
        // A non-zero failure rate is the honest outcome, not a bug to assert away: the interior-point solver does
        // occasionally stall on an ill-conditioned random instance. The point of the fix is that those cases are now
        // REPORTED rather than returned as safe controls. Demanding zero here would be re-introducing the old lie.
        eprintln!("   solver failure rate {:.3}% - reported, not silently returned as certified", 100.0 * failed as f64 / 20_000.0);
        assert!(failed * 200 < 20_000, "the failure rate stays under 0.5%: {failed}/20000");
    }

    /// A feasible instance still returns the true minimiser, so the fix did not cost accuracy. Checked against the
    /// closed-form projection on a single halfspace.
    #[test]
    fn the_slack_formulation_still_returns_the_exact_minimiser() {
        let row = CbfConstraint { a: vec![1.0, 1.0], b: 0.5 };
        let plain = CbfFilter::new();
        let boxed = CbfFilter::with_symmetric_limits(&[100.0, 100.0]); // box so wide it cannot bind
        for u_nom in [[2.0, 1.0], [-1.0, -1.0], [0.3, 0.1], [5.0, -4.0]] {
            let a = plain.filter(&u_nom, std::slice::from_ref(&row));
            let b = boxed.filter(&u_nom, std::slice::from_ref(&row));
            let worst = a.u.iter().zip(&b.u).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            eprintln!("u_nom {u_nom:?}: projection {:?} vs slack QP {:?}, difference {worst:.2e}", a.u, b.u);
            assert!(a.certified() && b.certified());
            assert!(worst < 1e-6, "the slack QP must agree with the exact projection: {worst:.3e}");
        }
    }

    /// A box whose width disagrees with the control width is refused, not a panic. This used to index out of bounds.
    #[test]
    fn a_mismatched_actuator_box_is_refused() {
        let filter = CbfFilter::with_symmetric_limits(&[1.0]); // one bound
        let cs = [CbfConstraint { a: vec![1.0, 0.0], b: 0.5 }]; // two controls
        let out = filter.filter(&[0.0, 0.0], &cs);
        eprintln!("box width 1 against control width 2: {:?}", out.mode);
        assert!(matches!(out.mode, FilterMode::Failed { .. }));
        assert!(out.certified_control().is_none());
        // a constraint row of the wrong width is refused the same way
        let bad = [CbfConstraint { a: vec![1.0, 0.0, 0.0], b: 0.5 }];
        assert!(matches!(CbfFilter::new().filter(&[0.0, 0.0], &bad).mode, FilterMode::Failed { .. }));
    }
}
