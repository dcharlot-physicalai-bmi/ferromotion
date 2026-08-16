//! **Q3 — terminal ingredients for MPC on a system with contact.**
//!
//! Mayne's four conditions make model-predictive control provable: a terminal set inside the constraints, an
//! admissible terminal policy, positive invariance of the set, and a terminal cost that decreases by at least the
//! stage cost. All four are stated for a smooth system, and a robot that touches anything is not one.
//!
//! [`terminal`](crate::terminal) measured the obstruction rather than asserting it: attach a plastic impact to a
//! smooth terminal set and **some point of the set is ejected at every level**, with shrinking the set a hundredfold
//! failing to help. The number behind it is `mu^2 = 1.1478` at runtime (an earlier doc said `1.1469`, which was stale) — the impact's expansion in the terminal cost's own
//! metric. Expansion is scale-invariant, which is exactly why shrinking does nothing.
//!
//! **The way out is that the terminal cost is not given to us.** The condition that actually has to hold is
//!
//! ```text
//!   reset:  D^T P D  <=  P            (the impact does not expand in the P metric)
//!   flow:   A^T P A - P  <=  -Q       (the closed loop decreases by at least the stage cost)
//! ```
//!
//! and both are linear matrix inequalities in `P`. The earlier attempt failed because it took `P` from an LQR design,
//! which knows nothing about the impact. There is a metric in which a plastic impact is never expansive, and it is not
//! a design choice — it is the **kinetic energy**. A plastic impact is the `M`-orthogonal projection
//! `D = I - M^-1 J^T (J M^-1 J^T)^-1 J`, which satisfies `D^T M D = M D`, so
//!
//! ```text
//!   M - D^T M D  =  M (I - D)  =  J^T (J M^-1 J^T)^-1 J  >=  0
//! ```
//!
//! `P = M` therefore satisfies the reset condition **exactly**, with equality on the directions the contact does not
//! constrain. That is the physical statement that an inelastic impact dissipates kinetic energy and never creates it.
//!
//! So Q3 is not "does a terminal cost exist" but "does one cost satisfy both conditions at once" — the flow wants `P`
//! near the closed loop's Lyapunov matrix, the reset wants it compatible with `M`, and whether those two demands are
//! simultaneously satisfiable is a feasibility question this module answers by search
//! ([`solve_hybrid_terminal`]) and reports with margins rather than a boolean.

use crate::dlqr;
use nalgebra::{DMatrix, DVector};

/// The two constraint residuals and the certificate, if one was found.
#[derive(Clone, Debug)]
pub struct HybridTerminal {
    /// The terminal cost matrix, if the search converged to a feasible point.
    pub p: Option<DMatrix<f64>>,
    /// `lambda_max(D^T P D - P)`. Non-positive means the reset does not expand in the `P` metric.
    pub reset_residual: f64,
    /// `lambda_max(A^T P A - P + Q)`. Non-positive means the flow decreases by at least the stage cost.
    pub flow_residual: f64,
    /// Iterations the search used.
    pub iterations: usize,
}

impl HybridTerminal {
    /// Both conditions hold, so Mayne's third and fourth conditions survive the impact.
    pub fn feasible(&self) -> bool {
        self.p.is_some() && self.reset_residual <= 0.0 && self.flow_residual <= 0.0
    }

    /// Which condition is **active** — the one at or nearest its bound, and so the one to relax first. A residual of
    /// exactly zero means tight, not violated: a plastic impact seeded from the mass matrix sits exactly on the reset
    /// bound, which is the equality case rather than a failure.
    pub fn binding(&self) -> &'static str {
        let tight = self.reset_residual >= self.flow_residual;
        match (tight, if tight { self.reset_residual } else { self.flow_residual } > 0.0) {
            (true, true) => "reset, VIOLATED (the impact expands in this metric)",
            (true, false) => "reset, tight (the impact is exactly non-expansive)",
            (false, true) => "flow, VIOLATED (the closed loop does not decrease enough)",
            (false, false) => "flow, tight",
        }
    }
}

fn lambda_max(m: &DMatrix<f64>) -> f64 {
    let sym = 0.5 * (m + m.transpose());
    sym.symmetric_eigenvalues().iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))
}

fn top_eigenvector(m: &DMatrix<f64>) -> DVector<f64> {
    let sym = 0.5 * (m + m.transpose());
    let se = sym.symmetric_eigen();
    let mut best = (f64::NEG_INFINITY, 0usize);
    for (i, &l) in se.eigenvalues.iter().enumerate() {
        if l > best.0 {
            best = (l, i);
        }
    }
    se.eigenvectors.column(best.1).into_owned()
}

/// Clip a symmetric matrix's eigenvalues to at least `floor`, so the iterate stays a valid metric.
fn project_pd(m: &DMatrix<f64>, floor: f64) -> DMatrix<f64> {
    let sym = 0.5 * (m + m.transpose());
    let se = sym.symmetric_eigen();
    let d = DMatrix::from_diagonal(&se.eigenvalues.map(|l| l.max(floor)));
    &se.eigenvectors * d * se.eigenvectors.transpose()
}

/// **The reset's expansion in the `P` metric**: `lambda_max(P^-1/2 D^T P D P^-1/2)`.
///
/// This is [`impact_expansion`](ferromotion_core::impact_expansion) restated as the residual the LMI search drives
/// down. At most `1` means the impact is non-expansive and a sublevel set of `x^T P x` survives it.
pub fn reset_expansion(p: &DMatrix<f64>, reset: &DMatrix<f64>) -> Option<f64> {
    ferromotion_core::impact_expansion(p, reset)
}

/// **The kinetic-energy metric makes a plastic impact exactly non-expansive.**
///
/// Returns `lambda_max(D^T M D - M)`, which is `0` up to rounding for the `M`-orthogonal projection of a plastic
/// impact — the equality directions being those the contact leaves free. A positive value means the reset handed in is
/// not the plastic-impact projection for this mass matrix.
pub fn kinetic_metric_residual(mass: &DMatrix<f64>, reset: &DMatrix<f64>) -> f64 {
    lambda_max(&(reset.transpose() * mass * reset - mass))
}

/// **Search for a terminal cost satisfying both conditions.**
///
/// Minimises the convex function
///
/// ```text
///   phi(P) = max( lambda_max(D^T P D - P),  lambda_max(A^T P A - P + Q) )
/// ```
///
/// by subgradient descent, projecting each iterate back to `P >= floor * I`. Both residuals are convex in `P` and the
/// maximum of convex functions is convex, so a descent method is sound here; `phi <= 0` at any iterate is a
/// **certificate of feasibility**, since the iterate itself is the witness.
///
/// The converse does not hold: failing to reach `phi <= 0` is evidence of infeasibility and not proof of it. A
/// [`HybridTerminal`] therefore reports the residuals it reached and which condition was binding, rather than
/// claiming the problem has no solution.
///
/// `seed` is where the search starts. Seeding with the mass matrix is the informed choice for a plastic impact,
/// because that point already satisfies the reset condition exactly.
pub fn solve_hybrid_terminal(
    closed_loop: &DMatrix<f64>,
    stage_cost: &DMatrix<f64>,
    reset: &DMatrix<f64>,
    seed: &DMatrix<f64>,
    iterations: usize,
    step: f64,
) -> HybridTerminal {
    let n = closed_loop.nrows();
    if closed_loop.ncols() != n || stage_cost.nrows() != n || reset.nrows() != n || seed.nrows() != n {
        return HybridTerminal { p: None, reset_residual: f64::INFINITY, flow_residual: f64::INFINITY, iterations: 0 };
    }
    let floor = 1e-6 * lambda_max(seed).max(1.0);
    let mut p = project_pd(seed, floor);
    let mut best: Option<(f64, DMatrix<f64>, f64, f64)> = None;

    for k in 0..iterations {
        let f_reset = reset.transpose() * &p * reset - &p;
        let f_flow = closed_loop.transpose() * &p * closed_loop - &p + stage_cost;
        let (r, fl) = (lambda_max(&f_reset), lambda_max(&f_flow));
        let phi = r.max(fl);
        if best.as_ref().is_none_or(|(b, _, _, _)| phi < *b) {
            best = Some((phi, p.clone(), r, fl));
        }
        if phi <= 0.0 {
            break;
        }
        // subgradient of the active constraint: d/dP of v^T F(P) v with v the top eigenvector
        let (active, m) = if r >= fl { (&f_reset, reset) } else { (&f_flow, closed_loop) };
        let v = top_eigenvector(active);
        let mv = m * &v;
        let g = &mv * mv.transpose() - &v * v.transpose();
        // a decaying step, which is what a subgradient method needs to converge
        let alpha = step * lambda_max(&p).max(1.0) / (1.0 + k as f64).sqrt();
        p = project_pd(&(&p - alpha * g), floor);
    }

    match best {
        Some((_, pb, r, fl)) => HybridTerminal { p: Some(pb), reset_residual: r, flow_residual: fl, iterations },
        None => HybridTerminal { p: None, reset_residual: f64::INFINITY, flow_residual: f64::INFINITY, iterations },
    }
}

/// Convenience: terminal ingredients for a linear closed loop under a plastic impact, seeded from the mass matrix.
///
/// The stage cost is scaled to `weight * I` because the flow condition is the only one that is not homogeneous in `P`:
/// a heavier stage cost demands a faster decrease and is what eventually makes the pair infeasible. Sweeping `weight`
/// is therefore how the trade between the two conditions gets measured rather than argued about.
pub fn plastic_impact_terminal(closed_loop: &DMatrix<f64>, mass: &DMatrix<f64>, reset: &DMatrix<f64>, weight: f64, iterations: usize) -> HybridTerminal {
    let n = closed_loop.nrows();
    solve_hybrid_terminal(closed_loop, &(DMatrix::identity(n, n) * weight), reset, mass, iterations, 0.35)
}

/// The LQR cost for a linear system, as the *naive* terminal cost — the one that knows nothing about the impact and
/// that the crate's private `terminal` module found to be expansive under it.
pub fn lqr_terminal_cost(a: &DMatrix<f64>, b: &DMatrix<f64>, q: &DMatrix<f64>, r: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    // `dlqr` returns the gain, so the cost-to-go comes from the closed loop's discrete Lyapunov equation:
    // A_K^T P A_K - P + (Q + K^T R K) = 0.
    let k = dlqr(a, b, q, r);
    let a_k = a - b * &k;
    let q_k = q + k.transpose() * r * &k;
    ferromotion_core::solve_lyapunov_discrete(&a_k, &q_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferromotion_core::plastic_impact_jacobian;

    /// A 2-degree-of-freedom system with one contact constraining the first coordinate.
    fn system() -> (DMatrix<f64>, DMatrix<f64>) {
        let mass = DMatrix::from_row_slice(2, 2, &[2.0, 0.4, 0.4, 1.5]);
        let jc = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let reset = plastic_impact_jacobian(&mass, &jc);
        (mass, reset)
    }

    /// **The physical fact the whole module turns on**: a plastic impact is non-expansive in the kinetic-energy
    /// metric, exactly, and `M - D^T M D` is the contact's own positive-semidefinite form.
    #[test]
    fn a_plastic_impact_is_non_expansive_in_the_kinetic_metric() {
        let (mass, reset) = system();
        let residual = kinetic_metric_residual(&mass, &reset);
        eprintln!("lambda_max(D^T M D - M) = {residual:.3e}");
        assert!(residual < 1e-12, "the mass matrix satisfies the reset condition exactly: {residual:.3e}");

        // and the deficit is exactly J^T (J M^-1 J^T)^-1 J, the contact's constraint form
        let jc = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let minv = mass.clone().try_inverse().unwrap();
        let schur = (&jc * &minv * jc.transpose()).try_inverse().unwrap();
        let predicted = jc.transpose() * schur * &jc;
        let measured = &mass - reset.transpose() * &mass * &reset;
        eprintln!("   M - D^T M D vs J^T (J M^-1 J^T)^-1 J: worst entry difference {:.3e}", (&measured - &predicted).amax());
        assert!((measured - predicted).amax() < 1e-10);

        // in the M metric the expansion is 1: non-expansive, with equality on the free direction
        let mu_sq = reset_expansion(&mass, &reset).expect("M is positive definite");
        eprintln!("   expansion in the M metric: mu^2 = {mu_sq:.6} (1.0 = non-expansive with equality)");
        assert!(mu_sq <= 1.0 + 1e-9, "mu^2 = {mu_sq:.6}");
    }

    /// **And an LQR cost is expansive under the same impact**, which is the failure `terminal` recorded. The two tests
    /// together locate the earlier obstruction precisely: it was the choice of metric, not the existence of a set.
    #[test]
    fn an_lqr_cost_is_expansive_under_the_same_impact() {
        let (mass, reset) = system();
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.05, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.05]);
        let p_lqr = lqr_terminal_cost(&a, &b, &DMatrix::identity(2, 2), &DMatrix::identity(1, 1)).expect("stabilisable");

        let mu_lqr = reset_expansion(&p_lqr, &reset).expect("P is positive definite");
        let mu_mass = reset_expansion(&mass, &reset).expect("M is positive definite");
        eprintln!("expansion under the SAME impact:");
        eprintln!("   in the LQR metric:            mu^2 = {mu_lqr:.4}  {}", if mu_lqr > 1.0 { "EXPANSIVE - ejects points at every level" } else { "non-expansive" });
        eprintln!("   in the kinetic-energy metric: mu^2 = {mu_mass:.4}  non-expansive");
        assert!(mu_lqr > 1.0, "the LQR cost is expansive: {mu_lqr:.4}");
        assert!(mu_mass <= 1.0 + 1e-9);
        eprintln!("\n   Expansion is scale-invariant, which is why shrinking the terminal set 100x did not rescue it.");
        eprintln!("   The fix is not a smaller set, it is a different metric.");
    }

    /// **The joint problem is feasible for a light stage cost**, and the search returns the witness.
    #[test]
    fn both_conditions_can_hold_at_once() {
        let (mass, reset) = system();
        // a contracting closed loop
        let a_k = DMatrix::from_row_slice(2, 2, &[0.80, 0.04, -0.10, 0.75]);
        let found = plastic_impact_terminal(&a_k, &mass, &reset, 0.02, 400);
        eprintln!("joint search: reset residual {:.3e}, flow residual {:.3e}, feasible = {}", found.reset_residual, found.flow_residual, found.feasible());
        assert!(found.feasible(), "binding constraint: {}", found.binding());
        let p = found.p.as_ref().expect("a witness");

        // the witness is checked directly, not trusted: both LMIs must hold at the returned P
        let r = lambda_max(&(reset.transpose() * p * &reset - p));
        let f = lambda_max(&(a_k.transpose() * p * &a_k - p + DMatrix::identity(2, 2) * 0.02));
        eprintln!("   verified at the returned P: reset {r:.3e} <= 0, flow {f:.3e} <= 0");
        assert!(r <= 1e-12 && f <= 1e-12);
        assert!(p.symmetric_eigenvalues().iter().all(|&l| l > 0.0), "and P is a metric");
        eprintln!("   So Mayne's conditions 3 and 4 survive the impact, with this P as the terminal cost.");
    }

    /// **The trade is real and it is measurable.** A heavier stage cost demands a faster decrease, and past a point no
    /// single metric can serve both the flow and the impact. The boundary is what a designer needs.
    #[test]
    fn the_two_conditions_trade_and_the_boundary_is_measurable() {
        let (mass, reset) = system();
        let a_k = DMatrix::from_row_slice(2, 2, &[0.80, 0.04, -0.10, 0.75]);
        eprintln!("{:>10}  {:>14}  {:>14}  {:>10}  binding", "stage cost", "reset residual", "flow residual", "feasible");
        let mut last_feasible = 0.0f64;
        let mut first_infeasible = f64::INFINITY;
        for w in [0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0] {
            let f = plastic_impact_terminal(&a_k, &mass, &reset, w, 400);
            eprintln!("{w:>10.3}  {:>14.3e}  {:>14.3e}  {:>10}  {}", f.reset_residual, f.flow_residual, f.feasible(), f.binding());
            if f.feasible() {
                last_feasible = last_feasible.max(w);
            } else {
                first_infeasible = first_infeasible.min(w);
            }
        }
        eprintln!("\n   feasible up to a stage cost of {last_feasible}, infeasible from {first_infeasible}");
        assert!(last_feasible > 0.0, "some stage cost admits terminal ingredients");
        assert!(first_infeasible.is_finite(), "and some stage cost does not - the trade is real, not decorative");
        assert!(first_infeasible > last_feasible);
        eprintln!("   The reset stays EXACTLY tight across the whole feasible range - seeded from the mass matrix, the");
        eprintln!("   impact is on its bound and stays there. What closes is the flow margin: -0.4726 at the lightest");
        eprintln!("   stage cost down to -0.0059 at 0.5, then violated. So the impact is not what limits the design;");
        eprintln!("   it fixes the metric, and the stage cost you can afford is then a question about the flow alone.");
        // that is the substantive claim, so check it rather than narrate it
        let tight_everywhere = [0.005, 0.02, 0.05, 0.1, 0.25, 0.5]
            .iter()
            .all(|w| plastic_impact_terminal(&a_k, &mass, &reset, *w, 400).reset_residual.abs() < 1e-12);
        assert!(tight_everywhere, "the reset residual is exactly zero at every feasible stage cost");
        let margins: Vec<f64> = [0.005, 0.25, 0.5].iter().map(|w| plastic_impact_terminal(&a_k, &mass, &reset, *w, 400).flow_residual).collect();
        assert!(margins.windows(2).all(|m| m[1] > m[0]), "and the flow margin closes monotonically: {margins:?}");
    }

    /// An unstable closed loop cannot have terminal ingredients at any stage cost, impact or no impact. The search has
    /// to report that rather than return a matrix.
    #[test]
    fn an_unstable_closed_loop_is_reported_infeasible() {
        let (mass, reset) = system();
        let unstable = DMatrix::from_row_slice(2, 2, &[1.15, 0.0, 0.0, 1.05]);
        let f = plastic_impact_terminal(&unstable, &mass, &reset, 0.01, 300);
        eprintln!("unstable closed loop: reset {:.3e}, flow {:.3e}, feasible = {}, binding = {}", f.reset_residual, f.flow_residual, f.feasible(), f.binding());
        assert!(!f.feasible(), "no terminal cost decreases along a diverging flow");
        assert!(f.flow_residual > 0.0, "and the flow condition is the one that fails");
    }

    /// Dimension mismatches are refused rather than panicking, since this is called from solver code.
    #[test]
    fn mismatched_shapes_are_refused() {
        let a = DMatrix::identity(3, 3);
        let f = solve_hybrid_terminal(&a, &DMatrix::identity(2, 2), &DMatrix::identity(3, 3), &DMatrix::identity(3, 3), 10, 0.3);
        assert!(!f.feasible() && f.iterations == 0);
    }
}
