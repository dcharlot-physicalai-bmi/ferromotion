//! **MPC terminal ingredients** — the four conditions that turn a receding-horizon optimiser into a
//! stabilising controller, and a demonstration of why contact does not have them.
//!
//! Model-predictive control optimises over a finite horizon and throws most of the answer away. Nothing about
//! that is stabilising on its own: a controller that is optimal over the next second can walk a system into a
//! state from which the next second is worse, forever. Mayne's theorem says what closes the gap, and it is
//! four conditions on a **terminal set** `X_f`, a **terminal cost** `F`, and a **local controller** `κ_f`:
//!
//! 1. `X_f ⊆ X` and `0 ∈ X_f` — the target is reachable and admissible;
//! 2. `κ_f(x) ∈ U` for all `x ∈ X_f` — the local controller is implementable;
//! 3. **positive invariance**, `f(x, κ_f(x)) ∈ X_f` — once inside, never leaves;
//! 4. the terminal cost is a **local control Lyapunov function**,
//!    `F(f(x, κ_f(x))) − F(x) ≤ −ℓ(x, κ_f(x))`.
//!
//! Given those, the optimal value function is a Lyapunov function for the closed loop and feasibility is
//! recursive: the horizon can always be extended by the terminal controller, so a feasible problem stays
//! feasible. [`check_terminal_ingredients`] verifies all four by sampling the terminal set and reports which
//! one fails and by how much, because "my MPC is stable" is otherwise an assumption.
//!
//! # The contact gap
//!
//! For a linear system these ingredients are constructive: take `F = xᵀPx` from the Riccati equation and
//! `κ_f = −Kx`, and condition 4 holds with *equality*. For hybrid and contact dynamics they are essentially
//! unconstructed, and the reason is not that nobody has tried. The impact map is **expansive**, so a smooth
//! sublevel set that is invariant under the continuous flow is thrown open by the reset; the mode sequence is
//! unknown ahead of time, so there is no single `κ_f`; complementarity makes the optimisation a nonconvex
//! program where a stationary point is not a global one; and the value function is discontinuous across
//! modes, so it cannot be a smooth Lyapunov function spanning a guard.
//!
//! [`contact_gap_witness`] makes the first of those concrete: it returns the point of a terminal set that an
//! expansive reset ejects, so the failure is a number rather than a caveat. That is the honest state of
//! contact-implicit MPC — deployed, effective, and carrying no stability or recursive-feasibility guarantee.

/// One-step dynamics, `(state, input) → next state`.
pub type Dynamics<'a> = &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>;
/// A state-feedback policy.
pub type Policy<'a> = &'a dyn Fn(&[f64]) -> Vec<f64>;
/// A scalar function of the state, such as a terminal cost.
pub type StateFn<'a> = &'a dyn Fn(&[f64]) -> f64;
/// A stage cost, `(state, input) → cost`.
pub type StageCost<'a> = &'a dyn Fn(&[f64], &[f64]) -> f64;
/// A membership test on inputs.
pub type InputSet<'a> = &'a dyn Fn(&[f64]) -> bool;

/// The outcome of checking the four terminal conditions, with the violations quantified.
#[derive(Clone, Debug)]
pub struct TerminalCheck {
    /// `0 ∈ X_f`: the origin is in the terminal set.
    pub contains_origin: bool,
    /// `κ_f(x) ∈ U` throughout `X_f`.
    pub input_admissible: bool,
    /// `f(x, κ_f(x)) ∈ X_f` throughout `X_f`.
    pub positively_invariant: bool,
    /// `F(f(x, κ_f(x))) − F(x) ≤ −ℓ(x, κ_f(x))` throughout `X_f`.
    pub cost_decreases: bool,
    /// Worst `F(next) − c` over the samples: how far outside the terminal set the successor of an interior
    /// point lands. Positive means invariance is broken.
    pub worst_invariance_violation: f64,
    /// Worst `F(next) − F(x) + ℓ` over the samples. Positive means the Lyapunov descent is broken.
    pub worst_descent_violation: f64,
    /// How many sample points were used.
    pub samples: usize,
}

impl TerminalCheck {
    /// Whether all four conditions held on the samples. Sampling is necessary rather than sufficient — a
    /// pass is evidence, not a proof — which is exactly the honest boundary of a numerical check.
    pub fn valid(&self) -> bool {
        self.contains_origin && self.input_admissible && self.positively_invariant && self.cost_decreases
    }

    /// The first condition that failed, for reporting.
    pub fn first_failure(&self) -> Option<&'static str> {
        if !self.contains_origin {
            Some("the terminal set does not contain the origin")
        } else if !self.input_admissible {
            Some("the terminal controller demands an inadmissible input")
        } else if !self.positively_invariant {
            Some("the terminal set is not positively invariant")
        } else if !self.cost_decreases {
            Some("the terminal cost is not a local control Lyapunov function")
        } else {
            None
        }
    }
}

/// Check Mayne's four terminal conditions on a sampled shell of the terminal set `X_f = {x : F(x) ≤ level}`.
///
/// * `f` — the one-step dynamics, `(x, u) → x⁺`.
/// * `kappa` — the terminal controller on `X_f`.
/// * `terminal_cost` — `F`.
/// * `stage_cost` — `ℓ(x, u)`.
/// * `input_ok` — membership test for `U`.
/// * `directions` — unit directions to probe; the set is sampled along each at several radii, so coverage
///   scales with how many are supplied.
///
/// Sampling along rays out to the boundary of the sublevel set is what makes this cheap and what makes it
/// honest: the binding constraint on all four conditions is at the boundary, so that is where the samples
/// are concentrated.
#[allow(clippy::too_many_arguments)]
pub fn check_terminal_ingredients(f: Dynamics, kappa: Policy, terminal_cost: StateFn, stage_cost: StageCost, input_ok: InputSet, level: f64, directions: &[Vec<f64>]) -> TerminalCheck {
    let n = directions.first().map(|d| d.len()).unwrap_or(0);
    let origin = vec![0.0; n];
    let mut out = TerminalCheck {
        contains_origin: terminal_cost(&origin) <= level,
        input_admissible: true,
        positively_invariant: true,
        cost_decreases: true,
        worst_invariance_violation: f64::NEG_INFINITY,
        worst_descent_violation: f64::NEG_INFINITY,
        samples: 0,
    };

    for dir in directions {
        // scale the direction to the boundary of the sublevel set, then sample inwards from it
        let unit: Vec<f64> = {
            let nrm = dir.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
            dir.iter().map(|v| v / nrm).collect()
        };
        let at = |r: f64| -> Vec<f64> { unit.iter().map(|v| v * r).collect() };
        // F is a positive quadratic in the intended use, so bisect for the radius where F = level
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        while terminal_cost(&at(hi)) < level && hi < 1e6 {
            hi *= 2.0;
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if terminal_cost(&at(mid)) <= level {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        for frac in [1.0f64, 0.75, 0.5, 0.25, 0.05] {
            let x = at(lo * frac);
            let u = kappa(&x);
            let next = f(&x, &u);
            out.samples += 1;
            if !input_ok(&u) {
                out.input_admissible = false;
            }
            let inv = terminal_cost(&next) - level;
            out.worst_invariance_violation = out.worst_invariance_violation.max(inv);
            if inv > 1e-9 {
                out.positively_invariant = false;
            }
            let desc = terminal_cost(&next) - terminal_cost(&x) + stage_cost(&x, &u);
            out.worst_descent_violation = out.worst_descent_violation.max(desc);
            if desc > 1e-9 {
                out.cost_decreases = false;
            }
        }
    }
    out
}

/// **A witness that an expansive reset breaks positive invariance**, whatever smooth terminal set is chosen.
///
/// Given a terminal cost `F`, its level, and a reset map, this searches the terminal set for the point whose
/// image lands furthest outside it, and returns `(point, F(reset(point)) − level)` when one exists. A
/// positive second component is a concrete counterexample to condition 3 — the reset takes an interior point
/// out of the set, so the horizon cannot be closed by any local controller acting after the impact.
///
/// This is the mechanical reason the certified-MPC recipe does not transfer to contact: the ingredient list
/// asks for a set the dynamics cannot leave, and an impact is precisely a map that leaves it.
pub fn contact_gap_witness(terminal_cost: StateFn, reset: Policy, level: f64, directions: &[Vec<f64>]) -> Option<(Vec<f64>, f64)> {
    let mut worst: Option<(Vec<f64>, f64)> = None;
    for dir in directions {
        let nrm = dir.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let unit: Vec<f64> = dir.iter().map(|v| v / nrm).collect();
        let at = |r: f64| -> Vec<f64> { unit.iter().map(|v| v * r).collect() };
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        while terminal_cost(&at(hi)) < level && hi < 1e6 {
            hi *= 2.0;
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if terminal_cost(&at(mid)) <= level {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let x = at(lo);
        let excess = terminal_cost(&reset(&x)) - level;
        if worst.as_ref().is_none_or(|(_, e)| excess > *e) {
            worst = Some((x, excess));
        }
    }
    worst.filter(|(_, e)| *e > 0.0)
}

/// Unit directions spread over a `dim`-dimensional sphere, deterministically, for use as the sample set.
pub fn probe_directions(dim: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|k| {
            // a low-discrepancy spread: distinct irrational frequencies per coordinate
            let t = (k as f64 + 1.0) * 0.618_033_988_749_895;
            (0..dim).map(|i| ((t * (i as f64 + 1.7)) * std::f64::consts::TAU).sin() + 0.3 * ((t * (i as f64 + 2.9)) * std::f64::consts::TAU).cos()).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// A double integrator with an LQR terminal cost: the case where the ingredients are constructive.
    struct Lqr {
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        p: DMatrix<f64>,
        k: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
    }

    fn lqr_setup() -> Lqr {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::identity(1, 1);
        let k = crate::dlqr(&a, &b, &q, &r);
        // P from the same Riccati recursion the gain came from
        let mut p = q.clone();
        for _ in 0..20_000 {
            let btp = b.transpose() * &p;
            let kk = (&r + &btp * &b).try_inverse().unwrap() * (&btp * &a);
            let acl = &a - &b * &kk;
            let pn = acl.transpose() * &p * &acl + &q + kk.transpose() * &r * &kk;
            if (&pn - &p).norm() < 1e-14 {
                p = pn;
                break;
            }
            p = pn;
        }
        Lqr { a, b, p, k, q, r }
    }

    /// **The constructive case.** For a linear system with an LQR terminal cost the Lyapunov condition holds
    /// with *equality*, not slack — that is the Riccati identity — so the worst descent violation should be
    /// zero to round-off. Anything else means the ingredients do not match the system they were built for.
    #[test]
    fn lqr_terminal_ingredients_satisfy_all_four_conditions() {
        let Lqr { a, b, p, k, q, r } = lqr_setup();
        let dv = |x: &[f64]| DVector::from_row_slice(x);
        let f = |x: &[f64], u: &[f64]| (&a * dv(x) + &b * dv(u)).iter().cloned().collect::<Vec<f64>>();
        let kappa = |x: &[f64]| (-(&k) * dv(x)).iter().cloned().collect::<Vec<f64>>();
        let fcost = |x: &[f64]| (dv(x).transpose() * &p * dv(x))[0];
        let ell = |x: &[f64], u: &[f64]| (dv(x).transpose() * &q * dv(x))[0] + (dv(u).transpose() * &r * dv(u))[0];
        let ok = |_: &[f64]| true;

        let dirs = probe_directions(2, 24);
        let c = check_terminal_ingredients(&f, &kappa, &fcost, &ell, &ok, 1.0, &dirs);
        eprintln!("LQR ingredients over {} samples: invariance slack {:.2e}, descent slack {:.2e}", c.samples, c.worst_invariance_violation, c.worst_descent_violation);
        assert!(c.valid(), "the LQR construction must satisfy all four: {:?}", c.first_failure());
        // the Riccati identity makes condition 4 an equality, so the violation sits at zero from below
        assert!(c.worst_descent_violation.abs() < 1e-9, "the Riccati identity should give equality, slack {:.2e}", c.worst_descent_violation);
        // and invariance is strict, since the closed loop contracts the cost
        assert!(c.worst_invariance_violation < -1e-6, "the terminal set should be strictly invariant, got {:.2e}", c.worst_invariance_violation);
    }

    /// Tighten the input set until the terminal controller cannot be implemented, and the check must name
    /// that condition rather than reporting a vague failure. A terminal set is only as valid as the authority
    /// available inside it.
    #[test]
    fn an_input_bound_that_bites_is_reported_as_the_failing_condition() {
        let Lqr { a, b, p, k, q, r } = lqr_setup();
        let dv = |x: &[f64]| DVector::from_row_slice(x);
        let f = |x: &[f64], u: &[f64]| (&a * dv(x) + &b * dv(u)).iter().cloned().collect::<Vec<f64>>();
        let kappa = |x: &[f64]| (-(&k) * dv(x)).iter().cloned().collect::<Vec<f64>>();
        let fcost = |x: &[f64]| (dv(x).transpose() * &p * dv(x))[0];
        let ell = |x: &[f64], u: &[f64]| (dv(x).transpose() * &q * dv(x))[0] + (dv(u).transpose() * &r * dv(u))[0];
        let dirs = probe_directions(2, 24);

        // generous bound: fine. tight bound: the controller saturates inside the terminal set.
        let loose = check_terminal_ingredients(&f, &kappa, &fcost, &ell, &|_| true, 1.0, &dirs);
        let tight = check_terminal_ingredients(&f, &kappa, &fcost, &ell, &|u: &[f64]| u[0].abs() <= 0.05, 1.0, &dirs);
        eprintln!("loose input set: {:?}; tight input set: {:?}", loose.first_failure(), tight.first_failure());
        assert!(loose.valid());
        assert!(!tight.valid());
        assert_eq!(tight.first_failure(), Some("the terminal controller demands an inadmissible input"));
        // shrinking the terminal set restores admissibility, which is the standard remedy and worth pinning
        let smaller = check_terminal_ingredients(&f, &kappa, &fcost, &ell, &|u: &[f64]| u[0].abs() <= 0.05, 0.002, &dirs);
        assert!(smaller.valid(), "a small enough terminal set should fit inside the input bound: {:?}", smaller.first_failure());
    }

    /// A terminal controller that does not stabilise breaks invariance and descent together, and both are
    /// reported with a magnitude rather than a boolean.
    #[test]
    fn a_non_stabilising_terminal_controller_breaks_invariance() {
        let Lqr { a, b, p, q, r, .. } = lqr_setup();
        let dv = |x: &[f64]| DVector::from_row_slice(x);
        let f = |x: &[f64], u: &[f64]| (&a * dv(x) + &b * dv(u)).iter().cloned().collect::<Vec<f64>>();
        let zero_input = |_: &[f64]| vec![0.0]; // do nothing: a double integrator coasts out of any set
        let fcost = |x: &[f64]| (dv(x).transpose() * &p * dv(x))[0];
        let ell = |x: &[f64], u: &[f64]| (dv(x).transpose() * &q * dv(x))[0] + (dv(u).transpose() * &r * dv(u))[0];
        let c = check_terminal_ingredients(&f, &zero_input, &fcost, &ell, &|_| true, 1.0, &probe_directions(2, 24));
        eprintln!("coasting terminal controller: {:?}, invariance violation {:.4}, descent violation {:.4}", c.first_failure(), c.worst_invariance_violation, c.worst_descent_violation);
        assert!(!c.valid());
        assert!(c.worst_invariance_violation > 0.0 || c.worst_descent_violation > 0.0, "one of the two must be violated");
    }

    /// **The contact gap, as a number.** Attach an expansive reset — a plastic impact reversing and
    /// amplifying velocity, which is what a footfall does to a transverse perturbation — and no sublevel set
    /// of a smooth terminal cost survives it. The witness is the specific state that gets ejected.
    #[test]
    fn an_expansive_reset_ejects_every_terminal_set() {
        let Lqr { p, .. } = lqr_setup();
        let dv = |x: &[f64]| DVector::from_row_slice(x);
        let fcost = |x: &[f64]| (dv(x).transpose() * &p * dv(x))[0];
        // velocity is reversed and amplified at the guard: the essential feature of an impact
        let reset = |x: &[f64]| vec![x[0], -1.4 * x[1]];
        let dirs = probe_directions(2, 48);

        // whatever level is chosen, there is a point the reset ejects
        for &level in &[0.01f64, 1.0, 100.0] {
            let w = contact_gap_witness(&fcost, &reset, level, &dirs).expect("an expansive reset must eject some point");
            eprintln!("terminal level {level:>6}: the reset ejects x = [{:.4}, {:.4}] by {:.4} in F", w.0[0], w.0[1], w.1);
            assert!(w.1 > 0.0);
            // and the ejection scales with the set, so shrinking the terminal set does not rescue it — which
            // is exactly why the standard remedy for an input bound does not work here
            let scaled = contact_gap_witness(&fcost, &reset, level * 0.01, &dirs).unwrap();
            assert!(scaled.1 > 0.0, "a hundredfold smaller terminal set is ejected too");
        }

        // A reset that contracts **in the terminal cost's own metric** is a different matter: it keeps every
        // sublevel set, so hybrid dynamics are not hopeless in general — it is expansion at the impact that
        // removes the ingredient.
        //
        // The metric qualifier is not pedantry. Damping only the velocity is a Euclidean contraction, and it
        // still ejects points here, because `F = xᵀPx` has a positive cross-term and shrinking one coordinate
        // can raise the form. That is precisely the quantity
        // `ferromotion_core::impact_expansion` measures — `μ² = λ_max(P^{−1/2} Δᵀ P Δ P^{−1/2})` — and it is
        // why an impact has to be judged in the certificate's metric and not in the state's.
        let velocity_only = |x: &[f64]| vec![x[0], 0.5 * x[1]];
        let mu_sq_vel = ferromotion_core::impact_expansion(&p, &DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.5])).unwrap();
        eprintln!("   damping velocity alone: mu^2 in the P metric = {mu_sq_vel:.4}");
        assert!(mu_sq_vel > 1.0, "it is expansive in the P metric, which is why it still ejects points");
        assert!(contact_gap_witness(&fcost, &velocity_only, 1.0, &dirs).is_some(), "and the witness must find one");

        // A uniform contraction is contracting in every metric, and keeps every sublevel set.
        let uniform = |x: &[f64]| vec![0.5 * x[0], 0.5 * x[1]];
        let mu_sq_uni = ferromotion_core::impact_expansion(&p, &(DMatrix::identity(2, 2) * 0.5)).unwrap();
        eprintln!("   uniform contraction:    mu^2 in the P metric = {mu_sq_uni:.4}");
        assert!(mu_sq_uni < 1.0, "a uniform contraction is contracting in the P metric too");
        assert!(contact_gap_witness(&fcost, &uniform, 1.0, &dirs).is_none(), "so it must keep the terminal set");
    }
}
