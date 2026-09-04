//! **Q4 — global certificates for whole-body QP stacks.**
//!
//! A whole-body controller solves a quadratic program every timestep. With only equality tasks that is a fixed linear
//! map ([`solve_hqp`](crate::solve_hqp) is exactly the null-space recursion for it), and the closed loop is one linear
//! system with one Lyapunov function. Real stacks are not that: torque bounds, friction cones and control-barrier rows
//! are **inequalities**, and the solution of an inequality-constrained QP is a *piecewise-affine* function of the
//! state — affine on each polyhedral region where the active set is constant.
//!
//! So the closed loop under a whole-body QP is a **switched linear system**, and that changes what a stability
//! argument has to prove. The usual practice is to check that the unconstrained gain is stabilising and stop. That is
//! not sufficient, and not by a technicality: a switched system whose every mode is individually stable can diverge
//! under a switching sequence the modes themselves produce. The active set is chosen by the state, so the system picks
//! its own switching sequence and nobody gets to assume it is benign.
//!
//! This module supplies both directions and keeps the logical status of each straight, because the middle ground is
//! where the honest answer usually lands:
//!
//! | tool | what it establishes |
//! |---|---|
//! | [`common_lyapunov`] | a single `P` decreasing on **every** mode — a **sufficient certificate** of switched stability |
//! | [`worst_case_growth`] | a switching sequence whose product has gain above one — a **definitive counterexample** |
//! | neither | undecided by these tools; say so rather than defaulting to "stable" |
//!
//! A common `P` is sufficient and not necessary, so failing to find one proves nothing. A growing product is
//! definitive, because it is a witness. [`QpCertificate::verdict`] reports which of the three happened.

use nalgebra::{DMatrix, DVector};

/// A control QP whose cost gradient and constraint bounds depend affinely on the state:
///
/// ```text
///   u*(x) = argmin_u  1/2 u^T H u + (G x)^T u     subject to   C u <= d + E x
/// ```
///
/// This is the shape a whole-body controller has once the tasks are folded into `H` and `G` and the limits into
/// `C, d, E`: torque bounds give constant rows, a friction cone gives rows depending on the normal force, and a
/// control-barrier constraint gives a row whose bound moves with the state.
#[derive(Clone, Debug)]
pub struct ControlQp {
    pub h: DMatrix<f64>,
    pub g: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DVector<f64>,
    pub e: DMatrix<f64>,
}

impl ControlQp {
    /// The affine law `u = K x + k` on the region where exactly `active` rows of `C` are tight.
    ///
    /// Solves the KKT system
    ///
    /// ```text
    ///   [ H   C_A^T ] [ u ]   [ -G x        ]
    ///   [ C_A   0   ] [ lam ] = [ d_A + E_A x ]
    /// ```
    ///
    /// and reads off the dependence on `x`. Returns `None` if the system is singular, which happens when the active
    /// rows are dependent — a real condition in a whole-body stack, and one that has to be reported rather than
    /// regularised away silently.
    pub fn affine_law(&self, active: &[usize]) -> Option<(DMatrix<f64>, DVector<f64>)> {
        let n = self.h.nrows();
        let nx = self.g.ncols();
        let m = active.len();
        if m == 0 {
            let hinv = self.h.clone().try_inverse()?;
            return Some((-(&hinv) * &self.g, DVector::zeros(n)));
        }
        let mut ca = DMatrix::zeros(m, n);
        let mut ea = DMatrix::zeros(m, nx);
        let mut da = DVector::zeros(m);
        for (r, &i) in active.iter().enumerate() {
            if i >= self.c.nrows() {
                return None;
            }
            ca.set_row(r, &self.c.row(i));
            ea.set_row(r, &self.e.row(i));
            da[r] = self.d[i];
        }
        let mut kkt = DMatrix::zeros(n + m, n + m);
        kkt.view_mut((0, 0), (n, n)).copy_from(&self.h);
        kkt.view_mut((0, n), (n, m)).copy_from(&ca.transpose());
        kkt.view_mut((n, 0), (m, n)).copy_from(&ca);
        let inv = kkt.try_inverse()?;
        // right-hand side is [-G x; d_A + E_A x]: split into the x-linear part and the constant
        let mut rhs_x = DMatrix::zeros(n + m, nx);
        rhs_x.view_mut((0, 0), (n, nx)).copy_from(&(-&self.g));
        rhs_x.view_mut((n, 0), (m, nx)).copy_from(&ea);
        let mut rhs_c = DVector::zeros(n + m);
        rhs_c.view_mut((n, 0), (m, 1)).copy_from(&da);
        let sol_x = &inv * rhs_x;
        let sol_c = &inv * rhs_c;
        Some((sol_x.view((0, 0), (n, nx)).into_owned(), DVector::from_iterator(n, sol_c.iter().take(n).copied())))
    }

    /// Solve the QP at `x` by enumerating active sets and keeping the first feasible KKT point with non-negative
    /// multipliers. Exact for small stacks, which is what a certificate needs — an iterative solver's answer would
    /// carry a residual into the certificate.
    pub fn solve(&self, x: &DVector<f64>) -> Option<(DVector<f64>, Vec<usize>)> {
        let rows = self.c.nrows();
        let mut best: Option<(f64, DVector<f64>, Vec<usize>)> = None;
        for mask in 0..1usize << rows {
            let active: Vec<usize> = (0..rows).filter(|i| mask >> i & 1 == 1).collect();
            let Some((k, k0)) = self.affine_law(&active) else { continue };
            let u = &k * x + &k0;
            // primal feasibility
            let slack = &self.c * &u - (&self.d + &self.e * x);
            if slack.iter().any(|s| *s > 1e-9) {
                continue;
            }
            let cost = 0.5 * (u.transpose() * &self.h * &u)[(0, 0)] + ((&self.g * x).transpose() * &u)[(0, 0)];
            if best.as_ref().is_none_or(|(c, _, _)| cost < *c - 1e-12) {
                best = Some((cost, u, active));
            }
        }
        best.map(|(_, u, a)| (u, a))
    }

    /// The active sets the controller actually visits over `samples` states drawn from the box `[-scale, scale]^nx`,
    /// with the affine closed loop `A + B K` for each.
    ///
    /// Enumerating every combinatorially possible active set would include regions the closed loop never enters, and a
    /// certificate over unreachable modes is needlessly conservative. Sampling finds the ones that occur; it is not a
    /// completeness proof, and [`QpCertificate`] records how many modes it was built from so the scope is visible.
    pub fn visited_modes(&self, a: &DMatrix<f64>, b: &DMatrix<f64>, samples: usize, scale: f64, seed: u64) -> Vec<(Vec<usize>, DMatrix<f64>)> {
        let mut rng = crate::Xorshift::new(seed);
        let nx = self.g.ncols();
        let mut out: Vec<(Vec<usize>, DMatrix<f64>)> = Vec::new();
        for _ in 0..samples {
            let x = DVector::from_iterator(nx, (0..nx).map(|_| scale * (2.0 * rng.uniform() - 1.0)));
            let Some((_, active)) = self.solve(&x) else { continue };
            if out.iter().any(|(s, _)| *s == active) {
                continue;
            }
            if let Some((k, _)) = self.affine_law(&active) {
                out.push((active, a + b * k));
            }
        }
        out
    }
}

/// What the certificate search concluded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// A single `P` decreases on every mode: switched stability is certified.
    Certified,
    /// A switching sequence was found whose product grows: the closed loop is unstable under switching the controller
    /// itself can produce.
    Refuted,
    /// Neither a common `P` nor a growing product was found. These tools do not decide it.
    Undecided,
}

/// The result, with the logical status of each half kept separate.
#[derive(Clone, Debug)]
pub struct QpCertificate {
    pub p: Option<DMatrix<f64>>,
    /// `max_i lambda_max(A_i^T P A_i - P + Q)` at the returned `P`. Non-positive certifies.
    pub worst_residual: f64,
    /// Largest product gain found over the switching sequences searched. Above one refutes.
    pub worst_growth: f64,
    /// The switching sequence achieving `worst_growth`.
    pub worst_sequence: Vec<usize>,
    /// How many distinct active-set modes the certificate covers.
    pub modes: usize,
    /// Spectral radius of the least stable individual mode — the number the usual practice checks.
    pub worst_mode_rho: f64,
}

impl QpCertificate {
    pub fn verdict(&self) -> Verdict {
        if self.p.is_some() && self.worst_residual <= 0.0 {
            Verdict::Certified
        } else if self.worst_growth > 1.0 {
            Verdict::Refuted
        } else {
            Verdict::Undecided
        }
    }

    /// **Whether checking the modes individually would have been misleading.** True when every mode is individually
    /// stable and yet a switching sequence grows — the case the usual practice cannot see.
    pub fn individually_stable_but_switching_unstable(&self) -> bool {
        self.worst_mode_rho < 1.0 && self.worst_growth > 1.0
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

fn project_pd(m: &DMatrix<f64>, floor: f64) -> DMatrix<f64> {
    let sym = 0.5 * (m + m.transpose());
    let se = sym.symmetric_eigen();
    let d = DMatrix::from_diagonal(&se.eigenvalues.map(|l| l.max(floor)));
    &se.eigenvectors * d * se.eigenvectors.transpose()
}

/// **A common Lyapunov function for a set of modes**, by subgradient descent on
/// `phi(P) = max_i lambda_max(A_i^T P A_i - P + Q)`.
///
/// Convex in `P`, so descent is sound, and `phi <= 0` at any iterate is a certificate because the iterate is the
/// witness. Returns the best `P` reached together with its worst residual, so a near-miss is visible rather than
/// collapsing to a boolean.
pub fn common_lyapunov(modes: &[DMatrix<f64>], q: &DMatrix<f64>, iterations: usize, step: f64) -> (Option<DMatrix<f64>>, f64) {
    let Some(first) = modes.first() else { return (None, f64::INFINITY) };
    let n = first.nrows();
    if modes.iter().any(|m| m.nrows() != n || m.ncols() != n) || q.nrows() != n {
        return (None, f64::INFINITY);
    }
    let mut p = DMatrix::identity(n, n);
    let mut best: Option<(f64, DMatrix<f64>)> = None;
    for k in 0..iterations {
        let residuals: Vec<(f64, usize)> = modes
            .iter()
            .enumerate()
            .map(|(i, a)| (lambda_max(&(a.transpose() * &p * a - &p + q)), i))
            .collect();
        let (worst, idx) = residuals.iter().fold((f64::NEG_INFINITY, 0), |acc, &(r, i)| if r > acc.0 { (r, i) } else { acc });
        if best.as_ref().is_none_or(|(b, _)| worst < *b) {
            best = Some((worst, p.clone()));
        }
        if worst <= 0.0 {
            break;
        }
        let a = &modes[idx];
        let f = a.transpose() * &p * a - &p + q;
        let v = top_eigenvector(&f);
        let av = a * &v;
        let g = &av * av.transpose() - &v * v.transpose();
        let alpha = step * lambda_max(&p).max(1.0) / (1.0 + k as f64).sqrt();
        p = project_pd(&(&p - alpha * g), 1e-8);
    }
    match best {
        Some((r, pb)) => (Some(pb), r),
        None => (None, f64::INFINITY),
    }
}

/// **Search switching sequences for a product that grows.**
///
/// Exhaustive over all `modes.len()^horizon` sequences, returning the largest `||A_{i_T} ... A_{i_1}||` found and the
/// sequence achieving it. A value above one is a **witness of instability**: that sequence, repeated, diverges.
///
/// This is a lower bound on the joint spectral radius, never an upper one, so it can refute stability but cannot
/// establish it. Guard the horizon: cost is exponential in it.
pub fn worst_case_growth(modes: &[DMatrix<f64>], horizon: usize) -> (f64, Vec<usize>) {
    let n = modes.len();
    if n == 0 || horizon == 0 || n.checked_pow(horizon as u32).is_none_or(|t| t > 200_000) {
        return (0.0, Vec::new());
    }
    let dim = modes[0].nrows();
    let mut best = (0.0f64, Vec::new());
    for code in 0..n.pow(horizon as u32) {
        let mut seq = Vec::with_capacity(horizon);
        let mut c = code;
        for _ in 0..horizon {
            seq.push(c % n);
            c /= n;
        }
        let mut prod = DMatrix::identity(dim, dim);
        for &i in &seq {
            prod = &modes[i] * prod;
        }
        // per-step gain, so the number is comparable across horizons
        let gain = ferromotion_core::finite_singular_values(&prod).and_then(|s| s.first().copied()).unwrap_or(f64::NAN).powf(1.0 / horizon as f64);
        if gain > best.0 {
            best = (gain, seq);
        }
    }
    best
}

/// **Certify (or refute) a whole-body QP closed loop.** Finds the visited active-set modes, tries for a common
/// Lyapunov function, and independently searches for a growing switching sequence.
#[allow(clippy::too_many_arguments)]
pub fn certify_qp_loop(
    qp: &ControlQp,
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    q: &DMatrix<f64>,
    samples: usize,
    scale: f64,
    horizon: usize,
    seed: u64,
) -> QpCertificate {
    let visited = qp.visited_modes(a, b, samples, scale, seed);
    let modes: Vec<DMatrix<f64>> = visited.iter().map(|(_, m)| m.clone()).collect();
    let worst_mode_rho = modes
        .iter()
        .map(|m| m.complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    let (p, worst_residual) = common_lyapunov(&modes, q, 600, 0.30);
    let (worst_growth, worst_sequence) = worst_case_growth(&modes, horizon);
    QpCertificate { p, worst_residual, worst_growth, worst_sequence, modes: modes.len(), worst_mode_rho }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A double integrator with a torque bound — the smallest whole-body-shaped QP that switches. Its open loop is
    /// marginally unstable (both eigenvalues at 1), which turns out to matter: see
    /// `saturation_hands_the_loop_back_to_the_open_loop`.
    fn saturating_qp(limit: f64, gain: f64) -> (ControlQp, DMatrix<f64>, DMatrix<f64>) {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        // cost 1/2 u^2 + (K x) u drives u toward -K x; rows +-u <= limit are the torque bound
        let qp = ControlQp {
            h: DMatrix::identity(1, 1),
            g: DMatrix::from_row_slice(1, 2, &[gain, gain * 0.6]),
            c: DMatrix::from_row_slice(2, 1, &[1.0, -1.0]),
            d: DVector::from_row_slice(&[limit, limit]),
            e: DMatrix::zeros(2, 2),
        };
        (qp, a, b)
    }

    /// The QP's solution must be piecewise affine, with the active set switching where the bound bites.
    #[test]
    fn the_qp_controller_is_piecewise_affine() {
        let (qp, _a, _b) = saturating_qp(1.0, 4.0);
        eprintln!("{:>8}  {:>10}  active rows", "x1", "u*");
        let mut seen = Vec::new();
        for x1 in [-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0] {
            let x = DVector::from_row_slice(&[x1, 0.0]);
            let (u, active) = qp.solve(&x).expect("feasible");
            eprintln!("{x1:>8.2}  {:>10.4}  {active:?}", u[0]);
            if !seen.contains(&active) {
                seen.push(active);
            }
        }
        assert!(seen.len() >= 2, "the bound has to actually bite somewhere: {seen:?}");
        // inside the bound the law is the unconstrained one; outside it is the saturated constant
        let inside = qp.solve(&DVector::from_row_slice(&[0.1, 0.0])).unwrap().0[0];
        let outside = qp.solve(&DVector::from_row_slice(&[1.0, 0.0])).unwrap().0[0];
        assert!((inside + 0.4).abs() < 1e-9, "unconstrained: u = -K x = -0.4, got {inside}");
        assert!((outside + 1.0).abs() < 1e-9, "saturated at the bound: u = -1, got {outside}");
    }

    /// A damped plant, so the **open loop itself** is stable. This is the distinction that decides whether a
    /// saturating QP can be certified at all.
    fn damped_qp(limit: f64, gain: f64) -> (ControlQp, DMatrix<f64>, DMatrix<f64>) {
        let a = DMatrix::from_row_slice(2, 2, &[0.90, 0.10, 0.00, 0.85]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 0.1]);
        let qp = ControlQp {
            h: DMatrix::identity(1, 1),
            g: DMatrix::from_row_slice(1, 2, &[gain, gain * 0.6]),
            c: DMatrix::from_row_slice(2, 1, &[1.0, -1.0]),
            d: DVector::from_row_slice(&[limit, limit]),
            e: DMatrix::zeros(2, 2),
        };
        (qp, a, b)
    }

    /// **Saturation hands the loop back to the open loop.** Where the torque bound is tight the QP's affine law is a
    /// constant, so its gain is zero and the closed loop on that region is `A` itself. Whether a saturating whole-body
    /// QP is certifiable therefore depends on a property of the *plant*, not of the controller.
    ///
    /// This is why the first version of this test — which assumed a low enough gain would certify a double integrator —
    /// was wrong. No gain certifies it: the saturated mode is the open loop, and a double integrator's open loop has
    /// spectral radius exactly 1.
    #[test]
    fn saturation_hands_the_loop_back_to_the_open_loop() {
        for (name, (qp, a, b)) in [("double integrator", saturating_qp(1.0, 4.0)), ("damped plant", damped_qp(1.0, 4.0))] {
            let open_rho = a.complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0, f64::max);
            let modes = qp.visited_modes(&a, &b, 400, 2.0, 3);
            eprintln!("{name}: open-loop rho {open_rho:.4}, {} visited modes", modes.len());
            for (active, m) in &modes {
                let rho = m.complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0, f64::max);
                let saturated = !active.is_empty();
                eprintln!("    active {active:?}{}: closed-loop rho {rho:.4}", if saturated { " (saturated)" } else { " (free)   " });
                if saturated {
                    // on a saturated region the gain is zero, so the mode IS the open loop
                    assert!((rho - open_rho).abs() < 1e-9, "a saturated mode is the open loop: {rho:.4} vs {open_rho:.4}");
                }
            }
        }
        eprintln!("\n   So the certifiability of a saturating stack is a question about the PLANT. A controller cannot");
        eprintln!("   certify a region in which it has been switched off.");
    }

    /// **A stack certifies when the plant's open loop is stable**, and then a single `P` covers every mode.
    #[test]
    fn a_stack_on_a_stable_plant_gets_a_common_lyapunov_function() {
        let (qp, a, b) = damped_qp(1.0, 4.0);
        let cert = certify_qp_loop(&qp, &a, &b, &(DMatrix::identity(2, 2) * 1e-3), 400, 2.0, 6, 7);
        eprintln!(
            "modes {}, worst mode rho {:.4}, worst residual {:.3e}, worst switching gain {:.4} -> {:?}",
            cert.modes, cert.worst_mode_rho, cert.worst_residual, cert.worst_growth, cert.verdict()
        );
        assert_eq!(cert.verdict(), Verdict::Certified, "residual {:.3e}", cert.worst_residual);
        // the witness is checked directly rather than trusted
        let p = cert.p.as_ref().unwrap();
        for (_, m) in qp.visited_modes(&a, &b, 400, 2.0, 7) {
            assert!(lambda_max(&(m.transpose() * p * &m - p + DMatrix::identity(2, 2) * 1e-3)) <= 1e-12);
        }
        eprintln!("   one P decreases on the free mode AND on the saturated mode, so the switching cannot hurt it.");
    }

    /// **The result that matters: every mode individually stable, and switching still diverges.**
    ///
    /// This is the case the usual practice — check the unconstrained gain, ship it — cannot see, and the reason a
    /// whole-body QP needs a switched certificate rather than a linear one.
    #[test]
    fn individually_stable_modes_can_switch_into_instability() {
        // two modes, each a stable spiral, that together shear a vector outward. This is the classical construction,
        // written out explicitly so the certificate machinery is being tested on a known answer.
        let m1 = DMatrix::from_row_slice(2, 2, &[0.95, 0.0, 8.0, 0.95]);
        let m2 = DMatrix::from_row_slice(2, 2, &[0.95, 8.0, 0.0, 0.95]);
        for (name, m) in [("mode 1", &m1), ("mode 2", &m2)] {
            let rho = m.complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0, f64::max);
            eprintln!("{name}: spectral radius {rho:.4} -> individually {}", if rho < 1.0 { "STABLE" } else { "unstable" });
            assert!(rho < 1.0);
        }
        let (growth, seq) = worst_case_growth(&[m1.clone(), m2.clone()], 8);
        eprintln!("\nworst switching sequence over 8 steps: {seq:?}");
        eprintln!("   per-step gain {growth:.4} -> the product GROWS, so the switched system is unstable");
        assert!(growth > 1.0, "a growing product is the witness: {growth:.4}");

        // and no common Lyapunov function exists, which is consistent: the search cannot find one
        let (_, residual) = common_lyapunov(&[m1, m2], &(DMatrix::identity(2, 2) * 1e-6), 800, 0.3);
        eprintln!("   common Lyapunov search: best worst-residual {residual:.3e} (positive, so no certificate)");
        assert!(residual > 0.0, "there is no common P, and the search correctly fails to produce one");
        eprintln!("\n   Checking each mode's spectral radius would have passed this system. It diverges.");
    }

    /// **The double integrator is refuted at every gain**, and the certificate says which way and why. A verdict is
    /// always one of the three; it is never a silent pass.
    #[test]
    fn a_saturating_stack_on_a_marginal_plant_is_refuted_at_every_gain() {
        let mut verdicts = Vec::new();
        for gain in [2.0, 20.0, 120.0] {
            let (qp, a, b) = saturating_qp(1.0, gain);
            let cert = certify_qp_loop(&qp, &a, &b, &(DMatrix::identity(2, 2) * 1e-3), 300, 2.0, 6, 11);
            eprintln!(
                "gain {gain:>6.0}: modes {}, worst mode rho {:.4}, growth {:.4}, verdict {:?}{}",
                cert.modes,
                cert.worst_mode_rho,
                cert.worst_growth,
                cert.verdict(),
                if cert.individually_stable_but_switching_unstable() { "  <- individually stable, switching unstable" } else { "" }
            );
            verdicts.push(cert.verdict());
        }
        assert!(verdicts.iter().all(|v| *v == Verdict::Refuted), "no gain rescues a marginal plant: {verdicts:?}");
        eprintln!("\n   Lowering the gain does not help, because the gain is irrelevant where the bound is tight.");
        eprintln!("   The fix is a plant with a stable open loop, or a guarantee that the saturated region is never");
        eprintln!("   entered - and the second is a reachability claim, not a Lyapunov one.");

        // and on a stable plant the same sweep certifies, so the difference is the plant and not the sweep
        let stable: Vec<Verdict> = [2.0, 20.0, 120.0]
            .iter()
            .map(|g| {
                let (qp, a, b) = damped_qp(1.0, *g);
                certify_qp_loop(&qp, &a, &b, &(DMatrix::identity(2, 2) * 1e-3), 300, 2.0, 6, 11).verdict()
            })
            .collect();
        eprintln!("   same sweep on the damped plant: {stable:?}");
        assert!(stable.contains(&Verdict::Certified), "the stable plant certifies somewhere: {stable:?}");
    }

    /// Degenerate input is refused rather than producing a certificate for nothing.
    #[test]
    fn empty_or_mismatched_input_is_refused() {
        assert_eq!(common_lyapunov(&[], &DMatrix::identity(2, 2), 10, 0.3).1, f64::INFINITY);
        let bad = [DMatrix::identity(2, 2), DMatrix::identity(3, 3)];
        assert_eq!(common_lyapunov(&bad, &DMatrix::identity(2, 2), 10, 0.3).1, f64::INFINITY);
        assert_eq!(worst_case_growth(&[], 4).0, 0.0);
        // and the horizon guard stops an exponential blow-up rather than hanging
        let many: Vec<DMatrix<f64>> = (0..10).map(|_| DMatrix::identity(2, 2)).collect();
        assert_eq!(worst_case_growth(&many, 12).0, 0.0, "10^12 sequences must be refused, not attempted");
    }
}
