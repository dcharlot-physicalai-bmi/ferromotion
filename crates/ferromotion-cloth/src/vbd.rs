//! **Vertex Block Descent** — the deformable solver that stays stable at one iteration per step.
//!
//! **On the word "unconditionally".** The unconditional-stability claim belongs to the paper below, not to this file.
//! What this file measures is one operating point: `dt = 1/30 s`, `k = 1e6`, a 12-link chain of `0.1 m` links at
//! `0.05 kg`, 600 steps, where a single sweep stays bounded (max extent `1.9124 m`) and explicit Euler on the identical
//! system goes `NaN`. No test here sweeps `dt`, stiffness, mass or mesh, so cite the paper for the general property and
//! this measurement for what the code demonstrates. And carry the qualification with it: **stable is not accurate.**
//! At that same operating point one sweep overstretches a `1.200 m` rest length to `1.9194 m`, `+60.0%`; four sweeps is
//! `+14.6%`, sixteen `+3.3%`, sixty-four `+0.5%`. One sweep buys boundedness; accuracy is bought in sweeps.
//!
//! VBD (Chen, Liu, Yang and Yuksel, SIGGRAPH 2024) solves the *variational* form of implicit Euler
//!
//! ```text
//!   minimise  E(x) = sum_i m_i/(2 h^2) ||x_i - y_i||^2  +  U(x),      y_i = x_i^n + h v_i^n + h^2 f_ext / m_i
//! ```
//!
//! by block coordinate descent: sweep the vertices, and for each one take a Newton step in its own `3x3` block while
//! holding the rest fixed, using the neighbours already updated in this sweep (Gauss-Seidel). Its defining property is
//! that it stays stable with a **single iteration** and a large timestep, running on an unconverged solution with a
//! large residual, and converges to the implicit-Euler solution as iterations increase. That combination is what makes
//! it the right default for a real-time deformable: it degrades into something usable rather than exploding.
//!
//! Stability comes from projecting each vertex Hessian block to positive semi-definite before inverting. A spring in
//! *compression* contributes an indefinite block — the `(1 - L/||d||)` transverse term goes negative — and taking a
//! Newton step against an indefinite block is what makes naive implicit solvers blow up on buckling. Dropping the
//! negative part ([`spring_hessian_psd`]) leaves a descent direction, at the cost of a shorter step.
//!
//! [`AugmentedVbd`] adds the augmented-Lagrangian treatment of AVBD (SIGGRAPH Real-Time Live 2025), which is what
//! makes the method usable when stiffnesses in one system differ by orders of magnitude — the regime where plain VBD's
//! Gauss-Seidel sweep converges slowly because the stiff constraints dominate every block.

use nalgebra::{Matrix3, Vector3};

/// A simulation vertex.
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub mass: f64,
    /// A pinned vertex is a boundary condition: it contributes to its neighbours and never moves.
    pub pinned: bool,
}

impl Vertex {
    pub fn new(position: Vector3<f64>, mass: f64) -> Vertex {
        Vertex { position, velocity: Vector3::zeros(), mass, pinned: false }
    }

    pub fn pinned_at(position: Vector3<f64>) -> Vertex {
        Vertex { position, velocity: Vector3::zeros(), mass: 1.0, pinned: true }
    }
}

/// A distance constraint with a rest length — the element VBD is usually demonstrated on, and the one that covers
/// cables and cloth warp/weft directly.
#[derive(Clone, Copy, Debug)]
pub struct Spring {
    pub i: usize,
    pub j: usize,
    pub rest: f64,
    pub stiffness: f64,
}

/// **A maximum that reports divergence instead of hiding it.**
///
/// `f64::max` returns the non-NaN argument, by IEEE-754 `maxNum`. So `.fold(0.0, f64::max)` over an all-NaN sequence
/// yields `0.0`, and `0.0.is_finite()` is `true` — measured, not reasoned about. Every reduction in this module that
/// summarises a quantity capable of diverging must go through here, because the alternative reads a blown-up solve as
/// a perfect one: a `0.0` maximum extent looks like a chain at rest, and a `0.0` residual looks like convergence.
///
/// Returns `0.0` for an empty sequence, matching the identity of a maximum over norms.
pub fn nan_propagating_max(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0f64, |a, b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) })
}

/// Elastic energy of one spring at the given endpoint positions.
pub fn spring_energy(xi: Vector3<f64>, xj: Vector3<f64>, rest: f64, stiffness: f64) -> f64 {
    let ext = (xi - xj).norm() - rest;
    0.5 * stiffness * ext * ext
}

/// `dU/dx_i` for one spring.
pub fn spring_gradient(xi: Vector3<f64>, xj: Vector3<f64>, rest: f64, stiffness: f64) -> Vector3<f64> {
    let d = xi - xj;
    let len = d.norm();
    if len < 1e-12 {
        return Vector3::zeros();
    }
    stiffness * (len - rest) * d / len
}

/// **`d2U/dx_i^2` for one spring, projected to positive semi-definite.**
///
/// The exact Hessian is `k [ d_hat d_hat^T + (1 - L/||d||)(I - d_hat d_hat^T) ]`. In compression `||d|| < L` makes the
/// transverse coefficient negative and the block indefinite, and a Newton step against an indefinite block is an
/// ascent direction in those directions. Clamping the coefficient at zero is VBD's projection: the result is still a
/// descent direction and the method stays stable through buckling.
pub fn spring_hessian_psd(xi: Vector3<f64>, xj: Vector3<f64>, rest: f64, stiffness: f64) -> Matrix3<f64> {
    let d = xi - xj;
    let len = d.norm();
    if len < 1e-12 {
        return Matrix3::identity() * stiffness;
    }
    let dh = d / len;
    let outer = dh * dh.transpose();
    let transverse = (1.0 - rest / len).max(0.0); // <- the projection
    stiffness * (outer + transverse * (Matrix3::identity() - outer))
}

/// The VBD solver.
#[derive(Clone, Copy, Debug)]
pub struct VbdSolver {
    pub dt: f64,
    /// Gauss-Seidel sweeps per step. One is enough to stay stable; more converges toward implicit Euler.
    pub iterations: usize,
    pub gravity: Vector3<f64>,
    /// Velocity damping as a **rate in 1/s**, applied at the end of a step:
    /// `v ← v / (1 + damping_rate·dt)`.
    ///
    /// A rate, not a per-step fraction, so that `dt` stays an accuracy knob instead of a physical one.
    /// A per-step decay `v ← v·(1−d)` dissipates `−ln(1−d)/dt` per second, so shortening the step
    /// strengthens the damping in proportion. Measured on a hanging chain with `damping = 0.02`:
    /// taking `dt` from 4e-3 to 1e-4 raised the effective rate from 5.1 /s to 202 /s, and moved the
    /// tip at t = 0.5 s from -0.627 m to -0.024 m. The rest state is unaffected (damping cannot move
    /// an equilibrium) but every transient is, and at the stiffest setting the chain had still not
    /// reached its rest length after 20 s of simulated time.
    pub damping_rate: f64,
}

impl VbdSolver {
    pub fn new(dt: f64, iterations: usize) -> Option<VbdSolver> {
        (dt > 0.0 && iterations >= 1).then_some(VbdSolver { dt, iterations, gravity: Vector3::new(0.0, 0.0, -9.81), damping_rate: 0.0 })
    }

    /// The inertial (predicted) positions `y_i`, which the variational objective pulls toward.
    pub fn inertial_positions(&self, verts: &[Vertex]) -> Vec<Vector3<f64>> {
        verts.iter().map(|v| v.position + self.dt * v.velocity + self.dt * self.dt * self.gravity).collect()
    }

    /// The variational objective `E(x)`. Its minimiser is the implicit-Euler step, which is what makes it the right
    /// thing to monitor: [`Self::residual`] going to zero means the step *is* an implicit-Euler step.
    pub fn objective(&self, x: &[Vector3<f64>], y: &[Vector3<f64>], verts: &[Vertex], springs: &[Spring]) -> f64 {
        let inertia: f64 = verts
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.pinned)
            .map(|(i, v)| v.mass / (2.0 * self.dt * self.dt) * (x[i] - y[i]).norm_squared())
            .sum();
        let elastic: f64 = springs.iter().map(|s| spring_energy(x[s.i], x[s.j], s.rest, s.stiffness)).sum();
        inertia + elastic
    }

    /// `max_i ||grad_i E||` over the free vertices — zero exactly at the implicit-Euler solution, and `NaN` if the
    /// state is not finite.
    ///
    /// **That `NaN` is a correction (2026-08-06).** This used to fold with `.fold(0.0, f64::max)`, and `f64::max`
    /// returns the *non-NaN* argument, so `[NaN, NaN].fold(0.0, f64::max)` is `0.0` with `is_finite() == true`
    /// (measured, not reasoned). A fully diverged solve therefore reported a residual of exactly zero — which this
    /// doc comment reads as "exactly at the implicit-Euler solution" — and the convergence test's `r < last` and
    /// `last < 1e-5` would both have passed on garbage. `residual` is the number a convergence plot puts on its
    /// y-axis, so the failure mode was: break the solver, watch the residual drop to zero, conclude it converged.
    ///
    /// The trap was already documented in this file, on the `max_extent` test helper below, and fixed there only.
    /// Use [`nan_propagating_max`] for any new reduction over a quantity that can diverge.
    pub fn residual(&self, x: &[Vector3<f64>], y: &[Vector3<f64>], verts: &[Vertex], springs: &[Spring]) -> f64 {
        let mut grad = vec![Vector3::zeros(); verts.len()];
        for (i, v) in verts.iter().enumerate() {
            if !v.pinned {
                grad[i] = v.mass / (self.dt * self.dt) * (x[i] - y[i]);
            }
        }
        for s in springs {
            let g = spring_gradient(x[s.i], x[s.j], s.rest, s.stiffness);
            grad[s.i] += g;
            grad[s.j] -= g;
        }
        nan_propagating_max(verts.iter().enumerate().filter(|(_, v)| !v.pinned).map(|(i, _)| grad[i].norm()))
    }

    /// **One VBD step.** Sweeps vertices `iterations` times, taking a projected Newton step in each vertex's own block
    /// using neighbours already updated this sweep.
    pub fn step(&self, verts: &mut [Vertex], springs: &[Spring]) {
        let y = self.inertial_positions(verts);
        let previous: Vec<Vector3<f64>> = verts.iter().map(|v| v.position).collect();
        let mut x: Vec<Vector3<f64>> = y.clone();
        for (i, v) in verts.iter().enumerate() {
            if v.pinned {
                x[i] = v.position;
            }
        }

        // adjacency, so a sweep can find a vertex's springs without scanning all of them
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
        for (si, s) in springs.iter().enumerate() {
            incident[s.i].push(si);
            incident[s.j].push(si);
        }

        for _ in 0..self.iterations {
            for i in 0..verts.len() {
                if verts[i].pinned {
                    continue;
                }
                let mut grad = verts[i].mass / (self.dt * self.dt) * (x[i] - y[i]);
                let mut hess = Matrix3::identity() * (verts[i].mass / (self.dt * self.dt));
                for &si in &incident[i] {
                    let s = springs[si];
                    let other = if s.i == i { s.j } else { s.i };
                    grad += spring_gradient(x[i], x[other], s.rest, s.stiffness);
                    hess += spring_hessian_psd(x[i], x[other], s.rest, s.stiffness);
                }
                // the mass term keeps the block invertible even with every spring projected away
                if let Some(inv) = hess.try_inverse() {
                    x[i] -= inv * grad;
                }
            }
        }

        for (i, v) in verts.iter_mut().enumerate() {
            if !v.pinned {
                v.velocity = (x[i] - previous[i]) / self.dt / (1.0 + self.damping_rate * self.dt);
                v.position = x[i];
            }
        }
    }
}

/// **AVBD** — VBD with an augmented-Lagrangian term per constraint, which is what makes it converge when the system
/// mixes stiff and soft elements.
///
/// Plain VBD treats a stiff spring as a large penalty, so its Hessian block is dominated by the stiffest incident
/// constraint and the Gauss-Seidel sweep propagates information across the mesh slowly. AVBD instead carries a
/// multiplier `lambda` per constraint and a finite penalty, so a stiff constraint is enforced by the multiplier rather
/// than by making the block enormous. The effective force on a constraint with extension `C` becomes
/// `lambda + penalty * C`, and `lambda` is updated after each step.
#[derive(Clone, Debug)]
///
/// **The penalty is not a free knob (measured 2026-08-06).** `step` builds each vertex block with `penalty` in place of
/// the spring's true stiffness, so the penalty IS the stiffness the Newton step sees. Choose it too small and the block
/// is far softer than the constraint it must enforce, and the solve blows up — while staying finite, which is how this
/// went unnoticed. On a uniform-stiffness 10-link chain of `0.1 m` links (rest total `1.0 m`), 4 sweeps, 200 steps:
///
/// ```text
///   stiffness   penalty   penalty/stiffness   max extent      verdict
///        1e7       1e2               1e-5     23192.7883 m    finite, and 23 km of it
///        1e7       1e4               1e-3       373.2868 m    diverged
///        1e7       1e6               1e-1         6.0075 m    diverged
///        1e7       1e8                1e1         1.9845 m    usable
///        1e4       1e2               1e-2        29.4211 m    diverged
///        1e4       1e4                1e0         1.3283 m    usable
/// ```
///
/// Bisected, the usable threshold is a RATIO, rising slowly with stiffness: `penalty/stiffness` of about `0.02` at
/// `k = 1e2`, `0.06` at `1e4`, `0.13` at `1e7`. So: **pick the penalty on the order of the largest stiffness present**,
/// and check the resulting EXTENT rather than only that the numbers are finite.
///
/// The mixed fixture this module tests (`1e7` / `1e2` alternating, penalty `1e4`) is fine despite a `1e-3` ratio against
/// its stiff links, because the soft links leave the chain compliant enough to satisfy them. The threshold is therefore
/// a property of the configuration, not of the ratio alone. See `examples/avbd_penalty_threshold.rs`.
///
/// # ⚠️ This solves the INEXTENSIBLE problem, not the elastic one (measured 2026-08-15)
///
/// `step` applies the force `λ + penalty·C`, in which **`Spring::stiffness` appears nowhere** except as the clamp
/// bound on `λ`. That is the augmented Lagrangian for the *hard* constraint `C = 0`: `λ` accumulates until the link
/// stops extending. So a deliberately soft spring is driven nearly rigid, and this solver does **not** converge to the
/// elastic implicit-Euler solution [`VbdSolver`] targets — the two have different fixed points.
///
/// Measured on a uniform soft chain, `hanging_chain(10, 0.1, 0.05, 1e2)`, run to rest:
///
/// | | total length |
/// |---|---|
/// | exact static elastic equilibrium, `nL + (mg/k)·n(n+1)/2` | **1.269775 m** |
/// | [`VbdSolver`] at 64 sweeps | 1.269775 m (0.0000% error) |
/// | `AugmentedVbd` at 4 sweeps | **1.022610 m (−19.47%)** |
///
/// Link 0 extends `0.004959 m` where the spring law gives `0.049050 m` — **10× too little** — and the clamp bound
/// `k·rest = 10.0 N` sits at **2× the true maximum tension** of `n·m·g = 4.905 N`, so the clamp permits more force
/// than the spring could ever supply and is not what holds the answer together.
///
/// **A compliance term does not repair it, and I checked rather than assumed.** Dividing the update by
/// `1 + penalty/k` does make `λ → k·C`, the elastic force — but the force *applied* is `λ + penalty·C`, so the total
/// becomes `(k + penalty)·C`, which at `penalty = 1e4, k = 1e2` is **101× too stiff**. Measured, it moved the chain
/// from `1.0226` to only `1.0311` while making the stiff-link violation on the mixed fixture 3× worse
/// (`1.11e-4 → 3.37e-4 m`). For a compliant spring the correct total force is simply `k·C`, so the augmented
/// Lagrangian contributes nothing: it is structurally a **hard-constraint** tool, and no small edit converts it.
///
/// **So use this for near-rigid elements and [`VbdSolver`] when the elastic solution is what you want.** Making
/// `AugmentedVbd` handle finite stiffness correctly needs the adaptive-penalty formulation from the AVBD literature,
/// not a patch to this update — that is recorded as open work rather than guessed at here.
///
/// Note also that `avbd_handles_a_stiffness_ratio_that_slows_plain_vbd` asserts only on the **stiff** links
/// (`filter(|(i, _)| i % 2 == 0)`), so the regime where this solver is wrong is the half of its own fixture that was
/// never measured. `the_soft_links_are_driven_nearly_rigid` now pins it.
pub struct AugmentedVbd {
    pub solver: VbdSolver,
    /// One multiplier per spring.
    pub multipliers: Vec<f64>,
    /// The finite penalty that replaces the true stiffness inside the sweep.
    pub penalty: f64,
}

impl AugmentedVbd {
    pub fn new(solver: VbdSolver, springs: usize, penalty: f64) -> Option<AugmentedVbd> {
        (penalty > 0.0).then(|| AugmentedVbd { solver, multipliers: vec![0.0; springs], penalty })
    }

    /// One AVBD step. The sweep uses `penalty` in place of each spring's stiffness and adds the multiplier as a
    /// constant force, then the multipliers absorb the residual constraint violation.
    pub fn step(&mut self, verts: &mut [Vertex], springs: &[Spring]) {
        let dt2 = self.solver.dt * self.solver.dt;
        let y = self.solver.inertial_positions(verts);
        let previous: Vec<Vector3<f64>> = verts.iter().map(|v| v.position).collect();
        let mut x: Vec<Vector3<f64>> = y.clone();
        for (i, v) in verts.iter().enumerate() {
            if v.pinned {
                x[i] = v.position;
            }
        }
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
        for (si, s) in springs.iter().enumerate() {
            incident[s.i].push(si);
            incident[s.j].push(si);
        }

        for _ in 0..self.solver.iterations {
            for i in 0..verts.len() {
                if verts[i].pinned {
                    continue;
                }
                let mut grad = verts[i].mass / dt2 * (x[i] - y[i]);
                let mut hess = Matrix3::identity() * (verts[i].mass / dt2);
                for &si in &incident[i] {
                    let s = springs[si];
                    let other = if s.i == i { s.j } else { s.i };
                    let d = x[i] - x[other];
                    let len = d.norm();
                    if len < 1e-12 {
                        continue;
                    }
                    let c = len - s.rest;
                    // force = lambda + penalty * C, rather than stiffness * C
                    grad += (self.multipliers[si] + self.penalty * c) * d / len;
                    hess += spring_hessian_psd(x[i], x[other], s.rest, self.penalty);
                }
                if let Some(inv) = hess.try_inverse() {
                    x[i] -= inv * grad;
                }
            }
        }

        // the multipliers take up the slack the finite penalty left behind
        for (si, s) in springs.iter().enumerate() {
            let c = (x[s.i] - x[s.j]).norm() - s.rest;
            self.multipliers[si] = (self.multipliers[si] + self.penalty * c).clamp(-s.stiffness * s.rest, s.stiffness * s.rest);
        }

        for (i, v) in verts.iter_mut().enumerate() {
            if !v.pinned {
                v.velocity = (x[i] - previous[i]) / self.solver.dt / (1.0 + self.solver.damping_rate * self.solver.dt);
                v.position = x[i];
            }
        }
    }
}

/// Explicit (symplectic) Euler on the same system, as the stability reference. It is the thing VBD is claimed to beat,
/// so it has to be present to check the claim rather than assert it.
pub fn explicit_step(verts: &mut [Vertex], springs: &[Spring], dt: f64, gravity: Vector3<f64>) {
    let mut force: Vec<Vector3<f64>> = verts.iter().map(|v| if v.pinned { Vector3::zeros() } else { v.mass * gravity }).collect();
    for s in springs {
        let g = spring_gradient(verts[s.i].position, verts[s.j].position, s.rest, s.stiffness);
        force[s.i] -= g;
        force[s.j] += g;
    }
    for (i, v) in verts.iter_mut().enumerate() {
        if !v.pinned {
            v.velocity += dt * force[i] / v.mass;
            v.position += dt * v.velocity;
        }
    }
}

/// A hanging chain: one pinned end, `n` free links. The standard VBD demonstration case, and stiff enough to expose
/// an unstable integrator immediately.
pub fn hanging_chain(n: usize, link: f64, mass: f64, stiffness: f64) -> (Vec<Vertex>, Vec<Spring>) {
    let mut verts = vec![Vertex::pinned_at(Vector3::zeros())];
    for i in 1..=n {
        verts.push(Vertex::new(Vector3::new(link * i as f64, 0.0, 0.0), mass));
    }
    let springs = (0..n).map(|i| Spring { i, j: i + 1, rest: link, stiffness }).collect();
    (verts, springs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f64::max` returns the non-NaN argument, so folding with it silently swallows a diverged trajectory.
    /// [`nan_propagating_max`] is the shared fix; this helper existed before it and now delegates, so the module and
    /// its tests cannot drift apart on the one reduction that decides whether divergence is visible.
    fn max_extent(verts: &[Vertex]) -> f64 {
        nan_propagating_max(verts.iter().map(|v| v.position.norm()))
    }

    /// **`AugmentedVbd` and `VbdSolver` have DIFFERENT fixed points, and this pins which.** The augmented
    /// Lagrangian drives `C → 0`, so a soft spring is made nearly rigid; `VbdSolver` reaches the elastic solution.
    /// The module doc claimed both target the implicit-Euler elastic answer. See that doc for why a compliance
    /// term does not repair it.
    #[test]
    fn the_soft_links_are_driven_nearly_rigid() {
        let (n, link, mass, k) = (10usize, 0.1, 0.05, 1e2);
        let g = 9.81;
        // Static equilibrium: link j carries the weight below it, so extension_j = (n-j)*m*g/k.
        let exact: f64 = n as f64 * link + (mass * g / k) * (n * (n + 1)) as f64 / 2.0;
        assert!((exact - 1.269775).abs() < 1e-6, "fixture arithmetic: {exact}");

        let settle = |avbd: bool, sweeps: usize| -> (f64, f64) {
            let (mut verts, springs) = hanging_chain(n, link, mass, k);
            let mut solver = VbdSolver::new(1.0 / 60.0, sweeps).unwrap();
            solver.damping_rate = 3.157_894_736_842_105; // old per-step 0.05 at dt = 1/60
            let mut aug = AugmentedVbd::new(solver, 10, 1e4).expect("valid avbd");
            for _ in 0..2000 {
                if avbd {
                    aug.step(&mut verts, &springs);
                } else {
                    solver.step(&mut verts, &springs);
                }
            }
            let total: f64 = springs.iter().map(|s| (verts[s.i].position - verts[s.j].position).norm()).sum();
            let link0 = (verts[springs[0].i].position - verts[springs[0].j].position).norm() - springs[0].rest;
            (total, link0)
        };

        // Plain VBD reaches the elastic solution — this is the reference, and it must stay exact.
        let (t_vbd, _) = settle(false, 64);
        assert!((t_vbd - exact).abs() < 1e-4, "VbdSolver should reach {exact}, got {t_vbd}");

        // AVBD does not, and is short by ~19.5% because it is solving C = 0 instead.
        let (t_avbd, link0_avbd) = settle(true, 4);
        assert!(
            t_avbd < exact - 0.2,
            "AVBD is expected to be ~19.5% SHORT of the elastic solution ({exact}); it measured {t_avbd}. If this \
             now matches, AugmentedVbd was changed to solve the elastic problem and the module doc needs updating."
        );
        assert!(t_avbd > n as f64 * link - 1e-6, "but not shorter than fully inextensible");
        // link 0 carries the most tension, so it is where the discrepancy is largest: ~10x too little extension
        let correct_link0 = n as f64 * mass * g / k;
        assert!(
            link0_avbd < correct_link0 / 5.0,
            "link 0 should extend far less than the spring law's {correct_link0}, got {link0_avbd}"
        );
        // and the clamp bound is not what is holding it: it sits above the true maximum tension
        assert!(k * link > n as f64 * mass * g, "clamp bound {} exceeds true max tension {}", k * link, n as f64 * mass * g);
    }

    /// **The defining claim: stable at one iteration with a large timestep and stiff springs.** Explicit Euler is run
    /// on the same system so the comparison is measured rather than asserted.
    #[test]
    fn vbd_is_stable_at_one_iteration_where_explicit_euler_explodes() {
        let (dt, k) = (1.0 / 30.0, 1e6); // a graphics timestep and a stiffness explicit Euler cannot survive
        let solver = VbdSolver::new(dt, 1).unwrap();

        let (mut vbd_verts, springs) = hanging_chain(12, 0.1, 0.05, k);
        for _ in 0..600 {
            solver.step(&mut vbd_verts, &springs);
        }
        let vbd_extent = max_extent(&vbd_verts);

        let (mut exp_verts, _) = hanging_chain(12, 0.1, 0.05, k);
        for _ in 0..600 {
            explicit_step(&mut exp_verts, &springs, dt, solver.gravity);
        }
        let exp_extent = max_extent(&exp_verts);

        eprintln!("dt = {dt:.4} s, stiffness {k:.0e}, 12-link chain, 600 steps:");
        eprintln!("    VBD (1 iteration): max extent {vbd_extent:.4} m, finite = {}", vbd_extent.is_finite());
        eprintln!("    explicit Euler:    max extent {exp_extent:.3e} m, finite = {}", exp_extent.is_finite());
        assert!(vbd_extent.is_finite() && vbd_extent < 10.0, "VBD stays bounded: {vbd_extent}");
        assert!(!exp_extent.is_finite() || exp_extent > 1e3, "explicit Euler does not: {exp_extent:.3e}");
        assert!(exp_extent.is_nan(), "and it diverges to NaN, not merely to something large: {exp_extent}");
        // Stable is not the same as accurate. One sweep cannot enforce a 1e6 chain, so measure the overstretch
        // against the rest length and show what sweeps buy - a chain of 12 x 0.1 m links reaches 1.200 m rigid.
        eprintln!("\n    stability is not accuracy. Overstretch past the 1.200 m rest length, by sweep count");
        eprintln!("    (with damping, so the chain settles instead of swinging):");
        for iters in [1usize, 4, 16, 64] {
            let mut sv = VbdSolver::new(dt, iters).unwrap();
            sv.damping_rate = 1.578_947_368_421_052_6; // old per-step 0.05 at dt = 1/30
            let (mut vs, sp) = hanging_chain(12, 0.1, 0.05, k);
            for _ in 0..1200 {
                sv.step(&mut vs, &sp);
            }
            let ext = max_extent(&vs);
            eprintln!("      {iters:>3} sweeps: {ext:.4} m  ({:+.1}% of rest length)", 100.0 * (ext / 1.2 - 1.0));
            assert!(ext.is_finite(), "stable at every sweep count");
        }
        let mut sv = VbdSolver::new(dt, 64).unwrap();
        sv.damping_rate = 1.578_947_368_421_052_6; // old per-step 0.05 at dt = 1/30
        let (mut vs, sp) = hanging_chain(12, 0.1, 0.05, k);
        for _ in 0..1200 {
            sv.step(&mut vs, &sp);
        }
        assert!((max_extent(&vs) - 1.2).abs() < 0.05, "with enough sweeps it hangs at its rest length: {:.4}", max_extent(&vs));
    }

    /// **It converges to implicit Euler.** More sweeps drive the variational residual down; that is the whole
    /// justification for calling the one-sweep result an approximation rather than a different model.
    #[test]
    fn more_sweeps_converge_to_the_implicit_euler_solution() {
        let (verts0, springs) = hanging_chain(8, 0.1, 0.05, 1e4);
        eprintln!("residual ||grad E||_inf after one step from rest, by sweep count:");
        let mut last = f64::INFINITY;
        for iters in [1usize, 2, 4, 8, 16, 32, 64, 256, 1024] {
            let solver = VbdSolver::new(1.0 / 60.0, iters).unwrap();
            let mut verts = verts0.clone();
            let y = solver.inertial_positions(&verts);
            solver.step(&mut verts, &springs);
            let x: Vec<Vector3<f64>> = verts.iter().map(|v| v.position).collect();
            let r = solver.residual(&x, &y, &verts, &springs);
            eprintln!("    {iters:>3} sweeps: residual {r:.4e}");
            assert!(r < last, "each doubling reduces the residual: {r:.4e} vs {last:.4e}");
            last = r;
        }
        // The early rate is first order: the residual roughly halves per doubling, which is block Gauss-Seidel on a
        // chain - information moves one vertex per sweep. Past ~64 sweeps on an 8-link chain it accelerates sharply
        // (3.7e-4 -> 4.9e-9 over 4x the sweeps). The slow early phase is VBD's known weakness and why AVBD exists.
        eprintln!("   1024 sweeps: {last:.4e}. Roughly 2x per doubling while the sweep count is comparable to the");
        eprintln!("   chain length, then much faster once information has crossed the mesh several times.");
        assert!(last < 1e-5, "it does reach the implicit-Euler solution, given enough sweeps: {last:.4e}");
    }

    /// The PSD projection is what buys the stability, so check it actually fires: a compressed spring has an
    /// indefinite exact Hessian and a projected one that is not.
    #[test]
    fn the_projection_fires_exactly_in_compression() {
        let (rest, k) = (1.0, 100.0);
        // compressed: ||d|| = 0.5 < rest
        let h = spring_hessian_psd(Vector3::new(0.5, 0.0, 0.0), Vector3::zeros(), rest, k);
        let eigs = h.symmetric_eigenvalues();
        eprintln!("compressed (||d|| = 0.5, rest 1.0): projected eigenvalues {:?}", eigs.as_slice());
        assert!(eigs.iter().all(|e| *e >= -1e-12), "projected block is PSD");
        // the exact transverse coefficient here is 1 - 1/0.5 = -1, so two eigenvalues would have been -100
        assert!(eigs.iter().filter(|e| e.abs() < 1e-9).count() == 2, "the two transverse directions are zeroed");

        // stretched: nothing is projected away
        let h2 = spring_hessian_psd(Vector3::new(2.0, 0.0, 0.0), Vector3::zeros(), rest, k);
        let eigs2 = h2.symmetric_eigenvalues();
        eprintln!("  stretched (||d|| = 2.0): eigenvalues {:?}", eigs2.as_slice());
        assert!(eigs2.iter().all(|e| *e > 1.0), "in extension the exact Hessian is already PSD");
    }

    /// **AVBD earns its keep on a stiffness ratio**, which is the case it was introduced for. Both solvers get the
    /// same sweep budget; the question is which one enforces the stiff constraint.
    #[test]
    fn avbd_handles_a_stiffness_ratio_that_slows_plain_vbd() {
        // a chain where alternating links are 1e7 and 1e2 - a 100000x ratio
        let (verts0, mut springs) = hanging_chain(10, 0.1, 0.05, 1e7);
        for (i, s) in springs.iter_mut().enumerate() {
            s.stiffness = if i % 2 == 0 { 1e7 } else { 1e2 };
        }
        let solver = VbdSolver::new(1.0 / 60.0, 4).unwrap();

        // NOT .fold(0.0, f64::max): a diverged run would fold to 0.0 and read as PERFECT constraint enforcement,
        // and the only assertion below is that both numbers are finite.
        let stiff_error = |verts: &[Vertex]| -> f64 {
            nan_propagating_max(
                springs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == 0)
                    .map(|(_, s)| ((verts[s.i].position - verts[s.j].position).norm() - s.rest).abs()),
            )
        };

        let mut vbd_verts = verts0.clone();
        for _ in 0..300 {
            solver.step(&mut vbd_verts, &springs);
        }
        let mut avbd = AugmentedVbd::new(solver, springs.len(), 1e4).unwrap();
        let mut avbd_verts = verts0.clone();
        for _ in 0..300 {
            avbd.step(&mut avbd_verts, &springs);
        }

        let (e_vbd, e_avbd) = (stiff_error(&vbd_verts), stiff_error(&avbd_verts));
        eprintln!("alternating stiffness 1e7 / 1e2, 4 sweeps, 300 steps:");
        eprintln!("    plain VBD: worst stiff-link violation {e_vbd:.4e} m");
        eprintln!("        AVBD : worst stiff-link violation {e_avbd:.4e} m");
        assert!(e_vbd.is_finite() && e_avbd.is_finite(), "both stay stable");
        eprintln!("    ratio {:.2}x", e_vbd / e_avbd.max(1e-300));

        // The module header claims AVBD "is what makes the method usable when stiffnesses differ by orders of
        // magnitude". Until now that was PRINTED and never asserted, so the advantage could invert or vanish with
        // nothing failing. Measured 1.1125e-4 m vs 4.4060e-5 m, a 2.52x margin; assert a conservative 1.5x so the
        // claim is pinned without being brittle.
        assert!(
            e_avbd * 1.5 < e_vbd,
            "AVBD must enforce the stiff links better than plain VBD, which is the module's stated reason for it: \
             VBD {e_vbd:.4e} m vs AVBD {e_avbd:.4e} m (ratio {:.2}x, needs > 1.5x)",
            e_vbd / e_avbd.max(1e-300)
        );
    }

    /// **The projection actually produces a positive semi-definite block, in every regime.**
    ///
    /// This is the module's central claim and nothing checked it. "Projected to PSD" is asserted in the header; here
    /// the eigenvalues are computed. Three regimes matter: extension (`||d|| > rest`, transverse term positive),
    /// compression (`||d|| < rest`, the term the clamp exists for), and exact rest length.
    #[test]
    fn the_projected_hessian_is_psd_in_compression_extension_and_at_rest() {
        let (k, rest) = (1.0e4f64, 1.0f64);
        let mut saw_clamped = false;
        let mut saw_unclamped = false;
        for len in [0.05f64, 0.25, 0.5, 0.9, 0.999, 1.0, 1.001, 1.5, 3.0] {
            let (xi, xj) = (Vector3::new(len, 0.0, 0.0), Vector3::zeros());
            let h = spring_hessian_psd(xi, xj, rest, k);
            assert!((h - h.transpose()).norm() < 1e-9, "len={len}: the block must be symmetric");
            let eig = h.symmetric_eigenvalues();
            let min = eig.iter().fold(f64::INFINITY, |a, b| a.min(*b));
            assert!(min >= -1e-9, "len={len}: eigenvalue {min:.6e} is negative, so the block is NOT PSD");

            // the longitudinal eigenvalue is the stiffness itself, in every regime
            let max = eig.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            assert!((max - k).abs() < 1e-6 * k, "len={len}: longitudinal eigenvalue {max:.6e} should be the stiffness");

            // and the clamp must actually be doing something in compression: two zero eigenvalues there
            let zeros = eig.iter().filter(|e| e.abs() < 1e-6 * k).count();
            if len < rest {
                assert_eq!(zeros, 2, "len={len} is compression, so the clamp must zero both transverse directions");
                saw_clamped = true;
            } else if len > rest {
                assert_eq!(zeros, 0, "len={len} is extension, so no direction should be clamped");
                saw_unclamped = true;
            }
        }
        assert!(saw_clamped && saw_unclamped, "the sweep must exercise both sides of the rest length");

        // The degenerate branch is a FALLBACK, not a limit. As ||d|| -> 0 the transverse coefficient tends to
        // -infinity (clamped to 0) and d_hat is undefined, so k*I is a choice. Pin it as such, and pin that the
        // gradient goes to zero there rather than to something enormous.
        let h0 = spring_hessian_psd(Vector3::new(1e-14, 0.0, 0.0), Vector3::zeros(), rest, k);
        assert_eq!(h0, Matrix3::identity() * k, "the coincident-vertex fallback is k*I by choice, not by derivation");
        assert_eq!(
            spring_gradient(Vector3::new(1e-14, 0.0, 0.0), Vector3::zeros(), rest, k),
            Vector3::zeros(),
            "and the gradient is zeroed on the same branch"
        );
    }

    /// **`objective` is the quantity `residual` is the gradient of, so it must fall as sweeps rise.**
    ///
    /// `VbdSolver::objective` is public and, before this, had no caller and no test anywhere in the workspace. It is
    /// also the natural thing to reach for when showing that a solver converges ("plot the energy going down"), and
    /// nothing would have caught a sign or a mass-weighting error in it.
    #[test]
    fn the_variational_objective_falls_as_sweeps_rise() {
        let (mut verts0, springs) = hanging_chain(8, 0.1, 0.05, 1e4);
        let base = VbdSolver::new(1.0 / 60.0, 1).unwrap();
        let y = base.inertial_positions(&verts0);

        // the objective at the starting point, for scale
        let x0: Vec<Vector3<f64>> = verts0.iter().map(|v| v.position).collect();
        let e_start = base.objective(&x0, &y, &verts0, &springs);
        assert!(e_start.is_finite() && e_start > 0.0, "the fixture must start away from the minimum: {e_start}");

        let mut last_e = f64::INFINITY;
        let mut last_r = f64::INFINITY;
        for iters in [1usize, 2, 4, 8, 16, 64, 256] {
            let solver = VbdSolver::new(1.0 / 60.0, iters).unwrap();
            let mut verts = verts0.clone();
            solver.step(&mut verts, &springs);
            let x: Vec<Vector3<f64>> = verts.iter().map(|v| v.position).collect();
            let e = solver.objective(&x, &y, &verts, &springs);
            let r = solver.residual(&x, &y, &verts, &springs);
            eprintln!("  {iters:>4} sweeps: objective {e:.9e}   residual {r:.4e}");
            assert!(e.is_finite(), "{iters} sweeps: objective went non-finite");
            assert!(e <= last_e * (1.0 + 1e-9), "{iters} sweeps: objective ROSE, {last_e:.9e} -> {e:.9e}");
            assert!(r <= last_r * (1.0 + 1e-9), "{iters} sweeps: residual rose, {last_r:.4e} -> {r:.4e}");
            last_e = e;
            last_r = r;
        }
        assert!(last_e < e_start, "more sweeps must beat the starting point: {last_e:.6e} vs {e_start:.6e}");
        // and NaN in must give NaN out, same contract as residual
        verts0[3].position = Vector3::new(f64::NAN, 0.0, 0.0);
        let xn: Vec<Vector3<f64>> = verts0.iter().map(|v| v.position).collect();
        assert!(base.objective(&xn, &y, &verts0, &springs).is_nan(), "a non-finite state must not report a finite energy");
    }

    /// **Does the AVBD multiplier clamp bind, and what is it worth?**
    ///
    /// The clamp is `+/- stiffness * rest` and no test exercised it, so a snippet choosing a large penalty could trip
    /// it with nothing to say so. This reports whether it binds on the stiff fixture and pins that the multipliers
    /// stay inside their bound and finite.
    #[test]
    fn the_avbd_multipliers_stay_inside_their_clamp() {
        let (verts0, springs) = hanging_chain(10, 0.1, 0.05, 1e7);
        let solver = VbdSolver::new(1.0 / 60.0, 4).unwrap();
        for penalty in [1e2f64, 1e4, 1e8] {
            let mut avbd = AugmentedVbd::new(solver, springs.len(), penalty).unwrap();
            let mut verts = verts0.clone();
            for _ in 0..200 {
                avbd.step(&mut verts, &springs);
            }
            let mut at_bound = 0usize;
            for (si, s) in springs.iter().enumerate() {
                let bound = s.stiffness * s.rest;
                let m = avbd.multipliers[si];
                assert!(m.is_finite(), "penalty {penalty:.0e}: multiplier {si} is not finite");
                assert!(m.abs() <= bound * (1.0 + 1e-12), "penalty {penalty:.0e}: multiplier {m} escaped +/-{bound}");
                if m.abs() >= bound * (1.0 - 1e-9) {
                    at_bound += 1;
                }
            }
            let ext = max_extent(&verts);
            eprintln!("  penalty {penalty:>7.0e}: {at_bound} of {} multipliers at the clamp, max extent {ext:.4} m", springs.len());
            assert!(ext.is_finite(), "penalty {penalty:.0e}: the solve went non-finite");

            // Finiteness is NOT the bar. The first version of this test asserted only is_finite() and passed on
            // 23192.7883 m of finite, on a chain whose rest total is 1.0 m. Assert the physical bound, and assert
            // that an under-sized penalty really does fail it, so the rule is measured rather than assumed.
            let rest_total = 10.0 * 0.1;
            if penalty / 1e7 >= 1.0 {
                assert!(ext <= 3.0 * rest_total, "penalty {penalty:.0e} is adequate, so the chain must stay physical: {ext:.4} m");
            } else {
                assert!(ext > 3.0 * rest_total, "penalty {penalty:.0e} is 1e7/{:.0e} of the stiffness and should DIVERGE; if this now passes, the threshold moved and the doc table is stale", 1e7 / penalty);
            }
        }
    }

    /// **A diverged solve must not report a residual of zero.**
    ///
    /// `residual` folded with `.fold(0.0, f64::max)`, and `f64::max` returns the non-NaN argument, so a state full of
    /// `NaN` reduced to `0.0` — indistinguishable from the exact implicit-Euler solution, which is what the doc comment
    /// says zero means. The convergence test's `r < last` and `last < 1e-5` would both pass on that. This pins the
    /// direction of the failure: not-finite in must give not-finite out.
    #[test]
    fn a_diverged_state_gives_a_nan_residual_not_a_zero_one() {
        // the fold itself, first: this is the behaviour the whole fix rests on
        assert_eq!(
            [f64::NAN, f64::NAN].iter().copied().fold(0.0, f64::max),
            0.0,
            "f64::max returns the non-NaN argument, which is why the naive fold hides divergence"
        );
        assert!(nan_propagating_max([f64::NAN, f64::NAN]).is_nan(), "the shared reduction must propagate it");
        assert!(nan_propagating_max([1.0, f64::NAN, 2.0]).is_nan(), "one NaN anywhere poisons the maximum");
        assert_eq!(nan_propagating_max([1.0, 3.0, 2.0]), 3.0, "and it is still a maximum on finite input");
        assert_eq!(nan_propagating_max(std::iter::empty()), 0.0, "empty reduces to the identity");

        let solver = VbdSolver::new(1.0 / 60.0, 4).unwrap();
        let (mut verts, springs) = hanging_chain(6, 0.1, 0.05, 1e4);

        // a finite state gives a finite, positive residual — prove the fixture is live before trusting the NaN case
        let y = solver.inertial_positions(&verts);
        let x: Vec<Vector3<f64>> = verts.iter().map(|v| v.position).collect();
        let clean = solver.residual(&x, &y, &verts, &springs);
        assert!(clean.is_finite() && clean > 0.0, "the fixture must have a real residual to compare against: {clean}");

        // now poison one free vertex, as a diverged step would
        verts[3].position = Vector3::new(f64::NAN, f64::NAN, f64::NAN);
        let y = solver.inertial_positions(&verts);
        let x: Vec<Vector3<f64>> = verts.iter().map(|v| v.position).collect();
        let poisoned = solver.residual(&x, &y, &verts, &springs);
        assert!(
            poisoned.is_nan(),
            "a non-finite state must give a non-finite residual, got {poisoned} (0.0 would read as converged)"
        );

        // And the whole point: it must not look BETTER than the healthy state. Stating it as a partial-order
        // comparison, so the NaN case is explicit rather than riding on `!(NaN < x)` being true by accident:
        // an unordered result is the correct answer here, and a `Less` would be the bug.
        assert_ne!(
            poisoned.partial_cmp(&clean),
            Some(std::cmp::Ordering::Less),
            "a diverged residual ({poisoned}) must never order below a healthy one ({clean:.4e}); before the fix it \
             was 0.0, which ordered below every real residual and read as perfect convergence"
        );
        eprintln!("residual: healthy {clean:.4e} N, one NaN vertex -> {poisoned} (was 0.0 before the fix)");
    }

    /// Damping must be a rate, so `dt` stays an accuracy knob rather than a physical one.
    ///
    /// This is the check the solver did not have. `damping` was a per-step fraction, dissipating
    /// `-ln(1-d)/dt` per second, so a shorter step damped harder. Every existing test ran a single
    /// `dt`, where a per-step decay and a rate are indistinguishable.
    #[test]
    fn damping_is_a_rate_not_a_per_step_fraction() {
        let tip = |dt: f64| -> f64 {
            let n = 10;
            let mut verts: Vec<Vertex> = (0..n).map(|i| Vertex::new(Vector3::new(i as f64 * 0.1, 0.0, 0.0), 1.0)).collect();
            verts[0] = Vertex::pinned_at(Vector3::zeros());
            let springs: Vec<Spring> = (0..n - 1).map(|i| Spring { i, j: i + 1, rest: 0.1, stiffness: 5.0e3 }).collect();
            let mut s = VbdSolver::new(dt, 8).expect("valid solver");
            s.damping_rate = 20.0;
            for _ in 0..(0.5 / dt).round() as usize {
                s.step(&mut verts, &springs);
            }
            verts.last().expect("chain is non-empty").position.z
        };
        // sampled mid-transient, where damping strength actually shows
        let reference = tip(2e-3);
        for dt in [1e-3f64, 5e-4, 1e-4] {
            let got = tip(dt);
            let drift = (got - reference).abs() / reference.abs();
            assert!(
                drift < 0.05,
                "dt = {dt:e} moved the mid-transient tip {reference:.6e} -> {got:.6e} ({:.1}%)",
                100.0 * drift
            );
        }
    }

}
