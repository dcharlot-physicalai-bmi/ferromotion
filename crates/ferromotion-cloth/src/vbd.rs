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
    /// Velocity damping applied at the end of a step, in `[0, 1]`.
    pub damping: f64,
}

impl VbdSolver {
    pub fn new(dt: f64, iterations: usize) -> Option<VbdSolver> {
        (dt > 0.0 && iterations >= 1).then_some(VbdSolver { dt, iterations, gravity: Vector3::new(0.0, 0.0, -9.81), damping: 0.0 })
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
                v.velocity = (x[i] - previous[i]) / self.dt * (1.0 - self.damping);
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
                v.velocity = (x[i] - previous[i]) / self.solver.dt * (1.0 - self.solver.damping);
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
            sv.damping = 0.05;
            let (mut vs, sp) = hanging_chain(12, 0.1, 0.05, k);
            for _ in 0..1200 {
                sv.step(&mut vs, &sp);
            }
            let ext = max_extent(&vs);
            eprintln!("      {iters:>3} sweeps: {ext:.4} m  ({:+.1}% of rest length)", 100.0 * (ext / 1.2 - 1.0));
            assert!(ext.is_finite(), "stable at every sweep count");
        }
        let mut sv = VbdSolver::new(dt, 64).unwrap();
        sv.damping = 0.05;
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

        // and the whole point: it must not look better than the healthy state
        assert!(!(poisoned < clean), "a diverged residual must never compare as smaller than a healthy one");
        eprintln!("residual: healthy {clean:.4e} N, one NaN vertex -> {poisoned} (was 0.0 before the fix)");
    }
}
