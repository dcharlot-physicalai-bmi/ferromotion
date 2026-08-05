//! **Correct gradients through contact** — what a differentiable simulator gets wrong, measured, and the fix.
//!
//! Every GPU-differentiable robotics simulator resolves contact with a penalty (a stiff spring in the gap) because a
//! penalty is smooth and therefore differentiable. The 2026 literature on this is blunt about the consequence:
//! realistically simulating hard contacts requires stiff solver settings, and stiff settings give **incorrect**
//! gradients under automatic differentiation, while the soft settings that give usable gradients widen the sim-to-real
//! gap. That is a genuine open problem, and it is stated as a trade-off: accuracy of the forward model against
//! accuracy of its derivative.
//!
//! It is not a trade-off. It is a consequence of differentiating the wrong object — and, as
//! [`crate::adaptive_contact`] later established by measurement, the wrong object is the **fixed-step integrator**,
//! not the penalty contact. See the correction at the end of this comment before relying on the attribution below.
//!
//! A rigid impact is a discontinuous transition, and the derivative of a trajectory through one is not the derivative
//! of any smooth approximation to it — it is the **saltation matrix** ([`saltation_matrix`](crate::saltation_matrix)),
//! which carries an extra rank-one term accounting for the fact that perturbing the state also moves the *time* of
//! impact. Penalty autodiff has no way to represent that term. It approximates it by resolving the impact over
//! several timesteps, which works only while the number of those timesteps varies smoothly with the initial
//! condition, and at realistic stiffness it does not: it varies in integer jumps.
//!
//! This module computes all three routes on the same system and reports the disagreement:
//!
//! 1. [`BouncingMass::jacobian_saltation`] — event-driven flow with the saltation matrix at the impact. Exact.
//! 2. [`PenaltyMass::jacobian_autodiff`] — the product of per-step Jacobians, which is what reverse-mode
//!    autodiff through a penalty simulator computes, to the last bit.
//! 3. [`BouncingMass::jacobian_finite_difference`] — the oracle, since neither of the above is allowed to be
//!    assumed correct.
//!
//! The comparison is only fair if both models implement the same physics, so [`PenaltyMass::effective_restitution`]
//! measures what restitution a given `(stiffness, damping)` pair actually realises, and the rigid model is given
//! that value rather than a nominal one. Comparing gradients of two different dynamics would prove nothing.
//!
//! **What the measurement found** (`examples/contact_gradient_audit.rs`, and asserted below). The trade-off as usually
//! stated is not a trade-off between two errors. The forward error shrinks with stiffness, and the penalty gradient
//! **diverges** with it: measured exponent `0.5127` against an exact `sqrt(stiffness)`, with the wrong sign at 5 of 7
//! stiffness settings. Uniform refinement of the timestep does not help: measured within `0.93x` to `1.24x` of the
//! fixed-step error.
//!
//! # Correction to the attribution above
//!
//! Those numbers stand. The explanation originally attached to them did not, and
//! [`crate::adaptive_contact`] is the instrument that overturned it. Three findings, each measured there:
//!
//! - **There is a limit, and it is the exact answer.** The earlier reasoning held that a spring of stiffness `k`
//!   amplifies the entry perturbation by `pi*sqrt(k)` and so has no limit to converge to. The *converged* Jacobian of
//!   the continuous penalty ODE approaches the rigid saltation Jacobian as roughly `1/k`. The penalty model is not
//!   the source of the divergence.
//! - **The divergence belongs to fixed-step semi-implicit Euler autodiff.** Decomposing the error into a
//!   discretisation part and a model part gives `1.5e2` against `1.7e-2` at `k = 1e6`: the integrator accounts for the
//!   whole thing.
//! - **A tolerance-driven stepper recovers the exact gradient.** ~519 steps reproduce the closed-form saltation
//!   Jacobian to three digits. So "uniform refinement does not help" is true and "adaptive integration does not help"
//!   is false; the earlier test of the published remedy shrank `dt` at a fixed ratio to `sqrt(k)`, which is uniform
//!   refinement, and never exercised a step-size controller.
//!
//! What survives is the practical recommendation, for a different reason than first given: differentiating a
//! fixed-step penalty rollout gives a gradient that is wrong by more than the magnitude of the Jacobian itself, and
//! the routes out are the exact event-driven Jacobian below ([`crate::hybrid_gradient`] generalises it) or a
//! tolerance-driven rollout. Softening the contact is neither.
//!
//! **Scope.** Demonstrated for a linear spring-damper penalty under semi-implicit Euler. A specific simulator's
//! soft-constraint formulation is a different function and its constants will differ; the claim asserted here is
//! about the penalty family measured, not about any named implementation.

use nalgebra::{DMatrix, DVector};

/// Planar gravity, m/s². Only used as a default.
pub const GRAVITY: f64 = 9.81;

/// A unit mass falling onto a plane, with a rigid impact law. The smallest system that has a contact event, and one
/// where every gradient route is available in closed form — which is why it is the right place to settle the question.
///
/// State is `[height, velocity]`.
#[derive(Clone, Copy, Debug)]
pub struct BouncingMass {
    pub gravity: f64,
    pub restitution: f64,
}

impl BouncingMass {
    pub fn new(gravity: f64, restitution: f64) -> Option<BouncingMass> {
        (gravity > 0.0 && (0.0..=1.0).contains(&restitution)).then_some(BouncingMass { gravity, restitution })
    }

    /// Time until the mass next reaches the plane from `[h, v]`, or `None` if it never does.
    pub fn time_to_impact(&self, x: [f64; 2]) -> Option<f64> {
        let [h, v] = x;
        if h < 0.0 {
            return None;
        }
        // h + v t - g t^2 / 2 = 0, taking the positive root
        let disc = v * v + 2.0 * self.gravity * h;
        (disc >= 0.0).then(|| (v + disc.sqrt()) / self.gravity).filter(|t| *t > 0.0)
    }

    /// Exact event-driven flow: integrate analytically to the guard, apply the impact law, continue. No timestep, so
    /// no timestep error, which is what makes this usable as a reference.
    ///
    /// Returns the state at `t` and how many impacts occurred.
    pub fn flow(&self, x0: [f64; 2], t: f64) -> ([f64; 2], usize) {
        let mut x = x0;
        let mut remaining = t;
        let mut events = 0usize;
        // a bound, not a convergence criterion: a restitution below 1 gives infinitely many impacts in finite time
        for _ in 0..10_000 {
            match self.time_to_impact(x) {
                Some(ti) if ti < remaining => {
                    x = self.free_flight(x, ti);
                    x[1] *= -self.restitution;
                    x[0] = 0.0;
                    remaining -= ti;
                    events += 1;
                }
                _ => break,
            }
        }
        (self.free_flight(x, remaining), events)
    }

    fn free_flight(&self, x: [f64; 2], t: f64) -> [f64; 2] {
        [x[0] + x[1] * t - 0.5 * self.gravity * t * t, x[1] - self.gravity * t]
    }

    /// The smooth flow's Jacobian over `t`: constant acceleration, so `[[1, t], [0, 1]]`.
    fn flight_jacobian(t: f64) -> [[f64; 2]; 2] {
        [[1.0, t], [0.0, 1.0]]
    }

    /// **The saltation matrix at an impact from velocity `v`.**
    ///
    /// Built through [`saltation_matrix`](crate::saltation_matrix) rather than hand-derived, so the general routine is
    /// what gets exercised. For this system it comes out as
    ///
    /// ```text
    ///   Xi = [ [ -e,               0  ],
    ///          [ -(1+e) g / v,    -e  ] ]
    /// ```
    ///
    /// The off-diagonal term is the whole story: it is unbounded as `v -> 0`, because a grazing impact's *timing* is
    /// arbitrarily sensitive to the state. No smooth approximation of the contact contains this term, and it does not
    /// appear as an error in the forward trajectory — only in its derivative.
    pub fn saltation(&self, v_minus: f64) -> Option<[[f64; 2]; 2]> {
        if v_minus >= 0.0 {
            return None; // not an impact
        }
        let reset = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -self.restitution]);
        let guard_normal = DVector::from_row_slice(&[1.0, 0.0]); // g(x) = h
        let f_minus = DVector::from_row_slice(&[v_minus, -self.gravity]);
        let f_plus = DVector::from_row_slice(&[-self.restitution * v_minus, -self.gravity]);
        let xi = crate::saltation_matrix(&reset, &guard_normal, &f_minus, &f_plus)?;
        Some([[xi[(0, 0)], xi[(0, 1)]], [xi[(1, 0)], xi[(1, 1)]]])
    }

    /// **The exact Jacobian** `d x(t) / d x0` for a trajectory with impacts: `Phi_after · Xi · Phi_before` at each
    /// event, composed in order.
    pub fn jacobian_saltation(&self, x0: [f64; 2], t: f64) -> Option<[[f64; 2]; 2]> {
        let mut jac = [[1.0, 0.0], [0.0, 1.0]];
        let mut x = x0;
        let mut remaining = t;
        for _ in 0..10_000 {
            match self.time_to_impact(x) {
                Some(ti) if ti < remaining => {
                    let pre = self.free_flight(x, ti);
                    jac = mat_mul(Self::flight_jacobian(ti), jac);
                    jac = mat_mul(self.saltation(pre[1])?, jac);
                    x = [0.0, -self.restitution * pre[1]];
                    remaining -= ti;
                }
                _ => break,
            }
        }
        Some(mat_mul(Self::flight_jacobian(remaining), jac))
    }

    /// Central differences on the event-driven flow — the oracle. Perturbations that change the impact *count* are
    /// rejected, because across such a perturbation the map genuinely is discontinuous and no Jacobian exists.
    pub fn jacobian_finite_difference(&self, x0: [f64; 2], t: f64, h: f64) -> Option<[[f64; 2]; 2]> {
        let (_, base_events) = self.flow(x0, t);
        let mut jac = [[0.0; 2]; 2];
        for j in 0..2 {
            let (mut plus, mut minus) = (x0, x0);
            plus[j] += h;
            minus[j] -= h;
            let (fp, ep) = self.flow(plus, t);
            let (fm, em) = self.flow(minus, t);
            if ep != base_events || em != base_events {
                return None; // the perturbation crossed an event count: no derivative here
            }
            for i in 0..2 {
                jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        Some(jac)
    }
}

/// A penalty contact: a stiff spring-damper in the gap, integrated at a fixed timestep with semi-implicit Euler. This
/// is what a differentiable simulator actually integrates, and what autodiff actually differentiates.
#[derive(Clone, Copy, Debug)]
pub struct PenaltyMass {
    pub gravity: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub dt: f64,
}

impl PenaltyMass {
    pub fn new(gravity: f64, stiffness: f64, damping: f64, dt: f64) -> Option<PenaltyMass> {
        (gravity > 0.0 && stiffness > 0.0 && damping >= 0.0 && dt > 0.0).then_some(PenaltyMass { gravity, stiffness, damping, dt })
    }

    /// Contact force at `[h, v]`: zero above the plane, a stiff spring-damper below it. One-sided damping so the
    /// contact cannot pull the mass back down.
    fn force(&self, x: [f64; 2]) -> f64 {
        let [h, v] = x;
        if h >= 0.0 {
            0.0
        } else {
            (-self.stiffness * h - self.damping * v).max(0.0)
        }
    }

    /// One semi-implicit Euler step.
    pub fn step(&self, x: [f64; 2]) -> [f64; 2] {
        let a = -self.gravity + self.force(x);
        let v = x[1] + a * self.dt;
        [x[0] + v * self.dt, v]
    }

    /// The per-step Jacobian, exactly as reverse-mode autodiff would accumulate it.
    fn step_jacobian(&self, x: [f64; 2]) -> [[f64; 2]; 2] {
        let (dh, dv) = if x[0] >= 0.0 || (-self.stiffness * x[0] - self.damping * x[1]) <= 0.0 {
            (0.0, 0.0) // the branch autodiff takes: no contact, no contribution
        } else {
            (-self.stiffness, -self.damping)
        };
        // v' = v + dt*(-g + F(h, v)),  h' = h + dt*v'
        let dv_dh = dh * self.dt;
        let dv_dv = 1.0 + dv * self.dt;
        [[1.0 + self.dt * dv_dh, self.dt * dv_dv], [dv_dh, dv_dv]]
    }

    pub fn rollout(&self, x0: [f64; 2], t: f64) -> [f64; 2] {
        let steps = (t / self.dt).round() as usize;
        (0..steps).fold(x0, |x, _| self.step(x))
    }

    /// **The Jacobian autodiff produces**: the product of per-step Jacobians along the realised trajectory. Bit-exact
    /// with what a tape would return, which is the point — no approximation is being blamed on the tape.
    pub fn jacobian_autodiff(&self, x0: [f64; 2], t: f64) -> [[f64; 2]; 2] {
        let steps = (t / self.dt).round() as usize;
        let mut x = x0;
        let mut jac = [[1.0, 0.0], [0.0, 1.0]];
        for _ in 0..steps {
            jac = mat_mul(self.step_jacobian(x), jac);
            x = self.step(x);
        }
        jac
    }

    /// **What restitution this penalty model actually realises** at a given impact speed, measured by dropping the
    /// mass and reading the rebound. Necessary for a fair comparison: a nominal restitution is not what the model does.
    pub fn effective_restitution(&self, impact_speed: f64) -> Option<f64> {
        if impact_speed <= 0.0 {
            return None;
        }
        let mut x = [0.0, -impact_speed];
        // run until the mass is airborne again and rising
        for _ in 0..1_000_000 {
            x = self.step(x);
            if x[0] > 0.0 && x[1] > 0.0 {
                // undo the height gained so the comparison is at the plane
                let v_at_plane = (x[1] * x[1] + 2.0 * self.gravity * x[0]).sqrt();
                return Some(v_at_plane / impact_speed);
            }
            if x[0] > 0.0 && x[1] <= 0.0 {
                return Some(0.0); // never rebounded
            }
        }
        None
    }
}

fn mat_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut c = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    c
}

/// Largest absolute entrywise difference between two 2x2 Jacobians.
pub fn jacobian_error(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> f64 {
    (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| (a[i][j] - b[i][j]).abs()).fold(0.0, f64::max)
}

/// Relative error, scaled by the magnitude of the reference — the honest statistic when entries span decades.
pub fn jacobian_relative_error(candidate: [[f64; 2]; 2], reference: [[f64; 2]; 2]) -> f64 {
    let scale = (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| reference[i][j].abs()).fold(0.0, f64::max);
    if scale <= 0.0 {
        return jacobian_error(candidate, reference);
    }
    jacobian_error(candidate, reference) / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The saltation matrix must agree with finite differences on the exact flow. Until this passes, nothing else in
    /// the module means anything.
    #[test]
    fn the_saltation_jacobian_is_the_true_jacobian() {
        let m = BouncingMass::new(GRAVITY, 0.6).unwrap();
        for (h, v, t) in [(1.0, 0.0, 0.8), (2.0, 1.0, 1.2), (0.5, -1.0, 0.5), (1.5, 2.0, 1.0)] {
            let exact = m.jacobian_saltation([h, v], t).unwrap();
            let fd = m.jacobian_finite_difference([h, v], t, 1e-7).expect("no event boundary here");
            let err = jacobian_relative_error(exact, fd);
            eprintln!("h={h} v={v} t={t}: saltation vs finite difference, relative error {err:.3e}");
            assert!(err < 1e-6, "saltation must match the oracle: {err:.3e}");
        }
    }

    /// **The penalty gradient diverges as `sqrt(stiffness)`.** It does not converge to the saltation matrix, or to
    /// anything else, so there is no stiffness at which penalty autodiff is right.
    #[test]
    fn the_penalty_gradient_diverges_as_root_stiffness() {
        let (h0, t, zeta) = (1.0, 0.8, 0.1606);
        let mut points = Vec::new();
        for exp in [4i32, 6, 8] {
            let k = 10f64.powi(exp);
            // dt shrinks with the contact duration, so this is the ADAPTIVE case: no discretisation excuse
            let pen = PenaltyMass::new(GRAVITY, k, 2.0 * zeta * k.sqrt(), 1e-4 * (100.0 / k).sqrt()).unwrap();
            points.push((k, pen.jacobian_autodiff([h0, 0.0], t)[1][0].abs()));
        }
        for w in points.windows(2) {
            let slope = (w[1].1 / w[0].1).ln() / (w[1].0 / w[0].0).ln();
            eprintln!("k {:.0e} -> {:.0e}: |grad| {:.2} -> {:.2}, slope {slope:.3}", w[0].0, w[1].0, w[0].1, w[1].1);
            assert!((slope - 0.5).abs() < 0.05, "the exponent is 1/2: {slope:.4}");
        }
        let overall = (points[2].1 / points[0].1).ln() / (points[2].0 / points[0].0).ln();
        eprintln!("   overall slope {overall:.4} against an exact 0.5 - so the gradient has no limit");
        // 0.5213 over 1e4..1e8; the residual above 1/2 is the finite-duration correction, which shrinks with k
        assert!((overall - 0.5).abs() < 0.03, "{overall:.4}");
    }

    /// And the practical consequence: at realistic stiffness the gradient points the wrong way.
    #[test]
    fn the_penalty_gradient_has_the_wrong_sign_at_realistic_stiffness() {
        let (h0, t, zeta, dt) = (1.0, 0.8, 0.1606, 1e-4);
        let exact = BouncingMass::new(GRAVITY, 0.6).unwrap().jacobian_saltation([h0, 0.0], t).unwrap();
        let mut wrong = 0usize;
        for exp in 2..=8 {
            let k = 10f64.powi(exp);
            let pen = PenaltyMass::new(GRAVITY, k, 2.0 * zeta * k.sqrt(), dt).unwrap();
            if pen.jacobian_autodiff([h0, 0.0], t)[1][0].signum() != exact[1][0].signum() {
                wrong += 1;
            }
        }
        eprintln!("{wrong} of 7 stiffness settings give a gradient of the wrong sign (exact entry {:+.4})", exact[1][0]);
        assert_eq!(wrong, 5, "a wrong sign sends gradient descent away from the optimum");
    }

    /// The off-diagonal saltation term is unbounded as the impact grazes. This is the term no smooth contact model has.
    #[test]
    fn the_timing_term_blows_up_at_a_grazing_impact() {
        let m = BouncingMass::new(GRAVITY, 0.6).unwrap();
        eprintln!("impact speed -> saltation entry Xi[1][0] (the time-of-impact term):");
        let mut last = 0.0f64;
        for v in [-4.0, -2.0, -1.0, -0.5, -0.1, -0.01] {
            let xi = m.saltation(v).unwrap();
            eprintln!("    v = {v:>6}: Xi[1][0] = {:>14.2}, Xi[0][0] = {:.4}", xi[1][0], xi[0][0]);
            assert!(xi[1][0].abs() > last.abs(), "monotone in 1/|v|");
            last = xi[1][0];
            assert!((xi[0][0] + m.restitution).abs() < 1e-12, "Xi[0][0] = -e");
        }
        assert!(last.abs() > 1e3, "unbounded as the impact grazes: {last:.2}");
    }
}
