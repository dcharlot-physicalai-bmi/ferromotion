//! Tolerance-driven adaptive integration of a penalty contact, and the error decomposition it makes possible.
//!
//! # Why this module exists
//!
//! The published remedy for bad penalty-contact gradients is adaptive time integration: replace the fixed-step
//! integrator with an embedded-error-estimate solver under a tolerance, and the reported gradient error falls by
//! orders of magnitude for the same compute budget.
//!
//! [`crate::contact_gradient`] reports that a penalty gradient diverges as `sqrt(k)` and that refining the timestep
//! does not rescue it. Those two statements look like a contradiction. They are not, and this module is the
//! instrument that shows why: **they measure different error terms against different references.**
//!
//! Write the gradient a differentiable simulator returns as `J_fixed(k, dt)`, and introduce two references:
//!
//! - `J_penalty(k)` — the Jacobian of the *continuous* penalty ODE, converged. Obtained here by central differences
//!   of an adaptive rollout at a tolerance tight enough that the answer stops moving. This is the reference the
//!   adaptive-integration literature measures against.
//! - `J_rigid` — the exact saltation Jacobian of the *rigid* (complementarity) system, at the restitution the
//!   penalty model actually realises. This is the reference [`crate::contact_gradient`] measures against.
//!
//! The total error then splits, exactly, into two terms that no amount of integration effort can mix:
//!
//! ```text
//! J_fixed - J_rigid  =  (J_fixed - J_penalty)  +  (J_penalty - J_rigid)
//!                        \_______  _______/       \_______  _______/
//!                          discretisation            model error
//! ```
//!
//! Adaptive integration attacks the first term only. The second is a property of the penalty model itself and
//! survives every integrator and every timestep. Whether it is large is a measurement, not an argument, and
//! [`decompose_gradient_error`] performs it.
//!
//! # What it measures
//!
//! On a unit mass dropped 0.5 m, damping tracking stiffness at a fixed ratio so the realised restitution holds
//! still (`examples/adaptive_gradient_decomposition.rs`):
//!
//! - **The discretisation term dominates**, by four orders of magnitude at `k = 1e6`: `1.5e2` against `1.7e-2`.
//!   At every stiffness from `1e4` up it accounts for the entire error to two decimal places.
//! - **The model term shrinks monotonically**, which is the direction that matters: it approaches the rigid saltation
//!   Jacobian rather than departing from it. The *rate* needs care. The four-point fit reports `-0.93`, but an
//!   adversarial audit showed that figure is set almost entirely by the `k = 1e3` point, which is out of the
//!   asymptotic regime twice over: its model error (`14.96`) is 280% of the largest rigid-Jacobian entry, and its
//!   decomposition is degenerate (the two error terms nearly cancel, `disc/total = 40.18`). The three well-resolved
//!   decades fit `k^-0.50` with `R^2 = 1.0000` — **`1/sqrt(k)`, not `1/k`**. Anchored at `k = 1e4`, `1/k` predicts
//!   `1.7e-3` at `k = 1e6` against a measured `1.696e-2`; `1/sqrt(k)` predicts it to 1%.
//! - **A tolerance-driven rollout recovers the exact answer.** At `k = 1e6` it reproduces the closed-form rigid
//!   saltation Jacobian in ~519 steps, verified in
//!   [`tests::adaptive_reproduces_the_integration_free_jacobian`]. Scoped honestly: the agreement is three digits on
//!   the scaled max-abs statistic the test uses; entrywise, `dv/dv0` agrees to `0.04%` and `dh/dh0` only to about
//!   1.5 digits (`-0.31926` vs `-0.30229`), and the test's pass margin is set by that worst entry.
//!
//!   The check is **integration-free only in `dv/dv0`**, where the rigid value is `1` exactly by algebra and does not
//!   depend on the restitution. The other entries are calibrated by a restitution that
//!   [`AdaptivePenalty::effective_restitution`] measures with the same adaptive rollout under test, so they are a
//!   consistency check rather than an independent one. Saying otherwise was an overstatement.
//!
//! So a bad penalty-contact gradient is the integrator's doing, not the contact model's. This corrects the
//! attribution in [`crate::contact_gradient`], whose measurement stands but whose explanation did not: what diverges
//! with stiffness is fixed-step semi-implicit Euler autodiff, not the penalty model.
//!
//! # Two traps this module exists to avoid
//!
//! Uniform-step finite differences are **unusable** at these stiffnesses. The same entry reads `5.80`, `-78.3`,
//! `694.2`, `-143.2` as the probe shrinks by decades: it measures the probe. Any conclusion drawn from a
//! uniform-step finite-difference Jacobian here is an artifact, which is why
//! [`tests::adaptive_is_probe_stable_and_uniform_is_not`] pins both behaviours.
//!
//! Central differences of an adaptive rollout carry an `O(h^2)` term that at high stiffness exceeds the model error
//! being measured against it, so [`AdaptivePenalty::jacobian_richardson`] extrapolates it away and
//! [`converged_reference`] reports what uncertainty is left. Model errors below that floor are not quotable.
//!
//! The integrator here is Dormand-Prince 5(4) rather than Tsitouras 5(4). Both are fifth-order explicit
//! Runge-Kutta pairs with an embedded fourth-order estimate driving a step-size controller, which is the mechanism
//! under test; the specific coefficient set is not.

use crate::contact_gradient::{BouncingMass, PenaltyMass, jacobian_error};

/// Controls for the adaptive stepper.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveOptions {
    /// Relative tolerance on the embedded error estimate.
    pub rtol: f64,
    /// Absolute tolerance, which keeps the controller sane as the state passes through zero.
    pub atol: f64,
    /// First trial step. The controller corrects it immediately, so it only needs to be in the right decade.
    pub dt_init: f64,
    /// Refuse to step below this. A stepper that has to go smaller is not converging, and saying so beats
    /// returning a number that took a million steps to be wrong.
    pub dt_min: f64,
    /// Hard cap on accepted plus rejected steps.
    pub max_steps: usize,
}

impl AdaptiveOptions {
    /// A tolerance-driven configuration. `dt_min` and `max_steps` are set so that a stiff contact is affordable but
    /// a non-converging one still terminates.
    pub fn with_tolerance(rtol: f64) -> AdaptiveOptions {
        AdaptiveOptions { rtol, atol: rtol * 1e-3, dt_init: 1e-6, dt_min: 1e-16, max_steps: 20_000_000 }
    }
}

/// What the stepper actually did. Reported rather than discarded, because "the tolerance was met" and "the
/// tolerance was met cheaply" are different claims and only one of them is usually being made.
#[derive(Clone, Copy, Debug, Default)]
pub struct AdaptiveStats {
    pub accepted: usize,
    pub rejected: usize,
    /// Smallest step the controller had to accept. A tiny value here is the signature of the contact kink.
    pub dt_min_used: f64,
}

/// Failure modes of the adaptive rollout. Distinguished so a caller can tell "too stiff for this budget" from
/// "the tolerance is unreachable".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveError {
    /// The controller demanded a step below `dt_min`.
    StepTooSmall,
    /// Ran out of step budget before reaching the end time.
    BudgetExhausted,
    /// The state left the reals.
    Diverged,
}

/// The penalty contact as a *continuous* ODE, with no timestep baked in.
///
/// [`PenaltyMass`](crate::contact_gradient::PenaltyMass) is the same physics fused to semi-implicit Euler, which is
/// what a simulator differentiates. Separating the two is the whole point: it makes "the model is wrong" and "the
/// integrator is coarse" independently measurable.
#[derive(Clone, Copy, Debug)]
pub struct AdaptivePenalty {
    pub gravity: f64,
    pub stiffness: f64,
    pub damping: f64,
}

impl AdaptivePenalty {
    pub fn new(gravity: f64, stiffness: f64, damping: f64) -> Option<AdaptivePenalty> {
        (gravity > 0.0 && stiffness > 0.0 && damping >= 0.0).then_some(AdaptivePenalty { gravity, stiffness, damping })
    }

    /// The same one-sided spring-damper as [`PenaltyMass`], so the two models are the same physics.
    fn force(&self, x: [f64; 2]) -> f64 {
        let [h, v] = x;
        if h >= 0.0 { 0.0 } else { (-self.stiffness * h - self.damping * v).max(0.0) }
    }

    fn rhs(&self, x: [f64; 2]) -> [f64; 2] {
        [x[1], -self.gravity + self.force(x)]
    }

    /// Integrate to `t` under a tolerance. Returns the end state and what it cost.
    // The negated comparison is deliberate: `!(t > 0.0)` rejects a NaN horizon, where `t <= 0.0` would accept it
    // and integrate for a nonsense number of steps. A NaN read as a valid measurement is the failure mode here.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn rollout(&self, x0: [f64; 2], t: f64, opts: AdaptiveOptions) -> Result<([f64; 2], AdaptiveStats), AdaptiveError> {
        if !(t > 0.0) {
            return Ok((x0, AdaptiveStats::default()));
        }
        let mut x = x0;
        let mut time = 0.0;
        let mut dt = opts.dt_init.min(t);
        let mut stats = AdaptiveStats { accepted: 0, rejected: 0, dt_min_used: f64::INFINITY };

        while time < t {
            if stats.accepted + stats.rejected >= opts.max_steps {
                return Err(AdaptiveError::BudgetExhausted);
            }
            // Never step past the end: the final step is clipped so `t` is hit exactly.
            let h = dt.min(t - time);
            if h < opts.dt_min {
                return Err(AdaptiveError::StepTooSmall);
            }
            let (y5, y4) = self.dopri_step(x, h);
            if !y5[0].is_finite() || !y5[1].is_finite() {
                return Err(AdaptiveError::Diverged);
            }

            // Scaled error norm, the standard mixed absolute/relative measure.
            let err = (0..2)
                .map(|i| {
                    let scale = opts.atol + opts.rtol * x[i].abs().max(y5[i].abs());
                    ((y5[i] - y4[i]) / scale).abs()
                })
                .fold(0.0, f64::max);

            if err <= 1.0 {
                x = y5;
                time += h;
                stats.accepted += 1;
                stats.dt_min_used = stats.dt_min_used.min(h);
            } else {
                stats.rejected += 1;
            }

            // Fifth-order controller with the usual safety factor and growth clamps.
            let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 5.0 };
            dt = h * factor.clamp(0.2, 5.0);
        }
        Ok((x, stats))
    }

    /// One Dormand-Prince 5(4) step, returning the fifth-order and embedded fourth-order states.
    fn dopri_step(&self, x: [f64; 2], h: f64) -> ([f64; 2], [f64; 2]) {
        let add = |a: [f64; 2], b: [f64; 2], s: f64| [a[0] + s * b[0], a[1] + s * b[1]];
        let k1 = self.rhs(x);
        let k2 = self.rhs(add(x, k1, h / 5.0));
        let k3 = self.rhs([
            x[0] + h * (3.0 / 40.0 * k1[0] + 9.0 / 40.0 * k2[0]),
            x[1] + h * (3.0 / 40.0 * k1[1] + 9.0 / 40.0 * k2[1]),
        ]);
        let k4 = self.rhs([
            x[0] + h * (44.0 / 45.0 * k1[0] - 56.0 / 15.0 * k2[0] + 32.0 / 9.0 * k3[0]),
            x[1] + h * (44.0 / 45.0 * k1[1] - 56.0 / 15.0 * k2[1] + 32.0 / 9.0 * k3[1]),
        ]);
        let k5 = self.rhs([
            x[0]
                + h * (19372.0 / 6561.0 * k1[0] - 25360.0 / 2187.0 * k2[0] + 64448.0 / 6561.0 * k3[0]
                    - 212.0 / 729.0 * k4[0]),
            x[1]
                + h * (19372.0 / 6561.0 * k1[1] - 25360.0 / 2187.0 * k2[1] + 64448.0 / 6561.0 * k3[1]
                    - 212.0 / 729.0 * k4[1]),
        ]);
        let k6 = self.rhs([
            x[0]
                + h * (9017.0 / 3168.0 * k1[0] - 355.0 / 33.0 * k2[0] + 46732.0 / 5247.0 * k3[0]
                    + 49.0 / 176.0 * k4[0]
                    - 5103.0 / 18656.0 * k5[0]),
            x[1]
                + h * (9017.0 / 3168.0 * k1[1] - 355.0 / 33.0 * k2[1] + 46732.0 / 5247.0 * k3[1]
                    + 49.0 / 176.0 * k4[1]
                    - 5103.0 / 18656.0 * k5[1]),
        ]);
        // The fifth-order state is the seventh stage node, so k7 is evaluated at it (FSAL structure).
        let y5 = [
            x[0] + h * (35.0 / 384.0 * k1[0] + 500.0 / 1113.0 * k3[0] + 125.0 / 192.0 * k4[0]
                - 2187.0 / 6784.0 * k5[0]
                + 11.0 / 84.0 * k6[0]),
            x[1] + h * (35.0 / 384.0 * k1[1] + 500.0 / 1113.0 * k3[1] + 125.0 / 192.0 * k4[1]
                - 2187.0 / 6784.0 * k5[1]
                + 11.0 / 84.0 * k6[1]),
        ];
        let k7 = self.rhs(y5);
        let y4 = [
            x[0]
                + h * (5179.0 / 57600.0 * k1[0] + 7571.0 / 16695.0 * k3[0] + 393.0 / 640.0 * k4[0]
                    - 92097.0 / 339200.0 * k5[0]
                    + 187.0 / 2100.0 * k6[0]
                    + k7[0] / 40.0),
            x[1]
                + h * (5179.0 / 57600.0 * k1[1] + 7571.0 / 16695.0 * k3[1] + 393.0 / 640.0 * k4[1]
                    - 92097.0 / 339200.0 * k5[1]
                    + 187.0 / 2100.0 * k6[1]
                    + k7[1] / 40.0),
        ];
        (y5, y4)
    }

    /// Central differences of the adaptive rollout: the reference the adaptive-integration literature measures
    /// against, built the way it builds it.
    ///
    /// `fd_step` perturbs each coordinate. Central differences are second-order, so the step is a genuine part of
    /// the reference and is reported by callers rather than hidden.
    pub fn jacobian_finite_difference(
        &self,
        x0: [f64; 2],
        t: f64,
        fd_step: f64,
        opts: AdaptiveOptions,
    ) -> Result<[[f64; 2]; 2], AdaptiveError> {
        let mut jac = [[0.0; 2]; 2];
        for j in 0..2 {
            let mut plus = x0;
            let mut minus = x0;
            plus[j] += fd_step;
            minus[j] -= fd_step;
            let (yp, _) = self.rollout(plus, t, opts)?;
            let (ym, _) = self.rollout(minus, t, opts)?;
            for i in 0..2 {
                jac[i][j] = (yp[i] - ym[i]) / (2.0 * fd_step);
            }
        }
        Ok(jac)
    }

    /// The same Dormand-Prince pair at a **uniform** step, with the error estimate discarded.
    ///
    /// This exists to separate two remedies that are easy to conflate. Going from fixed-step semi-implicit Euler to
    /// a tolerance-driven fifth-order pair changes two things at once: the integrator and the step control. Running
    /// the fifth-order pair at a uniform step holds the integrator fixed and removes only the adaptivity, so
    /// whichever of the two is doing the work becomes visible instead of inferred.
    // Negated comparisons again, for the same reason: a NaN horizon or step must be refused, not integrated.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn rollout_uniform(&self, x0: [f64; 2], t: f64, dt: f64) -> Option<[f64; 2]> {
        if !(t > 0.0) || !(dt > 0.0) {
            return None;
        }
        let steps = (t / dt).round().max(1.0) as usize;
        let h = t / steps as f64;
        let mut x = x0;
        for _ in 0..steps {
            x = self.dopri_step(x, h).0;
            if !x[0].is_finite() || !x[1].is_finite() {
                return None;
            }
        }
        Some(x)
    }

    /// Central differences of the uniform-step fifth-order rollout.
    pub fn jacobian_uniform(&self, x0: [f64; 2], t: f64, dt: f64, fd_step: f64) -> Option<[[f64; 2]; 2]> {
        let mut jac = [[0.0; 2]; 2];
        for j in 0..2 {
            let mut plus = x0;
            let mut minus = x0;
            plus[j] += fd_step;
            minus[j] -= fd_step;
            let yp = self.rollout_uniform(plus, t, dt)?;
            let ym = self.rollout_uniform(minus, t, dt)?;
            for i in 0..2 {
                jac[i][j] = (yp[i] - ym[i]) / (2.0 * fd_step);
            }
        }
        Some(jac)
    }

    /// Richardson-extrapolated central differences: `(4*J(h/2) - J(h)) / 3`.
    ///
    /// Central differences carry an `O(h^2)` truncation term, and at high stiffness that term is larger than the
    /// model error being measured against it. Extrapolation cancels it and leaves `O(h^4)`, which is what makes the
    /// reference tighter than the quantity under test instead of the other way round.
    pub fn jacobian_richardson(
        &self,
        x0: [f64; 2],
        t: f64,
        fd_step: f64,
        opts: AdaptiveOptions,
    ) -> Result<[[f64; 2]; 2], AdaptiveError> {
        let coarse = self.jacobian_finite_difference(x0, t, fd_step, opts)?;
        let fine = self.jacobian_finite_difference(x0, t, fd_step / 2.0, opts)?;
        let mut jac = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                jac[i][j] = (4.0 * fine[i][j] - coarse[i][j]) / 3.0;
            }
        }
        Ok(jac)
    }

    /// The restitution this continuous penalty model realises, measured by dropping the mass and reading the
    /// rebound. Needed so the rigid reference is the *same bounce*, not a nominal one.
    pub fn effective_restitution(&self, impact_speed: f64, opts: AdaptiveOptions) -> Option<f64> {
        if impact_speed <= 0.0 {
            return None;
        }
        // Integrate in short spans and inspect between them, so the airborne test sees the state near the plane.
        let span = 1e-4 / (self.stiffness.sqrt().max(1.0)) * 100.0;
        let mut x = [0.0, -impact_speed];
        for _ in 0..2_000_000 {
            let (next, _) = self.rollout(x, span, opts).ok()?;
            x = next;
            if x[0] > 0.0 && x[1] > 0.0 {
                let v_at_plane = (x[1] * x[1] + 2.0 * self.gravity * x[0]).sqrt();
                return Some(v_at_plane / impact_speed);
            }
            if x[0] > 0.0 && x[1] <= 0.0 {
                return Some(0.0);
            }
        }
        None
    }
}

/// The two error terms of a penalty-contact gradient, separated.
#[derive(Clone, Copy, Debug)]
pub struct ErrorDecomposition {
    pub stiffness: f64,
    /// The timestep the fixed-step gradient was taken at.
    pub dt: f64,
    /// `||J_fixed - J_penalty||`: what adaptive integration removes.
    pub discretisation: f64,
    /// `||J_penalty - J_rigid||`: what survives every integrator.
    pub model: f64,
    /// `||J_fixed - J_rigid||`, measured rather than assumed equal to the sum of the parts.
    pub total: f64,
    /// The converged Jacobian of the continuous penalty ODE.
    pub penalty: [[f64; 2]; 2],
    /// The exact saltation Jacobian of the rigid system at the realised restitution.
    pub rigid: [[f64; 2]; 2],
    /// The restitution the penalty model realised, which sets the rigid reference.
    pub restitution: f64,
    pub stats: AdaptiveStats,
}

impl ErrorDecomposition {
    /// How much of the total error adaptive integration can reach. The complement is the floor.
    pub fn reachable_fraction(&self) -> f64 {
        if self.total <= 0.0 { 0.0 } else { (self.discretisation / self.total).min(1.0) }
    }
}

/// Split a penalty-contact gradient error into its discretisation and model parts.
///
/// `reference_opts` should be tight enough that `J_penalty` has stopped moving; verify that with
/// [`converged_reference`] rather than trusting a tolerance to be small enough.
#[allow(clippy::too_many_arguments)]
pub fn decompose_gradient_error(
    gravity: f64,
    stiffness: f64,
    damping: f64,
    dt: f64,
    x0: [f64; 2],
    t: f64,
    fd_step: f64,
    reference_opts: AdaptiveOptions,
) -> Option<ErrorDecomposition> {
    let continuous = AdaptivePenalty::new(gravity, stiffness, damping)?;
    let fixed = PenaltyMass::new(gravity, stiffness, damping, dt)?;

    // The impact speed the rigid reference must match, taken from the drop itself.
    let impact_speed = (x0[1] * x0[1] + 2.0 * gravity * x0[0].max(0.0)).sqrt();
    let restitution = continuous.effective_restitution(impact_speed, reference_opts)?;
    let rigid = BouncingMass::new(gravity, restitution)?.jacobian_saltation(x0, t)?;

    let penalty = continuous.jacobian_richardson(x0, t, fd_step, reference_opts).ok()?;
    let (_, stats) = continuous.rollout(x0, t, reference_opts).ok()?;
    let fixed_jac = fixed.jacobian_autodiff(x0, t);

    Some(ErrorDecomposition {
        stiffness,
        dt,
        discretisation: jacobian_error(fixed_jac, penalty),
        model: jacobian_error(penalty, rigid),
        total: jacobian_error(fixed_jac, rigid),
        penalty,
        rigid,
        restitution,
        stats,
    })
}

/// Check that the converged reference has actually converged, by recomputing it at a tighter tolerance and a
/// different finite-difference step and reporting how far it moved.
///
/// A reference that moves under this check is not a reference, and every error measured against it is that
/// movement plus noise.
#[allow(clippy::too_many_arguments)]
pub fn converged_reference(
    gravity: f64,
    stiffness: f64,
    damping: f64,
    x0: [f64; 2],
    t: f64,
    fd_step: f64,
    loose: AdaptiveOptions,
    tight: AdaptiveOptions,
) -> Option<(f64, [[f64; 2]; 2])> {
    let continuous = AdaptivePenalty::new(gravity, stiffness, damping)?;
    let a = continuous.jacobian_richardson(x0, t, fd_step, loose).ok()?;
    let b = continuous.jacobian_richardson(x0, t, fd_step * 3.0, tight).ok()?;
    Some((jacobian_error(a, b), a))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Free flight is a quadratic in time, which a fifth-order pair integrates exactly. If the stepper cannot
    /// reproduce it to round-off then nothing else it reports means anything.
    #[test]
    fn free_flight_is_exact() {
        let p = AdaptivePenalty::new(9.81, 1e5, 10.0).unwrap();
        let opts = AdaptiveOptions::with_tolerance(1e-10);
        let (x, stats) = p.rollout([1.0, 0.0], 0.1, opts).unwrap();
        let expect_h = 1.0 - 0.5 * 9.81 * 0.01;
        let expect_v = -9.81 * 0.1;
        assert!((x[0] - expect_h).abs() < 1e-12, "h {} vs {}", x[0], expect_h);
        assert!((x[1] - expect_v).abs() < 1e-12, "v {} vs {}", x[1], expect_v);
        assert!(stats.accepted > 0);
    }

    /// The adaptive stepper and the fixed-step model are the same physics, so at a timestep fine enough for the
    /// fixed stepper they must agree on the trajectory. This is what makes the gradient comparison fair.
    #[test]
    fn adaptive_and_fixed_agree_on_the_trajectory() {
        let k = 1e5;
        let adaptive = AdaptivePenalty::new(9.81, k, 30.0).unwrap();
        let fixed = PenaltyMass::new(9.81, k, 30.0, 1e-7).unwrap();
        let (xa, _) = adaptive.rollout([0.5, 0.0], 0.4, AdaptiveOptions::with_tolerance(1e-11)).unwrap();
        let xf = fixed.rollout([0.5, 0.0], 0.4);
        assert!((xa[0] - xf[0]).abs() < 1e-3, "adaptive {:?} vs fixed {:?}", xa, xf);
    }

    /// A tolerance the controller cannot meet must be reported, not absorbed.
    #[test]
    fn unreachable_tolerance_is_reported() {
        let p = AdaptivePenalty::new(9.81, 1e10, 100.0).unwrap();
        let opts = AdaptiveOptions { rtol: 1e-14, atol: 1e-20, dt_init: 1e-6, dt_min: 1e-9, max_steps: 5_000 };
        assert!(p.rollout([0.5, 0.0], 1.0, opts).is_err());
    }

    /// Tightening the tolerance must move the answer toward a limit, not around at random. Two decades of
    /// tightening should shrink the change, which is the only evidence that a "converged" reference exists.
    #[test]
    fn reference_converges_under_tightening() {
        let p = AdaptivePenalty::new(9.81, 1e5, 30.0).unwrap();
        let coarse = p.rollout([0.5, 0.0], 0.4, AdaptiveOptions::with_tolerance(1e-6)).unwrap().0;
        let mid = p.rollout([0.5, 0.0], 0.4, AdaptiveOptions::with_tolerance(1e-9)).unwrap().0;
        let fine = p.rollout([0.5, 0.0], 0.4, AdaptiveOptions::with_tolerance(1e-12)).unwrap().0;
        let first = (coarse[0] - mid[0]).abs();
        let second = (mid[0] - fine[0]).abs();
        assert!(second < first, "not converging: {first:e} then {second:e}");
    }

    /// The decomposition must be self-consistent: the two parts are measured independently and the total is
    /// measured independently again, so the triangle inequality is a real check on all three.
    #[test]
    fn decomposition_obeys_the_triangle_inequality() {
        let d = decompose_gradient_error(
            9.81,
            1e5,
            30.0,
            1e-6,
            [0.5, 0.0],
            0.4,
            1e-7,
            AdaptiveOptions::with_tolerance(1e-11),
        )
        .expect("decomposition");
        assert!(
            d.total <= d.discretisation + d.model + 1e-9,
            "total {:e} exceeds parts {:e} + {:e}",
            d.total,
            d.discretisation,
            d.model
        );
        assert!(d.restitution > 0.0 && d.restitution <= 1.0, "restitution {}", d.restitution);
    }

    /// **The load-bearing result.** A tolerance-driven adaptive rollout of the penalty model reproduces the exact
    /// saltation Jacobian of the rigid system, which is closed form. The independence is partial and worth stating:
    /// `dv/dv0` is `1` by algebra whatever the restitution, so that entry is a genuinely integration-free check, while
    /// the others are calibrated by a restitution this same adaptive rollout measured.
    ///
    /// This is what makes the decomposition trustworthy rather than circular. Every other method here is scored
    /// against an adaptively-computed reference, so agreement among them proves nothing on its own.
    #[test]
    fn adaptive_reproduces_the_integration_free_jacobian() {
        let k: f64 = 1e6;
        let d = 2.0 * 0.1 * k.sqrt();
        let x0 = [0.5, 0.0];
        let t = 0.4;
        let opts = AdaptiveOptions::with_tolerance(1e-11);
        let p = AdaptivePenalty::new(9.81, k, d).unwrap();

        let e = p.effective_restitution((2.0f64 * 9.81 * x0[0]).sqrt(), opts).unwrap();
        let rigid = BouncingMass::new(9.81, e).unwrap().jacobian_saltation(x0, t).unwrap();
        let adaptive = p.jacobian_richardson(x0, t, 1e-6, opts).unwrap();

        let err = jacobian_error(adaptive, rigid);
        let scale = (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| rigid[i][j].abs()).fold(0.0, f64::max);
        assert!(err / scale < 5e-3, "adaptive {adaptive:?} vs rigid {rigid:?}, relative {:e}", err / scale);
    }

    /// A finite-difference Jacobian is only a measurement if it stops moving when the probe changes. The adaptive
    /// one does; a uniform-step one at this stiffness does not, by orders of magnitude, which is why uniform-step
    /// finite differences cannot be used to score anything here.
    #[test]
    fn adaptive_is_probe_stable_and_uniform_is_not() {
        let k: f64 = 1e6;
        let d = 2.0 * 0.1 * k.sqrt();
        let x0 = [0.5, 0.0];
        let t = 0.4;
        let opts = AdaptiveOptions::with_tolerance(1e-11);
        let p = AdaptivePenalty::new(9.81, k, d).unwrap();

        let a = p.jacobian_richardson(x0, t, 1e-5, opts).unwrap()[1][1];
        let b = p.jacobian_richardson(x0, t, 1e-6, opts).unwrap()[1][1];
        assert!((a - b).abs() < 1e-2, "adaptive moved with the probe: {a} vs {b}");

        let dt = t / 500_000.0;
        let u1 = p.jacobian_uniform(x0, t, dt, 1e-5).unwrap()[1][1];
        let u2 = p.jacobian_uniform(x0, t, dt, 1e-6).unwrap()[1][1];
        assert!((u1 - u2).abs() > 1.0, "uniform was expected to be probe-dependent here: {u1} vs {u2}");
    }

    /// The discretisation term dominates the model term at practical stiffness. This is the statement that
    /// reassigns the blame for a bad penalty gradient from the contact model to the integrator.
    #[test]
    fn discretisation_dominates_the_model_error() {
        let k: f64 = 1e6;
        let d = 2.0 * 0.1 * k.sqrt();
        let r = decompose_gradient_error(
            9.81,
            k,
            d,
            1.0 / k.sqrt(),
            [0.5, 0.0],
            0.4,
            1e-6,
            AdaptiveOptions::with_tolerance(1e-11),
        )
        .expect("decomposition");
        assert!(
            r.discretisation > 100.0 * r.model,
            "discretisation {:e} is not dominant over model {:e}",
            r.discretisation,
            r.model
        );
    }

    /// The reachable fraction is a fraction.
    #[test]
    fn reachable_fraction_is_bounded() {
        let d = decompose_gradient_error(
            9.81,
            1e6,
            30.0,
            1e-7,
            [0.5, 0.0],
            0.4,
            1e-7,
            AdaptiveOptions::with_tolerance(1e-11),
        )
        .expect("decomposition");
        let f = d.reachable_fraction();
        assert!((0.0..=1.0).contains(&f), "fraction {f}");
    }
}
