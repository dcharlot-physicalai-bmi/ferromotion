//! **The penalty contact in closed form** — no time-stepping, because it does not need any.
//!
//! [`crate::adaptive_contact`] integrates the penalty contact under a tolerance and measures how far the result sits
//! from the rigid saltation Jacobian. That module exists to *price* integrators. This one exists because the contact
//! does not require an integrator at all, and that fact is the foundation any provable bound on the smoothing gap
//! would rest on.
//!
//! # Why it is closed form
//!
//! Read the force the implementation actually uses ([`crate::adaptive_contact::AdaptivePenalty`]):
//!
//! ```text
//! F(h, v) = 0                              for h >= 0
//!         = max(0, -k*h - c*v)             for h <  0
//! ```
//!
//! That is **piecewise affine in `(h, v)`**, with three modes and two switching hyperplanes:
//!
//! | mode | condition | dynamics |
//! |---|---|---|
//! | flight | `h >= 0` | `v' = -g` |
//! | spring | `h < 0`, `k*h + c*v < 0` | `v' = -g - k*h - c*v` |
//! | clamped | `h < 0`, `k*h + c*v >= 0` | `v' = -g` |
//!
//! Every mode has an elementary exact flow: a quadratic in the two ballistic modes, and a damped harmonic oscillator
//! about the equilibrium `h_eq = -g/k` in the spring mode. So a contact is: enter the spring mode at `(0, -s)`, leave
//! it at the first root of one scalar function, coast ballistically, and exit at `h = 0`. Two scalar root-finds, no
//! discretisation error, and **stiffness improves the conditioning** rather than degrading it — a stiffer spring makes
//! the contact shorter and the phase advance faster, not the arithmetic worse.
//!
//! # What this is for
//!
//! Three things, in increasing ambition:
//!
//! 1. A reference for the integrators, exact to round-off rather than to a tolerance.
//! 2. The realised restitution and contact duration as **functions** of impact speed, rather than as drop-test
//!    measurements — which is what turns a sampled gap into a bounded one.
//! 3. The foundation for an interval enclosure over a *set* of impact speeds, which is the missing producer of
//!    `GapEvidence::Proved` in [`smoothing_tube`](../../ferromotion_control/smoothing_tube/index.html). That last step
//!    is not built here; this module supplies the exact scalar functions it would need.
//!
//! Everything below is verified against the tolerance-driven integrator rather than asserted, including the mode
//! sequence, which is a *derived* prediction and not an assumption.

/// Which mode the contact left the spring phase by. Reported rather than assumed, because the whole closed form
/// depends on the sequence being the one expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpringExit {
    /// The one-sided clamp shut while the body was still below the plane, which is the ordinary case for a lightly
    /// damped contact: `c*v` overtakes `-k*h` on the way back up.
    Clamped,
    /// The body reached the plane with the spring still pushing. Happens only when the damping is heavy enough that
    /// `k*h + c*v` stays negative all the way to `h = 0`.
    LeftDirectly,
}

/// An exactly-solved penalty contact: what the body did between entering and leaving the plane.
#[derive(Clone, Copy, Debug)]
pub struct ExactContact {
    /// Impact speed on entry, positive.
    pub impact_speed: f64,
    /// Exit speed, positive (upward).
    pub exit_speed: f64,
    /// Total time below the plane.
    pub duration: f64,
    /// Time spent in the spring mode before the clamp shut.
    pub spring_time: f64,
    /// Deepest penetration, negative.
    pub min_height: f64,
    /// How the spring phase ended.
    pub exit: SpringExit,
}

impl ExactContact {
    /// **The realised restitution**, as a function rather than a drop-test measurement.
    pub fn restitution(&self) -> f64 {
        if self.impact_speed > 0.0 { self.exit_speed / self.impact_speed } else { f64::NAN }
    }
}

/// The penalty contact, solved exactly.
#[derive(Clone, Copy, Debug)]
pub struct AffineContact {
    pub gravity: f64,
    pub stiffness: f64,
    pub damping: f64,
}

impl AffineContact {
    pub fn new(gravity: f64, stiffness: f64, damping: f64) -> Option<AffineContact> {
        (gravity > 0.0 && stiffness > 0.0 && damping >= 0.0).then_some(AffineContact { gravity, stiffness, damping })
    }

    /// Undamped natural frequency, `sqrt(k/m)` at unit mass.
    pub fn omega(&self) -> f64 {
        self.stiffness.sqrt()
    }

    /// Damping ratio. At or above `1` the spring mode is not oscillatory and the closed form below does not apply, so
    /// [`Self::solve`] refuses.
    pub fn zeta(&self) -> f64 {
        self.damping / (2.0 * self.omega())
    }

    /// Damped frequency, `omega * sqrt(1 - zeta^2)`.
    pub fn omega_d(&self) -> f64 {
        let z = self.zeta();
        self.omega() * (1.0 - z * z).max(0.0).sqrt()
    }

    /// Spring-mode equilibrium height, `-g/k`: where the spring would hold the body against gravity.
    pub fn equilibrium(&self) -> f64 {
        -self.gravity / self.stiffness
    }

    /// **Exact spring-mode flow.** State `(h, v)` a time `t` after entering the spring mode at `(h0, v0)`.
    ///
    /// Solved about the equilibrium: with `y = h - h_eq`, the mode is `y'' + c y' + k y = 0`, whose underdamped
    /// solution is elementary. The velocity returned is the analytic derivative of the height, not a difference.
    pub fn spring_flow(&self, h0: f64, v0: f64, t: f64) -> (f64, f64) {
        let (z, w, wd) = (self.zeta(), self.omega(), self.omega_d());
        let y0 = h0 - self.equilibrium();
        let decay = (-z * w * t).exp();
        let (cs, sn) = ((wd * t).cos(), (wd * t).sin());
        let y = decay * (y0 * cs + (v0 + z * w * y0) / wd * sn);
        let v = decay * (v0 * cs - (w * w * y0 + z * w * v0) / wd * sn);
        (y + self.equilibrium(), v)
    }

    /// The clamp indicator `k*h + c*v`. The spring mode is active while this is negative; the clamp shuts at its
    /// first zero. Evaluated from the exact flow, so its roots are the exact switching times.
    fn clamp_indicator(&self, h0: f64, v0: f64, t: f64) -> f64 {
        let (h, v) = self.spring_flow(h0, v0, t);
        self.stiffness * h + self.damping * v
    }

    /// Solve one contact from an impact at `(0, -impact_speed)`.
    ///
    /// Returns `None` when the closed form does not apply or its hypotheses fail: a non-positive impact speed, a
    /// non-oscillatory spring mode (`zeta >= 1`), or a switching time that cannot be bracketed. Refusing beats
    /// returning a number whose derivation did not hold.
    // Negated comparisons are deliberate: `!(x > 0.0)` rejects a NaN, where `x <= 0.0` would accept it and solve a
    // contact out of it. This module's whole value is that it refuses rather than guesses.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn solve(&self, impact_speed: f64) -> Option<ExactContact> {
        if !(impact_speed > 0.0) || self.zeta() >= 1.0 {
            return None;
        }
        let (h0, v0) = (0.0, -impact_speed);

        // Phase A: spring mode. The clamp indicator starts at c*(-s) < 0 and must reach zero. Bracket its first root
        // over one damped half-period, which is where the body has come back up.
        let half_period = core::f64::consts::PI / self.omega_d();
        let mut lo = 0.0;
        let mut hi = 0.0;
        let steps = 4096;
        let mut bracketed = false;
        for i in 1..=steps {
            let t = half_period * (i as f64) / (steps as f64);
            if self.clamp_indicator(h0, v0, t) >= 0.0 {
                lo = half_period * ((i - 1) as f64) / (steps as f64);
                hi = t;
                bracketed = true;
                break;
            }
        }
        if !bracketed {
            return None;
        }
        // Bisect to round-off. The indicator is analytic here, so this converges to the exact switching time.
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.clamp_indicator(h0, v0, mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let spring_time = 0.5 * (lo + hi);
        let (h1, v1) = self.spring_flow(h0, v0, spring_time);

        // Deepest penetration: where the velocity vanishes inside the spring phase, found the same way.
        let min_height = {
            let mut a = 0.0;
            let mut b = spring_time;
            for _ in 0..200 {
                let m = 0.5 * (a + b);
                if self.spring_flow(h0, v0, m).1 < 0.0 {
                    a = m;
                } else {
                    b = m;
                }
            }
            self.spring_flow(h0, v0, 0.5 * (a + b)).0.min(0.0)
        };

        // If the body already reached the plane, the spring phase ended by leaving rather than clamping.
        if h1 >= 0.0 {
            return Some(ExactContact {
                impact_speed,
                exit_speed: v1.max(0.0),
                duration: spring_time,
                spring_time,
                min_height,
                exit: SpringExit::LeftDirectly,
            });
        }

        // Phase B: clamped ballistic, h'' = -g, from (h1, v1) with h1 < 0 and v1 > 0. Exit at h = 0.
        // h(t) = h1 + v1 t - g t^2 / 2 = 0  ->  t = (v1 - sqrt(v1^2 + 2 g h1)) / g  is the ROOT BEFORE the apex;
        // since h1 < 0 the physical root is the one with the plus sign on the discriminant path upward.
        if v1 <= 0.0 {
            return None; // clamped while still descending: the body cannot leave, so the closed form does not apply
        }
        let disc = v1 * v1 + 2.0 * self.gravity * h1;
        if disc < 0.0 {
            return None; // never reaches the plane again: it clamped below its own apex
        }
        let t_b = (v1 - disc.sqrt()) / self.gravity;
        // Same reason: a NaN ballistic time must be refused, not accepted as non-negative.
        if !(t_b >= 0.0) {
            return None;
        }
        let exit_speed = v1 - self.gravity * t_b;
        Some(ExactContact {
            impact_speed,
            exit_speed: exit_speed.max(0.0),
            duration: spring_time + t_b,
            spring_time,
            min_height,
            exit: SpringExit::Clamped,
        })
    }

    /// Realised restitution as a function of impact speed, exactly. This is the scalar an interval enclosure over a
    /// set of impact speeds would bound, and the reason the closed form is worth having.
    pub fn restitution_at(&self, impact_speed: f64) -> Option<f64> {
        self.solve(impact_speed).map(|c| c.restitution())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_contact::{AdaptiveOptions, AdaptivePenalty};

    fn fixture(k: f64) -> AffineContact {
        // The smoothing-tube fixture's damping law.
        AffineContact::new(9.81, k, 2.0 * 0.1606 * k.sqrt()).unwrap()
    }

    /// **The foundation.** The closed form must reproduce the tolerance-driven integrator's restitution. If it does
    /// not, nothing built on it means anything — and no bound on the smoothing gap can rest on it.
    #[test]
    fn the_closed_form_matches_the_tolerance_driven_integrator() {
        let v = (2.0f64 * 9.81 * 1.0).sqrt();
        for &k in &[1e4f64, 1e5, 1e6, 1e7] {
            let exact = fixture(k);
            let ode = AdaptivePenalty::new(9.81, k, exact.damping).unwrap();
            let closed = exact.restitution_at(v).unwrap();
            let integrated = ode.effective_restitution(v, AdaptiveOptions::with_tolerance(1e-12)).unwrap();
            let rel = (closed - integrated).abs() / integrated;
            eprintln!(
                "k {k:.0e}: closed form {closed:.9}, integrator {integrated:.9}, relative {rel:.2e}  (zeta {:.4})",
                exact.zeta()
            );
            assert!(rel < 1e-5, "closed form and integrator disagree at k = {k:.0e}: {closed} vs {integrated}");
        }
    }

    /// The mode sequence is a PREDICTION of the closed form, not an assumption of it. For this lightly damped fixture
    /// the clamp must shut before the body reaches the plane, and the solver must report which happened.
    #[test]
    fn the_clamp_shuts_before_the_body_leaves() {
        for &k in &[1e4f64, 1e6] {
            let c = fixture(k).solve((2.0f64 * 9.81 * 1.0).sqrt()).unwrap();
            eprintln!(
                "k {k:.0e}: exit {:?}, spring {:.3e} s of {:.3e} s total, deepest {:.3e} m",
                c.exit, c.spring_time, c.duration, c.min_height
            );
            assert_eq!(c.exit, SpringExit::Clamped, "a lightly damped contact should clamp, not leave directly");
            assert!(c.spring_time < c.duration, "there must be a ballistic tail");
            assert!(c.min_height < 0.0, "the body must penetrate");
        }
    }

    /// Contact duration must scale as `1/sqrt(k)`: this is the `pi/omega` law the gap bound's tightness depends on,
    /// and it is the one place stiffness helps rather than hurts.
    #[test]
    fn contact_duration_scales_as_one_over_root_k() {
        let v = (2.0f64 * 9.81 * 1.0).sqrt();
        let a = fixture(1e4).solve(v).unwrap().duration;
        let b = fixture(1e6).solve(v).unwrap().duration;
        // 100x the stiffness is 10x the frequency, so 10x shorter.
        let ratio = a / b;
        eprintln!("duration at k=1e4 {a:.4e} s, at k=1e6 {b:.4e} s, ratio {ratio:.3} (expect ~10)");
        assert!((ratio - 10.0).abs() < 0.5, "duration should scale as 1/sqrt(k), ratio {ratio}");
    }

    /// Restitution must be a well-behaved function of impact speed, since that is the scalar an enclosure would bound.
    /// For a linear spring-damper it is very nearly speed-independent, which is what makes the bound tight.
    #[test]
    fn restitution_is_a_smooth_function_of_impact_speed() {
        let exact = fixture(1e6);
        let nominal = (2.0f64 * 9.81 * 1.0).sqrt();
        let mut vals = Vec::new();
        for f in [0.9, 0.95, 1.0, 1.05, 1.1] {
            let e = exact.restitution_at(nominal * f).unwrap();
            vals.push(e);
            eprintln!("impact {:.4} m/s: restitution {e:.9}", nominal * f);
        }
        let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
        eprintln!("over +/-10% of impact speed the restitution spans {:.3e}", hi - lo);
        assert!(hi - lo < 1e-3, "restitution should be near-constant in speed, spans {:.3e}", hi - lo);
        assert!(vals.windows(2).all(|w| w[1] >= w[0] - 1e-12) || vals.windows(2).all(|w| w[1] <= w[0] + 1e-12),
            "and monotone in speed, which is what makes an interval enclosure tight");
    }

    /// Hypotheses that do not hold must produce a refusal, not a number.
    #[test]
    fn broken_hypotheses_are_refused() {
        let exact = fixture(1e6);
        assert!(exact.solve(0.0).is_none(), "a zero impact is not a contact");
        assert!(exact.solve(-1.0).is_none());
        // Critically damped or beyond: the underdamped closed form does not apply.
        let heavy = AffineContact::new(9.81, 1e6, 2.0 * 1.0 * 1e3).unwrap();
        assert!(heavy.zeta() >= 1.0, "zeta {}", heavy.zeta());
        assert!(heavy.solve(4.0).is_none(), "an overdamped contact must be refused, not guessed");
        assert!(AffineContact::new(-1.0, 1e6, 1.0).is_none());
        assert!(AffineContact::new(9.81, 0.0, 1.0).is_none());
    }
}
