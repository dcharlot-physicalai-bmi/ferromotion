//! **Getting a gradient before you touch anything** — contacts from distance.
//!
//! # A different failure from the one [`crate::adaptive_contact`] fixes
//!
//! That module is about a gradient that is *wrong*: a fixed-step penalty rollout reports a contact Jacobian off by
//! more than the Jacobian's own magnitude, and the remedy is an exact event-driven derivative or a tolerance-driven
//! stepper. Both assume the contact happens.
//!
//! This module is about a gradient that is *exactly zero*. A finger that is not touching an object exerts no force on
//! it, so the object does not move, so
//!
//! ```text
//! d(object trajectory) / d(finger action) = 0     identically, not approximately
//! ```
//!
//! and every gradient-based method is dead on arrival. No amount of integration accuracy helps: the derivative is
//! genuinely zero, and it is zero because the physics says so. An optimiser started from a non-touching configuration
//! has nothing telling it which way to reach.
//!
//! # The straight-through trick
//!
//! The published remedy is to apply an artificial contact force at *positive* signed distance, and to apply it **only
//! in the backward pass**, via a stop-gradient construction:
//!
//! ```text
//! xdot = sg(F) + F~ - sg(F~)
//! ```
//!
//! The forward value is `F`, the true force, because `F~ - sg(F~)` is zero. The derivative is `dF~/dx`, because
//! `sg(F)` contributes none. So the simulated trajectory is exactly the real one, unmodified and still physical, while
//! the gradient is taken through a field that can feel the object from a distance.
//!
//! This workspace writes Jacobians explicitly rather than taping, which makes the trick unusually clear: the state
//! update uses [`PushTask::force`] and the sensitivity accumulation uses `force_partials`. They deliberately disagree,
//! and the disagreement is the whole method.
//!
//! # How well the gradient can be validated here, and why not better
//!
//! The obvious oracle is a central difference of the objective, and on this task it is **not probe-stable**. Measured
//! at `u = 2x` the contact threshold (`examples/contacts_from_distance.rs`):
//!
//! ```text
//! h      1e-1     1e-2     1e-3     1e-4     1e-5     1e-6     1e-7     1e-8
//! fd   1.60e-2  1.66e-2  1.66e-2  1.66e-2  4.90e-2  3.10e-1  2.92e0   2.90e1
//! ```
//!
//! Below `1e-5` it grows like `1/h`, which is the signature of a **discontinuity** rather than a derivative: the
//! objective depends on `u` partly through the integer number of timesteps spent in contact, and a small probe divides
//! a tiny jump by a tiny `h`. Above `1e-3` the probe is wide enough to average the jumps out and instead picks up the
//! objective's curvature.
//!
//! So the usable oracle is the plateau, `1.663e-2`, and the analytic gradient converges under timestep refinement to
//! `2.027e-2` (stable to 3% from `dt = 1e-4` down to `5e-6`). They agree in sign and order of magnitude and differ by
//! about 22%, and **neither validates the other more tightly than that**. The tests below assert exactly that much and
//! no more. The headline result does not depend on it: whether descent can start at all from an out-of-reach
//! configuration is a question about zero versus non-zero, not about 22%.
//!
//! # What is honest to claim here
//!
//! The reach profile below is ours, not a reproduction of any published spline. What is reproduced is the *mechanism*
//! (an artificial pre-contact force applied to the backward pass only) and what is measured is whether it buys descent
//! that the true gradient cannot deliver. A pre-contact gradient is a **search heuristic, not physics**: it points at
//! the object, and it is not the derivative of anything the robot will actually experience. The forward trajectory
//! stays exact so that every reported cost is a real cost.

/// Shape of the artificial pre-contact force used **only** for the gradient.
///
/// Zero at and beyond `reach`, rising to `gain` times the contact stiffness scale as the gap closes. Continuous at the
/// contact boundary when `power >= 1`, which matters because a discontinuity there would put a spurious spike into the
/// very quantity being repaired.
#[derive(Clone, Copy, Debug)]
pub struct CfdProfile {
    /// Distance at which the object becomes "visible" to the gradient, metres.
    pub reach: f64,
    /// Strength relative to the real contact stiffness.
    pub gain: f64,
    /// Falloff exponent. `1` is linear, higher concentrates the signal near contact.
    pub power: f64,
}

impl CfdProfile {
    pub fn new(reach: f64, gain: f64, power: f64) -> Option<CfdProfile> {
        (reach > 0.0 && gain > 0.0 && power >= 1.0).then_some(CfdProfile { reach, gain, power })
    }

    /// A profile that reaches `reach` metres with a modest gain, which is the setting that behaved best in the sweep
    /// recorded in `examples/contacts_from_distance.rs`.
    pub fn reaching(reach: f64) -> CfdProfile {
        CfdProfile { reach, gain: 0.05, power: 2.0 }
    }
}

/// A finger pushing an object along a line: the smallest system in which a pre-contact gradient is worth anything.
///
/// The finger is position-controlled, `x_f(t) = x_f0 + u * t`, so `u` is the single parameter an optimiser holds. The
/// object obeys `m * dv/dt = F_contact - drag * v`. The objective is where the object ends up.
#[derive(Clone, Copy, Debug)]
pub struct PushTask {
    /// Sum of the two radii: contact begins when the centre gap closes to this.
    pub contact_distance: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub object_mass: f64,
    /// Linear ground drag on the object, so a push produces a bounded displacement rather than a runaway.
    pub drag: f64,
    pub dt: f64,
    pub steps: usize,
    /// Where the finger starts.
    pub finger_start: f64,
    /// Where the object starts.
    pub object_start: f64,
    /// Where the object is wanted.
    pub target: f64,
}

impl PushTask {
    /// A task whose object starts out of reach, which is the configuration that produces a zero gradient.
    pub fn out_of_reach() -> PushTask {
        PushTask {
            contact_distance: 0.02,
            stiffness: 1.0e4,
            damping: 12.0,
            object_mass: 0.2,
            drag: 2.0,
            dt: 1.0e-4,
            steps: 4000,
            finger_start: 0.0,
            object_start: 0.10,
            target: 0.16,
        }
    }

    fn horizon(&self) -> f64 {
        self.dt * self.steps as f64
    }

    /// Signed gap between the surfaces. Positive means not touching.
    fn gap(&self, x_f: f64, x_o: f64) -> f64 {
        x_o - x_f - self.contact_distance
    }

    /// **The true contact force.** Zero unless the surfaces overlap, one-sided so the finger cannot pull.
    ///
    /// This is what the forward pass integrates, always, whether or not a pre-contact gradient is in use. The
    /// trajectory must stay physical or every cost reported against it is fiction.
    fn force(&self, gap: f64, v_rel: f64) -> f64 {
        if gap >= 0.0 {
            0.0
        } else {
            (-self.stiffness * gap - self.damping * v_rel).max(0.0)
        }
    }

    /// **The partial derivatives the gradient uses**, which deliberately need not be the derivatives of
    /// [`Self::force`]: `(dF/dgap, dF/dv_rel)`.
    ///
    /// Inside contact these are the true partials, *and they are zero when the one-sided `max` has clamped the force to
    /// zero*. Getting that wrong was a real bug here: reporting `-stiffness` through a clamped contact put a force
    /// derivative where there was no force, and the analytic gradient came out 19x short of the finite-difference
    /// oracle. Outside contact, with a profile supplied, the gap partial is the artificial pre-contact slope; without
    /// one it is zero, which is the honest derivative and also the reason descent stalls.
    fn force_partials(&self, gap: f64, v_rel: f64, profile: Option<CfdProfile>) -> (f64, f64) {
        if gap < 0.0 {
            // The clamp is part of the function, so it is part of the derivative.
            let raw = -self.stiffness * gap - self.damping * v_rel;
            return if raw > 0.0 { (-self.stiffness, -self.damping) } else { (0.0, 0.0) };
        }
        match profile {
            Some(p) if gap < p.reach => {
                // F~(g) = gain * k * (1 - g/reach)^power, so dF~/dg = -gain * k * power/reach * (1 - g/reach)^(power-1).
                let s = 1.0 - gap / p.reach;
                (-p.gain * self.stiffness * p.power / p.reach * s.powf(p.power - 1.0), 0.0)
            }
            _ => (0.0, 0.0),
        }
    }

    /// Roll the task forward at parameter `u`. Returns the object's final position and whether contact ever occurred.
    ///
    /// The forward pass never sees a profile. That is not an oversight; it is the point.
    pub fn rollout(&self, u: f64) -> (f64, bool) {
        let mut x_o = self.object_start;
        let mut v_o = 0.0;
        let mut touched = false;
        for k in 0..self.steps {
            let t = k as f64 * self.dt;
            let x_f = self.finger_start + u * t;
            let v_f = u;
            let gap = self.gap(x_f, x_o);
            if gap < 0.0 {
                touched = true;
            }
            let f = self.force(gap, v_o - v_f);
            let a = (f - self.drag * v_o) / self.object_mass;
            v_o += a * self.dt;
            x_o += v_o * self.dt;
        }
        (x_o, touched)
    }

    /// Objective: squared miss of the object's final position.
    pub fn objective(&self, u: f64) -> f64 {
        let (x, _) = self.rollout(u);
        (x - self.target) * (x - self.target)
    }

    /// **The gradient `dJ/du`**, accumulated along the realised trajectory.
    ///
    /// With `profile = None` this is the true derivative, and it is exactly zero whenever the finger never reaches the
    /// object. With a profile it is the straight-through gradient: the same forward states, a different slope.
    pub fn gradient(&self, u: f64, profile: Option<CfdProfile>) -> f64 {
        let mut x_o = self.object_start;
        let mut v_o = 0.0;
        // Sensitivities of the object's state to the parameter.
        let mut ds = 0.0f64; // d x_o / d u
        let mut dv = 0.0f64; // d v_o / d u

        for k in 0..self.steps {
            let t = k as f64 * self.dt;
            let x_f = self.finger_start + u * t;
            let v_f = u;
            let gap = self.gap(x_f, x_o);

            // d(gap)/du = d x_o/du - d x_f/du, and x_f = finger_start + u t so d x_f/du = t.
            let dgap = ds - t;
            let v_rel = v_o - v_f;
            // Damping acts on the relative velocity, and d(v_o - v_f)/du = dv - 1 because dv_f/du = 1.
            let (df_dgap, df_dvrel) = self.force_partials(gap, v_rel, profile);
            let df = df_dgap * dgap + df_dvrel * (dv - 1.0);

            let f = self.force(gap, v_rel);
            let a = (f - self.drag * v_o) / self.object_mass;
            let da = (df - self.drag * dv) / self.object_mass;

            v_o += a * self.dt;
            dv += da * self.dt;
            x_o += v_o * self.dt;
            ds += dv * self.dt;
        }
        // J = (x_o - target)^2
        2.0 * (x_o - self.target) * ds
    }

    /// Central-difference gradient of the objective: the oracle, since neither analytic route may be assumed correct.
    pub fn gradient_finite_difference(&self, u: f64, h: f64) -> f64 {
        (self.objective(u + h) - self.objective(u - h)) / (2.0 * h)
    }

    /// The parameter value at which the finger just reaches the object within the horizon. Below it, the true gradient
    /// is identically zero and there is nothing for descent to follow.
    pub fn contact_threshold(&self) -> f64 {
        (self.object_start - self.finger_start - self.contact_distance) / self.horizon()
    }
}

/// One descent run's outcome.
#[derive(Clone, Copy, Debug)]
pub struct DescentResult {
    pub u: f64,
    pub objective: f64,
    pub iterations: usize,
    /// Whether the finger ever touched the object at the final parameter.
    pub touched: bool,
    /// Whether any step actually moved the parameter. A run that never moves is not slow convergence; it is a dead
    /// gradient, and the two must not be reported the same way.
    pub moved: bool,
}

/// Gradient descent on `u` with a backtracking line search that only accepts steps reducing the **true** objective.
///
/// The search is deliberately honest: a pre-contact gradient is allowed to propose a direction and is not allowed to
/// grade the result. If the artificial force flattered a step, the line search rejects it on the real cost.
pub fn descend(task: &PushTask, u0: f64, profile: Option<CfdProfile>, iterations: usize, step0: f64) -> DescentResult {
    let mut u = u0;
    let mut moved = false;
    for _ in 0..iterations {
        let g = task.gradient(u, profile);
        if !g.is_finite() || g == 0.0 {
            break;
        }
        let j0 = task.objective(u);
        let mut step = step0;
        let mut accepted = false;
        for _ in 0..40 {
            let candidate = u - step * g;
            if task.objective(candidate) < j0 {
                u = candidate;
                accepted = true;
                moved = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    let (_, touched) = task.rollout(u);
    DescentResult { u, objective: task.objective(u), iterations, touched, moved }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The failure this module exists for.** With the object out of reach the true gradient is not small, it is
    /// exactly zero, and the finite-difference oracle agrees. Nothing about integration accuracy can help.
    #[test]
    fn the_true_gradient_is_exactly_zero_out_of_reach() {
        let task = PushTask::out_of_reach();
        let u = 0.5 * task.contact_threshold(); // half as fast as it needs to be
        let (_, touched) = task.rollout(u);
        assert!(!touched, "the fixture must genuinely not touch at u = {u}");

        let g = task.gradient(u, None);
        let fd = task.gradient_finite_difference(u, 1e-6);
        eprintln!("out of reach at u = {u:.5}: analytic gradient {g:e}, finite difference {fd:e}");
        assert_eq!(g, 0.0, "the true gradient must be exactly zero, not merely small");
        assert_eq!(fd, 0.0, "and the oracle must agree, or the fixture is not what it claims");
    }

    /// The straight-through gradient must be non-zero in the same configuration, and must point toward closing the gap.
    #[test]
    fn the_pre_contact_gradient_points_at_the_object() {
        let task = PushTask::out_of_reach();
        let u = 0.5 * task.contact_threshold();
        let profile = CfdProfile::reaching(0.06);
        let g = task.gradient(u, Some(profile));
        eprintln!("same state, straight-through gradient {g:e} (descent moves u by -step * g)");
        assert!(g.is_finite() && g != 0.0, "the pre-contact gradient must exist: {g}");
        // The objective wants the object further away than it is, so descent must INCREASE u (reach faster), which
        // means the gradient must be negative.
        assert!(g < 0.0, "the gradient should push the finger toward the object, got {g}");
    }

    /// **The forward pass must be untouched.** A profile may change the gradient and may not change one bit of the
    /// trajectory, or every cost reported against it is fiction.
    #[test]
    fn the_profile_never_changes_the_trajectory() {
        let task = PushTask::out_of_reach();
        for u in [0.2, 0.5, 1.0, 2.0] {
            let (a, ta) = task.rollout(u);
            // rollout takes no profile at all, which is the structural guarantee; this checks the objective too.
            let ja = task.objective(u);
            let (b, tb) = task.rollout(u);
            assert_eq!(a.to_bits(), b.to_bits(), "the rollout must be deterministic");
            assert_eq!(ta, tb);
            assert_eq!(ja.to_bits(), task.objective(u).to_bits());
        }
    }

    /// Inside contact the straight-through gradient must agree with the true one, because there the artificial force is
    /// not in play and the derivative is the real derivative.
    ///
    /// Compared against the finite difference on its **stable plateau**, not at a small probe. An earlier version of
    /// this test used `h = 1e-6` and demanded 5% agreement; it failed by 16x, and the fault was the oracle. On this
    /// objective a small probe reports `1/h`, not a derivative.
    #[test]
    fn in_contact_the_two_gradients_agree() {
        let task = PushTask::out_of_reach();
        let u = 2.0 * task.contact_threshold(); // comfortably touching
        let (_, touched) = task.rollout(u);
        assert!(touched);

        let truth = task.gradient(u, None);
        let cfd = task.gradient(u, Some(CfdProfile::reaching(0.06)));
        let fd = task.gradient_finite_difference(u, 1e-3); // on the plateau
        eprintln!("in contact at u = {u:.5}: true {truth:.6e}, cfd {cfd:.6e}, plateau fd {fd:.6e}");
        assert!(truth != 0.0, "in contact the true gradient must be alive");
        assert!(truth * fd > 0.0, "the analytic gradient must at least share the oracle's sign");
        let ratio = truth / fd;
        assert!((0.7..1.5).contains(&ratio), "same order of magnitude as the plateau oracle, got {ratio:.3}x");
        // The artificial force still acts over the approach, so the two analytic routes are close but need not be
        // identical; what matters is that they agree on direction.
        assert!(cfd * truth > 0.0, "cfd and truth must agree on direction: {cfd:e} vs {truth:e}");
    }

    /// **The analytic gradient's own convergence check**, which is the evidence the finite difference cannot supply.
    /// Refining the timestep by 20x must not move it, and that stability is what makes it quotable.
    #[test]
    fn the_analytic_gradient_is_stable_under_timestep_refinement() {
        let base = PushTask::out_of_reach();
        let mut values = Vec::new();
        for (dt, steps) in [(1e-4, 4_000usize), (2e-5, 20_000), (5e-6, 80_000)] {
            let t = PushTask { dt, steps, ..base };
            let g = t.gradient(2.0 * t.contact_threshold(), None);
            eprintln!("dt {dt:.0e} ({steps} steps): analytic {g:.6e}");
            values.push(g);
        }
        let (lo, hi) = values.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!((hi - lo) / hi < 0.05, "should be stable to a few percent: {lo:.4e} to {hi:.4e}");
    }

    /// **The trap, pinned.** A central difference of this objective is not probe-stable, and the failure grows as the
    /// probe shrinks, which is the opposite of the intuition. Recording it as a test stops the next person reaching for
    /// `h = 1e-8` and believing the answer.
    #[test]
    fn the_finite_difference_oracle_is_not_probe_stable() {
        let task = PushTask::out_of_reach();
        let u = 2.0 * task.contact_threshold();
        let plateau = task.gradient_finite_difference(u, 1e-3);
        let tiny = task.gradient_finite_difference(u, 1e-8);
        eprintln!("fd at h = 1e-3: {plateau:.4e};  at h = 1e-8: {tiny:.4e}  ({:.0}x)", tiny / plateau);
        assert!(tiny / plateau > 100.0, "a small probe should blow up here, got {:.1}x", tiny / plateau);
        // ... and the plateau itself must be a plateau, or there is no usable oracle at all.
        let a = task.gradient_finite_difference(u, 1e-2);
        let b = task.gradient_finite_difference(u, 1e-4);
        assert!((a - b).abs() / b.abs() < 0.05, "the plateau must be flat: {a:.4e} vs {b:.4e}");
    }

    /// **The result that decides whether any of this was worth building.** From an out-of-reach start, descent on the
    /// true gradient cannot move at all, and descent on the straight-through gradient reaches the object and reduces
    /// the real objective.
    #[test]
    fn descent_succeeds_from_out_of_reach_only_with_a_pre_contact_gradient() {
        let task = PushTask::out_of_reach();
        let u0 = 0.5 * task.contact_threshold();
        let j0 = task.objective(u0);

        let without = descend(&task, u0, None, 60, 1e3);
        let with = descend(&task, u0, Some(CfdProfile::reaching(0.06)), 60, 1e3);

        eprintln!("start: u = {u0:.5}, objective {j0:.6e}, touching false");
        eprintln!(
            "  true gradient:      u = {:.5}, objective {:.6e}, touched {}, moved {}",
            without.u, without.objective, without.touched, without.moved
        );
        eprintln!(
            "  straight-through:   u = {:.5}, objective {:.6e}, touched {}, moved {}",
            with.u, with.objective, with.touched, with.moved
        );

        assert!(!without.moved, "the true gradient cannot move the parameter at all");
        assert_eq!(without.objective, j0, "so the objective is untouched");
        assert!(with.moved, "the pre-contact gradient must get descent started");
        assert!(with.touched, "and must actually reach the object");
        assert!(with.objective < 0.5 * j0, "and reduce the real objective: {:.3e} from {:.3e}", with.objective, j0);
    }

    /// **The reach is a prior, not a free parameter.** A profile must exceed the finger's closest approach to see the
    /// object at all, so this method buys a first step in exchange for knowing roughly how far away the thing is. The
    /// example's own summary claimed "every other row solves the task" until this was measured: the 0.02 m reach fails.
    #[test]
    fn the_reach_must_exceed_the_closest_approach() {
        let task = PushTask::out_of_reach();
        let u0 = 0.5 * task.contact_threshold();
        let closest = task.object_start
            - task.contact_distance
            - (task.finger_start + u0 * task.dt * task.steps as f64);
        eprintln!("closest approach at u0 leaves a gap of {closest:.4} m");
        assert!(closest > 0.0);

        let blind = descend(&task, u0, Some(CfdProfile::reaching(closest * 0.5)), 60, 1e3);
        let seeing = descend(&task, u0, Some(CfdProfile::reaching(closest * 1.5)), 60, 1e3);
        eprintln!("  reach {:.4} m: moved {}", closest * 0.5, blind.moved);
        eprintln!("  reach {:.4} m: moved {}, objective {:.3e}", closest * 1.5, seeing.moved, seeing.objective);
        assert!(!blind.moved, "a reach under the gap must behave exactly like no profile");
        assert!(seeing.moved && seeing.touched, "a reach over the gap must get descent started");
    }

    /// A profile that reaches nowhere is the no-profile case, and a reach shorter than the initial gap must not help.
    /// Without this the win above could be an artefact of the profile being generous enough to touch everything.
    #[test]
    fn a_profile_that_cannot_see_the_object_does_not_help() {
        let task = PushTask::out_of_reach();
        let u0 = 0.5 * task.contact_threshold();
        // The finger's closest approach at u0 leaves a gap much larger than this reach.
        let myopic = CfdProfile::new(1e-4, 0.05, 2.0).expect("profile");
        let r = descend(&task, u0, Some(myopic), 60, 1e3);
        eprintln!("myopic reach 1e-4: moved {}, touched {}, objective {:.6e}", r.moved, r.touched, r.objective);
        assert!(!r.moved, "a profile that cannot reach the object must behave like no profile");
    }

    /// Malformed profiles are refused rather than silently clamped.
    #[test]
    fn bad_profiles_are_refused() {
        assert!(CfdProfile::new(0.0, 0.05, 2.0).is_none());
        assert!(CfdProfile::new(0.05, 0.0, 2.0).is_none());
        assert!(CfdProfile::new(0.05, 0.05, 0.5).is_none(), "power below 1 makes the slope singular at contact");
    }
}
