//! **Stribeck and LuGre friction** — what a joint actually resists with, below and around breakaway.
//!
//! [`crate::actuator::SeaJoint`] models friction as smoothed Coulomb plus viscous, which is the standard
//! simplification and wrong in the two regimes that matter most for a robot:
//!
//! * **At and below breakaway.** Real friction is *higher* at rest than in motion, then dips through a
//!   minimum as speed rises before viscous drag takes over. That non-monotone dip is the **Stribeck** effect,
//!   and it is why a joint sticks, then lurches — a Coulomb model cannot produce stick-slip at all, because it
//!   has no excess to release.
//! * **Before motion begins.** A joint deflects *microns* elastically before it breaks away. That
//!   **presliding** displacement is why position control has a deadband no gain can remove, and why a
//!   velocity-only friction law has no state to explain it with.
//!
//! # Two models, and why both
//!
//! [`Stribeck`] is a static map `v ↦ τ_f`: cheap, no state, correct in steady sliding, and the right thing for
//! a feedforward compensator. [`LuGre`] carries an internal bristle deflection, so it also reproduces
//! presliding, hysteresis, and the Dahl rate effect — at the cost of a state variable and a stiff time
//! constant.
//!
//! **They are not independent claims, and the test suite uses that.** In steady sliding LuGre must reduce
//! *exactly* to the Stribeck curve it was parameterised with. That cross-check is what makes either
//! trustworthy: the dynamic model is verified against the static one where they overlap, rather than each
//! being checked only against itself.
//!
//! ```text
//! Stribeck:  τ_f(v) = [τ_C + (τ_S − τ_C)·exp(−|v/v_s|^δ)]·sgn(v) + σ₂·v
//! LuGre:     ż = v − σ₀·|v|·z / g(v),    τ_f = σ₀·z + σ₁·ż + σ₂·v
//!            with g(v) = τ_C + (τ_S − τ_C)·exp(−|v/v_s|^δ)
//! ```

/// Static Stribeck friction curve: breakaway, the Stribeck dip, and viscous rise.
#[derive(Clone, Copy, Debug)]
pub struct Stribeck {
    /// Static (breakaway) friction magnitude — the value approached as `v → 0⁺`.
    pub tau_s: f64,
    /// Coulomb friction magnitude — the plateau at moderate sliding speed.
    pub tau_c: f64,
    /// Stribeck velocity: the scale over which static decays to Coulomb.
    pub v_s: f64,
    /// Shape exponent `δ`. 1 is exponential decay, 2 is the common Gaussian-like form.
    pub delta: f64,
    /// Viscous coefficient `σ₂`.
    pub sigma_2: f64,
}

impl Stribeck {
    /// A curve with the usual `δ = 2`.
    pub fn new(tau_s: f64, tau_c: f64, v_s: f64, sigma_2: f64) -> Self {
        Self { tau_s, tau_c, v_s, delta: 2.0, sigma_2 }
    }

    /// The speed-dependent magnitude `g(v) = τ_C + (τ_S − τ_C)·exp(−|v/v_s|^δ)`, without sign or viscous term.
    ///
    /// This is the function LuGre uses internally, which is what lets the two models be compared.
    pub fn g(&self, v: f64) -> f64 {
        self.tau_c + (self.tau_s - self.tau_c) * (-(v / self.v_s).abs().powf(self.delta)).exp()
    }

    /// Friction torque at velocity `v`.
    ///
    /// At exactly `v = 0` this returns `0`: a static curve cannot say what friction holds a stationary joint at,
    /// only what it *could* resist up to. Anything up to `τ_S` is admissible at rest, and choosing one requires
    /// the applied load — which is what [`LuGre`], with its state, can actually do.
    pub fn torque(&self, v: f64) -> f64 {
        if v == 0.0 {
            return 0.0;
        }
        self.g(v) * v.signum() + self.sigma_2 * v
    }

    /// The velocity at which total friction is smallest — the bottom of the Stribeck dip.
    ///
    /// Located by scanning, because the stationary point of
    /// `g(v) + σ₂v` has no closed form for general `δ`. Returns `None` when no interior minimum exists, which
    /// happens when `τ_S ≈ τ_C` (no dip to find) or viscous drag dominates immediately.
    pub fn dip_velocity(&self, v_max: f64, samples: usize) -> Option<f64> {
        if samples < 3 || v_max <= 0.0 {
            return None;
        }
        let mut best = (f64::INFINITY, 0.0);
        for i in 1..=samples {
            let v = v_max * i as f64 / samples as f64;
            let t = self.torque(v);
            if t < best.0 {
                best = (t, v);
            }
        }
        // An interior minimum, not an endpoint — an endpoint means the curve is monotone over this range.
        let v = best.1;
        if v <= v_max / samples as f64 * 1.5 || v >= v_max * 0.999 {
            return None;
        }
        Some(v)
    }
}

/// LuGre friction: a bristle-deflection state, so presliding and hysteresis are represented.
#[derive(Clone, Copy, Debug)]
pub struct LuGre {
    /// The Stribeck curve this model reduces to in steady sliding.
    pub curve: Stribeck,
    /// Bristle stiffness `σ₀` — the presliding spring rate.
    pub sigma_0: f64,
    /// Bristle damping `σ₁`.
    pub sigma_1: f64,
    /// Bristle deflection state `z`.
    pub z: f64,
}

impl LuGre {
    pub fn new(curve: Stribeck, sigma_0: f64, sigma_1: f64) -> Self {
        Self { curve, sigma_0, sigma_1, z: 0.0 }
    }

    /// `ż = v − σ₀·|v|·z / g(v)`.
    pub fn z_dot(&self, v: f64) -> f64 {
        v - self.sigma_0 * v.abs() * self.z / self.curve.g(v)
    }

    /// Friction torque `τ_f = σ₀·z + σ₁·ż + σ₂·v` at the present state and velocity.
    pub fn torque(&self, v: f64) -> f64 {
        self.sigma_0 * self.z + self.sigma_1 * self.z_dot(v) + self.curve.sigma_2 * v
    }

    /// Advance the bristle state by `dt` at velocity `v`, returning the friction torque over the step.
    ///
    /// The bristle time constant is `g(v)/(σ₀·|v|)`, which is *short* for a stiff `σ₀` — the reason LuGre is
    /// usually the stiffest part of a joint model and why [`LuGre::max_stable_dt`] is worth consulting.
    pub fn step(&mut self, dt: f64, v: f64) -> f64 {
        let tau = self.torque(v);
        self.z += dt * self.z_dot(v);
        tau
    }

    /// Largest `dt` for which the explicit bristle update is stable at speed `v`: `g(v)/(σ₀·|v|)`.
    ///
    /// Returns `f64::INFINITY` at `v = 0`, where the state does not decay at all — presliding is
    /// non-dissipative in this model, which is exactly why it behaves as a spring there.
    pub fn max_stable_dt(&self, v: f64) -> f64 {
        if v == 0.0 {
            return f64::INFINITY;
        }
        self.curve.g(v) / (self.sigma_0 * v.abs())
    }

    /// The steady-state bristle deflection at constant `v`: `z_ss = g(v)·sgn(v)/σ₀`.
    ///
    /// Setting `ż = 0` gives this directly, and substituting it into the torque recovers the Stribeck curve —
    /// the algebraic statement of the reduction the tests check numerically.
    pub fn steady_z(&self, v: f64) -> f64 {
        self.curve.g(v) * v.signum() / self.sigma_0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint() -> Stribeck {
        // A geared joint: 0.9 N·m to break away, 0.6 N·m sliding, dip scale 0.02 rad/s, light viscous.
        Stribeck::new(0.9, 0.6, 0.02, 0.05)
    }

    #[test]
    fn the_curve_has_breakaway_a_dip_and_a_viscous_rise() {
        let s = joint();
        // As v → 0⁺ the magnitude approaches static friction, which a Coulomb model cannot represent.
        let near_zero = s.torque(1e-9);
        assert!((near_zero - s.tau_s).abs() < 1e-6, "breakaway should approach {}, got {near_zero}", s.tau_s);
        // At high speed it approaches Coulomb plus viscous.
        let fast = s.torque(5.0);
        assert!((fast - (s.tau_c + s.sigma_2 * 5.0)).abs() < 1e-9, "high speed should be Coulomb + viscous");

        // **The dip: friction is genuinely LOWER at some intermediate speed than at either extreme.** This is
        // the signature of Stribeck and the mechanism behind stick-slip.
        let dip_v = s.dip_velocity(1.0, 20_000).expect("this parameterisation has a dip");
        let dip_t = s.torque(dip_v);
        assert!(dip_t < s.tau_s, "the dip must be below breakaway: {dip_t} vs {}", s.tau_s);
        assert!(dip_v > 0.0 && dip_v < 1.0, "the dip should be interior, at {dip_v}");
        // NOT below Coulomb: total friction is always >= tau_C, because once g(v) has decayed to tau_C the
        // viscous term only adds. Measured, the minimum here is 0.6030 against tau_C = 0.6. A first version of
        // this test asserted "below Coulomb" and was simply wrong about the model.
        assert!(dip_t >= s.tau_c, "the total minimum cannot fall below Coulomb: {dip_t} vs {}", s.tau_c);
        assert!(dip_t < s.tau_c * 1.02, "but it should approach it closely: {dip_t}");

        // **The real signature is NON-MONOTONICITY** — friction genuinely DECREASES with speed over a range,
        // which is the mechanism behind stick-slip and the thing Coulomb cannot produce.
        let slow = s.torque(0.001);
        let faster = s.torque(0.05);
        assert!(faster < slow, "friction must fall with speed somewhere: {faster} at 0.05 vs {slow} at 0.001");
        assert!(slow - faster > 0.2, "and by a substantial margin, got {}", slow - faster);

        // A curve with no static excess has no dip to find — so the detector is not just reporting noise.
        let flat = Stribeck::new(0.6, 0.6, 0.02, 0.05);
        assert!(flat.dip_velocity(1.0, 20_000).is_none(), "tau_s == tau_c means no Stribeck dip");
    }

    #[test]
    fn friction_opposes_motion_and_is_odd_in_velocity() {
        let s = joint();
        for v in [0.001, 0.02, 0.5, 3.0] {
            assert!(s.torque(v) > 0.0, "friction opposes positive motion");
            assert!((s.torque(-v) + s.torque(v)).abs() < 1e-12, "the law must be odd in v");
        }
        // At rest the static curve is silent: any value up to tau_s is admissible and choosing one needs the
        // applied load, which only the stateful model has.
        assert_eq!(s.torque(0.0), 0.0);
    }

    /// **The cross-check that makes both models trustworthy: LuGre in steady sliding must reduce exactly to the
    /// Stribeck curve it was built from.** Each model verified only against itself would prove nothing.
    #[test]
    fn lugre_reduces_to_the_stribeck_curve_in_steady_sliding() {
        let s = joint();
        for sigma_0 in [1.0e4, 1.0e5] {
            let mut l = LuGre::new(s, sigma_0, 0.0);
            for v in [0.005, 0.02, 0.1, 1.0, 4.0] {
                // Analytic steady state
                l.z = l.steady_z(v);
                assert!(l.z_dot(v).abs() < 1e-9, "steady_z should make z_dot vanish, got {}", l.z_dot(v));
                assert!(
                    (l.torque(v) - s.torque(v)).abs() < 1e-9,
                    "v={v}: LuGre steady torque {} vs Stribeck {}",
                    l.torque(v),
                    s.torque(v)
                );

                // And reached by integration from rest, not just asserted algebraically.
                let mut m = LuGre::new(s, sigma_0, 0.0);
                let dt = m.max_stable_dt(v) * 0.1;
                for _ in 0..200_000 {
                    m.step(dt, v);
                }
                assert!(
                    (m.torque(v) - s.torque(v)).abs() < 1e-6 * s.torque(v).abs().max(1e-3),
                    "v={v} sigma0={sigma_0}: integrated to {} vs Stribeck {}",
                    m.torque(v),
                    s.torque(v)
                );
            }
        }
    }

    /// **Presliding is elastic**: before breakaway the joint behaves as a spring of rate `σ₀`, which is why
    /// position control has a deadband. A velocity-only law has no state and cannot produce this at all.
    #[test]
    fn presliding_behaves_as_a_spring_of_stiffness_sigma_0() {
        let s = joint();
        let sigma_0 = 1.0e5;
        let mut l = LuGre::new(s, sigma_0, 0.0);
        // Creep forward very slowly, far below the Stribeck velocity, and accumulate displacement.
        let (v, dt) = (1.0e-5, 1.0e-3);
        let mut x = 0.0;
        for _ in 0..200 {
            l.step(dt, v);
            x += v * dt;
        }
        // z tracks displacement but LAGS it, because the decay term σ₀|v|z/g is already non-negligible here:
        // measured z/x ≈ 0.90 after 0.2 s. So the regime is spring-LIKE rather than an exact spring, and the
        // honest test is that torque is approximately LINEAR in displacement, checked at two displacements,
        // rather than that z equals x. A first version asserted z ≈ x to 0.1% and failed for that reason.
        assert!(l.z > 0.8 * x && l.z < x, "bristle deflection tracks but lags displacement: z={} x={x}", l.z);
        let tau_full = l.torque(v);

        let mut half = LuGre::new(s, sigma_0, 0.0);
        for _ in 0..100 {
            half.step(dt, v);
        }
        let tau_half = half.torque(v);
        // Doubling the displacement should roughly double the torque — spring-like in DISPLACEMENT, which no
        // velocity-only law can express, since v is identical in both runs.
        assert!(
            (tau_full / tau_half - 2.0).abs() < 0.15,
            "torque should be ~linear in displacement: {tau_full} vs {tau_half} (ratio {})",
            tau_full / tau_half
        );
        // And it is still below breakaway, so this really is the presliding regime rather than sliding.
        assert!(l.torque(v) < s.tau_s, "must not have broken away yet: {} vs {}", l.torque(v), s.tau_s);
    }

    /// Reversing direction produces **hysteresis**: the bristle must be unwound before it can load the other
    /// way, so force lags displacement. A static curve jumps discontinuously instead.
    #[test]
    fn reversal_produces_hysteresis_rather_than_a_jump() {
        let s = joint();
        let mut l = LuGre::new(s, 1.0e5, 0.0);
        let dt = 1.0e-5;
        // load forward into sliding
        for _ in 0..20_000 {
            l.step(dt, 0.05);
        }
        let forward = l.torque(0.05);
        assert!(forward > 0.0);
        // reverse instantly: the torque cannot flip instantly, because z must unwind first
        let immediately_after = l.torque(-0.05);
        assert!(
            immediately_after > 0.0,
            "torque should still oppose the OLD direction immediately after reversal, got {immediately_after}"
        );
        // after enough time at the new velocity it settles to the mirrored value
        for _ in 0..40_000 {
            l.step(dt, -0.05);
        }
        assert!(
            (l.torque(-0.05) + forward).abs() < 1e-3 * forward,
            "settled reverse torque {} should mirror {forward}",
            l.torque(-0.05)
        );
    }

    /// At zero velocity the bristle does not decay, so presliding is non-dissipative and the stable step is
    /// unbounded — the degenerate case a naive `g(v)/(σ₀|v|)` would divide by zero on.
    #[test]
    fn the_stable_step_is_reported_and_unbounded_at_rest() {
        let l = LuGre::new(joint(), 1.0e5, 0.0);
        assert_eq!(l.max_stable_dt(0.0), f64::INFINITY, "no decay at rest");
        // Faster sliding means a stiffer bristle equation and a smaller stable step.
        assert!(l.max_stable_dt(1.0) < l.max_stable_dt(0.01), "the bound must tighten with speed");
        assert!(l.max_stable_dt(1.0) > 0.0);
        // A stiffer bristle also tightens it, linearly.
        let stiff = LuGre::new(joint(), 1.0e6, 0.0);
        assert!((stiff.max_stable_dt(1.0) * 10.0 - l.max_stable_dt(1.0)).abs() < 1e-12);
    }
}
