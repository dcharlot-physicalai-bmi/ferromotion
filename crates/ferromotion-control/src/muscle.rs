//! **Hill muscle model** — biological actuation, and the cleanest example of *morphological
//! computation*: control work done by the body rather than the controller.
//!
//! A Hill muscle's force is the product of three factors (Hill 1938; Zajac 1989):
//! activation `a`, a **force-length** curve peaking at the optimal fiber length, and a
//! **force-velocity** curve — Hill's hyperbola — plus a passive elastic term:
//!
//! ```text
//!   F = F₀·[ a · f_l(l) · f_v(v) + f_pe(l) ]
//! ```
//!
//! The force-velocity relation is the interesting one. Because `∂F/∂v > 0` everywhere, a muscle
//! *automatically* resists being stretched and yields when shortening — **instantaneous, zero-delay
//! damping that no neural loop has to compute**. Biologists call this a *preflex*: the muscle rejects
//! a perturbation before any reflex could even fire. Together with the force-length curve (intrinsic
//! stiffness), a muscle at *constant activation* is already a self-stabilizing spring-damper. That is
//! what "the body is part of the controller" means, concretely. Pure Rust → WASM-clean.

/// A Hill-type muscle-tendon unit.
#[derive(Clone, Copy, Debug)]
pub struct HillMuscle {
    /// Maximum isometric force `F₀`.
    pub f0: f64,
    /// Optimal fiber length `l₀` (where `f_l` peaks).
    pub l0: f64,
    /// Maximum shortening velocity (positive magnitude).
    pub v_max: f64,
    /// Width of the force-length curve.
    pub width: f64,
    /// Hill's `A = a/F₀` (curvature of the hyperbola; ≈0.25).
    pub a_rel: f64,
    /// Eccentric plateau, as a multiple of `F₀` (≈1.5).
    pub f_ecc: f64,
    /// Strain beyond `l₀` at which the passive element reaches `F₀`.
    pub pe_strain: f64,
}

impl Default for HillMuscle {
    fn default() -> Self {
        Self { f0: 1000.0, l0: 0.1, v_max: 0.5, width: 0.45, a_rel: 0.25, f_ecc: 1.5, pe_strain: 0.6 }
    }
}

impl HillMuscle {
    /// Active force-length: a bell curve peaking at `l₀` — the muscle's intrinsic *stiffness*.
    pub fn force_length(&self, l: f64) -> f64 {
        (-((l / self.l0 - 1.0) / self.width).powi(2)).exp()
    }

    /// The constant that makes Hill's hyperbola meet the eccentric branch with matching slope at `v=0`.
    fn ecc_c(&self) -> f64 {
        self.v_max * self.a_rel * (self.f_ecc - 1.0) / (self.a_rel + 1.0)
    }

    /// Force-velocity (`v > 0` is lengthening). Shortening follows **Hill's hyperbola**; lengthening
    /// saturates at the eccentric plateau. `C¹` at `v = 0` by construction.
    pub fn force_velocity(&self, v: f64) -> f64 {
        if v <= 0.0 {
            if v <= -self.v_max {
                return 0.0; // nothing left at maximum shortening speed
            }
            (self.v_max + v) / (self.v_max - v / self.a_rel)
        } else {
            let c = self.ecc_c();
            (self.f_ecc * v + c) / (v + c)
        }
    }

    /// Passive elastic element: slack below `l₀`, stiffening beyond it.
    pub fn passive_force(&self, l: f64) -> f64 {
        let strain = l / self.l0 - 1.0;
        if strain <= 0.0 { 0.0 } else { (strain / self.pe_strain).powi(2) }
    }

    /// Total muscle force at activation `a`, length `l`, velocity `v`.
    pub fn force(&self, a: f64, l: f64, v: f64) -> f64 {
        self.f0 * (a.clamp(0.0, 1.0) * self.force_length(l) * self.force_velocity(v) + self.passive_force(l))
    }

    /// First-order activation dynamics `ȧ = (u − a)/τ` (deactivation is slower than activation).
    pub fn activation_rate(&self, a: f64, u: f64, tau_act: f64, tau_deact: f64) -> f64 {
        let tau = if u > a { tau_act } else { tau_deact };
        (u - a) / tau
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m() -> HillMuscle {
        HillMuscle::default()
    }

    #[test]
    fn shortening_branch_satisfies_hills_hyperbola_exactly() {
        // Hill (1938): (F + a)(v_shortening + b) = (F₀ + a)·b, with a = A·F₀ and b = A·v_max.
        // Our f_v should satisfy this identically, not approximately.
        let mu = m();
        let (a_h, b_h) = (mu.a_rel * mu.f0, mu.a_rel * mu.v_max);
        let rhs = (mu.f0 + a_h) * b_h;
        for i in 1..40 {
            let v = -mu.v_max * i as f64 / 40.0; // shortening
            let f = mu.f0 * mu.force_velocity(v); // fully activated, at optimal length
            let lhs = (f + a_h) * (-v + b_h); // −v = shortening speed
            assert!((lhs - rhs).abs() / rhs < 1e-12, "Hill's hyperbola violated at v={v}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn force_velocity_has_the_right_limits_and_is_c1_at_zero() {
        let mu = m();
        assert!((mu.force_velocity(0.0) - 1.0).abs() < 1e-12, "isometric force should be F₀");
        assert!(mu.force_velocity(-mu.v_max).abs() < 1e-12, "no force at maximum shortening velocity");
        assert!((mu.force_velocity(1e6) - mu.f_ecc).abs() < 1e-3, "lengthening should saturate at the eccentric plateau");
        // Eccentric force exceeds isometric — the classic Hill result.
        assert!(mu.force_velocity(0.05) > 1.0, "lengthening should produce more force than isometric");
        // Slope is continuous across v = 0 (the two branches were matched deliberately).
        let eps = 1e-7;
        let left = (mu.force_velocity(-eps) - mu.force_velocity(-2.0 * eps)) / eps;
        let right = (mu.force_velocity(2.0 * eps) - mu.force_velocity(eps)) / eps;
        assert!((left - right).abs() / left < 1e-3, "force-velocity is not C¹ at v=0: {left} vs {right}");
    }

    #[test]
    fn force_length_peaks_at_the_optimal_fiber_length() {
        let mu = m();
        assert!((mu.force_length(mu.l0) - 1.0).abs() < 1e-12);
        for f in [0.6, 0.8, 1.2, 1.5] {
            assert!(mu.force_length(f * mu.l0) < 1.0, "force-length should peak at l₀");
        }
        // Passive element is slack below l₀ and stiffens beyond it.
        assert_eq!(mu.passive_force(0.9 * mu.l0), 0.0);
        assert!(mu.passive_force(1.4 * mu.l0) > mu.passive_force(1.2 * mu.l0));
    }

    #[test]
    fn the_force_velocity_relation_is_intrinsic_damping_a_preflex() {
        // ∂F/∂v > 0 everywhere: stretch the muscle and it pulls back harder, all without a controller.
        let mu = m();
        let eps = 1e-6;
        for v in [-0.4, -0.2, -0.05, 0.0, 0.05, 0.2, 0.4] {
            let d = (mu.force(1.0, mu.l0, v + eps) - mu.force(1.0, mu.l0, v - eps)) / (2.0 * eps);
            assert!(d > 0.0, "muscle must resist lengthening at v={v} (∂F/∂v = {d})");
        }
    }

    #[test]
    fn constant_activation_self_stabilizes_morphological_computation() {
        // A mass pulled by a constant load, held by a muscle at FIXED activation — no feedback, no
        // controller, no reflex. The force-length curve supplies stiffness and force-velocity supplies
        // damping, so the body alone settles to equilibrium and rejects a velocity perturbation.
        let mu = m();
        let (mass, load, act) = (2.0, 300.0, 0.5); // load < peak muscle force at this activation
        let settle = |x0: f64, v0: f64| -> (f64, f64) {
            let (mut x, mut v, dt) = (x0, v0, 1e-5);
            for _ in 0..400_000 {
                let l = mu.l0 + x; // stretching the muscle as the load pulls out
                let f_m = mu.force(act, l, v);
                let acc = (load - f_m) / mass;
                v += acc * dt;
                x += v * dt;
            }
            (x, v)
        };
        let (x1, v1) = settle(0.0, 0.0);
        let (x2, v2) = settle(0.01, 0.5); // displaced *and* moving
        // Both come to rest …
        assert!(v1.abs() < 1e-3 && v2.abs() < 1e-3, "muscle did not settle: {v1}, {v2}");
        // … at the *same* equilibrium, from different perturbations — the body is the controller.
        assert!((x1 - x2).abs() < 1e-3, "different perturbations reached different equilibria: {x1} vs {x2}");
        // And that equilibrium really balances the load.
        let f_eq = mu.force(act, mu.l0 + x1, 0.0);
        assert!((f_eq - load).abs() / load < 1e-2, "equilibrium does not balance the load: {f_eq} vs {load}");
    }

    #[test]
    fn activation_dynamics_track_the_neural_drive() {
        let mu = m();
        let (mut a, dt) = (0.0, 1e-4);
        for _ in 0..50_000 {
            a += mu.activation_rate(a, 1.0, 0.01, 0.04) * dt;
        }
        assert!((a - 1.0).abs() < 1e-3, "activation should approach the drive: {a}");
        // Deactivation is slower than activation (τ_deact > τ_act).
        let fast = mu.activation_rate(0.5, 1.0, 0.01, 0.04).abs();
        let slow = mu.activation_rate(0.5, 0.0, 0.01, 0.04).abs();
        assert!(fast > slow, "activation should be faster than deactivation");
    }
}
