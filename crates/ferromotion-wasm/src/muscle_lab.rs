//! **Muscle lab** — the rig behind the *morphological computation* textbook chapter.
//!
//! A mass hangs on a Hill muscle held at **fixed activation**: no controller, no feedback, no reflex.
//! The muscle's force-length curve supplies stiffness and its force-velocity curve supplies damping,
//! so the body alone catches a perturbation — instantly, because that is physics rather than
//! computation.
//!
//! To make that claim falsifiable the lab runs a **controlled experiment**. We measure the muscle's
//! actual impedance at its operating point (`K = ∂F/∂x`, `B = ∂F/∂v`, by finite differences) and hand
//! *exactly those numbers* to a neural controller acting on delayed state. Identical mechanical
//! impedance; the only difference is the delay. Whatever happens next is the chapter's whole point,
//! and the reader can dial τ themselves.

use ferromotion_control::HillMuscle;
use wasm_bindgen::prelude::*;

/// A mass on a muscle: `m·ẍ = load − F(a, l₀+x, ẋ)`, with `x` the stretch away from optimal length.
#[wasm_bindgen]
pub struct MuscleRig {
    muscle: HillMuscle,
    mass: f64,
    load: f64,
    act: f64,
    x: f64,
    v: f64,
    /// When false, the force-velocity factor is pinned to 1 — a muscle with no intrinsic damping.
    fv_on: bool,
}

/// Muscle force with the force-velocity factor optionally disabled (a pure spring).
fn force_of(m: &HillMuscle, a: f64, l: f64, v: f64, fv_on: bool) -> f64 {
    if fv_on {
        m.force(a, l, v)
    } else {
        m.f0 * (a.clamp(0.0, 1.0) * m.force_length(l) + m.passive_force(l))
    }
}

#[wasm_bindgen]
impl MuscleRig {
    #[wasm_bindgen(constructor)]
    pub fn new(mass: f64, load: f64, activation: f64) -> MuscleRig {
        let muscle = HillMuscle::default();
        let mut r = MuscleRig { muscle, mass, load, act: activation, x: 0.0, v: 0.0, fv_on: true };
        r.x = r.equilibrium();
        r
    }

    /// The stretch at which the muscle exactly balances the load (bisection on `F(a, l₀+x, 0) = load`).
    pub fn equilibrium(&self) -> f64 {
        let f = |x: f64| force_of(&self.muscle, self.act, self.muscle.l0 + x, 0.0, self.fv_on) - self.load;
        let (mut lo, mut hi) = (-0.5 * self.muscle.l0, 2.0 * self.muscle.l0);
        if f(lo) * f(hi) > 0.0 {
            return 0.0; // no balance point in range (load beyond what this activation can hold)
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(lo) * f(mid) <= 0.0 { hi = mid } else { lo = mid }
        }
        0.5 * (lo + hi)
    }

    /// **Intrinsic stiffness** `K = ∂F/∂x` at the current operating point — from the force-length curve.
    pub fn stiffness(&self) -> f64 {
        let e = 1e-6;
        let f = |x: f64| force_of(&self.muscle, self.act, self.muscle.l0 + x, self.v, self.fv_on);
        (f(self.x + e) - f(self.x - e)) / (2.0 * e)
    }

    /// **Intrinsic damping** `B = ∂F/∂v` at the current operating point — from the force-velocity curve.
    /// This is the preflex: it is a property of the tissue, available with zero delay.
    pub fn damping(&self) -> f64 {
        let e = 1e-6;
        let f = |v: f64| force_of(&self.muscle, self.act, self.muscle.l0 + self.x, v, self.fv_on);
        (f(self.v + e) - f(self.v - e)) / (2.0 * e)
    }

    /// Advance the plant by `dt` (semi-implicit Euler).
    pub fn step(&mut self, dt: f64) {
        let f_m = force_of(&self.muscle, self.act, self.muscle.l0 + self.x, self.v, self.fv_on);
        let acc = (self.load - f_m) / self.mass;
        self.v += acc * dt;
        self.x += self.v * dt;
    }

    /// Perturb the mass — the reader kicking it.
    pub fn kick(&mut self, dv: f64) {
        self.v += dv;
    }

    pub fn displace(&mut self, x: f64) {
        self.x = x;
        self.v = 0.0;
    }

    pub fn x(&self) -> f64 { self.x }
    pub fn v(&self) -> f64 { self.v }
    pub fn force(&self) -> f64 {
        force_of(&self.muscle, self.act, self.muscle.l0 + self.x, self.v, self.fv_on)
    }

    pub fn set_activation(&mut self, a: f64) { self.act = a.clamp(0.0, 1.0) }
    pub fn activation(&self) -> f64 { self.act }
    pub fn set_load(&mut self, l: f64) { self.load = l }
    /// Toggle the force-velocity curve: the difference between a muscle and a plain spring.
    pub fn set_fv(&mut self, on: bool) { self.fv_on = on }
    pub fn l0(&self) -> f64 { self.muscle.l0 }
    pub fn f0(&self) -> f64 { self.muscle.f0 }
    pub fn v_max(&self) -> f64 { self.muscle.v_max }

    /// The force-velocity factor at a single velocity — for reading the curve interactively.
    pub fn fv_at(&self, v: f64) -> f64 {
        self.muscle.force_velocity(v)
    }

    /// Sample the force-velocity curve for plotting: `f_v(v)` over `[−v_max, +v_max]`.
    pub fn fv_curve(&self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let v = -self.muscle.v_max + 2.0 * self.muscle.v_max * i as f64 / (n - 1) as f64;
                self.muscle.force_velocity(v)
            })
            .collect()
    }

    /// Sample the force-length curve for plotting over `[0.4·l₀, 1.8·l₀]` (active + passive).
    pub fn fl_curve(&self, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let l = self.muscle.l0 * (0.4 + 1.4 * i as f64 / (n - 1) as f64);
                self.muscle.force_length(l) + self.muscle.passive_force(l)
            })
            .collect()
    }

    /// Live check of **Hill's hyperbola** `(F+a)(v+b) = (F₀+a)b` on the shortening branch: the
    /// worst relative residual over the curve. The reader can watch the identity hold on-device.
    pub fn hill_residual(&self) -> f64 {
        let (a_h, b_h) = (self.muscle.a_rel * self.muscle.f0, self.muscle.a_rel * self.muscle.v_max);
        let rhs = (self.muscle.f0 + a_h) * b_h;
        let mut worst: f64 = 0.0;
        for i in 1..60 {
            let v = -self.muscle.v_max * i as f64 / 60.0;
            let f = self.muscle.f0 * self.muscle.force_velocity(v);
            worst = worst.max(((f + a_h) * (-v + b_h) - rhs).abs() / rhs);
        }
        worst
    }
}

/// The same mass and load, held by an actuator driven by a **neural controller acting on delayed
/// state** — given exactly the muscle's measured `K` and `B`. The only difference is the delay.
#[wasm_bindgen]
pub struct DelayedRig {
    mass: f64,
    k: f64,
    b: f64,
    /// Delay in seconds (a human spinal reflex is ≈0.03 s; a cortical loop is far slower).
    tau: f64,
    x: f64,
    v: f64,
    hist: Vec<(f64, f64)>, // ring buffer of past (x, v)
    head: usize,
    steps: usize,
}

#[wasm_bindgen]
impl DelayedRig {
    /// `k`/`b` should be the impedance measured off the muscle, so the comparison is controlled.
    #[wasm_bindgen(constructor)]
    pub fn new(mass: f64, k: f64, b: f64, tau: f64, dt: f64) -> DelayedRig {
        let steps = ((tau / dt).round() as usize).max(1);
        DelayedRig { mass, k, b, tau, x: 0.0, v: 0.0, hist: vec![(0.0, 0.0); steps + 1], head: 0, steps }
    }

    /// `m·ẍ = −K·x(t−τ) − B·v(t−τ)` — the same restoring impedance the muscle has, applied late.
    pub fn step(&mut self, dt: f64) {
        let n = self.hist.len();
        self.hist[self.head] = (self.x, self.v);
        self.head = (self.head + 1) % n;
        let (xd, vd) = self.hist[self.head]; // the oldest sample = state τ ago
        let acc = (-self.k * xd - self.b * vd) / self.mass;
        self.v += acc * dt;
        self.x += self.v * dt;
    }

    pub fn kick(&mut self, dv: f64) { self.v += dv }
    pub fn displace(&mut self, x: f64) {
        self.x = x;
        self.v = 0.0;
        for h in self.hist.iter_mut() { *h = (x, 0.0) }
    }
    pub fn x(&self) -> f64 { self.x }
    pub fn v(&self) -> f64 { self.v }
    pub fn tau(&self) -> f64 { self.tau }
    pub fn delay_steps(&self) -> usize { self.steps }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rig() -> MuscleRig {
        MuscleRig::new(2.0, 300.0, 0.5)
    }

    #[test]
    fn the_rig_starts_at_a_real_equilibrium_and_stays_there() {
        // The constructor solves for the balance point; with no perturbation nothing should move.
        let mut r = rig();
        let x0 = r.x();
        assert!((r.force() - r.load).abs() / r.load < 1e-6, "equilibrium does not balance the load");
        for _ in 0..200_000 {
            r.step(1e-5);
        }
        assert!((r.x() - x0).abs() < 1e-6, "drifted off equilibrium: {} vs {x0}", r.x());
        assert!(r.v().abs() < 1e-6, "should be at rest");
    }

    #[test]
    fn measured_impedance_is_a_restoring_spring_and_a_damper() {
        // K > 0 (force-length ⇒ stiffness) and B > 0 (force-velocity ⇒ damping) — the preflex.
        let r = rig();
        assert!(r.stiffness() > 0.0, "muscle should be a restoring spring, K = {}", r.stiffness());
        assert!(r.damping() > 0.0, "muscle should be a damper, B = {}", r.damping());
        // Killing the force-velocity curve removes the damping and only the damping.
        let mut s = rig();
        s.set_fv(false);
        assert!(s.damping().abs() < 1e-9, "no f_v should mean no intrinsic damping, got {}", s.damping());
        assert!(s.stiffness() > 0.0, "stiffness must survive: {}", s.stiffness());
    }

    #[test]
    fn the_force_velocity_curve_is_what_catches_the_perturbation() {
        // Same kick, same activation, same load. The only difference is f_v.
        let settle = |fv: bool| -> f64 {
            let mut r = rig();
            r.set_fv(fv);
            r.displace(r.equilibrium());
            r.kick(0.4);
            let mut peak: f64 = 0.0;
            for i in 0..300_000 {
                r.step(1e-5);
                if i > 150_000 {
                    peak = peak.max((r.x() - r.equilibrium()).abs()); // amplitude in the second half
                }
            }
            peak
        };
        let with_fv = settle(true);
        let without = settle(false);
        // With f_v the wobble is gone; without it the mass is still ringing.
        assert!(with_fv < 1e-3, "muscle should have absorbed the kick, residual {with_fv}");
        assert!(without > 10.0 * with_fv, "without f_v it should still be oscillating: {without}");
    }

    #[test]
    fn hills_hyperbola_holds_live_in_the_rig() {
        assert!(rig().hill_residual() < 1e-12, "residual {}", rig().hill_residual());
    }

    #[test]
    fn curves_are_sane_for_plotting() {
        let r = rig();
        let fv = r.fv_curve(101);
        assert_eq!(fv.len(), 101);
        assert!(fv[0].abs() < 1e-9, "f_v should vanish at maximum shortening");
        assert!((fv[50] - 1.0).abs() < 1e-9, "midpoint is isometric ⇒ 1");
        assert!(fv[100] > 1.0, "eccentric end should exceed isometric");
        assert!(fv.windows(2).all(|w| w[1] >= w[0] - 1e-12), "f_v must be monotone increasing");
        let fl = r.fl_curve(101);
        assert_eq!(fl.len(), 101);
        assert!(fl.iter().all(|&f| f >= 0.0));
    }

    #[test]
    fn a_zero_delay_neural_controller_matches_the_muscle() {
        // Sanity on the comparison itself: with the muscle's own K and B and no delay, the
        // controller reproduces the muscle's recovery. Any later difference is therefore the delay.
        let r = rig();
        let (k, b) = (r.stiffness(), r.damping());
        let dt = 1e-5;
        let mut d = DelayedRig::new(r.mass, k, b, dt, dt); // one step ≈ no delay
        d.displace(0.0);
        d.kick(0.2);
        let mut m = rig();
        m.displace(m.equilibrium());
        m.kick(0.2);
        let eq = m.equilibrium();
        for _ in 0..20_000 {
            d.step(dt);
            m.step(dt);
        }
        // Both settle toward their rest point on the same timescale (linearization ⇒ not identical).
        assert!(d.x().abs() < 0.02, "undelayed controller should be recovering: {}", d.x());
        assert!((m.x() - eq).abs() < 0.02, "muscle should be recovering: {}", m.x() - eq);
    }

    #[test]
    fn delay_destabilizes_the_identical_impedance() {
        // THE CHAPTER'S POINT. Same mass, same K, same B — only the delay differs. Find the delay
        // at which the neural controller loses the fight, by measurement rather than assertion.
        let r = rig();
        let (k, b, mass) = (r.stiffness(), r.damping(), r.mass);
        let dt = 1e-5;
        let amplitude_after = |tau: f64| -> f64 {
            let mut d = DelayedRig::new(mass, k, b, tau, dt);
            d.displace(0.0);
            d.kick(0.2);
            let mut peak: f64 = 0.0;
            for i in 0..60_000 {
                d.step(dt);
                if i > 30_000 {
                    peak = peak.max(d.x().abs());
                }
                if !d.x().is_finite() || d.x().abs() > 1e3 {
                    return f64::INFINITY; // blown up
                }
            }
            peak
        };
        let quick = amplitude_after(dt); // effectively instantaneous
        assert!(quick < 0.05, "with no delay the controller should be settling, got {quick}");
        // A slow loop cannot hold what the tissue holds for free.
        let slow = amplitude_after(0.05);
        assert!(slow > 10.0 * quick, "a 50 ms loop should do far worse than none: {slow} vs {quick}");
    }

    #[test]
    fn the_critical_delay_matches_the_classical_delay_margin() {
        // The destabilization above is not a simulation artifact: it is the textbook delay margin.
        // For plant m·s² under feedback (K + B·s)·e^{−sτ}, gain crossover solves m²ω⁴ = K² + B²ω²
        // and the loop goes unstable at τ_c = atan2(Bω, K)/ω. Bisect the simulation and compare.
        let r = rig();
        let (k, b, mass) = (r.stiffness(), r.damping(), r.mass);
        let dt = 2e-6;
        let stable = |tau: f64| -> bool {
            let mut d = DelayedRig::new(mass, k, b, tau, dt);
            d.displace(0.0);
            d.kick(0.2);
            let mut peak: f64 = 0.0;
            for i in 0..400_000 {
                d.step(dt);
                if i > 200_000 {
                    peak = peak.max(d.x().abs());
                }
                if !d.x().is_finite() || d.x().abs() > 1e3 {
                    return false;
                }
            }
            peak < 0.01
        };
        let (mut lo, mut hi) = (1e-6, 5e-3);
        assert!(stable(lo) && !stable(hi), "bisection must bracket the critical delay");
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if stable(mid) { lo = mid } else { hi = mid }
        }
        let u = (b * b + (b.powi(4) + 4.0 * mass * mass * k * k).sqrt()) / (2.0 * mass * mass);
        let w = u.sqrt();
        let tau_analytic = (b * w).atan2(k) / w;
        assert!(
            (lo - tau_analytic).abs() / tau_analytic < 0.02,
            "measured critical delay {:.4} ms vs analytic margin {:.4} ms",
            lo * 1e3,
            tau_analytic * 1e3
        );
        // The punchline, as an assertion: a spinal reflex (~30 ms) is far beyond this margin, so the
        // muscle's own impedance is not something a neural loop could deliver at all.
        assert!(0.030 > 10.0 * tau_analytic, "reflex delay should dwarf the margin");
    }
}
