//! **McKibben muscle** — the pneumatic actuator whose whole behaviour follows from a braid angle.
//!
//! An elastomer bladder inside a helical braid. Pressurise it, the bladder pushes the braid outward, the braid
//! is inextensible so it must shorten, and the muscle contracts. [`HillMuscle`](crate::HillMuscle) models the *biological*
//! force-velocity relationship; this models the pneumatic hardware that borrows its name, and the two have very
//! little in common mechanically.
//!
//! Everything here comes out of one geometric fact and one application of virtual work:
//!
//! ```text
//!   F = P · (−dV/dL)
//! ```
//!
//! The force is the pressure times how fast the enclosed volume changes with length. That is the whole model,
//! and the closed form below is its consequence rather than an independent correlation — which is why the tests
//! check the closed form against a **numerically differentiated volume**, a path that shares no algebra with it.
//!
//! # The magic angle, and the sign change at it
//!
//! With initial braid angle `θ₀` (from the muscle's axis) and contraction ratio `ε = 1 − L/L₀`:
//!
//! ```text
//!   F = (π D₀² P / 4) · (3(1 − ε)² cos²θ₀ − 1) / sin²θ₀
//! ```
//!
//! The bracket vanishes when `cos θ = 1/√3`, i.e. **θ = 54.7356°**. That angle governs everything:
//!
//! * Force falls to zero there, so free contraction stops at `ε_max = 1 − 1/(√3 cos θ₀)`.
//! * A braid wound at `θ₀ > 54.74°` has `1/(√3 cos θ₀) > 1`, so `ε_max < 0`: **the muscle extends instead of
//!   contracting**. Same hardware, opposite sign, decided by how the braid was wound.
//! * At exactly the magic angle the muscle does nothing at all at any pressure.
//!
//! [`Mckibben::free_contraction`] returns the signed value and does not clamp it, because a negative one is a
//! real device (an extensor) and not an error to be hidden.
//!
//! # Why it behaves like a spring, and what that costs
//!
//! Force falls monotonically with contraction at fixed pressure, from a large blocked force at `ε = 0` to zero
//! at `ε_max`. So a McKibben muscle is a pressure-programmable spring, not a force source: its output depends on
//! its own position. Useful for compliant contact, awkward for trajectory tracking, and the reason these are
//! usually run in antagonistic pairs — the pair's *difference* sets torque while its *sum* sets stiffness, two
//! controls from two pressures. [`Mckibben::antagonistic_torque`] and
//! [`antagonistic_stiffness`](Mckibben::antagonistic_stiffness) make that decomposition explicit.
//!
//! # What this ideal model leaves out
//!
//! Gaylord's model assumes a frictionless, infinitely thin, perfectly inextensible braid and a bladder with no
//! stiffness of its own. Real muscles fall short of the predicted force by **10-30%** near zero contraction and
//! show hysteresis of a similar order from braid friction and bladder deformation. The free-contraction figure
//! is likewise optimistic: this model gives about 38% for a 20° braid where measured devices reach 25-30%. Stated
//! plainly, because the ideal figures are the ones that get quoted into specifications and they are not
//! conservative.

use std::f64::consts::PI;

/// The braid angle at which the force term vanishes, `arccos(1/√3)`, in radians. `54.7356°`.
pub const MAGIC_ANGLE: f64 = 0.955_316_618_124_509_3;

/// A McKibben (braided pneumatic) muscle in its ideal Gaylord form.
#[derive(Clone, Copy, Debug)]
pub struct Mckibben {
    /// Initial (uncontracted) bladder diameter (m).
    pub d0: f64,
    /// Initial (uncontracted) length (m).
    pub l0: f64,
    /// Initial braid angle from the muscle axis (rad).
    pub theta0: f64,
}

impl Mckibben {
    /// A muscle from its resting geometry. Returns `None` unless the diameter and length are positive and the
    /// braid angle lies strictly in `(0, π/2)`.
    ///
    /// The open interval matters: at `θ₀ = 0` the braid is axial and `sin²θ₀ = 0` makes the force undefined, and
    /// at `θ₀ = π/2` it is circumferential and the muscle has no length to change.
    pub fn new(d0: f64, l0: f64, theta0: f64) -> Option<Mckibben> {
        if !d0.is_finite() || !l0.is_finite() || !theta0.is_finite() {
            return None;
        }
        if d0 <= 0.0 || l0 <= 0.0 || theta0 <= 0.0 || theta0 >= 0.5 * PI {
            return None;
        }
        Some(Mckibben { d0, l0, theta0 })
    }

    /// Whether this muscle contracts (`θ₀ < 54.74°`) rather than extends.
    pub fn is_contractile(&self) -> bool {
        self.theta0 < MAGIC_ANGLE
    }

    /// Total braid fibre length implied by the resting geometry, `b = L₀/cos θ₀`.
    pub fn fibre_length(&self) -> f64 {
        self.l0 / self.theta0.cos()
    }

    /// Braid angle at contraction ratio `eps`, from `cos θ = (1 − ε) cos θ₀`.
    ///
    /// Returns `None` if the implied cosine leaves `[-1, 1]`, which is a contraction the geometry cannot reach.
    pub fn angle_at(&self, eps: f64) -> Option<f64> {
        // `eps > 1` is a NEGATIVE length, and the cosine bound alone does not catch it: at eps = 2 the cosine
        // is -cos(theta0), comfortably inside [-1, 1], and `acos` happily returns an obtuse angle for a muscle
        // turned inside out. `eps = 1` is the real limit (zero length, theta = 90 degrees) and is allowed.
        if !eps.is_finite() || eps > 1.0 {
            return None;
        }
        let c = (1.0 - eps) * self.theta0.cos();
        if !(-1.0..=1.0).contains(&c) {
            return None;
        }
        Some(c.acos())
    }

    /// Enclosed volume (m³) at contraction ratio `eps`.
    ///
    /// `V = b³ sin²θ cos θ / (4 π n²)` with `n` braid turns, written here in terms of the resting geometry so
    /// `n` cancels. This is the quantity whose derivative *is* the force, so it is the honest place to start.
    pub fn volume(&self, eps: f64) -> Option<f64> {
        let theta = self.angle_at(eps)?;
        // n = b sin(theta0) / (pi D0) from the resting diameter; b^3/n^2 then reduces cleanly.
        let b = self.fibre_length();
        let n = b * self.theta0.sin() / (PI * self.d0);
        Some(b * b * b * theta.sin() * theta.sin() * theta.cos() / (4.0 * PI * n * n))
    }

    /// **Axial force** (N) at gauge pressure `p` and contraction ratio `eps`.
    ///
    /// `F = (π D₀² P / 4)(3(1 − ε)² cos²θ₀ − 1)/sin²θ₀`. Positive is contractile (pulling the ends together).
    pub fn force(&self, p: f64, eps: f64) -> f64 {
        let (c0, s0) = (self.theta0.cos(), self.theta0.sin());
        let one_minus = 1.0 - eps;
        PI * self.d0 * self.d0 * p / 4.0 * (3.0 * one_minus * one_minus * c0 * c0 - 1.0) / (s0 * s0)
    }

    /// Blocked force (N) at pressure `p`: the force at zero contraction.
    pub fn blocked_force(&self, p: f64) -> f64 {
        self.force(p, 0.0)
    }

    /// **Free contraction ratio**: `ε` where the force reaches zero, `1 − 1/(√3 cos θ₀)`.
    ///
    /// **Signed and unclamped.** Negative means the device is an extensor, which is a real muscle wound past the
    /// magic angle rather than an error. Clamping it at zero would erase the distinction the braid angle makes.
    pub fn free_contraction(&self) -> f64 {
        1.0 - 1.0 / (3f64.sqrt() * self.theta0.cos())
    }

    /// Force gradient with contraction, `dF/dε` (N per unit ratio). Negative for a contractile muscle: it
    /// weakens as it shortens, which is what makes it spring-like.
    pub fn force_gradient(&self, p: f64, eps: f64) -> f64 {
        let (c0, s0) = (self.theta0.cos(), self.theta0.sin());
        PI * self.d0 * self.d0 * p / 4.0 * (-6.0 * (1.0 - eps) * c0 * c0) / (s0 * s0)
    }

    /// Effective axial stiffness (N/m) at a working point: `−dF/dL = (dF/dε)·(−1/L₀)·(−1)`.
    ///
    /// Positive for a contractile muscle, meaning it resists being stretched further from its equilibrium.
    pub fn stiffness(&self, p: f64, eps: f64) -> f64 {
        -self.force_gradient(p, eps) / self.l0
    }

    /// Work (J) done contracting from `eps_a` to `eps_b` at constant pressure, `∫F dx` with `dx = −L₀ dε`.
    ///
    /// Integrated in closed form: the force is quadratic in `ε`, so this is exact rather than quadrature.
    pub fn work(&self, p: f64, eps_a: f64, eps_b: f64) -> f64 {
        // F(eps) = K (3 c0^2 (1-eps)^2 - 1)/s0^2 with K = pi D0^2 P/4. Integrate over x = L0 eps.
        let (c0, s0) = (self.theta0.cos(), self.theta0.sin());
        let k = PI * self.d0 * self.d0 * p / 4.0 / (s0 * s0);
        let anti = |e: f64| {
            let u = 1.0 - e;
            // integral of (3 c0^2 u^2 - 1) de, with u = 1 - e, is -c0^2 u^3 - e
            -c0 * c0 * u * u * u - e
        };
        self.l0 * k * (anti(eps_b) - anti(eps_a))
    }

    /// Torque (N·m) from an antagonistic pair on a pulley of radius `r`, at joint contraction ratio `eps`.
    ///
    /// One muscle contracts as the other extends, so the joint sees the **difference** of the two forces. This is
    /// the control channel a paired arrangement buys.
    pub fn antagonistic_torque(&self, r: f64, p_a: f64, p_b: f64, eps: f64) -> f64 {
        let span = self.free_contraction();
        // The pair is mounted so that one is at eps and the other at (span - eps).
        r * (self.force(p_a, eps) - self.force(p_b, span - eps))
    }

    /// Joint stiffness (N·m/rad) from an antagonistic pair: the **sum** of the two muscles' contributions.
    ///
    /// Two pressures give two independent controls — torque from their difference, stiffness from their sum —
    /// which is the reason to use a pair rather than a muscle against a spring.
    pub fn antagonistic_stiffness(&self, r: f64, p_a: f64, p_b: f64, eps: f64) -> f64 {
        let span = self.free_contraction();
        r * r * (self.stiffness(p_a, eps) + self.stiffness(p_b, span - eps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn muscle() -> Mckibben {
        // 20 mm bladder, 200 mm long, 20 degree braid: a common commercial geometry.
        Mckibben::new(0.02, 0.2, 20f64.to_radians()).expect("valid geometry")
    }

    #[test]
    fn the_closed_form_force_is_the_pressure_times_the_volume_gradient() {
        // THE oracle. Gaylord's formula is derived from virtual work, F = P (-dV/dL), so differentiating the
        // volume numerically and comparing is a check on the whole derivation through a path that shares no
        // algebra with the closed form. If either the volume or the force expression were wrong, this fails.
        let m = muscle();
        let p = 300e3; // 3 bar gauge
        let h = 1e-7;
        for &eps in &[0.0f64, 0.05, 0.1, 0.2, 0.3, 0.35] {
            let v_plus = m.volume(eps + h).expect("reachable");
            let v_minus = m.volume(eps - h).expect("reachable");
            // dV/dL = (dV/deps) * (deps/dL) and L = L0 (1 - eps), so deps/dL = -1/L0.
            let dv_deps = (v_plus - v_minus) / (2.0 * h);
            let dv_dl = dv_deps * (-1.0 / m.l0);
            let f_virtual = -p * dv_dl;
            let f_closed = m.force(p, eps);
            assert!(
                (f_virtual - f_closed).abs() < 1e-5 * f_closed.abs().max(1.0),
                "eps={eps}: virtual work gives {f_virtual:.6} N, closed form {f_closed:.6} N"
            );
        }
    }

    #[test]
    fn the_magic_angle_is_where_the_force_vanishes() {
        // 54.7356 degrees, and the constant must be the arccos rather than a rounded degree value.
        assert!((MAGIC_ANGLE.cos() - 1.0 / 3f64.sqrt()).abs() < 1e-15);
        assert!((MAGIC_ANGLE.to_degrees() - 54.7356).abs() < 1e-3);
        // 3 cos^2 - 1 = 0 there, exactly.
        assert!((3.0 * MAGIC_ANGLE.cos() * MAGIC_ANGLE.cos() - 1.0).abs() < 1e-15);

        // A muscle wound AT the magic angle does nothing at any pressure or contraction.
        let m = Mckibben::new(0.02, 0.2, MAGIC_ANGLE).expect("valid");
        assert!(m.force(1e6, 0.0).abs() < 1e-9, "no blocked force at the magic angle");
        assert!(m.free_contraction().abs() < 1e-12, "and no free contraction");
        assert!(!m.is_contractile() || m.theta0 < MAGIC_ANGLE);
    }

    #[test]
    fn a_braid_past_the_magic_angle_extends_instead_of_contracting() {
        // Same hardware, opposite sign, decided by the winding. The signed free contraction is what carries it,
        // which is why it is not clamped.
        let contractile = Mckibben::new(0.02, 0.2, 20f64.to_radians()).expect("valid");
        let extensor = Mckibben::new(0.02, 0.2, 70f64.to_radians()).expect("valid");
        assert!(contractile.is_contractile());
        assert!(!extensor.is_contractile());

        assert!(contractile.free_contraction() > 0.0, "a 20 degree braid contracts");
        assert!(extensor.free_contraction() < 0.0, "a 70 degree braid extends, got {}", extensor.free_contraction());

        // The blocked forces have opposite signs at the same pressure.
        let p = 300e3;
        assert!(contractile.blocked_force(p) > 0.0);
        assert!(extensor.blocked_force(p) < 0.0, "an extensor pushes, got {}", extensor.blocked_force(p));

        // And the sign flips exactly at the magic angle, checked from both sides.
        let just_under = Mckibben::new(0.02, 0.2, MAGIC_ANGLE - 1e-6).expect("valid");
        let just_over = Mckibben::new(0.02, 0.2, MAGIC_ANGLE + 1e-6).expect("valid");
        assert!(just_under.blocked_force(p) > 0.0);
        assert!(just_over.blocked_force(p) < 0.0);
    }

    #[test]
    fn the_free_contraction_closed_form_matches_a_root_find_on_the_force() {
        // Two routes to the same number: the algebraic solution of F = 0, and bisection on F itself.
        let m = muscle();
        let p = 250e3;
        let closed = m.free_contraction();
        assert!(closed > 0.0 && closed < 1.0);

        let (mut lo, mut hi) = (0.0f64, 0.99f64);
        assert!(m.force(p, lo) > 0.0 && m.force(p, hi) < 0.0, "the root must be bracketed");
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if m.force(p, mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let found = 0.5 * (lo + hi);
        assert!((found - closed).abs() < 1e-12, "root find {found} vs closed form {closed}");

        // It does not depend on pressure: the zero crossing is geometric.
        for pp in [50e3, 100e3, 600e3] {
            assert!(m.force(pp, closed).abs() < 1e-9, "the zero is pressure-independent, failed at {pp}");
        }
        // The ideal figure for a 20 degree braid is about 38%, which is optimistic against measured devices.
        assert!((closed - 0.385).abs() < 0.005, "expected about 0.385, got {closed:.4}");
    }

    #[test]
    fn force_is_proportional_to_pressure_and_to_the_square_of_diameter() {
        // Both scalings exactly, since either being wrong is a silent factor in every force this produces.
        let m = muscle();
        let base = m.force(100e3, 0.1);
        for k in [0.5f64, 2.0, 7.3] {
            assert!(
                (m.force(k * 100e3, 0.1) / base - k).abs() < 1e-12,
                "force must be linear in pressure, failed at {k}x"
            );
        }
        for k in [0.5f64, 2.0, 3.0] {
            let bigger = Mckibben::new(k * m.d0, m.l0, m.theta0).expect("valid");
            let ratio = bigger.force(100e3, 0.1) / base;
            assert!((ratio - k * k).abs() < 1e-9, "force must go as D0^2: {k}x diameter gave {ratio}x");
        }
        // Length must NOT affect force, only stroke: the formula has no L0 in it.
        let longer = Mckibben::new(m.d0, 3.0 * m.l0, m.theta0).expect("valid");
        assert!((longer.force(100e3, 0.1) - base).abs() < 1e-9, "force must not depend on length");
        // But the stiffness does, because stiffness is per unit absolute displacement.
        assert!(longer.stiffness(100e3, 0.1) < m.stiffness(100e3, 0.1), "a longer muscle is softer");
    }

    #[test]
    fn force_falls_monotonically_with_contraction_so_it_is_a_spring_not_a_source() {
        // The property that shapes how these are controlled.
        let m = muscle();
        let p = 300e3;
        let mut prev = m.blocked_force(p);
        assert!(prev > 0.0);
        for k in 1..=1000 {
            let eps = m.free_contraction() * k as f64 / 1000.0;
            let f = m.force(p, eps);
            assert!(f <= prev + 1e-12, "force must not rise with contraction, at eps={eps}");
            prev = f;
        }
        assert!(prev.abs() < 1e-6, "and reaches zero at the free contraction, got {prev}");

        // The gradient is negative throughout and matches a numerical derivative.
        let h = 1e-7;
        for &eps in &[0.0f64, 0.1, 0.2, 0.3] {
            let numeric = (m.force(p, eps + h) - m.force(p, eps - h)) / (2.0 * h);
            assert!(m.force_gradient(p, eps) < 0.0, "the gradient must be negative at eps={eps}");
            assert!(
                (m.force_gradient(p, eps) - numeric).abs() < 1e-4 * numeric.abs(),
                "eps={eps}: analytic {} vs numeric {numeric}",
                m.force_gradient(p, eps)
            );
        }
        // Stiffness is positive, so the muscle resists being pulled off its working point.
        assert!(m.stiffness(p, 0.1) > 0.0);
    }

    #[test]
    fn the_work_integral_matches_quadrature_of_the_force() {
        // The closed-form integral is exact for a quadratic force, so this catches a slip in the
        // antiderivative rather than a discretisation error.
        let m = muscle();
        let p = 300e3;
        let (a, b) = (0.0, 0.3);
        let n = 200_000;
        let mut quad = 0.0;
        for k in 0..n {
            let e0 = a + (b - a) * k as f64 / n as f64;
            let e1 = a + (b - a) * (k + 1) as f64 / n as f64;
            // dx = L0 de, and the force acts along the contraction.
            quad += 0.5 * (m.force(p, e0) + m.force(p, e1)) * m.l0 * (e1 - e0);
        }
        let closed = m.work(p, a, b);
        assert!(
            (closed - quad).abs() < 1e-6 * quad.abs(),
            "closed-form work {closed:.6} vs quadrature {quad:.6}"
        );
        assert!(closed > 0.0, "contracting under pressure must do positive work");
        // Zero-width interval does no work, and reversing the interval negates it.
        assert_eq!(m.work(p, 0.2, 0.2), 0.0);
        assert!((m.work(p, b, a) + closed).abs() < 1e-9 * closed);
    }

    #[test]
    fn an_antagonistic_pair_separates_torque_from_stiffness() {
        // Two pressures, two independent controls: the reason for the arrangement. Checked by varying the pair
        // pressures in a way that holds one channel fixed and moves the other.
        let m = muscle();
        let r = 0.03;
        let mid = 0.5 * m.free_contraction();

        // Equal pressures at the midpoint give zero torque by symmetry.
        assert!(m.antagonistic_torque(r, 300e3, 300e3, mid).abs() < 1e-9, "a balanced pair holds still");

        // Raising one pressure produces torque in that direction, and lowering it reverses.
        assert!(m.antagonistic_torque(r, 400e3, 300e3, mid) > 0.0);
        assert!(m.antagonistic_torque(r, 300e3, 400e3, mid) < 0.0);

        // Adding the SAME amount to both raises stiffness while leaving torque unchanged: the decomposition.
        let t_low = m.antagonistic_torque(r, 300e3, 300e3, mid);
        let t_high = m.antagonistic_torque(r, 600e3, 600e3, mid);
        assert!((t_high - t_low).abs() < 1e-9, "co-contraction must not create torque");
        let k_low = m.antagonistic_stiffness(r, 300e3, 300e3, mid);
        let k_high = m.antagonistic_stiffness(r, 600e3, 600e3, mid);
        assert!(k_high > k_low, "co-contraction must raise stiffness: {k_low:.3} -> {k_high:.3}");
        assert!((k_high / k_low - 2.0).abs() < 1e-9, "and linearly in pressure, ratio {}", k_high / k_low);

        // Both channels are positive-definite in the useful range, so a controller can invert the map.
        assert!(k_low > 0.0);
    }

    #[test]
    fn malformed_geometry_is_rejected_rather_than_producing_infinities() {
        // theta0 = 0 makes sin^2 theta0 = 0 and the force undefined; pi/2 leaves no length to change.
        assert!(Mckibben::new(0.02, 0.2, 0.0).is_none(), "an axial braid must be rejected");
        assert!(Mckibben::new(0.02, 0.2, 0.5 * PI).is_none(), "a circumferential braid must be rejected");
        assert!(Mckibben::new(0.0, 0.2, 0.35).is_none());
        assert!(Mckibben::new(0.02, 0.0, 0.35).is_none());
        assert!(Mckibben::new(f64::NAN, 0.2, 0.35).is_none());
        // A contraction the geometry cannot reach is reported, not extrapolated.
        let m = muscle();
        assert!(m.angle_at(2.0).is_none(), "eps > 1 is unreachable");
        assert!(m.volume(2.0).is_none());
        // And eps = 1 is exactly reachable (zero length, theta = 90 degrees).
        let theta = m.angle_at(1.0).expect("eps = 1 is the geometric limit");
        assert!((theta - 0.5 * PI).abs() < 1e-12);
    }

    #[test]
    fn the_volume_peaks_at_the_magic_angle_which_is_why_the_force_vanishes_there() {
        // The two facts are the same fact: F = P(-dV/dL), so F = 0 exactly where V is stationary in L. Checking
        // that the volume maximum coincides with the force zero ties the geometry to the mechanics.
        let m = muscle();
        let eps_free = m.free_contraction();
        let v_at_free = m.volume(eps_free).expect("reachable");
        for k in 0..=2000 {
            let eps = eps_free * (k as f64 / 2000.0) * 1.5;
            if let Some(v) = m.volume(eps) {
                assert!(
                    v <= v_at_free * (1.0 + 1e-9),
                    "volume at eps={eps} exceeds the value at free contraction"
                );
            }
        }
        // And the braid angle there IS the magic angle.
        let theta = m.angle_at(eps_free).expect("reachable");
        assert!(
            (theta - MAGIC_ANGLE).abs() < 1e-9,
            "the free-contraction angle should be the magic angle: {theta} vs {MAGIC_ANGLE}"
        );
    }
}
