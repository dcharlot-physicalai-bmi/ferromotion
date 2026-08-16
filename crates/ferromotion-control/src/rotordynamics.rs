//! **Rotordynamics** — what a fast-spinning rotor does to the machine holding it.
//!
//! Every joint in the actuator layer so far has been treated as a torque source. But the thing producing that
//! torque is a rotor turning at thousands of rpm, and nobody simulates it as a joint: its spin is orders of
//! magnitude faster than any body rate, so it is factored out of the multibody model entirely. That
//! factoring-out loses two effects, and both are large enough to matter.
//!
//! * **A spinning rotor resists being turned, at right angles.** Rotate the housing of a running motor and the
//!   rotor pushes back on an axis perpendicular to both its spin and your rotation. On a humanoid wrist or a
//!   drone arm that is a real disturbance torque with no term in the multibody equations, because the spin
//!   degree of freedom was removed. [`gyroscopic_moment`] is that torque.
//! * **There are speeds a rotor cannot be run at.** Residual mass eccentricity — microns, unavoidable — drives
//!   the shaft at exactly its own natural frequency somewhere in the operating range, and the response there is
//!   set entirely by damping. [`Rotor`] is the model, and the useful part is *which way out*: see below.
//!
//! # The two results worth knowing before designing a drive
//!
//! **Running above the critical speed is quieter than running near it.** The whirl amplitude is
//! `e·r²/√((1−r²)² + (2ζr)²)` with `r = Ω/ω_n`, which peaks near `r = 1` at `e/(2ζ)` and then **falls back to
//! `e` exactly** as `r → ∞`. Above resonance the rotor spins about its *mass centre* rather than its geometric
//! centre, so the bearings see the eccentricity and nothing more. The design decision is therefore not "stay
//! below resonance" but "pick a side and pass through quickly".
//!
//! **A disc-shaped rotor has no conical critical speed at all.** Gyroscopic stiffening raises the forward whirl
//! branch with speed, and if the polar-to-diametral inertia ratio `γ = I_p/I_d` exceeds 1 that branch outruns
//! the synchronous line and never crosses it: `Ω_cr = ω_n/√(1−γ)` has no real solution.
//! [`conical_critical_speed`](Rotor::conical_critical_speed) returns `None` there, which is a physical
//! statement and not a failure. For a thin disc `γ = 2` exactly, so the common case *is* the exempt case —
//! while a long slender rotor (`γ = 6r²/L² ≪ 1`) has one and must be designed around it.
//!
//! # What the tests pin
//!
//! The gyroscopic moment is checked to be **exactly workless** — `M · ω_precession = 0` to machine precision,
//! for randomised spin and precession axes — because that is the invariant separating a genuine gyroscopic term
//! from one that pumps energy, and an implementation with the cross product's operands swapped satisfies the
//! magnitude check while getting the sign wrong. The whirl frequencies are checked by substituting them back
//! into the characteristic polynomial rather than against an algebraic form retyped into the assertion. The
//! closed-form response is checked against a **time-domain integration** in both amplitude and phase, which is
//! an independent path.

use nalgebra::Vector3;

/// The gyroscopic moment on a rotor whose spin angular momentum is `h_spin` when its axis is precessed at
/// `omega_precession`: `M = ω × H`.
///
/// Both arguments are in the same frame, and the result is the moment that must be **applied to the rotor** to
/// sustain that precession. The reaction on the structure is its negation, which is what
/// [`Rotor::reaction_on_structure`] returns and what a robot arm actually feels.
///
/// The order of the cross product is load-bearing and is not recoverable from a magnitude check: swapping the
/// operands negates the result, leaving `|M|` correct and the direction reversed, so the disturbance torque
/// would be modelled pushing the wrong way. The tests fix it by the workless condition and by an explicit
/// right-hand-rule case.
pub fn gyroscopic_moment(h_spin: &Vector3<f64>, omega_precession: &Vector3<f64>) -> Vector3<f64> {
    omega_precession.cross(h_spin)
}

/// A spinning rotor on a flexible support: the **Jeffcott** model, plus its gyroscopic properties.
///
/// One lumped mass on an isotropic spring with viscous damping, whirling because its centre of mass sits
/// `eccentricity` away from its geometric centre. It is the simplest model that produces a critical speed, and
/// it gets the amplitude and phase right for a single mode, which is what a drive designer needs.
#[derive(Clone, Copy, Debug)]
pub struct Rotor {
    /// Rotor mass (kg).
    pub mass: f64,
    /// Support stiffness (N/m), isotropic.
    pub stiffness: f64,
    /// Support viscous damping (N·s/m).
    pub damping: f64,
    /// Mass-centre offset from the geometric centre (m). Microns, in practice.
    pub eccentricity: f64,
    /// Polar moment of inertia, about the spin axis (kg·m²).
    pub polar_inertia: f64,
    /// Diametral moment of inertia, about a transverse axis through the centre (kg·m²).
    pub diametral_inertia: f64,
}

/// A Jeffcott rotor's steady-state whirl at one speed.
#[derive(Clone, Copy, Debug)]
pub struct WhirlResponse {
    /// Whirl radius of the geometric centre (m).
    pub amplitude: f64,
    /// Phase lag of the displacement behind the eccentricity vector (rad), in `[0, π]`.
    pub phase: f64,
    /// Peak dynamic force transmitted to the support (N).
    pub bearing_force: f64,
}

impl Rotor {
    /// A **thin disc** of `mass` and `radius` on a support of the given stiffness and damping.
    ///
    /// Its inertias are `I_p = ½ m r²` and `I_d = ¼ m r²`, so `γ = 2` and it is one of the rotors with **no**
    /// conical critical speed.
    pub fn thin_disc(mass: f64, radius: f64, stiffness: f64, damping: f64, eccentricity: f64) -> Rotor {
        Rotor {
            mass,
            stiffness,
            damping,
            eccentricity,
            polar_inertia: 0.5 * mass * radius * radius,
            diametral_inertia: 0.25 * mass * radius * radius,
        }
    }

    /// A **slender rotor** of `mass`, `radius` and `length` — a motor armature. `I_p = ½ m r²`,
    /// `I_d = m(3r² + L²)/12`, so `γ < 1` whenever `L > √3·r` and it *does* have a conical critical speed.
    pub fn slender(
        mass: f64,
        radius: f64,
        length: f64,
        stiffness: f64,
        damping: f64,
        eccentricity: f64,
    ) -> Rotor {
        Rotor {
            mass,
            stiffness,
            damping,
            eccentricity,
            polar_inertia: 0.5 * mass * radius * radius,
            diametral_inertia: mass * (3.0 * radius * radius + length * length) / 12.0,
        }
    }

    /// Undamped natural frequency of the support, `√(k/m)` (rad/s). The **critical speed** in translation.
    pub fn natural_frequency(&self) -> f64 {
        (self.stiffness / self.mass).sqrt()
    }

    /// Viscous damping ratio, `c / (2√(km))`.
    pub fn damping_ratio(&self) -> f64 {
        self.damping / (2.0 * (self.stiffness * self.mass).sqrt())
    }

    /// Polar-to-diametral inertia ratio `γ = I_p/I_d`. Above `1` the conical mode has no critical speed.
    pub fn inertia_ratio(&self) -> f64 {
        self.polar_inertia / self.diametral_inertia
    }

    /// Spin angular momentum vector for spin rate `spin` (rad/s) about unit axis `axis`.
    pub fn spin_momentum(&self, spin: f64, axis: &Vector3<f64>) -> Vector3<f64> {
        axis.normalize() * (self.polar_inertia * spin)
    }

    /// The gyroscopic moment this rotor imposes **on its housing** when spinning at `spin` about `axis` while
    /// the housing rotates at `omega_precession`.
    ///
    /// This is the disturbance a robot arm feels from its own motor, and it is absent from a multibody model
    /// that factored the rotor spin out. Note it is the *negation* of [`gyroscopic_moment`]: that function
    /// returns what must be applied to the rotor, this returns the reaction.
    pub fn reaction_on_structure(
        &self,
        spin: f64,
        axis: &Vector3<f64>,
        omega_precession: &Vector3<f64>,
    ) -> Vector3<f64> {
        -gyroscopic_moment(&self.spin_momentum(spin, axis), omega_precession)
    }

    /// Steady-state whirl at spin speed `spin` (rad/s).
    ///
    /// ```text
    ///   amplitude = e r² / √((1 − r²)² + (2ζr)²)
    ///   phase     = atan2(2ζr, 1 − r²)
    /// ```
    ///
    /// with `r = Ω/ω_n`. At `r = 1` the amplitude is exactly `e/(2ζ)` and the phase exactly `π/2`; as
    /// `r → ∞` the amplitude falls to `e` and the phase to `π`, the rotor then turning about its mass centre.
    pub fn whirl(&self, spin: f64) -> WhirlResponse {
        let wn = self.natural_frequency();
        let z = self.damping_ratio();
        let r = spin / wn;
        let r2 = r * r;
        let denom = ((1.0 - r2) * (1.0 - r2) + (2.0 * z * r) * (2.0 * z * r)).sqrt();
        let amplitude = if denom == 0.0 { f64::INFINITY } else { self.eccentricity * r2 / denom };
        let phase = (2.0 * z * r).atan2(1.0 - r2);
        // The support sees the spring and damper acting on the whirl orbit.
        let k_term = self.stiffness * amplitude;
        let c_term = self.damping * spin * amplitude;
        WhirlResponse {
            amplitude,
            phase,
            bearing_force: (k_term * k_term + c_term * c_term).sqrt(),
        }
    }

    /// The speed ratio `r = Ω/ω_n` at which the whirl amplitude peaks: `1/√(1 − 2ζ²)`.
    ///
    /// Note this is **above** `1`, not at it — the peak of the forced response is not the natural frequency, and
    /// the difference grows with damping. Returns `None` for `ζ ≥ 1/√2 ≈ 0.7071`, where there is no peak at all
    /// and the amplitude rises monotonically to `e`. That is a genuine qualitative change rather than a missing
    /// number, so it is not reported as a large one.
    pub fn peak_speed_ratio(&self) -> Option<f64> {
        let z = self.damping_ratio();
        let d = 1.0 - 2.0 * z * z;
        if d <= 0.0 {
            None
        } else {
            Some(1.0 / d.sqrt())
        }
    }

    /// The two **whirl frequencies** at spin speed `spin`: `(forward, backward)`, both positive (rad/s).
    ///
    /// Gyroscopic coupling splits the conical mode into a forward branch that **stiffens** with speed and a
    /// backward branch that softens. They satisfy
    ///
    /// ```text
    ///   ω² ∓ γ Ω ω − ω_n² = 0,     γ = I_p/I_d
    /// ```
    ///
    /// with the upper sign for forward whirl. At `Ω = 0` the two are degenerate at `ω_n`; the split is what a
    /// Campbell diagram plots. The tests verify the returned values by substituting them back into this
    /// polynomial, so the assertion does not depend on the closed form being retyped correctly.
    pub fn whirl_frequencies(&self, spin: f64) -> (f64, f64) {
        let wn = self.natural_frequency();
        let g = self.inertia_ratio() * spin;
        // ω = (±γΩ + √((γΩ)² + 4ω_n²)) / 2 — the positive root of each sign choice.
        let disc = (g * g + 4.0 * wn * wn).sqrt();
        (0.5 * (g + disc), 0.5 * (-g + disc))
    }

    /// Residual of the whirl characteristic polynomial, for verifying a frequency belongs to a branch.
    ///
    /// `forward = true` uses the `−γΩω` sign. Exposed rather than kept in the tests because it is the honest way
    /// for a caller to check a frequency it obtained elsewhere.
    pub fn whirl_residual(&self, omega: f64, spin: f64, forward: bool) -> f64 {
        let wn = self.natural_frequency();
        let g = self.inertia_ratio() * spin;
        let sign = if forward { -1.0 } else { 1.0 };
        omega * omega + sign * g * omega - wn * wn
    }

    /// The **conical critical speed**: where synchronous excitation (`ω = Ω`) meets the forward whirl branch.
    ///
    /// `Ω_cr = ω_n/√(1 − γ)`, so it exists only for `γ < 1`. Returns `None` for `γ ≥ 1`, where gyroscopic
    /// stiffening raises the forward branch faster than the synchronous line and the two never meet — a
    /// disc-shaped rotor (`γ = 2`) has **no conical critical speed at any speed whatever**.
    ///
    /// This is not the same as the translational critical speed, which is `ω_n` regardless of `γ` because
    /// translation involves no tilting and therefore no gyroscopic term.
    pub fn conical_critical_speed(&self) -> Option<f64> {
        let g = self.inertia_ratio();
        if g >= 1.0 {
            None
        } else {
            Some(self.natural_frequency() / (1.0 - g).sqrt())
        }
    }

    /// Explicit-Euler stability bound for [`step`](Rotor::step), `2/ω_n`. A useful step is a small fraction.
    pub fn max_stable_dt(&self) -> f64 {
        2.0 / self.natural_frequency()
    }

    /// One time-domain step of the planar whirl, for the independent check on [`whirl`](Rotor::whirl).
    ///
    /// State is the geometric centre's position and velocity in the plane. The forcing is the rotating
    /// unbalance `m e Ω²` at angle `Ω t`, which is the physical origin of the closed form.
    pub fn step(
        &self,
        dt: f64,
        spin: f64,
        t: f64,
        pos: &mut Vector3<f64>,
        vel: &mut Vector3<f64>,
    ) {
        let f = self.mass * self.eccentricity * spin * spin;
        let force = Vector3::new(f * (spin * t).cos(), f * (spin * t).sin(), 0.0);
        let accel = (force - self.stiffness * *pos - self.damping * *vel) / self.mass;
        // Semi-implicit: velocity first, then position from the new velocity.
        *vel += accel * dt;
        *pos += *vel * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn disc() -> Rotor {
        // 0.5 kg, 40 mm radius, on a support giving ~200 rad/s natural frequency, 5% damping, 10 um offset.
        let (m, k) = (0.5f64, 20_000.0f64);
        let zeta = 0.05;
        let c = zeta * 2.0 * (k * m).sqrt();
        Rotor::thin_disc(m, 0.04, k, c, 10e-6)
    }

    #[test]
    fn the_gyroscopic_moment_is_exactly_workless() {
        // THE invariant. A gyroscopic term neither adds nor removes energy, so its moment is orthogonal to the
        // precession it accompanies. An implementation with the cross product's operands swapped passes any
        // magnitude check and fails this one only by sign — so this is checked alongside an explicit
        // right-hand-rule case below rather than alone.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        for _ in 0..500 {
            let h = Vector3::new(next(), next(), next()) * 10.0;
            let w = Vector3::new(next(), next(), next()) * 3.0;
            let m = gyroscopic_moment(&h, &w);
            let scale = m.norm().max(1e-12);
            assert!(
                m.dot(&w).abs() < 1e-12 * scale * w.norm().max(1.0),
                "the moment must do no work on the precession: {}",
                m.dot(&w)
            );
            // It is also orthogonal to the spin momentum, so it changes H's direction and not its magnitude.
            assert!(m.dot(&h).abs() < 1e-12 * scale * h.norm().max(1.0), "and must not change |H|");
        }
    }

    #[test]
    fn the_gyroscopic_moment_has_the_right_hand_sense() {
        // The sign, fixed by a case anyone can check by hand. Spin about +z, precess about +x: the moment on
        // the rotor is x cross z = -y.
        let h = Vector3::new(0.0, 0.0, 5.0);
        let w = Vector3::new(2.0, 0.0, 0.0);
        let m = gyroscopic_moment(&h, &w);
        assert!((m - Vector3::new(0.0, -10.0, 0.0)).norm() < 1e-12, "expected -10 y, got {m:?}");

        // Swapping the operands must give the opposite, which is exactly the error this pins.
        let swapped = gyroscopic_moment(&w, &h);
        assert!((swapped + m).norm() < 1e-12, "the cross product must be antisymmetric");

        // And the reaction on the structure is the negation of the moment on the rotor.
        let r = disc();
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let spin = 5.0 / r.polar_inertia; // gives |H| = 5
        let reaction = r.reaction_on_structure(spin, &axis, &w);
        assert!((reaction + m).norm() < 1e-9, "reaction {reaction:?} should be -({m:?})");
    }

    #[test]
    fn a_gyroscopic_moment_vanishes_when_the_precession_is_along_the_spin() {
        // Spinning up faster is not a precession: no moment. A model that used |w||H| would report a large one.
        let h = Vector3::new(0.0, 0.0, 7.0);
        for s in [-5.0, 0.0, 3.0, 100.0] {
            let w = Vector3::new(0.0, 0.0, s);
            assert!(gyroscopic_moment(&h, &w).norm() < 1e-12, "collinear precession must give no moment");
        }
        // And it is maximal when perpendicular, scaling as sin of the angle between.
        for deg in [0.0f64, 30.0, 45.0, 60.0, 90.0] {
            let a = deg.to_radians();
            let w = Vector3::new(a.sin(), 0.0, a.cos()) * 4.0;
            let got = gyroscopic_moment(&h, &w).norm();
            let want = 4.0 * 7.0 * a.sin();
            assert!((got - want).abs() < 1e-12, "at {deg} deg: {got} vs {want}");
        }
    }

    #[test]
    fn the_whirl_amplitude_hits_its_closed_form_landmarks() {
        let r = disc();
        let e = r.eccentricity;
        let z = r.damping_ratio();
        let wn = r.natural_frequency();
        assert!((z - 0.05).abs() < 1e-12, "the fixture should be 5% damped, got {z}");

        // At rest there is no whirl at all: the r^2 numerator, not a constant.
        assert_eq!(r.whirl(0.0).amplitude, 0.0);

        // At r = 1 the amplitude is exactly e/(2 zeta) — the quality factor, and 10x the eccentricity here.
        let at_res = r.whirl(wn).amplitude;
        assert!((at_res - e / (2.0 * z)).abs() < 1e-18, "at resonance: {at_res} vs {}", e / (2.0 * z));
        assert!((at_res / e - 10.0).abs() < 1e-9, "5% damping gives a 10x magnification");
        // And the phase is exactly 90 degrees there, which is the definition of resonance.
        assert!((r.whirl(wn).phase - PI / 2.0).abs() < 1e-15);

        // As r grows the amplitude falls BACK to e: the rotor turns about its mass centre and the bearings see
        // only the eccentricity. This is the result that makes running above resonance the right answer.
        let far = r.whirl(1000.0 * wn).amplitude;
        assert!((far / e - 1.0).abs() < 1e-5, "far above resonance the amplitude should approach e, got {}", far / e);
        assert!(far < at_res / 9.0, "and be far below the resonant peak");
        // Phase approaches 180 degrees: displacement opposite the eccentricity, which is what "about the mass
        // centre" means geometrically.
        assert!((r.whirl(1000.0 * wn).phase - PI).abs() < 1e-3);
    }

    #[test]
    fn the_amplitude_peak_sits_above_the_natural_frequency_and_disappears_when_overdamped() {
        // A precise, non-obvious closed form: the forced peak is at 1/sqrt(1 - 2 zeta^2), not at 1.
        for zeta in [0.02f64, 0.05, 0.2, 0.5, 0.7] {
            let (m, k) = (0.5f64, 20_000.0f64);
            let r = Rotor::thin_disc(m, 0.04, k, zeta * 2.0 * (k * m).sqrt(), 10e-6);
            let ratio = r.peak_speed_ratio().expect("a peak exists below zeta = 1/sqrt(2)");
            assert!(ratio > 1.0, "the peak must be above the natural frequency, got {ratio}");
            // Verify it IS the maximum by scanning, rather than trusting the formula.
            let wn = r.natural_frequency();
            let best = r.whirl(ratio * wn).amplitude;
            for k in 0..=4000 {
                let rr = 0.2 + 4.8 * k as f64 / 4000.0;
                assert!(
                    r.whirl(rr * wn).amplitude <= best * (1.0 + 1e-9),
                    "zeta={zeta}: r={rr} beat the predicted peak at {ratio}"
                );
            }
        }

        // At and above zeta = 1/sqrt(2) there is no peak: the response rises monotonically to e. A sharp
        // qualitative change, reported as None rather than as a very large or a clamped number.
        // 1/sqrt(2) is the exact boundary, so it is written as the constant rather than as a literal that
        // silently differs from it in the last bits.
        for zeta in [std::f64::consts::FRAC_1_SQRT_2, 0.8, 1.0, 2.0] {
            let (m, k) = (0.5f64, 20_000.0f64);
            let r = Rotor::thin_disc(m, 0.04, k, zeta * 2.0 * (k * m).sqrt(), 10e-6);
            assert!(r.peak_speed_ratio().is_none(), "zeta={zeta} should have no peak");
            // And confirm monotonicity, which is the physical content of that None.
            let wn = r.natural_frequency();
            let mut prev = 0.0;
            for k in 1..=3000 {
                let a = r.whirl(0.01 * k as f64 * wn).amplitude;
                assert!(a >= prev - 1e-18, "zeta={zeta}: amplitude must not fall, at r={}", 0.01 * k as f64);
                prev = a;
            }
            assert!(prev <= r.eccentricity * (1.0 + 1e-6), "and must not exceed e");
        }
    }

    #[test]
    fn the_time_domain_response_reproduces_the_closed_form_in_amplitude_and_phase() {
        // Two independent paths. The closed form could be self-consistently wrong; an integration of the
        // actual equation of motion could not be wrong in the same way.
        let r = disc();
        let wn = r.natural_frequency();
        for &ratio in &[0.5f64, 1.0, 1.7, 3.0] {
            let spin = ratio * wn;
            let dt = 0.002 * r.max_stable_dt().min(2.0 * PI / spin);
            let mut pos = Vector3::zeros();
            let mut vel = Vector3::zeros();
            // Run long enough for the transient to die: 40 damped time constants.
            let settle = 40.0 / (r.damping_ratio() * wn);
            let n_settle = (settle / dt) as usize;
            let mut t = 0.0;
            for _ in 0..n_settle {
                r.step(dt, spin, t, &mut pos, &mut vel);
                t += dt;
            }
            // Measure the orbit radius and the phase relative to the forcing over one full revolution.
            let period = 2.0 * PI / spin;
            let n_per = (period / dt) as usize;
            let mut max_rad = 0.0f64;
            let mut min_rad = f64::INFINITY;
            let (mut c_s, mut c_c) = (0.0f64, 0.0f64);
            for _ in 0..n_per {
                r.step(dt, spin, t, &mut pos, &mut vel);
                t += dt;
                let rad = (pos.x * pos.x + pos.y * pos.y).sqrt();
                max_rad = max_rad.max(rad);
                min_rad = min_rad.min(rad);
                // Correlate x against the forcing's cos and sin to recover the phase lag.
                c_c += pos.x * (spin * t).cos();
                c_s += pos.x * (spin * t).sin();
            }
            let want = r.whirl(spin);
            // The orbit is circular for an isotropic support, so max and min radius agree.
            assert!(
                (max_rad - min_rad).abs() < 1e-3 * max_rad,
                "ratio={ratio}: the orbit should be circular, {min_rad:.3e} to {max_rad:.3e}"
            );
            assert!(
                (max_rad - want.amplitude).abs() < 5e-3 * want.amplitude,
                "ratio={ratio}: measured radius {max_rad:.6e} vs closed form {:.6e}",
                want.amplitude
            );
            // Phase lag: x = A cos(Omega t - phi), so the correlations give phi.
            let measured_phase = c_s.atan2(c_c);
            assert!(
                (measured_phase - want.phase).abs() < 5e-3,
                "ratio={ratio}: measured phase {measured_phase:.4} vs closed form {:.4}",
                want.phase
            );
        }
    }

    #[test]
    fn the_whirl_frequencies_satisfy_their_own_characteristic_polynomial() {
        // Verified by substitution, not by retyping the algebra into the assertion.
        let r = disc();
        let wn = r.natural_frequency();
        for &spin in &[0.0f64, 50.0, 200.0, 1000.0, 5000.0] {
            let (fwd, bwd) = r.whirl_frequencies(spin);
            assert!(fwd > 0.0 && bwd > 0.0, "both branches are positive frequencies");
            assert!(
                r.whirl_residual(fwd, spin, true).abs() < 1e-9 * (wn * wn),
                "forward residual at {spin}: {}",
                r.whirl_residual(fwd, spin, true)
            );
            assert!(
                r.whirl_residual(bwd, spin, false).abs() < 1e-9 * (wn * wn),
                "backward residual at {spin}: {}",
                r.whirl_residual(bwd, spin, false)
            );
        }

        // At zero spin the two are degenerate at the natural frequency: no gyroscopic term, no split.
        let (f0, b0) = r.whirl_frequencies(0.0);
        assert!((f0 - wn).abs() < 1e-12 && (b0 - wn).abs() < 1e-12, "degenerate at rest: {f0}, {b0}");

        // Forward stiffens and backward softens, monotonically. This is the Campbell diagram's shape.
        let mut prev = (f0, b0);
        for k in 1..=500 {
            let spin = 20.0 * k as f64;
            let (f, b) = r.whirl_frequencies(spin);
            assert!(f > prev.0, "forward whirl must rise with speed, at {spin}");
            assert!(b < prev.1, "backward whirl must fall with speed, at {spin}");
            prev = (f, b);
        }
        // Their product is the squared natural frequency exactly, at every speed — the roots of the pair.
        for &spin in &[0.0f64, 137.0, 4321.0] {
            let (f, b) = r.whirl_frequencies(spin);
            assert!((f * b - wn * wn).abs() < 1e-9 * wn * wn, "product should be wn^2 at {spin}");
        }
    }

    #[test]
    fn a_disc_rotor_has_no_conical_critical_speed_and_a_slender_one_does() {
        // The striking result, with its sharp boundary at gamma = 1.
        let d = disc();
        assert!((d.inertia_ratio() - 2.0).abs() < 1e-12, "a thin disc has gamma = 2 exactly");
        assert!(
            d.conical_critical_speed().is_none(),
            "gyroscopic stiffening removes the conical critical speed entirely for gamma >= 1"
        );
        // The operational statement: the forward whirl branch never meets the synchronous line.
        for k in 1..=20_000 {
            let spin = 5.0 * k as f64;
            let (fwd, _) = d.whirl_frequencies(spin);
            assert!(fwd > spin, "forward whirl must stay above the synchronous line, at {spin}");
        }

        // A slender rotor: 0.5 kg, 8 mm radius, 120 mm long.
        let s = Rotor::slender(0.5, 0.008, 0.12, 20_000.0, 20.0, 10e-6);
        let g = s.inertia_ratio();
        assert!(g < 1.0, "a slender rotor should have gamma < 1, got {g}");
        let cr = s.conical_critical_speed().expect("a slender rotor has one");
        assert!(cr > s.natural_frequency(), "and it lies above the translational critical speed");
        // Verify by intersection rather than by the formula: at the critical speed, forward whirl equals spin.
        let (fwd, _) = s.whirl_frequencies(cr);
        assert!(
            (fwd - cr).abs() < 1e-6 * cr,
            "at the conical critical speed the forward branch meets the synchronous line: {fwd} vs {cr}"
        );
        // And the crossing is a genuine crossing: below it forward whirl exceeds spin, above it does not.
        let (below, _) = s.whirl_frequencies(0.9 * cr);
        let (above, _) = s.whirl_frequencies(1.1 * cr);
        assert!(below > 0.9 * cr, "below the crossing, whirl leads");
        assert!(above < 1.1 * cr, "above it, spin leads");

        // The boundary is exactly gamma = 1, checked from both sides.
        let mut near = s;
        near.polar_inertia = 0.999 * near.diametral_inertia;
        assert!(near.conical_critical_speed().is_some());
        near.polar_inertia = 1.0 * near.diametral_inertia;
        assert!(near.conical_critical_speed().is_none(), "gamma = 1 exactly has no solution");
    }

    #[test]
    fn the_translational_critical_speed_is_unaffected_by_the_inertia_ratio() {
        // Translation involves no tilting, so there is no gyroscopic term and gamma cannot enter. A model that
        // applied the conical correction everywhere would fail this.
        let base = disc();
        let wn = base.natural_frequency();
        for factor in [0.1f64, 1.0, 10.0] {
            let mut r = base;
            r.polar_inertia = base.polar_inertia * factor;
            assert!((r.natural_frequency() - wn).abs() < 1e-12, "wn must not depend on I_p");
            // The whirl response is a translational quantity, so it must not move either.
            assert_eq!(r.whirl(wn).amplitude, base.whirl(wn).amplitude);
        }
    }

    #[test]
    fn a_motors_gyroscopic_reaction_is_a_real_disturbance_at_robot_scale() {
        // The engineering consequence, as a number a designer can weigh against a joint's torque budget.
        // A 100 g rotor, 20 mm radius, at 8000 rpm, on a wrist that is being rotated at 3 rad/s.
        let r = Rotor::thin_disc(0.1, 0.02, 20_000.0, 20.0, 10e-6);
        let rpm = 8000.0;
        let spin = rpm * 2.0 * PI / 60.0;
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let precession = Vector3::new(3.0, 0.0, 0.0);
        let m = r.reaction_on_structure(spin, &axis, &precession);

        // I_p = 0.5 * 0.1 * 0.02^2 = 2e-5 kg m^2. H = 2e-5 * 837.8 = 0.01676. |M| = 3 * 0.01676 = 0.0503 N m.
        let expect = r.polar_inertia * spin * 3.0;
        assert!((m.norm() - expect).abs() < 1e-12, "{} vs {expect}", m.norm());
        assert!(
            (m.norm() - 0.0503).abs() < 5e-4,
            "about 50 mN m at this scale, got {:.5}",
            m.norm()
        );
        // Perpendicular to both, so it appears on an axis the joint is not commanding — which is exactly why
        // it reads as a disturbance rather than as a modelling error.
        assert!(m.dot(&precession).abs() < 1e-15);
        assert!(m.dot(&axis).abs() < 1e-15);
        assert!(m.y.abs() > 0.05 && m.x.abs() < 1e-15 && m.z.abs() < 1e-15, "it lands on y: {m:?}");

        // It scales linearly in both spin and precession rate, so doubling either doubles the disturbance.
        let double_spin = r.reaction_on_structure(2.0 * spin, &axis, &precession);
        assert!((double_spin.norm() / m.norm() - 2.0).abs() < 1e-12);
        let double_prec = r.reaction_on_structure(spin, &axis, &(precession * 2.0));
        assert!((double_prec.norm() / m.norm() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_bearing_force_is_bounded_above_resonance_even_though_speed_keeps_rising() {
        // A subtlety worth pinning: the whirl amplitude falls to e above resonance, but the damper force goes
        // as c*Omega*amplitude, so the transmitted force does NOT fall to a constant. Asserting "quieter above
        // resonance" without qualification would be wrong about the bearings.
        let r = disc();
        let wn = r.natural_frequency();
        let at_res = r.whirl(wn).bearing_force;
        let above = r.whirl(3.0 * wn).bearing_force;
        assert!(above < at_res, "well above resonance the bearing force is lower: {above:.4} vs {at_res:.4}");

        // But it grows again with speed, because the damping term is proportional to Omega.
        let far = r.whirl(100.0 * wn).bearing_force;
        let farther = r.whirl(200.0 * wn).bearing_force;
        assert!(farther > far, "the damper's contribution rises with speed: {far:.4} then {farther:.4}");
        // Roughly linearly, once the amplitude has settled at e.
        assert!((farther / far - 2.0).abs() < 0.05, "and about linearly, ratio {:.3}", farther / far);
    }
}
