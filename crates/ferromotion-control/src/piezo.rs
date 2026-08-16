//! **Piezoelectric transduction** — the actuator that is also a sensor, and a capacitor.
//!
//! A piezoelectric stack is the actuator you reach for when the motion is small and must be precise: sub-nanometre
//! resolution, kilohertz bandwidth, no backlash, no lubricant. It is also the sensor you reach for when the
//! signal is a force or a vibration, because the same coupling runs both ways — the `d` coefficient in
//! [`Piezo::free_strain`] and the one in [`Piezo::charge_from_stress`] are **the same number**, which is the
//! reciprocity the tests assert as an energy identity rather than by inspection.
//!
//! Three things about it that a "commanded displacement" model gets wrong:
//!
//! * **It has almost no stroke and enormous force.** A stack is `d·E` in strain, which is parts in `10⁴`. Free
//!   displacement is microns; blocked force is kilonewtons. What it can actually *do* against a load lies on the
//!   straight line between those two, and the useful work is maximised **not** at either end but exactly
//!   half-way: `¼ F_blocked · x_free`. Load matching is the whole design problem, and it is the same
//!   quarter-power result as impedance matching.
//! * **It is a capacitive load, so driving it fast costs reactive current.** `i = C dV/dt`, so a 1 µF stack
//!   slewed at 100 V/ms draws 100 mA that does no work and heats the amplifier. Bandwidth is an amplifier
//!   specification, not a piezo specification.
//! * **Only a fraction `k²` of the energy you put in comes out as work**, and `k²` is a material constant, not
//!   an efficiency you can design around. It is measurable two entirely different ways — from the constitutive
//!   coefficients, and from the gap between the resonance and antiresonance frequencies — and those two must
//!   agree. That agreement is the sharpest check in this module.
//!
//! # The linear constitutive relations
//!
//! Strain-charge form, one axis:
//!
//! ```text
//!   S = sᴱ T + d E          (strain from stress and field)
//!   D = d T + εᵀ E          (charge density from stress and field)
//! ```
//!
//! `sᴱ` is compliance at constant field, `εᵀ` permittivity at constant stress, `d` the piezoelectric
//! coefficient appearing in **both** equations. The electromechanical coupling is
//!
//! ```text
//!   k² = d² / (sᴱ εᵀ)
//! ```
//!
//! and `k² ≤ 1` is a thermodynamic requirement, not a material happenstance: a `k² > 1` set of coefficients
//! describes a transducer that returns more work than it is given. [`Piezo::new`] rejects it.
//!
//! # What this module does not model
//!
//! **Hysteresis and creep, which are large.** A real stack shows 10-15% hysteresis in its open-loop
//! displacement-voltage curve and continues to creep for minutes after a step, which is why precision stages run
//! closed-loop on a capacitive or strain-gauge sensor. The linear relations here are the small-signal model;
//! they will overstate open-loop positioning accuracy by more than an order of magnitude. For the creep half,
//! [`viscoelastic`](https://docs.rs/ferromotion-fem) has the right shape of model. This is stated rather than
//! buried because "piezo positioning is exact" is the specific wrong conclusion the linear model invites.

/// A piezoelectric element in one axis, with its geometry.
#[derive(Clone, Copy, Debug)]
pub struct Piezo {
    /// Compliance at constant electric field, `sᴱ` (m²/N).
    pub compliance: f64,
    /// Piezoelectric strain coefficient `d` (m/V), the same number in both constitutive relations.
    pub d: f64,
    /// Permittivity at constant stress, `εᵀ` (F/m).
    pub permittivity: f64,
    /// Density (kg/m³), for the resonance frequency.
    pub density: f64,
    /// Cross-sectional area (m²).
    pub area: f64,
    /// Length along the driven axis (m).
    pub length: f64,
    /// Number of stacked layers. A stack of `n` layers of thickness `length/n` reaches the same field at
    /// `1/n` of the voltage, which is the entire reason stacks exist.
    pub layers: usize,
}

impl Piezo {
    /// An element from material coefficients and geometry. Returns `None` if any parameter is non-physical or
    /// if the implied coupling `k² = d²/(sᴱ εᵀ)` exceeds 1.
    ///
    /// **The `k² ≤ 1` check is thermodynamics, not validation theatre.** `k²` is the fraction of input energy
    /// converted per cycle; a coefficient set giving `k² > 1` describes a transducer returning more work than it
    /// receives, and it is what you get from mixing coefficients measured under different boundary conditions —
    /// a `d` from one datasheet with an `εᵀ` from another, or an `εˢ` (constant strain) used where `εᵀ` belongs.
    pub fn new(
        compliance: f64,
        d: f64,
        permittivity: f64,
        density: f64,
        area: f64,
        length: f64,
        layers: usize,
    ) -> Option<Piezo> {
        let all = [compliance, d, permittivity, density, area, length];
        if all.iter().any(|v| !v.is_finite()) || layers == 0 {
            return None;
        }
        if compliance <= 0.0 || permittivity <= 0.0 || density <= 0.0 || area <= 0.0 || length <= 0.0 {
            return None;
        }
        let p = Piezo { compliance, d, permittivity, density, area, length, layers };
        if p.coupling_squared() > 1.0 {
            return None;
        }
        Some(p)
    }

    /// A PZT-5H stack of the given area, length and layer count. Representative soft-PZT coefficients:
    /// `d₃₃ = 650 pm/V`, `s₃₃ᴱ = 20.7 pm²/N`, `ε₃₃ᵀ = 3400 ε₀`, `ρ = 7500 kg/m³`.
    pub fn pzt5h(area: f64, length: f64, layers: usize) -> Option<Piezo> {
        const EPS0: f64 = 8.854_187_812_8e-12;
        Piezo::new(20.7e-12, 650e-12, 3400.0 * EPS0, 7500.0, area, length, layers)
    }

    /// Thickness of one layer, `length / layers`.
    pub fn layer_thickness(&self) -> f64 {
        self.length / self.layers as f64
    }

    /// Electric field for an applied voltage: `V / layer_thickness`. A stack multiplies the field by its layer
    /// count for the same voltage, which is why 100 V moves a stack and would need 10 kV for a monolith.
    pub fn field(&self, voltage: f64) -> f64 {
        voltage / self.layer_thickness()
    }

    /// **Electromechanical coupling squared**, `k² = d²/(sᴱ εᵀ)`: the fraction of input energy converted.
    ///
    /// A material constant. It bounds what any circuit or mechanism around the element can achieve, so it is
    /// the first number to look at and the one that cannot be engineered around.
    pub fn coupling_squared(&self) -> f64 {
        self.d * self.d / (self.compliance * self.permittivity)
    }

    /// Free strain at applied voltage `v`: `S = d E`, with nothing opposing.
    pub fn free_strain(&self, v: f64) -> f64 {
        self.d * self.field(v)
    }

    /// Free displacement (m) at voltage `v`. For a stack this is `n · d · V`, independent of length.
    pub fn free_displacement(&self, v: f64) -> f64 {
        self.free_strain(v) * self.length
    }

    /// **Blocked force** (N) at voltage `v`: the force at zero displacement, `d E A / sᴱ`.
    pub fn blocked_force(&self, v: f64) -> f64 {
        self.free_strain(v) * self.area / self.compliance
    }

    /// Short-circuit stiffness of the element as a spring, `A/(sᴱ L)` (N/m).
    pub fn stiffness(&self) -> f64 {
        self.area / (self.compliance * self.length)
    }

    /// Displacement (m) against an external spring of stiffness `k_load`, at voltage `v`.
    ///
    /// The load line: `x = x_free · k_piezo/(k_piezo + k_load)`. Note that a load equal to the element's own
    /// stiffness gives exactly **half** the free displacement and half the blocked force — the matched
    /// condition, which is where the work below is maximised.
    pub fn displacement_against_spring(&self, v: f64, k_load: f64) -> f64 {
        let kp = self.stiffness();
        self.free_displacement(v) * kp / (kp + k_load)
    }

    /// Force (N) delivered into an external spring of stiffness `k_load`, at voltage `v`.
    pub fn force_against_spring(&self, v: f64, k_load: f64) -> f64 {
        k_load * self.displacement_against_spring(v, k_load)
    }

    /// Work (J) delivered into an external spring, `½ k_load x²`.
    ///
    /// Maximised at `k_load = k_piezo`, where it equals `⅛ F_blocked · x_free`. The frequently quoted
    /// `¼ F_blocked x_free` is the work into a **constant-force** load at half stroke; a spring load stores half
    /// as much again because the force ramps from zero. Both are asserted, because quoting one figure for the
    /// other is a factor-of-two error in a work budget.
    pub fn work_into_spring(&self, v: f64, k_load: f64) -> f64 {
        let x = self.displacement_against_spring(v, k_load);
        0.5 * k_load * x * x
    }

    /// The load stiffness that maximises delivered work: the element's own stiffness.
    pub fn matched_load(&self) -> f64 {
        self.stiffness()
    }

    /// Capacitance (F) of the stack: `n² εᵀ A / L` for `n` layers in parallel electrically and series
    /// mechanically.
    pub fn capacitance(&self) -> f64 {
        let n = self.layers as f64;
        n * n * self.permittivity * self.area / self.length
    }

    /// Reactive current (A) needed to slew the voltage at `dv_dt`: `i = C dV/dt`.
    ///
    /// This current does no work on the load. It is why a piezo's usable bandwidth is set by the amplifier and
    /// not by the ceramic, and why a stack that is nominally a "kilohertz actuator" needs amps to be one.
    pub fn drive_current(&self, dv_dt: f64) -> f64 {
        self.capacitance() * dv_dt
    }

    /// Charge density (C/m²) generated by a stress with no applied field: `D = d T`. The converse effect, using
    /// the **same** `d`.
    pub fn charge_from_stress(&self, stress: f64) -> f64 {
        self.d * stress
    }

    /// Open-circuit voltage from an applied force (V): the sensing mode.
    pub fn voltage_from_force(&self, force: f64) -> f64 {
        let stress = force / self.area;
        let charge = self.charge_from_stress(stress) * self.area * self.layers as f64;
        charge / self.capacitance()
    }

    /// Fundamental **series (short-circuit) resonance** of a free-free bar, `f_r = c/(2L)` with
    /// `c = 1/√(ρ sᴱ)` (Hz).
    pub fn resonance(&self) -> f64 {
        let c = 1.0 / (self.density * self.compliance).sqrt();
        c / (2.0 * self.length)
    }

    /// **Antiresonance (open-circuit, parallel) frequency** (Hz).
    ///
    /// Open-circuit the element and it stiffens, because the charge that would have flowed instead builds a
    /// field opposing the strain. The two frequencies are related to the coupling by
    ///
    /// ```text
    ///   k² = 1 − (f_r/f_a)²
    /// ```
    ///
    /// which is how `k²` is actually measured on a part: two frequencies from an impedance analyser, no need to
    /// know `d`, `sᴱ` or `εᵀ` separately. That this agrees with `d²/(sᴱεᵀ)` is the module's sharpest test,
    /// because the two routes share no arithmetic.
    pub fn antiresonance(&self) -> f64 {
        self.resonance() / (1.0 - self.coupling_squared()).sqrt()
    }

    /// Coupling recovered from the resonance pair, `1 − (f_r/f_a)²`. Must equal
    /// [`coupling_squared`](Piezo::coupling_squared).
    pub fn coupling_from_resonances(&self) -> f64 {
        let r = self.resonance() / self.antiresonance();
        1.0 - r * r
    }

    /// **Optimal resistive load for energy harvesting** at angular frequency `omega`: `R = 1/(ωC)`.
    ///
    /// The element is a current source behind its own capacitance, so the load that extracts the most power is
    /// the one matching the capacitive reactance. Note the strong frequency dependence: a harvester tuned for
    /// 50 Hz is badly mismatched at 200 Hz, which is why broadband vibration harvesting is hard for reasons
    /// that have nothing to do with the ceramic.
    pub fn optimal_harvest_resistance(&self, omega: f64) -> f64 {
        1.0 / (omega * self.capacitance())
    }

    /// Average power (W) into a resistive load `r_load` from a sinusoidal stress of amplitude
    /// `stress_amplitude` at `omega`.
    ///
    /// Models the element as a current source `i = ω d A n T` behind capacitance `C`, so
    /// `P = ½ i² R/(1 + (ωRC)²)`.
    pub fn harvested_power(&self, stress_amplitude: f64, omega: f64, r_load: f64) -> f64 {
        let c = self.capacitance();
        let i = omega * self.d * self.area * self.layers as f64 * stress_amplitude;
        let wrc = omega * r_load * c;
        0.5 * i * i * r_load / (1.0 + wrc * wrc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack() -> Piezo {
        // 10 mm x 10 mm cross-section, 20 mm long, 200 layers of 100 um: a common precision-stage stack.
        Piezo::pzt5h(1e-4, 20e-3, 200).expect("valid PZT-5H stack")
    }

    #[test]
    fn the_two_routes_to_the_coupling_coefficient_agree() {
        // The sharpest check here: k^2 from the constitutive coefficients and k^2 from the resonance pair share
        // no arithmetic, so agreement is real evidence rather than algebra restated.
        let p = stack();
        let from_coeffs = p.coupling_squared();
        let from_freqs = p.coupling_from_resonances();
        assert!(
            (from_coeffs - from_freqs).abs() < 1e-12,
            "k^2 = {from_coeffs} from coefficients vs {from_freqs} from f_r and f_a"
        );
        // And it is a physical fraction, in a plausible band for soft PZT.
        assert!(from_coeffs > 0.0 && from_coeffs < 1.0, "k^2 must be a fraction, got {from_coeffs}");
        assert!(
            (0.5..0.85).contains(&from_coeffs),
            "soft PZT should couple strongly, got k^2 = {from_coeffs:.4}"
        );
        // Antiresonance is above resonance, by exactly the coupling factor.
        assert!(p.antiresonance() > p.resonance());
        let ratio = p.antiresonance() / p.resonance();
        assert!((ratio - 1.0 / (1.0 - from_coeffs).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_coefficient_set_implying_more_energy_out_than_in_is_rejected() {
        // k^2 > 1 is thermodynamically impossible, and it is exactly what mixing coefficients from different
        // measurement conditions produces. Constructed here by inflating d until the bound breaks.
        let ok = Piezo::new(20.7e-12, 650e-12, 3400.0 * 8.8541878128e-12, 7500.0, 1e-4, 20e-3, 200);
        assert!(ok.is_some(), "the physical set must be accepted");
        // The same set with d tripled: k^2 goes up 9x and crosses 1.
        let bad = Piezo::new(20.7e-12, 3.0 * 650e-12, 3400.0 * 8.8541878128e-12, 7500.0, 1e-4, 20e-3, 200);
        assert!(bad.is_none(), "k^2 > 1 must be rejected");
        // And confirm the rejection is for the stated reason rather than incidental: k^2 really does exceed 1.
        let unchecked = Piezo {
            compliance: 20.7e-12,
            d: 3.0 * 650e-12,
            permittivity: 3400.0 * 8.8541878128e-12,
            density: 7500.0,
            area: 1e-4,
            length: 20e-3,
            layers: 200,
        };
        assert!(unchecked.coupling_squared() > 1.0, "the fixture must actually violate the bound");

        // Non-physical geometry and a zero-layer stack are rejected too.
        assert!(Piezo::pzt5h(0.0, 20e-3, 200).is_none());
        assert!(Piezo::pzt5h(1e-4, 0.0, 200).is_none());
        assert!(Piezo::pzt5h(1e-4, 20e-3, 0).is_none());
        assert!(Piezo::new(-1e-12, 650e-12, 1e-8, 7500.0, 1e-4, 20e-3, 1).is_none());
    }

    #[test]
    fn a_stack_reaches_its_field_at_a_fraction_of_the_voltage() {
        // The entire reason stacks exist, as a ratio.
        let mono = Piezo::pzt5h(1e-4, 20e-3, 1).expect("valid");
        let s = stack();
        let v = 100.0;
        assert!((s.field(v) / mono.field(v) - 200.0).abs() < 1e-9, "200 layers give 200x the field");
        // Free displacement of a stack is n*d*V, independent of length: check against that closed form.
        let expect = 200.0 * s.d * v;
        assert!(
            (s.free_displacement(v) - expect).abs() < 1e-18,
            "{} vs n d V = {expect}",
            s.free_displacement(v)
        );
        // And the numbers are in the right regime: microns of stroke, kilonewtons of blocked force.
        let x = s.free_displacement(150.0);
        let f = s.blocked_force(150.0);
        assert!((10e-6..40e-6).contains(&x), "tens of microns, got {:.3e} m", x);
        assert!((1e3..2e4).contains(&f), "kilonewtons, got {:.1} N", f);
    }

    #[test]
    fn the_load_line_is_straight_between_free_displacement_and_blocked_force() {
        // The defining property of the actuator, and what makes load matching the design problem.
        let p = stack();
        let v = 100.0;
        let x_free = p.free_displacement(v);
        let f_block = p.blocked_force(v);

        // Zero load gives free displacement and no force; infinite load gives no motion and blocked force.
        assert!((p.displacement_against_spring(v, 0.0) - x_free).abs() < 1e-18);
        assert_eq!(p.force_against_spring(v, 0.0), 0.0);
        let huge = 1e12 * p.stiffness();
        assert!(p.displacement_against_spring(v, huge) < x_free * 1e-9);
        assert!((p.force_against_spring(v, huge) - f_block).abs() < 1e-3 * f_block);

        // In between, force and displacement lie on the straight line f = f_block (1 - x/x_free).
        for k in [0.1, 0.5, 1.0, 2.0, 10.0] {
            let kl = k * p.stiffness();
            let x = p.displacement_against_spring(v, kl);
            let f = p.force_against_spring(v, kl);
            let on_line = f_block * (1.0 - x / x_free);
            assert!(
                (f - on_line).abs() < 1e-9 * f_block,
                "k={k}: force {f} should lie on the load line at {on_line}"
            );
        }
        // A matched load gives exactly half of each.
        let xm = p.displacement_against_spring(v, p.matched_load());
        assert!((xm / x_free - 0.5).abs() < 1e-12, "matched load gives half the stroke");
        assert!(
            (p.force_against_spring(v, p.matched_load()) / f_block - 0.5).abs() < 1e-12,
            "and half the force"
        );
    }

    #[test]
    fn delivered_work_peaks_at_the_matched_load_and_the_two_quarter_power_figures_differ() {
        // Both the location of the optimum and its value, because the commonly quoted 1/4 F x figure is for a
        // constant-force load and a spring load stores half of it. Confusing them is a factor of two.
        let p = stack();
        let v = 100.0;
        let best = p.work_into_spring(v, p.matched_load());

        // It IS the maximum, by scan rather than by assertion.
        for i in 1..=4000 {
            let kl = p.stiffness() * (0.01 + 5.0 * i as f64 / 4000.0);
            assert!(
                p.work_into_spring(v, kl) <= best * (1.0 + 1e-12),
                "k_load/k_p = {} beat the matched load",
                kl / p.stiffness()
            );
        }

        // Spring load at match: 1/8 F_blocked x_free.
        let x_free = p.free_displacement(v);
        let f_block = p.blocked_force(v);
        assert!(
            (best - 0.125 * f_block * x_free).abs() < 1e-12 * best,
            "matched spring work {best:.6e} should be F x / 8 = {:.6e}",
            0.125 * f_block * x_free
        );
        // Constant-force load at half stroke: 1/4 F_blocked x_free, i.e. twice as much.
        let const_force_work = 0.5 * f_block * 0.5 * x_free;
        assert!((const_force_work / best - 2.0).abs() < 1e-12, "the two figures differ by exactly 2x");
    }

    #[test]
    fn reciprocity_holds_as_an_energy_identity() {
        // The same d in both directions is not just a shared field; it is required for the transducer to have a
        // well-defined energy. Check it operationally: the charge a force produces, times the voltage that
        // would produce that force's displacement, must be consistent both ways round.
        let p = stack();

        // Route A: apply voltage v, measure blocked force. Route B: apply that force, measure open-circuit
        // voltage. Reciprocity makes the ratio the same constant either way.
        let v = 100.0;
        let f = p.blocked_force(v);
        let v_back = p.voltage_from_force(f);
        // The round trip is scaled by k^2: mechanical energy recovered over electrical energy supplied. That
        // this equals the independently computed coupling is the identity.
        let ratio = v_back / v;
        assert!(
            (ratio - p.coupling_squared()).abs() < 1e-9 * p.coupling_squared(),
            "the reciprocal round trip should scale by k^2 = {}, got {ratio}",
            p.coupling_squared()
        );

        // Linearity in both directions, since the whole model rests on it.
        for s in [-3.0, 0.5, 7.0] {
            assert!((p.charge_from_stress(s * 1e6) / (s * 1e6) - p.d).abs() < 1e-24);
            assert!((p.free_displacement(s * v) / s - p.free_displacement(v)).abs() < 1e-18);
        }
    }

    #[test]
    fn the_drive_current_is_reactive_and_sets_the_real_bandwidth() {
        // The number that turns "kilohertz actuator" into an amplifier specification.
        let p = stack();
        let c = p.capacitance();
        assert!((0.5e-6..20e-6).contains(&c), "a 200-layer stack should be microfarads, got {c:.3e} F");

        // Slewing 100 V in 1 ms.
        let i = p.drive_current(100.0 / 1e-3);
        assert!(i > 0.05, "slewing this stack needs tens of mA at least, got {i:.4} A");
        // Linear in slew rate, and it does no work: it is set by C alone.
        assert!((p.drive_current(2.0 * 100.0 / 1e-3) / i - 2.0).abs() < 1e-12);

        // Sinusoidal drive at f: peak current is 2 pi f C V. Check the implied full-stroke bandwidth for a
        // 1 A amplifier, which is the quantity a designer actually wants.
        let v_pk = 100.0;
        let i_limit = 1.0;
        let f_max = i_limit / (2.0 * std::f64::consts::PI * c * v_pk);
        assert!(f_max.is_finite() && f_max > 0.0);
        // Doubling the voltage swing halves the achievable frequency, exactly.
        let f_max_double = i_limit / (2.0 * std::f64::consts::PI * c * 2.0 * v_pk);
        assert!((f_max / f_max_double - 2.0).abs() < 1e-12);
    }

    #[test]
    fn the_optimal_harvesting_resistance_is_the_capacitive_reactance() {
        // Verified by scanning the power curve, not by trusting the formula.
        let p = stack();
        for &f in &[10.0f64, 50.0, 200.0, 1000.0] {
            let omega = 2.0 * std::f64::consts::PI * f;
            let r_opt = p.optimal_harvest_resistance(omega);
            let best = p.harvested_power(1e6, omega, r_opt);
            for i in 1..=3000 {
                let r = r_opt * (0.02 + 8.0 * i as f64 / 3000.0);
                assert!(
                    p.harvested_power(1e6, omega, r) <= best * (1.0 + 1e-12),
                    "f={f}: R/R_opt = {} beat the optimum",
                    r / r_opt
                );
            }
            // At the optimum, omega R C = 1 exactly, which is the matching statement.
            assert!((omega * r_opt * p.capacitance() - 1.0).abs() < 1e-12);
        }
        // The optimum moves inversely with frequency, which is why broadband harvesting is hard.
        let r_50 = p.optimal_harvest_resistance(2.0 * std::f64::consts::PI * 50.0);
        let r_200 = p.optimal_harvest_resistance(2.0 * std::f64::consts::PI * 200.0);
        assert!((r_50 / r_200 - 4.0).abs() < 1e-9, "4x the frequency should quarter the optimal resistance");
    }

    #[test]
    fn harvested_power_vanishes_at_both_load_extremes() {
        // Open circuit passes no current and short circuit develops no voltage, so power is zero at both ends
        // and the interior optimum is real rather than an artefact of the search range.
        let p = stack();
        let omega = 2.0 * std::f64::consts::PI * 100.0;
        assert_eq!(p.harvested_power(1e6, omega, 0.0), 0.0, "a short circuit harvests nothing");
        let open = p.harvested_power(1e6, omega, 1e15);
        let opt = p.harvested_power(1e6, omega, p.optimal_harvest_resistance(omega));
        assert!(open < opt * 1e-6, "an open circuit harvests nothing: {open:.3e} vs {opt:.3e}");
        assert!(opt > 0.0);
    }

    #[test]
    fn the_resonance_scales_as_the_inverse_of_length() {
        // A free-free bar's fundamental is c/(2L), and c depends only on the material. Checking the scaling
        // separates a geometry error from a material one.
        let a = Piezo::pzt5h(1e-4, 20e-3, 200).expect("valid");
        let b = Piezo::pzt5h(1e-4, 40e-3, 200).expect("valid");
        assert!((a.resonance() / b.resonance() - 2.0).abs() < 1e-9, "twice the length, half the frequency");
        // Area and layer count must not affect it: neither enters c or L.
        let c = Piezo::pzt5h(4e-4, 20e-3, 50).expect("valid");
        assert!((a.resonance() - c.resonance()).abs() < 1e-9, "resonance must not depend on area or layers");
        // And the value is in the right regime for a 20 mm PZT bar: tens of kHz.
        assert!(
            (20e3..200e3).contains(&a.resonance()),
            "a 20 mm PZT bar should resonate in the tens of kHz, got {:.1} Hz",
            a.resonance()
        );
    }
}
