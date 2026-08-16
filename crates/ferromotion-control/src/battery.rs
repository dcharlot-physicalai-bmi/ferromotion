//! **Battery supply model** — the input side of joules-per-task, and why torque fades as the pack empties.
//!
//! Everything upstream treats voltage as a constant. A real pack does not: terminal voltage is
//! `OCV(SOC) − I·R_int`, so drawing current *lowers the voltage available to the motors*, which lowers the
//! achievable speed and torque, which changes the current drawn. A robot near the end of a mission is a
//! different machine from the same robot at full charge, and nothing in the stack could express that.
//!
//! Pairs directly with [`crate::MotorThermal`]: that module accounts for where the energy goes as heat in the
//! winding, this one accounts for where it comes from and what it costs at the source. Together they close the
//! path from a commanded torque to joules drawn from a pack — which is the quantity an energy-per-task figure
//! is actually about.
//!
//! # The model
//!
//! A Thévenin equivalent circuit, which is what pack datasheets and BMS firmware are specified against:
//!
//! ```text
//! V_terminal = OCV(SOC) − I·R_int
//! dSOC/dt    = −I / Q          (coulomb counting)
//! ```
//!
//! `OCV(SOC)` is the open-circuit voltage curve — for lithium chemistries a plateau with knees at both ends,
//! approximated here by a piecewise-linear table so a real measured curve can be substituted without changing
//! the model.
//!
//! # Energy conservation is the oracle
//!
//! The energy removed from the cell splits exactly into work delivered at the terminals plus internal
//! dissipation:
//!
//! ```text
//! ∫ OCV·I dt  =  ∫ V_term·I dt  +  ∫ I²R_int dt
//! ```
//!
//! That identity is checked to machine precision in the tests, and it is what makes the internal resistance a
//! *loss* rather than a fudge factor. It is also the reason a high-current draw is doubly expensive: the useful
//! fraction falls as `V_term/OCV`, so the same joules removed deliver less work.

/// A pack modelled as an OCV source behind an internal resistance.
#[derive(Clone, Debug)]
pub struct Battery {
    /// Capacity in coulombs. A 2 Ah cell is `2 × 3600 = 7200` C.
    pub capacity_c: f64,
    /// Internal resistance (Ω).
    pub r_internal: f64,
    /// State of charge in `[0, 1]`.
    pub soc: f64,
    /// Open-circuit voltage table as ascending `(soc, volts)` points, linearly interpolated.
    pub ocv_curve: Vec<(f64, f64)>,
}

impl Battery {
    /// A pack with a generic lithium-ion OCV shape: knees near empty and full, a plateau between.
    ///
    /// `nominal` scales the whole curve, so `Battery::lithium(7200.0, 0.05, 24.0)` is a 2 Ah, 24 V pack.
    pub fn lithium(capacity_c: f64, r_internal: f64, nominal: f64) -> Self {
        // Normalised shape (fraction of nominal) against SOC — the characteristic Li-ion curve.
        let shape = [
            (0.00, 0.82),
            (0.05, 0.90),
            (0.20, 0.95),
            (0.50, 1.00),
            (0.80, 1.05),
            (0.95, 1.10),
            (1.00, 1.13),
        ];
        Self {
            capacity_c,
            r_internal,
            soc: 1.0,
            ocv_curve: shape.iter().map(|(s, f)| (*s, f * nominal)).collect(),
        }
    }

    /// Open-circuit voltage at the present SOC, linearly interpolated from the table.
    ///
    /// Clamped at both ends rather than extrapolated: a curve fitted over `[0, 1]` says nothing outside it, and
    /// extrapolating an OCV knee produces confidently wrong voltages.
    pub fn ocv(&self) -> f64 {
        self.ocv_at(self.soc)
    }

    /// Open-circuit voltage at an arbitrary SOC.
    pub fn ocv_at(&self, soc: f64) -> f64 {
        let c = &self.ocv_curve;
        if c.is_empty() {
            return 0.0;
        }
        let s = soc.clamp(c[0].0, c[c.len() - 1].0);
        for w in c.windows(2) {
            let ((s0, v0), (s1, v1)) = (w[0], w[1]);
            if s >= s0 && s <= s1 {
                let t = if (s1 - s0).abs() < f64::EPSILON { 0.0 } else { (s - s0) / (s1 - s0) };
                return v0 + t * (v1 - v0);
            }
        }
        c[c.len() - 1].1
    }

    /// Terminal voltage under load current `i` (positive = discharge): `OCV − I·R_int`.
    ///
    /// Can go negative for an absurd current, which is physical for this model and a signal the draw is outside
    /// what the pack can supply — see [`Battery::max_power_current`].
    pub fn terminal_voltage(&self, i: f64) -> f64 {
        self.ocv() - i * self.r_internal
    }

    /// Power delivered at the terminals under current `i`: `V_term·I`.
    pub fn delivered_power(&self, i: f64) -> f64 {
        self.terminal_voltage(i) * i
    }

    /// Internal dissipation under current `i`: `I²·R_int`.
    pub fn loss_power(&self, i: f64) -> f64 {
        i * i * self.r_internal
    }

    /// The current at which delivered power is **maximal**: `OCV / (2·R_int)`.
    ///
    /// Beyond it, drawing more current delivers *less* power — the terminal voltage falls faster than the
    /// current rises. At that point efficiency is exactly 50 %, so it is a ceiling rather than an operating
    /// point, but it bounds what any controller can ask of the pack.
    pub fn max_power_current(&self) -> f64 {
        self.ocv() / (2.0 * self.r_internal)
    }

    /// Fraction of removed energy that reaches the terminals: `V_term/OCV`.
    pub fn efficiency(&self, i: f64) -> f64 {
        let ocv = self.ocv();
        if ocv == 0.0 {
            return 0.0;
        }
        self.terminal_voltage(i) / ocv
    }

    /// Draw `i` amperes for `dt` seconds. Returns `(delivered_joules, lost_joules)`.
    ///
    /// SOC is clamped to `[0, 1]`, so a pack cannot be discharged past empty or charged past full by coulomb
    /// counting alone. A negative `i` charges.
    pub fn step(&mut self, dt: f64, i: f64) -> (f64, f64) {
        let delivered = self.delivered_power(i) * dt;
        let lost = self.loss_power(i) * dt;
        self.soc = (self.soc - i * dt / self.capacity_c).clamp(0.0, 1.0);
        (delivered, lost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack() -> Battery {
        Battery::lithium(7200.0, 0.05, 24.0) // 2 Ah, 50 mΩ, 24 V nominal
    }

    /// **The energy identity**: what leaves the cell equals what reaches the terminals plus what heats the
    /// internal resistance. This is what makes `r_internal` a loss rather than a fitted constant.
    #[test]
    fn removed_energy_splits_exactly_into_delivered_and_dissipated() {
        let mut b = pack();
        let (dt, i) = (0.001, 8.0);
        let (mut delivered, mut lost, mut removed) = (0.0, 0.0, 0.0);
        for _ in 0..200_000 {
            removed += b.ocv() * i * dt; // energy leaving the ideal source
            let (d, l) = b.step(dt, i);
            delivered += d;
            lost += l;
        }
        assert!(delivered > 0.0 && lost > 0.0);
        assert!(
            (removed - (delivered + lost)).abs() < 1e-9 * removed,
            "energy must balance: removed {removed}, delivered {delivered} + lost {lost} = {}",
            delivered + lost
        );
    }

    /// Coulomb counting is exact: charge drawn equals capacity times the SOC change.
    #[test]
    fn coulomb_counting_is_exact_and_clamped() {
        let mut b = pack();
        let (dt, i) = (0.01, 5.0);
        let n = 10_000;
        let soc0 = b.soc;
        for _ in 0..n {
            b.step(dt, i);
        }
        let drawn = i * dt * n as f64;
        // Relative, not absolute: this is 10,000 accumulated subtractions, so the error floor is set by
        // floating-point accumulation rather than by the model. Measured 6.6e-12 relative, which is as good as
        // f64 gets over that many steps — an absolute 1e-9 bound was simply the wrong kind of tolerance.
        assert!(
            ((soc0 - b.soc) * b.capacity_c - drawn).abs() < 1e-9 * drawn,
            "SOC change should account for {drawn} C, got {}",
            (soc0 - b.soc) * b.capacity_c
        );

        // Cannot go below empty, however long the draw continues.
        for _ in 0..1_000_000 {
            b.step(dt, i);
        }
        assert_eq!(b.soc, 0.0, "SOC must clamp at empty");
        // Nor above full when charging.
        for _ in 0..1_000_000 {
            b.step(dt, -i);
        }
        assert_eq!(b.soc, 1.0, "SOC must clamp at full");
    }

    /// **Voltage sags by exactly `I·R`, and that is what steals torque.** A constant-voltage model misses it
    /// entirely.
    #[test]
    fn terminal_voltage_sags_linearly_with_current() {
        let b = pack();
        let ocv = b.ocv();
        for i in [0.0, 1.0, 10.0, 40.0] {
            assert!(
                (b.terminal_voltage(i) - (ocv - i * b.r_internal)).abs() < 1e-12,
                "sag must be exactly I·R at {i} A"
            );
        }
        // At 40 A through 50 mΩ the pack loses 2 V — over 7 % of a 27 V full-charge OCV.
        let sag = ocv - b.terminal_voltage(40.0);
        assert!((sag - 2.0).abs() < 1e-12, "40 A × 0.05 Ω = 2 V, got {sag}");
        assert!(sag / ocv > 0.07, "that is a material fraction of the supply: {}", sag / ocv);
        // Charging raises the terminal voltage above OCV, which is why a charger must exceed it.
        assert!(b.terminal_voltage(-10.0) > ocv);
    }

    /// **Past the maximum-power current, drawing more delivers less** — and efficiency there is exactly 50 %.
    #[test]
    fn delivered_power_peaks_then_falls() {
        let b = pack();
        let i_star = b.max_power_current();
        let p_star = b.delivered_power(i_star);
        for f in [0.5, 0.8, 1.2, 2.0] {
            assert!(
                b.delivered_power(i_star * f) < p_star + 1e-9,
                "power at {f}× the peak current should not exceed the peak"
            );
        }
        // Efficiency at the power peak is exactly one half — half the energy heats the pack.
        assert!((b.efficiency(i_star) - 0.5).abs() < 1e-12, "got {}", b.efficiency(i_star));
        // Beyond twice it, the terminal voltage is negative: the draw is outside what the pack can supply.
        assert!(b.terminal_voltage(i_star * 2.5) < 0.0);
        // And efficiency falls monotonically with current, which is why a peak draw is doubly expensive.
        assert!(b.efficiency(1.0) > b.efficiency(10.0));
        assert!(b.efficiency(10.0) > b.efficiency(40.0));
    }

    /// The OCV curve is **clamped, not extrapolated** — a table fitted on `[0,1]` says nothing outside it, and
    /// extrapolating a knee produces confidently wrong voltages.
    #[test]
    fn the_ocv_curve_is_interpolated_and_clamped() {
        let b = pack();
        // Monotone in SOC across the table.
        let mut prev = f64::NEG_INFINITY;
        for k in 0..=100 {
            let v = b.ocv_at(k as f64 / 100.0);
            assert!(v >= prev - 1e-12, "OCV should not fall with SOC at {k}%");
            prev = v;
        }
        // Endpoints, and clamping beyond them.
        assert!((b.ocv_at(0.0) - 0.82 * 24.0).abs() < 1e-12);
        assert!((b.ocv_at(1.0) - 1.13 * 24.0).abs() < 1e-12);
        assert_eq!(b.ocv_at(-5.0), b.ocv_at(0.0), "below empty clamps rather than extrapolating");
        assert_eq!(b.ocv_at(9.0), b.ocv_at(1.0), "above full clamps too");
        // Interpolation hits a midpoint exactly: halfway between the 0.5 and 0.8 knots.
        let mid = b.ocv_at(0.65);
        assert!((mid - (1.00 + 1.05) / 2.0 * 24.0).abs() < 1e-12, "linear interpolation, got {mid}");
        // A degenerate empty curve is 0 rather than a panic.
        let empty = Battery { capacity_c: 1.0, r_internal: 1.0, soc: 1.0, ocv_curve: vec![] };
        assert_eq!(empty.ocv(), 0.0);
    }

    /// **A draining pack becomes a weaker machine**: the same current yields less delivered power as SOC falls,
    /// because OCV falls. This is the effect the constant-voltage assumption erases.
    #[test]
    fn the_same_current_delivers_less_power_as_the_pack_empties() {
        let mut b = pack();
        let i = 10.0;
        let full = b.delivered_power(i);
        // Drain to roughly a tenth.
        while b.soc > 0.1 {
            b.step(0.01, i);
        }
        let low = b.delivered_power(i);
        assert!(low < full * 0.95, "delivered power should fall materially: {low} vs {full}");
        // and the sag is unchanged — it is OCV that moved, not the resistance
        let sag_full = pack().ocv() - pack().terminal_voltage(i);
        let sag_low = b.ocv() - b.terminal_voltage(i);
        assert!((sag_full - sag_low).abs() < 1e-12, "I·R does not depend on SOC in this model");
    }
}
