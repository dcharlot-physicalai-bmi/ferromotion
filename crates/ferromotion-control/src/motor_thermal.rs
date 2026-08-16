//! **Motor winding temperature** — the limit that actually bounds a robot's torque.
//!
//! [`crate::actuator::DcMotor`] gives the electrical side: resistance, inductance, `kₑ`, `k_t`, cogging. What
//! nothing in the stack knew is **how hot the winding is**, and that is what sets the real torque envelope. A
//! motor will deliver several times its continuous torque for a few seconds and destroy itself doing it for a
//! minute; a planner that sees only a torque bound cannot tell those apart, so it either wastes the peak or
//! burns the motor.
//!
//! This is also where the joules go. Copper loss `I²R` is the dominant term at the low speeds a manipulator
//! spends most of its time at — a stalled joint holding a load produces heat and *no mechanical work at all* —
//! so an energy-per-task figure that ignores winding temperature is ignoring the largest contributor.
//!
//! # The model
//!
//! A two-node lumped network, which is what motor datasheets are specified against:
//!
//! ```text
//! C_w·Ṫ_w = P_loss − (T_w − T_h)/R_wh
//! C_h·Ṫ_h = (T_w − T_h)/R_wh − (T_h − T_amb)/R_ha
//! ```
//!
//! winding → housing → ambient, each hop a thermal resistance (K/W) and each node a heat capacity (J/K).
//!
//! # Copper's temperature coefficient makes this a feedback loop, not a lookup
//!
//! Resistance rises with temperature, `R(T) = R₂₅·(1 + α·(T − 25))` with `α ≈ 0.00393 /K` for copper. So
//! hotter winding → higher resistance → more loss → hotter winding. At around **100 K** above calibration
//! that is a **+39 %** resistance increase, and it means:
//!
//! * steady-state temperature is *not* linear in `I²`;
//! * above a critical current **there is no steady state at all** — the loop diverges. That is thermal
//!   runaway, and it is a property of the physics rather than a numerical artifact.
//!
//! [`MotorThermal::equilibrium_rise`] solves the fixed point where one exists and returns `None` where it does
//! not, because a large finite number there would be a fiction. [`MotorThermal::continuous_current`] inverts
//! the relation to give the current a motor can hold indefinitely at a stated ambient and limit — the number a
//! planner actually needs, and the one a datasheet quotes.

/// Copper's resistance temperature coefficient, per kelvin.
pub const ALPHA_COPPER: f64 = 0.003_93;

/// Two-node lumped thermal model of a motor: winding and housing.
#[derive(Clone, Copy, Debug)]
pub struct MotorThermal {
    /// Winding resistance at the calibration temperature (Ω).
    pub r_25: f64,
    /// Calibration temperature for `r_25` (°C). Datasheets almost always use 25.
    pub t_ref: f64,
    /// Resistance temperature coefficient (1/K). [`ALPHA_COPPER`] for copper windings.
    pub alpha: f64,
    /// Winding heat capacity (J/K) — small, so the winding responds in seconds.
    pub c_winding: f64,
    /// Housing heat capacity (J/K) — large, so the housing responds in minutes.
    pub c_housing: f64,
    /// Winding-to-housing thermal resistance (K/W).
    pub r_wh: f64,
    /// Housing-to-ambient thermal resistance (K/W).
    pub r_ha: f64,
    /// Winding temperature state (°C).
    pub t_winding: f64,
    /// Housing temperature state (°C).
    pub t_housing: f64,
}

impl MotorThermal {
    /// A motor thermal model, both nodes initialised at `ambient`.
    pub fn new(r_25: f64, c_winding: f64, c_housing: f64, r_wh: f64, r_ha: f64, ambient: f64) -> Self {
        Self {
            r_25,
            t_ref: 25.0,
            alpha: ALPHA_COPPER,
            c_winding,
            c_housing,
            r_wh,
            r_ha,
            t_winding: ambient,
            t_housing: ambient,
        }
    }

    /// Winding resistance at the current temperature: `R₂₅·(1 + α·(T − T_ref))`.
    pub fn resistance(&self) -> f64 {
        self.r_25 * (1.0 + self.alpha * (self.t_winding - self.t_ref))
    }

    /// Copper loss at current `i`, using the **temperature-corrected** resistance: `I²·R(T)`.
    ///
    /// Using `R₂₅` here instead is the standard way to under-predict heating by tens of percent once a motor
    /// is hot.
    pub fn copper_loss(&self, i: f64) -> f64 {
        i * i * self.resistance()
    }

    /// Advance both nodes by `dt` under winding current `i` and the given `ambient`.
    ///
    /// Explicit Euler, which is adequate because the winding time constant `C_w·R_wh` is seconds while a
    /// control step is milliseconds — but the constraint is real: see [`MotorThermal::max_stable_dt`].
    pub fn step(&mut self, dt: f64, i: f64, ambient: f64) {
        let p = self.copper_loss(i);
        let q_wh = (self.t_winding - self.t_housing) / self.r_wh;
        let q_ha = (self.t_housing - ambient) / self.r_ha;
        self.t_winding += dt * (p - q_wh) / self.c_winding;
        self.t_housing += dt * (q_wh - q_ha) / self.c_housing;
    }

    /// The largest `dt` for which [`MotorThermal::step`] is stable, from the faster (winding) node:
    /// `C_w·R_wh`, which is that node's time constant.
    ///
    /// Explicit Euler on a linear decay needs `dt < 2τ`; this returns `τ` as the practical bound, since running
    /// at the stability edge gives an oscillating temperature that is useless even when bounded.
    pub fn max_stable_dt(&self) -> f64 {
        self.c_winding * self.r_wh
    }

    /// Steady-state winding rise above ambient at constant current `i`, or `None` if no steady state exists
    /// (thermal runaway).
    ///
    /// At equilibrium all the loss flows to ambient through `R_wh + R_ha =: R_tot`, so
    /// `ΔT = I²·R₂₅·(1 + α·(T_amb + ΔT − T_ref))·R_tot`. Solving for `ΔT`:
    ///
    /// ```text
    /// ΔT = I²R₂₅R_tot·(1 + α(T_amb − T_ref)) / (1 − I²R₂₅R_tot·α)
    /// ```
    ///
    /// The denominator is where the physics bites: once `I²·R₂₅·R_tot·α ≥ 1` there is **no** fixed point and the
    /// winding runs away. `None` is the honest answer, not a large number.
    pub fn equilibrium_rise(&self, i: f64, ambient: f64) -> Option<f64> {
        let r_tot = self.r_wh + self.r_ha;
        let g = i * i * self.r_25 * r_tot;
        let denom = 1.0 - g * self.alpha;
        if denom <= 0.0 {
            return None; // thermal runaway: the feedback loop has no fixed point
        }
        Some(g * (1.0 + self.alpha * (ambient - self.t_ref)) / denom)
    }

    /// The current the motor can hold **indefinitely** without exceeding `t_limit` at the given `ambient` — the
    /// continuous rating, and the bound a planner should be given instead of a bare torque limit.
    ///
    /// Inverts [`MotorThermal::equilibrium_rise`]: from `ΔT` allowed, `I² = ΔT / (R₂₅·R_tot·(1 + α(T_lim − T_ref)))`,
    /// where the resistance is evaluated **at the limit temperature** because that is the worst case the motor
    /// must survive. Returns `None` if the limit is at or below ambient.
    pub fn continuous_current(&self, t_limit: f64, ambient: f64) -> Option<f64> {
        let rise = t_limit - ambient;
        if rise <= 0.0 {
            return None;
        }
        let r_hot = self.r_25 * (1.0 + self.alpha * (t_limit - self.t_ref));
        Some((rise / (r_hot * (self.r_wh + self.r_ha))).sqrt())
    }

    /// How long a current `i` may be applied from the present state before the winding reaches `t_limit`, or
    /// `None` if it never will.
    ///
    /// Integrated numerically because `R(T)` makes the ODE nonlinear — the closed-form exponential of a
    /// constant-resistance model under-predicts the time to limit, in the unsafe direction.
    pub fn time_to_limit(&self, i: f64, ambient: f64, t_limit: f64, dt: f64, max_t: f64) -> Option<f64> {
        let mut m = *self;
        let mut t = 0.0;
        while t < max_t {
            if m.t_winding >= t_limit {
                return Some(t);
            }
            m.step(dt, i, ambient);
            t += dt;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small geared joint motor: ~1 Ω, winding responds in seconds, housing in minutes.
    fn joint_motor(ambient: f64) -> MotorThermal {
        MotorThermal::new(1.0, 8.0, 400.0, 1.2, 1.8, ambient)
    }

    #[test]
    fn the_equilibrium_matches_a_long_integration() {
        // The analytic fixed point and the integrated dynamics must agree — one checks the other, and neither
        // is the reference for itself.
        for i in [0.3, 0.8, 1.5] {
            let mut m = joint_motor(25.0);
            let predicted = m.equilibrium_rise(i, 25.0).expect("should have a steady state");
            let dt = 0.01;
            for _ in 0..600_000 {
                m.step(dt, i, 25.0);
            }
            let settled = m.t_winding - 25.0;
            assert!(
                (settled - predicted).abs() < 1e-3 * predicted.max(1.0),
                "i={i}: integrated to {settled} K rise, fixed point says {predicted}"
            );
        }
    }

    /// **The temperature coefficient makes the rise superlinear in `I²`, and that is the whole point.** A
    /// constant-resistance model would give exact proportionality.
    #[test]
    fn heating_is_superlinear_because_resistance_rises_with_temperature() {
        let m = joint_motor(25.0);
        let r1 = m.equilibrium_rise(1.0, 25.0).unwrap();
        let r2 = m.equilibrium_rise(2.0, 25.0).unwrap();
        // With constant R, doubling current would quadruple the rise exactly.
        assert!(
            r2 > 4.0 * r1 * 1.02,
            "doubling current should raise temperature MORE than 4x ({r2} vs 4×{r1}) once R(T) is included"
        );
        // and the mechanism is visible in the resistance itself: ~+39% at 100 K above calibration
        let mut hot = joint_motor(25.0);
        hot.t_winding = 125.0;
        assert!(
            (hot.resistance() / hot.r_25 - 1.393).abs() < 1e-3,
            "100 K should raise copper resistance ~39%, got {}",
            hot.resistance() / hot.r_25
        );
    }

    /// **Thermal runaway is refused, not approximated.** Once `I²·R₂₅·R_tot·α ≥ 1` the feedback loop has no
    /// fixed point, and a large finite answer would be a fiction.
    #[test]
    fn beyond_a_critical_current_there_is_no_steady_state() {
        let m = joint_motor(25.0);
        let r_tot = m.r_wh + m.r_ha;
        // I_crit where the denominator vanishes
        let i_crit = (1.0 / (m.r_25 * r_tot * m.alpha)).sqrt();
        assert!(m.equilibrium_rise(i_crit * 0.95, 25.0).is_some(), "just below critical there is a fixed point");
        assert!(m.equilibrium_rise(i_crit * 1.05, 25.0).is_none(), "just above it there is none");

        // And the dynamics agree: above critical the winding keeps climbing rather than settling.
        let mut runaway = joint_motor(25.0);
        let mut prev = runaway.t_winding;
        for _ in 0..200_000 {
            runaway.step(0.005, i_crit * 1.2, 25.0);
        }
        assert!(runaway.t_winding > prev + 500.0, "runaway should climb without bound, reached {}", runaway.t_winding);
        prev = runaway.t_winding;
        for _ in 0..50_000 {
            runaway.step(0.005, i_crit * 1.2, 25.0);
        }
        assert!(runaway.t_winding > prev, "and keep climbing");
    }

    /// The continuous rating must be exactly the current whose equilibrium sits at the limit — the two
    /// functions are inverses and a sign or a `t_ref` slip would break the round trip.
    #[test]
    fn the_continuous_rating_round_trips_against_the_equilibrium() {
        let m = joint_motor(25.0);
        for (limit, ambient) in [(100.0, 25.0), (155.0, 25.0), (100.0, 45.0), (80.0, 20.0)] {
            let i = m.continuous_current(limit, ambient).expect("a limit above ambient is achievable");
            let rise = m.equilibrium_rise(i, ambient).expect("the continuous current cannot be runaway");
            assert!(
                (ambient + rise - limit).abs() < 1e-6,
                "limit={limit} ambient={ambient}: I={i} settles at {} not {limit}",
                ambient + rise
            );
        }
        // A hotter ambient lowers the continuous rating — the derating a datasheet tabulates.
        let cool = m.continuous_current(100.0, 20.0).unwrap();
        let warm = m.continuous_current(100.0, 60.0).unwrap();
        assert!(warm < cool, "a hotter ambient must lower the rating: {warm} vs {cool}");
        assert!(m.continuous_current(25.0, 25.0).is_none(), "no headroom at ambient");
        assert!(m.continuous_current(20.0, 25.0).is_none(), "a limit below ambient is unachievable");
    }

    /// **Peak torque is a duration, not a number.** A current well above the continuous rating is survivable
    /// for a bounded time, and that time is what a planner needs — this is the distinction a bare torque bound
    /// cannot express.
    #[test]
    fn overload_is_survivable_for_a_bounded_time() {
        let m = joint_motor(25.0);
        let cont = m.continuous_current(100.0, 25.0).unwrap();

        // At the continuous rating the winding approaches the limit but never exceeds it, so there is no
        // finite time-to-limit.
        assert!(
            m.time_to_limit(cont * 0.98, 25.0, 100.0, 0.01, 3600.0).is_none(),
            "at/below the continuous rating the limit is never reached"
        );

        // Above it, the time to limit is finite and shrinks as the overload grows.
        let t2 = m.time_to_limit(cont * 2.0, 25.0, 100.0, 0.005, 3600.0).expect("2x should reach the limit");
        let t3 = m.time_to_limit(cont * 3.0, 25.0, 100.0, 0.005, 3600.0).expect("3x should reach it sooner");
        assert!(t3 < t2, "a bigger overload must reach the limit sooner: {t3} vs {t2}");
        assert!(t2 > 0.0 && t2 < 3600.0, "2x overload time should be finite and non-trivial: {t2}");
        // and the winding node is the fast one — seconds to tens of seconds, not minutes
        assert!(t3 < 120.0, "a 3x overload should bite within a couple of minutes, got {t3}");
    }

    /// The two nodes have separated time constants: the winding responds in seconds, the housing in minutes.
    /// That separation is why a short overload is survivable at all, and it is what `max_stable_dt` reports.
    #[test]
    fn the_winding_is_fast_and_the_housing_is_slow() {
        let m = joint_motor(25.0);
        let tau_w = m.c_winding * m.r_wh;
        let tau_h = m.c_housing * m.r_ha;
        assert!(tau_h > 20.0 * tau_w, "the housing must be much slower: {tau_h} vs {tau_w}");
        assert!((m.max_stable_dt() - tau_w).abs() < 1e-12);

        // A control step is milliseconds, far inside the bound — but check the bound is actually informative
        // rather than trivially large.
        assert!(m.max_stable_dt() > 1.0 && m.max_stable_dt() < 100.0, "tau_w = {}", m.max_stable_dt());

        // Over one winding time constant at a step overload the winding moves substantially while the housing
        // barely does — the physical basis of a peak rating.
        let mut hot = joint_motor(25.0);
        let i = m.continuous_current(100.0, 25.0).unwrap() * 3.0;
        let steps = (tau_w / 0.01) as usize;
        for _ in 0..steps {
            hot.step(0.01, i, 25.0);
        }
        assert!(hot.t_winding - 25.0 > 10.0, "winding should heat quickly, rose {}", hot.t_winding - 25.0);
        assert!(hot.t_housing - 25.0 < 2.0, "housing should barely move, rose {}", hot.t_housing - 25.0);
    }
}
