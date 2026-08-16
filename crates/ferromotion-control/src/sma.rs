//! **Shape-memory alloy** — the actuator whose state depends on where it has been.
//!
//! A nitinol wire contracts a few percent when heated and recovers when cooled, at a work density around
//! `10 MJ/m³` — roughly a hundred times an electric motor's and more than any other actuator in common use. It
//! is how you get large force from something the size of a hair. What it costs you is everything in this
//! module's title.
//!
//! * **The transformation is hysteretic, and the hysteresis is intrinsic.** Heating and cooling follow different
//!   paths, separated by tens of kelvin. This is not a defect to be calibrated out: it is the same first-order
//!   phase transition that stores the shape memory. A model without it predicts a single-valued
//!   temperature-to-strain map that does not exist, and open-loop position control built on that map is wrong by
//!   the full width of the loop.
//! * **The state is a memory, not a function of the inputs.** Martensite fraction `ξ` at a given `(T, σ)`
//!   depends on whether you arrived heating or cooling, and on how far you got before turning around. A partial
//!   cycle lands *inside* the loop, not on either branch.
//! * **Bandwidth is set by cooling, and cooling is passive.** Heating is Joule heating: as fast as the current
//!   source allows. Cooling is convection into whatever is nearby. The asymmetry is typically an order of
//!   magnitude or more, so the actuation rate is a thermal design problem and not an electrical one.
//!   [`Sma::cycle_time`] makes the asymmetry explicit rather than letting a symmetric time constant imply
//!   a bandwidth the wire does not have.
//!
//! # The model
//!
//! **Liang-Rogers cosine** kinetics, which is the standard one-dimensional model. Four transformation
//! temperatures at zero stress — `M_f ≤ M_s ≤ A_s ≤ A_f` — and two Clausius-Clapeyron slopes shifting them
//! under load:
//!
//! ```text
//!   heating (martensite → austenite):   ξ = ½[cos(a_A (T − A_s − σ/C_A)) + 1]
//!   cooling (austenite → martensite):   ξ = ½[cos(a_M (T − M_f − σ/C_M)) + 1]
//! ```
//!
//! with `a_A = π/(A_f − A_s)` and `a_M = π/(M_s − M_f)`. The cosine is a smooth interpolation whose endpoints
//! are **exact**: `ξ = 0` at `A_f` and `ξ = 1` at `M_f`, which the tests check to machine precision because a
//! model that only approaches full transformation leaves a residual stroke error that looks like miscalibration.
//!
//! Constitutively, with `ξ` known:
//!
//! ```text
//!   σ = E(ξ)(ε − ε_L ξ) + Θ(T − T₀),        E(ξ) = E_A + ξ(E_M − E_A)
//! ```
//!
//! The **modulus itself depends on phase** — martensite is roughly a third as stiff as austenite — so an SMA
//! actuator's stiffness changes by 3x over its stroke. A constant-`E` model gets the force wrong at one end of
//! the travel or the other, and there is no single `E` that is right at both.
//!
//! # What the tests pin
//!
//! `ξ` stays in `[0, 1]` for arbitrary input sequences including reversals; the cosine endpoints are exact; the
//! Clausius-Clapeyron shift is linear in stress with the stated slope; a full thermal cycle **closes** and
//! encloses positive area, while a partial cycle lands strictly inside the loop; and detwinning does mechanical
//! work whose sign is checked rather than assumed.

use std::f64::consts::PI;

/// Which way the last transformation step was going. The state variable that makes this model path-dependent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Last step raised the temperature: on or under the austenite (heating) branch.
    Heating,
    /// Last step lowered it: on or under the martensite (cooling) branch.
    Cooling,
}

/// A one-dimensional shape-memory alloy element, Liang-Rogers kinetics.
#[derive(Clone, Copy, Debug)]
pub struct Sma {
    /// Martensite finish temperature at zero stress (K).
    pub m_f: f64,
    /// Martensite start temperature at zero stress (K).
    pub m_s: f64,
    /// Austenite start temperature at zero stress (K).
    pub a_s: f64,
    /// Austenite finish temperature at zero stress (K).
    pub a_f: f64,
    /// Clausius-Clapeyron slope for the martensite transformation, `dσ/dT` (Pa/K).
    pub c_m: f64,
    /// Clausius-Clapeyron slope for the austenite transformation (Pa/K).
    pub c_a: f64,
    /// Austenite Young's modulus (Pa).
    pub e_austenite: f64,
    /// Martensite Young's modulus (Pa). Typically about a third of austenite's.
    pub e_martensite: f64,
    /// Maximum recoverable (transformation) strain, `ε_L`. Four to eight percent for nitinol.
    pub eps_l: f64,
}

/// The path-dependent state of an [`Sma`] element.
#[derive(Clone, Copy, Debug)]
pub struct SmaState {
    /// Martensite fraction, in `[0, 1]`. `1` is fully martensitic (cold, long), `0` fully austenitic.
    pub xi: f64,
    /// Current temperature (K).
    pub temperature: f64,
    /// Current stress (Pa).
    pub stress: f64,
    /// Which branch the last step was on.
    pub direction: Direction,
    /// The martensite fraction at which the **current branch began**, latched on reversal.
    ///
    /// Liang-Rogers kinetics are written relative to the start of the current transformation, not relative to
    /// the previous instant. Re-applying the formula each step with the *current* `xi` as its start compounds
    /// the cosine factor: heating from `xi = 1` to the middle of the austenite window in 100 steps gave
    /// **0.0144** where it should give 0.5, and holding the temperature constant drove `xi` to `3e-126`. Both
    /// are the same bug. The branch start must be latched.
    pub xi_branch_start: f64,
}

impl Sma {
    /// Representative **nitinol**: `M_f = 292`, `M_s = 309`, `A_s = 315`, `A_f = 330` K,
    /// `C_M = 8 MPa/K`, `C_A = 13.8 MPa/K`, `E_A = 67 GPa`, `E_M = 26.3 GPa`, `ε_L = 0.067`.
    pub fn nitinol() -> Sma {
        Sma {
            m_f: 292.0,
            m_s: 309.0,
            a_s: 315.0,
            a_f: 330.0,
            c_m: 8.0e6,
            c_a: 13.8e6,
            e_austenite: 67.0e9,
            e_martensite: 26.3e9,
            eps_l: 0.067,
        }
    }

    /// Validate the temperature ordering `M_f ≤ M_s ≤ A_s ≤ A_f` and the positivity of the rest.
    ///
    /// The ordering is not a convention: it is what makes the two branches distinct and the loop have an
    /// interior. A parameter set violating it describes a material with negative hysteresis, and the cosine
    /// kinetics would divide by a non-positive width.
    pub fn is_valid(&self) -> bool {
        self.m_f <= self.m_s
            && self.m_s <= self.a_s
            && self.a_s <= self.a_f
            && self.m_s > self.m_f
            && self.a_f > self.a_s
            && self.c_m > 0.0
            && self.c_a > 0.0
            && self.e_austenite > 0.0
            && self.e_martensite > 0.0
            && self.eps_l > 0.0
    }

    /// Hysteresis width at zero stress, `A_f − M_s` (K). The gap a single-valued model would have to pretend
    /// away.
    pub fn hysteresis_width(&self) -> f64 {
        self.a_f - self.m_s
    }

    /// The four transformation temperatures shifted by stress, via Clausius-Clapeyron:
    /// `(M_f, M_s, A_s, A_f)` each raised by `σ/C`.
    ///
    /// Load **raises** the transformation temperatures, so a wire under load needs to be hotter to contract.
    /// This is the effect that makes an SMA actuator's timing depend on what it is lifting.
    pub fn shifted_temperatures(&self, stress: f64) -> (f64, f64, f64, f64) {
        let dm = stress / self.c_m;
        let da = stress / self.c_a;
        (self.m_f + dm, self.m_s + dm, self.a_s + da, self.a_f + da)
    }

    /// Martensite fraction on the **heating** branch, from a starting fraction `xi_start` held when heating
    /// began.
    ///
    /// Exactly `xi_start` at or below `A_s` and exactly `0` at or above `A_f`, so a completed heating stroke
    /// leaves no residual martensite.
    pub fn xi_heating(&self, temperature: f64, stress: f64, xi_start: f64) -> f64 {
        let (_, _, a_s, a_f) = self.shifted_temperatures(stress);
        if temperature <= a_s {
            return xi_start;
        }
        if temperature >= a_f {
            return 0.0;
        }
        let a = PI / (a_f - a_s);
        0.5 * xi_start * ((a * (temperature - a_s)).cos() + 1.0)
    }

    /// Martensite fraction on the **cooling** branch, from a starting fraction `xi_start`.
    ///
    /// Exactly `xi_start` at or above `M_s` and exactly `1` at or below `M_f`.
    pub fn xi_cooling(&self, temperature: f64, stress: f64, xi_start: f64) -> f64 {
        let (m_f, m_s, _, _) = self.shifted_temperatures(stress);
        if temperature >= m_s {
            return xi_start;
        }
        if temperature <= m_f {
            return 1.0;
        }
        let a = PI / (m_s - m_f);
        // xi = (1 - xi_start)/2 * cos(a (T - M_f)) + (1 + xi_start)/2.
        //
        // The sign on the cosine was inverted in the first version, and NONE of the endpoint tests saw it: both
        // ends are hard-coded early returns above, and the window's midpoint is exactly where cos = 0, so the
        // error vanished at all three points I had checked. Only the monotonicity assertion caught it. Any
        // interior point away from the midpoint distinguishes the two forms, which is what
        // `the_cooling_branch_interior_matches_its_analytic_form` now checks.
        0.5 * (1.0 - xi_start) * (a * (temperature - m_f)).cos() + 0.5 * (1.0 + xi_start)
    }

    /// A fresh state: fully martensitic at `temperature`, unloaded, on the cooling branch.
    pub fn cold(&self, temperature: f64) -> SmaState {
        SmaState {
            xi: 1.0,
            temperature,
            stress: 0.0,
            direction: Direction::Cooling,
            xi_branch_start: 1.0,
        }
    }

    /// Advance to a new temperature and stress, following whichever branch the motion implies.
    ///
    /// **A reversal latches the current `ξ` as the new branch's start**, which is what puts a partial cycle
    /// inside the loop rather than on either boundary. Without that latch, turning around mid-transformation
    /// would snap the state onto the opposite branch and the model would report motion the wire does not make.
    pub fn step(&self, st: &mut SmaState, temperature: f64, stress: f64) {
        let heating = temperature > st.temperature;
        let cooling = temperature < st.temperature;
        let new_dir = if heating {
            Direction::Heating
        } else if cooling {
            Direction::Cooling
        } else {
            st.direction
        };
        // Latch ONLY on reversal. The kinetics are relative to the branch's start, so carrying the current
        // value in as the start compounds the cosine every step; see `SmaState::xi_branch_start`.
        if new_dir != st.direction {
            st.xi_branch_start = st.xi;
        }
        let start = st.xi_branch_start;
        // No min/max guard here: with the start correctly latched the formulas are already monotone within a
        // branch, and clamping against the branch start would freeze a partial cycle in place.
        st.xi = match new_dir {
            Direction::Heating => self.xi_heating(temperature, stress, start),
            Direction::Cooling => self.xi_cooling(temperature, stress, start),
        }
        .clamp(0.0, 1.0);
        st.temperature = temperature;
        st.stress = stress;
        st.direction = new_dir;
    }

    /// Young's modulus at martensite fraction `xi`: `E_A + ξ(E_M − E_A)`.
    ///
    /// Varies by about 3x across the stroke for nitinol, which is why a constant-`E` actuator model cannot be
    /// right at both ends of the travel.
    pub fn modulus(&self, xi: f64) -> f64 {
        self.e_austenite + xi * (self.e_martensite - self.e_austenite)
    }

    /// Recoverable strain at fraction `xi`: `ε_L ξ`. The wire is longest when fully martensitic.
    pub fn transformation_strain(&self, xi: f64) -> f64 {
        self.eps_l * xi
    }

    /// Stress for a given total strain and state: `σ = E(ξ)(ε − ε_L ξ)`.
    ///
    /// The thermoelastic `Θ(T − T₀)` term is omitted: it is small against the transformation term for nitinol
    /// and carrying it would require a reference temperature this API does not have. Stated rather than
    /// silently dropped.
    pub fn stress_at(&self, strain: f64, xi: f64) -> f64 {
        self.modulus(xi) * (strain - self.transformation_strain(xi))
    }

    /// Force (N) a wire of cross-section `area` develops at a given strain and state.
    pub fn force(&self, area: f64, strain: f64, xi: f64) -> f64 {
        area * self.stress_at(strain, xi)
    }

    /// Free (zero-stress) actuation strain between fully martensitic and fully austenitic: `ε_L`.
    pub fn free_stroke(&self) -> f64 {
        self.eps_l
    }

    /// Work density (J/m³) available from a full stroke against a constant stress `sigma`.
    ///
    /// `σ ε_L`, maximised where the material can still transform. This is the figure that makes SMA
    /// interesting: at 200 MPa and 6.7% strain it is **13.4 MJ/m³**, roughly two orders above an electric
    /// motor's.
    pub fn work_density(&self, sigma: f64) -> f64 {
        sigma * self.eps_l
    }

    /// Heating and cooling times (s) for a wire, as `(heat, cool)`.
    ///
    /// Heating is Joule: `t = ρ c_p V ΔT / (I² R)` reduced to `thermal_mass ΔT / power`. Cooling is convective:
    /// `t = (thermal_mass/(h A)) ln(ΔT_initial/ΔT_final)`, a time constant no electrical input can shorten.
    ///
    /// **The asymmetry is the point.** Returning both makes it impossible to quote a bandwidth from the heating
    /// number alone, which is the standard way an SMA actuator's cycle rate gets overstated.
    pub fn cycle_time(
        &self,
        thermal_mass: f64,
        heating_power: f64,
        conv_coeff_area: f64,
        delta_t: f64,
        ambient_margin: f64,
    ) -> (f64, f64) {
        let heat = thermal_mass * delta_t / heating_power;
        // Newtonian cooling from delta_t above ambient down to ambient_margin above it.
        let tau = thermal_mass / conv_coeff_area;
        let cool = tau * (delta_t / ambient_margin).ln();
        (heat, cool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire() -> Sma {
        Sma::nitinol()
    }

    #[test]
    fn the_parameter_set_is_ordered_and_the_ordering_is_checked() {
        let s = wire();
        assert!(s.is_valid());
        assert!(s.m_f <= s.m_s && s.m_s <= s.a_s && s.a_s <= s.a_f, "M_f <= M_s <= A_s <= A_f");
        // A set with the austenite window inverted is invalid: the cosine would divide by a negative width.
        let mut bad = s;
        bad.a_f = bad.a_s - 5.0;
        assert!(!bad.is_valid());
        let mut bad2 = s;
        bad2.m_f = bad2.m_s + 5.0;
        assert!(!bad2.is_valid());
        // Hysteresis is a real, positive width for the physical set.
        assert!(s.hysteresis_width() > 10.0, "nitinol's loop is tens of kelvin, got {}", s.hysteresis_width());
    }

    #[test]
    fn the_cosine_endpoints_are_exact() {
        // A model that only approaches full transformation leaves a residual stroke error that reads as
        // miscalibration, so both ends are checked to machine precision rather than to a tolerance.
        let s = wire();
        assert_eq!(s.xi_heating(s.a_f, 0.0, 1.0), 0.0, "fully austenitic at A_f, exactly");
        assert_eq!(s.xi_heating(s.a_f + 50.0, 0.0, 1.0), 0.0);
        assert_eq!(s.xi_heating(s.a_s, 0.0, 1.0), 1.0, "untransformed at A_s, exactly");
        assert_eq!(s.xi_heating(s.a_s - 50.0, 0.0, 1.0), 1.0);

        assert_eq!(s.xi_cooling(s.m_f, 0.0, 0.0), 1.0, "fully martensitic at M_f, exactly");
        assert_eq!(s.xi_cooling(s.m_f - 50.0, 0.0, 0.0), 1.0);
        assert_eq!(s.xi_cooling(s.m_s, 0.0, 0.0), 0.0, "untransformed at M_s, exactly");
        assert_eq!(s.xi_cooling(s.m_s + 50.0, 0.0, 0.0), 0.0);

        // Midpoint of the heating window is exactly half transformed, by the cosine's symmetry.
        let mid = 0.5 * (s.a_s + s.a_f);
        assert!((s.xi_heating(mid, 0.0, 1.0) - 0.5).abs() < 1e-15);
        let midm = 0.5 * (s.m_f + s.m_s);
        assert!((s.xi_cooling(midm, 0.0, 0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn the_cooling_branch_interior_matches_its_analytic_form() {
        // The test that was missing. The endpoints are hard-coded early returns and the window midpoint sits
        // exactly where the cosine vanishes, so an inverted cosine sign passed the endpoint AND midpoint checks
        // and was caught only by monotonicity. Interior points away from the midpoint separate the two forms.
        let s = wire();
        let a = PI / (s.m_s - s.m_f);
        for &xi_start in &[0.0f64, 0.25, 0.5] {
            for k in 1..20 {
                let frac = k as f64 / 20.0;
                if (frac - 0.5).abs() < 1e-9 {
                    continue; // the blind spot itself
                }
                let t = s.m_f + (s.m_s - s.m_f) * frac;
                let want = 0.5 * (1.0 - xi_start) * (a * (t - s.m_f)).cos() + 0.5 * (1.0 + xi_start);
                let got = s.xi_cooling(t, 0.0, xi_start);
                assert!((got - want).abs() < 1e-14, "T={t} xi_start={xi_start}: {got} vs {want}");
                // And the inverted form must NOT match, or this point is another blind spot.
                let inverted = 0.5 * (1.0 + xi_start) - 0.5 * (1.0 - xi_start) * (a * (t - s.m_f)).cos();
                assert!(
                    (got - inverted).abs() > 1e-6,
                    "T={t}: this point cannot distinguish the sign, pick another"
                );
            }
        }
        // Monotone decreasing in temperature across the whole window, from 1 at M_f to xi_start at M_s.
        let mut prev = s.xi_cooling(s.m_f, 0.0, 0.0);
        assert_eq!(prev, 1.0);
        for k in 1..=500 {
            let t = s.m_f + (s.m_s - s.m_f) * k as f64 / 500.0;
            let now = s.xi_cooling(t, 0.0, 0.0);
            assert!(now <= prev + 1e-15, "cooling xi must fall as T rises, at {t}");
            prev = now;
        }
        assert!(prev.abs() < 1e-15, "and reach xi_start at M_s, got {prev}");
    }

    #[test]
    fn xi_stays_in_range_under_arbitrary_input_sequences() {
        // Including reversals mid-transformation, which is where a latch bug would push xi out of bounds.
        let s = wire();
        let mut st = s.cold(280.0);
        let mut state = 0xC0FF_EE12_3456_789Au64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as f64 / u64::MAX as f64
        };
        for _ in 0..20_000 {
            let t = 270.0 + 90.0 * next();
            let sigma = 400.0e6 * next();
            s.step(&mut st, t, sigma);
            assert!(
                (0.0..=1.0).contains(&st.xi) && st.xi.is_finite(),
                "xi left [0,1]: {} at T={t} sigma={sigma}",
                st.xi
            );
        }
    }

    #[test]
    fn a_full_cycle_closes_and_the_two_branches_are_genuinely_apart() {
        // The loop must return to its starting state, and the outbound and return paths must differ — otherwise
        // there is no hysteresis and the model has collapsed to a single-valued map.
        let s = wire();
        let mut st = s.cold(280.0);
        let start = st.xi;
        assert_eq!(start, 1.0);

        // Heat fully, recording the path.
        let mut heating_path = Vec::new();
        for k in 0..=200 {
            let t = 280.0 + 70.0 * k as f64 / 200.0;
            s.step(&mut st, t, 0.0);
            heating_path.push((t, st.xi));
        }
        assert!(st.xi < 1e-15, "a full heat must reach austenite, got {}", st.xi);

        // Cool fully back.
        let mut cooling_path = Vec::new();
        for k in 0..=200 {
            let t = 350.0 - 70.0 * k as f64 / 200.0;
            s.step(&mut st, t, 0.0);
            cooling_path.push((t, st.xi));
        }
        assert!((st.xi - start).abs() < 1e-15, "the cycle must close, got {} vs {start}", st.xi);

        // The branches are apart: at temperatures inside the loop, heating and cooling give different xi.
        let mut max_gap = 0.0f64;
        for t in [310.0, 315.0, 320.0, 325.0] {
            let h = heating_path.iter().min_by(|a, b| (a.0 - t).abs().partial_cmp(&(b.0 - t).abs()).unwrap()).unwrap().1;
            let c = cooling_path.iter().min_by(|a, b| (a.0 - t).abs().partial_cmp(&(b.0 - t).abs()).unwrap()).unwrap().1;
            max_gap = max_gap.max((h - c).abs());
        }
        assert!(max_gap > 0.3, "the two branches must be well separated, largest gap only {max_gap:.3}");

        // Monotone within each branch: heating never increases xi, cooling never decreases it.
        for w in heating_path.windows(2) {
            assert!(w[1].1 <= w[0].1 + 1e-15, "heating must not create martensite");
        }
        for w in cooling_path.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-15, "cooling must not remove martensite");
        }
    }

    #[test]
    fn a_partial_cycle_lands_strictly_inside_the_loop() {
        // The property that makes this a memory rather than a function of the inputs, and the one the reversal
        // latch exists for. Heat half way, turn around, and the state must be on neither boundary.
        let s = wire();
        let mut st = s.cold(280.0);
        // Heat to the middle of the austenite window.
        let mid = 0.5 * (s.a_s + s.a_f);
        for k in 1..=100 {
            s.step(&mut st, 280.0 + (mid - 280.0) * k as f64 / 100.0, 0.0);
        }
        let xi_turn = st.xi;
        assert!((0.05..0.95).contains(&xi_turn), "must turn around mid-transformation, xi = {xi_turn}");

        // Now cool a little. The state must move toward martensite from xi_turn, not jump to the full cooling
        // branch (which at this temperature would still read xi = 0 since T > M_s).
        for k in 1..=20 {
            s.step(&mut st, mid - 5.0 * k as f64 / 20.0, 0.0);
        }
        assert!(st.xi >= xi_turn - 1e-12, "cooling from a partial state must not reduce martensite");
        // Above M_s nothing new forms, so it holds exactly where it was: the interior of the loop.
        assert!(
            (st.xi - xi_turn).abs() < 1e-12,
            "above M_s a partial state must hold, {} vs {xi_turn}",
            st.xi
        );
        assert!(st.xi > 0.0 && st.xi < 1.0, "and it is strictly inside the loop");

        // Reheating from there must resume from the held value, not from 1.0.
        let before = st.xi;
        s.step(&mut st, mid + 1.0, 0.0);
        assert!(st.xi <= before + 1e-12, "resuming heating must not add martensite");
    }

    #[test]
    fn stress_raises_the_transformation_temperatures_linearly() {
        // Clausius-Clapeyron, checked as a slope rather than as a direction. This is why an SMA actuator's
        // timing depends on its load.
        let s = wire();
        for sigma in [0.0, 50e6, 200e6, 400e6] {
            let (m_f, m_s, a_s, a_f) = s.shifted_temperatures(sigma);
            assert!((m_f - (s.m_f + sigma / s.c_m)).abs() < 1e-12);
            assert!((m_s - (s.m_s + sigma / s.c_m)).abs() < 1e-12);
            assert!((a_s - (s.a_s + sigma / s.c_a)).abs() < 1e-12);
            assert!((a_f - (s.a_f + sigma / s.c_a)).abs() < 1e-12);
            // Ordering survives the shift, which it must for the branches to stay distinct.
            assert!(m_f <= m_s && a_s <= a_f);
        }
        // 200 MPa raises A_f by 200e6/13.8e6 = 14.5 K: a load changes the switching point by more than the
        // width of many temperature controllers' deadband.
        let (_, _, _, a_f_loaded) = s.shifted_temperatures(200e6);
        assert!((a_f_loaded - s.a_f - 14.49).abs() < 0.05, "shift was {}", a_f_loaded - s.a_f);

        // Operationally: under load, a temperature that fully transformed the free wire no longer does.
        let mut free = s.cold(280.0);
        let mut loaded = s.cold(280.0);
        for k in 1..=200 {
            let t = 280.0 + 50.0 * k as f64 / 200.0; // up to 330 K = A_f at zero stress
            s.step(&mut free, t, 0.0);
            s.step(&mut loaded, t, 200e6);
        }
        assert!(free.xi < 1e-12, "the free wire completes");
        assert!(loaded.xi > 0.2, "the loaded one does not, xi still {}", loaded.xi);
    }

    #[test]
    fn the_modulus_changes_by_about_three_times_across_the_stroke() {
        // Why a constant-E actuator model cannot be right at both ends.
        let s = wire();
        let e_hot = s.modulus(0.0);
        let e_cold = s.modulus(1.0);
        assert!((e_hot - s.e_austenite).abs() < 1e-6);
        assert!((e_cold - s.e_martensite).abs() < 1e-6);
        let ratio = e_hot / e_cold;
        assert!((2.0..4.0).contains(&ratio), "austenite should be a few times stiffer, ratio {ratio:.2}");
        // Linear in xi, and monotone, so there is no interior extremum a controller could sit on.
        for k in 0..=100 {
            let xi = k as f64 / 100.0;
            assert!((s.modulus(xi) - (e_hot + xi * (e_cold - e_hot))).abs() < 1e-6);
        }
        // The force a wire develops at fixed strain therefore differs by more than the modulus ratio alone,
        // because the transformation strain moves too.
        let f_cold = s.force(1e-6, 0.04, 1.0);
        let f_hot = s.force(1e-6, 0.04, 0.0);
        assert!(f_hot > f_cold, "the hot wire pulls harder at the same strain: {f_hot:.3} vs {f_cold:.3}");
    }

    #[test]
    fn the_work_density_is_the_reason_to_accept_all_of_this() {
        // The headline figure, as a number rather than an adjective.
        let s = wire();
        let w = s.work_density(200e6);
        assert!((w - 13.4e6).abs() < 0.1e6, "200 MPa at 6.7% should give 13.4 MJ/m^3, got {:.3e}", w);
        // Two orders above an electric motor's roughly 0.1 MJ/m^3.
        assert!(w / 0.1e6 > 100.0, "should be two orders above a motor, ratio {:.0}", w / 0.1e6);
        // Linear in stress, so the figure is only meaningful with its stress stated.
        assert!((s.work_density(400e6) / w - 2.0).abs() < 1e-12);
        assert_eq!(s.work_density(0.0), 0.0, "no load, no work");
        // And the free stroke is the transformation strain, which bounds the motion regardless of stress.
        assert!((s.free_stroke() - s.eps_l).abs() < 1e-15);
        assert!((0.03..0.09).contains(&s.free_stroke()), "a few percent, got {}", s.free_stroke());
    }

    #[test]
    fn cooling_dominates_the_cycle_time_and_no_current_can_change_that() {
        // The asymmetry, made explicit. Heating is electrical and can be forced; cooling is convective and
        // cannot, so quoting a bandwidth from the heating figure overstates the achievable rate.
        let s = wire();
        // A 100 um nitinol wire, 100 mm long: volume ~7.85e-10 m^3, rho*c_p ~ 6.45e6 J/(m^3 K).
        let volume = std::f64::consts::PI * (50e-6f64).powi(2) * 0.1;
        let thermal_mass = 6.45e6 * volume;
        let surface = std::f64::consts::PI * 100e-6 * 0.1;
        let h_still_air = 50.0; // W/(m^2 K)
        let (heat, cool) = s.cycle_time(thermal_mass, 2.0, h_still_air * surface, 50.0, 2.0);
        assert!(heat > 0.0 && cool > 0.0);
        assert!(cool > 5.0 * heat, "cooling should dominate: heat {heat:.4} s, cool {cool:.4} s");

        // Ten times the electrical power makes heating ten times faster and cooling not at all faster.
        let (heat10, cool10) = s.cycle_time(thermal_mass, 20.0, h_still_air * surface, 50.0, 2.0);
        assert!((heat / heat10 - 10.0).abs() < 1e-9, "heating scales with power");
        assert!((cool - cool10).abs() < 1e-12, "cooling is untouched by drive power");

        // Forced convection is the only lever, and it works: 10x the coefficient gives 10x faster cooling.
        let (_, cool_forced) = s.cycle_time(thermal_mass, 2.0, 10.0 * h_still_air * surface, 50.0, 2.0);
        assert!((cool / cool_forced - 10.0).abs() < 1e-9, "cooling scales with hA, and only with hA");
    }

    #[test]
    fn holding_temperature_constant_holds_the_state() {
        // A step with no temperature change must not move xi, or a controller sitting at a setpoint would see
        // the actuator drift with no input change.
        let s = wire();
        let mut st = s.cold(280.0);
        for k in 1..=100 {
            s.step(&mut st, 280.0 + 40.0 * k as f64 / 100.0, 0.0);
        }
        let held = st.xi;
        let dir = st.direction;
        // Read the temperature out first: `st` is mutably borrowed by `step`, so it cannot also be read
        // from the argument list.
        let hold_t = st.temperature;
        for _ in 0..1000 {
            s.step(&mut st, hold_t, 0.0);
        }
        assert_eq!(st.xi, held, "a zero-temperature-change step must not move the state");
        assert_eq!(st.direction, dir, "nor flip the branch");
    }
}
