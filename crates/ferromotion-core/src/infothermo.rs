//! **Information and thermodynamics of embodiment** — the floors under a physical agent, in joules and
//! in bits.
//!
//! Every other module here computes what a machine *does*. This one computes what it *cannot* do,
//! whatever the algorithm: how little energy a decision can cost, how much work an observation is worth,
//! and how much a feedback loop can suppress a disturbance before it must amplify one elsewhere. These
//! are conservation statements, not engineering targets, so they are the right place to look when a
//! design promises something that sounds too good.
//!
//! Three floors, each independently checkable:
//!
//! * **Landauer**: erasing a bit costs at least `kT ln 2` — about `2.9 zJ` at room temperature. Any
//!   irreversible decision that discards information pays this, which is what makes joules-per-decision
//!   a physical quantity rather than an implementation detail.
//! * **The feedback second law** (Sagawa and Ueda): a measurement worth `I` bits licenses at most
//!   `kT I ln 2` of extractable work. Information is not free energy, but it converts at a fixed rate,
//!   so a controller that claims to extract more than its sensors justify is wrong somewhere.
//! * **The Bode sensitivity integral**: for a stable loop that rolls off, `∫₀^∞ ln|S(jω)| dω = 0`.
//!   Feedback does not remove sensitivity, it *moves* it. Push disturbance rejection down in one band
//!   and it rises in another — the waterbed. This is the closest thing control theory has to a
//!   conservation law, and it is why "just tune the gains higher" stops working.

use nalgebra::Complex;

/// Boltzmann's constant, joules per kelvin.
pub const BOLTZMANN: f64 = 1.380649e-23;

/// **Landauer's bound**: the least energy that erasing `bits` of information can dissipate at
/// temperature `temperature` kelvin, `bits · kT ln 2` joules. Returns `0` for a non-positive count and
/// `None` for a non-physical temperature.
pub fn landauer_energy(bits: f64, temperature: f64) -> Option<f64> {
    if temperature <= 0.0 {
        return None;
    }
    Some(bits.max(0.0) * BOLTZMANN * temperature * std::f64::consts::LN_2)
}

/// **The feedback second law**: the most work extractable from a measurement carrying `bits` of mutual
/// information with the system, `kT · bits · ln 2` joules. The same constant as Landauer's bound, which
/// is the point — acquiring and discarding information are the two directions of one exchange rate.
pub fn max_extractable_work(bits: f64, temperature: f64) -> Option<f64> {
    landauer_energy(bits, temperature)
}

/// How many bits a decision may irreversibly discard on a given energy budget at a given temperature.
/// The inverse of [`landauer_energy`], and the honest form of "how much can this robot decide per
/// joule" — an absolute ceiling no processor design can beat.
pub fn bits_affordable(joules: f64, temperature: f64) -> Option<f64> {
    let per_bit = landauer_energy(1.0, temperature)?;
    Some((joules.max(0.0) / per_bit).max(0.0))
}

/// The **Bode sensitivity integral** of a feedback loop, `∫₀^Ω ln|S(jω)| dω` with
/// `S = 1/(1 + L)`, evaluated numerically out to `omega_max` on a log-spaced grid.
///
/// For a stable loop whose gain rolls off at least as fast as `1/ω²`, the exact value over the whole
/// axis is zero: every decibel of disturbance rejection bought at one frequency is paid back at
/// another. With right-half-plane poles the value is `π Σ Re(pᵢ)` and strictly positive, meaning an
/// unstable plant must be *more* sensitive on balance than an open loop — instability is not free.
///
/// `l` is the open-loop transfer function evaluated at `jω`. The grid is logarithmic because `ln|S|` has
/// its structure near crossover, and a linear grid wastes samples where nothing happens.
pub fn bode_sensitivity_integral(l: &dyn Fn(f64) -> Complex<f64>, omega_max: f64, samples: usize) -> f64 {
    // trapezoid on a log grid from a decade below the smallest interesting frequency
    let (lo, hi) = (1e-4_f64, omega_max.max(1e-3));
    let n = samples.max(16);
    let mut total = 0.0;
    let mut prev_w = lo;
    let mut prev_f = ln_sensitivity(l, lo);
    for k in 1..=n {
        let w = lo * (hi / lo).powf(k as f64 / n as f64);
        let f = ln_sensitivity(l, w);
        total += 0.5 * (f + prev_f) * (w - prev_w);
        prev_w = w;
        prev_f = f;
    }
    total
}

fn ln_sensitivity(l: &dyn Fn(f64) -> Complex<f64>, w: f64) -> f64 {
    let s = Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) + l(w));
    let m = s.norm();
    if m <= 1e-300 {
        -690.0 // ln of an underflow; the integrand is integrable but the sample is not representable
    } else {
        m.ln()
    }
}

/// Entropy produced when `dissipated` joules are lost to a bath at `temperature` kelvin, in joules per
/// kelvin. Non-negative for any physical dissipation, and the bookkeeping that connects a contact
/// impact, a damper, or a resistor to the second law.
pub fn entropy_production(dissipated: f64, temperature: f64) -> Option<f64> {
    if temperature <= 0.0 {
        return None;
    }
    Some(dissipated.max(0.0) / temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Landauer's bound at room temperature is about 2.9 zeptojoules per bit — the number worth knowing,
    /// because it sets the scale everything else is measured against.
    #[test]
    fn landauer_bound_at_room_temperature() {
        let e = landauer_energy(1.0, 300.0).unwrap();
        eprintln!("erasing one bit at 300 K costs at least {:.3e} J ({:.2} zJ)", e, e * 1e21);
        assert!((e - 2.871e-21).abs() < 1e-24, "expected ~2.87e-21 J, got {e}");
        // linear in the bit count, zero for nothing erased, and refuses a non-physical temperature
        assert!((landauer_energy(1000.0, 300.0).unwrap() - 1000.0 * e).abs() < 1e-30);
        assert_eq!(landauer_energy(-5.0, 300.0).unwrap(), 0.0);
        assert!(landauer_energy(1.0, 0.0).is_none() && landauer_energy(1.0, -3.0).is_none());
    }

    /// The exchange rate runs both ways: the work a measurement licenses is exactly the energy erasing
    /// that measurement would cost. A round trip nets nothing, which is the second law holding.
    #[test]
    fn information_and_work_convert_at_the_same_rate() {
        let (bits, t) = (12.0, 293.0);
        let w = max_extractable_work(bits, t).unwrap();
        let e = landauer_energy(bits, t).unwrap();
        assert!((w - e).abs() < 1e-30, "acquiring and erasing must be the same rate: {w} vs {e}");
        // and the affordable-bits inverse is consistent
        let b = bits_affordable(e, t).unwrap();
        assert!((b - bits).abs() < 1e-9, "round trip should return {bits} bits, got {b}");
    }

    /// A joule is an enormous number of bits at room temperature, which is the honest framing of the
    /// Landauer floor: it is not what limits real robots, and knowing the gap is the useful part.
    #[test]
    fn a_joule_buys_an_astronomical_number_of_bits() {
        let b = bits_affordable(1.0, 300.0).unwrap();
        eprintln!("one joule at 300 K affords {:.3e} bit erasures — real hardware runs far above this floor", b);
        assert!(b > 1e20 && b < 1e21, "expected ~3.5e20 bits, got {b}");
    }

    /// **The waterbed, measured.** A stable loop that rolls off has zero net log-sensitivity: the
    /// integral is zero, so rejection gained in one band is lost in another. Raising the gain moves the
    /// sensitivity around and cannot make the total negative.
    #[test]
    fn bode_integral_is_zero_for_a_stable_rolled_off_loop() {
        for &k in &[1.0_f64, 5.0, 20.0] {
            // L(s) = k / ((s+1)(s+2)): stable, relative degree 2
            let l = move |w: f64| {
                let s = Complex::new(0.0, w);
                Complex::new(k, 0.0) / ((s + Complex::new(1.0, 0.0)) * (s + Complex::new(2.0, 0.0)))
            };
            let integral = bode_sensitivity_integral(&l, 1e4, 200_000);
            eprintln!("stable loop, gain {k:>4}: integral of ln|S| = {integral:+.4} (theory 0)");
            assert!(integral.abs() < 0.05, "a stable rolled-off loop must integrate to zero, got {integral}");
        }
    }

    /// An unstable plant is *more* sensitive on balance: the integral becomes `π Σ Re(p)` over the
    /// right-half-plane poles, strictly positive. Instability has to be paid for somewhere, and this is
    /// where.
    #[test]
    fn an_unstable_pole_makes_the_integral_positive() {
        // L(s) = k(s+3)/((s−1)(s+2)(s+5)): one RHP pole at s = +1, so theory says integral = pi
        let k = 30.0;
        let l = move |w: f64| {
            let s = Complex::new(0.0, w);
            let num = Complex::new(k, 0.0) * (s + Complex::new(3.0, 0.0));
            let den = (s - Complex::new(1.0, 0.0)) * (s + Complex::new(2.0, 0.0)) * (s + Complex::new(5.0, 0.0));
            num / den
        };
        let integral = bode_sensitivity_integral(&l, 1e4, 200_000);
        eprintln!("one unstable pole at s=+1: integral of ln|S| = {integral:+.4} (theory pi = {:.4})", std::f64::consts::PI);
        assert!(integral > 0.5, "an unstable plant must integrate strictly positive, got {integral}");
        assert!((integral - std::f64::consts::PI).abs() < 0.5, "should be near pi, got {integral}");
    }

    /// Dissipation produces entropy, never destroys it — the ledger that ties a contact impact or a
    /// damper to the second law.
    #[test]
    fn dissipation_produces_entropy() {
        let s = entropy_production(3.56, 300.0).unwrap(); // the 3.56 J a quadruped foot strike removes
        eprintln!("a foot strike dissipating 3.56 J at 300 K produces {s:.4e} J/K of entropy");
        assert!(s > 0.0 && (s - 3.56 / 300.0).abs() < 1e-12);
        assert_eq!(entropy_production(-1.0, 300.0).unwrap(), 0.0, "dissipation cannot be negative");
        assert!(entropy_production(1.0, 0.0).is_none());
    }
}
