//! **Shear, partial slip, and a slip signal that arrives before the slip.**
//!
//! The gel model in [`lib`](crate) is normal-load only: an indenter presses in, the surface deforms, photometric stereo
//! renders it. Nothing in the crate mentions shear, friction or slip — and slip is the signal in-hand manipulation
//! actually runs on. A grip that is about to fail does not announce itself in the normal force; it announces itself in
//! the tangential field.
//!
//! The physics is Cattaneo-Mindlin partial slip, and its useful property is that slip is **not** a threshold event.
//! For a sphere pressed with normal load `P` and dragged with tangential load `Q < mu P`, the contact does not stick
//! until `Q = mu P` and then let go. An annulus at the periphery slips as soon as `Q > 0`, and the stuck core shrinks:
//!
//! ```text
//!   c / a = ( 1 - Q / (mu P) )^(1/3)
//! ```
//!
//! So the stuck fraction of the contact patch falls continuously to zero as the grip approaches failure, which makes it
//! a **warning** rather than an alarm. A controller watching `c/a` sees trouble while `Q` is still well inside the
//! friction cone; a controller watching only whether the object moved sees it afterwards.
//!
//! **The oracle for the traction field is that it integrates to the applied load.** `sum tau dA = Q` is not something
//! this module can fudge, and it is checked to `1e-3` relative across the load range rather than assumed from the
//! algebra.

use nalgebra::Vector3;

/// A contact under combined normal and tangential load.
#[derive(Clone, Copy, Debug)]
pub struct ShearLoad {
    /// Normal load, newtons. Must be positive for a contact to exist.
    pub normal: f64,
    /// Tangential load magnitude, newtons.
    pub tangential: f64,
    /// Coulomb friction coefficient.
    pub mu: f64,
    /// Hertzian contact radius, metres.
    pub contact_radius: f64,
}

impl ShearLoad {
    /// The load ratio `Q / (mu P)`: zero is pure normal load, one is gross slip.
    pub fn load_ratio(&self) -> Option<f64> {
        (self.normal > 0.0 && self.mu > 0.0).then(|| self.tangential.abs() / (self.mu * self.normal))
    }

    /// Hertzian peak pressure `p0 = 3P / (2 pi a^2)`.
    pub fn peak_pressure(&self) -> Option<f64> {
        (self.contact_radius > 0.0).then(|| 3.0 * self.normal / (2.0 * std::f64::consts::PI * self.contact_radius * self.contact_radius))
    }
}

/// How much of the contact patch is still stuck.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlipState {
    /// No tangential load: the whole patch is stuck.
    Stuck,
    /// A stuck core of radius `c` inside a slipping annulus. `stick_fraction` is the **area** fraction `(c/a)^2`, which
    /// is what a tactile sensor sees.
    PartialSlip { stick_radius_ratio: f64, stick_fraction: f64 },
    /// `Q >= mu P`: nothing is stuck and the contact is sliding.
    GrossSlip,
}

impl SlipState {
    /// Area fraction of the patch that is stuck: `1` when fully stuck, `0` at gross slip.
    pub fn stick_fraction(&self) -> f64 {
        match self {
            SlipState::Stuck => 1.0,
            SlipState::PartialSlip { stick_fraction, .. } => *stick_fraction,
            SlipState::GrossSlip => 0.0,
        }
    }

    pub fn is_gross(&self) -> bool {
        matches!(self, SlipState::GrossSlip)
    }
}

/// **The stick radius ratio `c/a = (1 - Q/(mu P))^(1/3)`** and the resulting slip state.
pub fn slip_state(load: &ShearLoad) -> Option<SlipState> {
    let ratio = load.load_ratio()?;
    if ratio <= 0.0 {
        return Some(SlipState::Stuck);
    }
    if ratio >= 1.0 {
        return Some(SlipState::GrossSlip);
    }
    let c_over_a = (1.0 - ratio).cbrt();
    Some(SlipState::PartialSlip { stick_radius_ratio: c_over_a, stick_fraction: c_over_a * c_over_a })
}

/// **Shear traction at radius `r`**, the Cattaneo-Mindlin distribution:
///
/// ```text
///   tau(r) = mu p0 [ sqrt(1 - r^2/a^2) - (c/a) sqrt(1 - r^2/c^2) ]   for r <= c   (stuck core)
///   tau(r) = mu p0   sqrt(1 - r^2/a^2)                               for c < r <= a (slipping annulus)
/// ```
///
/// In the annulus the traction sits exactly on the friction limit `mu p(r)`; in the core it is reduced by the second
/// term, which is what lets the core carry load without sliding.
pub fn shear_traction(load: &ShearLoad, r: f64) -> Option<f64> {
    let a = load.contact_radius;
    if a <= 0.0 || r < 0.0 || r > a {
        return None;
    }
    let p0 = load.peak_pressure()?;
    let outer = (1.0 - (r / a) * (r / a)).max(0.0).sqrt();
    let state = slip_state(load)?;
    let c_over_a = match state {
        SlipState::Stuck => 1.0,
        SlipState::PartialSlip { stick_radius_ratio, .. } => stick_radius_ratio,
        SlipState::GrossSlip => 0.0,
    };
    let c = c_over_a * a;
    if r <= c && c > 0.0 {
        let inner = (1.0 - (r / c) * (r / c)).max(0.0).sqrt();
        Some(load.mu * p0 * (outer - c_over_a * inner))
    } else {
        Some(load.mu * p0 * outer)
    }
}

/// Integrate the traction field over the patch: `sum tau(r) 2 pi r dr`. Must equal the applied tangential load, which
/// is the oracle for the whole distribution.
pub fn integrated_tangential_force(load: &ShearLoad, samples: usize) -> Option<f64> {
    let a = load.contact_radius;
    if a <= 0.0 || samples < 2 {
        return None;
    }
    let dr = a / samples as f64;
    let mut total = 0.0;
    for k in 0..samples {
        let r = (k as f64 + 0.5) * dr;
        total += shear_traction(load, r)? * 2.0 * std::f64::consts::PI * r * dr;
    }
    Some(total)
}

/// What a tactile sensor reports about an in-progress grip.
#[derive(Clone, Copy, Debug)]
pub struct SlipSignal {
    pub state: SlipState,
    /// `1 - Q/(mu P)`: how much tangential load the contact could still take, as a fraction of its capacity. Zero at
    /// gross slip.
    pub margin: f64,
    /// Direction of the tangential load in the sensor plane, if it has one.
    pub direction: Option<Vector3<f64>>,
}

impl SlipSignal {
    /// **Whether the grip should be tightened now.** True while the contact still holds but has lost more than
    /// `1 - threshold` of its stuck area — a warning, not a failure report.
    pub fn incipient(&self, stick_threshold: f64) -> bool {
        !self.state.is_gross() && self.state.stick_fraction() < stick_threshold
    }
}

/// Read the slip signal from the normal and tangential force at a contact.
pub fn slip_signal(normal: f64, tangential: Vector3<f64>, mu: f64, contact_radius: f64) -> Option<SlipSignal> {
    let load = ShearLoad { normal, tangential: tangential.norm(), mu, contact_radius };
    let state = slip_state(&load)?;
    let ratio = load.load_ratio()?;
    Some(SlipSignal {
        state,
        margin: (1.0 - ratio).max(0.0),
        direction: (tangential.norm() > 1e-12).then(|| tangential.normalize()),
    })
}

/// **Tangential surface displacement at radius `r`**, normalised by the rigid-core displacement.
///
/// The stuck core moves rigidly, so the field is flat at `1` for `r <= c` and falls through the slipping annulus. This is
/// the shape a gel's tracked markers show, and the flat-then-falling profile is how the stuck radius is measured from an
/// image rather than computed from forces.
pub fn tangential_displacement_profile(load: &ShearLoad, r: f64) -> Option<f64> {
    let a = load.contact_radius;
    if a <= 0.0 || r < 0.0 || r > a {
        return None;
    }
    let state = slip_state(load)?;
    match state {
        SlipState::Stuck => Some(1.0),
        SlipState::GrossSlip => Some(0.0),
        SlipState::PartialSlip { stick_radius_ratio, .. } => {
            let c = stick_radius_ratio * a;
            if r <= c {
                Some(1.0)
            } else {
                // in the annulus the surface has slipped, so its displacement relative to the core falls off toward the
                // edge; the Hertzian profile gives the shape
                let t = ((a * a - r * r) / (a * a - c * c)).max(0.0).sqrt();
                Some(t)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(q_over_mup: f64) -> ShearLoad {
        let (p, mu, a) = (5.0, 0.6, 4e-3);
        ShearLoad { normal: p, tangential: q_over_mup * mu * p, mu, contact_radius: a }
    }

    /// The endpoints must be exact: fully stuck at zero tangential load, gross slip at the friction limit.
    #[test]
    fn the_endpoints_are_exact() {
        assert_eq!(slip_state(&load(0.0)).unwrap(), SlipState::Stuck);
        assert_eq!(slip_state(&load(1.0)).unwrap(), SlipState::GrossSlip);
        assert_eq!(slip_state(&load(1.5)).unwrap(), SlipState::GrossSlip);
        assert_eq!(slip_state(&load(0.0)).unwrap().stick_fraction(), 1.0);
        assert_eq!(slip_state(&load(1.0)).unwrap().stick_fraction(), 0.0);
    }

    /// **The warning property**: the stuck area falls continuously as the grip approaches failure, so a controller sees
    /// trouble coming instead of discovering it.
    #[test]
    fn the_stuck_area_shrinks_long_before_gross_slip() {
        eprintln!("{:>10}  {:>10}  {:>14}  {:>10}", "Q/(mu P)", "c/a", "stuck area", "margin");
        let mut prev = 2.0;
        for ratio in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let l = load(ratio);
            let s = slip_state(&l).unwrap();
            let c_over_a = match s {
                SlipState::Stuck => 1.0,
                SlipState::PartialSlip { stick_radius_ratio, .. } => stick_radius_ratio,
                SlipState::GrossSlip => 0.0,
            };
            eprintln!("{ratio:>10.2}  {c_over_a:>10.4}  {:>14.4}  {:>10.4}", s.stick_fraction(), 1.0 - ratio);
            assert!(s.stick_fraction() < prev, "the stuck area is strictly decreasing");
            prev = s.stick_fraction();
        }
        // the useful number: half the stuck AREA is gone well before the friction limit
        let half_area = slip_state(&load(0.65)).unwrap().stick_fraction();
        eprintln!("\n   at 65% of the friction limit the stuck area is already down to {:.1}%", 100.0 * half_area);
        assert!(half_area < 0.55, "more than 45% of the patch is already slipping at Q = 0.65 mu P");
    }

    /// **The oracle: the traction field integrates to the applied tangential load.** This is the check the algebra
    /// cannot talk its way out of.
    #[test]
    fn the_traction_field_integrates_to_the_applied_load() {
        eprintln!("{:>10}  {:>14}  {:>14}  {:>10}", "Q/(mu P)", "applied Q (N)", "integral (N)", "rel error");
        for ratio in [0.05, 0.2, 0.5, 0.8, 0.95] {
            let l = load(ratio);
            let integral = integrated_tangential_force(&l, 20_000).unwrap();
            let rel = (integral - l.tangential).abs() / l.tangential;
            eprintln!("{ratio:>10.2}  {:>14.5}  {integral:>14.5}  {rel:>10.2e}", l.tangential);
            assert!(rel < 1e-3, "the distribution must carry exactly the applied load: {rel:.3e}");
        }
        // and at gross slip the whole patch is at the friction limit, so the integral is mu P
        let gross = load(1.0);
        let integral = integrated_tangential_force(&gross, 20_000).unwrap();
        eprintln!("   gross slip: integral {integral:.5} N against mu P = {:.5} N", gross.mu * gross.normal);
        assert!((integral - gross.mu * gross.normal).abs() / (gross.mu * gross.normal) < 1e-3);
    }

    /// In the slipping annulus the traction sits exactly on the friction limit; in the stuck core it is below it. That
    /// distinction is the whole content of partial slip.
    #[test]
    fn the_annulus_is_at_the_friction_limit_and_the_core_is_not() {
        let l = load(0.5);
        let p0 = l.peak_pressure().unwrap();
        let SlipState::PartialSlip { stick_radius_ratio, .. } = slip_state(&l).unwrap() else { panic!("partial slip") };
        let c = stick_radius_ratio * l.contact_radius;
        eprintln!("Q = 0.5 mu P: c/a = {stick_radius_ratio:.4}, c = {:.4} mm", c * 1e3);
        for frac in [0.2, 0.6, 0.95] {
            let r = frac * c;
            let tau = shear_traction(&l, r).unwrap();
            let limit = l.mu * p0 * (1.0 - (r / l.contact_radius).powi(2)).sqrt();
            eprintln!("   core   r/c = {frac:.2}: tau = {tau:.1} Pa, friction limit {limit:.1} Pa, ratio {:.3}", tau / limit);
            assert!(tau < limit * (1.0 - 1e-9), "the core carries less than the limit");
        }
        for frac in [1.05, 1.3, 1.8] {
            let r = (frac * c).min(l.contact_radius * 0.999);
            let tau = shear_traction(&l, r).unwrap();
            let limit = l.mu * p0 * (1.0 - (r / l.contact_radius).powi(2)).sqrt();
            eprintln!("   annulus r/c = {frac:.2}: tau = {tau:.1} Pa, friction limit {limit:.1} Pa, ratio {:.6}", tau / limit);
            assert!((tau - limit).abs() < 1e-6 * limit.max(1.0), "the annulus is exactly at the limit");
        }
    }

    /// The displacement profile is flat over the stuck core and falls through the annulus — the shape a gel's markers
    /// show, which is how `c` gets measured from an image rather than from forces.
    #[test]
    fn the_displacement_profile_is_flat_then_falling() {
        let l = load(0.4);
        let SlipState::PartialSlip { stick_radius_ratio, .. } = slip_state(&l).unwrap() else { panic!() };
        let a = l.contact_radius;
        eprintln!("normalised tangential displacement across the patch (c/a = {stick_radius_ratio:.4}):");
        // sample in r/a so every point is distinct and inside the patch; sampling in r/c let two annulus points clamp
        // to the edge and tie, which is not a property of the profile
        let mut core_value: Option<f64> = None;
        let mut prev_annulus = 2.0;
        for frac in [0.0, 0.3, 0.6, 0.83, 0.87, 0.92, 0.96, 1.0] {
            let r = frac * a;
            let u = tangential_displacement_profile(&l, r).unwrap();
            let region = if frac <= stick_radius_ratio { "core   " } else { "annulus" };
            eprintln!("   r/a = {frac:.2} ({region}): u = {u:.4}");
            if frac <= stick_radius_ratio {
                if let Some(prev) = core_value {
                    assert!((u - prev).abs() < 1e-12, "the core moves rigidly: {u} vs {prev}");
                }
                core_value = Some(u);
                assert!((u - 1.0).abs() < 1e-12, "and its displacement is the reference value 1");
            } else {
                assert!(u < prev_annulus, "the annulus displacement falls toward the edge: {u} after {prev_annulus}");
                prev_annulus = u;
            }
        }
        assert!(tangential_displacement_profile(&l, a).unwrap().abs() < 1e-12, "and reaches zero at the rim");
    }

    /// **The signal fires before the failure.** An incipient-slip warning at a stuck-area threshold triggers at a
    /// strictly lower tangential load than gross slip, and the gap is the reaction time a controller gets.
    #[test]
    fn the_warning_arrives_before_the_slip() {
        let (p, mu, a) = (5.0, 0.6, 4e-3);
        let threshold = 0.5; // warn once half the stuck area is gone
        let mut warn_at = None;
        let mut gross_at = None;
        for k in 0..=1000 {
            let q = mu * p * k as f64 / 1000.0;
            let sig = slip_signal(p, Vector3::new(q, 0.0, 0.0), mu, a).unwrap();
            if warn_at.is_none() && sig.incipient(threshold) {
                warn_at = Some(q);
            }
            if gross_at.is_none() && sig.state.is_gross() {
                gross_at = Some(q);
            }
        }
        let (w, g) = (warn_at.expect("the warning fires"), gross_at.expect("gross slip is reached"));
        eprintln!("friction capacity mu P = {:.3} N", mu * p);
        eprintln!("   incipient warning at Q = {w:.3} N ({:.0}% of capacity)", 100.0 * w / (mu * p));
        eprintln!("   gross slip        at Q = {g:.3} N ({:.0}% of capacity)", 100.0 * g / (mu * p));
        eprintln!("   warning margin: {:.3} N of tangential load, {:.0}% of capacity", g - w, 100.0 * (g - w) / (mu * p));
        assert!(w < g, "the warning must precede the failure");
        assert!((g - w) / (mu * p) > 0.2, "and by a usable margin: {:.3}", (g - w) / (mu * p));
    }

    /// Degenerate input is refused rather than answered.
    #[test]
    fn degenerate_input_is_refused() {
        assert!(slip_state(&ShearLoad { normal: 0.0, tangential: 1.0, mu: 0.5, contact_radius: 1e-3 }).is_none(), "no normal load, no contact");
        assert!(slip_state(&ShearLoad { normal: 1.0, tangential: 1.0, mu: 0.0, contact_radius: 1e-3 }).is_none(), "no friction, no stick");
        let l = load(0.5);
        assert!(shear_traction(&l, -1e-3).is_none(), "a negative radius");
        assert!(shear_traction(&l, 2.0 * l.contact_radius).is_none(), "outside the patch");
        assert!(integrated_tangential_force(&ShearLoad { contact_radius: 0.0, ..l }, 100).is_none());
        // a pure normal load has no tangential direction to report
        assert!(slip_signal(5.0, Vector3::zeros(), 0.6, 4e-3).unwrap().direction.is_none());
    }
}
