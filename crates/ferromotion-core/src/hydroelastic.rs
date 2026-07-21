//! **Hydroelastic (pressure-field) contact** — smooth, distributed contact forces, the compliant-contact
//! model from Drake. Point-contact models ([`crate::gjk`]/[`crate::epa`] give one point + normal + depth) make
//! the force a discontinuous function of configuration: as the single witness point jumps across a face the
//! force chatters, which wrecks stiff integrators and contact-rich planning. The hydroelastic model instead
//! assigns each soft body a scalar **pressure field** `p(x)` that is zero on its surface and rises into its
//! interior; where two bodies overlap, the contact surface is the **equi-pressure isosurface** `p_A = p_B`,
//! and the net wrench is the *integral of pressure over that surface*. Because the surface and the integrand
//! move continuously with configuration, so does the force — no chatter.
//!
//! This module implements the workhorse case with an exact analytic oracle: a **soft sphere with a linear
//! pressure field pressing into a rigid plane**. With `p(r) = E·(1 − r/R)` (max `E` at the center, zero at the
//! surface), the contact patch is the disk of radius `a = √(R² − c²)` (`c = R − δ` the center height above the
//! plane at penetration `δ`), and the normal force integrates in closed form:
//! `F = 2πE[ a²/2 − (R³ − c³)/(3R) ]`. Two equal soft spheres reduce to this: their equi-pressure surface is
//! the **midplane**, so each presses against it exactly like the rigid-plane case with `c = d/2`. Verified:
//! numerical polar integration matches the closed form; the force is continuous and vanishes at first contact
//! (`δ → 0`); the pressure centroid sits on the contact axis; and the equal-sphere reduction agrees with the
//! plane oracle. Pure Rust → WASM-clean. Unequal compliant-compliant pairs need the general (curved)
//! isosurface and are out of this module's analytic scope.

use core::f64::consts::PI;

/// The result of a hydroelastic contact query.
#[derive(Clone, Copy, Debug)]
pub struct HydroContact {
    /// Net normal force (along the contact normal), = ∫ pressure dA over the patch.
    pub force: f64,
    /// Radius of the (disk) contact patch.
    pub patch_radius: f64,
    /// Signed distance of the pressure centroid from the contact axis (0 by symmetry for sphere/plane).
    pub centroid_offset: f64,
}

/// Linear pressure field of a soft ball: `p(r) = E·(1 − r/R)` for `r ≤ R`, else 0. `r` = distance from the
/// ball center, `radius` = `R`, `modulus` = `E` (peak pressure at the center).
#[inline]
pub fn linear_pressure(r: f64, radius: f64, modulus: f64) -> f64 {
    (modulus * (1.0 - r / radius)).max(0.0)
}

/// Closed-form normal force for a soft sphere (linear pressure field, radius `R`, modulus `E`) pressing depth
/// `penetration` (δ) into a rigid plane. Returns 0 for non-positive penetration.
pub fn sphere_plane_force_closed_form(radius: f64, penetration: f64, modulus: f64) -> f64 {
    if penetration <= 0.0 {
        return 0.0;
    }
    let r = radius;
    let c = (r - penetration).max(0.0); // center height above the plane
    let a2 = (r * r - c * c).max(0.0); // squared patch radius, = R² − c²
    // F = 2πE [ a²/2 − (R³ − c³)/(3R) ]
    2.0 * PI * modulus * (a2 / 2.0 - (r * r * r - c * c * c) / (3.0 * r))
}

/// Numerically integrate the pressure over the contact disk for a soft sphere on a rigid plane (polar
/// quadrature, `n` radial × `n` angular samples). Used to *verify* the closed form; also returns the patch
/// radius and (on-axis) centroid.
pub fn sphere_plane_contact(radius: f64, penetration: f64, modulus: f64, n: usize) -> HydroContact {
    if penetration <= 0.0 {
        return HydroContact { force: 0.0, patch_radius: 0.0, centroid_offset: 0.0 };
    }
    let r = radius;
    let c = (r - penetration).max(0.0);
    let a = (r * r - c * c).max(0.0).sqrt();
    let mut force = 0.0;
    let mut moment_x = 0.0;
    let dr = a / n as f64;
    let dth = 2.0 * PI / n as f64;
    for i in 0..n {
        let s = (i as f64 + 0.5) * dr; // radial distance on the plane from the axis
        let dist_from_center = (s * s + c * c).sqrt();
        let p = linear_pressure(dist_from_center, r, modulus);
        let ring_area = s * dr * dth; // area element in polar coords
        for j in 0..n {
            let th = (j as f64 + 0.5) * dth;
            let da = ring_area;
            force += p * da;
            moment_x += p * da * (s * th.cos());
        }
    }
    let centroid_offset = if force > 0.0 { moment_x / force } else { 0.0 };
    HydroContact { force, patch_radius: a, centroid_offset }
}

/// Two **equal** soft spheres (same radius `R`, same modulus `E`) whose centers are `center_distance` apart.
/// Their equi-pressure surface is the midplane, so each presses against it exactly like the rigid-plane case
/// with center height `c = d/2` and penetration `R − d/2`. Returns `None` when they are not in contact.
pub fn equal_spheres_contact(radius: f64, modulus: f64, center_distance: f64) -> Option<HydroContact> {
    let half = center_distance / 2.0;
    if half >= radius {
        return None; // no overlap
    }
    let penetration = radius - half;
    Some(sphere_plane_contact(radius, penetration, modulus, 200))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_integration_matches_the_closed_form() {
        // THE ORACLE. Polar quadrature of the pressure field must converge to the analytic integral.
        let (r, e) = (1.0, 100.0);
        for &delta in &[0.05, 0.2, 0.5, 0.9] {
            let num = sphere_plane_contact(r, delta, e, 400).force;
            let exact = sphere_plane_force_closed_form(r, delta, e);
            let rel = (num - exact).abs() / exact.max(1e-9);
            assert!(rel < 1e-2, "δ={delta}: numerical {num} vs closed form {exact} (rel {rel:.4})");
        }
    }

    #[test]
    fn force_is_continuous_and_vanishes_at_first_contact() {
        // THE HEADLINE. The whole point of hydroelastic contact: force → 0 smoothly as penetration → 0 (no
        // jump), unlike a point model that can snap on. Also monotonically increasing with depth.
        let (r, e) = (1.0, 100.0);
        assert_eq!(sphere_plane_force_closed_form(r, 0.0, e), 0.0, "no force at zero penetration");
        assert_eq!(sphere_plane_force_closed_form(r, -0.1, e), 0.0, "no force when separated");
        let f_tiny = sphere_plane_force_closed_form(r, 1e-4, e);
        assert!(f_tiny > 0.0 && f_tiny < 1e-3, "force turns on continuously from zero: {f_tiny}");
        // monotonic in penetration
        let mut prev = 0.0;
        for k in 1..=20 {
            let f = sphere_plane_force_closed_form(r, k as f64 * 0.05, e);
            assert!(f > prev, "force must increase with depth");
            prev = f;
        }
    }

    #[test]
    fn pressure_is_zero_at_the_surface_and_peaks_at_the_center() {
        assert!((linear_pressure(1.0, 1.0, 50.0)).abs() < 1e-12, "zero pressure at the surface r=R");
        assert!((linear_pressure(0.0, 1.0, 50.0) - 50.0).abs() < 1e-12, "peak modulus at the center");
        assert_eq!(linear_pressure(1.5, 1.0, 50.0), 0.0, "no pressure outside the ball");
    }

    #[test]
    fn the_pressure_centroid_lies_on_the_contact_axis() {
        // By radial symmetry the net pressure centroid is on the axis (offset ~0).
        let hc = sphere_plane_contact(1.0, 0.4, 100.0, 200);
        assert!(hc.centroid_offset.abs() < 1e-6, "centroid should be on-axis: {}", hc.centroid_offset);
        assert!(hc.patch_radius > 0.0, "there should be a finite contact patch");
    }

    #[test]
    fn equal_spheres_reduce_to_the_rigid_plane_oracle() {
        // THE DISCRIMINATOR (the hydroelastic idea). Two equal soft spheres share a midplane contact surface,
        // so their force equals the single-sphere-into-rigid-plane force at half the center distance.
        let (r, e) = (1.0, 100.0);
        let d = 1.5; // centers 1.5 apart → overlap 0.5, each penetrates the midplane by R − d/2 = 0.25
        let hc = equal_spheres_contact(r, e, d).expect("in contact");
        let expected = sphere_plane_force_closed_form(r, r - d / 2.0, e);
        let rel = (hc.force - expected).abs() / expected;
        assert!(rel < 1e-2, "equal-sphere force {} vs plane oracle {} (rel {rel:.4})", hc.force, expected);
        assert!(equal_spheres_contact(r, e, 2.1).is_none(), "separated spheres are not in contact");
    }
}
