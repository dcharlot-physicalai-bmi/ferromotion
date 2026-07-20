//! **Geodetic coordinate conversions** (WGS-84). The transforms every outdoor / GNSS-fused robot needs to
//! relate a GPS fix to a local metric frame: **LLA** (geodetic latitude, longitude, altitude) ↔ **ECEF**
//! (Earth-Centered Earth-Fixed Cartesian) ↔ **ENU** (a local East-North-Up tangent plane at a reference
//! point). GPS reports LLA on the WGS-84 ellipsoid; state estimators and planners want a flat local frame —
//! ENU at the mission origin — so fusing GNSS with IMU/wheel/visual odometry starts here.
//!
//! Uses the WGS-84 ellipsoid (`a`, flattening `f`); `ecef→lla` is Bowring's closed-form (sub-mm). Verified:
//! `lla→ecef→lla` round-trips to sub-millimetre; the equator/prime-meridian maps to `(a, 0, 0)` and the pole
//! to `(0, 0, b)`; a point due east of the reference has ENU `≈ (east, 0, 0)`; and ENU round-trips. Angles
//! are **radians**. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

/// WGS-84 semi-major axis (metres).
pub const WGS84_A: f64 = 6_378_137.0;
/// WGS-84 flattening.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

fn e_sq() -> f64 {
    WGS84_F * (2.0 - WGS84_F) // first eccentricity squared, e² = f(2−f)
}
fn semi_minor() -> f64 {
    WGS84_A * (1.0 - WGS84_F) // b = a(1−f)
}

/// Geodetic `(lat, lon, alt)` [rad, rad, m] → ECEF `(X, Y, Z)` [m].
pub fn lla_to_ecef(lat: f64, lon: f64, alt: f64) -> Vector3<f64> {
    let e2 = e_sq();
    let n = WGS84_A / (1.0 - e2 * lat.sin() * lat.sin()).sqrt(); // prime vertical radius
    Vector3::new(
        (n + alt) * lat.cos() * lon.cos(),
        (n + alt) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + alt) * lat.sin(),
    )
}

/// ECEF `(X, Y, Z)` → geodetic `(lat, lon, alt)` [rad, rad, m] via Bowring's method.
pub fn ecef_to_lla(ecef: &Vector3<f64>) -> (f64, f64, f64) {
    let (x, y, z) = (ecef.x, ecef.y, ecef.z);
    let a = WGS84_A;
    let b = semi_minor();
    let e2 = e_sq();
    let ep2 = (a * a - b * b) / (b * b); // second eccentricity squared
    let p = (x * x + y * y).sqrt();
    let lon = y.atan2(x);
    // Bowring's parametric-latitude seed, then one closed-form correction (sub-mm accurate)
    let theta = (z * a).atan2(p * b);
    let lat = (z + ep2 * b * theta.sin().powi(3)).atan2(p - e2 * a * theta.cos().powi(3));
    let n = a / (1.0 - e2 * lat.sin() * lat.sin()).sqrt();
    let alt = if lat.cos().abs() > 1e-9 { p / lat.cos() - n } else { z.abs() - b };
    (lat, lon, alt)
}

// rotation from ECEF to the local ENU frame at reference (lat, lon)
fn ecef_to_enu_rotation(lat: f64, lon: f64) -> Matrix3<f64> {
    let (sl, cl) = (lat.sin(), lat.cos());
    let (so, co) = (lon.sin(), lon.cos());
    Matrix3::new(
        -so, co, 0.0, // East
        -sl * co, -sl * so, cl, // North
        cl * co, cl * so, sl, // Up
    )
}

/// ECEF point → local ENU `(East, North, Up)` [m] about a reference geodetic origin `(ref_lat, ref_lon,
/// ref_alt)`.
pub fn ecef_to_enu(ecef: &Vector3<f64>, ref_lat: f64, ref_lon: f64, ref_alt: f64) -> Vector3<f64> {
    let origin = lla_to_ecef(ref_lat, ref_lon, ref_alt);
    ecef_to_enu_rotation(ref_lat, ref_lon) * (ecef - origin)
}

/// Local ENU `(East, North, Up)` → ECEF, about the same reference origin.
pub fn enu_to_ecef(enu: &Vector3<f64>, ref_lat: f64, ref_lon: f64, ref_alt: f64) -> Vector3<f64> {
    let origin = lla_to_ecef(ref_lat, ref_lon, ref_alt);
    ecef_to_enu_rotation(ref_lat, ref_lon).transpose() * enu + origin
}

/// Convenience: geodetic point → local ENU about a reference geodetic origin.
pub fn lla_to_enu(lat: f64, lon: f64, alt: f64, ref_lat: f64, ref_lon: f64, ref_alt: f64) -> Vector3<f64> {
    ecef_to_enu(&lla_to_ecef(lat, lon, alt), ref_lat, ref_lon, ref_alt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn lla_ecef_round_trips_to_sub_millimetre() {
        // THE ORACLE. A handful of geodetic points survive LLA→ECEF→LLA to sub-mm / sub-µrad.
        for &(lat_deg, lon_deg, alt) in &[(37.42, -122.08, 30.0), (-33.87, 151.21, 58.0), (0.0, 0.0, 0.0), (78.0, 15.0, 1200.0)] {
            let (lat, lon) = (lat_deg * PI / 180.0, lon_deg * PI / 180.0);
            let ecef = lla_to_ecef(lat, lon, alt);
            let (lat2, lon2, alt2) = ecef_to_lla(&ecef);
            assert!((lat2 - lat).abs() < 1e-10 && (lon2 - lon).abs() < 1e-10, "angle round trip at {lat_deg},{lon_deg}");
            assert!((alt2 - alt).abs() < 1e-3, "alt round trip: {alt2} vs {alt}");
        }
    }

    #[test]
    fn cardinal_points_map_to_the_expected_ecef() {
        // Equator & prime meridian ⇒ (a, 0, 0); north pole ⇒ (0, 0, b).
        let eq = lla_to_ecef(0.0, 0.0, 0.0);
        assert!((eq - Vector3::new(WGS84_A, 0.0, 0.0)).norm() < 1e-6, "equator/prime meridian {eq:?}");
        let pole = lla_to_ecef(PI / 2.0, 0.0, 0.0);
        assert!((pole - Vector3::new(0.0, 0.0, semi_minor())).norm() < 1e-6, "north pole {pole:?}");
    }

    #[test]
    fn a_point_due_east_has_enu_along_east() {
        // A reference near the equator and a second point a small longitude step east at the same latitude:
        // its ENU should be almost purely +East, with tiny North and a small (curvature) Up.
        let (rlat, rlon, ralt) = (0.1, 0.5, 0.0);
        let east_pt = lla_to_enu(0.1, 0.5 + 1e-4, 0.0, rlat, rlon, ralt);
        assert!(east_pt.x > 0.0, "should be east: {east_pt:?}");
        assert!(east_pt.y.abs() < 1e-3 * east_pt.x, "north component negligible");
        // the reference maps to the ENU origin
        let at_ref = lla_to_enu(rlat, rlon, ralt, rlat, rlon, ralt);
        assert!(at_ref.norm() < 1e-6, "reference is the ENU origin");
    }

    #[test]
    fn enu_round_trips_through_ecef() {
        let (rlat, rlon, ralt) = (0.65, -2.13, 100.0);
        let enu = Vector3::new(1234.0, -567.0, 89.0);
        let back = ecef_to_enu(&enu_to_ecef(&enu, rlat, rlon, ralt), rlat, rlon, ralt);
        assert!((back - enu).norm() < 1e-6, "ENU round trip {back:?} vs {enu:?}");
    }
}
