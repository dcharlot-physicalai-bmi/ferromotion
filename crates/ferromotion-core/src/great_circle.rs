//! **Great-circle navigation** on a sphere — the surface-navigation primitives for drones, marine, and
//! aerial waypoint following: **haversine distance** between two lat/lon points, the **initial bearing** to
//! steer toward a target, the **destination** reached by travelling a bearing for a distance, and the
//! **cross-track distance** (how far off the intended great-circle course a vehicle has drifted). Unlike the
//! ellipsoid *coordinate* conversions in [`crate::geodetic`] (LLA↔ECEF↔ENU), this is spherical *navigation*
//! math — the equations a waypoint follower or route planner runs directly on latitude/longitude.
//!
//! Angles are **radians**; distances are in the same units as the `radius` argument (pass the Earth mean
//! radius `≈ 6 371 000 m`). Verified: a quarter-equator is `πR/2` and antipodes are `πR`; bearings to due-
//! north/east are `0`/`π⁄2`; `destination` inverts `haversine`+`bearing` (a round-trip returns the input);
//! and a point on the course has zero cross-track error. Pure Rust → WASM-clean.

use std::f64::consts::PI;

/// Mean Earth radius (metres).
pub const EARTH_RADIUS: f64 = 6_371_000.0;

/// Great-circle (haversine) distance between `(lat1, lon1)` and `(lat2, lon2)`.
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64, radius: f64) -> f64 {
    let (dlat, dlon) = (lat2 - lat1, lon2 - lon1);
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * radius * a.sqrt().clamp(0.0, 1.0).asin()
}

/// Initial bearing (radians, clockwise from north, in `(−π, π]`) along the great circle from `(lat1, lon1)`
/// toward `(lat2, lon2)`.
pub fn initial_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlon = lon2 - lon1;
    let y = dlon.sin() * lat2.cos();
    let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    y.atan2(x)
}

/// The `(lat, lon)` reached by travelling `distance` along the great circle from `(lat1, lon1)` on the given
/// initial `bearing`.
pub fn destination(lat1: f64, lon1: f64, bearing: f64, distance: f64, radius: f64) -> (f64, f64) {
    let ad = distance / radius; // angular distance
    let lat2 = (lat1.sin() * ad.cos() + lat1.cos() * ad.sin() * bearing.cos()).clamp(-1.0, 1.0).asin();
    let lon2 = lon1 + (bearing.sin() * ad.sin() * lat1.cos()).atan2(ad.cos() - lat1.sin() * lat2.sin());
    // wrap longitude to (−π, π]
    let lon2 = (lon2 + 3.0 * PI).rem_euclid(2.0 * PI) - PI;
    (lat2, lon2)
}

/// Signed cross-track distance of point `(lat, lon)` from the great circle through `start → end` (positive =
/// right of the course). Its magnitude is how far off-course the point lies.
pub fn cross_track_distance(lat: f64, lon: f64, start: (f64, f64), end: (f64, f64), radius: f64) -> f64 {
    let d13 = haversine_distance(start.0, start.1, lat, lon, radius) / radius; // angular
    let theta13 = initial_bearing(start.0, start.1, lat, lon);
    let theta12 = initial_bearing(start.0, start.1, end.0, end.1);
    radius * ((d13.sin() * (theta13 - theta12).sin()).clamp(-1.0, 1.0)).asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f64 = 1.0; // unit sphere for clean checks

    fn deg(d: f64) -> f64 {
        d * PI / 180.0
    }

    #[test]
    fn distances_match_known_arcs() {
        // THE ORACLE. A quarter of the equator is πR/2; antipodal points are πR apart.
        assert!((haversine_distance(0.0, 0.0, 0.0, deg(90.0), R) - PI / 2.0).abs() < 1e-12, "quarter equator");
        assert!((haversine_distance(deg(90.0), 0.0, deg(-90.0), 0.0, R) - PI).abs() < 1e-12, "pole to pole = πR");
        assert!(haversine_distance(deg(37.0), deg(-122.0), deg(37.0), deg(-122.0), R).abs() < 1e-12, "same point = 0");
    }

    #[test]
    fn bearings_point_the_right_way() {
        // Due north ⇒ bearing 0 ; due east from the equator ⇒ bearing π/2.
        assert!(initial_bearing(0.0, 0.0, deg(10.0), 0.0).abs() < 1e-9, "north is 0");
        assert!((initial_bearing(0.0, 0.0, 0.0, deg(10.0)) - PI / 2.0).abs() < 1e-9, "east is π/2");
    }

    #[test]
    fn destination_inverts_distance_and_bearing() {
        // THE ROUND TRIP. From a start, head on bearing β for distance d; the arrival is exactly d away and
        // on bearing β.
        let (lat1, lon1) = (deg(48.0), deg(2.0));
        let (bearing, dist) = (deg(70.0), 0.3);
        let (lat2, lon2) = destination(lat1, lon1, bearing, dist, R);
        assert!((haversine_distance(lat1, lon1, lat2, lon2, R) - dist).abs() < 1e-9, "arrival distance = d");
        assert!((initial_bearing(lat1, lon1, lat2, lon2) - bearing).abs() < 1e-9, "arrival bearing = β");
    }

    #[test]
    fn a_point_on_the_course_has_zero_cross_track() {
        // A waypoint generated ALONG the great circle from start toward end lies on the course.
        let start = (deg(10.0), deg(20.0));
        let end = (deg(40.0), deg(60.0));
        let bearing = initial_bearing(start.0, start.1, end.0, end.1);
        let on = destination(start.0, start.1, bearing, 0.2, R);
        let xte = cross_track_distance(on.0, on.1, start, end, R);
        assert!(xte.abs() < 1e-9, "on-course point should have zero cross-track: {xte}");
        // a point pushed off to the side has non-zero (and sign-consistent) cross-track
        let off = destination(on.0, on.1, bearing + PI / 2.0, 0.05, R);
        assert!(cross_track_distance(off.0, off.1, start, end, R).abs() > 0.01, "off-course point deviates");
    }
}
