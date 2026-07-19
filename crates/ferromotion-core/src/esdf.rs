//! **ESDF — Euclidean Signed Distance Field** from occupancy (Oleynikova et al. *Voxblox*, IROS 2017;
//! Han et al. *FIESTA* 2019) — the distance field a robot *measures from the world*, as opposed to the
//! analytic primitive fields in [`crate::sdf`]. Given a set of occupied points (a depth/LiDAR scan, an
//! occupancy map), the ESDF returns the distance and gradient to the nearest obstacle at any query — the
//! substrate reactive controllers ([`crate::Chomp`], RMPflow) consume to plan around real, sensed
//! environments. Built directly on the [`crate::KdTree`] spatial index: the distance *is* the
//! nearest-neighbour query, so this is a thin, exact wrapper.
//!
//! Verified: the field matches the brute-force nearest-occupied distance, is zero on obstacles and grows
//! away from them, its gradient is the unit outward direction, and it recovers the analytic distance for a
//! planar obstacle. Pure `nalgebra` → WASM-clean.

use crate::spatial::KdTree;
use nalgebra::Vector3;

/// A Euclidean distance field over a set of occupied points.
#[derive(Clone, Debug)]
pub struct Esdf {
    tree: KdTree,
    /// Distances beyond this are clamped (a truncation radius, à la a TSDF); `f64::INFINITY` = no clamp.
    pub truncation: f64,
}

impl Esdf {
    /// Build an ESDF from occupied points.
    pub fn from_occupied(occupied: Vec<Vector3<f64>>, truncation: f64) -> Esdf {
        Esdf { tree: KdTree::build(occupied), truncation }
    }

    /// Distance to the nearest occupied point (clamped to `truncation`).
    pub fn distance(&self, p: &Vector3<f64>) -> f64 {
        self.tree.nearest(p).map(|(_, d)| d.min(self.truncation)).unwrap_or(self.truncation)
    }

    /// The (unit) gradient `∇d` — the outward direction from the nearest obstacle toward `p`. Zero on an
    /// obstacle point.
    pub fn gradient(&self, p: &Vector3<f64>) -> Vector3<f64> {
        match self.tree.nearest(p) {
            Some((i, d)) if d > 1e-12 => (p - self.tree.point(i)) / d,
            _ => Vector3::zeros(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud() -> Vec<Vector3<f64>> {
        // a wall of occupied points on the plane z = 0
        let mut pts = Vec::new();
        let g = 11;
        for i in 0..g {
            for j in 0..g {
                pts.push(Vector3::new(i as f64 * 0.1 - 0.5, j as f64 * 0.1 - 0.5, 0.0));
            }
        }
        pts
    }

    fn brute(pts: &[Vector3<f64>], p: &Vector3<f64>) -> f64 {
        pts.iter().map(|q| (q - p).norm()).fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn the_field_matches_brute_force_nearest_occupied() {
        // THE ORACLE. The ESDF distance equals the exhaustive nearest-occupied distance.
        let pts = cloud();
        let esdf = Esdf::from_occupied(pts.clone(), f64::INFINITY);
        for q in &[Vector3::new(0.0, 0.0, 0.3), Vector3::new(0.2, -0.1, 1.0), Vector3::new(-0.4, 0.3, 0.05)] {
            assert!((esdf.distance(q) - brute(&pts, q)).abs() < 1e-9, "ESDF {} vs brute {}", esdf.distance(q), brute(&pts, q));
        }
    }

    #[test]
    fn the_field_is_zero_on_an_obstacle_and_grows_away() {
        let pts = cloud();
        let esdf = Esdf::from_occupied(pts.clone(), f64::INFINITY);
        assert!(esdf.distance(&pts[0]) < 1e-12, "distance on an obstacle point should be 0");
        let near = esdf.distance(&Vector3::new(0.0, 0.0, 0.1));
        let far = esdf.distance(&Vector3::new(0.0, 0.0, 0.8));
        assert!(far > near, "distance should grow away from the wall: {near} → {far}");
    }

    #[test]
    fn above_a_planar_wall_the_distance_is_the_height() {
        // For a point well inside the wall's extent, the nearest obstacle is directly below ⇒ distance ≈ z,
        // and the gradient points straight up (+z).
        let esdf = Esdf::from_occupied(cloud(), f64::INFINITY);
        let p = Vector3::new(0.02, -0.01, 0.5); // above the middle of the wall
        assert!((esdf.distance(&p) - 0.5).abs() < 0.06, "distance above the wall ≈ height: {}", esdf.distance(&p));
        let g = esdf.gradient(&p);
        assert!(g.z > 0.95 && g.norm() > 0.99, "gradient should be ~+z unit: {g}");
    }

    #[test]
    fn truncation_clamps_far_distances() {
        let esdf = Esdf::from_occupied(cloud(), 0.3);
        assert!((esdf.distance(&Vector3::new(0.0, 0.0, 2.0)) - 0.3).abs() < 1e-12, "far distance should clamp to the truncation");
    }
}
