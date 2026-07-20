//! **Algebraic circle & sphere fitting** (Kåsa least squares). Fit a circle to 2-D points or a sphere to
//! 3-D points in **closed form** by linearizing the implicit equation: since a point on a circle satisfies
//! `x² + y² = 2aₓx + 2a_yy + c` (with centre `(aₓ, a_y)` and `c = r² − ‖a‖²`), stacking that over the points
//! is an ordinary linear least-squares solve — no iteration, no initial guess. Its everyday uses in physical
//! AI: **magnetometer / accelerometer calibration** (the raw readings lie on an offset sphere — the fit
//! centre is the hard-iron bias), estimating a wheel/turn **radius of curvature** from odometry, and fitting
//! circular/cylindrical features in laser scans.
//!
//! Verified: points on a known circle/sphere recover its centre and radius exactly; an offset (biased)
//! sphere recovers the bias; and the fit stays accurate under measurement noise. Pure `nalgebra` →
//! WASM-clean.

use nalgebra::{DMatrix, DVector, Vector2, Vector3};

/// Fit a circle to 2-D points, returning `(centre, radius)`. Needs ≥ 3 non-collinear points.
pub fn fit_circle(points: &[Vector2<f64>]) -> Option<(Vector2<f64>, f64)> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    // rows [2xᵢ, 2yᵢ, 1] · [aₓ, a_y, c] = xᵢ²+yᵢ²
    let mut a = DMatrix::zeros(n, 3);
    let mut b = DVector::zeros(n);
    for (i, p) in points.iter().enumerate() {
        a[(i, 0)] = 2.0 * p.x;
        a[(i, 1)] = 2.0 * p.y;
        a[(i, 2)] = 1.0;
        b[i] = p.x * p.x + p.y * p.y;
    }
    let sol = (a.transpose() * &a).try_inverse()? * a.transpose() * b;
    let center = Vector2::new(sol[0], sol[1]);
    let radius = (sol[2] + center.norm_squared()).max(0.0).sqrt();
    Some((center, radius))
}

/// Fit a sphere to 3-D points, returning `(centre, radius)`. Needs ≥ 4 non-coplanar points.
pub fn fit_sphere(points: &[Vector3<f64>]) -> Option<(Vector3<f64>, f64)> {
    let n = points.len();
    if n < 4 {
        return None;
    }
    let mut a = DMatrix::zeros(n, 4);
    let mut b = DVector::zeros(n);
    for (i, p) in points.iter().enumerate() {
        a[(i, 0)] = 2.0 * p.x;
        a[(i, 1)] = 2.0 * p.y;
        a[(i, 2)] = 2.0 * p.z;
        a[(i, 3)] = 1.0;
        b[i] = p.norm_squared();
    }
    let sol = (a.transpose() * &a).try_inverse()? * a.transpose() * b;
    let center = Vector3::new(sol[0], sol[1], sol[2]);
    let radius = (sol[3] + center.norm_squared()).max(0.0).sqrt();
    Some((center, radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn it_recovers_a_known_circle() {
        // THE ORACLE. Points sampled on a circle of centre (2, −1) radius 3 recover it exactly.
        let (c, r) = (Vector2::new(2.0, -1.0), 3.0);
        let pts: Vec<Vector2<f64>> = (0..12).map(|i| { let a = TAU * i as f64 / 12.0; c + Vector2::new(r * a.cos(), r * a.sin()) }).collect();
        let (fc, fr) = fit_circle(&pts).unwrap();
        assert!((fc - c).norm() < 1e-9 && (fr - r).abs() < 1e-9, "circle {fc:?} r={fr}");
    }

    #[test]
    fn it_recovers_a_known_sphere_and_the_hard_iron_bias() {
        // THE APPLICATION. Magnetometer-style: readings lie on a sphere offset by the hard-iron bias; the fit
        // centre IS that bias.
        let (bias, r) = (Vector3::new(-0.4, 0.15, 0.6), 1.0);
        let mut seed = 1u64;
        let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5 };
        let pts: Vec<Vector3<f64>> = (0..40).map(|_| bias + Vector3::new(u(), u(), u()).normalize() * r).collect();
        let (c, fr) = fit_sphere(&pts).unwrap();
        assert!((c - bias).norm() < 1e-9 && (fr - r).abs() < 1e-9, "sphere centre (bias) {c:?} r={fr}");
    }

    #[test]
    fn the_fit_is_accurate_under_noise() {
        let (c, r) = (Vector2::new(-1.5, 2.5), 4.0);
        let mut seed = 9u64;
        let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.05 };
        let pts: Vec<Vector2<f64>> = (0..60).map(|i| { let a = TAU * i as f64 / 60.0; c + Vector2::new(r * a.cos() + u(), r * a.sin() + u()) }).collect();
        let (fc, fr) = fit_circle(&pts).unwrap();
        assert!((fc - c).norm() < 0.05 && (fr - r).abs() < 0.05, "noisy fit {fc:?} r={fr}");
    }

    #[test]
    fn too_few_points_return_none() {
        assert!(fit_circle(&[Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)]).is_none());
        assert!(fit_sphere(&[Vector3::zeros(); 3]).is_none());
    }
}
