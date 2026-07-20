//! **Doppler radar ego-velocity** — recover a sensor's own 3-D velocity from a *single* automotive/4-D radar
//! scan. Each radar return gives a **bearing** `d̂ᵢ` (unit direction to the point) and a **radial velocity**
//! `v_rᵢ` (the Doppler measurement). For a **static** point, the measured radial velocity is exactly the
//! projection of the sensor's motion onto that bearing: `v_rᵢ = −d̂ᵢ · v_ego`. Stacking this linear
//! constraint over the scan recovers `v_ego` by least squares — an instantaneous, drift-free velocity
//! estimate (no integration) that anchors radar-inertial odometry and works in the rain/dust/dark where
//! cameras and LiDAR struggle. Moving objects violate the static assumption, so a robust variant rejects
//! them with RANSAC (reusing [`crate::ransac`]).
//!
//! Verified: from a static-world scan with a known ego-velocity the least-squares estimate recovers it
//! exactly; and with a large fraction of moving-target outliers the RANSAC variant still recovers it and
//! flags the movers. Pure `nalgebra` → WASM-clean.

use crate::ransac::ransac;
use nalgebra::{Matrix3, Vector3};

/// Least-squares ego-velocity from bearings `d̂ᵢ` (unit) and radial velocities `v_rᵢ`, solving
/// `min_v Σ (v_rᵢ + d̂ᵢ·v)²` ⇒ `(Σ d̂ᵢ d̂ᵢᵀ) v = −Σ v_rᵢ d̂ᵢ`. Needs ≥ 3 non-degenerate bearings.
pub fn ego_velocity_ls(bearings: &[Vector3<f64>], radial: &[f64]) -> Option<Vector3<f64>> {
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for (d, &vr) in bearings.iter().zip(radial) {
        a += d * d.transpose();
        b -= vr * d;
    }
    a.try_inverse().map(|inv| inv * b)
}

/// The ego-velocity, and a per-return inlier mask, robust to moving-target outliers via RANSAC (3-point
/// minimal fits, inlier if the radial-velocity residual is within `noise_bound`).
pub fn ego_velocity_ransac(bearings: &[Vector3<f64>], radial: &[f64], noise_bound: f64, iters: usize, seed: u64) -> Option<(Vector3<f64>, Vec<bool>)> {
    let fit = |idx: &[usize]| -> Option<Vector3<f64>> {
        let b: Vec<Vector3<f64>> = idx.iter().map(|&i| bearings[i]).collect();
        let r: Vec<f64> = idx.iter().map(|&i| radial[i]).collect();
        ego_velocity_ls(&b, &r)
    };
    let is_inlier = |v: &Vector3<f64>, i: usize| (radial[i] + bearings[i].dot(v)).abs() < noise_bound;
    let res = ransac(bearings.len(), 3, iters, seed, fit, is_inlier)?;
    // refit on the consensus inliers for a tight estimate
    let inl_b: Vec<Vector3<f64>> = (0..bearings.len()).filter(|&i| res.inliers[i]).map(|i| bearings[i]).collect();
    let inl_r: Vec<f64> = (0..bearings.len()).filter(|&i| res.inliers[i]).map(|i| radial[i]).collect();
    let v = ego_velocity_ls(&inl_b, &inl_r)?;
    Some((v, res.inliers))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5
    }

    // a fan of unit bearings and the static-world radial velocities they'd measure under `v_ego`
    fn static_scan(v_ego: Vector3<f64>, n: usize, seed: &mut u64) -> (Vec<Vector3<f64>>, Vec<f64>) {
        let mut b = Vec::new();
        let mut r = Vec::new();
        for _ in 0..n {
            let d = Vector3::new(rand(seed), rand(seed), 0.3 * rand(seed)).normalize();
            r.push(-d.dot(&v_ego));
            b.push(d);
        }
        (b, r)
    }

    #[test]
    fn least_squares_recovers_the_ego_velocity_exactly() {
        // THE ORACLE. A static-world scan ⇒ v_ego is recovered from the Doppler+bearing constraints.
        let v_ego = Vector3::new(3.0, -1.0, 0.2);
        let mut seed = 42u64;
        let (b, r) = static_scan(v_ego, 20, &mut seed);
        let v = ego_velocity_ls(&b, &r).unwrap();
        assert!((v - v_ego).norm() < 1e-9, "recovered {v} vs {v_ego}");
    }

    #[test]
    fn ransac_rejects_moving_targets() {
        // THE HEADLINE. 30 returns, 10 of them moving objects (their radial velocity is corrupted). RANSAC
        // recovers the true ego-velocity and flags the movers as outliers.
        let v_ego = Vector3::new(2.5, 0.8, -0.4);
        let mut seed = 7u64;
        let (b, mut r) = static_scan(v_ego, 30, &mut seed);
        for ri in r.iter_mut().take(10) {
            *ri += 4.0 + 3.0 * rand(&mut seed); // moving-target Doppler offset
        }
        let (v, inliers) = ego_velocity_ransac(&b, &r, 0.05, 300, 1).unwrap();
        assert!((v - v_ego).norm() < 1e-6, "robust estimate {v} vs {v_ego}");
        // the 10 movers should be flagged outliers, the 20 static ones inliers
        assert!(inliers[10..].iter().filter(|&&x| x).count() >= 19, "static returns are inliers");
        assert!(inliers[..10].iter().filter(|&&x| x).count() <= 1, "movers are outliers");
    }

    #[test]
    fn a_stationary_sensor_reads_zero() {
        let mut seed = 3u64;
        let (b, r) = static_scan(Vector3::zeros(), 12, &mut seed);
        assert!(ego_velocity_ls(&b, &r).unwrap().norm() < 1e-12, "stationary ⇒ zero ego-velocity");
    }
}
