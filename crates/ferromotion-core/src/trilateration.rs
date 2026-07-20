//! **Trilateration / multilateration** — recover a position from **range** measurements to known anchor
//! points. This is the geometry behind indoor positioning (UWB tags, Bluetooth/WiFi ranging), acoustic /
//! sonar beacon localization, and GNSS pseudorange fixes: each anchor `aᵢ` with a measured distance `rᵢ`
//! constrains the unknown position to a sphere `‖p − aᵢ‖ = rᵢ`, and their intersection is the fix.
//!
//! The quadratic range equations linearize by **differencing**: subtracting one anchor's equation from the
//! rest cancels the `‖p‖²` term, leaving `2(aᵢ − a₀)ᵀ p = (‖aᵢ‖² − ‖a₀‖²) − (rᵢ² − r₀²)` — an ordinary linear
//! least-squares solve that fuses any number of (possibly noisy, over-determined) anchors. Needs `≥ 4`
//! anchors in 3-D (`≥ 3` non-collinear in a plane). Verified: exact ranges recover the true position to
//! machine precision, an over-determined noisy set recovers it closely, and too few anchors return `None`.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector3};

/// Estimate a 3-D position from anchor points and their measured ranges (least squares). Returns `None` if
/// fewer than 4 anchors or the geometry is degenerate.
pub fn trilaterate(anchors: &[Vector3<f64>], ranges: &[f64]) -> Option<Vector3<f64>> {
    let n = anchors.len();
    if n < 4 || ranges.len() != n {
        return None;
    }
    let (a0, r0) = (anchors[0], ranges[0]);
    // rows i=1..n: 2(aᵢ − a₀)ᵀ p = (‖aᵢ‖² − ‖a₀‖²) − (rᵢ² − r₀²)
    let mut m = DMatrix::zeros(n - 1, 3);
    let mut b = DVector::zeros(n - 1);
    for i in 1..n {
        let d = anchors[i] - a0;
        for c in 0..3 {
            m[(i - 1, c)] = 2.0 * d[c];
        }
        b[i - 1] = anchors[i].norm_squared() - a0.norm_squared() - (ranges[i] * ranges[i] - r0 * r0);
    }
    let mtm = m.transpose() * &m;
    let sol = mtm.try_inverse()? * m.transpose() * b;
    Some(Vector3::new(sol[0], sol[1], sol[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchors() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(0.0, 5.0, 0.0),
            Vector3::new(0.0, 0.0, 5.0),
            Vector3::new(5.0, 5.0, 2.0),
            Vector3::new(2.0, 4.0, 5.0),
        ]
    }

    #[test]
    fn it_recovers_a_known_position_from_exact_ranges() {
        // THE ORACLE. Exact ranges to the anchors pin the position precisely.
        let p = Vector3::new(2.0, 1.5, 3.0);
        let a = anchors();
        let r: Vec<f64> = a.iter().map(|ai| (p - ai).norm()).collect();
        let est = trilaterate(&a, &r).unwrap();
        assert!((est - p).norm() < 1e-9, "recovered {est:?} vs {p:?}");
    }

    #[test]
    fn it_is_robust_to_range_noise_when_over_determined() {
        let p = Vector3::new(-1.0, 2.5, 1.0);
        let a = anchors();
        let mut seed = 5u64;
        let mut noise = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.04 };
        let r: Vec<f64> = a.iter().map(|ai| (p - ai).norm() + noise()).collect();
        let est = trilaterate(&a, &r).unwrap();
        assert!((est - p).norm() < 0.1, "noisy fix {est:?} vs {p:?}");
    }

    #[test]
    fn too_few_anchors_return_none() {
        let a = &anchors()[..3];
        let r = [1.0, 2.0, 3.0];
        assert!(trilaterate(a, &r).is_none(), "3 anchors is insufficient in 3D");
    }
}
