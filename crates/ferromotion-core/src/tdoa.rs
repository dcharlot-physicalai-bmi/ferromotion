//! **Time-Difference-of-Arrival (TDOA) localization** — locate an **un-clocked** emitter from the
//! *differences* in signal arrival time across pairs of synchronized receivers. Where
//! [`crate::trilaterate`] needs an absolute range to each anchor (so the source must share the anchors'
//! clock), TDOA needs only that the *receivers* are synchronized: each range-difference `dᵢ = ‖p − aᵢ‖ −
//! ‖p − a₀‖` places the source on a **hyperboloid** with foci `aᵢ, a₀`, and their intersection is the fix.
//! This is the geometry of acoustic source localization, gunshot / drone detection, GPS-denied RF
//! localization, and cellular positioning.
//!
//! The hyperbolic equations linearize by carrying the reference range `R₀ = ‖p − a₀‖` as an extra unknown:
//! `2(aᵢ − a₀)ᵀ p + 2 dᵢ R₀ = ‖aᵢ‖² − ‖a₀‖² − dᵢ²`, linear in `(p, R₀)` — a plain least-squares solve
//! (Chan–Ho / spherical-interpolation style). Needs `≥ 5` receivers in 3-D. Verified: from exact
//! range-differences it recovers the true source to machine precision, an over-determined noisy set recovers
//! it closely, and too few receivers return `None`. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector3};

/// Localize an emitter in 3-D from receiver positions and their range-differences relative to receiver 0
/// (`tdoa[i] = ‖p − aᵢ‖ − ‖p − a₀‖`, with `tdoa[0] = 0`; pass differences × the wave speed if you have
/// times). Returns `None` for fewer than 5 receivers or a degenerate geometry.
pub fn tdoa_localize(anchors: &[Vector3<f64>], tdoa: &[f64]) -> Option<Vector3<f64>> {
    let n = anchors.len();
    if n < 5 || tdoa.len() != n {
        return None;
    }
    let a0 = anchors[0];
    // unknowns [px, py, pz, R0]; rows i=1..n: 2(aᵢ−a₀)ᵀ p + 2 dᵢ R₀ = ‖aᵢ‖² − ‖a₀‖² − dᵢ²
    let mut m = DMatrix::zeros(n - 1, 4);
    let mut b = DVector::zeros(n - 1);
    for i in 1..n {
        let d = anchors[i] - a0;
        let di = tdoa[i];
        m[(i - 1, 0)] = 2.0 * d.x;
        m[(i - 1, 1)] = 2.0 * d.y;
        m[(i - 1, 2)] = 2.0 * d.z;
        m[(i - 1, 3)] = 2.0 * di;
        b[i - 1] = anchors[i].norm_squared() - a0.norm_squared() - di * di;
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
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(0.0, 10.0, 0.0),
            Vector3::new(10.0, 10.0, 0.0),
            Vector3::new(5.0, 5.0, 8.0),
            Vector3::new(2.0, 8.0, 3.0),
        ]
    }

    // exact range-differences relative to anchor 0 for a source at p
    fn tdoa_of(a: &[Vector3<f64>], p: Vector3<f64>) -> Vec<f64> {
        let r0 = (p - a[0]).norm();
        a.iter().map(|ai| (p - ai).norm() - r0).collect()
    }

    #[test]
    fn it_recovers_a_known_source_from_exact_tdoa() {
        // THE ORACLE. Exact range-differences pin the source.
        let p = Vector3::new(3.0, 4.0, 2.0);
        let a = anchors();
        let est = tdoa_localize(&a, &tdoa_of(&a, p)).unwrap();
        assert!((est - p).norm() < 1e-8, "recovered {est:?} vs {p:?}");
    }

    #[test]
    fn it_is_robust_to_tdoa_noise() {
        let p = Vector3::new(6.0, 2.0, 4.0);
        let a = anchors();
        let mut t = tdoa_of(&a, p);
        let mut seed = 11u64;
        let mut noise = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.02 };
        for ti in t.iter_mut().skip(1) {
            *ti += noise();
        }
        let est = tdoa_localize(&a, &t).unwrap();
        assert!((est - p).norm() < 0.2, "noisy TDOA fix {est:?} vs {p:?}");
    }

    #[test]
    fn too_few_receivers_return_none() {
        let a = &anchors()[..4];
        assert!(tdoa_localize(a, &[0.0, 1.0, 2.0, 3.0]).is_none(), "4 receivers is insufficient for 3D TDOA");
    }
}
