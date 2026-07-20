//! **TEASER-style robust point-cloud registration** (Yang, Shi & Carlone, T-RO 2021). Recover the rigid
//! transform `(R, t)` aligning two point sets with putative correspondences that may be **mostly outliers**
//! — a regime where plain least-squares (Kabsch/Umeyama) and even RANSAC struggle. TEASER's insight is to
//! **decouple** the estimation: differences of correspondences (**translation-invariant measurements**,
//! TIMs) cancel the unknown translation, leaving a rotation-only problem; rotation is then solved with
//! **graduated non-convexity + truncated least squares (GNC-TLS)**, which smoothly rejects outliers without
//! an initial guess; and the translation follows robustly (component-wise median) once the rotation is
//! known. This complements the crate's [`crate::gnc`] (scalar linear GNC) and ICP-family registration with
//! the init-free, high-outlier global aligner.
//!
//! (The full TEASER++ adds maximum-clique inlier pre-pruning to reach >99% outliers; here the GNC-TLS
//! rotation on all TIMs already tolerates large outlier fractions.) Verified: it recovers a known transform
//! from correspondences with a high fraction of gross outliers, to which Kabsch is oblivious. Pure
//! `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

/// The recovered rigid registration `dst ≈ R·src + t`.
#[derive(Clone, Copy, Debug)]
pub struct Registration {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

// Weighted rotation-only fit: R minimizing Σ wₖ‖vₖ − R uₖ‖² (weighted Kabsch on vectors, no centroid).
fn weighted_rotation(u: &[Vector3<f64>], v: &[Vector3<f64>], w: &[f64]) -> Matrix3<f64> {
    let mut h = Matrix3::zeros();
    for ((uk, vk), &wk) in u.iter().zip(v).zip(w) {
        h += wk * vk * uk.transpose();
    }
    let svd = h.svd(true, true);
    let (um, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (um * vt).determinant().signum();
    um * d * vt
}

// GNC-TLS truncated-least-squares weight for residual r at control μ and noise bound c̄.
fn tls_weight(r: f64, mu: f64, barc: f64) -> f64 {
    let (r2, barc2) = (r * r, barc * barc);
    let lo = mu / (mu + 1.0) * barc2;
    let hi = (mu + 1.0) / mu * barc2;
    if r2 >= hi {
        0.0
    } else if r2 <= lo {
        1.0
    } else {
        (barc / r) * (mu * (mu + 1.0)).sqrt() - mu
    }
}

/// Register `src` onto `dst` (same length, index-aligned correspondences, some fraction outliers) robust to
/// gross outliers. `noise_bound` is the expected inlier residual scale.
pub fn register(src: &[Vector3<f64>], dst: &[Vector3<f64>], noise_bound: f64) -> Registration {
    // translation-invariant measurements: all correspondence differences
    let n = src.len();
    let mut u = Vec::new();
    let mut v = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let du = src[i] - src[j];
            if du.norm() > 1e-9 {
                u.push(du);
                v.push(dst[i] - dst[j]);
            }
        }
    }
    let barc = 2.0 * noise_bound; // TIMs are differences of two points ⇒ noise scales up
    // --- GNC-TLS rotation ---
    let mut w = vec![1.0; u.len()];
    let mut r = weighted_rotation(&u, &v, &w);
    let r_max = u.iter().zip(&v).map(|(uk, vk)| (vk - r * uk).norm()).fold(0.0, f64::max);
    let mut mu = if 2.0 * r_max * r_max - barc * barc > 1e-9 { barc * barc / (2.0 * r_max * r_max - barc * barc) } else { 1e-4 };
    for _ in 0..100 {
        // update weights from current residuals
        for (k, (uk, vk)) in u.iter().zip(&v).enumerate() {
            w[k] = tls_weight((vk - r * uk).norm(), mu, barc);
        }
        r = weighted_rotation(&u, &v, &w);
        mu *= 1.4;
        // stop once the weights have essentially binarized
        if w.iter().all(|&wk| !(1e-3..=1.0 - 1e-3).contains(&wk)) && mu > 1.0 {
            break;
        }
    }
    // --- robust translation: median of (dst − R·src) per axis ---
    let mut res: Vec<Vector3<f64>> = src.iter().zip(dst).map(|(s, d)| d - r * s).collect();
    let mut t = Vector3::zeros();
    for axis in 0..3 {
        res.sort_by(|a, b| a[axis].partial_cmp(&b[axis]).unwrap());
        t[axis] = res[res.len() / 2][axis];
    }
    Registration { rotation: r, translation: t }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn so3(w: Vector3<f64>) -> Matrix3<f64> {
        let th = w.norm();
        if th < 1e-12 {
            return Matrix3::identity();
        }
        let k = w / th;
        let kx = Matrix3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        Matrix3::identity() + th.sin() * kx + (1.0 - th.cos()) * kx * kx
    }

    fn rand(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5
    }

    #[test]
    fn it_recovers_a_transform_despite_many_outliers() {
        // THE ORACLE / HEADLINE. 30 correspondences, 12 of them gross outliers (40%). TEASER recovers the
        // true (R, t); plain Kabsch on all points would be badly biased.
        let r_true = so3(Vector3::new(0.3, -0.5, 0.4));
        let t_true = Vector3::new(1.5, -0.8, 2.0);
        let mut seed = 20260720u64;
        let n = 30;
        let src: Vec<Vector3<f64>> = (0..n).map(|_| Vector3::new(rand(&mut seed) * 4.0, rand(&mut seed) * 4.0, rand(&mut seed) * 4.0)).collect();
        let mut dst: Vec<Vector3<f64>> = src.iter().map(|s| r_true * s + t_true).collect();
        // corrupt 12 correspondences with random points
        for d in dst.iter_mut().take(12) {
            *d = Vector3::new(rand(&mut seed) * 8.0, rand(&mut seed) * 8.0, rand(&mut seed) * 8.0);
        }
        let reg = register(&src, &dst, 0.01);
        let rot_err = (reg.rotation - r_true).norm();
        let t_err = (reg.translation - t_true).norm();
        assert!(rot_err < 1e-3, "rotation error {rot_err}");
        assert!(t_err < 1e-2, "translation error {t_err}");

        // sanity: naive Kabsch (all points, no robustness) is far worse
        let src_c: Vector3<f64> = src.iter().sum::<Vector3<f64>>() / n as f64;
        let dst_c: Vector3<f64> = dst.iter().sum::<Vector3<f64>>() / n as f64;
        let mut h = Matrix3::zeros();
        for (s, d) in src.iter().zip(&dst) {
            h += (d - dst_c) * (s - src_c).transpose();
        }
        let svd = h.svd(true, true);
        let naive_r = svd.u.unwrap() * svd.v_t.unwrap();
        assert!((naive_r - r_true).norm() > 10.0 * rot_err, "robust estimate should crush naive Kabsch");
    }

    #[test]
    fn it_is_exact_with_no_outliers() {
        let r_true = so3(Vector3::new(-0.2, 0.6, 0.1));
        let t_true = Vector3::new(-1.0, 2.0, 0.5);
        let mut seed = 7u64;
        let src: Vec<Vector3<f64>> = (0..12).map(|_| Vector3::new(rand(&mut seed) * 3.0, rand(&mut seed) * 3.0, rand(&mut seed) * 3.0)).collect();
        let dst: Vec<Vector3<f64>> = src.iter().map(|s| r_true * s + t_true).collect();
        let reg = register(&src, &dst, 0.001);
        assert!((reg.rotation - r_true).norm() < 1e-6 && (reg.translation - t_true).norm() < 1e-6, "clean recovery");
    }
}
