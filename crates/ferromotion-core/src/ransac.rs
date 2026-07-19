//! **RANSAC — Random Sample Consensus** (Fischler & Bolles, 1981): fit a model to data corrupted by gross
//! **outliers** by repeatedly fitting from a minimal random sample and keeping the model with the largest
//! **inlier consensus set**. It is the robust-estimation workhorse of perception — robust line/plane fitting,
//! robust homography / [`crate::essential`] matrix / [`crate::pnp`] — anywhere a fraction of the
//! correspondences are wrong. It complements [`crate::gnc`] (graduated non-convexity, a continuous outlier
//! down-weighting) with the discrete sample-and-count approach that tolerates very high outlier ratios.
//!
//! This is a **generic** RANSAC over any model type `M`: you supply a minimal-sample fitter and an
//! inlier test, and it returns the best model plus its inlier mask. Deterministic (seeded) → reproducible.
//! Verified: it recovers a line and a plane from data with 40–50% outliers and identifies the inliers, and —
//! wrapping the 8-point algorithm — it recovers the true two-view pose despite mismatched correspondences.
//! Pure `nalgebra` → WASM-clean.

/// A deterministic SplitMix64 sampler, so RANSAC is reproducible.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The RANSAC outcome: the winning model, a per-datum inlier mask, and the inlier count.
#[derive(Clone, Debug)]
pub struct RansacResult<M> {
    pub model: M,
    pub inliers: Vec<bool>,
    pub n_inliers: usize,
}

/// Run RANSAC over `n` data indices. `sample_size` is the minimal set the model needs; `fit` builds a model
/// from a sample (returning `None` if degenerate); `is_inlier(model, i)` tests datum `i`. Returns the model
/// with the largest consensus set after `iters` trials (deterministic in `seed`), or `None` if nothing fit.
pub fn ransac<M>(
    n: usize,
    sample_size: usize,
    iters: usize,
    seed: u64,
    fit: impl Fn(&[usize]) -> Option<M>,
    is_inlier: impl Fn(&M, usize) -> bool,
) -> Option<RansacResult<M>> {
    if n < sample_size {
        return None;
    }
    let mut rng = Rng::new(seed);
    let mut best: Option<RansacResult<M>> = None;
    let mut sample = vec![0usize; sample_size];
    for _ in 0..iters {
        // draw `sample_size` distinct indices
        for s in 0..sample_size {
            loop {
                let idx = rng.below(n);
                if !sample[..s].contains(&idx) {
                    sample[s] = idx;
                    break;
                }
            }
        }
        let Some(model) = fit(&sample) else { continue };
        let inliers: Vec<bool> = (0..n).map(|i| is_inlier(&model, i)).collect();
        let n_inliers = inliers.iter().filter(|&&b| b).count();
        if best.as_ref().is_none_or(|b| n_inliers > b.n_inliers) {
            best = Some(RansacResult { model, inliers, n_inliers });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::essential::{eight_point, recover_pose};
    use nalgebra::{Matrix3, Vector2, Vector3};

    // deterministic uniform in [-1,1)
    fn urand(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    }

    #[test]
    fn it_recovers_a_line_from_data_with_many_outliers() {
        // A line y = 2x + 1; 60 inliers on it, 40 uniform-noise outliers. RANSAC must find the line and flag
        // the inliers.
        let (a, b) = (2.0, 1.0); // y = a x + b
        let mut pts: Vec<Vector2<f64>> = Vec::new();
        let mut seed = 42u64;
        for i in 0..60 {
            let x = i as f64 * 0.1 - 3.0;
            pts.push(Vector2::new(x, a * x + b + 0.002 * urand(&mut seed)));
        }
        for _ in 0..40 {
            pts.push(Vector2::new(6.0 * urand(&mut seed), 12.0 * urand(&mut seed)));
        }
        // model = (a,b,c) with a x + b y + c = 0, normalized
        let fit = |s: &[usize]| -> Option<Vector3<f64>> {
            let (p, q) = (pts[s[0]], pts[s[1]]);
            let dir = q - p;
            if dir.norm() < 1e-9 {
                return None;
            }
            let nrm = Vector2::new(-dir.y, dir.x).normalize();
            Some(Vector3::new(nrm.x, nrm.y, -nrm.dot(&p)))
        };
        let is_inlier = |m: &Vector3<f64>, i: usize| (m.x * pts[i].x + m.y * pts[i].y + m.z).abs() < 0.05;
        let res = ransac(pts.len(), 2, 200, 7, fit, is_inlier).expect("a line should be found");
        assert!(res.n_inliers >= 58, "should recover ~all 60 inliers: {}", res.n_inliers);
        // the first 60 (true inliers) should be flagged, the outliers mostly not
        assert!(res.inliers[..60].iter().filter(|&&b| b).count() >= 58, "true inliers flagged");
        assert!(res.inliers[60..].iter().filter(|&&b| b).count() <= 3, "few outliers flagged");
    }

    #[test]
    fn it_recovers_a_plane_from_data_with_outliers() {
        // plane z = 0.5x - 0.3y + 2 (normal (−0.5, 0.3, 1)); 50 inliers, 50 outliers.
        let mut pts: Vec<Vector3<f64>> = Vec::new();
        let mut seed = 99u64;
        for _ in 0..50 {
            let (x, y) = (5.0 * urand(&mut seed), 5.0 * urand(&mut seed));
            pts.push(Vector3::new(x, y, 0.5 * x - 0.3 * y + 2.0 + 0.002 * urand(&mut seed)));
        }
        for _ in 0..50 {
            pts.push(Vector3::new(5.0 * urand(&mut seed), 5.0 * urand(&mut seed), 10.0 * urand(&mut seed)));
        }
        let fit = |s: &[usize]| -> Option<(Vector3<f64>, f64)> {
            let (p, q, r) = (pts[s[0]], pts[s[1]], pts[s[2]]);
            let nrm = (q - p).cross(&(r - p));
            if nrm.norm() < 1e-9 {
                return None;
            }
            let nn = nrm.normalize();
            Some((nn, -nn.dot(&p)))
        };
        let is_inlier = |m: &(Vector3<f64>, f64), i: usize| (m.0.dot(&pts[i]) + m.1).abs() < 0.05;
        let res = ransac(pts.len(), 3, 300, 11, fit, is_inlier).expect("a plane should be found");
        assert!(res.n_inliers >= 48, "should recover ~all 50 plane inliers: {}", res.n_inliers);
    }

    #[test]
    fn it_makes_the_eight_point_algorithm_robust_to_mismatches() {
        // THE HEADLINE. Two-view correspondences, but 5 of 30 are wrong matches. Plain 8-point on all points
        // is corrupted; RANSAC over 8-point samples recovers the true pose.
        fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
            Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
        }
        fn so3(phi: &Vector3<f64>) -> Matrix3<f64> {
            let th = phi.norm();
            let k = phi / th;
            let kx = skew(&k);
            Matrix3::identity() + th.sin() * kx + (1.0 - th.cos()) * kx * kx
        }
        let r_true = so3(&Vector3::new(0.03, 0.15, -0.08));
        let t_true = Vector3::new(1.0, 0.05, 0.1);
        let mut seed = 5u64;
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for _ in 0..30 {
            let p = Vector3::new(2.0 * urand(&mut seed), 2.0 * urand(&mut seed), 5.0 + 2.0 * urand(&mut seed));
            x1.push(Vector2::new(p.x / p.z, p.y / p.z));
            let pc = r_true * p + t_true;
            x2.push(Vector2::new(pc.x / pc.z, pc.y / pc.z));
        }
        // corrupt 5 matches by shuffling their second-view points
        for slot in x2.iter_mut().take(5) {
            *slot = Vector2::new(3.0 * urand(&mut seed), 3.0 * urand(&mut seed));
        }
        let (x1c, x2c) = (x1.clone(), x2.clone());
        let fit = move |s: &[usize]| -> Option<Matrix3<f64>> {
            let a: Vec<Vector2<f64>> = s.iter().map(|&i| x1c[i]).collect();
            let b: Vec<Vector2<f64>> = s.iter().map(|&i| x2c[i]).collect();
            Some(eight_point(&a, &b))
        };
        let (x1i, x2i) = (x1.clone(), x2.clone());
        let is_inlier = move |e: &Matrix3<f64>, i: usize| {
            let v1 = Vector3::new(x1i[i].x, x1i[i].y, 1.0);
            let v2 = Vector3::new(x2i[i].x, x2i[i].y, 1.0);
            (v2.transpose() * e * v1)[(0, 0)].abs() < 1e-3
        };
        let res = ransac(30, 8, 500, 3, fit, is_inlier).expect("robust E should be found");
        assert!(res.n_inliers >= 24, "should keep the ~25 good matches: {}", res.n_inliers);
        // refit the pose on the consensus set and compare to ground truth
        let inl1: Vec<Vector2<f64>> = (0..30).filter(|&i| res.inliers[i]).map(|i| x1[i]).collect();
        let inl2: Vec<Vector2<f64>> = (0..30).filter(|&i| res.inliers[i]).map(|i| x2[i]).collect();
        let e = eight_point(&inl1, &inl2);
        let (r, t) = recover_pose(&e, &inl1, &inl2);
        assert!((r - r_true).norm() < 1e-6, "rotation recovered despite mismatches: {}", (r - r_true).norm());
        assert!(t.normalize().dot(&t_true.normalize()) > 1.0 - 1e-6, "translation direction recovered");
    }
}
