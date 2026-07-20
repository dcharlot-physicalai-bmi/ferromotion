//! **k-means clustering** (Lloyd's algorithm with **k-means++** seeding). Partition points into `k` groups
//! that minimize the within-cluster sum of squares (inertia) `Σⱼ Σ_{x∈Cⱼ} ‖x − μⱼ‖²`. It is the standard
//! unsupervised tool for vector quantization / codebooks, feature and descriptor grouping, key-frame
//! selection, and point-cloud segmentation. The k-means++ seeding spreads the initial centres by choosing
//! each with probability proportional to its squared distance from the nearest existing centre, which avoids
//! the poor local minima random seeding falls into.
//!
//! Distinct from [`crate::LloydCoverage`], which runs the *same* Lloyd iteration for multi-robot *coverage*
//! over a continuous density; here the "density" is a discrete point set and the goal is a labelling. Each
//! Lloyd step provably does not increase inertia, so the objective is monotone. Deterministic (seeded).
//! Verified: it recovers well-separated clusters (correct labels, centroids on the true means); inertia
//! decreases monotonically; and k-means++ seeding beats a deliberately bad initialization. Pure `nalgebra`
//! → WASM-clean.

use nalgebra::DVector;

/// A deterministic SplitMix64 sampler.
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
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The result of clustering: centroids, per-point labels, the final inertia, and the inertia after each
/// Lloyd iteration (a monotonically non-increasing curve).
#[derive(Clone, Debug)]
pub struct KMeans {
    pub centroids: Vec<DVector<f64>>,
    pub labels: Vec<usize>,
    pub inertia: f64,
    pub history: Vec<f64>,
}

fn nearest(centroids: &[DVector<f64>], x: &DVector<f64>) -> (usize, f64) {
    let mut best = 0;
    let mut bd = f64::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = (c - x).norm_squared();
        if d < bd {
            bd = d;
            best = i;
        }
    }
    (best, bd)
}

/// k-means++ seeding: `k` initial centres spread by squared-distance sampling.
fn kpp_init(points: &[DVector<f64>], k: usize, rng: &mut Rng) -> Vec<DVector<f64>> {
    let mut centroids = vec![points[rng.next_u64() as usize % points.len()].clone()];
    while centroids.len() < k {
        let d2: Vec<f64> = points.iter().map(|p| nearest(&centroids, p).1).collect();
        let total: f64 = d2.iter().sum();
        if total <= 0.0 {
            centroids.push(points[rng.next_u64() as usize % points.len()].clone());
            continue;
        }
        // sample proportional to D²
        let mut target = rng.unif() * total;
        let mut chosen = points.len() - 1;
        for (i, &d) in d2.iter().enumerate() {
            target -= d;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(points[chosen].clone());
    }
    centroids
}

/// Cluster `points` into `k` groups by Lloyd's algorithm with k-means++ seeding, running up to `iters`
/// iterations (stops early on convergence). Deterministic in `seed`.
pub fn kmeans(points: &[DVector<f64>], k: usize, iters: usize, seed: u64) -> KMeans {
    let mut rng = Rng::new(seed);
    let dim = points[0].len();
    let mut centroids = kpp_init(points, k, &mut rng);
    let mut labels = vec![0usize; points.len()];
    let mut history = Vec::new();

    for _ in 0..iters.max(1) {
        // assignment step
        let mut inertia = 0.0;
        for (i, p) in points.iter().enumerate() {
            let (c, d) = nearest(&centroids, p);
            labels[i] = c;
            inertia += d;
        }
        history.push(inertia);
        // update step
        let mut sums = vec![DVector::zeros(dim); k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            sums[labels[i]] += p;
            counts[labels[i]] += 1;
        }
        let mut moved = false;
        for c in 0..k {
            if counts[c] > 0 {
                let new_c = &sums[c] / counts[c] as f64;
                if (&new_c - &centroids[c]).norm() > 1e-12 {
                    moved = true;
                }
                centroids[c] = new_c;
            }
        }
        if !moved {
            break;
        }
    }
    // final inertia with the converged centroids
    let mut inertia = 0.0;
    for (i, p) in points.iter().enumerate() {
        let (c, d) = nearest(&centroids, p);
        labels[i] = c;
        inertia += d;
    }
    history.push(inertia);
    KMeans { centroids, labels, inertia, history }
}

#[cfg(test)]
mod tests {
    use super::*;

    // three well-separated 2-D blobs around (0,0), (10,0), (5,8)
    fn blobs() -> (Vec<DVector<f64>>, Vec<[f64; 2]>) {
        let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]];
        let mut seed = 7u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 1.2
        };
        let mut pts = Vec::new();
        for c in &centers {
            for _ in 0..40 {
                pts.push(DVector::from_row_slice(&[c[0] + u(), c[1] + u()]));
            }
        }
        (pts, centers.to_vec())
    }

    #[test]
    fn it_recovers_well_separated_clusters() {
        // THE ORACLE. Three tight, separated blobs ⇒ each centroid lands on a true blob centre, and every
        // blob's 40 points share one label.
        let (pts, centers) = blobs();
        let km = kmeans(&pts, 3, 50, 1);
        // each true centre has a matching centroid nearby
        for c in &centers {
            let cv = DVector::from_row_slice(c);
            let d = km.centroids.iter().map(|k| (k - &cv).norm()).fold(f64::INFINITY, f64::min);
            assert!(d < 0.5, "no centroid near true centre {c:?}: min dist {d}");
        }
        // points 0..40 (blob 0) all share a label, etc.
        for b in 0..3 {
            let first = km.labels[b * 40];
            assert!(km.labels[b * 40..(b + 1) * 40].iter().all(|&l| l == first), "blob {b} split across labels");
        }
    }

    #[test]
    fn inertia_decreases_monotonically() {
        // THE MONOTONICITY GUARANTEE. Each Lloyd iteration weakly lowers the within-cluster sum of squares.
        let (pts, _) = blobs();
        let km = kmeans(&pts, 3, 50, 2);
        for w in km.history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "inertia rose: {} → {}", w[0], w[1]);
        }
    }

    #[test]
    fn kpp_seeding_beats_a_bad_initialization() {
        // k-means++ should reach a lower inertia than Lloyd started from three centres bunched in one blob.
        let (pts, _) = blobs();
        let km = kmeans(&pts, 3, 50, 3);
        // deliberately bad init: all three seeds inside blob 0, then run plain Lloyd
        let mut centroids = vec![pts[0].clone(), pts[1].clone(), pts[2].clone()];
        let mut labels = vec![0usize; pts.len()];
        for _ in 0..50 {
            for (i, p) in pts.iter().enumerate() {
                labels[i] = nearest(&centroids, p).0;
            }
            let mut sums = vec![DVector::zeros(2); 3];
            let mut counts = [0usize; 3];
            for (i, p) in pts.iter().enumerate() {
                sums[labels[i]] += p;
                counts[labels[i]] += 1;
            }
            for c in 0..3 {
                if counts[c] > 0 {
                    centroids[c] = &sums[c] / counts[c] as f64;
                }
            }
        }
        let bad_inertia: f64 = pts.iter().map(|p| nearest(&centroids, p).1).sum();
        assert!(km.inertia <= bad_inertia + 1e-9, "k-means++ ({}) should not be worse than the bad init ({bad_inertia})", km.inertia);
    }

    #[test]
    fn it_is_deterministic() {
        let (pts, _) = blobs();
        let a = kmeans(&pts, 3, 50, 42);
        let b = kmeans(&pts, 3, 50, 42);
        assert_eq!(a.labels, b.labels, "same seed should give identical labels");
    }
}
