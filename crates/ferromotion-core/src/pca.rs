//! **Principal component analysis & oriented bounding boxes**. PCA finds the orthogonal axes along which a
//! point cloud varies most — the eigenvectors of its covariance, ordered by variance. Aligning a box to
//! those axes gives an **oriented bounding box (OBB)** that hugs an elongated or tilted cloud far more
//! tightly than an axis-aligned box, which matters for broad-phase collision, grasp/part alignment, shape
//! fitting, and pose initialization from raw points (e.g. a segmented object from a depth sensor).
//!
//! Exact and deterministic (symmetric-eigen of the `3×3` covariance). Verified: PCA recovers the axes and
//! per-axis variances of a cloud generated with a known orientation; the OBB contains every point; and for a
//! tilted, elongated cloud the OBB is strictly tighter than the axis-aligned box. Pure `nalgebra` →
//! WASM-clean.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

/// PCA of a point cloud: the centroid, the principal axes (matrix **columns**, ordered by descending
/// variance, right-handed), and the variance along each axis.
pub struct Pca {
    pub centroid: Vector3<f64>,
    pub axes: Matrix3<f64>,
    pub variances: Vector3<f64>,
}

/// Compute the PCA of `points` (needs at least one point).
pub fn pca(points: &[Vector3<f64>]) -> Pca {
    let n = points.len().max(1) as f64;
    let centroid = points.iter().sum::<Vector3<f64>>() / n;
    let mut cov = Matrix3::zeros();
    for p in points {
        let d = p - centroid;
        cov += d * d.transpose();
    }
    cov /= n;
    let eig = SymmetricEigen::new(cov);
    // order eigenpairs by descending eigenvalue
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
    let mut axes = Matrix3::zeros();
    for (col, &src) in idx.iter().enumerate() {
        axes.set_column(col, &eig.eigenvectors.column(src).into_owned());
    }
    // make right-handed
    if axes.determinant() < 0.0 {
        axes.set_column(2, &(-axes.column(2)));
    }
    let variances = Vector3::new(eig.eigenvalues[idx[0]], eig.eigenvalues[idx[1]], eig.eigenvalues[idx[2]]);
    Pca { centroid, axes, variances }
}

/// An oriented bounding box: centre, orthonormal axes (columns), and half-extents along each axis.
#[derive(Clone, Copy, Debug)]
pub struct Obb {
    pub center: Vector3<f64>,
    pub axes: Matrix3<f64>,
    pub half_extents: Vector3<f64>,
}

impl Obb {
    /// Whether a point lies inside the box (within `eps`).
    pub fn contains(&self, p: &Vector3<f64>) -> bool {
        let local = self.axes.transpose() * (p - self.center);
        (0..3).all(|i| local[i].abs() <= self.half_extents[i] + 1e-9)
    }

    /// The box volume.
    pub fn volume(&self) -> f64 {
        8.0 * self.half_extents.x * self.half_extents.y * self.half_extents.z
    }

    /// The eight corners.
    pub fn corners(&self) -> [Vector3<f64>; 8] {
        let mut c = [Vector3::zeros(); 8];
        for (k, corner) in c.iter_mut().enumerate() {
            let sx = if k & 1 == 0 { -1.0 } else { 1.0 };
            let sy = if k & 2 == 0 { -1.0 } else { 1.0 };
            let sz = if k & 4 == 0 { -1.0 } else { 1.0 };
            let local = Vector3::new(sx * self.half_extents.x, sy * self.half_extents.y, sz * self.half_extents.z);
            *corner = self.center + self.axes * local;
        }
        c
    }
}

/// Fit an oriented bounding box to a point cloud via its PCA axes.
pub fn obb_from_points(points: &[Vector3<f64>]) -> Obb {
    let p = pca(points);
    let (mut lo, mut hi) = (Vector3::repeat(f64::INFINITY), Vector3::repeat(f64::NEG_INFINITY));
    for pt in points {
        let local = p.axes.transpose() * (pt - p.centroid);
        for i in 0..3 {
            lo[i] = lo[i].min(local[i]);
            hi[i] = hi[i].max(local[i]);
        }
    }
    let center_local = (lo + hi) / 2.0;
    Obb { center: p.centroid + p.axes * center_local, axes: p.axes, half_extents: (hi - lo) / 2.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }
    fn so3(phi: &Vector3<f64>) -> Matrix3<f64> {
        let th = phi.norm();
        if th < 1e-12 {
            return Matrix3::identity();
        }
        let k = phi / th;
        let kx = skew(&k);
        Matrix3::identity() + th.sin() * kx + (1.0 - th.cos()) * kx * kx
    }

    // a cloud stretched (3, 1, 0.3) along the axes of rotation R, centred at `c`
    fn cloud(r: &Matrix3<f64>, c: &Vector3<f64>) -> Vec<Vector3<f64>> {
        let mut seed = 1u64;
        let mut u = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64) - 0.5
        };
        (0..400).map(|_| c + r * Vector3::new(3.0 * u(), 1.0 * u(), 0.3 * u())).collect()
    }

    #[test]
    fn pca_recovers_the_cloud_axes_and_variances() {
        // THE ORACLE. The principal axis aligns with the most-stretched direction R·e_x, and variances rank
        // 3 > 1 > 0.3 (uniform spread s over width w has variance w²/12).
        let r = so3(&Vector3::new(0.3, -0.5, 0.2));
        let p = pca(&cloud(&r, &Vector3::new(2.0, -1.0, 0.5)));
        // principal axis parallel to R·e_x (sign-agnostic)
        let cos = p.axes.column(0).dot(&r.column(0)).abs();
        assert!(cos > 0.99, "principal axis should align with R·e_x: cos {cos}");
        assert!(p.variances[0] > p.variances[1] && p.variances[1] > p.variances[2], "variances should be ordered: {:?}", p.variances);
        // variance of a uniform spread of half-width 1.5 is (3)²/12 = 0.75
        assert!((p.variances[0] - 0.75).abs() < 0.15, "top variance ≈ 0.75, got {}", p.variances[0]);
    }

    #[test]
    fn the_obb_contains_every_point() {
        let r = so3(&Vector3::new(0.7, 0.1, -0.4));
        let pts = cloud(&r, &Vector3::new(-1.0, 2.0, 3.0));
        let obb = obb_from_points(&pts);
        for p in &pts {
            assert!(obb.contains(p), "point {p:?} outside its OBB");
        }
        // axes are orthonormal
        assert!((obb.axes.transpose() * obb.axes - Matrix3::identity()).norm() < 1e-9, "axes should be orthonormal");
    }

    #[test]
    fn the_obb_is_tighter_than_the_axis_aligned_box() {
        // THE VALUE PROP. For a tilted, elongated cloud the OBB volume is well below the AABB volume.
        let r = so3(&Vector3::new(0.6, -0.6, 0.5));
        let pts = cloud(&r, &Vector3::zeros());
        let obb = obb_from_points(&pts);
        let (mut lo, mut hi) = (Vector3::repeat(f64::INFINITY), Vector3::repeat(f64::NEG_INFINITY));
        for p in &pts {
            for i in 0..3 {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let aabb_vol = (hi - lo).iter().product::<f64>();
        assert!(obb.volume() < 0.7 * aabb_vol, "OBB should be tighter: {} vs AABB {aabb_vol}", obb.volume());
    }

    #[test]
    fn contains_rejects_a_far_exterior_point() {
        let obb = obb_from_points(&cloud(&Matrix3::identity(), &Vector3::zeros()));
        assert!(!obb.contains(&Vector3::new(100.0, 0.0, 0.0)), "far point should be outside");
        assert!(obb.contains(&obb.center), "centre is inside");
    }
}
