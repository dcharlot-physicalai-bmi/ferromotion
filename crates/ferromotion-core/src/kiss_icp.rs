//! **KISS-ICP** (Vizzo, Guadagnino, Mersch, Wiesmann, Behley, Stachniss, RA-L 2023) — LiDAR/point-cloud
//! **odometry**: estimate the sensor's pose over a stream of scans by registering each against a local map.
//! Where [`crate::Icp`] is a single-shot alignment, this is the full loop that turns registration into an
//! odometry system, built on the [`crate::KdTree`] spatial index. Its "keep-it-simple" recipe: (1) a
//! **constant-velocity motion model** predicts the next pose and seeds ICP; (2) plain **point-to-point**
//! ICP against a **voxel-downsampled local map** (no normals, no features); (3) an **adaptive
//! correspondence threshold** that tightens as the estimate improves.
//!
//! Verified by recovering a known sensor trajectory (translation + rotation) from synthetic scans of a
//! static scene. Pure `nalgebra` → WASM-clean. (Intra-scan motion deskew — which needs per-point
//! timestamps — is the one KISS-ICP piece left out here; the CV model, local map, and adaptive threshold
//! are the odometry core.)

use crate::icp::umeyama;
use crate::spatial::KdTree;
use nalgebra::{Matrix3, Vector3};

type Pose = (Matrix3<f64>, Vector3<f64>); // (rotation, translation): world = R·local + t

fn compose(a: &Pose, b: &Pose) -> Pose {
    (a.0 * b.0, a.0 * b.1 + a.1)
}
fn invert(a: &Pose) -> Pose {
    let rt = a.0.transpose();
    (rt, -(rt * a.1))
}

/// A KISS-ICP odometry estimator.
#[derive(Clone, Debug)]
pub struct KissIcp {
    pub voxel_size: f64,
    /// Base (max) correspondence distance; the adaptive threshold never exceeds this.
    pub max_corr: f64,
    pub iters: usize,
    map: Vec<Vector3<f64>>,
    pose: Pose,
    velocity: Pose, // last frame-to-frame motion (for constant-velocity prediction)
    poses: Vec<Pose>,
}

impl KissIcp {
    pub fn new(voxel_size: f64, max_corr: f64) -> KissIcp {
        KissIcp {
            voxel_size,
            max_corr,
            iters: 20,
            map: Vec::new(),
            pose: (Matrix3::identity(), Vector3::zeros()),
            velocity: (Matrix3::identity(), Vector3::zeros()),
            poses: Vec::new(),
        }
    }

    /// Register a new scan (points in the sensor frame) and return the estimated sensor pose (sensor→world).
    pub fn register(&mut self, scan: &[Vector3<f64>]) -> Pose {
        if self.map.is_empty() {
            self.pose = (Matrix3::identity(), Vector3::zeros());
            self.insert_scan(scan, &self.pose.clone());
            self.poses.push(self.pose);
            return self.pose;
        }
        // constant-velocity prediction seeds ICP
        let mut est = compose(&self.pose, &self.velocity);
        let tree = KdTree::build(self.map.clone());
        let mut threshold = self.max_corr;
        for _ in 0..self.iters {
            // transform scan into the world by the current estimate, match against the map
            let mut src = Vec::new();
            let mut dst = Vec::new();
            let mut residuals = Vec::new();
            for p in scan {
                let pw = est.0 * p + est.1;
                if let Some((j, d)) = tree.nearest(&pw)
                    && d < threshold
                {
                    src.push(pw);
                    dst.push(tree.point(j));
                    residuals.push(d);
                }
            }
            if src.len() < 3 {
                break;
            }
            let (dr, dt) = umeyama(&src, &dst); // aligns current-world → corrected-world
            est = compose(&(dr, dt), &est);
            // adaptive threshold: shrink toward 3× the median residual
            let mut r = residuals.clone();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = r[r.len() / 2];
            threshold = (3.0 * med).clamp(self.voxel_size, self.max_corr);
            if dt.norm() + (dr - Matrix3::identity()).norm() < 1e-9 {
                break;
            }
        }
        self.velocity = compose(&invert(&self.pose), &est);
        self.pose = est;
        self.insert_scan(scan, &est);
        self.poses.push(est);
        est
    }

    /// Voxel-downsample the scan (world frame) and add it to the local map.
    fn insert_scan(&mut self, scan: &[Vector3<f64>], pose: &Pose) {
        use std::collections::HashSet;
        let mut seen: HashSet<[i64; 3]> = self.map.iter().map(|p| self.cell(p)).collect();
        for p in scan {
            let pw = pose.0 * p + pose.1;
            let c = self.cell(&pw);
            if seen.insert(c) {
                self.map.push(pw);
            }
        }
    }

    fn cell(&self, p: &Vector3<f64>) -> [i64; 3] {
        [(p.x / self.voxel_size).floor() as i64, (p.y / self.voxel_size).floor() as i64, (p.z / self.voxel_size).floor() as i64]
    }

    /// The estimated pose trajectory.
    pub fn trajectory(&self) -> &[Pose] {
        &self.poses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screw::exp_so3;

    fn scene() -> Vec<Vector3<f64>> {
        let mut seed = 0xA5A5A5A5u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) * 8.0 - 4.0
        };
        (0..400).map(|_| Vector3::new(rng(), rng(), rng())).collect()
    }

    // Ground-truth sensor pose at step k: a gentle screw motion.
    fn gt_pose(k: usize) -> Pose {
        let r = exp_so3(&Vector3::new(0.0, 0.0, 0.04 * k as f64));
        let t = Vector3::new(0.12 * k as f64, 0.05 * k as f64, 0.0);
        (r, t)
    }

    #[test]
    fn it_recovers_a_known_sensor_trajectory() {
        // THE HEADLINE. A sensor moves on a known trajectory through a static scene; feeding it the scans
        // (scene expressed in each sensor frame), KISS-ICP odometry recovers the poses.
        let world = scene();
        let mut odom = KissIcp::new(0.15, 1.5);
        let steps = 8;
        for k in 0..steps {
            let (r, t) = gt_pose(k);
            let (ri, ti) = invert(&(r, t)); // world → sensor
            let scan: Vec<Vector3<f64>> = world.iter().map(|p| ri * p + ti).collect();
            let (er, et) = odom.register(&scan);
            // recovered pose should match ground truth (frame 0 fixes the gauge, and gt(0)=I)
            assert!((et - t).norm() < 0.05, "step {k}: translation {et} vs gt {t}");
            assert!((er - r).abs().max() < 0.02, "step {k}: rotation error {}", (er - r).abs().max());
        }
        assert_eq!(odom.trajectory().len(), steps);
    }

    #[test]
    fn the_local_map_grows_but_stays_downsampled() {
        let world = scene();
        let mut odom = KissIcp::new(0.3, 1.5);
        for k in 0..5 {
            let (r, t) = gt_pose(k);
            let (ri, ti) = invert(&(r, t));
            let scan: Vec<Vector3<f64>> = world.iter().map(|p| ri * p + ti).collect();
            odom.register(&scan);
        }
        // the voxel-downsampled map has fewer points than 5 full scans
        assert!(odom.map.len() < 5 * world.len(), "map should be downsampled");
        assert!(odom.map.len() > 50, "map should have accumulated structure");
    }
}
