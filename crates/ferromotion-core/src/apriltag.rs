//! **Fiducial-marker decode & pose** (AprilTag / ArUco). Fiducial markers are the ubiquitous ground-truth
//! and manipulation-target tool: a printed square whose interior encodes an ID with an **error-correcting
//! code**, seen by a camera to give both the tag's identity and its full 6-DoF pose. This module implements
//! the two *algorithmically clean* halves — the parts with exact oracles:
//!
//! - **Payload decode**: read the interior bit grid, and over all four 90° rotations find the codebook entry
//!   within the code's Hamming-correction radius — recovering the tag **ID** and its **orientation**, and
//!   rejecting corrupted payloads beyond the correction distance.
//! - **Tag pose**: from the four detected corner pixels and the camera intrinsics, recover the tag's pose by
//!   [`crate::homography`] estimation + decomposition (a planar target), refined by [`crate::pnp`]'s
//!   Gauss–Newton — reusing the crate's calibrated-vision stack.
//!
//! (Finding the quad in raw pixels — thresholding, edge/segment grouping, quad fitting — is image-processing
//! that has no analytic oracle and is out of scope here; feed this module the corners a detector produces.)
//! Verified: a known ID round-trips through encode→decode, survives sub-radius bit corruption, and is
//! rejected beyond it; and a tag rendered at a known pose is recovered from its projected corners. Pure
//! `nalgebra` → WASM-clean.

use crate::camera::PinholeCamera;
use crate::pnp::pnp_gn;
use nalgebra::{Matrix3, Vector2, Vector3};

/// Read an `n×n` boolean bit grid (row-major) into a code word (bit 0 = grid[0]).
fn grid_to_code(bits: &[bool]) -> u64 {
    bits.iter().enumerate().filter(|(_, b)| **b).fold(0u64, |c, (i, _)| c | (1u64 << i))
}

/// Rotate an `n×n` grid 90° clockwise.
fn rotate90(bits: &[bool], n: usize) -> Vec<bool> {
    let mut out = vec![false; n * n];
    for r in 0..n {
        for c in 0..n {
            out[c * n + (n - 1 - r)] = bits[r * n + c];
        }
    }
    out
}

/// Decode a fiducial payload: the `n×n` bit grid is matched (over all 4 rotations) against `codebook`,
/// accepting a codeword within `max_errors` Hamming distance. Returns `(id, rotation)` — the codebook index
/// and the number of 90° clockwise rotations needed to bring the observed grid to the canonical code — or
/// `None` if no codeword is close enough.
pub fn decode_payload(bits: &[bool], n: usize, codebook: &[u64], max_errors: u32) -> Option<(usize, usize)> {
    let mut grid = bits.to_vec();
    for rot in 0..4 {
        let code = grid_to_code(&grid);
        for (id, &cw) in codebook.iter().enumerate() {
            if (code ^ cw).count_ones() <= max_errors {
                return Some((id, rot));
            }
        }
        grid = rotate90(&grid, n);
    }
    None
}

/// Decompose a planar homography (calibrated / normalized image coordinates) into a pose `(R, t)` with the
/// target on its own `Z = 0` plane. `H` maps target `(X, Y, 1)` to the image ray.
fn homography_to_pose(h: &Matrix3<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let h1 = h.column(0);
    let h2 = h.column(1);
    let h3 = h.column(2);
    let lambda = 1.0 / h1.norm();
    let mut r1 = h1 * lambda;
    let mut r2 = h2 * lambda;
    let mut t = h3 * lambda;
    // put the target in front of the camera
    if t.z < 0.0 {
        r1 = -r1;
        r2 = -r2;
        t = -t;
    }
    let r3 = r1.cross(&r2);
    let r0 = Matrix3::from_columns(&[r1, r2, r3]);
    // nearest rotation (orthonormalize) via SVD
    let svd = r0.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    (u * d * vt, t)
}

/// Recover a square tag's pose `(R, t)` (`X_cam = R·X_tag + t`) from its four **corner pixels** (in the
/// tag's corner order) and the physical `tag_size` (edge length), given the camera intrinsics. The tag
/// corners in the tag frame are `(±s/2, ±s/2, 0)`.
pub fn tag_pose(corners: [Vector2<f64>; 4], tag_size: f64, camera: &PinholeCamera) -> (Matrix3<f64>, Vector3<f64>) {
    let s = tag_size / 2.0;
    // tag-frame corners (Z=0), same order as the pixel corners
    let obj = [Vector2::new(-s, -s), Vector2::new(s, -s), Vector2::new(s, s), Vector2::new(-s, s)];
    // normalized image points (undistorted bearings, z=1 → take x,y)
    let norm: Vec<Vector2<f64>> = corners.iter().map(|px| { let r = camera.unproject(px); Vector2::new(r.x, r.y) }).collect();
    // homography from planar tag → image, then decompose to a pose
    let h = crate::homography::homography_dlt(&obj, &norm);
    let (r0, t0) = homography_to_pose(&h);
    // refine with PnP Gauss–Newton over the 4 correspondences
    let obj3: Vec<Vector3<f64>> = obj.iter().map(|p| Vector3::new(p.x, p.y, 0.0)).collect();
    pnp_gn(&obj3, &norm, r0, t0, 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }
    fn so3(w: Vector3<f64>) -> Matrix3<f64> {
        let th = w.norm();
        if th < 1e-12 {
            return Matrix3::identity();
        }
        let k = w / th;
        let kx = skew(&k);
        Matrix3::identity() + th.sin() * kx + (1.0 - th.cos()) * kx * kx
    }

    #[test]
    fn a_known_id_round_trips_and_survives_correctable_corruption() {
        // THE ORACLE. A codebook with large pairwise Hamming distance: encode → decode recovers the ID;
        // flipping up to `max_errors` bits still decodes; flipping more rejects.
        let n = 6;
        let codebook = [0x0000_0000u64, 0x0FFF_0FFF, 0x3333_CCCC, 0xAAAA_5555];
        let id = 2;
        let cw = codebook[id];
        let grid: Vec<bool> = (0..n * n).map(|i| (cw >> i) & 1 == 1).collect();
        assert_eq!(decode_payload(&grid, n, &codebook, 3), Some((2, 0)), "clean decode");
        // corrupt 2 bits (within radius)
        let mut corrupt = grid.clone();
        corrupt[0] = !corrupt[0];
        corrupt[5] = !corrupt[5];
        assert_eq!(decode_payload(&corrupt, n, &codebook, 3), Some((2, 0)), "2-bit corruption still decodes");
        // a random pattern far from every codeword is rejected
        let junk: Vec<bool> = (0..n * n).map(|i| i % 3 == 0).collect();
        assert_eq!(decode_payload(&junk, n, &codebook, 3), None, "junk should be rejected");
    }

    #[test]
    fn decode_recovers_the_rotation() {
        // Rotating the observed grid by 90° should still decode, reporting the rotation count that
        // re-aligns it.
        let n = 4;
        let codebook = [0b1011_0010_1101_0110u64];
        let grid: Vec<bool> = (0..n * n).map(|i| (codebook[0] >> i) & 1 == 1).collect();
        // observe it rotated 90° counter-clockwise = 3 clockwise rotations bring it back
        let mut observed = grid.clone();
        for _ in 0..3 {
            observed = rotate90(&observed, n);
        }
        let (id, rot) = decode_payload(&observed, n, &codebook, 0).expect("rotated tag decodes");
        assert_eq!(id, 0);
        assert_eq!(rot, 1, "one more clockwise rotation realigns it");
    }

    #[test]
    fn tag_pose_is_recovered_from_projected_corners() {
        // THE VISION ORACLE. Render a tag at a known pose, project its corners, and recover the pose.
        let cam = PinholeCamera::new(600.0, 600.0, 320.0, 240.0);
        let r_true = so3(Vector3::new(0.1, -0.25, 0.15));
        let t_true = Vector3::new(0.05, -0.03, 1.2);
        let size = 0.16;
        let s = size / 2.0;
        let obj = [Vector3::new(-s, -s, 0.0), Vector3::new(s, -s, 0.0), Vector3::new(s, s, 0.0), Vector3::new(-s, s, 0.0)];
        let corners: [Vector2<f64>; 4] = std::array::from_fn(|i| cam.project(&(r_true * obj[i] + t_true)));
        let (r, t) = tag_pose(corners, size, &cam);
        assert!((r - r_true).norm() < 1e-5, "rotation error {}", (r - r_true).norm());
        assert!((t - t_true).norm() < 1e-6, "translation error {}", (t - t_true).norm());
    }
}
