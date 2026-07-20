//! **Quaternion (rotation) averaging** — Markley's eigenvector method (Markley, Cheng, Crassidis & Oshman,
//! 2007). Averaging orientations is not the same as averaging their quaternion components: a quaternion and
//! its negation `q` and `−q` are the *same* rotation, so a naive component mean can cancel to nonsense, and
//! it ignores the unit-norm constraint. The correct weighted mean maximizes `Σ wᵢ (qᵀqᵢ)²`, whose solution
//! is the **dominant eigenvector** of the `4×4` accumulation matrix `M = Σ wᵢ qᵢ qᵢᵀ`. Because each term uses
//! the outer product `qᵢ qᵢᵀ`, the double-cover sign ambiguity disappears automatically.
//!
//! This is the standard tool for fusing several attitude estimates (multi-sensor, multi-camera calibration),
//! blending particle-filter orientation samples, and smoothing rotation tracks. Distinct from
//! [`crate::wahba`], which solves attitude *from vector observations*. Verified: the average of identical
//! rotations is that rotation; the average of rotations placed symmetrically about a mean recovers the mean;
//! flipping input signs leaves the result unchanged; and weighting biases toward the heavier rotation. Pure
//! `nalgebra` → WASM-clean.

use nalgebra::{Matrix4, SymmetricEigen, UnitQuaternion, Vector4};

/// The weighted average rotation of `quats` (weights default to 1). Returns the Markley mean — the dominant
/// eigenvector of `Σ wᵢ qᵢ qᵢᵀ` as a unit quaternion.
pub fn average_quaternions(quats: &[UnitQuaternion<f64>], weights: Option<&[f64]>) -> UnitQuaternion<f64> {
    assert!(!quats.is_empty(), "need at least one quaternion");
    let mut m = Matrix4::zeros();
    for (i, q) in quats.iter().enumerate() {
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        let qq = q.quaternion();
        let v = Vector4::new(qq.w, qq.i, qq.j, qq.k);
        m += w * v * v.transpose();
    }
    let eig = SymmetricEigen::new(m);
    // eigenvector for the largest eigenvalue
    let mut best = 0;
    for i in 1..4 {
        if eig.eigenvalues[i] > eig.eigenvalues[best] {
            best = i;
        }
    }
    let v = eig.eigenvectors.column(best);
    UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[0], v[1], v[2], v[3]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn q(axis: Vector3<f64>, angle: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
    }

    #[test]
    fn the_average_of_identical_rotations_is_that_rotation() {
        // THE ORACLE (degenerate). N copies of one rotation average to it.
        let r = q(Vector3::new(0.3, -0.7, 0.5), 1.1);
        let avg = average_quaternions(&[r, r, r, r], None);
        assert!(avg.angle_to(&r) < 1e-9, "average should equal the input: off by {}", avg.angle_to(&r));
    }

    #[test]
    fn it_recovers_the_mean_of_symmetric_rotations() {
        // Rotations placed symmetrically about a central orientation (rotated ±δ about a common axis) should
        // average back to the centre.
        let center = q(Vector3::new(0.2, 0.4, -0.3), 0.8);
        let ax = Vector3::new(1.0, -0.5, 0.2);
        let plus = center * q(ax, 0.25);
        let minus = center * q(ax, -0.25);
        let avg = average_quaternions(&[plus, minus], None);
        assert!(avg.angle_to(&center) < 1e-6, "symmetric average should be the centre: off by {}", avg.angle_to(&center));
    }

    #[test]
    fn it_is_invariant_to_quaternion_sign() {
        // q and −q are the same rotation; flipping signs of some inputs must not change the average.
        let a = q(Vector3::new(1.0, 0.0, 0.0), 0.5);
        let b = q(Vector3::new(0.0, 1.0, 0.0), 0.9);
        let c = q(Vector3::new(0.0, 0.0, 1.0), 1.3);
        let base = average_quaternions(&[a, b, c], None);
        // negate b's quaternion (same rotation)
        let bn = UnitQuaternion::from_quaternion(-b.quaternion());
        let flipped = average_quaternions(&[a, bn, c], None);
        assert!(base.angle_to(&flipped) < 1e-9, "sign flip changed the average by {}", base.angle_to(&flipped));
    }

    #[test]
    fn weighting_biases_toward_the_heavier_rotation() {
        // A heavy weight on one rotation pulls the mean close to it.
        let a = q(Vector3::new(0.0, 0.0, 1.0), 0.0); // identity
        let b = q(Vector3::new(0.0, 0.0, 1.0), 1.0); // 1 rad about z
        let avg = average_quaternions(&[a, b], Some(&[100.0, 1.0]));
        assert!(avg.angle_to(&a) < avg.angle_to(&b), "heavy weight on a should pull the mean toward a");
        assert!(avg.angle_to(&a) < 0.1, "mean should be very close to the heavily-weighted rotation: {}", avg.angle_to(&a));
    }
}
