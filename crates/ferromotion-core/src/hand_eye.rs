//! **Hand-eye calibration** (`AX = XB`) — recover the unknown rigid transform `X ∈ SE(3)` between two
//! rigidly-coupled frames (classically a robot gripper and a camera mounted on it, but equally any two
//! sensors on one body) from **pairs of their relative motions**: as the body moves, the gripper undergoes
//! motion `Aᵢ` and the camera `Bᵢ`, and every pair satisfies `Aᵢ X = X Bᵢ`. Solving it is the prerequisite
//! to fusing a robot's kinematics with what its camera sees, and the same equation calibrates any
//! multi-sensor rig.
//!
//! The classic decoupling (Tsai–Lenz / Shiu–Ahmad): the rotation equation `R_{Aᵢ} R_X = R_X R_{Bᵢ}` means
//! `A` and `B` are conjugate rotations, so their axis-angle vectors satisfy `R_X · log(R_{Bᵢ}) = log(R_{Aᵢ})`
//! — a **rotation-from-vector-pairs** problem (Kabsch/Wahba, reusing the crate's [`crate::wahba`] machinery).
//! With `R_X` known, translation follows from the **linear** system `(R_{Aᵢ} − I) t_X = R_X t_{Bᵢ} − t_{Aᵢ}`
//! stacked over motions. Needs `≥ 2` motions with non-parallel rotation axes. Verified: from motions
//! synthesized with a known `X`, it recovers `X` to solver precision. Pure `nalgebra` → WASM-clean.

use crate::screw::{log_so3, pose, rot_of, trans_of};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Vector3};

// rotation aligning source vectors to destination vectors: R minimizing Σ‖dstᵢ − R srcᵢ‖² (Kabsch).
fn rotation_from_pairs(src: &[Vector3<f64>], dst: &[Vector3<f64>]) -> Matrix3<f64> {
    let mut h = Matrix3::zeros();
    for (s, d) in src.iter().zip(dst) {
        h += d * s.transpose();
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut w = Matrix3::identity();
    w[(2, 2)] = (u * vt).determinant().signum();
    u * w * vt
}

/// Solve the hand-eye problem `Aᵢ X = X Bᵢ` for `X` from motion pairs `(Aᵢ, Bᵢ)` (each a `4×4` SE(3)
/// homogeneous transform). Returns `X`.
pub fn hand_eye_calibration(motions: &[(Matrix4<f64>, Matrix4<f64>)]) -> Matrix4<f64> {
    // --- rotation: R_X maps each B rotation axis to the corresponding A rotation axis ---
    let a_axes: Vec<Vector3<f64>> = motions.iter().map(|(a, _)| log_so3(&rot_of(a))).collect();
    let b_axes: Vec<Vector3<f64>> = motions.iter().map(|(_, b)| log_so3(&rot_of(b))).collect();
    let r_x = rotation_from_pairs(&b_axes, &a_axes);

    // --- translation: stack (R_Aᵢ − I) t_X = R_X t_Bᵢ − t_Aᵢ and least-squares solve ---
    let n = motions.len();
    let mut m = DMatrix::zeros(3 * n, 3);
    let mut d = DVector::zeros(3 * n);
    for (i, (a, b)) in motions.iter().enumerate() {
        let block = rot_of(a) - Matrix3::identity();
        let rhs = r_x * trans_of(b) - trans_of(a);
        for r in 0..3 {
            for c in 0..3 {
                m[(3 * i + r, c)] = block[(r, c)];
            }
            d[3 * i + r] = rhs[r];
        }
    }
    let mtm = m.transpose() * &m;
    let t_x = mtm.try_inverse().map(|inv| inv * m.transpose() * d).unwrap_or_else(|| DVector::zeros(3));
    pose(&r_x, &Vector3::new(t_x[0], t_x[1], t_x[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screw::exp_so3;

    fn se3(w: Vector3<f64>, t: Vector3<f64>) -> Matrix4<f64> {
        pose(&exp_so3(&w), &t)
    }

    #[test]
    fn it_recovers_a_known_hand_eye_transform() {
        // THE ORACLE. Fix a true X; synthesize gripper motions Bᵢ and set Aᵢ = X Bᵢ X⁻¹ (which satisfies
        // AᵢX = XBᵢ). Recover X.
        let x_true = se3(Vector3::new(0.3, -0.5, 0.4), Vector3::new(0.1, 0.05, -0.2));
        let x_inv = x_true.try_inverse().unwrap();
        // motions with non-parallel rotation axes
        let bs = [
            se3(Vector3::new(0.5, 0.1, 0.0), Vector3::new(0.2, -0.1, 0.3)),
            se3(Vector3::new(0.0, 0.4, 0.2), Vector3::new(-0.1, 0.2, 0.1)),
            se3(Vector3::new(0.1, -0.2, 0.5), Vector3::new(0.05, 0.15, -0.05)),
            se3(Vector3::new(-0.3, 0.2, 0.1), Vector3::new(0.1, 0.0, 0.2)),
        ];
        let motions: Vec<(Matrix4<f64>, Matrix4<f64>)> = bs.iter().map(|&b| (x_true * b * x_inv, b)).collect();
        let x = hand_eye_calibration(&motions);
        assert!((x - x_true).abs().max() < 1e-8, "recovered X off by {}", (x - x_true).abs().max());
    }

    #[test]
    fn two_motions_suffice() {
        let x_true = se3(Vector3::new(0.2, 0.3, -0.1), Vector3::new(-0.05, 0.1, 0.2));
        let x_inv = x_true.try_inverse().unwrap();
        let bs = [se3(Vector3::new(0.4, 0.0, 0.1), Vector3::new(0.1, 0.2, -0.1)), se3(Vector3::new(0.0, 0.3, -0.2), Vector3::new(-0.2, 0.05, 0.15))];
        let motions: Vec<(Matrix4<f64>, Matrix4<f64>)> = bs.iter().map(|&b| (x_true * b * x_inv, b)).collect();
        let x = hand_eye_calibration(&motions);
        assert!((x - x_true).abs().max() < 1e-7, "two-motion recovery off by {}", (x - x_true).abs().max());
    }
}
