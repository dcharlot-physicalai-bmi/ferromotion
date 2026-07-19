//! **Attitude determination from vector observations** — Wahba's problem (Wahba, 1965): recover a rigid
//! body's orientation from two or more direction measurements expressed in both a reference frame (star
//! catalog, sun ephemeris, magnetic-field model) and the body frame (star tracker, sun sensor,
//! magnetometer). Find the rotation `A` minimizing `½ Σ aᵢ‖bᵢ − A rᵢ‖²`.
//!
//! Two classic solvers, the workhorses of spacecraft attitude determination:
//! - [`triad`] — the **deterministic** two-vector solution: build a right-handed triad from each pair and
//!   read off the rotation between them. Fast, exact for two clean vectors, uses no optimization.
//! - [`davenport_q_method`] — the **optimal least-squares** solution (Davenport's q-method, the exact form
//!   QUEST approximates): the best-fit quaternion is the top eigenvector of the 4×4 Davenport matrix `K`.
//!   Uses *all* observations with weights, so it beats TRIAD under noise.
//!
//! This pairs with [`crate::Cw`] rendezvous to round out the spacecraft GNC toolset (determine attitude,
//! then control relative motion). Verified: both recover a known rotation from clean observations to machine
//! precision; the q-method's Wahba cost is optimal; and with noisy measurements the multi-vector q-method
//! beats two-vector TRIAD. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Matrix4, Quaternion, SymmetricEigen, UnitQuaternion, Vector3};

/// **TRIAD** — deterministic attitude from two reference/body vector pairs (`r1`,`r2` in the reference
/// frame; `b1`,`b2` the same directions measured in the body frame). Returns the rotation `A` with
/// `bᵢ ≈ A·rᵢ`. The first pair is trusted for the primary axis; the second fixes the remaining freedom.
pub fn triad(r1: &Vector3<f64>, r2: &Vector3<f64>, b1: &Vector3<f64>, b2: &Vector3<f64>) -> Matrix3<f64> {
    // reference triad
    let t1r = r1.normalize();
    let t2r = t1r.cross(r2).normalize();
    let t3r = t1r.cross(&t2r);
    // body triad
    let t1b = b1.normalize();
    let t2b = t1b.cross(b2).normalize();
    let t3b = t1b.cross(&t2b);
    let mr = Matrix3::from_columns(&[t1r, t2r, t3r]);
    let mb = Matrix3::from_columns(&[t1b, t2b, t3b]);
    mb * mr.transpose()
}

/// **Davenport q-method** — the optimal least-squares solution to Wahba's problem over any number of
/// observations with weights. `refs[i]` is direction `i` in the reference frame, `obs[i]` the same
/// direction in the body frame, `weights[i]` its confidence. Returns the attitude quaternion `A` with
/// `obs ≈ A·refs`. This is the exact eigenvector problem QUEST solves approximately.
pub fn davenport_q_method(refs: &[Vector3<f64>], obs: &[Vector3<f64>], weights: &[f64]) -> UnitQuaternion<f64> {
    // attitude profile matrix B = Σ aᵢ bᵢ rᵢᵀ
    let mut b = Matrix3::zeros();
    for ((r, o), &a) in refs.iter().zip(obs).zip(weights) {
        b += a * o * r.transpose();
    }
    let s = b + b.transpose();
    let sigma = b.trace();
    // z = Σ aᵢ (bᵢ × rᵢ), read off the antisymmetric part of B
    let z = Vector3::new(b[(1, 2)] - b[(2, 1)], b[(2, 0)] - b[(0, 2)], b[(0, 1)] - b[(1, 0)]);

    // Davenport K = [[S − σI, z], [zᵀ, σ]] (4×4 symmetric); optimal q is its top eigenvector.
    let mut k = Matrix4::zeros();
    let s_minus = s - Matrix3::identity() * sigma;
    k.fixed_view_mut::<3, 3>(0, 0).copy_from(&s_minus);
    k.fixed_view_mut::<3, 1>(0, 3).copy_from(&z);
    k.fixed_view_mut::<1, 3>(3, 0).copy_from(&z.transpose());
    k[(3, 3)] = sigma;

    let eig = SymmetricEigen::new(k);
    // pick the eigenvector for the largest eigenvalue
    let mut best = 0;
    for i in 1..4 {
        if eig.eigenvalues[i] > eig.eigenvalues[best] {
            best = i;
        }
    }
    let v = eig.eigenvectors.column(best);
    // Eigenvector ordering is [q_vec; q_scalar]. Davenport/Markley's attitude matrix A(q) uses the
    // −2q₄[q_vec×] convention, i.e. A(q) = R_hamilton(q)ᵀ; nalgebra's Quaternion is Hamilton (+2w[v×]), so
    // we conjugate the vector part to make nalgebra's rotation equal Davenport's A (body ≈ A·reference).
    let q = Quaternion::new(v[3], -v[0], -v[1], -v[2]);
    UnitQuaternion::from_quaternion(q)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fixed, non-trivial reference rotation (deterministic — no RNG).
    fn true_rotation() -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(Vector3::new(0.3, -0.7, 0.5)), 1.1)
    }

    // Three distinct reference directions.
    fn refs() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(1.0, 0.2, -0.1).normalize(),
            Vector3::new(-0.3, 1.0, 0.4).normalize(),
            Vector3::new(0.2, -0.5, 1.0).normalize(),
        ]
    }

    #[test]
    fn triad_recovers_a_known_rotation_exactly() {
        // THE ORACLE (deterministic solver). Clean two-vector observations ⇒ exact rotation.
        let a = true_rotation().to_rotation_matrix();
        let r = refs();
        let b: Vec<_> = r.iter().map(|v| a * v).collect();
        let est = triad(&r[0], &r[1], &b[0], &b[1]);
        assert!((est - a.matrix()).norm() < 1e-12, "TRIAD should recover A: err {}", (est - a.matrix()).norm());
    }

    #[test]
    fn the_q_method_recovers_a_known_rotation_exactly() {
        // THE ORACLE (optimal solver). Clean multi-vector observations ⇒ exact quaternion.
        let a = true_rotation();
        let r = refs();
        let b: Vec<_> = r.iter().map(|v| a * v).collect();
        let est = davenport_q_method(&r, &b, &[1.0, 1.0, 1.0]);
        // quaternion sign is arbitrary; compare rotation matrices
        assert!((est.to_rotation_matrix().matrix() - a.to_rotation_matrix().matrix()).norm() < 1e-12,
            "q-method should recover A: err {}", (est.to_rotation_matrix().matrix() - a.to_rotation_matrix().matrix()).norm());
    }

    #[test]
    fn the_q_method_solution_minimizes_the_wahba_cost() {
        // Optimality: the returned attitude has lower Wahba loss than any nearby perturbed attitude, even
        // with (deterministic) measurement noise breaking exact recovery.
        let a = true_rotation();
        let r = refs();
        // deterministic small perturbations to the observations
        let noise = [Vector3::new(0.02, -0.01, 0.015), Vector3::new(-0.012, 0.018, -0.008), Vector3::new(0.009, 0.011, -0.02)];
        let b: Vec<_> = r.iter().zip(&noise).map(|(v, dn)| (a * v + dn).normalize()).collect();
        let w = [1.0, 1.0, 1.0];
        let cost = |att: &UnitQuaternion<f64>| -> f64 {
            r.iter().zip(&b).map(|(rr, bb)| (bb - att * rr).norm_squared()).sum::<f64>()
        };
        let est = davenport_q_method(&r, &b, &w);
        let c_est = cost(&est);
        // perturb the estimate in several directions; none should do better
        for axis in [Vector3::x(), Vector3::y(), Vector3::z()] {
            for &d in &[0.05_f64, -0.05] {
                let perturbed = UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), d) * est;
                assert!(cost(&perturbed) >= c_est - 1e-12, "q-method must be the local optimum: {} < {}", cost(&perturbed), c_est);
            }
        }
    }

    #[test]
    fn the_multivector_q_method_beats_two_vector_triad_under_noise() {
        // THE HEADLINE / value prop. With noisy measurements, the least-squares q-method using ALL THREE
        // observations recovers the attitude more accurately than TRIAD using only two.
        let a = true_rotation();
        let am = a.to_rotation_matrix();
        let r = refs();
        let noise = [Vector3::new(0.03, -0.02, 0.01), Vector3::new(-0.02, 0.025, -0.015), Vector3::new(0.018, 0.012, -0.028)];
        let b: Vec<_> = r.iter().zip(&noise).map(|(v, dn)| (am * v + dn).normalize()).collect();

        let triad_est = triad(&r[0], &r[1], &b[0], &b[1]);
        let q_est = davenport_q_method(&r, &b, &[1.0, 1.0, 1.0]).to_rotation_matrix();

        let triad_err = (triad_est - am.matrix()).norm();
        let q_err = (q_est.matrix() - am.matrix()).norm();
        assert!(q_err < triad_err, "q-method (all 3 vectors) should beat TRIAD (2 vectors): q {q_err} vs triad {triad_err}");
    }
}
