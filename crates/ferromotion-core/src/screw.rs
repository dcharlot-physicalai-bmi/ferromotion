//! **Screw theory & the SE(3) Lie group** (Lynch & Park, *Modern Robotics*; Brockett's product of
//! exponentials) — the algebraic substrate the rest of the kinematics/dynamics stack is written in:
//! rigid-body poses as elements of SE(3), velocities as **twists** and forces as **wrenches** in se(3),
//! the exponential/logarithm between them, the **adjoint** that moves a twist between frames, the little
//! adjoint (Lie bracket), and **product-of-exponentials** forward kinematics.
//!
//! Convention: a twist is `ξ = [ω; v] ∈ ℝ⁶` (angular part first), a pose is a homogeneous `4×4` matrix `T`,
//! and `exp([ξ])` is the screw motion. Everything is exact (Rodrigues + the closed-form SE(3) map) and
//! verified by the group laws themselves — `exp∘log = id`, `[Ad_T ξ]∧ = T[ξ]∧T⁻¹`, the bracket, and PoE
//! forward kinematics matching an explicit chain. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Matrix4, Matrix6, Vector3, Vector6};

const EPS: f64 = 1e-9;

/// Skew-symmetric matrix `[ω]×` of a 3-vector (`hat` on so(3)).
pub fn hat3(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

/// Inverse of [`hat3`]: the axis vector of a skew-symmetric matrix.
pub fn vee3(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

/// `exp` on SO(3): rotation matrix of a rotation vector `ω` (axis·angle), via Rodrigues' formula.
pub fn exp_so3(w: &Vector3<f64>) -> Matrix3<f64> {
    let theta = w.norm();
    if theta < EPS {
        return Matrix3::identity() + hat3(w); // 1st-order; exact as θ→0
    }
    let k = hat3(&(w / theta));
    Matrix3::identity() + theta.sin() * k + (1.0 - theta.cos()) * (k * k)
}

/// `log` on SO(3): the rotation vector of a rotation matrix.
pub fn log_so3(r: &Matrix3<f64>) -> Vector3<f64> {
    let tr = r.trace();
    let cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos.acos();
    if theta < EPS {
        return vee3(&((r - r.transpose()) * 0.5));
    }
    let axis = vee3(&((r - r.transpose()) * (0.5 / theta.sin())));
    axis * theta
}

/// Assemble a homogeneous pose from a rotation and a translation.
pub fn pose(r: &Matrix3<f64>, p: &Vector3<f64>) -> Matrix4<f64> {
    let mut t = Matrix4::identity();
    t.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    t.fixed_view_mut::<3, 1>(0, 3).copy_from(p);
    t
}

/// Rotation part of a pose.
pub fn rot_of(t: &Matrix4<f64>) -> Matrix3<f64> {
    t.fixed_view::<3, 3>(0, 0).into()
}
/// Translation part of a pose.
pub fn trans_of(t: &Matrix4<f64>) -> Vector3<f64> {
    t.fixed_view::<3, 1>(0, 3).into()
}

/// `exp` on SE(3): the pose produced by a twist `ξ = [ω; v]` (a screw motion).
pub fn exp_se3(xi: &Vector6<f64>) -> Matrix4<f64> {
    let w = Vector3::new(xi[0], xi[1], xi[2]);
    let v = Vector3::new(xi[3], xi[4], xi[5]);
    let theta = w.norm();
    let r = exp_so3(&w);
    if theta < EPS {
        return pose(&r, &v);
    }
    let k = hat3(&(w / theta));
    // V = I + (1−cosθ)/θ · K + (θ−sinθ)/θ · K²  (so that p = V·v)
    let vmat = Matrix3::identity() + (1.0 - theta.cos()) / theta * k + (theta - theta.sin()) / theta * (k * k);
    pose(&r, &(vmat * v))
}

/// `log` on SE(3): the twist `ξ = [ω; v]` such that `exp([ξ]) = T`.
pub fn log_se3(t: &Matrix4<f64>) -> Vector6<f64> {
    let r = rot_of(t);
    let p = trans_of(t);
    let w = log_so3(&r);
    let theta = w.norm();
    let v = if theta < EPS {
        p
    } else {
        let k = hat3(&(w / theta));
        // V⁻¹ = I − ½[w]× + (1 − (θ/2)·cot(θ/2))·K²
        let cot = 1.0 / (theta / 2.0).tan();
        let vinv = Matrix3::identity() - 0.5 * hat3(&w) + (1.0 - (theta / 2.0) * cot) * (k * k);
        vinv * p
    };
    Vector6::new(w.x, w.y, w.z, v.x, v.y, v.z)
}

/// The **adjoint** `Ad_T` (6×6): maps a twist expressed in one frame to another, `ξ_a = Ad_{T_ab} ξ_b`.
/// For `[ω; v]`: `Ad_T = [[R, 0], [[p]×R, R]]`.
pub fn adjoint(t: &Matrix4<f64>) -> Matrix6<f64> {
    let r = rot_of(t);
    let p = trans_of(t);
    let mut ad = Matrix6::zeros();
    ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&(hat3(&p) * r));
    ad
}

/// The **little adjoint** `ad_ξ` (6×6): the Lie bracket operator, `ad_x y = [x, y]`.
/// For `[ω; v]`: `ad_ξ = [[[ω]×, 0], [[v]×, [ω]×]]`.
pub fn ad(xi: &Vector6<f64>) -> Matrix6<f64> {
    let w = Vector3::new(xi[0], xi[1], xi[2]);
    let v = Vector3::new(xi[3], xi[4], xi[5]);
    let mut m = Matrix6::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat3(&w));
    m.fixed_view_mut::<3, 3>(3, 3).copy_from(&hat3(&w));
    m.fixed_view_mut::<3, 3>(3, 0).copy_from(&hat3(&v));
    m
}

/// A screw axis `S = [ω; v]` for a revolute joint about a unit `axis` passing through point `q` (space
/// frame): `ω = axis`, `v = −axis × q`.
pub fn revolute_axis(axis: &Vector3<f64>, q: &Vector3<f64>) -> Vector6<f64> {
    let a = axis.normalize();
    let v = -a.cross(q);
    Vector6::new(a.x, a.y, a.z, v.x, v.y, v.z)
}

/// **Product-of-exponentials** forward kinematics (space form): `T(θ) = ∏ exp([Sᵢ]θᵢ) · M`, where `Sᵢ`
/// are space-frame screw axes and `M` is the end-effector home pose.
pub fn poe_fk(screws: &[Vector6<f64>], theta: &[f64], m_home: &Matrix4<f64>) -> Matrix4<f64> {
    let mut t = Matrix4::identity();
    for (s, &th) in screws.iter().zip(theta) {
        t *= exp_se3(&(s * th));
    }
    t * m_home
}

/// **Screw linear interpolation (ScLERP)** between poses `a` and `b` at `t ∈ [0, 1]`:
/// `T(t) = a · exp(t · log(a⁻¹ b))`. By Chasles' theorem `a⁻¹ b` is a screw motion (simultaneous rotation
/// about and translation along one axis); ScLERP traverses it at constant twist — the SE(3) analogue of
/// quaternion SLERP, and the frame-independent, minimum-twist way to blend Cartesian keyframes. Straight-
/// line position + independently-slerped orientation, by contrast, is kinked and frame-dependent.
pub fn sclerp(a: &Matrix4<f64>, b: &Matrix4<f64>, t: f64) -> Matrix4<f64> {
    let rel = a.try_inverse().expect("pose invertible") * b;
    a * exp_se3(&(log_se3(&rel) * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &Matrix4<f64>, b: &Matrix4<f64>, tol: f64) -> bool {
        (a - b).abs().max() < tol
    }

    fn sample(seed: f64) -> Matrix4<f64> {
        let w = Vector3::new(0.3 * seed, -0.5 * seed, 0.2 + 0.1 * seed);
        pose(&exp_so3(&w), &Vector3::new(1.0 + seed, -2.0 * seed, 0.5))
    }

    #[test]
    fn sclerp_hits_both_endpoints() {
        let (a, b) = (sample(0.5), sample(1.3));
        assert!(approx(&sclerp(&a, &b, 0.0), &a, 1e-12), "T(0) = A");
        assert!(approx(&sclerp(&a, &b, 1.0), &b, 1e-9), "T(1) = B");
    }

    #[test]
    fn sclerp_moves_at_a_constant_screw_velocity() {
        // THE DEFINING PROPERTY. Equal-time samples are related by the SAME incremental transform
        // T(tᵢ)⁻¹ T(tᵢ₊₁) = exp(Δt·ξ): a constant screw twist along the whole path.
        let (a, b) = (sample(0.4), sample(1.1));
        let dt = 0.1;
        let mut reference: Option<Vector6<f64>> = None;
        let mut ti = 0.0;
        while ti < 1.0 - 1e-9 {
            let step = sclerp(&a, &b, ti).try_inverse().unwrap() * sclerp(&a, &b, ti + dt);
            let xi = log_se3(&step);
            match &reference {
                None => reference = Some(xi),
                Some(r) => assert!((xi - r).norm() < 1e-9, "screw step not constant at t={ti}"),
            }
            ti += dt;
        }
    }

    #[test]
    fn sclerp_pure_translation_is_a_straight_line() {
        let a = pose(&Matrix3::identity(), &Vector3::zeros());
        let b = pose(&Matrix3::identity(), &Vector3::new(4.0, -2.0, 6.0));
        for &t in &[0.25, 0.5, 0.75] {
            let mid = sclerp(&a, &b, t);
            assert!((trans_of(&mid) - Vector3::new(4.0, -2.0, 6.0) * t).norm() < 1e-12, "straight line at t={t}");
            assert!((rot_of(&mid) - Matrix3::identity()).abs().max() < 1e-12, "no spurious rotation");
        }
    }

    #[test]
    fn exp_log_round_trip_on_se3() {
        // exp∘log = id and log∘exp = id — the defining group-map identities.
        let xis = [
            Vector6::new(0.3, -0.5, 0.7, 1.2, -0.4, 0.9),
            Vector6::new(0.0, 0.0, 0.0, 0.5, -0.2, 0.1), // pure translation
            Vector6::new(0.01, -0.02, 0.005, 0.3, 0.1, -0.2), // near-zero rotation
        ];
        for xi in xis {
            let t = exp_se3(&xi);
            let back = log_se3(&t);
            assert!((back - xi).norm() < 1e-9, "log∘exp: {back} vs {xi}");
            // and exp∘log on the pose
            assert!(approx(&exp_se3(&back), &t, 1e-9), "exp∘log pose mismatch");
        }
    }

    #[test]
    fn exp_so3_matches_nalgebra_rotation() {
        use nalgebra::Rotation3;
        let w = Vector3::new(0.4, -0.7, 0.2);
        let r = exp_so3(&w);
        let rn = Rotation3::new(w);
        assert!((r - rn.matrix()).abs().max() < 1e-12, "Rodrigues vs nalgebra");
        assert!((log_so3(&r) - w).norm() < 1e-12, "log_so3 round trip");
    }

    #[test]
    fn the_adjoint_conjugates_the_twist() {
        // THE INVARIANT. [Ad_T ξ]∧ = T [ξ]∧ T⁻¹ — the adjoint is conjugation in the Lie algebra.
        let t = exp_se3(&Vector6::new(0.2, 0.5, -0.3, 0.7, -0.1, 0.4));
        let xi = Vector6::new(-0.3, 0.2, 0.6, 0.4, 0.8, -0.5);
        let ad_xi = adjoint(&t) * xi;
        // build [ξ]∧ (4×4) and conjugate
        let hat = |x: &Vector6<f64>| {
            let mut h = Matrix4::zeros();
            h.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat3(&Vector3::new(x[0], x[1], x[2])));
            h.fixed_view_mut::<3, 1>(0, 3).copy_from(&Vector3::new(x[3], x[4], x[5]));
            h
        };
        let conj = t * hat(&xi) * t.try_inverse().unwrap();
        assert!((hat(&ad_xi) - conj).abs().max() < 1e-10, "Ad_T ξ ≠ T[ξ]T⁻¹");
    }

    #[test]
    fn the_little_adjoint_is_the_lie_bracket() {
        // ad_x y = [x,y] = ([x]∧[y]∧ − [y]∧[x]∧)∨
        let x = Vector6::new(0.2, -0.4, 0.5, 0.1, 0.6, -0.3);
        let y = Vector6::new(-0.1, 0.3, 0.2, 0.7, -0.2, 0.4);
        let bracket = ad(&x) * y;
        let hat = |v: &Vector6<f64>| {
            let mut h = Matrix4::zeros();
            h.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat3(&Vector3::new(v[0], v[1], v[2])));
            h.fixed_view_mut::<3, 1>(0, 3).copy_from(&Vector3::new(v[3], v[4], v[5]));
            h
        };
        let comm = hat(&x) * hat(&y) - hat(&y) * hat(&x);
        // read the twist back out of the 4×4 commutator
        let br_w = vee3(&comm.fixed_view::<3, 3>(0, 0).into());
        let br_v = Vector3::new(comm[(0, 3)], comm[(1, 3)], comm[(2, 3)]);
        let expect = Vector6::new(br_w.x, br_w.y, br_w.z, br_v.x, br_v.y, br_v.z);
        assert!((bracket - expect).norm() < 1e-12, "ad_x y ≠ bracket: {bracket} vs {expect}");
    }

    #[test]
    fn poe_forward_kinematics_matches_an_explicit_planar_arm() {
        // A 2R planar arm: PoE must reproduce the closed-form end-effector pose.
        let (l1, l2) = (1.0, 0.7);
        let z = Vector3::new(0.0, 0.0, 1.0);
        let s1 = revolute_axis(&z, &Vector3::zeros());
        let s2 = revolute_axis(&z, &Vector3::new(l1, 0.0, 0.0));
        let m = pose(&Matrix3::identity(), &Vector3::new(l1 + l2, 0.0, 0.0));
        for &(t1, t2) in &[(0.3, -0.5), (1.2, 0.8), (-0.6, 0.4)] {
            let t = poe_fk(&[s1, s2], &[t1, t2], &m);
            let ee = trans_of(&t);
            let ex = l1 * t1.cos() + l2 * (t1 + t2).cos();
            let ey = l1 * t1.sin() + l2 * (t1 + t2).sin();
            assert!((ee.x - ex).abs() < 1e-10 && (ee.y - ey).abs() < 1e-10, "PoE ee ({},{}) vs explicit ({ex},{ey})", ee.x, ee.y);
            // orientation is the summed joint angle about z
            let ang = log_so3(&rot_of(&t)).z;
            assert!((ang - (t1 + t2)).abs() < 1e-10, "PoE orientation {ang} vs {}", t1 + t2);
        }
    }

    #[test]
    fn the_wrench_twist_pairing_is_frame_invariant() {
        // Power = F·V is a scalar independent of frame: with V_a = Ad_T V_b and F_b = Ad_Tᵀ F_a,
        // F_a·V_a = F_b·V_b.
        let t = exp_se3(&Vector6::new(0.4, -0.2, 0.6, 0.5, 0.3, -0.7));
        let v_b = Vector6::new(0.2, 0.5, -0.3, 0.1, -0.4, 0.6); // twist in frame b
        let f_a = Vector6::new(-0.3, 0.2, 0.5, 0.7, -0.1, 0.4); // wrench in frame a
        let v_a = adjoint(&t) * v_b;
        let f_b = adjoint(&t).transpose() * f_a;
        assert!((f_a.dot(&v_a) - f_b.dot(&v_b)).abs() < 1e-12, "power not frame-invariant");
    }
}
