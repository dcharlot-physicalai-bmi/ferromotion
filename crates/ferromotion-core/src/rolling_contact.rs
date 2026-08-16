//! **Rolling contact kinematics** — how a contact point migrates across two surfaces.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §5.6, Theorem 5.11
//! (Montana's equations of contact). Real grasping is mostly *moving* contacts: fingers are surfaces, and
//! manipulating an object rolls them across it. This gives the derivative of the **contact coordinates** as a
//! function of the relative body velocity — the piece a fixed-contact grasp model cannot express and in-hand
//! manipulation is built on.
//!
//! # Contact coordinates
//!
//! `η = (α_f, α_o, ψ)`: the chart coordinates of the contact point on the finger and on the object, plus the
//! **contact angle** `ψ` between the two surfaces' `∂c/∂u` directions. Five scalars, and Montana's theorem
//! gives all five derivatives.
//!
//! # Geometric parameters
//!
//! Each surface contributes `(M, K, T)` at the contact — metric tensor, curvature form, torsion form (MLS
//! eq. 5.17–5.24). `M` normalises tangent vectors for the chart's scaling, `K` measures how the unit normal
//! turns across the surface, `T` measures how the Gauss frame twists. A flat surface has `K = 0` and `T = 0`.
//!
//! # The equations (MLS eq. 5.28)
//!
//! ```text
//! α̇_f = M_f⁻¹ (K_f + K̃_o)⁻¹ ( [−ω_y; ω_x] − K̃_o [v_x; v_y] )
//! α̇_o = M_o⁻¹ R_ψ (K_f + K̃_o)⁻¹ ( [−ω_y; ω_x] + K_f [v_x; v_y] )
//! ψ̇   = ω_z + T_f M_f α̇_f + T_o M_o α̇_o
//! 0    = v_z
//! ```
//!
//! with `R_ψ = [[cos ψ, −sin ψ], [−sin ψ, −cos ψ]]` and `K̃_o = R_ψ K_o R_ψ` — the object's curvature expressed
//! in the finger's contact axes. `K_f + K̃_o` is the **relative curvature form**.
//!
//! **The relative curvature must be invertible, and that is a modelling constraint rather than a numerical
//! one.** MLS is explicit: it goes singular when a convex and a concave surface share a radius of curvature,
//! and there small object motions cause unbounded contact motion — continuity is genuinely lost, not merely
//! ill-conditioned. [`contact_derivative`] returns `None` there rather than a large number, because a large
//! number would be a fiction.
//!
//! `v_z = 0` is the contact-maintenance condition and is *checked*, not assumed: a non-zero normal velocity
//! means the surfaces are separating or interpenetrating and the contact coordinates are not defined.

use nalgebra::{Matrix2, Vector2, Vector3};

/// The geometric parameters of a surface at a contact point: `(M, K, T)`.
#[derive(Clone, Copy, Debug)]
pub struct GeometricParams {
    /// Metric tensor `M`, the positive-definite square root of the first fundamental form.
    pub m: Matrix2<f64>,
    /// Curvature form `K`.
    pub k: Matrix2<f64>,
    /// Torsion form `T`, a **row** vector (MLS notes this explicitly).
    pub t: Vector2<f64>,
}

impl GeometricParams {
    /// A plane: unit metric, no curvature, no torsion.
    pub fn plane() -> Self {
        Self { m: Matrix2::identity(), k: Matrix2::zeros(), t: Vector2::zeros() }
    }

    /// A sphere of radius `r`, at a chart point where the parameterisation is locally orthonormal.
    ///
    /// Curvature `K = (1/r)·I`: a sphere turns its normal at the same rate in every tangent direction, which
    /// is what makes it the clean analytic case. Sign convention follows MLS, where a convex surface has
    /// positive curvature.
    pub fn sphere(r: f64) -> Self {
        Self { m: Matrix2::identity(), k: Matrix2::identity() / r, t: Vector2::zeros() }
    }
}

/// Contact coordinate derivatives `(α̇_f, α̇_o, ψ̇)`.
#[derive(Clone, Copy, Debug)]
pub struct ContactRates {
    pub alpha_f_dot: Vector2<f64>,
    pub alpha_o_dot: Vector2<f64>,
    pub psi_dot: f64,
}

/// `R_ψ = [[cos ψ, −sin ψ], [−sin ψ, −cos ψ]]` (MLS §5.6.2).
///
/// Note this is **not** a rotation matrix — its determinant is `−1`. It is the orientation of the finger's
/// contact axes relative to the object's, which includes a reflection because the two surfaces face each
/// other. Writing a plain rotation here is the natural mistake and produces plausible wrong answers.
pub fn r_psi(psi: f64) -> Matrix2<f64> {
    let (c, s) = (psi.cos(), psi.sin());
    Matrix2::new(c, -s, -s, -c)
}

/// **Montana's equations of contact** — MLS Theorem 5.11, eq. (5.28).
///
/// `vel` is the relative **body** velocity of the finger's local frame with respect to the object's, as
/// `(v_x, v_y, v_z)`; `omega` likewise `(ω_x, ω_y, ω_z)`. In that frame `(ω_x, ω_y)` are the rolling rates in
/// the tangent plane, `ω_z` the spin about the contact normal, `(v_x, v_y)` the sliding velocities, and `v_z`
/// the normal velocity.
///
/// Returns `None` when the relative curvature `K_f + K̃_o` is singular (see the module note — this is a real
/// loss of continuity), or when `v_z` is non-zero beyond `tol`, meaning the surfaces are not in contact.
///
/// Pure **rolling** is `v_x = v_y = 0, ω_z = 0`; pure **sliding** is `ω_x = ω_y = ω_z = 0`.
pub fn contact_derivative(
    finger: &GeometricParams,
    object: &GeometricParams,
    psi: f64,
    vel: Vector3<f64>,
    omega: Vector3<f64>,
    tol: f64,
) -> Option<ContactRates> {
    if vel.z.abs() > tol {
        return None; // v_z != 0: separating or interpenetrating, contact coordinates undefined
    }
    let rp = r_psi(psi);
    let k_tilde_o = rp * object.k * rp; // K̃_o, the object's curvature in the finger's contact axes
    let relative = finger.k + k_tilde_o;
    let rel_inv = relative.try_inverse()?; // singular relative curvature

    let roll = Vector2::new(-omega.y, omega.x);
    let slide = Vector2::new(vel.x, vel.y);

    let m_f_inv = finger.m.try_inverse()?;
    let m_o_inv = object.m.try_inverse()?;

    let alpha_f_dot = m_f_inv * (rel_inv * (roll - k_tilde_o * slide));
    let alpha_o_dot = m_o_inv * (rp * (rel_inv * (roll + finger.k * slide)));
    // ψ̇ = ω_z + T_f M_f α̇_f + T_o M_o α̇_o — the torsion forms are row vectors, hence the dots.
    let psi_dot = omega.z
        + finger.t.dot(&(finger.m * alpha_f_dot))
        + object.t.dot(&(object.m * alpha_o_dot));

    Some(ContactRates { alpha_f_dot, alpha_o_dot, psi_dot })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The physical oracle: a sphere rolling without slipping on a plane traverses at `r·ω`.**
    ///
    /// This is the no-slip condition, independent of Montana's equations, so it checks the implementation
    /// rather than restating it. With the plane as the object (`K_o = 0`) the relative curvature is `(1/r)·I`,
    /// so its inverse contributes exactly the factor `r`.
    #[test]
    fn a_sphere_rolling_on_a_plane_traverses_at_r_omega() {
        let plane = GeometricParams::plane();
        for r in [0.02, 0.05, 0.25, 1.0] {
            let sphere = GeometricParams::sphere(r);
            for w in [0.5, 2.0, -3.0] {
                // pure rolling about the contact-frame y axis: no sliding, no spin
                let rates = contact_derivative(
                    &sphere,
                    &plane,
                    0.0,
                    Vector3::zeros(),
                    Vector3::new(0.0, w, 0.0),
                    1e-12,
                )
                .expect("a sphere on a plane has invertible relative curvature");

                // |α̇_o| is the speed of the contact point across the PLANE, and no-slip fixes it at r*|w|.
                let speed = rates.alpha_o_dot.norm();
                assert!(
                    (speed - r * w.abs()).abs() < 1e-12,
                    "r={r} w={w}: contact traversed at {speed}, no-slip requires {}",
                    r * w.abs()
                );
                // Rolling about y moves the contact along x, not y.
                assert!(rates.alpha_o_dot.y.abs() < 1e-12, "rolling about y must not move v");
                // Neither surface has torsion and there is no spin, so ψ is stationary.
                assert!(rates.psi_dot.abs() < 1e-12, "psi should not drift: {}", rates.psi_dot);
                // The contact migrates across the SPHERE at the same rate, `r·|w|`, and that is a statement
                // about the CHART rather than about the physics. `GeometricParams::sphere` uses `M = I`, so
                // its chart is locally arclength-parameterised: chart distance IS arc length, and the arc
                // swept by rolling through `w` is `r·w`. A first version of this test asserted `|w|`, which
                // would be right only for an ANGULAR chart — where the radius would instead appear in `M`.
                // Both surfaces measure arc here, so both see `r·|w|`.
                assert!(
                    (rates.alpha_f_dot.norm() - r * w.abs()).abs() < 1e-12,
                    "arclength chart: the contact should sweep the sphere at r*|w| = {}, got {}",
                    r * w.abs(),
                    rates.alpha_f_dot.norm()
                );
            }
        }
    }

    /// Rolling faster moves the contact faster, and a bigger ball moves it further per radian. Both are
    /// linear, which is the structure of eq. (5.28) — worth pinning because a stray metric factor breaks it.
    #[test]
    fn contact_speed_is_linear_in_both_rate_and_radius() {
        let plane = GeometricParams::plane();
        let speed = |r: f64, w: f64| {
            contact_derivative(
                &GeometricParams::sphere(r),
                &plane,
                0.0,
                Vector3::zeros(),
                Vector3::new(0.0, w, 0.0),
                1e-12,
            )
            .unwrap()
            .alpha_o_dot
            .norm()
        };
        let base = speed(0.05, 1.0);
        assert!((speed(0.05, 3.0) - 3.0 * base).abs() < 1e-12, "linear in rate");
        assert!((speed(0.15, 1.0) - 3.0 * base).abs() < 1e-12, "linear in radius");
    }

    /// **Singular relative curvature is refused, not approximated.** MLS's own example: a convex surface
    /// against a concave one of the same radius. There, small object motions cause unbounded contact motion —
    /// continuity is lost, so a large finite answer would be a fiction.
    #[test]
    fn a_matched_convex_concave_pair_is_refused() {
        let r = 0.05;
        let convex = GeometricParams::sphere(r);
        // Concave of the same radius: curvature of equal magnitude and opposite sign, so K_f + K̃_o = 0.
        let concave = GeometricParams { m: Matrix2::identity(), k: -Matrix2::identity() / r, t: Vector2::zeros() };
        // At psi = 0, R_psi = diag(1, -1), so K̃_o = R K R has the same diagonal as K — the sum vanishes.
        let k_tilde = r_psi(0.0) * concave.k * r_psi(0.0);
        assert!((convex.k + k_tilde).norm() < 1e-12, "the fixture must actually be singular");
        assert!(
            contact_derivative(&convex, &concave, 0.0, Vector3::zeros(), Vector3::new(0.0, 1.0, 0.0), 1e-12)
                .is_none(),
            "a matched convex/concave pair has singular relative curvature and must be refused"
        );
        // A mismatched pair is fine — the singularity is the coincidence of radii, not concavity itself.
        let concave_other = GeometricParams { m: Matrix2::identity(), k: -Matrix2::identity() / (2.0 * r), t: Vector2::zeros() };
        assert!(
            contact_derivative(&convex, &concave_other, 0.0, Vector3::zeros(), Vector3::new(0.0, 1.0, 0.0), 1e-12)
                .is_some(),
            "different radii are not singular"
        );
    }

    /// Two flat surfaces have zero curvature, so the relative curvature is singular and rolling is undefined —
    /// which is correct: a plane on a plane cannot roll, it can only slide.
    #[test]
    fn plane_on_plane_cannot_roll() {
        let p = GeometricParams::plane();
        assert!(
            contact_derivative(&p, &p, 0.0, Vector3::zeros(), Vector3::new(0.0, 1.0, 0.0), 1e-12).is_none(),
            "flat on flat has K_f + K̃_o = 0"
        );
    }

    /// `v_z != 0` means the surfaces are separating or interpenetrating: the contact coordinates are not
    /// defined and MLS's fourth equation, `0 = v_z`, is a condition rather than an output.
    #[test]
    fn a_non_zero_normal_velocity_is_refused() {
        let sphere = GeometricParams::sphere(0.05);
        let plane = GeometricParams::plane();
        let omega = Vector3::new(0.0, 1.0, 0.0);
        assert!(contact_derivative(&sphere, &plane, 0.0, Vector3::new(0.0, 0.0, 0.01), omega, 1e-12).is_none());
        assert!(contact_derivative(&sphere, &plane, 0.0, Vector3::new(0.0, 0.0, -0.01), omega, 1e-12).is_none());
        // Exactly in contact is accepted.
        assert!(contact_derivative(&sphere, &plane, 0.0, Vector3::zeros(), omega, 1e-12).is_some());
    }

    /// `R_ψ` has determinant `−1`: it is not a rotation. The two surfaces face each other, so the mapping
    /// between their contact axes includes a reflection. Substituting a plain rotation is the natural error and
    /// gives plausible wrong answers, so pin it.
    #[test]
    fn r_psi_is_a_reflection_not_a_rotation() {
        for psi in [0.0, 0.3, 1.2, -2.0, std::f64::consts::PI] {
            let m = r_psi(psi);
            assert!((m.determinant() + 1.0).abs() < 1e-12, "det R_psi should be -1, got {}", m.determinant());
            // still orthogonal, so it preserves lengths
            assert!((m * m.transpose() - Matrix2::identity()).norm() < 1e-12, "R_psi must be orthogonal");
            // and it is an involution at every psi: R_psi * R_psi = I
            assert!((m * m - Matrix2::identity()).norm() < 1e-12, "R_psi should be its own inverse");
        }
    }

    /// Pure sliding: no rolling rates at all. The contact still migrates, driven by `K̃_o` and `K_f` acting on
    /// the sliding velocity, which is the second term of each equation and would be missed by a test that only
    /// ever rolls.
    #[test]
    fn pure_sliding_still_moves_the_contact() {
        let sphere = GeometricParams::sphere(0.05);
        let plane = GeometricParams::plane();
        let slide = Vector3::new(0.02, 0.0, 0.0);
        let rates = contact_derivative(&sphere, &plane, 0.0, slide, Vector3::zeros(), 1e-12).unwrap();
        // On the plane side K_f * slide drives it: with K_f = I/r and (K_f + K̃_o)⁻¹ = r·I, α̇_o = R_ψ · slide.
        assert!(
            (rates.alpha_o_dot.norm() - slide.xy().norm()).abs() < 1e-12,
            "sliding should carry the plane-side contact at the slide speed, got {}",
            rates.alpha_o_dot.norm()
        );
        // The finger-side term is −K̃_o · slide, and K̃_o = 0 for a plane object, so the sphere-side contact
        // does not move under pure sliding. That asymmetry is real and is what distinguishes rolling.
        assert!(
            rates.alpha_f_dot.norm() < 1e-12,
            "with a flat object, pure sliding should not migrate the contact on the finger: {}",
            rates.alpha_f_dot.norm()
        );
    }
}
