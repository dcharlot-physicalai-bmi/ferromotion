//! **von Mises plasticity** — permanent deformation, so a squeezed object stays squeezed.
//!
//! [`crate::FemSim`] is purely hyperelastic: every deformation is recoverable, so a gripper can crush a can
//! and the can springs back perfectly. That is wrong for most of what a robot manipulates — metal, clay, foam
//! past its cell-collapse point, and any object where "did I damage it?" is the question the grasp planner
//! should be answering.
//!
//! This is the standard multiplicative split `F = F_e · F_p` with a **von Mises yield surface** and a **radial
//! return** mapping, computed in principal **Hencky** (logarithmic) strain space — the formulation used across
//! computational plasticity and its graphics descendants, and the one whose return map is a projection rather
//! than an iteration.
//!
//! # The algorithm
//!
//! 1. SVD the trial deformation gradient, `F = U Σ Vᵀ`.
//! 2. Take principal logarithmic strains `ε = ln σᵢ`. Logs are what make the split additive: a multiplicative
//!    `F_e·F_p` becomes `ε_e + ε_p`.
//! 3. Split off the deviatoric part, `ε_dev = ε − tr(ε)/3`.
//! 4. Yield when `‖ε_dev‖ > τ_Y/(2μ)`. Below that the step is **exactly elastic** and `F_e = F` unchanged.
//! 5. Above it, scale `ε_dev` back onto the yield surface — radial return — and rebuild
//!    `F_e = U exp(ε_new) Vᵀ`. Whatever was removed becomes `F_p`.
//!
//! # Two properties that make this checkable rather than plausible
//!
//! * **Plastic flow is volume-preserving, exactly.** The return map only touches the deviatoric part, so the
//!   trace of the plastic strain is zero and `det(F_p) = 1` to machine precision. Not approximately — this is
//!   the defining property of `J2` flow, and it is a sharp test that catches a mis-split trace immediately.
//! * **Hydrostatic loading never yields, at any magnitude.** Compress uniformly by 10× and von Mises does not
//!   care, because `ε_dev = 0`. A pressure-dependent criterion (Drucker-Prager, for soil) *would* yield, so
//!   this test also distinguishes which criterion is implemented.
//!
//! Rate-independent and perfectly plastic: no hardening, no rate dependence. Both are real extensions;
//! neither is guessed at here.

use nalgebra::Matrix3;

/// Result of a return-mapping step.
#[derive(Clone, Copy, Debug)]
pub struct ReturnMap {
    /// Elastic part of the deformation gradient — what the stress should be computed from.
    pub f_elastic: Matrix3<f64>,
    /// Plastic increment absorbed by this step. `det ≈ 1`.
    pub f_plastic: Matrix3<f64>,
    /// Whether the trial state was outside the yield surface.
    pub yielded: bool,
    /// Magnitude of deviatoric Hencky strain removed. Zero when elastic.
    pub plastic_strain_increment: f64,
}

/// The yield threshold in deviatoric Hencky strain: `τ_Y / (2μ)`.
///
/// Expressing the criterion as a strain rather than a stress is what makes the return map a straight
/// projection in `ε` space instead of a nonlinear solve.
pub fn yield_strain(tau_y: f64, mu: f64) -> f64 {
    tau_y / (2.0 * mu)
}

/// **Radial-return mapping onto the von Mises yield surface.**
///
/// `f_trial` is the deformation gradient the elastic predictor produced; `tau_y` the yield stress; `mu` the
/// shear modulus. Returns the elastic/plastic split.
///
/// Below yield this returns `f_trial` **unchanged and bit-identical**, with `f_plastic = I` — an elastic
/// simulation must not drift merely because a plasticity check is present.
///
/// Returns `None` if `f_trial` is singular or inverted (`det ≤ 0`), where the logarithmic strain is undefined.
/// That is the same domain restriction [`crate::FemSim`]'s `ln J` term has, and refusing is better than
/// returning a complex-valued strain's real part.
pub fn radial_return(f_trial: &Matrix3<f64>, tau_y: f64, mu: f64) -> Option<ReturnMap> {
    if f_trial.determinant() <= 0.0 {
        return None; // log strain undefined for an inverted or degenerate element
    }
    let svd = f_trial.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let s = svd.singular_values;
    if s.iter().any(|x| *x <= 0.0) {
        return None;
    }

    // Principal Hencky strains. The log is what turns the multiplicative split into an additive one.
    let eps = [s[0].ln(), s[1].ln(), s[2].ln()];
    let trace = eps[0] + eps[1] + eps[2];
    let mean = trace / 3.0;
    let dev = [eps[0] - mean, eps[1] - mean, eps[2] - mean];
    let dev_norm = (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]).sqrt();

    let y = yield_strain(tau_y, mu);
    if dev_norm <= y || dev_norm == 0.0 {
        // Elastic: hand back exactly what came in.
        return Some(ReturnMap {
            f_elastic: *f_trial,
            f_plastic: Matrix3::identity(),
            yielded: false,
            plastic_strain_increment: 0.0,
        });
    }

    // Radial return: shrink the deviatoric part onto the surface, keep the volumetric part untouched. Because
    // only the deviator is scaled, the removed strain is trace-free and the plastic flow preserves volume.
    let scale = y / dev_norm;
    let eps_new = [mean + dev[0] * scale, mean + dev[1] * scale, mean + dev[2] * scale];
    let s_new = Matrix3::from_diagonal(&nalgebra::Vector3::new(
        eps_new[0].exp(),
        eps_new[1].exp(),
        eps_new[2].exp(),
    ));
    let f_elastic = u * s_new * v_t;

    // F_p = F_e⁻¹ F, the part this step absorbed.
    let f_plastic = f_elastic.try_inverse()? * f_trial;

    Some(ReturnMap {
        f_elastic,
        f_plastic,
        yielded: true,
        plastic_strain_increment: dev_norm - y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn stretch(a: f64, b: f64, c: f64) -> Matrix3<f64> {
        Matrix3::from_diagonal(&Vector3::new(a, b, c))
    }

    #[test]
    fn below_yield_the_deformation_is_returned_bit_identically() {
        // An elastic simulation must not drift just because a plasticity check runs. Bit-identical, not close.
        let (tau_y, mu) = (1.0e6, 1.0e6); // y = 0.5, a generous elastic range
        for f in [
            stretch(1.0, 1.0, 1.0),
            stretch(1.05, 0.98, 1.0),
            stretch(1.2, 1.0 / 1.2, 1.0),
            Matrix3::new(1.02, 0.01, 0.0, -0.01, 0.99, 0.02, 0.0, 0.01, 1.01),
        ] {
            let r = radial_return(&f, tau_y, mu).expect("valid gradient");
            assert!(!r.yielded, "should be elastic: {f}");
            assert_eq!(r.f_elastic, f, "elastic branch must be bit-identical");
            assert_eq!(r.f_plastic, Matrix3::identity());
            assert_eq!(r.plastic_strain_increment, 0.0);
        }
    }

    /// **Plastic flow preserves volume exactly** — the defining property of `J2` flow, and a sharp check on the
    /// deviatoric split. A trace leaking into the plastic part shows up here immediately.
    #[test]
    fn plastic_flow_is_exactly_volume_preserving() {
        let (tau_y, mu) = (2.0e4, 1.0e6); // y = 0.01, easy to exceed
        for f in [
            stretch(1.5, 0.8, 0.9),
            stretch(2.0, 1.0, 0.4),
            stretch(0.5, 1.7, 1.1),
            Matrix3::new(1.4, 0.2, 0.1, -0.1, 0.8, 0.05, 0.02, -0.1, 1.1),
        ] {
            let r = radial_return(&f, tau_y, mu).expect("valid");
            assert!(r.yielded, "this fixture should yield");
            assert!(
                (r.f_plastic.determinant() - 1.0).abs() < 1e-12,
                "det(F_p) must be exactly 1, got {}",
                r.f_plastic.determinant()
            );
            // and the split reconstructs the trial gradient
            assert!(
                (r.f_elastic * r.f_plastic - f).norm() < 1e-12,
                "F_e · F_p must reproduce F"
            );
            // volumetric part is untouched, so det(F_e) == det(F)
            assert!(
                (r.f_elastic.determinant() - f.determinant()).abs() < 1e-12,
                "the volumetric response must be purely elastic"
            );
        }
    }

    /// **Hydrostatic loading never yields, at any magnitude** — von Mises is deviatoric. This also identifies
    /// *which* criterion is implemented: a pressure-dependent one (Drucker-Prager, for soil) would yield here.
    #[test]
    fn hydrostatic_compression_never_yields() {
        let (tau_y, mu) = (1.0, 1.0e6); // y = 5e-7, essentially zero elastic range
        for j in [0.1f64, 0.5, 0.9, 1.1, 2.0, 10.0] {
            let s = j.cbrt();
            let f = stretch(s, s, s);
            let r = radial_return(&f, tau_y, mu).expect("valid");
            assert!(
                !r.yielded,
                "uniform scaling by {j} has zero deviatoric strain and must not yield under von Mises"
            );
            assert_eq!(r.f_elastic, f);
        }
        // But adding any shear at the same volume does yield, so the test above is not vacuous.
        let sheared = stretch(1.3, 1.0 / 1.3, 1.0);
        assert!(radial_return(&sheared, tau_y, mu).unwrap().yielded, "deviatoric strain must yield");
    }

    /// The returned state sits **on** the yield surface, not merely inside it — the projection is exact.
    #[test]
    fn the_returned_state_lies_on_the_yield_surface() {
        let (tau_y, mu) = (3.0e4, 1.0e6);
        let y = yield_strain(tau_y, mu);
        for f in [stretch(1.6, 0.7, 0.95), stretch(2.5, 0.5, 0.9)] {
            let r = radial_return(&f, tau_y, mu).unwrap();
            let s = r.f_elastic.svd(false, false).singular_values;
            let eps = [s[0].ln(), s[1].ln(), s[2].ln()];
            let mean = (eps[0] + eps[1] + eps[2]) / 3.0;
            let dev_norm = ((eps[0] - mean).powi(2) + (eps[1] - mean).powi(2) + (eps[2] - mean).powi(2)).sqrt();
            assert!(
                (dev_norm - y).abs() < 1e-10,
                "returned deviatoric strain {dev_norm} should sit exactly on the surface {y}"
            );
        }
    }

    /// **Load past yield, unload, and the deformation is permanent** — the behaviour the whole module exists
    /// for, and the one a purely elastic model cannot produce.
    #[test]
    fn loading_past_yield_leaves_permanent_deformation() {
        let (tau_y, mu) = (2.0e4, 1.0e6);
        // Squeeze well past yield, accumulating the plastic part as a grasp would over several steps.
        let mut f_p_total = Matrix3::identity();
        let mut applied = Matrix3::identity();
        for _ in 0..5 {
            applied = stretch(1.08, 1.0 / 1.08, 1.0) * applied;
            let trial = applied * f_p_total.try_inverse().unwrap();
            let r = radial_return(&trial, tau_y, mu).unwrap();
            if r.yielded {
                f_p_total = r.f_plastic * f_p_total;
            }
        }
        // Something was retained, and it is volume-preserving.
        assert!((f_p_total - Matrix3::identity()).norm() > 1e-3, "plastic deformation should accumulate");
        assert!(
            (f_p_total.determinant() - 1.0).abs() < 1e-10,
            "accumulated plastic flow stays volume-preserving, det = {}",
            f_p_total.determinant()
        );
        // Releasing the load entirely leaves the plastic shape: the elastic part returns to identity but the
        // body does not return to its rest configuration.
        let released = radial_return(&Matrix3::identity(), tau_y, mu).unwrap();
        assert!(!released.yielded, "an unloaded element is elastic");
        assert!(
            (f_p_total - Matrix3::identity()).norm() > 1e-3,
            "the permanent set survives unloading — this is the point"
        );
    }

    /// An inverted or degenerate gradient has no real logarithmic strain, so it is refused — the same domain
    /// restriction `FemSim`'s `ln J` term carries.
    #[test]
    fn an_inverted_gradient_is_refused() {
        let (tau_y, mu) = (1.0e4, 1.0e6);
        assert!(radial_return(&stretch(1.0, 1.0, -0.5), tau_y, mu).is_none(), "det < 0");
        assert!(radial_return(&stretch(1.0, 1.0, 0.0), tau_y, mu).is_none(), "det = 0");
        assert!(radial_return(&Matrix3::zeros(), tau_y, mu).is_none(), "fully degenerate");
        // A large but valid stretch is fine.
        assert!(radial_return(&stretch(5.0, 0.3, 0.7), tau_y, mu).is_some());
    }

    /// A larger yield stress means a larger elastic range, and the increment scales as the excess — linear in
    /// how far past the surface the trial state sat.
    #[test]
    fn a_higher_yield_stress_widens_the_elastic_range() {
        let mu = 1.0e6;
        let f = stretch(1.4, 1.0 / 1.4, 1.0);
        let soft = radial_return(&f, 1.0e4, mu).unwrap(); // y = 0.005
        let stiff = radial_return(&f, 1.0e6, mu).unwrap(); // y = 0.5
        assert!(soft.yielded, "a low yield stress should yield");
        assert!(!stiff.yielded, "a high one should stay elastic");
        assert!(soft.plastic_strain_increment > 0.0);
        // The increment is the excess over the surface, so raising tau_y shrinks it monotonically.
        let mid = radial_return(&f, 5.0e4, mu).unwrap();
        assert!(
            mid.plastic_strain_increment < soft.plastic_strain_increment,
            "raising the yield stress must reduce the plastic increment"
        );
    }
}
