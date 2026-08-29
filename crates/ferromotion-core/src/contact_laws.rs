//! **Does a contact solve actually obey the physics?** — the three laws, checked on the answer.
//!
//! Every frictional contact solver in this crate returns impulses. Whether those impulses satisfy the physical
//! laws of contact is a *separate question* from whether the solver converged, and the two come apart in a way
//! that matters: a solver can converge perfectly to the solution of the **wrong problem**. `PgsResult::residual`
//! is the largest impulse change in the final sweep, and it goes to zero for a pyramidal-cone solve whose
//! friction forces sit outside the true Coulomb cone by a fixed margin. Convergence is not correctness.
//!
//! The laws, following Le Lidec et al., *Contact Models in Robotics: a Comparative Analysis* (arXiv:2304.06372),
//! which reduces contact correctness to three conditions and gives per-contact residuals for each:
//!
//! * **Signorini** — a contact may only push, and a loaded contact may not still be closing:
//!   `0 ≤ λ_n ⟂ c_n ≥ 0`. Violated by the CCP relaxation (MuJoCo, Drake) for sliding contacts, by an amount
//!   the paper characterises as `Δt·μ·‖c_T‖` — proportional to both timestep and sliding speed.
//! * **Coulomb** — the impulse lies in the friction cone `‖λ_T‖ ≤ μ λ_n`. Violated by every solver that
//!   linearises the cone into a pyramid (ODE, Bullet, DART, and this crate's own `contact_ipm`,
//!   `contact_pgs` and `whole_body_contact`), which biases friction toward the pyramid's corners.
//! * **Maximum dissipation** — among all admissible friction impulses, the one chosen must maximise
//!   dissipation, which for a sliding contact means `λ_T = −μ λ_n c_T/‖c_T‖`: exactly anti-parallel to the
//!   slide, at the cone boundary. Violated by RaiSim, whose contacts slide faster than they should.
//!
//! # Why this is a free function and not a solver method
//!
//! Because five of this crate's seven contact paths reported nothing at all. `contact_pgs` and `contact_ipm`
//! carried a violation number; `contact`, `affine_contact`, `planar_contact`, `floating_contact` and `ipc` did
//! not. A checker that takes impulses and contact velocities works for all of them, including solvers not
//! written yet, and cannot drift away from any one of them the way a copied method would.
//!
//! # What it cannot tell you
//!
//! These residuals evaluate the returned point against the laws. They say nothing about whether the *contact
//! set* was right — a missed collision produces a perfectly law-abiding solve of a problem that is not the one
//! in front of you. And a solver deliberately using a relaxation will report a violation that is a design
//! choice rather than a defect; the number is the size of that choice, which is the point.

use nalgebra::Vector3;

/// Per-contact violation of each physical law, in impulse units.
///
/// All three are non-negative, and zero means the law holds exactly at the returned point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactLawResidual {
    pub contact: usize,
    /// **Signorini.** The larger of two failures: a pulling normal impulse (`max(0, −λ_n)`), and a loaded
    /// contact that is still closing (`λ_n · max(0, −c_n)`, the complementarity product).
    pub signorini: f64,
    /// **Coulomb.** Distance of the impulse outside the *true* second-order cone, `max(0, ‖λ_T‖ − μ λ_n)`.
    /// This is the quantity a pyramidal solver pays: its impulse can sit up to a factor of `√2` (in 3D, at a
    /// pyramid corner) outside the cone it claims to enforce.
    pub coulomb: f64,
    /// **Maximum dissipation.** For a sliding contact, how far the friction impulse is from the unique
    /// maximally-dissipative choice `−μ λ_n c_T/‖c_T‖`. Zero for a sticking contact, where any impulse inside
    /// the cone is admissible and the principle imposes nothing.
    pub max_dissipation: f64,
    /// Whether this contact was sliding, which is the only regime in which `max_dissipation` binds.
    pub sliding: bool,
}

impl ContactLawResidual {
    /// The worst of the three, for a single go/no-go number.
    pub fn worst(&self) -> f64 {
        self.signorini.max(self.coulomb).max(self.max_dissipation)
    }
}

/// **Check a contact solve against Signorini, Coulomb and maximum dissipation.**
///
/// `lambda[i]` is the contact impulse as `[t_x, t_y, n]` and `c[i]` the resulting contact-frame velocity in the
/// same layout — the convention [`crate::solve_contacts_pgs`] already uses. `mu[i]` is the friction coefficient.
///
/// Residuals are scaled by the largest impulse present, so the numbers mean the same thing on a 10 g finger and
/// a 1 t press. `slide_tol` is the tangential speed above which a contact counts as sliding; below it the
/// maximum-dissipation principle imposes no constraint and reporting one would be an artifact of noise.
///
/// Returns one entry per contact, so a caller can see *which* contact and *which* law, rather than a single
/// number that says only that something is wrong.
pub fn contact_law_residuals(
    lambda: &[Vector3<f64>],
    c: &[Vector3<f64>],
    mu: &[f64],
    slide_tol: f64,
) -> Vec<ContactLawResidual> {
    let n = lambda.len().min(c.len()).min(mu.len());
    // Scale by the largest impulse so the residual is dimensionless in the same way across problems. A floor
    // keeps an all-zero solve (no contact active) from dividing by zero and reporting NaN as a violation.
    let scale = lambda.iter().take(n).fold(1e-12f64, |m, l| m.max(l.norm()));

    (0..n)
        .map(|i| {
            let (l, v, m) = (lambda[i], c[i], mu[i]);
            let (lt, ln) = (Vector3::new(l.x, l.y, 0.0), l.z);
            let ct = Vector3::new(v.x, v.y, 0.0);
            let (lt_norm, ct_norm) = (lt.norm(), ct.norm());

            // Signorini: no pulling, and no loaded contact still closing.
            let pulling = (-ln).max(0.0);
            let closing = if ln > 1e-9 * scale { ln * (-v.z).max(0.0) } else { 0.0 };
            let signorini = (pulling / scale).max(closing / scale);

            // Coulomb against the TRUE cone. A pyramidal solver is inside its own pyramid and can be outside
            // this, which is exactly the error the linearisation buys.
            let coulomb = (lt_norm - m * ln.max(0.0)).max(0.0) / scale;

            // Maximum dissipation binds only while sliding: the friction impulse must be anti-parallel to the
            // slide and saturated at the cone boundary. Sticking contacts admit any impulse in the cone, so
            // reporting a residual there would penalise a correct answer.
            let sliding = ct_norm > slide_tol;
            let max_dissipation = if sliding && ln > 1e-9 * scale {
                let ideal = ct * (-m * ln / ct_norm);
                (lt - ideal).norm() / scale
            } else {
                0.0
            };

            ContactLawResidual { contact: i, signorini, coulomb, max_dissipation, sliding }
        })
        .collect()
}

/// The worst residual over all contacts and all three laws — one number for a gate.
///
/// Zero for an empty contact set, which is correct: no contacts cannot violate a contact law.
pub fn worst_contact_law_residual(
    lambda: &[Vector3<f64>],
    c: &[Vector3<f64>],
    mu: &[f64],
    slide_tol: f64,
) -> f64 {
    contact_law_residuals(lambda, c, mu, slide_tol).iter().map(|r| r.worst()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A solve that obeys all three laws must report zero on all three. Without this the other tests could
    /// pass on a checker that always fires.
    #[test]
    fn a_lawful_solve_reports_nothing() {
        // Sticking contact: pressing, not closing, friction well inside the cone.
        let lambda = [Vector3::new(0.1, 0.0, 1.0)];
        let c = [Vector3::new(0.0, 0.0, 0.0)];
        let r = &contact_law_residuals(&lambda, &c, &[0.5], 1e-6)[0];
        assert_eq!(r.signorini, 0.0, "{r:?}");
        assert_eq!(r.coulomb, 0.0, "{r:?}");
        assert_eq!(r.max_dissipation, 0.0, "a sticking contact imposes no dissipation constraint: {r:?}");
        assert!(!r.sliding);
    }

    /// Each law must be detectable on its own, or a composite number could hide which one broke.
    #[test]
    fn each_law_is_detected_separately() {
        // Signorini, form 1: a contact that PULLS.
        let r = &contact_law_residuals(&[Vector3::new(0.0, 0.0, -1.0)], &[Vector3::zeros()], &[0.5], 1e-6)[0];
        assert!(r.signorini > 0.9, "a pulling contact must violate Signorini: {r:?}");
        assert_eq!(r.coulomb, 0.0, "and must not be blamed on Coulomb: {r:?}");

        // Signorini, form 2: a LOADED contact still closing (negative normal velocity).
        let r = &contact_law_residuals(&[Vector3::new(0.0, 0.0, 1.0)], &[Vector3::new(0.0, 0.0, -0.3)], &[0.5], 1e-6)[0];
        assert!(r.signorini > 0.29, "a loaded, still-closing contact violates complementarity: {r:?}");

        // Coulomb: friction outside the cone, normal fine, not sliding. The residual is SCALED by the impulse
        // norm — 0.4 outside on an impulse of ‖(0.9,0,1)‖ = 1.345 reports 0.297. The first version of this
        // assertion expected the unscaled 0.4, which was the test forgetting the normalisation the function
        // documents and this same test file relies on two cases below.
        let lam = Vector3::new(0.9, 0.0, 1.0);
        let r = &contact_law_residuals(&[lam], &[Vector3::zeros()], &[0.5], 1e-6)[0];
        assert!(
            (r.coulomb - 0.4 / lam.norm()).abs() < 1e-9,
            "0.9 tangential against a 0.5 cone is 0.4 outside, scaled by {:.3}: {r:?}",
            lam.norm()
        );
        assert_eq!(r.signorini, 0.0, "and Signorini is satisfied: {r:?}");

        // Maximum dissipation: sliding in +x, friction pointing the WRONG way (also +x).
        let wrong = Vector3::new(0.5, 0.0, 1.0);
        let r = &contact_law_residuals(&[wrong], &[Vector3::new(1.0, 0.0, 0.0)], &[0.5], 1e-6)[0];
        assert!(r.sliding);
        // Friction pointing WITH the slide instead of against it is 2·μλ_n away from the lawful impulse.
        assert!(
            (r.max_dissipation - 1.0 / wrong.norm()).abs() < 1e-9,
            "friction along the slide is 2*mu*ln = 1.0 from lawful, scaled: {r:?}"
        );
        // The correct impulse for that slide reports zero, which is what makes the check meaningful.
        let ok = &contact_law_residuals(&[Vector3::new(-0.5, 0.0, 1.0)], &[Vector3::new(1.0, 0.0, 0.0)], &[0.5], 1e-6)[0];
        assert!(ok.max_dissipation < 1e-12, "anti-parallel and saturated is the lawful answer: {ok:?}");
    }

    /// **What the pyramidal cone actually costs, as a number.**
    ///
    /// A solver that linearises the friction cone into an axis-aligned pyramid lets the tangential impulse
    /// reach `μλ_n` along *each* axis independently, so at a corner its magnitude is `√2·μλ_n` in 3D — outside
    /// the true cone it claims to enforce, by 41%. That impulse is perfectly converged and perfectly wrong,
    /// and it is why `PgsResult::residual` going to zero is not evidence of a correct answer.
    ///
    /// This crate's `contact_ipm`, `contact_pgs` and `whole_body_contact` all use the pyramidal form.
    #[test]
    fn the_pyramidal_cone_sits_outside_the_true_one_at_its_corners() {
        let (mu, ln) = (0.5, 1.0);
        // At a pyramid corner: μλ_n on each tangential axis. A pyramidal solver considers this feasible.
        let corner = Vector3::new(mu * ln, mu * ln, ln);
        let r = &contact_law_residuals(&[corner], &[Vector3::zeros()], &[mu], 1e-6)[0];
        let expected = (2.0f64.sqrt() - 1.0) * mu * ln / corner.norm();
        assert!(
            (r.coulomb - expected).abs() < 1e-9,
            "corner impulse should sit {expected:.4} outside the true cone, got {}",
            r.coulomb
        );
        assert!(r.coulomb > 0.0, "the linearisation is not free");

        // The same magnitude aligned to ONE axis is exactly on the true cone, so the residual is a property of
        // direction, not of size — which is the anisotropy the survey describes.
        let edge = Vector3::new(mu * ln, 0.0, ln);
        let re = &contact_law_residuals(&[edge], &[Vector3::zeros()], &[mu], 1e-6)[0];
        assert!(re.coulomb < 1e-12, "on-axis is exactly feasible: {re:?}");
    }

    /// An empty contact set cannot violate a contact law, and must not report NaN from an empty scale.
    #[test]
    fn no_contacts_is_not_a_violation() {
        assert_eq!(worst_contact_law_residual(&[], &[], &[], 1e-6), 0.0);
        // An all-zero solve (contacts present, none active) likewise.
        let z = [Vector3::zeros(); 2];
        let w = worst_contact_law_residual(&z, &z, &[0.5, 0.5], 1e-6);
        assert!(w.is_finite() && w == 0.0, "inactive contacts must report 0, got {w}");
    }

    /// **Where this crate's own PGS solver stops obeying the laws, measured.**
    ///
    /// Le Lidec et al. Fig. 13 puts a light cube under a heavy one: the mass ratio makes the contact problem
    /// ill-conditioned, and the paper reports per-contact (PGS-family) solvers hitting their iteration cap
    /// before converging and producing unrealistic trajectories. [`crate::solve_contacts_pgs`] is a per-contact
    /// solver, so it should show the same behaviour, and it does.
    ///
    /// Measured here, 1000-iteration cap, µ = 0.4:
    ///
    /// | mass ratio | iterations | solver residual | law residual | light-cube `v_z` |
    /// |---|---|---|---|---|
    /// | 1:1 | 25, converged | 5.9e-13 | 2.9e-10 | — |
    /// | 10² | hit the cap | 4.7e-10 | 4.7e-7 | −4.7e-7 |
    /// | 10⁶ | hit the cap | 9.8e-6 | **9.8e-3** | **−9.8e-3** |
    ///
    /// At 10⁶ the light cube is still closing at ~1 cm/s when the solver returns: it is being pushed through
    /// the ground. **The solver residual is three orders smaller than the law residual** — 9.8e-6 against
    /// 9.8e-3 — so a caller gating on convergence sees a reassuring number while Signorini is violated a
    /// thousand times worse. That gap is the entire reason this module exists.
    ///
    /// This is a property of per-contact splitting, not a defect unique to this implementation. The fix when
    /// it bites is a different solver (`contact_ipm` solves the whole system at once), not more iterations —
    /// the 10⁶ row does not improve between a 50-iteration budget and a 1000-iteration one.
    #[test]
    fn ill_conditioning_breaks_per_contact_pgs_and_the_laws_say_so() {
        use crate::{solve_contacts_pgs, PgsContact};
        use nalgebra::{DMatrix, DVector};

        let solve = |ratio: f64, iters: usize| -> (usize, f64, f64) {
            let (m_light, m_heavy) = (1e-3, 1e-3 * ratio);
            let mut m = DMatrix::zeros(6, 6);
            for k in 0..3 {
                m[(k, k)] = m_light;
                m[(k + 3, k + 3)] = m_heavy;
            }
            let dt = 1e-3;
            let mut vf = DVector::zeros(6);
            vf[2] = -9.81 * dt;
            vf[5] = -9.81 * dt;
            let mut ja = DMatrix::zeros(3, 6);
            let mut jb = DMatrix::zeros(3, 6);
            for k in 0..3 {
                ja[(k, k)] = 1.0;
                jb[(k, k)] = -1.0;
                jb[(k, k + 3)] = 1.0;
            }
            let cs = vec![PgsContact { j: ja, phi: 0.0, mu: 0.4 }, PgsContact { j: jb, phi: 0.0, mu: 0.4 }];
            let r = solve_contacts_pgs(&m, &vf, &cs, dt, iters, None);
            let cv: Vec<Vector3<f64>> = cs
                .iter()
                .map(|c| {
                    let u = &c.j * &r.v_next;
                    Vector3::new(u[0], u[1], u[2])
                })
                .collect();
            let worst = contact_law_residuals(&r.lambda, &cv, &[0.4, 0.4], 1e-6)
                .iter()
                .map(|l| l.worst())
                .fold(0.0f64, f64::max);
            (r.iters, r.residual, worst)
        };

        // Well-conditioned: converges early and the laws hold to near machine precision.
        let (iters, _, law) = solve(1.0, 1000);
        assert!(iters < 1000, "equal masses should converge inside the cap, used {iters}");
        assert!(law < 1e-8, "and obey the laws: {law:e}");

        // Ill-conditioned: caps out, and the laws are violated by something a user would feel.
        let (iters, solver, law) = solve(1e6, 1000);
        assert_eq!(iters, 1000, "a 10^6 mass ratio should exhaust the cap");
        assert!(law > 1e-3, "and leave a violation worth reporting, got {law:e}");

        // THE POINT: the solver's own residual is orders smaller than the physical violation, so convergence
        // is not evidence of correctness. If this ever stops holding, the certificate has lost its reason to
        // exist and this test should be re-read rather than deleted.
        assert!(
            law > 100.0 * solver,
            "law residual {law:e} should dwarf the solver residual {solver:e}; that gap is why this module exists"
        );

        // More iterations do not fix ill-conditioning — the failure is structural, not budgetary.
        let (_, _, law_50) = solve(1e6, 50);
        let (_, _, law_1000) = solve(1e6, 1000);
        assert!(
            (law_50 - law_1000).abs() / law_50 < 0.05,
            "20x the iterations should barely move it: {law_50:e} vs {law_1000:e}"
        );
    }
}
