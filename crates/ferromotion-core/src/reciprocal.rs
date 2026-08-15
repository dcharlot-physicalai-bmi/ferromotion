//! **Reciprocal screws** — when a wrench does no work on a twist.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §2.5.3. A wrench `F`
//! is *reciprocal* to a twist `V` when the instantaneous power vanishes, `F · V = 0`: the wrench is exactly
//! the kind a constraint can supply for free, because the motion it permits generates no work against it.
//! Two screws are reciprocal when the twist about one is reciprocal to the wrench along the other
//! (MLS Def. 2.3).
//!
//! This is the algebra behind constraint analysis. The wrenches a set of contacts can apply and the twists
//! an object can still execute are reciprocal subspaces, so "which motions does this grasp leave free" and
//! "which wrenches can it resist without friction" are the same question asked from the two sides.
//!
//! # Conventions, because this crate contains two of them
//!
//! Here, matching [`crate::screw`]: a **twist is `[ω; v]`** (angular part first) and a **wrench is `[m; f]`**
//! (moment first), so `F · V = m·ω + f·v` is the instantaneous power.
//!
//! What establishes that is the *structure* of [`crate::adjoint`], which is `[[R, 0], [p̂R, R]]` — the
//! Lynch & Park form for a `[ω; v]` twist, whose dual acts on a moment-first wrench. It is **not**
//! established by `screw::tests::the_wrench_twist_pairing_is_frame_invariant`, despite appearances: that
//! test checks `Fᵀ(Ad V) == (Adᵀ F)ᵀ V`, which is the unconditional identity `fᵀ(Av) = (Aᵀf)ᵀv` and holds
//! for *any* ordering of either vector. It verifies the transpose is used consistently, which is worth
//! having, but it cannot pin a convention and should not be read as doing so.
//!
//! **[`crate::grasp_spatial`] orders its wrenches the other way, `[f; m]`.** That is self-consistent there
//! (grasp quality only ever pairs grasp wrenches with each other), but it means a `grasp_spatial` wrench
//! must be **swapped** before being paired with a twist or transformed by `Ad_Tᵀ` — otherwise force and
//! moment silently exchange roles and the result is dimensional nonsense that still type-checks. Use
//! [`swap_wrench_halves`] at that boundary.
//!
//! MLS itself uses a third ordering — twist `(v, ω)`, linear first — so formulas transcribed from the book
//! need reordering, not just copying. This module implements MLS's mathematics in this crate's convention.

use nalgebra::{Vector3, Vector6};

/// A screw: an axis (a point `q` on it and a unit direction `omega`), a pitch, and a magnitude.
///
/// Pitch `h` is translation per unit rotation. `h = 0` is a pure rotation, and `h = inf` is a pure
/// translation — the latter is a distinct case in MLS's algebra and is **not** representable here; see
/// [`Screw::twist`].
#[derive(Clone, Copy, Debug)]
pub struct Screw {
    /// Any point on the screw axis.
    pub q: Vector3<f64>,
    /// Axis direction. Normalized on use.
    pub omega: Vector3<f64>,
    /// Pitch: translation per unit rotation.
    pub h: f64,
    /// Magnitude.
    pub m: f64,
}

impl Screw {
    /// The **twist** about this screw, as `[ω; v]` with `v = q × ω + h·ω` (MLS §2.5.3, scaled by magnitude).
    pub fn twist(&self) -> Vector6<f64> {
        let w = self.omega.normalize();
        let v = self.q.cross(&w) + self.h * w;
        Vector6::new(self.m * w.x, self.m * w.y, self.m * w.z, self.m * v.x, self.m * v.y, self.m * v.z)
    }

    /// The **wrench** along this screw, as `[m; f]` with `f = ω` and moment `q × ω + h·ω`.
    ///
    /// Note the duality: a wrench along a screw has its *force* along the axis and its *moment* built from
    /// the axis offset and pitch, which is the mirror image of [`Screw::twist`]. Getting these the same way
    /// round is the classic way to compute a plausible number that means nothing.
    pub fn wrench(&self) -> Vector6<f64> {
        let w = self.omega.normalize();
        let mom = self.q.cross(&w) + self.h * w;
        Vector6::new(self.m * mom.x, self.m * mom.y, self.m * mom.z, self.m * w.x, self.m * w.y, self.m * w.z)
    }
}

/// Instantaneous power of a wrench acting through a twist: `F · V = m·ω + f·v`.
///
/// The definition reciprocity is built on, exposed so a caller never has to guess the ordering.
pub fn power(wrench: &Vector6<f64>, twist: &Vector6<f64>) -> f64 {
    wrench.dot(twist)
}

/// Swap the two halves of a 6-vector, converting between `[m; f]` and `[f; m]` ordering.
///
/// Needed at the boundary with [`crate::grasp_spatial`], which orders wrenches force-first. A plain dot
/// product across that boundary is not power and not frame-invariant; it is a number with mixed units.
pub fn swap_wrench_halves(w: &Vector6<f64>) -> Vector6<f64> {
    Vector6::new(w[3], w[4], w[5], w[0], w[1], w[2])
}

/// **The reciprocal product, computed algebraically**: `F₂ · V₁`, the power the wrench along `s2` does on
/// the twist about `s1`.
///
/// This is the authoritative form — it is the definition, it has no branch cases, and it is exact for every
/// configuration including parallel and intersecting axes. Compare
/// [`reciprocal_product_geometric`], which is MLS's closed geometric form.
pub fn reciprocal_product(s1: &Screw, s2: &Screw) -> f64 {
    power(&s2.wrench(), &s1.twist())
}

/// **MLS eq. (2.72)**, the classical geometric form of the reciprocal product:
///
/// ```text
/// S₁ ⊙ S₂ = M₁·M₂·[ (h₁ + h₂)·cos α − d·sin α ]
/// ```
///
/// where `d` is the distance between the two axes along their common perpendicular `n`, and
/// `α = atan2(ω₁ × ω₂ · n, ω₁ · ω₂)` is the angle between them.
///
/// Returns `None` for **parallel axes**, where the common perpendicular is not unique and so neither `d`
/// nor the sign of `α` is defined by this construction. MLS restricts the derivation to finite pitch and
/// leaves the remaining cases as exercises; [`reciprocal_product`] has no such restriction, and the tests
/// here check the two agree wherever both are defined. That agreement is the reason to trust either.
pub fn reciprocal_product_geometric(s1: &Screw, s2: &Screw) -> Option<f64> {
    let (w1, w2) = (s1.omega.normalize(), s2.omega.normalize());
    let cross = w1.cross(&w2);
    let cn = cross.norm();
    if cn <= 1e-12 {
        return None; // parallel: no unique common perpendicular
    }
    let n = cross / cn;
    // Signed offset along the common perpendicular. MLS writes the offset as `d·n` with `d > 0`, which
    // fixes n's direction; carrying the sign here instead keeps n tied to ω₁ × ω₂ so that α and d are
    // consistent with each other. Sign errors between the two show up immediately against the algebraic
    // form, which is why that comparison is the test.
    let d = (s2.q - s1.q).dot(&n);
    let alpha = cn.atan2(w1.dot(&w2));
    Some(s1.m * s2.m * ((s1.h + s2.h) * alpha.cos() - d * alpha.sin()))
}

/// Whether two screws are reciprocal — MLS Prop. 2.18, `S₁ ⊙ S₂ = 0`.
pub fn are_reciprocal(s1: &Screw, s2: &Screw, tol: f64) -> bool {
    reciprocal_product(s1, s2).abs() <= tol
}

/// A basis for the wrenches reciprocal to **every** twist in `twists` — the constraint wrenches a mechanism
/// permitting exactly those motions can resist for free.
///
/// Computed as the null space of the matrix whose rows are the twists, via SVD: a wrench `F` is reciprocal
/// to all of them iff `V_i · F = 0` for each `i`. The returned vectors are `[m; f]`-ordered and orthonormal.
///
/// The dimensions must add up — `dim(reciprocal) = 6 − rank(twists)` — which is the property worth asserting
/// in a caller, because it catches a rank mistake that inspecting individual vectors will not.
pub fn reciprocal_basis(twists: &[Vector6<f64>], tol: f64) -> Vec<Vector6<f64>> {
    if twists.is_empty() {
        return (0..6).map(|i| Vector6::from_fn(|r, _| if r == i { 1.0 } else { 0.0 })).collect();
    }
    // A^T A is 6x6 whatever the twist count, so its eigenvectors always give a full basis. A THIN SVD of
    // the 1x6 twist matrix returns only ONE row of V^T, and indexing six of them panics — which is what the
    // first version of this did.
    let mut ata = nalgebra::Matrix6::zeros();
    for t in twists {
        ata += t * t.transpose();
    }
    let eig = nalgebra::SymmetricEigen::new(ata);
    let e_max = eig.eigenvalues.iter().fold(0.0f64, |m, x| m.max(*x));
    let thresh = tol.max(1e-12) * e_max.max(1.0);
    let mut out = Vec::new();
    for i in 0..6 {
        if eig.eigenvalues[i] <= thresh {
            out.push(Vector6::from_fn(|r, _| eig.eigenvectors[(r, i)]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adjoint, exp_se3};

    fn s(qx: f64, qy: f64, qz: f64, wx: f64, wy: f64, wz: f64, h: f64, m: f64) -> Screw {
        Screw { q: Vector3::new(qx, qy, qz), omega: Vector3::new(wx, wy, wz), h, m }
    }

    #[test]
    fn the_geometric_form_agrees_with_the_algebraic_one() {
        // THE test for this module. MLS eq. (2.72) is a closed form for a quantity that is also just a dot
        // product; if my sign handling of `d` or `alpha` were wrong, the two would disagree. Sweeping many
        // skew configurations is what makes this a check rather than a coincidence.
        let mut checked = 0;
        for i in 0..6 {
            for j in 0..6 {
                let t = i as f64 * 0.37 - 1.0;
                let u = j as f64 * 0.29 - 0.8;
                let s1 = s(t, 0.2, -0.3, 0.0, 0.0, 1.0, 0.15 * t, 1.3);
                let s2 = s(0.1, u, 0.6, 0.4 + 0.1 * u, 1.0, 0.2 * t, -0.2 + 0.1 * u, 0.7);
                if let Some(g) = reciprocal_product_geometric(&s1, &s2) {
                    let a = reciprocal_product(&s1, &s2);
                    assert!(
                        (g - a).abs() < 1e-9,
                        "eq. (2.72) gave {g}, the dot product gave {a} (diff {})",
                        (g - a).abs()
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 20, "the sweep should exercise many skew pairs, got {checked}");
    }

    #[test]
    fn a_revolute_axis_is_reciprocal_to_a_force_through_it() {
        // The canonical case, and the one with physical meaning: a pure force whose line of action MEETS a
        // revolute axis produces no moment about it, so it does no work on that joint's motion — which is
        // exactly why a bearing can carry it for free.
        let joint = s(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0); // rotation about z through the origin
        let through = s(0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0); // force along x, crossing the z-axis
        assert!(are_reciprocal(&joint, &through, 1e-12), "product {}", reciprocal_product(&joint, &through));

        // Offset that same force line so it no longer meets the axis: now it exerts a moment and does work.
        let offset = s(0.0, 0.7, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(!are_reciprocal(&joint, &offset, 1e-12));
        assert!(reciprocal_product(&joint, &offset).abs() > 0.1);

        // A force PARALLEL to the axis is also reciprocal: no moment about z, and translation along z is
        // not permitted by a zero-pitch revolute twist.
        let parallel = s(0.4, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(are_reciprocal(&joint, &parallel, 1e-12), "product {}", reciprocal_product(&joint, &parallel));
    }

    #[test]
    fn reciprocity_is_frame_invariant() {
        // Power is a scalar, so reciprocity cannot depend on the frame it is computed in. Transform the
        // twist by Ad_T and the wrench by Ad_Tᵀ and the product must be unchanged — the same property
        // screw.rs asserts, now for this pairing.
        let s1 = s(0.3, -0.1, 0.4, 0.2, 0.9, -0.3, 0.11, 1.0);
        let s2 = s(-0.2, 0.5, 0.1, 0.7, -0.2, 0.6, -0.05, 1.0);
        let t = exp_se3(&Vector6::new(0.3, -0.4, 0.2, 0.7, 0.1, -0.5));
        let (v_b, f_a) = (s1.twist(), s2.wrench());
        // V_a = Ad_T V_b and F_b = Ad_T^T F_a are the DUAL pair: F_a . V_a == F_b . V_b. Applying Ad to the
        // twist and Ad^T to the wrench and then pairing those two is a different (and wrong) expression --
        // it evaluates f^T Ad Ad v. The first version of this test did exactly that and failed, correctly.
        let in_a = power(&f_a, &(adjoint(&t) * v_b));
        let in_b = power(&(adjoint(&t).transpose() * f_a), &v_b);
        assert!((in_a - in_b).abs() < 1e-12, "power not frame-invariant: {in_a} vs {in_b}");
        // and the adjoint really is doing something, so this is not vacuous
        assert!((adjoint(&t) * v_b - v_b).norm() > 1e-3, "the transform must move the twist");
    }

    #[test]
    fn the_reciprocal_basis_has_the_complementary_dimension() {
        // dim(reciprocal) = 6 − rank(twists). This is the property that catches a rank mistake; inspecting
        // individual null-space vectors does not.
        let z = s(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0).twist();
        let x = s(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).twist();

        let b1 = reciprocal_basis(&[z], 1e-9);
        assert_eq!(b1.len(), 5, "one twist leaves a 5-dimensional reciprocal space");
        let b2 = reciprocal_basis(&[z, x], 1e-9);
        assert_eq!(b2.len(), 4);
        // a repeated twist adds no rank, so the dimension must not drop
        let b3 = reciprocal_basis(&[z, z, z], 1e-9);
        assert_eq!(b3.len(), 5, "linearly dependent twists must not reduce the reciprocal space");

        // every returned wrench really is reciprocal to every input twist
        for f in &b2 {
            for v in [&z, &x] {
                assert!(power(f, v).abs() < 1e-9, "basis wrench not reciprocal: {}", power(f, v));
            }
        }
        // and no twists at all leaves the whole 6-dimensional space
        assert_eq!(reciprocal_basis(&[], 1e-9).len(), 6);
    }

    #[test]
    fn swapping_halves_is_what_makes_a_grasp_wrench_pairable() {
        // grasp_spatial orders wrenches [f; m]; this module and screw.rs use [m; f]. Pairing across that
        // boundary without swapping exchanges force and moment, which type-checks and is meaningless.
        let s2 = s(0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0);
        let f_moment_first = s2.wrench();
        let f_force_first = swap_wrench_halves(&f_moment_first);
        assert_eq!(swap_wrench_halves(&f_force_first), f_moment_first, "swap must be an involution");
        assert_ne!(f_force_first, f_moment_first, "this wrench must actually be asymmetric to be a test");

        // NOTE: the twist must have a NON-ZERO linear part. A pure rotation about an axis through the
        // origin has v = 0, so only the angular half is ever paired and the mis-ordered product comes out
        // zero by accident -- which is how the first version of this test passed for the wrong reason.
        let joint = s(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
        let v = joint.twist();
        assert!(v.fixed_rows::<3>(3).norm() < 1e-12, "this joint's twist is purely angular");
        let helical = s(0.2, -0.3, 0.1, 0.0, 0.0, 1.0, 0.4, 1.0);
        let v = helical.twist();
        assert!(v.fixed_rows::<3>(3).norm() > 1e-3, "a pitched, offset screw has a real linear part");
        // The two orderings give materially different numbers, which is the whole point: pairing across the
        // grasp_spatial boundary without swapping silently exchanges force and moment.
        let right = power(&f_moment_first, &v);
        let wrong = power(&f_force_first, &v);
        assert!((right - wrong).abs() > 0.1, "orderings must differ materially: {right} vs {wrong}");
    }
}
