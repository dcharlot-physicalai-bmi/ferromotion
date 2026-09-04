//! **How many fingers does a hand need?** — Carathéodory, Steinitz, and exceptional surfaces.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §5.4.1. Before choosing
//! a hand's kinematics you have to choose the number and type of its fingers, and that is answerable from
//! convex analysis rather than experiment: force closure means the origin lies in the interior of the convex
//! hull of the achievable wrenches, and two classical theorems bound how many generators that takes.
//!
//! * **Carathéodory** (MLS Thm 5.4): if a set positively spans `Rᵖ`, it has at least `p + 1` elements. So a
//!   force-closure grasp needs **at least `p + 1`** contacts. The planar intuition: given any two vectors
//!   `vᵢ, vⱼ` in `R²`, `−(vᵢ + vⱼ)` never lies in their positive span.
//! * **Steinitz** (MLS Thm 5.5): if `q ∈ int(co S)` then some subset of at most `2p` elements of `S` already
//!   has `q` in the interior of its hull. So **at most `2p`** contacts are ever needed.
//!
//! Together, for frictionless point contacts: `p = 3` planar gives `4 ≤ k ≤ 6`, and `p = 6` spatial gives
//! `7 ≤ k ≤ 12`. Friction lowers the floor sharply — each contact then contributes a *cone* of generators
//! rather than a single wrench — which is the quantitative reason a three-fingered frictional hand competes
//! with a many-fingered frictionless one.
//!
//! # Exceptional surfaces, and why a sphere is the hard case
//!
//! MLS defines `Λ(Σ)` as the set of wrenches available from frictionless point contacts anywhere on a
//! surface `Σ`, and calls `Σ` **exceptional** when the convex hull of `Λ(Σ)` contains no neighbourhood of the
//! origin. Such an object **can never be grasped with frictionless point contacts at all**, at any number.
//! MLS's canonical examples are the **sphere in `R³` and the circle in `R²`**.
//!
//! The reason is worth stating because it is the same fact the grasp-quality code runs into from the other
//! side: on a circle every inward normal points at the centre, so every line of action passes through the
//! reference point and every torque is *identically zero*. `Λ(circle)` therefore lies in the plane `τ = 0`, a
//! proper subspace, and a set confined to a subspace cannot contain a neighbourhood of the origin. That is
//! precisely the configuration [`crate::wrench_rank`] reports as rank-deficient and
//! [`crate::force_closure_q1`] gates to exactly zero — so the rank gate is not merely a numerical safeguard,
//! it is this theorem showing up in arithmetic.

/// The contact model, which sets how many generators one finger contributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactModel {
    /// Frictionless point contact: one wrench per contact, along the surface normal.
    FrictionlessPoint,
    /// Point contact with friction: a friction cone, so several generators per contact.
    PointWithFriction,
    /// Soft finger: friction plus a torsional moment about the contact normal.
    SoftFinger,
}

/// **Carathéodory's lower bound**: a set positively spanning `Rᵖ` has at least `p + 1` elements (MLS Thm 5.4).
pub fn caratheodory_lower_bound(p: usize) -> usize {
    p + 1
}

/// **Steinitz's upper bound**: at most `2p` generators are ever needed to put the origin in the interior of a
/// hull that contains it at all (MLS Thm 5.5).
pub fn steinitz_upper_bound(p: usize) -> usize {
    2 * p
}

/// Lower bound on the number of **contacts** for a force-closure grasp, by wrench-space dimension and contact
/// model — MLS Table 5.3.
///
/// `p = 3` is planar, `p = 6` spatial. Returns `None` for a dimension the table does not cover, rather than
/// extrapolating: the frictional entries are not a formula in `p`, they are results for the two cases the book
/// tabulates, and inventing a third would be inventing a theorem.
///
/// | | planar `p = 3` | spatial `p = 6` |
/// |---|---|---|
/// | frictionless point | 4 | 12 (7 if polyhedral) |
/// | point with friction | 3 | 4 |
/// | soft finger | 3 | 4 |
///
/// Note the frictionless spatial entry: **12** for a general non-exceptional surface but **7** for a
/// polyhedral one, because a polyhedron's finitely many face normals are a far smaller generator set to choose
/// from. Pass `polyhedral` to get that case.
pub fn min_contacts(p: usize, model: ContactModel, polyhedral: bool) -> Option<usize> {
    match (p, model) {
        (3, ContactModel::FrictionlessPoint) => Some(4),
        (3, ContactModel::PointWithFriction | ContactModel::SoftFinger) => Some(3),
        (6, ContactModel::FrictionlessPoint) => Some(if polyhedral { 7 } else { 12 }),
        (6, ContactModel::PointWithFriction | ContactModel::SoftFinger) => Some(4),
        _ => None,
    }
}

/// The Carathéodory–Steinitz bracket on contacts for **frictionless point contacts** in `Rᵖ`: `(p+1, 2p)`.
///
/// `p = 3` gives `(4, 6)` and `p = 6` gives `(7, 12)`, the two brackets MLS states explicitly.
pub fn frictionless_contact_bracket(p: usize) -> (usize, usize) {
    (caratheodory_lower_bound(p), steinitz_upper_bound(p))
}

/// Whether a generator set is confined to a proper subspace of `Rᵖ` and so **cannot** contain a neighbourhood
/// of the origin — the algebraic half of MLS's *exceptional surface* condition.
///
/// A surface is exceptional when the hull of its available wrenches has no neighbourhood of the origin. Rank
/// deficiency is a *sufficient* reason for that and the one that catches the canonical cases: on a circle or a
/// sphere every frictionless normal passes through the centre, so every torque vanishes identically and the
/// generators lie in a proper subspace.
///
/// It is **not** the whole condition. A full-rank generator set can still fail to contain the origin in its
/// hull — every generator on one side of some hyperplane — which is the case
/// [`crate::is_force_closure`] decides exactly. Use this to recognise the structural impossibility and that
/// for the configuration-specific question.
pub fn is_rank_deficient(generators: &[Vec<f64>], p: usize, tol: f64) -> bool {
    if generators.is_empty() {
        return true;
    }
    let m = generators.len();
    let a = nalgebra::DMatrix::from_fn(p, m, |r, c| generators[c].get(r).copied().unwrap_or(0.0));
    let Some(sv) = crate::finite_singular_values(&a) else { return true };
    let s_max = sv.iter().fold(0.0f64, |acc, x| acc.max(*x));
    if s_max <= tol {
        return true;
    }
    sv.iter().filter(|s| **s > tol * s_max).count() < p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_classical_bounds_reproduce_the_brackets_mls_states() {
        // MLS §5.4.1: p = 3 (planar) => 4 <= k <= 6; p = 6 (spatial) => 7 <= k <= 12.
        assert_eq!(frictionless_contact_bracket(3), (4, 6));
        assert_eq!(frictionless_contact_bracket(6), (7, 12));
        // and the bounds are consistent — a lower bound above an upper one would be a transcription error
        for p in 1..8 {
            let (lo, hi) = frictionless_contact_bracket(p);
            assert!(lo <= hi, "p={p}: lower {lo} exceeds upper {hi}");
        }
        // Carathéodory in the plane: no two vectors positively span R², so 3 is the floor there.
        assert_eq!(caratheodory_lower_bound(2), 3);
    }

    #[test]
    fn friction_lowers_the_floor_and_the_table_is_not_extrapolated() {
        // Table 5.3. Friction lowers the planar floor 4 -> 3 and the spatial floor 12 -> 4, which is the
        // quantitative reason a three-fingered frictional hand competes with a many-fingered frictionless one.
        assert_eq!(min_contacts(3, ContactModel::FrictionlessPoint, false), Some(4));
        assert_eq!(min_contacts(3, ContactModel::PointWithFriction, false), Some(3));
        assert_eq!(min_contacts(3, ContactModel::SoftFinger, false), Some(3));
        assert_eq!(min_contacts(6, ContactModel::FrictionlessPoint, false), Some(12));
        assert_eq!(min_contacts(6, ContactModel::PointWithFriction, false), Some(4));
        assert_eq!(min_contacts(6, ContactModel::SoftFinger, false), Some(4));
        // A polyhedron's finitely many face normals are a smaller set to choose from: 12 -> 7.
        assert_eq!(min_contacts(6, ContactModel::FrictionlessPoint, true), Some(7));
        // `polyhedral` must not change the frictional entries, which the table gives without that split.
        assert_eq!(min_contacts(6, ContactModel::PointWithFriction, true), Some(4));
        // Dimensions the book does not tabulate return None rather than an invented number.
        assert_eq!(min_contacts(4, ContactModel::FrictionlessPoint, false), None);
        assert_eq!(min_contacts(2, ContactModel::SoftFinger, false), None);
    }

    #[test]
    fn the_circle_is_an_exceptional_surface_which_is_the_rank_gate_restated() {
        // MLS names the circle in R² and the sphere in R³ as exceptional: never graspable with frictionless
        // point contacts, at ANY number. The reason is that every inward normal passes through the centre, so
        // every torque is identically zero and the generators lie in a proper subspace.
        //
        // Planar wrench [fx, fy, tau] from a frictionless contact at angle theta on the unit circle.
        let on_circle = |deg: f64| {
            let a: f64 = deg.to_radians();
            let (px, py) = (a.cos(), a.sin());
            let (nx, ny) = (-px, -py); // inward normal
            vec![nx, ny, px * ny - py * nx] // tau = p x n, identically 0 here
        };
        // However many contacts, and however spread out, the set stays rank-deficient.
        for count in [3usize, 6, 12, 32] {
            let gens: Vec<Vec<f64>> =
                (0..count).map(|i| on_circle(360.0 * i as f64 / count as f64)).collect();
            for g in &gens {
                assert!(g[2].abs() < 1e-12, "every torque on a circle must vanish, got {}", g[2]);
            }
            assert!(
                is_rank_deficient(&gens, 3, 1e-9),
                "the circle is exceptional: {count} frictionless contacts still cannot span R³"
            );
            // Steinitz's bound is no help here, and the sweep is what shows it: 12 and 32 contacts both
            // EXCEED 2p = 6 and are still rank-deficient. The bound says at most 2p are *needed* when the
            // origin is already interior to the hull; on an exceptional surface it never is, so no count
            // suffices. (A `count <= 2p || count > 2p` assertion would be a tautology, not a check.)
        }

        // A polygon is NOT exceptional: its normals do not all pass through one point, so torques are real.
        let on_square = |i: usize| -> Vec<f64> {
            let (p, n) = match i {
                0 => ([1.0, 0.3], [-1.0, 0.0]),
                1 => ([-1.0, -0.3], [1.0, 0.0]),
                2 => ([0.3, 1.0], [0.0, -1.0]),
                _ => ([-0.3, -1.0], [0.0, 1.0]),
            };
            vec![n[0], n[1], p[0] * n[1] - p[1] * n[0]]
        };
        let square: Vec<Vec<f64>> = (0..4).map(on_square).collect();
        assert!(!is_rank_deficient(&square, 3, 1e-9), "a square is not an exceptional surface");
        // and it meets Carathéodory's floor for the frictionless planar case
        assert_eq!(square.len(), min_contacts(3, ContactModel::FrictionlessPoint, false).unwrap());

        // Degenerate input is deficient rather than panicking.
        assert!(is_rank_deficient(&[], 3, 1e-9));
        assert!(is_rank_deficient(&[vec![0.0, 0.0, 0.0]], 3, 1e-9));
    }
}
