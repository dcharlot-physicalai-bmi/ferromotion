//! **Paden-Kahan subproblems** — closed-form, geometrically meaningful inverse kinematics.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §3.3.2, building on
//! Paden's thesis and Kahan's unpublished work. Where a numerical IK solver ([`crate::diffik`], or the
//! Levenberg-Marquardt solver in [`crate::kinematic_tree`]) iterates toward *one* solution from a seed,
//! these solve small geometric problems **exactly** — no iteration, no seed, no convergence question — and
//! return **every** solution, because the underlying geometry is a circle meeting a plane, a circle, or a
//! sphere, and those intersect in zero, one, or two points.
//!
//! The method is to reduce a manipulator's IK to a sequence of these, using the product-of-exponentials
//! form ([`crate::poe_fk`]). MLS is explicit that the set is *not* exhaustive — some manipulators cannot be
//! reduced to them — but between them they cover the common designs, and they are numerically stable in a
//! way a general nonlinear solve is not.
//!
//! All three take a **zero-pitch, unit-magnitude twist**, i.e. a pure rotation about an axis. A twist here
//! is given as an axis direction `omega` (normalized internally) and any point `r` on the axis, which is the
//! `(omega, q)` parameterisation of a revolute joint rather than the 6-vector — it is what the geometry
//! actually needs.
//!
//! **Every solution is verified by substitution, not asserted.** Each subproblem's tests re-apply
//! [`exp_so3`] to the returned angle and check the defining equation holds, which is the only claim that
//! matters and the one a sign error breaks.

use crate::exp_so3;
use nalgebra::Vector3;

/// Project `v` onto the plane perpendicular to the unit vector `omega`.
fn perp(v: &Vector3<f64>, omega: &Vector3<f64>) -> Vector3<f64> {
    v - omega * omega.dot(v)
}

/// **Subproblem 1: rotation about a single axis.** Rotate `p` about the axis `(omega, r)` until it
/// coincides with `q`; return `theta` with `exp(omega_hat * theta) * (p - r) + r == q`.
///
/// MLS eq. (3.17)-(3.19). Returns `None` when the problem is inconsistent — the two necessary conditions
/// are that `p` and `q` have equal components *along* the axis and equal distances *from* it:
///
/// ```text
/// omega . u == omega . v      and      |u'| == |v'|
/// ```
///
/// where `u = p - r`, `v = q - r` and `'` is the perpendicular projection. A rotation is an isometry that
/// fixes the axis, so a `q` that fails either condition is simply not on `p`'s orbit and no angle exists.
/// Returning `None` rather than a least-squares angle is deliberate: this is an exact method, and a caller
/// that wants a nearest-point answer wants a different function.
///
/// When `u' == 0` both points lie *on* the axis, every angle is a solution, and this returns `Some(0.0)` as
/// the representative — MLS notes the infinity of solutions in this degenerate case.
pub fn subproblem1(
    omega: &Vector3<f64>,
    r: &Vector3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
    tol: f64,
) -> Option<f64> {
    let w = omega.normalize();
    let (u, v) = (p - r, q - r);
    if (w.dot(&u) - w.dot(&v)).abs() > tol {
        return None; // different heights along the axis
    }
    let (up, vp) = (perp(&u, &w), perp(&v, &w));
    if (up.norm() - vp.norm()).abs() > tol {
        return None; // different radii from the axis
    }
    if up.norm() <= tol {
        return Some(0.0); // both on the axis: any angle works
    }
    Some(w.dot(&up.cross(&vp)).atan2(up.dot(&vp)))
}

/// **Subproblem 3: rotation to a given distance.** Rotate `p` about the axis `(omega, r)` until it is a
/// distance `delta` from `q`. Returns the (up to two) solutions.
///
/// MLS eq. (3.26)-(3.29). Geometrically a circle meeting a sphere, so there are two solutions in general,
/// one when they are tangent, and none when they miss — the returned `Vec` has that length, which is the
/// honest signature for this problem.
///
/// The construction projects onto the plane perpendicular to the axis and shrinks the target distance
/// accordingly, `delta'^2 = delta^2 - (omega . (p - q))^2`; a negative `delta'^2` means the sphere does not
/// reach the circle's plane at all and there is no solution. Then the law of cosines on the triangle formed
/// by the axis, the rotated `u'` and `v'` gives `theta = theta0 +/- acos(...)`, where the two signs are the
/// solution and its "flip" about the `u'`-`v'` bisector.
pub fn subproblem3(
    omega: &Vector3<f64>,
    r: &Vector3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
    delta: f64,
    tol: f64,
) -> Vec<f64> {
    let w = omega.normalize();
    let (u, v) = (p - r, q - r);
    let (up, vp) = (perp(&u, &w), perp(&v, &w));
    let (nu, nv) = (up.norm(), vp.norm());
    if nu <= tol || nv <= tol {
        return Vec::new(); // a point on the axis does not move; no angle changes its distance
    }
    // shrink delta by the fixed component along the axis
    let along = w.dot(&(p - q));
    let d2 = delta * delta - along * along;
    if d2 < -tol {
        return Vec::new();
    }
    let theta0 = w.dot(&up.cross(&vp)).atan2(up.dot(&vp));
    let cos_phi = (nu * nu + nv * nv - d2.max(0.0)) / (2.0 * nu * nv);
    if cos_phi > 1.0 + tol || cos_phi < -1.0 - tol {
        return Vec::new(); // circle and sphere do not intersect
    }
    let phi = cos_phi.clamp(-1.0, 1.0).acos();
    if phi <= tol {
        return vec![theta0]; // tangent: the two roots coincide
    }
    vec![theta0 + phi, theta0 - phi]
}

/// **Subproblem 2: rotation about two subsequent axes.** Find `(theta1, theta2)` such that rotating `p`
/// about axis 2 by `theta2` and then about axis 1 by `theta1` lands on `q`. The axes must **intersect**, at
/// the point `r`.
///
/// MLS eq. (3.20)-(3.25). The construction finds the intermediate point `c` that `p` reaches after the
/// second rotation, writing `z = c - r` in the (generally non-orthogonal) basis
/// `z = alpha*omega1 + beta*omega2 + gamma*(omega1 x omega2)`. Because a rotation preserves the component
/// along its own axis, `omega2 . u = omega2 . z` and `omega1 . v = omega1 . z` give a 2x2 linear system for
/// `alpha` and `beta`, and `|z| = |u|` then gives `gamma^2` — which may be negative (no solution), zero
/// (one), or positive (two). Each `c` is fed back through [`subproblem1`].
///
/// Returns up to two `(theta1, theta2)` pairs. **Parallel axes are rejected** with an empty result: when
/// `omega1 x omega2 == 0` the basis above degenerates, and MLS handles coincident axes separately by
/// reduction to Subproblem 1 with `theta1 + theta2` free — a one-parameter family, not a discrete pair, so
/// it cannot be returned in this shape. Use [`subproblem1`] directly for that case.
pub fn subproblem2(
    omega1: &Vector3<f64>,
    omega2: &Vector3<f64>,
    r: &Vector3<f64>,
    p: &Vector3<f64>,
    q: &Vector3<f64>,
    tol: f64,
) -> Vec<(f64, f64)> {
    let (w1, w2) = (omega1.normalize(), omega2.normalize());
    let cross = w1.cross(&w2);
    let cn2 = cross.norm_squared();
    if cn2 <= tol * tol {
        return Vec::new(); // parallel or coincident: not a discrete two-solution problem
    }
    let (u, v) = (p - r, q - r);
    let dot12 = w1.dot(&w2);
    let den = dot12 * dot12 - 1.0;
    if den.abs() <= f64::EPSILON {
        return Vec::new();
    }
    let alpha = (dot12 * w2.dot(&u) - w1.dot(&v)) / den;
    let beta = (dot12 * w1.dot(&v) - w2.dot(&u)) / den;
    let g2 = (u.norm_squared() - alpha * alpha - beta * beta - 2.0 * alpha * beta * dot12) / cn2;
    if g2 < -tol {
        return Vec::new();
    }
    let gammas: Vec<f64> = if g2.abs() <= tol { vec![0.0] } else { let g = g2.sqrt(); vec![g, -g] };

    let mut out = Vec::new();
    for gamma in gammas {
        let c = r + w1 * alpha + w2 * beta + cross * gamma;
        // theta2 rotates p to c about axis 2; theta1 rotates c to q about axis 1.
        if let (Some(t2), Some(t1)) =
            (subproblem1(&w2, r, p, &c, tol), subproblem1(&w1, r, &c, q, tol))
        {
            out.push((t1, t2));
        }
    }
    out
}

/// Apply a zero-pitch twist: rotate `p` about the axis `(omega, r)` by `theta`.
///
/// The action a subproblem solves for, offered so a caller can verify a returned angle by substitution
/// rather than trusting it.
pub fn rotate_about_axis(
    omega: &Vector3<f64>,
    r: &Vector3<f64>,
    theta: f64,
    p: &Vector3<f64>,
) -> Vector3<f64> {
    r + exp_so3(&(omega.normalize() * theta)) * (p - r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn subproblem1_recovers_the_angle_that_generated_the_point() {
        // Round-trip: pick an angle, rotate, and ask the subproblem to recover it.
        let w = Vector3::new(0.3, -0.5, 0.81).normalize();
        let r = Vector3::new(0.2, 0.1, -0.4);
        let p = Vector3::new(1.0, 0.7, 0.3);
        for theta in [0.0, 0.4, -1.2, 2.9, 3.0] {
            let q = rotate_about_axis(&w, &r, theta, &p);
            let got = subproblem1(&w, &r, &p, &q, TOL).expect("consistent by construction");
            assert!((got - theta).abs() < 1e-9, "theta={theta}, recovered {got}");
            // and the defining equation, which is the claim that matters
            assert!((rotate_about_axis(&w, &r, got, &p) - q).norm() < 1e-9);
        }
    }

    #[test]
    fn subproblem1_refuses_an_unreachable_point() {
        // A rotation is an isometry fixing the axis, so it cannot change the height along the axis nor the
        // radius from it. Both violations must be refused rather than least-squares fitted.
        let w = Vector3::new(0.0, 0.0, 1.0);
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        assert!(subproblem1(&w, &r, &p, &Vector3::new(0.0, 1.0, 0.5), TOL).is_none(), "wrong height");
        assert!(subproblem1(&w, &r, &p, &Vector3::new(0.0, 2.0, 0.0), TOL).is_none(), "wrong radius");
        // on-axis points: any angle works, and a representative is returned
        assert_eq!(subproblem1(&w, &r, &Vector3::new(0.0, 0.0, 3.0), &Vector3::new(0.0, 0.0, 3.0), TOL), Some(0.0));
    }

    #[test]
    fn subproblem3_returns_two_one_or_zero_solutions_and_each_hits_the_distance() {
        // Circle meets sphere. Unit circle in the xy-plane about z, target q on the +x axis at distance 2.
        let w = Vector3::new(0.0, 0.0, 1.0);
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(2.0, 0.0, 0.0);

        // delta = 1.5 cuts the circle twice
        let two = subproblem3(&w, &r, &p, &q, 1.5, TOL);
        assert_eq!(two.len(), 2, "a secant sphere gives two solutions, got {two:?}");
        for t in &two {
            let hit = rotate_about_axis(&w, &r, *t, &p);
            assert!(((hit - q).norm() - 1.5).abs() < 1e-9, "distance not met: {}", (hit - q).norm());
        }
        assert!((two[0] - two[1]).abs() > 1e-6, "the two roots must be distinct");

        // delta = 1.0 is tangent at theta = 0 (p is already distance 1 from q, at the near point)
        let one = subproblem3(&w, &r, &p, &q, 1.0, TOL);
        assert_eq!(one.len(), 1, "tangency gives one solution, got {one:?}");
        assert!(one[0].abs() < 1e-9);

        // delta = 0.5 never reaches the circle; delta = 5.0 overshoots it
        assert!(subproblem3(&w, &r, &p, &q, 0.5, TOL).is_empty(), "too close to intersect");
        assert!(subproblem3(&w, &r, &p, &q, 5.0, TOL).is_empty(), "too far to intersect");
    }

    #[test]
    fn subproblem3_handles_an_axis_offset_along_omega() {
        // The delta' projection is what makes an out-of-plane q work; this fails if it is dropped.
        let w = Vector3::new(0.0, 0.0, 1.0);
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(1.6, 0.0, 0.8); // 0.8 above the circle's plane
        let sols = subproblem3(&w, &r, &p, &q, 1.3, TOL);
        assert!(!sols.is_empty(), "should intersect");
        for t in &sols {
            let hit = rotate_about_axis(&w, &r, *t, &p);
            assert!(((hit - q).norm() - 1.3).abs() < 1e-9, "got {}", (hit - q).norm());
        }
    }

    #[test]
    fn subproblem2_solves_two_intersecting_axes_by_substitution() {
        // Two perpendicular axes meeting at the origin — the shoulder/elbow pattern the subproblem exists
        // for. Generate a reachable q from known angles, then check every returned pair reproduces it.
        let w1 = Vector3::new(0.0, 0.0, 1.0);
        let w2 = Vector3::new(0.0, 1.0, 0.0);
        let r = Vector3::zeros();
        let p = Vector3::new(0.8, 0.2, 0.5);
        let (t1_true, t2_true) = (0.7, -0.5);
        let q = rotate_about_axis(&w1, &r, t1_true, &rotate_about_axis(&w2, &r, t2_true, &p));

        let sols = subproblem2(&w1, &w2, &r, &p, &q, 1e-9);
        assert!(!sols.is_empty(), "the generating pair must be found");
        for (t1, t2) in &sols {
            let got = rotate_about_axis(&w1, &r, *t1, &rotate_about_axis(&w2, &r, *t2, &p));
            assert!((got - q).norm() < 1e-8, "pair ({t1}, {t2}) gives {got:?}, want {q:?}");
        }
        // the true pair is among them
        assert!(
            sols.iter().any(|(a, b)| (a - t1_true).abs() < 1e-7 && (b - t2_true).abs() < 1e-7),
            "generating pair ({t1_true}, {t2_true}) missing from {sols:?}"
        );
    }

    #[test]
    fn subproblem2_rejects_parallel_axes_rather_than_returning_nonsense() {
        // With omega1 x omega2 == 0 the (omega1, omega2, omega1 x omega2) basis degenerates. MLS reduces
        // this case to Subproblem 1 with theta1 + theta2 free — a one-parameter family, which cannot be
        // expressed as discrete pairs, so an empty result is the correct answer for this signature.
        let w = Vector3::new(0.0, 0.0, 1.0);
        let r = Vector3::zeros();
        let p = Vector3::new(1.0, 0.0, 0.0);
        let q = Vector3::new(0.0, 1.0, 0.0);
        assert!(subproblem2(&w, &w, &r, &p, &q, 1e-9).is_empty(), "coincident axes");
        assert!(subproblem2(&w, &(-w), &r, &p, &q, 1e-9).is_empty(), "antiparallel axes");
        // and the caller's fallback genuinely works for that case
        assert!(subproblem1(&w, &r, &p, &q, 1e-9).is_some());
    }

    #[test]
    fn subproblem2_reports_no_solution_when_the_circles_miss() {
        // q at the wrong radius from the intersection point is unreachable: both rotations fix |x - r|.
        let w1 = Vector3::new(0.0, 0.0, 1.0);
        let w2 = Vector3::new(0.0, 1.0, 0.0);
        let r = Vector3::zeros();
        let p = Vector3::new(0.8, 0.2, 0.5);
        let far = Vector3::new(9.0, 0.0, 0.0); // |far| >> |p|
        assert!(subproblem2(&w1, &w2, &r, &p, &far, 1e-9).is_empty());
    }
}
