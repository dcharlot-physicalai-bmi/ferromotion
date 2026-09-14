//! **From a compound collider to a solver's constraint set** — the step that was missing between
//! [`compound_contacts`](crate::compound_contacts) and [`solve_contacts_pgs`](crate::solve_contacts_pgs).
//!
//! [`compound_contacts`](crate::compound_contacts) answers a question about *geometry*: which parts of
//! two bodies are within a margin, how deep, and along what normal. [`PgsContact`] asks a question
//! about *mechanics*: give me a `3 × nv` matrix mapping generalized velocity to the contact point's
//! velocity, with **the normal on row 2**, plus a signed gap and a friction coefficient. Nothing joined
//! the two, which is why the compound collider shipped with zero consumers: a caller had to know the
//! contact-frame convention, the Jacobian difference and the sign of the normal, and get all three
//! right, before the collider was of any use at all.
//!
//! # The sign chain, stated once
//!
//! Three conventions meet here and each is easy to get backwards:
//!
//! * [`CompoundContact::normal`](crate::CompoundContact::normal) points **from A toward B**, and
//!   translating A by `gap·normal` brings the pair to touching. So the gap *decreases* along `+normal`,
//!   and the direction A must move to get clear is `−normal`.
//! * [`solve_contacts_pgs`](crate::solve_contacts_pgs) reads `J.row(2) · v` and drives the normal
//!   impulse up when it is **negative**. So row 2 must be the direction along which a positive velocity
//!   means *separating* — which is `−normal`, the same direction as `d(gap)/dt`.
//! * The constraint is on the **relative** motion, so the Jacobian is `J_A − J_B`. A static
//!   environment contributes zero, which is what [`StaticBody`] is.
//!
//! Putting those together, row 2 is `(−normal)ᵀ (J_A − J_B)` and `phi` is the signed gap. Rows 0 and 1
//! are any orthonormal pair spanning the contact plane; the Coulomb cone here is circular, so the
//! choice does not change the answer, but the pair must be orthonormal or the friction bound is scaled.
//!
//! # ⛔ One contact per part pair, which is exact for a point-like contact and an approximation for a face
//!
//! This produces **one** constraint per part pair within the margin. For a contact that is genuinely
//! point-like — a ball foot, a rounded fingertip, a capsule end — that is the whole truth. For two flat
//! faces meeting it is not: a box resting squarely on the floor needs a contact *manifold* of several
//! points to resist tipping, and the support function of a flat face returns an arbitrary vertex of it,
//! so the single point this produces sits at a corner. Manifold generation is a separate piece of work
//! and is deliberately not attempted here rather than approximated silently.
//!
//! What this does mean is that the *count* is right: a plate across two separated blocks yields two
//! constraints, which is the distinction the compound collider exists to make and the reason a
//! closest-pair query cannot feed a solver.
//!
//! # The oracle
//!
//! `the_block_on_an_incline_slides_exactly_when_mu_falls_below_tan_theta` puts a point-like body on a
//! slope tilted 30° and checks the textbook result: the body accelerates at `g(sin θ − μ cos θ)` down
//! the slope while `μ < tan θ` and holds still once `μ ≥ tan θ`. Both regimes are one expression,
//! `g·max(0, sin θ − μ cos θ)`, so the assertion crosses the knee rather than sitting on one side of
//! it. The normal impulse is `m g cos θ dt` throughout.
//!
//! That oracle is external to every line here, and it is sensitive to all three signs above: reverse
//! the normal row and the body falls through the slope at the full `g`; drop the orthonormality of the
//! tangent pair and the critical `μ` moves.

use crate::gjk::{ConvexPoints, Support};
use crate::{compound_contacts, CompoundContact, CompoundHull, PgsContact, Robot};
use nalgebra::{DMatrix, Vector3};

/// **How a body's generalized velocity moves a material point on one of its parts.** The one thing a
/// contact solver needs from a body and cannot get from geometry.
///
/// A pair of bodies in contact must share a generalized velocity vector — the constraint is on their
/// *relative* motion, so both Jacobians have to be expressed in the same coordinates. For a robot
/// against a fixed world that means one [`StaticBody`] with the robot's own `nv`; for self-collision it
/// means two views of the same robot, each mapping its own part indices to different links.
pub trait ContactJacobian {
    /// Length of the generalized velocity vector. Both bodies of a pair must agree.
    fn nv(&self) -> usize;
    /// The `3 × nv` Jacobian of the **world** velocity of the material point currently at `p_world`,
    /// which lies on part `part` of this body. A part that cannot move returns zeros.
    fn point_jacobian(&self, part: usize, p_world: &Vector3<f64>) -> DMatrix<f64>;
}

/// An immobile body: the ground, a table, a fixture. Contributes no columns of motion, so the
/// constraint reduces to the other body's own Jacobian.
#[derive(Clone, Copy, Debug)]
pub struct StaticBody {
    pub nv: usize,
}

impl ContactJacobian for StaticBody {
    fn nv(&self) -> usize {
        self.nv
    }
    fn point_jacobian(&self, _part: usize, _p: &Vector3<f64>) -> DMatrix<f64> {
        DMatrix::zeros(3, self.nv)
    }
}

/// A serial chain whose compound parts are each rigidly attached to a link.
///
/// ⛔ `frames[k]` is a **count of joints**, not a zero-based link index: it is the `upto` argument
/// [`Robot::point_jacobian`] and [`Robot::frame_pose`] take, so a part on the **last** link of an
/// `n`-joint chain has `frames[k] == n`, not `n − 1`. Passing the index instead silently drops the
/// last joint's column, and a dropped column is not a small error: the first version of this module's
/// own incline test passed `2` for a three-joint slider, the normal row lost its `z` component
/// entirely, and the block fell through the slope at the full `g` with a contact reported and an
/// impulse of exactly zero.
///
/// A part index with no entry in `frames` is treated as immobile rather than panicking, because a
/// caller assembling a hull from a description can end up with more parts than it mapped.
#[derive(Clone, Copy, Debug)]
pub struct SerialLinkParts<'a> {
    pub robot: &'a Robot,
    pub q: &'a [f64],
    pub frames: &'a [usize],
}

impl ContactJacobian for SerialLinkParts<'_> {
    fn nv(&self) -> usize {
        self.robot.dof()
    }
    fn point_jacobian(&self, part: usize, p: &Vector3<f64>) -> DMatrix<f64> {
        match self.frames.get(part) {
            Some(&f) => self.robot.point_jacobian(self.q, f, p),
            None => DMatrix::zeros(3, self.robot.dof()),
        }
    }
}

/// The constraint set, paired with the geometry that produced it so an impulse can be attributed back
/// to a part pair.
#[derive(Clone, Debug)]
pub struct CompoundPgs {
    /// Ready for [`solve_contacts_pgs`](crate::solve_contacts_pgs).
    pub contacts: Vec<PgsContact>,
    /// Same length and order as `contacts`: `lambda[i]` from the solver belongs to `geometry[i]`, whose
    /// `part_a`/`part_b` say which pieces of the two bodies met.
    pub geometry: Vec<CompoundContact>,
    /// Pairs inside the margin that produced **no** constraint because neither GJK nor EPA could give a
    /// direction — an exactly-touching pair, or one whose Minkowski difference is flat.
    ///
    /// ⛔ Non-zero means the constraint set is incomplete. Inventing a normal for those pairs would put
    /// a constraint on an axis nothing measured, so they are dropped and counted instead of guessed.
    pub skipped: usize,
}

/// Two orthonormal vectors spanning the plane perpendicular to the unit vector `n`.
fn tangents(n: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    // Cross against whichever axis `n` leans on LEAST. The cross product's length is the sine of the
    // angle between the two, so the least-aligned axis keeps it furthest from zero; crossing against a
    // fixed axis loses all precision exactly when the normal happens to be that axis, which for a
    // contact normal is the common case rather than a corner one.
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let a = if ax <= ay && ax <= az {
        Vector3::x()
    } else if ay <= az {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let t1 = n.cross(&a).normalize();
    (t1, n.cross(&t1))
}

/// The point at which to evaluate the Jacobians.
///
/// Separated: the midpoint of GJK's witness pair, which lies on the shortest segment between the two
/// shapes. Penetrating: **A's deepest point into B**, brought back half the penetration along the
/// normal so the point sits midway between the two surfaces. GJK's witness points are meaningless once
/// the shapes overlap and EPA returns none, so the support function is the only thing left.
///
/// ⛔ **This is taken on A, and that asymmetry is deliberate.** The first version averaged the two
/// bodies' support points, which is wrong whenever either shape presents a FLAT FACE: the support
/// function of a face returns an arbitrary vertex of it, so a spike penetrating a 4 m floor slab got a
/// contact point at the slab's corner, metres from the contact, and a body with any revolute joint then
/// got a lever arm to match. Only the *normal* coordinate of a face's support point carries
/// information; its tangential position does not.
///
/// Taking the point on A means A's support function must be the informative one, so **pass the more
/// point-like body first**. That is the same single-contact-per-pair limit the module docs state, seen
/// from the other side: a genuine face-against-face pair needs a manifold, not a better guess at one
/// point.
fn contact_point(pa: &ConvexPoints, c: &CompoundContact) -> Vector3<f64> {
    if c.intersecting {
        // `normal` points A→B and `gap` is negative here, so this steps back toward A by half the depth.
        pa.support(&c.normal) + 0.5 * c.gap * c.normal
    } else {
        0.5 * (c.witness_a + c.witness_b)
    }
}

/// **Build a Gauss-Seidel constraint set from two compound colliders.**
///
/// `a` and `b` must already be in world coordinates (see
/// [`CompoundHull::transformed`](crate::CompoundHull::transformed)), and `ja`/`jb` must describe the
/// same generalized velocity. `margin` is the contact-activation distance, passed straight to
/// [`compound_contacts`](crate::compound_contacts); `mu` is the friction coefficient of the material
/// pair.
///
/// # Errors
///
/// The two bodies disagreeing on `nv`, a non-finite or negative `mu`, or a provider returning a
/// wrongly-shaped Jacobian. All three are caller mistakes that would otherwise produce a silently wrong
/// solve, so they are reported rather than absorbed.
pub fn compound_pgs_contacts(
    a: &CompoundHull,
    ja: &dyn ContactJacobian,
    b: &CompoundHull,
    jb: &dyn ContactJacobian,
    margin: f64,
    mu: f64,
) -> Result<CompoundPgs, String> {
    let nv = ja.nv();
    if nv != jb.nv() {
        return Err(format!("the two bodies must share a generalized velocity: nv {} vs {}", nv, jb.nv()));
    }
    if nv == 0 {
        return Err("a body with no generalized velocity has no constraint to write".into());
    }
    if !mu.is_finite() || mu < 0.0 {
        return Err(format!("the friction coefficient must be finite and non-negative, got {mu}"));
    }

    let mut out = CompoundPgs { contacts: Vec::new(), geometry: Vec::new(), skipped: 0 };
    for c in compound_contacts(a, b, margin) {
        // A zero normal is the honest answer for an exactly-touching pair, and it is unusable: it would
        // put row 2 of the Jacobian at zero, which the solver reads as a contact that no motion can
        // ever violate.
        let nn = c.normal.norm();
        // `|| is_nan()` rather than a bare `<=`, because every comparison against NaN is false and a
        // NaN normal must be skipped too. That clause is defensive and is NOT exercised by a test:
        // `ConvexPoints::support` skips non-finite vertices and `from_convex_parts` refuses a hull
        // containing one, so no public route reaches here with a NaN. Stated rather than implied.
        if nn <= 0.5 || nn.is_nan() {
            out.skipped += 1;
            continue;
        }
        let Some(pa) = a.parts.get(c.part_a) else {
            out.skipped += 1;
            continue;
        };
        let p = contact_point(pa, &c);
        let (jac_a, jac_b) = (ja.point_jacobian(c.part_a, &p), jb.point_jacobian(c.part_b, &p));
        for (which, m) in [("A", &jac_a), ("B", &jac_b)] {
            if m.nrows() != 3 || m.ncols() != nv {
                return Err(format!("body {which} returned a {}×{} Jacobian, expected 3×{nv}", m.nrows(), m.ncols()));
            }
        }
        let jrel = jac_a - jac_b;

        // Row 2 is the SEPARATING direction, which is −normal: see the sign chain in the module docs.
        let n_sep = c.normal / nn * -1.0;
        let (t1, t2) = tangents(&n_sep);
        let mut j = DMatrix::zeros(3, nv);
        for (r, axis) in [t1, t2, n_sep].iter().enumerate() {
            j.row_mut(r).copy_from(&(axis.transpose() * &jrel));
        }
        out.contacts.push(PgsContact { j, phi: c.gap, mu });
        out.geometry.push(c);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{forward_dynamics, mass_matrix, solve_contacts_pgs_with, Iso, Joint, LinkInertia, PgsStabilization};
    use nalgebra::{DVector, Matrix3, Translation3, UnitQuaternion};

    const G: f64 = 9.81;
    const MASS: f64 = 2.0;
    const DT: f64 = 1e-3;

    /// A body free to translate and nothing else: three prismatic joints along the world axes, with all
    /// the mass on the last link. Its mass matrix is `m·I`, so the Delassus block is isotropic and the
    /// Gauss-Seidel sweep reaches the exact answer immediately — which is what lets the analytic
    /// comparison below be at 1e-9 rather than at a convergence tolerance.
    ///
    /// It also removes the contact POINT from the problem: a prismatic chain's point Jacobian is the
    /// same identity everywhere, so nothing here depends on where on the part the contact was placed.
    /// That is deliberate. The single-point-per-pair approximation is a separate matter from the sign
    /// and frame conventions this test exists to pin.
    fn slider() -> (Robot, Vec<LinkInertia>) {
        let robot = Robot {
            joints: vec![
                Joint::prismatic(Iso::identity(), Vector3::x()),
                Joint::prismatic(Iso::identity(), Vector3::y()),
                Joint::prismatic(Iso::identity(), Vector3::z()),
            ],
            ee_offset: Iso::identity(),
        };
        let mut inertia = vec![LinkInertia::zero(); 3];
        inertia[2] = LinkInertia { mass: MASS, com: Vector3::zeros(), inertia: Matrix3::identity() * 1e-3 };
        (robot, inertia)
    }

    fn box_pts(c: Vector3<f64>, h: Vector3<f64>) -> ConvexPoints {
        ConvexPoints {
            pts: (0..8)
                .map(|i| {
                    c + Vector3::new(
                        if i & 1 == 0 { -h.x } else { h.x },
                        if i & 2 == 0 { -h.y } else { h.y },
                        if i & 4 == 0 { -h.z } else { h.z },
                    )
                })
                .collect(),
        }
    }

    /// A point-like body: a tetrahedron whose apex is its unique support in the `-up` direction, so the
    /// deepest point is a single vertex and the contact is genuinely a point rather than a face.
    fn spike(apex: Vector3<f64>, up: Vector3<f64>) -> CompoundHull {
        let (t1, t2) = tangents(&up);
        let mut pts = vec![apex];
        for k in 0..3 {
            let a = k as f64 * core::f64::consts::TAU / 3.0;
            pts.push(apex + up * 0.05 + (t1 * a.cos() + t2 * a.sin()) * 0.03);
        }
        CompoundHull::from_convex_parts(vec![ConvexPoints { pts }]).expect("a tetrahedron is a part")
    }

    /// A slab of half-extents `(2, 2, 0.5)` tilted by `theta` about y, positioned so the centre of its
    /// top face is the world origin and that face's outward normal is `R·ẑ`.
    fn incline(theta: f64) -> (CompoundHull, Vector3<f64>) {
        let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), theta);
        let n = r * Vector3::z();
        let local_top = Vector3::new(0.0, 0.0, 0.5);
        let tf = Iso::from_parts(Translation3::from(-(r * local_top)), r);
        let slab = CompoundHull::from_convex_parts(vec![box_pts(Vector3::zeros(), Vector3::new(2.0, 2.0, 0.5))])
            .expect("one part");
        (slab.transformed(&tf), n)
    }

    /// **The tangent pair must be orthonormal for EVERY normal, the world axes included.**
    ///
    /// ⛔ This test exists because a mutation survived without it. Replacing the least-aligned-axis
    /// choice with a fixed `x̂` passed all eight of this module's other tests: the slope fixtures rotate
    /// about `ŷ`, and no such rotation produces a normal exactly parallel to `x̂` — even `θ = π/2` gives
    /// `(1, 0, 6.1e-17)`, whose cross product with `x̂` is a clean `ŷ`. The degenerate case is
    /// unreachable through the geometry, so it has to be asked of the helper directly.
    ///
    /// Crossing against a fixed axis gives a zero vector when the normal IS that axis, `normalize()`
    /// turns that into NaN, and the whole contact frame is then NaN with no error raised anywhere.
    #[test]
    fn the_tangent_pair_is_orthonormal_for_every_normal_including_the_world_axes() {
        let mut cases: Vec<Vector3<f64>> = Vec::new();
        for k in 0..3 {
            cases.push(Vector3::ith(k, 1.0));
            cases.push(-Vector3::ith(k, 1.0));
        }
        // obliques, near-axis, and two-axis diagonals: the places a tie-break can go wrong
        for v in [
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1e-18, 1e-18),
            Vector3::new(-1e-18, 1.0, -1e-18),
            Vector3::new(0.3, -0.7, 0.64),
        ] {
            cases.push(v.normalize());
        }

        let mut worst = 0.0f64;
        for n in &cases {
            let (t1, t2) = tangents(n);
            assert!(
                t1.iter().chain(t2.iter()).all(|c| c.is_finite()),
                "normal {:?} gave a non-finite tangent pair {:?} / {:?}",
                n.as_slice(), t1.as_slice(), t2.as_slice()
            );
            for (label, e) in [
                ("|t1| − 1", t1.norm() - 1.0),
                ("|t2| − 1", t2.norm() - 1.0),
                ("t1·t2", t1.dot(&t2)),
                ("t1·n", t1.dot(n)),
                ("t2·n", t2.dot(n)),
            ] {
                assert!(e.abs() < 1e-14, "normal {:?}: {label} = {e:.3e}", n.as_slice());
                worst = worst.max(e.abs());
            }
        }
        eprintln!("  {} normals including all six world axes: worst orthonormality error {worst:.3e}", cases.len());
    }

    /// **THE ORACLE.** A block on a slope: it accelerates down at `g(sin θ − μ cos θ)` while
    /// `μ < tan θ` and holds still once `μ ≥ tan θ`, and the normal impulse is `m g cos θ dt` either
    /// way. Both regimes are the single expression `g·max(0, sin θ − μ cos θ)`, so the assertion crosses
    /// the knee instead of describing one side of it.
    ///
    /// Nothing in this fixture is axis-aligned with the contact: the slope normal is
    /// `(sin 30°, 0, cos 30°)`, so a Jacobian that assumed the world `z` axis was the normal — which is
    /// all the floor-contact path in this crate had ever needed — cannot pass.
    #[test]
    fn the_block_on_an_incline_slides_exactly_when_mu_falls_below_tan_theta() {
        let theta = core::f64::consts::FRAC_PI_6; // 30°, tan θ = 0.5773502691896257
        let (slab, n) = incline(theta);
        let (robot, inertia) = slider();

        // A millimetre of penetration, with gap feedback switched OFF (erp = 0). The analytic result is
        // the rigid velocity-level one; leaving the default stabilisation on would add a recovery
        // velocity and the comparison would be against a different problem.
        let pen = 1e-3;
        let stab = PgsStabilization { slop: 0.0, erp: 0.0, max_correction: 0.0 };
        assert_eq!(stab.normal_bias(-pen, DT), 0.0, "the fixture must carry no gap feedback at all");

        let apex = -n * pen;
        let q = [apex.x, apex.y, apex.z];
        let body = spike(apex, n);

        // the fixture's own premise: an isotropic mass matrix, so the sweep is exact in one pass
        let mm = mass_matrix(&robot, &inertia, &q);
        assert!(
            (mm.clone() - DMatrix::identity(3, 3) * MASS).amax() < 1e-12,
            "a three-prismatic slider must have mass matrix m·I, got {mm}"
        );

        let qdd = forward_dynamics(&robot, &inertia, &q, &[0.0; 3], &[0.0; 3], Vector3::new(0.0, 0.0, -G));
        let v_free = DVector::from_iterator(3, qdd.iter().map(|a| a * DT));
        assert!((v_free[2] + G * DT).abs() < 1e-12, "free fall for one step is −g·dt on z, got {}", v_free[2]);

        let downhill = Vector3::new(theta.cos(), 0.0, -theta.sin());
        let tan_theta = theta.tan();
        let mut rows = Vec::new();
        for &mu in &[0.0, 0.3, 0.5, 0.55, tan_theta, 0.6, 0.8, 1.2] {
            let set = compound_pgs_contacts(&body, &SerialLinkParts { robot: &robot, q: &q, frames: &[3] }, &slab, &StaticBody { nv: 3 }, 1e-4, mu)
                .expect("the bodies agree on nv");
            assert_eq!(set.contacts.len(), 1, "one part against one part is one contact");
            assert_eq!(set.skipped, 0, "a penetrating pair has a normal, so nothing may be skipped");
            assert!((set.geometry[0].gap + pen).abs() < 1e-6, "the fixture penetrates by {pen}, got {}", set.geometry[0].gap);

            let res = solve_contacts_pgs_with(&mm, &v_free, &set.contacts, DT, 40, None, stab);
            let v = Vector3::new(res.v_next[0], res.v_next[1], res.v_next[2]);

            // ONE expression across the knee, which is what makes this a single property rather than
            // two cases that happen to agree.
            let expected = G * DT * (theta.sin() - mu * theta.cos()).max(0.0);
            rows.push((mu, (v.norm() * 1e9).round() / 1e9, (expected * 1e9).round() / 1e9));
            assert!(
                (v.norm() - expected).abs() < 1e-9,
                "μ = {mu}: speed after one step must be g·dt·max(0, sin θ − μ cos θ) = {expected:.9}, got {:.9}",
                v.norm()
            );
            if expected > 0.0 {
                assert!(
                    (v.normalize() - downhill).norm() < 1e-9,
                    "μ = {mu}: the motion must be straight down the slope {downhill:?}, got {:?}",
                    v.normalize()
                );
            }
            // the normal impulse is the same in both regimes: the slope carries the weight either way
            assert!(
                (res.lambda[0].z - MASS * G * theta.cos() * DT).abs() < 1e-12,
                "μ = {mu}: the normal impulse must be m·g·cos θ·dt = {:.9}, got {:.9}",
                MASS * G * theta.cos() * DT,
                res.lambda[0].z
            );
        }
        eprintln!("  θ = 30°, tan θ = {tan_theta:.6}   (μ, |v| after one step, analytic):");
        for (mu, got, want) in &rows {
            eprintln!("    μ = {mu:<6.4}  {got:.9}  {want:.9}");
        }
    }

    /// The contact frame is a rotation, not merely three directions. For the slider the relative
    /// Jacobian is the identity, so the three rows ARE the frame and `J·Jᵀ` must be exactly `I`.
    ///
    /// A tangent pair that is not orthonormal still passes a stick/slide test at one friction value —
    /// it moves the critical `μ` rather than breaking it outright — so the frame is pinned directly.
    #[test]
    fn the_contact_frame_is_orthonormal_and_row_two_is_the_separating_direction() {
        let (robot, inertia) = slider();
        let _ = inertia;
        // A spread of normals, including each world axis, where a fixed cross-product axis degenerates.
        let mut worst_gram = 0.0f64;
        let mut worst_dot = 1.0f64;
        for theta in [0.0f64, 0.3, core::f64::consts::FRAC_PI_6, 1.2, core::f64::consts::FRAC_PI_2 - 1e-9] {
            let (slab, n) = incline(theta);
            let apex = -n * 1e-3;
            let q = [apex.x, apex.y, apex.z];
            let set = compound_pgs_contacts(&spike(apex, n), &SerialLinkParts { robot: &robot, q: &q, frames: &[3] }, &slab, &StaticBody { nv: 3 }, 1e-4, 0.5)
                .expect("nv agrees");
            let j = &set.contacts[0].j;
            worst_gram = worst_gram.max((j * j.transpose() - DMatrix::identity(3, 3)).amax());
            // row 2 must be the slope's outward normal: the direction the body moves to get clear
            let row2 = Vector3::new(j[(2, 0)], j[(2, 1)], j[(2, 2)]);
            worst_dot = worst_dot.min(row2.dot(&n));
        }
        eprintln!("  over five slope angles: worst |J·Jᵀ − I| = {worst_gram:.3e}, worst row2·n̂ = {worst_dot:.12}");
        assert!(worst_gram < 1e-12, "the contact frame must be orthonormal, worst deviation {worst_gram:.3e}");
        assert!(worst_dot > 1.0 - 1e-12, "row 2 must be the separating direction (+n̂), worst dot {worst_dot:.12}");
    }

    /// **The second constraint has to be load-bearing, or the count proves nothing.**
    ///
    /// A first version of this test rested the body on two flat blocks. It got two constraints and the
    /// normal impulses summed to the weight — and the measured split was `[0.01962, 0.0]`: the first
    /// contact carried everything and the second carried nothing at all. Under a translation-only body
    /// on a flat floor one support is sufficient, so a test built that way asserts a COUNT and never
    /// finds out whether the second constraint does any work. Deleting it would have changed nothing.
    ///
    /// A V-groove fixes that, and it makes the numbers analytic. Two planes at ±30° hold a frictionless
    /// body only because their normals bracket gravity; by symmetry each impulse is exactly
    /// `m g dt / (2 cos θ)`. Hand a solver only the closest pair — which is all
    /// [`compound_distance`](crate::compound_distance) returns — and the body slides down the plane it
    /// kept at `g dt sin θ`. That contrast is the claim the compound collider exists to make, measured
    /// on both sides.
    #[test]
    fn a_v_groove_needs_both_constraints_and_one_alone_lets_the_body_slide() {
        let (robot, inertia) = slider();
        let theta = core::f64::consts::FRAC_PI_6;
        let pen = 1e-3;

        // two slabs, each with its top face through the origin, tilted toward each other
        let mut parts = Vec::new();
        let mut normals = Vec::new();
        for sign in [1.0f64, -1.0] {
            let (slab, n) = incline(sign * theta);
            normals.push(n);
            parts.push(slab.parts[0].clone());
        }
        let groove = CompoundHull::from_convex_parts(parts).expect("two planes");
        assert!(
            (normals[0].dot(&normals[1]) - (2.0 * theta).cos()).abs() < 1e-12,
            "the two faces must meet at 2θ, got {}", normals[0].dot(&normals[1])
        );

        // the apex sits `pen/cos θ` below the vertex, so it penetrates EACH plane by exactly `pen`
        let apex = Vector3::new(0.0, 0.0, -pen / theta.cos());
        let q = [apex.x, apex.y, apex.z];
        let body = spike(apex, Vector3::z());

        let set = compound_pgs_contacts(&body, &SerialLinkParts { robot: &robot, q: &q, frames: &[3] }, &groove, &StaticBody { nv: 3 }, 1e-4, 0.0)
            .expect("nv agrees");
        assert_eq!(set.contacts.len(), 2, "one foot in a V touches two planes, got {}", set.contacts.len());
        assert_eq!(set.skipped, 0);
        assert_ne!(set.geometry[0].part_b, set.geometry[1].part_b, "the two constraints must name different planes");
        for g in &set.geometry {
            assert!((g.gap + pen).abs() < 1e-6, "each plane is penetrated by {pen}, got {}", g.gap);
        }
        // the control on the collider itself: the closest-pair query returns ONE of these two
        let single = crate::compound_distance(&body, &groove).expect("a pair");
        eprintln!("  compound_distance returns 1 contact (part {}↔{}); compound_pgs_contacts returns {}", single.part_a, single.part_b, set.contacts.len());

        let mm = mass_matrix(&robot, &inertia, &q);
        let qdd = forward_dynamics(&robot, &inertia, &q, &[0.0; 3], &[0.0; 3], Vector3::new(0.0, 0.0, -G));
        let v_free = DVector::from_iterator(3, qdd.iter().map(|a| a * DT));
        let stab = PgsStabilization { slop: 0.0, erp: 0.0, max_correction: 0.0 };

        // ---- both constraints: held, and each impulse is pinned by symmetry ----
        let both = solve_contacts_pgs_with(&mm, &v_free, &set.contacts, DT, 400, None, stab);
        let each = MASS * G * DT / (2.0 * theta.cos());
        let v_both = Vector3::new(both.v_next[0], both.v_next[1], both.v_next[2]);
        eprintln!(
            "  both planes: impulses {:?} vs m·g·dt/(2cos θ) = {:.9}, |v| = {:.3e}",
            both.lambda.iter().map(|l| (l.z * 1e9).round() / 1e9).collect::<Vec<_>>(), each, v_both.norm()
        );
        for (i, l) in both.lambda.iter().enumerate() {
            assert!((l.z - each).abs() < 1e-9, "plane {i}: impulse must be {each:.9}, got {:.9}", l.z);
        }
        assert!(v_both.norm() < 1e-9, "a body wedged in a V must not move, got {:?}", v_both);

        // ---- one constraint only, which is what a closest-pair query would have handed over ----
        let one = solve_contacts_pgs_with(&mm, &v_free, &set.contacts[..1], DT, 400, None, stab);
        let v_one = Vector3::new(one.v_next[0], one.v_next[1], one.v_next[2]);
        let slide = G * DT * theta.sin();
        eprintln!("  one plane only: |v| = {:.9} vs g·dt·sin θ = {:.9}  <- the constraint a planner's answer loses", v_one.norm(), slide);
        assert!(
            (v_one.norm() - slide).abs() < 1e-9,
            "with one plane the body must slide at g·dt·sin θ = {slide:.9}, got {:.9}", v_one.norm()
        );
        assert!(v_one.norm() > 1e3 * v_both.norm().max(1e-15), "the two answers must be qualitatively different, {:.3e} vs {:.3e}", v_one.norm(), v_both.norm());
    }

    /// **`phi` really is the signed gap, and the incline oracle cannot see that.** Both dynamic tests
    /// above switch gap feedback off (`erp = 0`) so the analytic rigid answer applies — which means
    /// `phi` is multiplied by zero and a mutation replacing it with `distance` passes them untouched.
    ///
    /// Turned on, `phi` has a closed form of its own: under [`PgsStabilization::exact`] a body resting
    /// at depth `d` with no other load is pushed out at exactly `d/dt`. `distance` is `0` for every
    /// penetrating pair, so the same mutation gives zero velocity here.
    #[test]
    fn the_gap_drives_the_push_out_velocity_at_depth_over_dt() {
        let (robot, inertia) = slider();
        let (slab, n) = incline(0.4);
        let mm = mass_matrix(&robot, &inertia, &[0.0; 3]);
        let stab = PgsStabilization::exact();

        let mut rows = Vec::new();
        for d in [1e-4f64, 5e-4, 1e-3, 4e-3] {
            let apex = -n * d;
            let q = [apex.x, apex.y, apex.z];
            let set = compound_pgs_contacts(&spike(apex, n), &SerialLinkParts { robot: &robot, q: &q, frames: &[3] }, &slab, &StaticBody { nv: 3 }, 1e-2, 0.5)
                .expect("nv agrees");
            assert_eq!(set.contacts.len(), 1);
            // at rest, no gravity: the ONLY thing that can move the body is the gap term
            let res = solve_contacts_pgs_with(&mm, &DVector::zeros(3), &set.contacts, DT, 40, None, stab);
            let v = Vector3::new(res.v_next[0], res.v_next[1], res.v_next[2]);
            rows.push((d, (v.norm() * 1e9).round() / 1e9, d / DT));
            assert!((v.norm() - d / DT).abs() < 1e-9, "depth {d}: push-out must be d/dt = {:.6}, got {:.6}", d / DT, v.norm());
            assert!((v.normalize() - n).norm() < 1e-9, "the push-out must be along the surface normal {n:?}, got {:?}", v.normalize());
        }
        eprintln!("  exact stabilisation, at rest   (depth, |v|, d/dt): {rows:?}");
    }

    /// **A body resting exactly ON a surface must never reach the solver as a deep penetration.**
    ///
    /// ⛔ This is the test that found the second defect of the session. Before it, a foot whose tip sat
    /// exactly on a slab's face was handed to the solver as **1.775 m of penetration** on a diagonal
    /// normal — EPA's seed-orientation degeneracy, [`epa`](crate::epa). Under
    /// [`PgsStabilization::exact`] that is a 1775 m/s launch; under the default it is a
    /// `max_correction` clamp firing every step on a body that is simply at rest.
    ///
    /// Exact touching is where both the GJK and the EPA branch degenerate, and what comes back depends
    /// on the orientation of the geometry about the contact normal: sometimes a zero normal, which is
    /// dropped and counted, and sometimes a real normal at zero gap, which is a perfectly good
    /// constraint. So the assertion is the orientation-independent one — the pair is accounted for
    /// exactly once, and any constraint it does produce has a gap of zero.
    #[test]
    fn a_body_resting_exactly_on_a_surface_is_never_a_deep_penetration() {
        let (robot, _) = slider();
        let (slab, n) = incline(0.0);
        assert!((n - Vector3::z()).norm() < 1e-15, "the flat fixture's normal is +z");

        let mut rows = Vec::new();
        for phase in [0.0f64, 0.7, core::f64::consts::FRAC_PI_2, 1.4, 2.9, 4.1] {
            // the apex exactly on the plane z = 0, rotated about the contact normal
            let foot = CompoundHull::from_convex_parts(vec![ConvexPoints {
                pts: core::iter::once(Vector3::zeros())
                    .chain((0..3).map(|k| {
                        let a = phase + k as f64 * core::f64::consts::TAU / 3.0;
                        Vector3::new(0.03 * a.cos(), 0.03 * a.sin(), 0.05)
                    }))
                    .collect(),
            }])
            .expect("a tetrahedron is a part");
            let set = compound_pgs_contacts(&foot, &SerialLinkParts { robot: &robot, q: &[0.0; 3], frames: &[3] }, &slab, &StaticBody { nv: 3 }, 1e-4, 0.5)
                .expect("nv agrees");

            rows.push((phase, set.contacts.len(), set.skipped, set.contacts.first().map(|c| c.phi)));
            assert_eq!(set.contacts.len() + set.skipped, 1, "phase {phase}: the one pair must be accounted for exactly once");
            assert_eq!(set.geometry.len(), set.contacts.len(), "phase {phase}: geometry and contacts must stay in step");
            for c in &set.contacts {
                assert!(
                    c.phi.abs() < 1e-12,
                    "phase {phase}: a body resting ON the surface has a zero gap, got {} (the shipped bug reported −1.775)",
                    c.phi
                );
                // ⛔ and a constraint that IS produced must carry a usable direction. The slider's
                // relative Jacobian is the identity, so row 2 is the normal itself; a zero row reads to
                // the solver as a contact no motion can ever violate, which is why the zero-normal
                // pairs are dropped rather than passed on.
                let row2 = Vector3::new(c.j[(2, 0)], c.j[(2, 1)], c.j[(2, 2)]);
                assert!(
                    (row2.norm() - 1.0).abs() < 1e-12,
                    "phase {phase}: a produced constraint must have a unit normal row, got {:.3e}",
                    row2.norm()
                );
            }
        }
        eprintln!("  apex exactly on the surface  (phase, contacts, skipped, phi): {rows:?}");
        // both outcomes must actually occur in this sweep, or the test is only exercising one branch
        assert!(rows.iter().any(|r| r.1 == 1), "some orientation must yield a usable constraint");
        assert!(rows.iter().any(|r| r.2 == 1), "some orientation must yield a counted skip");
    }

    /// **The contact POINT is load-bearing as soon as the body can rotate**, and the slider cannot see
    /// that: a prismatic chain's point Jacobian is the same identity everywhere, so the point could be
    /// anywhere at all and every test above would still pass.
    ///
    /// A revolute lever pins it analytically. One joint about `ŷ` at the origin, a foot at
    /// `x = 0.4`, contact normal `+ẑ`: the joint column is `ŷ × p`, so row 2 is `ẑ·(ŷ × p) = −p.x`.
    /// The row therefore reports the lever arm directly, and a wrong contact point shows up as a wrong
    /// arm. This is the test that caught the support-point average.
    #[test]
    fn a_revolute_lever_reads_its_arm_off_the_contact_point() {
        let arm = 0.4;
        let pen = 1e-3;
        let robot = Robot { joints: vec![Joint::revolute(Iso::identity(), Vector3::y())], ee_offset: Iso::identity() };
        let (slab, n) = incline(0.0);
        let apex = Vector3::new(arm, 0.0, -pen);
        let set = compound_pgs_contacts(&spike(apex, n), &SerialLinkParts { robot: &robot, q: &[0.0], frames: &[1] }, &slab, &StaticBody { nv: 1 }, 1e-4, 0.5)
            .expect("nv agrees");
        assert_eq!(set.contacts.len(), 1);
        let row2 = set.contacts[0].j[(2, 0)];
        // the point is A's deepest vertex brought back half the depth along the normal, so its x is the
        // apex's x exactly; the analytic arm is therefore −0.4 and not −0.4 ± anything
        eprintln!("  revolute lever: arm {arm}, row 2 = {row2:.12} (analytic −p.x = {:.12})", -arm);
        assert!((row2 + arm).abs() < 1e-12, "row 2 must be −p.x = {:.12}, got {row2:.12}", -arm);
        // and it must scale with the arm, which is what rules out a constant of the right magnitude
        let far = compound_pgs_contacts(&spike(Vector3::new(1.3, 0.0, -pen), n), &SerialLinkParts { robot: &robot, q: &[0.0], frames: &[1] }, &slab, &StaticBody { nv: 1 }, 1e-4, 0.5)
            .expect("nv agrees");
        assert!((far.contacts[0].j[(2, 0)] + 1.3).abs() < 1e-12, "a longer arm must read longer, got {}", far.contacts[0].j[(2, 0)]);

        // ---- and the point's NORMAL coordinate, which row 2 alone cannot see ----
        // For `n_sep = ẑ` the tangent pair is `(ŷ, −x̂)` and `ŷ × p = (p.z, 0, −p.x)`, so row 1 is
        // `−x̂ᵀ(ŷ × p) = −p.z`: the row reports the contact point's height directly. That is what makes
        // the half-depth step-back load-bearing — without it the point would sit ON body A's surface at
        // `−pen` instead of midway between the two surfaces at `−pen/2`.
        let row1 = set.contacts[0].j[(1, 0)];
        eprintln!("  row 1 = {row1:.9} (analytic −p.z = +pen/2 = {:.9}; on A's surface it would be {:.9})", pen / 2.0, pen);
        assert!((row1 - pen / 2.0).abs() < 1e-12, "row 1 must be −p.z = +pen/2 = {:.9}, got {row1:.9}", pen / 2.0);

        // the SEPARATED branch of `contact_point` is a different expression and needs its own case: the
        // midpoint of GJK's witness pair, so a foot a gap above the plane puts the point at gap/2.
        let gap = 2e-3;
        let above = compound_pgs_contacts(&spike(Vector3::new(arm, 0.0, gap), n), &SerialLinkParts { robot: &robot, q: &[0.0], frames: &[1] }, &slab, &StaticBody { nv: 1 }, 1e-2, 0.5)
            .expect("nv agrees");
        assert_eq!(above.contacts.len(), 1, "a foot {gap} above the plane is inside a 1e-2 margin");
        assert!((above.contacts[0].phi - gap).abs() < 1e-12, "the gap must be +{gap}, got {}", above.contacts[0].phi);
        let row1_above = above.contacts[0].j[(1, 0)];
        eprintln!("  separated: row 1 = {row1_above:.9} (−(witness midpoint z) = −gap/2 = {:.9}; A's own witness would give {:.9})", -gap / 2.0, -gap);
        assert!((row1_above + gap / 2.0).abs() < 1e-12, "row 1 must be −(witness MIDPOINT z) = {:.9}, got {row1_above:.9}", -gap / 2.0);
    }

    /// **Two bodies that both move**, which is the reason this takes a trait per body rather than a
    /// robot and a floor. The constraint is on RELATIVE motion, so the Jacobian is `J_A − J_B`; with a
    /// static environment that subtraction is invisible, because `J_B` is zero.
    ///
    /// A and B here move on disjoint halves of one six-coordinate velocity, so the normal row must come
    /// out as `[+n̂ᵀ | −n̂ᵀ]` and a sign error in the subtraction is visible in the second half alone.
    #[test]
    fn the_constraint_is_on_relative_motion_so_the_two_jacobians_subtract() {
        /// A body whose three coordinates translate it, starting at column `base` of a longer vector.
        struct Sliding {
            nv: usize,
            base: usize,
        }
        impl ContactJacobian for Sliding {
            fn nv(&self) -> usize {
                self.nv
            }
            fn point_jacobian(&self, _part: usize, _p: &Vector3<f64>) -> DMatrix<f64> {
                let mut j = DMatrix::zeros(3, self.nv);
                for r in 0..3 {
                    j[(r, self.base + r)] = 1.0;
                }
                j
            }
        }

        let theta = 0.55f64;
        let (slab, n) = incline(theta);
        let apex = -n * 1e-3;
        let set = compound_pgs_contacts(&spike(apex, n), &Sliding { nv: 6, base: 0 }, &slab, &Sliding { nv: 6, base: 3 }, 1e-4, 0.5)
            .expect("both bodies are nv 6");
        assert_eq!(set.contacts.len(), 1);
        let j = &set.contacts[0].j;
        assert_eq!((j.nrows(), j.ncols()), (3, 6));
        let row2: Vec<f64> = (0..6).map(|c| j[(2, c)]).collect();
        eprintln!("  row 2 = {:?}   (n̂ = {:?})", row2.iter().map(|v| (v * 1e12).round() / 1e12).collect::<Vec<_>>(), n.as_slice());
        for k in 0..3 {
            assert!((row2[k] - n[k]).abs() < 1e-12, "the first half must be +n̂: {row2:?} vs {:?}", n.as_slice());
            assert!((row2[3 + k] + n[k]).abs() < 1e-12, "the second half must be −n̂: {row2:?} vs {:?}", n.as_slice());
        }
        // the whole 3×6 must be an isometry on each half, which is what an added-instead-of-subtracted
        // Jacobian would break without changing the first half at all
        let gram = j * j.transpose();
        assert!((gram.clone() - DMatrix::identity(3, 3) * 2.0).amax() < 1e-12, "each half is orthonormal, so J·Jᵀ = 2I, got {gram}");
    }

    /// The refusals. Each is a caller mistake that would otherwise produce a silently wrong solve.
    #[test]
    fn a_mismatched_generalized_velocity_and_a_bad_friction_coefficient_are_refused() {
        let (robot, _) = slider();
        let (slab, n) = incline(0.3);
        let apex = -n * 1e-3;
        let q = [apex.x, apex.y, apex.z];
        let body = spike(apex, n);
        let good = SerialLinkParts { robot: &robot, q: &q, frames: &[3] };

        let mismatched = compound_pgs_contacts(&body, &good, &slab, &StaticBody { nv: 4 }, 1e-4, 0.5);
        assert!(mismatched.is_err(), "nv 3 against nv 4 must be refused, got {:?}", mismatched.map(|s| s.contacts.len()));
        for bad_mu in [-0.1, f64::NAN, f64::INFINITY] {
            let r = compound_pgs_contacts(&body, &good, &slab, &StaticBody { nv: 3 }, 1e-4, bad_mu);
            assert!(r.is_err(), "μ = {bad_mu} must be refused");
        }
        // a non-finite margin yields no pairs at all rather than an error, matching `compound_contacts`
        let nan_margin = compound_pgs_contacts(&body, &good, &slab, &StaticBody { nv: 3 }, f64::NAN, 0.5).expect("not an error");
        assert!(nan_margin.contacts.is_empty(), "a NaN margin selects nothing");

        // a part index the caller never mapped is treated as immobile, not a panic
        let unmapped = compound_pgs_contacts(&body, &SerialLinkParts { robot: &robot, q: &q, frames: &[] }, &slab, &StaticBody { nv: 3 }, 1e-4, 0.5)
            .expect("still builds");
        assert_eq!(unmapped.contacts.len(), 1, "the pair is still found");
        assert!(unmapped.contacts[0].j.amax() == 0.0, "an unmapped part cannot move, so its Jacobian is zero");
    }

    /// ⭐⭐ **The chain, end to end: mesh → voxels → decomposition → contacts → solver.**
    ///
    /// [`convex_decompose`](crate::convex_decompose) says it closes the chain to this module. That was
    /// an unverified claim until this test, and the way to verify it is not to check that a contact
    /// exists — it is to put a body somewhere the two representations DISAGREE and ask the solver what
    /// happens.
    ///
    /// The place is the mouth of the U-bracket. A body resting on the channel floor is supported from
    /// BELOW by the slab. The bracket's convex hull has no channel: the same body is 0.4 m inside it,
    /// and the nearest escape is sideways through a `y` face, so the hull's answer is a shove in `y`
    /// while the body keeps falling at full gravity.
    ///
    /// ⛔ The hull arm is the control and it is load-bearing. Without it, "the body rests" is also what
    /// you would see from a decomposition that had simply filled the channel with a part — the body
    /// would rest on a lie. The control proves the mouth is somewhere the hull is WRONG.
    #[test]
    fn a_body_rests_on_the_floor_of_a_decomposed_channel_where_the_hull_shoves_it_sideways() {
        use crate::acd::{convex_decompose, AcdOptions};
        use crate::link_geometry::{primitive_mesh, LinkGeometry};
        use crate::voxel::SolidVoxels;

        let mut mesh = crate::mesh3::TriMesh3 { verts: Vec::new(), tris: Vec::new() };
        for (g, at) in [
            (LinkGeometry::Box { size: Vector3::new(1.2, 0.8, 0.4) }, Vector3::zeros()),
            (LinkGeometry::Box { size: Vector3::new(0.2, 0.8, 1.0) }, Vector3::new(-0.5, 0.0, 0.5)),
            (LinkGeometry::Box { size: Vector3::new(0.2, 0.8, 1.0) }, Vector3::new(0.5, 0.0, 0.5)),
        ] {
            let m = primitive_mesh(&g, 24).expect("a primitive meshes");
            let base = mesh.verts.len();
            mesh.verts.extend(m.verts.iter().map(|q| q + at));
            mesh.tris.extend(m.tris.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        }
        let opts = AcdOptions::default();
        let (parts, rep) = convex_decompose(&mesh, &opts).expect("the bracket decomposes");
        let hull = CompoundHull::from_mesh_hull(&mesh).expect("the bracket hulls");

        // ⛔ The floor is read from the voxelisation, not assumed to be the slab's nominal 0.2 m. The
        // parts are hulls over CELL CORNERS, so the channel floor is a cell boundary and the fixture
        // must sit on the one the decomposition actually produced, or the penetration is not `pen`.
        let v = SolidVoxels::from_mesh(&mesh, opts.resolution).expect("voxelises");
        let ci = |a: usize, x: f64| ((x - v.origin[a]) / v.cell).round() as usize;
        let (i, j) = (ci(0, 0.0), ci(1, 0.0));
        let top = (0..v.dims[2]).filter(|&k| v.get(i, j, k)).max().expect("a column under the channel");
        let floor = v.origin.z + (top as f64 + 0.5) * v.cell;

        let pen = 1e-3;
        let apex = Vector3::new(0.0, 0.0, floor - pen);
        let body = spike(apex, Vector3::z());
        let q = [apex.x, apex.y, apex.z];

        let (robot, inertia) = slider();
        let stab = PgsStabilization { slop: 0.0, erp: 0.0, max_correction: 0.0 };
        let mm = mass_matrix(&robot, &inertia, &q);
        let qdd = forward_dynamics(&robot, &inertia, &q, &[0.0; 3], &[0.0; 3], Vector3::new(0.0, 0.0, -G));
        let v_free = DVector::from_iterator(3, qdd.iter().map(|a| a * DT));
        assert!((v_free[2] + G * DT).abs() < 1e-12, "free fall for one step is −g·dt on z");

        let step = |against: &CompoundHull| {
            let set = compound_pgs_contacts(
                &body,
                &SerialLinkParts { robot: &robot, q: &q, frames: &[3] },
                against,
                &StaticBody { nv: 3 },
                1e-4,
                0.6,
            )
            .expect("the bodies agree on nv");
            let res = solve_contacts_pgs_with(&mm, &v_free, &set.contacts, DT, 60, None, stab);
            (set, Vector3::new(res.v_next[0], res.v_next[1], res.v_next[2]))
        };

        let (dset, dv) = step(&parts);
        let (hset, hv) = step(&hull);
        eprintln!(
            "  floor at z = {floor:+.4}, body apex {:+.4}\n  \
             decomposed ({} parts): {} contact(s), deepest gap {:+.5} m, v_next {:?}\n  \
             single hull  (1 part): {} contact(s), deepest gap {:+.5} m, v_next {:?}",
            apex.z, rep.parts, dset.contacts.len(),
            dset.geometry.iter().map(|g| g.gap).fold(f64::INFINITY, f64::min), dv,
            hset.contacts.len(),
            hset.geometry.iter().map(|g| g.gap).fold(f64::INFINITY, f64::min), hv,
        );

        // the control first: the hull has the body deep inside and pushes it out of a SIDE face
        let hg = hset.geometry.iter().map(|g| g.gap).fold(f64::INFINITY, f64::min);
        assert!(hg < -0.1, "the control needs the body deep inside the hull, got a gap of {hg:+.4} m");
        assert!(
            hset.geometry.iter().all(|g| g.normal.z.abs() < 0.5),
            "the control needs the hull's escape to be SIDEWAYS, got a normal with z = {:?}",
            hset.geometry.iter().map(|g| g.normal.z).collect::<Vec<_>>()
        );
        assert!(
            (hv.z + G * DT).abs() < 1e-9,
            "under the hull the body is unsupported and must still fall at −g·dt = {:.6}, got {:.6}",
            -G * DT,
            hv.z
        );

        // and the decomposition: supported from below, at the penetration the fixture was built for
        assert!(!dset.contacts.is_empty(), "the decomposition must produce a contact at the floor");
        let floor_row = dset
            .geometry
            .iter()
            .find(|g| g.normal.z < -0.99)
            .expect("one contact normal must point from the body DOWN into the slab");
        assert!(
            (floor_row.gap + pen).abs() < 1e-6,
            "the fixture penetrates the floor by {pen}, got {:+.6}",
            floor_row.gap
        );
        assert!(
            dv.z.abs() < 1e-9,
            "resting on the slab, the vertical velocity after one step must be 0, got {:.9}",
            dv.z
        );
    }

    /// ⭐⭐ **Is the answer this chain produces PHYSICAL?** Nothing asked until now.
    ///
    /// `a_body_rests_on_the_floor_of_a_decomposed_channel_where_the_hull_shoves_it_sideways` shows the
    /// chain puts the body in the right place. That is a different question from whether the impulses it
    /// got there with obey Signorini, Coulomb and maximum dissipation — see
    /// [`crate::contact_law_residuals`], and the measurement on [`crate::PgsResult::violation`] showing
    /// a solver's own number can be five orders out from the law.
    ///
    /// The fixture is a block wedged in the U-bracket's channel, touching the slab below and a wall
    /// beside it. That is **two simultaneous contacts, on two different parts of the decomposition, with
    /// perpendicular normals** — the case the whole compound path exists for, and one a single hull
    /// cannot produce at all because it has no channel to wedge anything into. A tangential drive along
    /// `y` makes both contacts slide, so maximum dissipation binds rather than being vacuously zero.
    #[test]
    fn the_decomposed_collider_feeds_the_solver_a_physically_lawful_answer() {
        use crate::acd::{convex_decompose, AcdOptions};
        use crate::contact_law_residuals;
        use crate::link_geometry::{primitive_mesh, LinkGeometry};
        use crate::voxel::SolidVoxels;

        let mut mesh = crate::mesh3::TriMesh3 { verts: Vec::new(), tris: Vec::new() };
        for (g, at) in [
            (LinkGeometry::Box { size: Vector3::new(1.2, 0.8, 0.4) }, Vector3::zeros()),
            (LinkGeometry::Box { size: Vector3::new(0.2, 0.8, 1.0) }, Vector3::new(-0.5, 0.0, 0.5)),
            (LinkGeometry::Box { size: Vector3::new(0.2, 0.8, 1.0) }, Vector3::new(0.5, 0.0, 0.5)),
        ] {
            let m = primitive_mesh(&g, 24).expect("a primitive meshes");
            let base = mesh.verts.len();
            mesh.verts.extend(m.verts.iter().map(|q| q + at));
            mesh.tris.extend(m.tris.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        }
        let opts = AcdOptions::default();
        let (bracket, rep) = convex_decompose(&mesh, &opts).expect("decomposes");

        // ⛔ The channel's floor AND its left wall's inner face, both read from the voxelisation rather
        // than assumed to be the nominal 0.2 and −0.4: the parts are hulls over CELL CORNERS, so both
        // surfaces are cell boundaries.
        //
        // ⚠ The first version of this fixture laid a plate across the two WALL TOPS. It produced two
        // contacts and every residual came out exactly `0` — because both normals are `+z` and this body
        // is a three-prismatic slider, so the two Jacobians are identical and the pair is one constraint
        // wearing two hats. A fixture where everything is zero is a statement about the fixture.
        let v = SolidVoxels::from_mesh(&mesh, opts.resolution).expect("voxelises");
        let ci = |a: usize, x: f64| ((x - v.origin[a]) / v.cell).round() as usize;
        let j = ci(1, 0.0);
        let k_mid = ci(2, 0.5); // well above the slab, between the walls
        let floor = {
            let i = ci(0, 0.0);
            let k = (0..v.dims[2]).filter(|&k| v.get(i, j, k)).max().expect("a column under the channel");
            v.origin.z + (k as f64 + 0.5) * v.cell
        };
        let inner_x = {
            let i0 = ci(0, 0.0);
            let i = (0..i0).filter(|&i| v.get(i, j, k_mid)).max().expect("a left wall at mid height");
            v.origin.x + (i as f64 + 0.5) * v.cell
        };

        let pen = 1e-3;
        let (hx, hy, hz) = (0.15, 0.2, 0.15);
        let centre = Vector3::new(inner_x - pen + hx, 0.0, floor - pen + hz);
        let plate = CompoundHull::from_convex_parts(vec![box_pts(centre, Vector3::new(hx, hy, hz))])
            .expect("a block wedged in the channel");

        let (robot, inertia) = slider();
        let q = [centre.x, centre.y, centre.z];
        let mu = 0.6;
        let set = compound_pgs_contacts(
            &plate,
            &SerialLinkParts { robot: &robot, q: &q, frames: &[3] },
            &bracket,
            &StaticBody { nv: 3 },
            1e-4,
            mu,
        )
        .expect("the bodies agree on nv");

        let stab = PgsStabilization { slop: 0.0, erp: 0.0, max_correction: 0.0 };
        let mm = mass_matrix(&robot, &inertia, &q);
        let qdd = forward_dynamics(&robot, &inertia, &q, &[0.0; 3], &[0.0; 3], Vector3::new(0.0, 0.0, -G));
        // a tangential drive along y, so BOTH contacts slide and maximum dissipation actually binds
        let mut v_free = DVector::from_iterator(3, qdd.iter().map(|a| a * DT));
        v_free[1] += 2.0;
        let res = solve_contacts_pgs_with(&mm, &v_free, &set.contacts, DT, 400, None, stab);

        let cv: Vec<Vector3<f64>> = set
            .contacts
            .iter()
            .map(|c| {
                let u = &c.j * &res.v_next;
                Vector3::new(u[0], u[1], u[2])
            })
            .collect();
        let mus = vec![mu; set.contacts.len()];
        let laws = contact_law_residuals(&res.lambda, &cv, &mus, 1e-6);
        let worst = laws.iter().map(crate::ContactLawResidual::worst).fold(0.0f64, f64::max);
        eprintln!(
            "  block wedged in a {}-part decomposition: {} contact(s), {} skipped; solver resid {:.2e}, violation {:.2e}; WORST LAW {worst:.3e}",
            rep.parts, set.contacts.len(), set.skipped, res.residual, res.violation
        );
        for l in &laws {
            eprintln!("    contact {}: sig {:.2e} coul {:.2e} maxdiss {:.2e} (sliding {})", l.contact, l.signorini, l.coulomb, l.max_dissipation, l.sliding);
        }

        // ⛔ the fixture must actually exercise the compound path, or the law check below is about one
        // contact and says nothing the single-hull case would not
        assert!(
            set.contacts.len() >= 2,
            "a block wedged against the floor and a wall must produce a contact for each, got {}",
            set.contacts.len()
        );
        // ⛔ and they must genuinely compete: two parallel normals on a translational body are one
        // constraint, which is how the first fixture came out all zeros
        let worst_dot = set
            .geometry
            .iter()
            .flat_map(|a| set.geometry.iter().map(move |b| (a, b)))
            .filter(|(a, b)| !core::ptr::eq(*a, *b))
            .map(|(a, b)| a.normal.dot(&b.normal).abs())
            .fold(1.0f64, f64::min);
        assert!(
            worst_dot < 0.5,
            "the contact normals must not be parallel or the pair is one constraint: worst |dot| {worst_dot:.3}"
        );
        assert_eq!(set.skipped, 0, "every pair inside the margin must have produced a constraint");
        assert!(
            set.geometry.iter().map(|g| g.part_b).collect::<std::collections::HashSet<_>>().len() >= 2,
            "the contacts must land on DIFFERENT parts of the decomposition: {:?}",
            set.geometry.iter().map(|g| g.part_b).collect::<Vec<_>>()
        );
        assert!(res.v_next[2].abs() < 1e-9, "the block is supported and must not fall: {}", res.v_next[2]);
        assert!(laws.iter().any(|l| l.sliding), "the drive must make at least one contact slide, or maximum dissipation is vacuous here");
        assert!(worst < 1e-6, "the chain produced an unphysical answer: worst law residual {worst:e}");
    }
}
