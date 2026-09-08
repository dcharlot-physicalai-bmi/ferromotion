//! **EPA — Expanding Polytope Algorithm** (penetration depth for overlapping convex shapes). GJK
//! ([`crate::gjk`]) reports *that* two convex shapes overlap; EPA reports *how much* — the minimum
//! translation `depth·n̂` that just separates them and the contact normal `n̂`. It seeds from a
//! **GJK-boolean** tetrahedron that strictly encloses the origin of the Minkowski difference `A ⊖ B`, then
//! iteratively expands the face closest to the origin (querying the support function outward) until it
//! reaches the boundary — that closest boundary point *is* the penetration vector. This completes the
//! crate's collision narrow-phase (GJK distance + witness points already existed; penetration was the
//! documented gap).
//!
//! **Scope (honest).** The seed comes from the proper GJK **`do_simplex`** evolution (Muratori). This is
//! robust for **polytopes** (boxes, convex hulls, meshes) — EPA's actual purpose. For *perfectly smooth*
//! strictly-convex shapes (a `Ball`/`Capsule`) `do_simplex` can terminate with the origin exactly on a
//! tetrahedron face, which is a degenerate EPA seed; such primitives have a **trivial analytic penetration**
//! anyway, provided here as [`sphere_penetration`]. So: use `epa` for polytopes, the analytic form for round
//! primitives. Verified: overlapping boxes and overlapping convex hulls recover the minimum-translation
//! depth and axis, disjoint shapes return `None`, and the analytic sphere penetration matches its closed
//! form. Pure `nalgebra` → WASM-clean.

use crate::gjk::Support;
use nalgebra::{Matrix3, Vector3};

/// Penetration data for two overlapping convex shapes.
#[derive(Clone, Copy, Debug)]
pub struct Penetration {
    /// Minimum separation distance (penetration depth).
    pub depth: f64,
    /// Unit contact normal, pointing **from `B` toward `A`**: the direction `A` must move to get
    /// clear. The contract is one statement, and it is what the tests check functionally rather than
    /// by asserting a sign: **translating `A` by `+depth·normal` brings the pair to exactly
    /// touching.**
    ///
    /// Both routes here honour it. [`epa`] has to negate the face normal it computes to do so, for
    /// the reason given at that line; the disagreement between the two routes was live until
    /// 2026-09-08.
    pub normal: Vector3<f64>,
}

// A Minkowski-difference support point.
#[derive(Clone, Copy)]
struct Sp {
    v: Vector3<f64>,
}

fn support<A: Support, B: Support>(a: &A, b: &B, dir: &Vector3<f64>) -> Sp {
    Sp { v: a.support(dir) - b.support(&(-dir)) }
}

fn triple(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> Vector3<f64> {
    a.cross(b).cross(c)
}

enum Step {
    Done(bool), // true ⇒ origin enclosed (tetrahedron)
    Continue,
}

fn line(s: &mut Vec<Sp>, dir: &mut Vector3<f64>) -> Step {
    let (a, b) = (s[1], s[0]);
    let ab = b.v - a.v;
    let ao = -a.v;
    if ab.dot(&ao) > 0.0 {
        let mut d = triple(&ab, &ao, &ab);
        if d.norm() < 1e-12 {
            // origin on the line: any perpendicular
            let axis = if ab.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
            d = ab.cross(&axis);
        }
        *dir = d;
    } else {
        *s = vec![a];
        *dir = ao;
    }
    Step::Done(false)
}

fn triangle(s: &mut Vec<Sp>, dir: &mut Vector3<f64>) -> Step {
    let (a, b, c) = (s[2], s[1], s[0]);
    let (ab, ac, ao) = (b.v - a.v, c.v - a.v, -a.v);
    let abc = ab.cross(&ac);
    if abc.cross(&ac).dot(&ao) > 0.0 {
        if ac.dot(&ao) > 0.0 {
            *s = vec![c, a];
            let mut d = triple(&ac, &ao, &ac);
            if d.norm() < 1e-12 {
                let axis = if ac.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
                d = ac.cross(&axis);
            }
            *dir = d;
            return Step::Done(false);
        }
        *s = vec![b, a];
        return Step::Continue;
    }
    if ab.cross(&abc).dot(&ao) > 0.0 {
        *s = vec![b, a];
        return Step::Continue;
    }
    if abc.dot(&ao) > 0.0 {
        *dir = abc;
    } else {
        *s = vec![b, c, a];
        *dir = -abc;
    }
    Step::Done(false)
}

fn tetra(s: &mut Vec<Sp>, dir: &mut Vector3<f64>) -> Step {
    let (a, b, c, d) = (s[3], s[2], s[1], s[0]);
    let (ab, ac, ad, ao) = (b.v - a.v, c.v - a.v, d.v - a.v, -a.v);
    if ab.cross(&ac).dot(&ao) > 0.0 {
        *s = vec![c, b, a];
        return Step::Continue;
    }
    if ac.cross(&ad).dot(&ao) > 0.0 {
        *s = vec![d, c, a];
        return Step::Continue;
    }
    if ad.cross(&ab).dot(&ao) > 0.0 {
        *s = vec![b, d, a];
        return Step::Continue;
    }
    let _ = dir;
    Step::Done(true) // origin enclosed
}

// GJK-boolean evolution to an origin-enclosing tetrahedron, or None if disjoint.
fn enclosing_tetra<A: Support, B: Support>(a: &A, b: &B) -> Option<Vec<Sp>> {
    let mut dir = Vector3::new(1.0, 0.0, 0.0);
    let mut s = vec![support(a, b, &dir)];
    dir = -s[0].v;
    for _ in 0..64 {
        if dir.norm() < 1e-12 {
            dir = Vector3::new(0.0, 1.0, 0.0);
        }
        let p = support(a, b, &dir);
        if p.v.dot(&dir) < 0.0 {
            return None; // disjoint
        }
        s.push(p);
        let done = loop {
            let step = match s.len() {
                2 => line(&mut s, &mut dir),
                3 => triangle(&mut s, &mut dir),
                4 => tetra(&mut s, &mut dir),
                _ => Step::Done(false),
            };
            match step {
                Step::Done(enclosed) => break enclosed,
                Step::Continue => {}
            }
        };
        if done {
            return Some(s);
        }
    }
    None
}

struct Face {
    idx: [usize; 3],
    normal: Vector3<f64>,
    dist: f64,
}

fn make_face(verts: &[Sp], i: usize, j: usize, k: usize) -> Option<Face> {
    let (a, b, c) = (verts[i].v, verts[j].v, verts[k].v);
    let mut n = (b - a).cross(&(c - a));
    let nn = n.norm();
    if nn < 1e-14 {
        return None;
    }
    n /= nn;
    let mut dist = n.dot(&a);
    if dist < 0.0 {
        n = -n; // outward (origin is inside)
        dist = -dist;
    }
    Some(Face { idx: [i, j, k], normal: n, dist })
}

/// **An upper bound on the penetration depth that the geometry itself cannot exceed.**
///
/// The true depth is `min over all unit directions d of support_{A⊖B}(d)·d` — that is what "distance
/// from the origin to the boundary" means. So the minimum over ANY sample of directions is an upper
/// bound on it, and a claimed depth larger than that bound is definitely wrong. The check is one-sided
/// by construction: it can reject an over-estimate and can never reject a true answer, whatever
/// directions are sampled. Six axes and the reverse of the candidate normal cost seven support calls.
///
/// # ⛔ Why this is needed: the origin sitting ON the boundary was assumed impossible
///
/// [`make_face`] orients every face outward by flipping any whose signed distance comes out negative,
/// on the stated assumption that the origin is inside. Two shapes exactly TOUCHING put the origin on
/// the boundary, that assumption is false, a face through the origin gets flipped, and the expansion
/// then settles on a supporting face somewhere else entirely and reports ITS distance as a depth.
///
/// Measured: a tetrahedron whose apex sat exactly on a slab's top face was reported as penetrating by
/// **1.775 m** along a diagonal normal. The same tetrahedron rotated 90° about the contact normal gave
/// the correct `0`. An answer that depends on vertex ordering is a degeneracy, not a tolerance.
///
/// # Two guards that did not work, so the next attempt does not repeat them
///
/// | attempt | why it failed |
/// |---|---|
/// | "the origin must be strictly inside the seed simplex" | GJK's termination simplex can legitimately have the origin ON one of its faces while the shapes genuinely overlap — two axis-aligned cubes do exactly that — so it refused a real 0.4 m overlap. |
/// | "for each seed face through the origin, ask whether the body reaches past it" | incomplete, and provably so: the supporting plane at the origin need not be among the seed's four faces. With the slab as an analytic [`Cuboid`](crate::Cuboid) the seed happened to contain it and the guard fired; with the same slab as a [`ConvexPoints`](crate::ConvexPoints) hull — which is what the compound collider uses — the support tie-break moved the seed, its faces came out tilted by 0.012 rad from the true normal, the body reached 4.9e-2 past every one of them, and the 1.775 sailed through. |
///
/// The lesson is that the seed cannot answer a question about `A ⊖ B`; only the support function can.
fn depth_upper_bound<A: Support, B: Support>(a: &A, b: &B, nf: &Vector3<f64>) -> f64 {
    let mut bound = f64::INFINITY;
    for d in [Vector3::x(), -Vector3::x(), Vector3::y(), -Vector3::y(), Vector3::z(), -Vector3::z(), -nf] {
        let e = support(a, b, &d).v.dot(&d);
        if e.is_finite() {
            bound = bound.min(e);
        }
    }
    bound
}

/// **EPA** penetration depth and contact normal for two convex shapes, or `None` if they are disjoint.
pub fn epa<A: Support, B: Support>(a: &A, b: &B) -> Option<Penetration> {
    let mut verts = enclosing_tetra(a, b)?;
    // degenerate (flat) seed ⇒ touching, treat as no penetration
    let vol = {
        let o = Matrix3::from_columns(&[verts[1].v - verts[0].v, verts[2].v - verts[0].v, verts[3].v - verts[0].v]);
        o.determinant().abs()
    };
    if vol < 1e-12 {
        return None;
    }
    let mut faces: Vec<Face> = Vec::new();
    for &(i, j, k) in &[(0usize, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)] {
        faces.push(make_face(&verts, i, j, k)?);
    }
    for _ in 0..96 {
        // A total order suffices here: distances are non-negative, so a NaN compares GREATEST and is
        // never selected as the closest face.
        let ci = (0..faces.len()).min_by(|&x, &y| faces[x].dist.total_cmp(&faces[y].dist))?;
        let normal = faces[ci].normal;
        let dist = faces[ci].dist;
        let p = support(a, b, &normal);
        let pd = p.v.dot(&normal);
        if pd - dist < 1e-7 {
            // ⛔ VERIFY before returning: see `depth_upper_bound`. The expansion can settle on a
            // supporting face that is not the closest one when the origin lies on the boundary, and
            // there is no way to tell from the polytope alone.
            let scale = verts.iter().map(|s| s.v.norm()).fold(0.0f64, f64::max).max(1.0);
            if dist > depth_upper_bound(a, b, &normal) + 8.0 * f64::EPSILON * scale {
                return None;
            }
            // ⛔ **NEGATED, and that is the whole point of this line.** `normal` here is the OUTWARD
            // face normal of the closest face of the Minkowski difference `A ⊖ B`, so the closest
            // boundary point to the origin is `+dist·normal` and shifting the set by `−dist·normal`
            // is what puts the origin on the boundary — i.e. translating `A` by `−dist·normal` is
            // what separates the shapes. That is the OPPOSITE of what [`Penetration`] documents and
            // of what [`sphere_penetration`] returns, and the disagreement shipped: measured on two
            // unit cubes, `A` at the origin and `B` at `+0.75x`, this function returned `(+1,0,0)`
            // while the analytic ball path returned the direction away from `B`. Translating `A` by
            // `+depth·normal` DOUBLED the overlap, 0.25 → 0.50.
            //
            // Neither of this module's own box tests could see it: both assert
            // `normal.x.abs() > 0.98`, which pins the axis and discards exactly the sign that was
            // wrong. `the_separating_translation_is_plus_depth_times_normal_on_both_paths` replaces
            // that with the functional statement, on both paths.
            return Some(Penetration { depth: dist, normal: -normal });
        }
        // expand: drop faces visible from p, stitch the horizon to the new vertex
        let pi = verts.len();
        verts.push(p);
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut kept: Vec<Face> = Vec::new();
        for f in faces.drain(..) {
            if f.normal.dot(&p.v) - f.dist > 1e-9 {
                for &(u, v) in &[(f.idx[0], f.idx[1]), (f.idx[1], f.idx[2]), (f.idx[2], f.idx[0])] {
                    if let Some(pos) = horizon.iter().position(|&(a2, b2)| a2 == v && b2 == u) {
                        horizon.swap_remove(pos);
                    } else {
                        horizon.push((u, v));
                    }
                }
            } else {
                kept.push(f);
            }
        }
        faces = kept;
        for (u, v) in horizon {
            if let Some(nf) = make_face(&verts, u, v, pi) {
                faces.push(nf);
            }
        }
        if faces.is_empty() {
            break;
        }
    }
    None
}

/// Analytic penetration of two [`crate::gjk::Ball`]s (the closed form EPA would recover; use this for round
/// primitives, where the iterative seed degenerates). `None` if disjoint.
pub fn sphere_penetration(a: &crate::gjk::Ball, b: &crate::gjk::Ball) -> Option<Penetration> {
    let d = b.center - a.center;
    let dist = d.norm();
    let depth = a.radius + b.radius - dist;
    if depth <= 0.0 {
        return None;
    }
    let normal = if dist > 1e-12 { -d / dist } else { Vector3::x() };
    Some(Penetration { depth, normal })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gjk::{Ball, ConvexPoints, Cuboid};

    #[test]
    fn epa_box_penetration_matches_the_overlap_axis() {
        // THE ORACLE. Two unit boxes overlapping 0.5 along x (nudged in y,z to break symmetry): depth 0.5,
        // min-translation axis x.
        let a = Cuboid { center: Vector3::zeros(), half: Vector3::new(1.0, 1.0, 1.0), rot: Matrix3::identity() };
        let b = Cuboid { center: Vector3::new(1.5, 0.02, 0.03), half: Vector3::new(1.0, 1.0, 1.0), rot: Matrix3::identity() };
        let pen = epa(&a, &b).expect("overlapping boxes penetrate");
        assert!((pen.depth - 0.5).abs() < 1e-2, "box depth {} vs 0.5", pen.depth);
        assert!(pen.normal.x.abs() > 0.98, "min-translation axis is x: {:?}", pen.normal);
    }

    #[test]
    fn epa_handles_overlapping_convex_polytopes() {
        // THE HEADLINE (EPA's real domain). Two unit cubes as generic convex point sets, overlapping 0.4
        // along x. EPA recovers depth 0.4 and the x normal.
        let cube = |cx: f64| ConvexPoints {
            pts: (0..8).map(|i| Vector3::new(cx + if i & 1 == 0 { -0.5 } else { 0.5 }, if i & 2 == 0 { -0.5 } else { 0.5 } + 0.01, if i & 4 == 0 { -0.5 } else { 0.5 } + 0.02)).collect(),
        };
        let pen = epa(&cube(0.0), &cube(0.6)).expect("overlapping polytopes penetrate");
        assert!((pen.depth - 0.4).abs() < 1e-2, "polytope depth {} vs 0.4", pen.depth);
        assert!(pen.normal.x.abs() > 0.98, "min-translation axis is x: {:?}", pen.normal);
    }

    /// **The contract [`Penetration`] states, checked as a translation rather than as a sign, on both
    /// routes.** `depth` and `normal` exist to tell a caller which way to move `A`; a test that
    /// asserts `normal.x.abs() > 0.98` has checked the axis and thrown that answer away, which is how
    /// the polytope route ran with the normal reversed while two tests watched it.
    ///
    /// Measured before the fix, `A` a unit cube at the origin and `B` one at `+0.75x`: the polytope
    /// route returned `(+1,0,0)`, and translating `A` by `+depth·normal` took the overlap from 0.25 to
    /// 0.50 while the analytic ball route separated correctly. Same struct, opposite meanings.
    #[test]
    fn the_separating_translation_is_plus_depth_times_normal_on_both_paths() {
        // ---- the polytope route ----
        let cube = |c: Vector3<f64>| ConvexPoints {
            pts: (0..8)
                .map(|i| c + Vector3::new(if i & 1 == 0 { -0.5 } else { 0.5 }, if i & 2 == 0 { -0.5 } else { 0.5 }, if i & 4 == 0 { -0.5 } else { 0.5 }))
                .collect(),
        };
        let shift = |p: &ConvexPoints, t: Vector3<f64>| ConvexPoints { pts: p.pts.iter().map(|q| q + t).collect() };
        // Off-axis on purpose: the deepest axis is still x, but a fixture centred on x would let a
        // normal that is right only up to a permutation of the axes pass.
        let (a, b) = (cube(Vector3::zeros()), cube(Vector3::new(0.75, 0.13, -0.09)));
        let pen = epa(&a, &b).expect("overlapping cubes penetrate");
        let depth_after = |t: Vector3<f64>| epa(&shift(&a, t), &b).map(|q| q.depth).unwrap_or(0.0);
        let (sep, into) = (depth_after(pen.depth * pen.normal), depth_after(-pen.depth * pen.normal));
        eprintln!(
            "  polytope: depth {:.4} normal {:?} -> after +depth·n depth {:.3e}, after −depth·n depth {:.4}",
            pen.depth, pen.normal.as_slice(), sep, into
        );
        assert!((pen.normal.norm() - 1.0).abs() < 1e-9, "the normal must be a unit vector, got {}", pen.normal.norm());
        assert!(sep < 1e-6, "+depth·normal must bring the pair to touching, got depth {sep:.3e}");
        // the opposite translation must go DEEPER, which is what pins the direction rather than the
        // axis. Before the fix this was the branch that measured ~0.
        assert!(into > pen.depth * 1.5, "−depth·normal must deepen the overlap past {:.4}, got {into:.4}", pen.depth);

        // ---- the analytic ball route, same statement ----
        let (ra, rb) = (1.0, 1.0);
        let cb = Vector3::new(1.2, 0.4, 0.3);
        let ball = |c: Vector3<f64>, r: f64| Ball { center: c, radius: r };
        let q = sphere_penetration(&ball(Vector3::zeros(), ra), &ball(cb, rb)).expect("overlapping balls penetrate");
        let gap_after = |t: Vector3<f64>| (cb - t).norm() - (ra + rb);
        let (bsep, binto) = (gap_after(q.depth * q.normal), gap_after(-q.depth * q.normal));
        eprintln!("  ball:     depth {:.4} normal {:?} -> after +depth·n gap {:+.3e}, after −depth·n gap {:+.4}", q.depth, q.normal.as_slice(), bsep, binto);
        assert!(bsep.abs() < 1e-12, "+depth·normal must bring the balls to touching, got gap {bsep:+.3e}");
        assert!(binto < -q.depth * 1.5, "−depth·normal must deepen the overlap, got gap {binto:+.4}");

        // ---- and the two routes must now AGREE on direction for the same body ordering ----
        // A ball pair and a cube pair, both A at the origin with B along +x, must give normals on the
        // same side. This is the cross-route assertion that did not exist and would have failed.
        let axis_poly = epa(&cube(Vector3::zeros()), &cube(Vector3::new(0.75, 0.0, 0.0))).expect("cubes").normal;
        let axis_ball = sphere_penetration(&ball(Vector3::zeros(), 1.0), &ball(Vector3::new(1.5, 0.0, 0.0), 1.0)).expect("balls").normal;
        eprintln!("  cross-route, B along +x: polytope normal {:?}, ball normal {:?}", axis_poly.as_slice(), axis_ball.as_slice());
        assert!(axis_poly.dot(&axis_ball) > 0.999, "the two routes must point the same way: {:?} vs {:?}", axis_poly.as_slice(), axis_ball.as_slice());
        assert!(axis_poly.x < -0.999, "B is at +x, so the direction A must move to get clear is −x, got {:?}", axis_poly.as_slice());
    }

    /// **Exactly touching is not PENETRATION — whatever the vertex ordering, and on both support paths.**
    ///
    /// The regression for [`depth_upper_bound`]. A tetrahedron whose apex sat exactly on a slab's top
    /// face was reported as penetrating by **1.775 m** along a diagonal normal.
    ///
    /// ⛔ The slab is checked BOTH as an analytic [`Cuboid`] and as a [`ConvexPoints`] hull of the same
    /// eight corners, because the two differ in how a flat face's support ties are broken, that moves
    /// the seed simplex, and it decided whether the bug appeared. A guard built against the analytic box
    /// alone was measured to pass every test here while the point-set path — the one the compound
    /// collider actually uses — still returned the 1.775.
    ///
    /// ⛔ The property asserted is **not** `is_none()`. Depending on orientation the expansion either
    /// trips the bound and refuses, or converges to a depth of exactly zero with the correct normal.
    /// Both are honest descriptions of a touching pair; a positive depth is not.
    #[test]
    fn an_exactly_touching_pair_never_reports_a_penetration() {
        let analytic = Cuboid { center: Vector3::new(0.0, 0.0, -0.5), half: Vector3::new(2.0, 2.0, 0.5), rot: Matrix3::identity() };
        let hull = ConvexPoints {
            pts: (0..8)
                .map(|i| Vector3::new(if i & 1 == 0 { -2.0 } else { 2.0 }, if i & 2 == 0 { -2.0 } else { 2.0 }, if i & 4 == 0 { -1.0 } else { 0.0 }))
                .collect(),
        };
        let spike = |apex: Vector3<f64>, phase: f64| ConvexPoints {
            pts: core::iter::once(apex)
                .chain((0..3).map(|k| {
                    let a = phase + k as f64 * core::f64::consts::TAU / 3.0;
                    apex + Vector3::new(0.03 * a.cos(), 0.03 * a.sin(), 0.05)
                }))
                .collect(),
        };
        let phases = [0.0f64, 0.7, core::f64::consts::FRAC_PI_2, 1.4, 2.9, 4.1];

        let mut lines = Vec::new();
        for phase in phases {
            let tip = spike(Vector3::zeros(), phase);
            let (ca, ch) = (epa(&tip, &analytic), epa(&tip, &hull));
            for (which, got) in [("analytic Cuboid", ca), ("ConvexPoints hull", ch)] {
                if let Some(p) = got {
                    assert!(
                        p.depth <= 1e-12,
                        "phase {phase}, B as {which}: an apex exactly on the surface is touching, so the depth must be zero or absent, got {} (the shipped bug reported 1.775)",
                        p.depth
                    );
                }
            }
            lines.push((
                (phase * 100.0).round() / 100.0,
                ca.map(|p| p.depth == 0.0),
                ch.map(|p| p.depth == 0.0),
            ));
        }
        eprintln!("  touching, (phase, analytic, hull) as Some(depth==0)/None: {lines:?}");
        // both outcomes must occur, or the sweep is exercising one branch of the guard only
        assert!(lines.iter().any(|l| l.2.is_none()), "some orientation must trip the bound outright");
        assert!(lines.iter().any(|l| l.2 == Some(true)), "some orientation must converge to an exact zero");

        // ---- and no GENUINE penetration may be refused, on either path ----
        // The bound is one-sided by construction, so the floor is set only by the arithmetic: measured,
        // both paths report down to 1e-17 and the point-set path refuses 1e-18, which is below the
        // resolution of the coordinates it is differencing. Stated, not assumed.
        let mut worst = 0.0f64;
        for e in 3..18 {
            let d = 10f64.powi(-e);
            let tip = spike(Vector3::new(0.0, 0.0, -d), 0.0);
            for (which, got) in [("analytic Cuboid", epa(&tip, &analytic)), ("ConvexPoints hull", epa(&tip, &hull))] {
                let p = got.unwrap_or_else(|| panic!("a penetration of {d:.0e} is real and must be reported, B as {which}"));
                worst = worst.max((p.depth - d).abs());
                assert!(p.normal.z > 0.999, "the apex is below the top face, so A must move +z to clear: {:?}", p.normal.as_slice());
            }
        }
        eprintln!("  genuine penetration reported on both paths from 1e-3 down to 1e-17 (worst depth error {worst:.1e})");
        assert!(worst < 1e-18, "the reported depth must be the real one, worst error {worst:.1e}");
    }

    #[test]
    fn epa_returns_none_for_disjoint_shapes() {
        let a = Cuboid { center: Vector3::zeros(), half: Vector3::new(1.0, 1.0, 1.0), rot: Matrix3::identity() };
        let b = Cuboid { center: Vector3::new(5.0, 0.0, 0.0), half: Vector3::new(1.0, 1.0, 1.0), rot: Matrix3::identity() };
        assert!(epa(&a, &b).is_none(), "disjoint shapes have no penetration");
    }

    #[test]
    fn analytic_sphere_penetration_matches_the_closed_form() {
        // Smooth primitives use the analytic form (EPA's iterative seed degenerates on perfect spheres).
        let a = Ball { center: Vector3::zeros(), radius: 1.0 };
        let cb = Vector3::new(1.2, 0.4, 0.3);
        let b = Ball { center: cb, radius: 1.0 };
        let pen = sphere_penetration(&a, &b).expect("overlapping balls penetrate");
        assert!((pen.depth - (2.0 - cb.norm())).abs() < 1e-9, "sphere depth");
        assert!(pen.normal.dot(&(-cb.normalize())) > 0.999, "normal along −c_b");
        assert!(sphere_penetration(&Ball { center: Vector3::new(5.0, 0.0, 0.0), radius: 1.0 }, &a).is_none(), "disjoint ⇒ None");
    }
}
