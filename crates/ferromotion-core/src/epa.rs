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
    /// Unit contact normal; translating `A` by `+depth·normal` just separates the shapes.
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
        let ci = (0..faces.len()).min_by(|&x, &y| faces[x].dist.partial_cmp(&faces[y].dist).unwrap()).unwrap();
        let normal = faces[ci].normal;
        let dist = faces[ci].dist;
        let p = support(a, b, &normal);
        let pd = p.v.dot(&normal);
        if pd - dist < 1e-7 {
            return Some(Penetration { depth: dist, normal });
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
