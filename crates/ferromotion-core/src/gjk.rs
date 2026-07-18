//! **GJK narrowphase + continuous collision** (Gilbert–Johnson–Keerthi 1988; conservative-advancement
//! CCD, Mirtich 1996) — the convex-geometry bedrock every physics engine and planner sits on. A convex
//! shape is defined solely by its **support function** `s(d) = argmax_{x∈shape} d·x`; GJK then walks a
//! simplex over the Minkowski difference `A ⊖ B` to find the **distance and witness points** between two
//! convex shapes (and a cheap boolean **intersection** test), and conservative advancement bounds the
//! **time-of-impact** of two *moving* shapes so fast motions can't tunnel between discrete checks.
//!
//! This complements [`crate::dcol`] (a heavier conic *scaling* proximity, which also covers the
//! penetration regime): GJK is the cheap iterative primitive with exact witness points. Verified against
//! closed-form sphere/box distances and an analytic time-of-impact. Pure `nalgebra` → WASM-clean.
//! (EPA penetration-depth recovery on overlap is a planned follow-up; use `dcol` for penetration today.)

use nalgebra::{Matrix3, Vector3};

/// A convex shape, given by its support function `support(d)` = the farthest point in direction `d`.
pub trait Support {
    fn support(&self, dir: &Vector3<f64>) -> Vector3<f64>;
}

/// A ball (sphere): support is `center + r·d̂`.
#[derive(Clone, Copy, Debug)]
pub struct Ball {
    pub center: Vector3<f64>,
    pub radius: f64,
}
impl Support for Ball {
    fn support(&self, dir: &Vector3<f64>) -> Vector3<f64> {
        let n = dir.norm();
        if n < 1e-12 {
            self.center
        } else {
            self.center + dir * (self.radius / n)
        }
    }
}

/// An oriented box: support picks the corner farthest along `d` in the box's local frame.
#[derive(Clone, Copy, Debug)]
pub struct Cuboid {
    pub center: Vector3<f64>,
    pub half: Vector3<f64>,
    pub rot: Matrix3<f64>,
}
impl Support for Cuboid {
    fn support(&self, dir: &Vector3<f64>) -> Vector3<f64> {
        let local = self.rot.transpose() * dir;
        let corner = Vector3::new(
            local.x.signum() * self.half.x,
            local.y.signum() * self.half.y,
            local.z.signum() * self.half.z,
        );
        self.center + self.rot * corner
    }
}

/// An arbitrary convex point set: support is the vertex maximizing `d·x`.
#[derive(Clone, Debug)]
pub struct ConvexPoints {
    pub pts: Vec<Vector3<f64>>,
}
impl Support for ConvexPoints {
    fn support(&self, dir: &Vector3<f64>) -> Vector3<f64> {
        *self.pts.iter().max_by(|a, b| a.dot(dir).partial_cmp(&b.dot(dir)).unwrap()).unwrap()
    }
}

/// A support point on the Minkowski difference `A ⊖ B`, tracking the contributing support on each shape
/// (so witness points can be recovered by the same barycentric weights).
#[derive(Clone, Copy, Debug)]
struct Sup {
    v: Vector3<f64>,  // support_a − support_b
    a: Vector3<f64>,  // support on A
}

fn minkowski<A: Support, B: Support>(a: &A, b: &B, dir: &Vector3<f64>) -> Sup {
    let sa = a.support(dir);
    let sb = b.support(&(-dir));
    Sup { v: sa - sb, a: sa }
}

/// The outcome of a GJK query.
#[derive(Clone, Copy, Debug)]
pub struct GjkResult {
    /// Distance between the shapes (0 if intersecting).
    pub distance: f64,
    pub intersecting: bool,
    /// Witness point on A and on B (the closest pair; meaningful when disjoint).
    pub witness_a: Vector3<f64>,
    pub witness_b: Vector3<f64>,
}

/// GJK distance between two convex shapes, with witness points.
pub fn gjk<A: Support, B: Support>(a: &A, b: &B) -> GjkResult {
    let mut dir = Vector3::new(1.0, 0.0, 0.0);
    let mut simplex: Vec<Sup> = vec![minkowski(a, b, &dir)];
    dir = -simplex[0].v;
    for _ in 0..64 {
        if dir.norm() < 1e-12 {
            // origin is on the simplex ⇒ touching/intersecting
            return GjkResult { distance: 0.0, intersecting: true, witness_a: simplex[0].a, witness_b: simplex[0].a };
        }
        let p = minkowski(a, b, &dir);
        // no progress toward the origin ⇒ converged (disjoint)
        if p.v.dot(&dir) - simplex.iter().map(|s| s.v.dot(&dir)).fold(f64::NEG_INFINITY, f64::max) < 1e-10 {
            break;
        }
        simplex.push(p);
        // reduce to the sub-simplex closest to the origin; get the search direction
        if let Some(d) = closest_dir(&mut simplex) {
            dir = d;
        } else {
            // origin enclosed ⇒ intersecting
            let (wa, _wb) = witnesses(&simplex, &Vector3::zeros());
            return GjkResult { distance: 0.0, intersecting: true, witness_a: wa, witness_b: wa };
        }
    }
    // closest point on the final simplex to the origin
    let cp = closest_point_on_simplex(&simplex.iter().map(|s| s.v).collect::<Vec<_>>());
    let (wa, wb) = witnesses(&simplex, &cp);
    GjkResult { distance: cp.norm(), intersecting: false, witness_a: wa, witness_b: wb }
}

/// True iff the two convex shapes overlap (GJK boolean — cheaper than the full distance).
pub fn intersects<A: Support, B: Support>(a: &A, b: &B) -> bool {
    gjk(a, b).intersecting
}

/// Given the current simplex, reduce it to the sub-simplex closest to the origin and return the next
/// search direction (from the closest point toward the origin). `None` ⇒ the origin is inside (a full
/// tetrahedron containing it).
fn closest_dir(simplex: &mut Vec<Sup>) -> Option<Vector3<f64>> {
    let pts: Vec<Vector3<f64>> = simplex.iter().map(|s| s.v).collect();
    let (cp, keep) = closest_and_keep(&pts);
    // rebuild the simplex from the kept indices
    *simplex = keep.iter().map(|&i| simplex[i]).collect();
    if keep.len() == 4 {
        return None; // tetrahedron encloses the origin
    }
    let d = -cp;
    if d.norm() < 1e-12 { None } else { Some(d) }
}

/// Closest point on a simplex (1..4 points) to the origin, plus which vertices support it.
fn closest_and_keep(pts: &[Vector3<f64>]) -> (Vector3<f64>, Vec<usize>) {
    match pts.len() {
        1 => (pts[0], vec![0]),
        2 => {
            let (c, w) = closest_seg(pts[0], pts[1]);
            (c, w.into_iter().collect())
        }
        3 => {
            let (c, w) = closest_tri(pts[0], pts[1], pts[2]);
            (c, w)
        }
        _ => closest_tetra(pts[0], pts[1], pts[2], pts[3]),
    }
}

fn closest_seg(a: Vector3<f64>, b: Vector3<f64>) -> (Vector3<f64>, Vec<usize>) {
    let ab = b - a;
    let t = (-a.dot(&ab)) / ab.dot(&ab).max(1e-18);
    if t <= 0.0 {
        (a, vec![0])
    } else if t >= 1.0 {
        (b, vec![1])
    } else {
        (a + ab * t, vec![0, 1])
    }
}

fn closest_tri(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> (Vector3<f64>, Vec<usize>) {
    // project origin onto triangle plane, then clamp to edges/vertices (Ericson).
    let (ab, ac, ap) = (b - a, c - a, -a);
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, vec![0]);
    }
    let bp = -b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, vec![1]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return (a + ab * t, vec![0, 1]);
    }
    let cp = -c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, vec![2]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return (a + ac * t, vec![0, 2]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * t, vec![1, 2]);
    }
    // inside face region
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, vec![0, 1, 2])
}

fn closest_tetra(a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>, d: Vector3<f64>) -> (Vector3<f64>, Vec<usize>) {
    // if the origin is inside the tetra, keep all four
    let orient = |p: Vector3<f64>, q: Vector3<f64>, r: Vector3<f64>, s: Vector3<f64>| ((q - p).cross(&(r - p))).dot(&(s - p));
    let ref_sign = orient(a, b, c, d).signum();
    let faces = [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)];
    let idx = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
    let mut best = (Vector3::zeros(), vec![], f64::INFINITY);
    let mut inside = true;
    for (f, ix) in faces.iter().zip(idx.iter()) {
        // origin on the same side as the opposite vertex?
        if orient(f.0, f.1, f.2, Vector3::zeros()).signum() != ref_sign {
            inside = false;
            let (cp, keep) = closest_tri(f.0, f.1, f.2);
            let dist = cp.norm();
            if dist < best.2 {
                best = (cp, keep.iter().map(|&k| ix[k]).collect(), dist);
            }
        }
    }
    if inside {
        (Vector3::zeros(), vec![0, 1, 2, 3])
    } else {
        (best.0, best.1)
    }
}

/// Closest point on a simplex to the origin (value only).
fn closest_point_on_simplex(pts: &[Vector3<f64>]) -> Vector3<f64> {
    closest_and_keep(pts).0
}

/// Recover witness points on A and B: the barycentric weights of `cp` on the (Minkowski) simplex applied
/// to the tracked A-supports; the B-witness is then `A-witness − cp` (since v = a − b).
fn witnesses(simplex: &[Sup], cp: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let pts: Vec<Vector3<f64>> = simplex.iter().map(|s| s.v).collect();
    let w = barycentric(&pts, cp);
    let mut wa = Vector3::zeros();
    for (i, &wi) in w.iter().enumerate() {
        wa += simplex[i].a * wi;
    }
    (wa, wa - cp)
}

/// Barycentric weights of point `p` w.r.t. a simplex (least-squares; clamped/normalized).
fn barycentric(pts: &[Vector3<f64>], p: &Vector3<f64>) -> Vec<f64> {
    match pts.len() {
        1 => vec![1.0],
        2 => {
            let ab = pts[1] - pts[0];
            let t = ((p - pts[0]).dot(&ab) / ab.dot(&ab).max(1e-18)).clamp(0.0, 1.0);
            vec![1.0 - t, t]
        }
        _ => {
            // solve for weights via normal equations on the first two edges (triangle); general fallback
            let a = pts[0];
            let ab = pts[1] - a;
            let ac = pts[2] - a;
            let ap = p - a;
            let (d00, d01, d11) = (ab.dot(&ab), ab.dot(&ac), ac.dot(&ac));
            let (d20, d21) = (ap.dot(&ab), ap.dot(&ac));
            let denom = (d00 * d11 - d01 * d01).abs().max(1e-18);
            let v = (d11 * d20 - d01 * d21) / denom;
            let w = (d00 * d21 - d01 * d20) / denom;
            let u = 1.0 - v - w;
            if pts.len() == 3 {
                vec![u, v, w]
            } else {
                vec![u, v, w, 0.0] // tetra witness handled at faces; last vertex unused
            }
        }
    }
}

/// **Conservative-advancement CCD**: earliest time `t ∈ [0,1]` at which convex `a` (moving by `va`) and
/// `b` (moving by `vb`) first come within `tol`, or `None` if they stay apart over the step. Uses the GJK
/// distance and the closing speed along the separating direction to take safe sub-steps.
pub fn ccd_toi<A, B>(a: &A, va: &Vector3<f64>, b: &B, vb: &Vector3<f64>, tol: f64) -> Option<f64>
where
    A: Support + Translate,
    B: Support + Translate,
{
    let rel = va - vb;
    let rel_speed = rel.norm();
    if rel_speed < 1e-12 {
        return None;
    }
    let mut t = 0.0;
    for _ in 0..64 {
        let at = a.translated(&(va * t));
        let bt = b.translated(&(vb * t));
        let g = gjk(&at, &bt);
        if g.distance <= tol {
            return Some(t);
        }
        // safe advance: distance can shrink no faster than the relative speed
        let dt = (g.distance - tol) / rel_speed;
        t += dt;
        if t >= 1.0 {
            return None;
        }
    }
    Some(t)
}

/// Rigid translation of a support shape (for CCD sweeping).
pub trait Translate: Sized {
    fn translated(&self, by: &Vector3<f64>) -> Self;
}
impl Translate for Ball {
    fn translated(&self, by: &Vector3<f64>) -> Self {
        Ball { center: self.center + by, radius: self.radius }
    }
}
impl Translate for Cuboid {
    fn translated(&self, by: &Vector3<f64>) -> Self {
        Cuboid { center: self.center + by, half: self.half, rot: self.rot }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    #[test]
    fn gjk_distance_between_two_balls_is_exact() {
        let a = Ball { center: v(0.0, 0.0, 0.0), radius: 0.5 };
        let b = Ball { center: v(3.0, 0.0, 0.0), radius: 0.7 };
        let g = gjk(&a, &b);
        let expect = 3.0 - 0.5 - 0.7;
        assert!((g.distance - expect).abs() < 1e-6, "ball distance {} vs {expect}", g.distance);
        assert!(!g.intersecting);
        // witnesses lie on each surface and their separation equals the distance
        assert!((g.witness_a - v(0.5, 0.0, 0.0)).norm() < 1e-5, "witness A {:?}", g.witness_a);
        assert!(((g.witness_a - g.witness_b).norm() - g.distance).abs() < 1e-6, "witness separation ≠ distance");
    }

    #[test]
    fn gjk_distance_between_two_boxes_is_exact() {
        let a = Cuboid { center: v(0.0, 0.0, 0.0), half: v(0.5, 0.5, 0.5), rot: Matrix3::identity() };
        let b = Cuboid { center: v(2.0, 0.0, 0.0), half: v(0.5, 0.5, 0.5), rot: Matrix3::identity() };
        let g = gjk(&a, &b);
        assert!((g.distance - 1.0).abs() < 1e-6, "box gap {} vs 1.0", g.distance); // 2.0 − 0.5 − 0.5
    }

    #[test]
    fn overlap_is_detected_and_disjoint_is_not() {
        let a = Ball { center: v(0.0, 0.0, 0.0), radius: 1.0 };
        let hit = Ball { center: v(1.2, 0.0, 0.0), radius: 0.5 };
        let miss = Ball { center: v(3.0, 0.0, 0.0), radius: 0.5 };
        assert!(intersects(&a, &hit), "overlapping balls must intersect");
        assert!(!intersects(&a, &miss), "separated balls must not intersect");
    }


    #[test]
    fn ccd_time_of_impact_matches_the_analytic_value() {
        // a ball at x=0 (r=0.5) flying +x by 4 over the step; a static ball at x=3 (r=0.5). They touch when
        // the centers are 1.0 apart ⇒ the mover reaches x=2.0 ⇒ t = 2.0/4.0 = 0.5.
        let a = Ball { center: v(0.0, 0.0, 0.0), radius: 0.5 };
        let b = Ball { center: v(3.0, 0.0, 0.0), radius: 0.5 };
        let toi = ccd_toi(&a, &v(4.0, 0.0, 0.0), &b, &v(0.0, 0.0, 0.0), 1e-4).expect("should collide");
        assert!((toi - 0.5).abs() < 1e-3, "TOI {toi} vs 0.5");
    }

    #[test]
    fn ccd_reports_no_impact_when_they_pass_clear() {
        let a = Ball { center: v(0.0, 0.0, 0.0), radius: 0.3 };
        let b = Ball { center: v(0.0, 3.0, 0.0), radius: 0.3 }; // far in y
        let toi = ccd_toi(&a, &v(4.0, 0.0, 0.0), &b, &v(0.0, 0.0, 0.0), 1e-4);
        assert!(toi.is_none(), "shapes that never approach must report no impact");
    }
}
