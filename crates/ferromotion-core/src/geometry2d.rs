//! **Planar computational geometry** — the small kit of exact 2-D primitives the rest of the stack keeps
//! needing: the **convex hull** (Andrew's monotone chain), signed **polygon area** and **centroid**
//! (shoelace), a **point-in-polygon** test, and the **minimum enclosing circle** (Welzl). These are the
//! building blocks of *support-polygon* stability margins (is the ZMP inside the foot-support hull, and how
//! far from the edge?), robot footprints and swept areas, grasp/wrench hulls, and coverage regions.
//!
//! All exact and deterministic. Verified: the hull of a point set contains every point and its vertices are
//! extreme; the shoelace area/centroid match closed forms; point-in-polygon separates inside from outside;
//! and the minimum enclosing circle contains all points with the smallest possible radius. Pure `nalgebra`
//! → WASM-clean.

use nalgebra::Vector2;

fn cross(o: &Vector2<f64>, a: &Vector2<f64>, b: &Vector2<f64>) -> f64 {
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
}

/// The **convex hull** of a point set by Andrew's monotone chain, returned counter-clockwise without the
/// duplicated closing vertex. Collinear boundary points are dropped.
pub fn convex_hull(points: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap().then(a.y.partial_cmp(&b.y).unwrap()));
    pts.dedup_by(|a, b| (*a - *b).norm() < 1e-12);
    let n = pts.len();
    if n < 3 {
        return pts;
    }
    let mut hull: Vec<Vector2<f64>> = Vec::with_capacity(2 * n);
    // lower hull
    for p in &pts {
        while hull.len() >= 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    // upper hull
    let lower_len = hull.len() + 1;
    for p in pts.iter().rev() {
        while hull.len() >= lower_len && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(*p);
    }
    hull.pop(); // drop the repeated first point
    hull
}

/// Signed area of a polygon (CCW positive) by the shoelace formula.
pub fn signed_area(poly: &[Vector2<f64>]) -> f64 {
    let n = poly.len();
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    a / 2.0
}

/// Area of a polygon (always non-negative).
pub fn polygon_area(poly: &[Vector2<f64>]) -> f64 {
    signed_area(poly).abs()
}

/// Centroid of a simple polygon (area-weighted, shoelace form).
pub fn polygon_centroid(poly: &[Vector2<f64>]) -> Vector2<f64> {
    let n = poly.len();
    let a = signed_area(poly);
    if a.abs() < 1e-15 {
        // degenerate: fall back to the vertex average
        return poly.iter().sum::<Vector2<f64>>() / n as f64;
    }
    let (mut cx, mut cy) = (0.0, 0.0);
    for i in 0..n {
        let j = (i + 1) % n;
        let c = poly[i].x * poly[j].y - poly[j].x * poly[i].y;
        cx += (poly[i].x + poly[j].x) * c;
        cy += (poly[i].y + poly[j].y) * c;
    }
    Vector2::new(cx / (6.0 * a), cy / (6.0 * a))
}

/// Whether point `p` lies inside a simple polygon (ray-casting; boundary counts as inside within `eps`).
pub fn point_in_polygon(poly: &[Vector2<f64>], p: &Vector2<f64>) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        // on-edge check
        let ab = b - a;
        let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
        if (p - (a + t * ab)).norm() < 1e-9 {
            return true;
        }
        if (a.y > p.y) != (b.y > p.y) {
            let x_cross = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if p.x < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// A circle: centre and radius.
#[derive(Clone, Copy, Debug)]
pub struct Circle {
    pub center: Vector2<f64>,
    pub radius: f64,
}

fn circle_two(a: &Vector2<f64>, b: &Vector2<f64>) -> Circle {
    Circle { center: (a + b) / 2.0, radius: (a - b).norm() / 2.0 }
}

fn circle_three(a: &Vector2<f64>, b: &Vector2<f64>, c: &Vector2<f64>) -> Option<Circle> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-14 {
        return None; // collinear
    }
    let (a2, b2, c2) = (a.norm_squared(), b.norm_squared(), c.norm_squared());
    let ux = (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d;
    let uy = (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d;
    let center = Vector2::new(ux, uy);
    Some(Circle { center, radius: (center - a).norm() })
}

/// The **minimum enclosing circle** of a point set (Welzl's incremental algorithm). Returns the smallest
/// circle containing every point.
pub fn min_enclosing_circle(points: &[Vector2<f64>]) -> Circle {
    let p = points;
    if p.is_empty() {
        return Circle { center: Vector2::zeros(), radius: 0.0 };
    }
    let inside = |c: &Circle, q: &Vector2<f64>| (c.center - q).norm() <= c.radius + 1e-9;
    let mut c = Circle { center: p[0], radius: 0.0 };
    for i in 1..p.len() {
        if inside(&c, &p[i]) {
            continue;
        }
        c = Circle { center: p[i], radius: 0.0 };
        for j in 0..i {
            if inside(&c, &p[j]) {
                continue;
            }
            c = circle_two(&p[i], &p[j]);
            for k in 0..j {
                if inside(&c, &p[k]) {
                    continue;
                }
                if let Some(c3) = circle_three(&p[i], &p[j], &p[k]) {
                    c = c3;
                }
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_convex_hull_contains_every_point_and_is_extreme() {
        // THE ORACLE. Interior points are excluded from the hull, and every input point lies inside it.
        let pts = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::new(2.0, 2.0),
            Vector2::new(0.0, 2.0),
            Vector2::new(1.0, 1.0),   // interior
            Vector2::new(0.5, 0.3),   // interior
            Vector2::new(1.0, 0.0),   // on an edge ⇒ dropped (collinear)
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4, "square hull should have 4 corners, got {}", hull.len());
        assert!(signed_area(&hull) > 0.0, "hull should be CCW");
        for p in &pts {
            assert!(point_in_polygon(&hull, p), "point {p:?} should be inside the hull");
        }
    }

    #[test]
    fn shoelace_area_and_centroid_match_closed_forms() {
        // Unit square: area 1, centroid (0.5, 0.5).
        let sq = [Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0), Vector2::new(1.0, 1.0), Vector2::new(0.0, 1.0)];
        assert!((polygon_area(&sq) - 1.0).abs() < 1e-12);
        assert!((polygon_centroid(&sq) - Vector2::new(0.5, 0.5)).norm() < 1e-12);
        // Right triangle: area 0.5, centroid at the average of the vertices.
        let tri = [Vector2::new(0.0, 0.0), Vector2::new(3.0, 0.0), Vector2::new(0.0, 3.0)];
        assert!((polygon_area(&tri) - 4.5).abs() < 1e-12);
        assert!((polygon_centroid(&tri) - Vector2::new(1.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn point_in_polygon_separates_inside_from_outside() {
        let poly = [Vector2::new(0.0, 0.0), Vector2::new(4.0, 0.0), Vector2::new(4.0, 3.0), Vector2::new(0.0, 3.0)];
        assert!(point_in_polygon(&poly, &Vector2::new(2.0, 1.5)), "interior point");
        assert!(!point_in_polygon(&poly, &Vector2::new(5.0, 1.5)), "exterior point");
        assert!(!point_in_polygon(&poly, &Vector2::new(2.0, -0.5)), "below the polygon");
        assert!(point_in_polygon(&poly, &Vector2::new(0.0, 1.5)), "on the boundary counts as inside");
    }

    #[test]
    fn the_minimum_enclosing_circle_is_smallest_and_covers_all() {
        // Two-point set: the MEC is the diameter circle.
        let two = [Vector2::new(-1.0, 0.0), Vector2::new(3.0, 0.0)];
        let c = min_enclosing_circle(&two);
        assert!((c.center - Vector2::new(1.0, 0.0)).norm() < 1e-9 && (c.radius - 2.0).abs() < 1e-9, "two-point MEC is the diameter");

        // A cloud with one far corner: the MEC covers everything, and no smaller circle could (radius equals
        // the max distance from the centre to a point, tightly).
        let cloud = [Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0), Vector2::new(1.0, 1.0), Vector2::new(0.4, 0.6), Vector2::new(0.9, 0.2)];
        let c = min_enclosing_circle(&cloud);
        let far = cloud.iter().map(|p| (p - c.center).norm()).fold(0.0, f64::max);
        for p in &cloud {
            assert!((p - c.center).norm() <= c.radius + 1e-9, "point {p:?} outside the MEC");
        }
        assert!((c.radius - far).abs() < 1e-6, "radius should be tight against the farthest point");
        // for the unit square the MEC radius is the half-diagonal √2/2
        assert!((c.radius - std::f64::consts::SQRT_2 / 2.0).abs() < 1e-6, "unit-square MEC radius = √2/2, got {}", c.radius);
    }
}
