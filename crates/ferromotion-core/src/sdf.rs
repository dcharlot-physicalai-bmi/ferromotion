//! **Signed distance fields** for collision — the world representation behind GPU motion generators
//! like cuRobo. Each obstacle primitive exposes an analytic signed distance `d(p)` (negative inside)
//! and its exact gradient `∇d` (the outward surface normal, unit-length by the eikonal property
//! `‖∇d‖ = 1`). A scene is their union (`d = min`), and a robot approximated by **collision spheres**
//! queries it for the minimum clearance and a smooth avoidance direction — cheap, differentiable
//! collision for planning and reactive control. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector3;

/// An analytic signed-distance primitive.
#[derive(Clone, Copy, Debug)]
pub enum Sdf {
    Sphere { center: Vector3<f64>, radius: f64 },
    /// Axis-aligned box: `|p − center|` compared against the half-extents.
    Box { center: Vector3<f64>, half: Vector3<f64> },
    /// Half-space `n·p − offset` (with `n` unit).
    Plane { normal: Vector3<f64>, offset: f64 },
    Capsule { a: Vector3<f64>, b: Vector3<f64>, radius: f64 },
    /// Torus around the `+y` axis at `center`: `major` tube-centre radius, `minor` tube radius.
    Torus { center: Vector3<f64>, major: f64, minor: f64 },
}

impl Sdf {
    /// Signed distance to the surface (negative inside).
    pub fn distance(&self, p: &Vector3<f64>) -> f64 {
        match self {
            Sdf::Sphere { center, radius } => (p - center).norm() - radius,
            Sdf::Box { center, half } => {
                let q = (p - center).map(|v| v.abs()) - half;
                q.map(|v| v.max(0.0)).norm() + q.max().min(0.0)
            }
            Sdf::Plane { normal, offset } => normal.dot(p) - offset,
            Sdf::Capsule { a, b, radius } => {
                let ab = b - a;
                let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
                (p - (a + t * ab)).norm() - radius
            }
            Sdf::Torus { center, major, minor } => {
                let d = p - center;
                let planar = (d.x * d.x + d.z * d.z).sqrt() - major;
                (planar * planar + d.y * d.y).sqrt() - minor
            }
        }
    }

    /// Exact gradient `∇d` (outward unit normal, away from the surface).
    pub fn gradient(&self, p: &Vector3<f64>) -> Vector3<f64> {
        match self {
            Sdf::Sphere { center, .. } => safe_normalize(p - center),
            Sdf::Box { center, half } => {
                let d = p - center;
                let s = d.map(|v| if v < 0.0 { -1.0 } else { 1.0 });
                let q = d.map(|v| v.abs()) - half;
                if q.max() > 0.0 {
                    // Outside: gradient of ‖max(q,0)‖.
                    let qp = q.map(|v| v.max(0.0));
                    safe_normalize(s.component_mul(&qp))
                } else {
                    // Inside: point out through the nearest face (the largest, least-negative q).
                    let imax = (0..3).max_by(|&i, &j| q[i].partial_cmp(&q[j]).unwrap()).unwrap();
                    let mut g = Vector3::zeros();
                    g[imax] = s[imax];
                    g
                }
            }
            Sdf::Plane { normal, .. } => *normal,
            Sdf::Capsule { a, b, .. } => {
                let ab = b - a;
                let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
                safe_normalize(p - (a + t * ab))
            }
            Sdf::Torus { center, major, .. } => {
                let d = p - center;
                let planar = (d.x * d.x + d.z * d.z).sqrt();
                let q0 = planar - major;
                let qn = (q0 * q0 + d.y * d.y).sqrt().max(1e-12);
                let ps = planar.max(1e-12);
                // exact eikonal gradient (already unit): (q0/qn)·∇planar + (d.y/qn)·ê_y
                Vector3::new((q0 / qn) * (d.x / ps), d.y / qn, (q0 / qn) * (d.z / ps))
            }
        }
    }
}

/// CSG **union** of two signed distances (the closer surface): `min(d1, d2)`.
pub fn op_union(d1: f64, d2: f64) -> f64 {
    d1.min(d2)
}

/// CSG **intersection** of two signed distances (inside both): `max(d1, d2)`.
pub fn op_intersect(d1: f64, d2: f64) -> f64 {
    d1.max(d2)
}

/// CSG **subtraction** `A − B` (inside `A`, outside `B`): `max(d1, −d2)`.
pub fn op_subtract(d1: f64, d2: f64) -> f64 {
    d1.max(-d2)
}

/// **Smooth union** (Quilez polynomial smin) with blend radius `k > 0`: a rounded, C¹ blend of two shapes
/// that never exceeds their hard union `min(d1, d2)` and approaches it as `k → 0`.
pub fn op_smooth_union(d1: f64, d2: f64, k: f64) -> f64 {
    let h = (0.5 + 0.5 * (d2 - d1) / k).clamp(0.0, 1.0);
    d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
}

fn safe_normalize(v: Vector3<f64>) -> Vector3<f64> {
    let n = v.norm();
    if n > 1e-12 { v / n } else { Vector3::new(0.0, 0.0, 1.0) }
}

/// A collision scene: the union of primitives, `d(p) = min_i d_i(p)`.
#[derive(Clone, Debug, Default)]
pub struct SdfScene {
    pub prims: Vec<Sdf>,
}

impl SdfScene {
    pub fn distance(&self, p: &Vector3<f64>) -> f64 {
        self.prims.iter().map(|s| s.distance(p)).fold(f64::INFINITY, f64::min)
    }

    /// Gradient of the union — the gradient of the nearest primitive.
    pub fn gradient(&self, p: &Vector3<f64>) -> Vector3<f64> {
        let nearest = self.prims.iter().min_by(|x, y| x.distance(p).partial_cmp(&y.distance(p)).unwrap());
        nearest.map(|s| s.gradient(p)).unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0))
    }

    /// Minimum clearance of a set of collision spheres `(center, radius)` (negative ⇒ penetration).
    pub fn min_clearance(&self, spheres: &[(Vector3<f64>, f64)]) -> f64 {
        spheres.iter().map(|&(c, r)| self.distance(&c) - r).fold(f64::INFINITY, f64::min)
    }

    /// The worst (least-clearance) sphere and the avoidance direction on its center (`∇` of clearance).
    pub fn worst_contact(&self, spheres: &[(Vector3<f64>, f64)]) -> (usize, f64, Vector3<f64>) {
        let mut best = (0usize, f64::INFINITY);
        for (i, &(c, r)) in spheres.iter().enumerate() {
            let clr = self.distance(&c) - r;
            if clr < best.1 {
                best = (i, clr);
            }
        }
        let (c, _) = spheres[best.0];
        (best.0, best.1, self.gradient(&c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_grad(s: &Sdf, p: &Vector3<f64>) -> Vector3<f64> {
        let eps = 1e-6;
        Vector3::from_iterator((0..3).map(|i| {
            let (mut pp, mut pm) = (*p, *p);
            pp[i] += eps;
            pm[i] -= eps;
            (s.distance(&pp) - s.distance(&pm)) / (2.0 * eps)
        }))
    }

    #[test]
    fn primitive_distances_and_gradients_are_exact() {
        let cases = [
            Sdf::Sphere { center: Vector3::new(1.0, 0.0, 0.0), radius: 0.5 },
            Sdf::Box { center: Vector3::new(0.0, 0.0, 0.0), half: Vector3::new(1.0, 0.5, 0.75) },
            Sdf::Plane { normal: Vector3::new(0.0, 1.0, 0.0), offset: -0.3 },
            Sdf::Capsule { a: Vector3::new(-1.0, 0.0, 0.0), b: Vector3::new(1.0, 0.0, 0.0), radius: 0.4 },
        ];
        let probes = [Vector3::new(2.0, 1.0, 0.5), Vector3::new(0.3, 0.2, -0.4), Vector3::new(-1.5, 0.9, 0.2)];
        for s in &cases {
            for p in &probes {
                // Gradient matches finite differences …
                let (g, fd) = (s.gradient(p), fd_grad(s, p));
                assert!((g - fd).norm() < 1e-4, "gradient off for {s:?} at {p:?}: {g:?} vs {fd:?}");
                // … and satisfies the eikonal property ‖∇d‖ = 1 outside the surface.
                if s.distance(p) > 0.05 {
                    assert!((g.norm() - 1.0).abs() < 1e-4, "‖∇d‖ ≠ 1 for {s:?} at {p:?}: {}", g.norm());
                }
            }
        }
    }

    #[test]
    fn known_distances() {
        let sphere = Sdf::Sphere { center: Vector3::zeros(), radius: 1.0 };
        assert!((sphere.distance(&Vector3::new(3.0, 0.0, 0.0)) - 2.0).abs() < 1e-12); // outside
        assert!((sphere.distance(&Vector3::new(0.5, 0.0, 0.0)) + 0.5).abs() < 1e-12); // inside (−0.5)
        let b = Sdf::Box { center: Vector3::zeros(), half: Vector3::new(1.0, 1.0, 1.0) };
        assert!((b.distance(&Vector3::new(2.0, 0.0, 0.0)) - 1.0).abs() < 1e-12); // 1 past the face
        assert!(b.distance(&Vector3::zeros()) < 0.0); // center is inside
    }

    #[test]
    fn scene_union_and_robot_sphere_clearance() {
        let scene = SdfScene {
            prims: vec![
                Sdf::Box { center: Vector3::new(2.0, 0.0, 0.0), half: Vector3::new(0.5, 0.5, 0.5) },
                Sdf::Sphere { center: Vector3::new(-2.0, 0.0, 0.0), radius: 0.5 },
            ],
        };
        // The union is the nearer of the two obstacles.
        assert!((scene.distance(&Vector3::new(1.0, 0.0, 0.0)) - 0.5).abs() < 1e-9); // 0.5 from the box face

        // A robot collision sphere near the box: clearance = box distance − sphere radius.
        let spheres = [(Vector3::new(1.0, 0.0, 0.0), 0.2), (Vector3::new(0.0, 3.0, 0.0), 0.2)];
        let clr = scene.min_clearance(&spheres);
        assert!((clr - 0.3).abs() < 1e-9, "clearance {clr} (expected 0.3)"); // 0.5 − 0.2
        let (idx, worst, dir) = scene.worst_contact(&spheres);
        assert_eq!(idx, 0, "the sphere by the box is the worst");
        assert!((worst - 0.3).abs() < 1e-9);
        // Avoidance direction points away from the box (−x here).
        assert!(dir.x < -0.9, "avoidance direction should point away from the box: {dir:?}");

        // An overlapping sphere reports negative clearance.
        let overlap = [(Vector3::new(1.6, 0.0, 0.0), 0.3)]; // box face at x=1.5, sphere reaches 1.3
        assert!(scene.min_clearance(&overlap) < 0.0, "overlap should be a collision");
    }

    #[test]
    fn the_torus_distance_and_gradient_are_exact() {
        // THE ORACLE. Torus (major R=2, minor r=0.5) around +y. The outer-equator surface point (R+r,0,0)
        // is on the surface (d=0); the centre is R−r inside the hole; and the gradient matches a finite
        // difference (and is unit by the eikonal property).
        let t = Sdf::Torus { center: Vector3::zeros(), major: 2.0, minor: 0.5 };
        assert!(t.distance(&Vector3::new(2.5, 0.0, 0.0)).abs() < 1e-12, "outer equator on surface");
        assert!((t.distance(&Vector3::new(0.0, 0.0, 0.0)) - 1.5).abs() < 1e-12, "centre distance R−r");
        assert!(t.distance(&Vector3::new(2.0, 0.5, 0.0)).abs() < 1e-9, "top of the tube on surface");
        for p in [Vector3::new(1.5, 0.3, 0.4), Vector3::new(-2.2, 0.1, 0.3), Vector3::new(0.5, 0.5, 1.9)] {
            let g = t.gradient(&p);
            assert!((g.norm() - 1.0).abs() < 1e-6, "gradient should be unit: {}", g.norm());
            assert!((g - fd_grad(&t, &p)).norm() < 1e-4, "torus gradient vs finite diff");
        }
    }

    #[test]
    fn the_csg_operators_combine_shapes_correctly() {
        let a = Sdf::Sphere { center: Vector3::new(-0.4, 0.0, 0.0), radius: 1.0 };
        let b = Sdf::Sphere { center: Vector3::new(0.4, 0.0, 0.0), radius: 1.0 };
        let inside_both = Vector3::new(0.0, 0.0, 0.0); // inside both spheres
        let only_a = Vector3::new(-1.2, 0.0, 0.0); // inside a, outside b
        // union: inside if inside either
        assert!(op_union(a.distance(&only_a), b.distance(&only_a)) < 0.0, "union contains only_a");
        // intersection: inside only where both are
        assert!(op_intersect(a.distance(&inside_both), b.distance(&inside_both)) < 0.0, "intersection at center");
        assert!(op_intersect(a.distance(&only_a), b.distance(&only_a)) > 0.0, "only_a is outside the intersection");
        // subtraction A−B: inside a but outside b
        assert!(op_subtract(a.distance(&only_a), b.distance(&only_a)) < 0.0, "A−B contains only_a");
        assert!(op_subtract(a.distance(&inside_both), b.distance(&inside_both)) > 0.0, "A−B excludes the B region");
    }

    #[test]
    fn smooth_union_bounds_and_limits_to_the_hard_union() {
        let (d1, d2) = (0.4, 0.7);
        // never exceeds the hard min
        assert!(op_smooth_union(d1, d2, 0.3) <= d1.min(d2) + 1e-12, "smooth min ≤ hard min");
        // approaches the hard min as k → 0
        assert!((op_smooth_union(d1, d2, 1e-6) - d1.min(d2)).abs() < 1e-4, "k→0 recovers hard union");
        // and it is a genuine rounding (strictly below min when the two are comparable)
        assert!(op_smooth_union(0.5, 0.5, 0.4) < 0.5, "equal distances get rounded down");
    }
}
