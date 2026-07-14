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
        }
    }
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
}
