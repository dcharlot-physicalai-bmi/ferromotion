//! DCOL — differentiable collision between convex primitives (Tracy, Howell, Manchester, ICRA 2023).
//! Proximity is the **minimum uniform scaling** `α` applied to both primitives before they touch:
//! `α < 1` ⇒ interpenetrating, `α = 1` ⇒ touching, `α > 1` ⇒ separated. It's a single convex program
//! (LP/SOCP), smooth in the primitive poses even during penetration — a differentiable signed-distance
//! surrogate that feeds obstacle/contact constraints. Solved with `clarabel`. Pure Rust → WASM-clean.

use crate::Iso;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::{DMatrix, Vector3};

/// A convex primitive (in its local frame; posed by an [`Iso`] at query time).
#[derive(Clone, Debug)]
pub enum Primitive {
    /// Ball of the given radius, centered at the frame origin.
    Sphere { radius: f64 },
    /// Convex polytope as half-spaces `nᵀ·x ≤ offset` (normals in the local frame).
    Polytope { halfspaces: Vec<(Vector3<f64>, f64)> },
}

impl Primitive {
    /// Axis-aligned box with the given half-extents (as a 6-face polytope).
    pub fn box_half(hx: f64, hy: f64, hz: f64) -> Primitive {
        Primitive::Polytope {
            halfspaces: vec![
                (Vector3::x(), hx),
                (-Vector3::x(), hx),
                (Vector3::y(), hy),
                (-Vector3::y(), hy),
                (Vector3::z(), hz),
                (-Vector3::z(), hz),
            ],
        }
    }
}

// Append one primitive's "x ∈ α·prim" constraints (rows of A, entries of b, one cone).
fn append(prim: &Primitive, pose: &Iso, a: &mut Vec<[f64; 4]>, b: &mut Vec<f64>, cones: &mut Vec<SupportedConeT<f64>>) {
    let c = pose.translation.vector; // scaling center = primitive origin
    match prim {
        Primitive::Sphere { radius } => {
            // (α·r, x − c) ∈ SOC(4)  ⟺  ‖x − c‖ ≤ α·r.  s = b − A·z.
            a.push([0.0, 0.0, 0.0, -radius]); // s0 = α·r
            b.push(0.0);
            a.push([-1.0, 0.0, 0.0, 0.0]); // s1 = x1 − c1
            b.push(-c.x);
            a.push([0.0, -1.0, 0.0, 0.0]);
            b.push(-c.y);
            a.push([0.0, 0.0, -1.0, 0.0]);
            b.push(-c.z);
            cones.push(SupportedConeT::SecondOrderConeT(4));
        }
        Primitive::Polytope { halfspaces } => {
            let r = pose.rotation.to_rotation_matrix();
            for (n_local, off) in halfspaces {
                let nw = r * n_local; // world normal
                // nw·x − α·off ≤ nw·c  →  s = nw·c − (nw·x − α·off) ≥ 0.
                a.push([nw.x, nw.y, nw.z, -off]);
                b.push(nw.dot(&c));
            }
            cones.push(SupportedConeT::NonnegativeConeT(halfspaces.len()));
        }
    }
}

/// DCOL proximity `α` between two posed primitives (`< 1` ⇒ collision). Minimizes `α` over a world
/// witness point that lies in both `α`-scaled primitives.
pub fn proximity(a: &Primitive, ta: &Iso, b: &Primitive, tb: &Iso) -> f64 {
    // Variables z = [x(3), α]; minimize α.
    let mut rows: Vec<[f64; 4]> = Vec::new();
    let mut bvec: Vec<f64> = Vec::new();
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    // α ≥ 0  (−α ≤ 0).
    rows.push([0.0, 0.0, 0.0, -1.0]);
    bvec.push(0.0);
    cones.push(SupportedConeT::NonnegativeConeT(1));
    append(a, ta, &mut rows, &mut bvec, &mut cones);
    append(b, tb, &mut rows, &mut bvec, &mut cones);

    let m = rows.len();
    let a_dense = DMatrix::from_fn(m, 4, |i, j| rows[i][j]);
    let a_csc = dense_to_csc(&a_dense);
    let p = CscMatrix::new(4, 4, vec![0; 5], vec![], vec![]); // zero quadratic → LP/SOCP
    let q = [0.0, 0.0, 0.0, 1.0];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p, &q, &a_csc, &bvec, &cones, settings).unwrap();
    solver.solve();
    solver.solution.x[3]
}

fn dense_to_csc(a: &DMatrix<f64>) -> CscMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..m {
            if a[(i, j)] != 0.0 {
                rowval.push(i);
                nzval.push(a[(i, j)]);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

/// Analytic gradient of sphere-sphere proximity w.r.t. sphere B's center: `∂α/∂c_b`. (The general
/// primitive gradient comes from the KKT dual solution — the DCOL implicit-function-theorem path.)
pub fn proximity_grad_spheres(ca: Vector3<f64>, ra: f64, cb: Vector3<f64>, rb: f64) -> Vector3<f64> {
    let d = cb - ca;
    let dist = d.norm();
    if dist < 1e-12 {
        return Vector3::zeros();
    }
    d / (dist * (ra + rb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Translation3;

    fn at(x: f64, y: f64, z: f64) -> Iso {
        Iso::from_parts(Translation3::new(x, y, z), Default::default())
    }

    #[test]
    fn sphere_sphere_proximity_matches_closed_form() {
        let (sa, sb) = (Primitive::Sphere { radius: 1.0 }, Primitive::Sphere { radius: 0.5 });
        // Separated: centers 3 apart, radii sum 1.5 → α = 3/1.5 = 2.0.
        let sep = proximity(&sa, &at(0.0, 0.0, 0.0), &sb, &at(3.0, 0.0, 0.0));
        assert!((sep - 2.0).abs() < 1e-5, "separated α = {sep}");
        assert!(sep > 1.0, "should be separated");
        // Overlapping: centers 1 apart → α = 1/1.5 < 1.
        let hit = proximity(&sa, &at(0.0, 0.0, 0.0), &sb, &at(1.0, 0.0, 0.0));
        assert!((hit - 1.0 / 1.5).abs() < 1e-5 && hit < 1.0, "colliding α = {hit}");
    }

    #[test]
    fn sphere_box_collision_detection() {
        let sphere = Primitive::Sphere { radius: 0.5 };
        let unit_box = Primitive::box_half(1.0, 1.0, 1.0);
        // Sphere well outside the box → α > 1.
        let far = proximity(&sphere, &at(3.0, 0.0, 0.0), &unit_box, &at(0.0, 0.0, 0.0));
        assert!(far > 1.0, "far case should be separated: {far}");
        // Sphere overlapping the box face → α < 1.
        let near = proximity(&sphere, &at(1.2, 0.0, 0.0), &unit_box, &at(0.0, 0.0, 0.0));
        assert!(near < 1.0, "overlapping case should collide: {near}");
    }

    #[test]
    fn sphere_gradient_matches_finite_difference() {
        let (ca, ra, rb) = (Vector3::new(0.0, 0.0, 0.0), 1.0, 0.5);
        let cb = Vector3::new(2.0, 0.5, 0.0);
        let g = proximity_grad_spheres(ca, ra, cb, rb);
        let (sa, sb) = (Primitive::Sphere { radius: ra }, Primitive::Sphere { radius: rb });
        let eps = 1e-6;
        for i in 0..3 {
            let mut cp = cb;
            cp[i] += eps;
            let fd = (proximity(&sa, &at(ca.x, ca.y, ca.z), &sb, &at(cp.x, cp.y, cp.z))
                - proximity(&sa, &at(ca.x, ca.y, ca.z), &sb, &at(cb.x, cb.y, cb.z)))
                / eps;
            assert!((g[i] - fd).abs() < 1e-4, "grad[{i}]: analytic {} vs fd {}", g[i], fd);
        }
    }
}
