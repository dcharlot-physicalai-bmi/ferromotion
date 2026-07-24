//! **Unstructured-mesh finite volume** (Honest Fluids — the SU2/OpenFOAM beachhead on arbitrary
//! meshes). The structured [`crate::fvm`] proved the flux-form idea; this is the capability that
//! actually distinguishes production CFD: solving on an *unstructured triangular mesh*, where the
//! cells have no grid indices and the connectivity is an explicit list. It assembles the discrete
//! Poisson operator from per-element gradient contributions — the vertex-centered box-scheme, which
//! coincides with the P1 Galerkin stiffness matrix — and factors it with a sparse Cholesky.
//!
//! Verified with the **Method of Manufactured Solutions**, the gold standard for unstructured CFD:
//! pick `φ*`, feed the analytic source `S = −∇²φ*`, solve, and measure the error convergence order
//! as the mesh refines. Second order is confirmed on a regular triangulation *and on a jittered,
//! genuinely irregular mesh* — proof the solver isn't secretly exploiting grid structure.

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Side};

/// A 2-D triangular mesh: vertex coordinates and a triangle list (no grid indices — unstructured).
pub struct TriMesh {
    pub verts: Vec<[f64; 2]>,
    pub tris: Vec<[usize; 3]>,
    pub boundary: Vec<bool>,
}

impl TriMesh {
    /// Triangulate the unit square: an `n × n` vertex grid, each cell split into two triangles.
    /// `jitter` displaces interior vertices by up to `jitter·h` — turning the regular triangulation
    /// into a genuinely irregular unstructured mesh (deterministic hash offsets, no RNG).
    pub fn unit_square(n: usize, jitter: f64) -> TriMesh {
        let h = 1.0 / (n - 1) as f64;
        let mut verts = Vec::with_capacity(n * n);
        let mut boundary = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let (mut x, mut y) = (i as f64 * h, j as f64 * h);
                let is_b = i == 0 || j == 0 || i == n - 1 || j == n - 1;
                if !is_b && jitter > 0.0 {
                    // deterministic pseudo-offset in [-1,1] from the integer coordinates
                    let hx = (((i * 73856093) ^ (j * 19349663)) & 0xffff) as f64 / 65535.0;
                    let hy = (((i * 83492791) ^ (j * 12582917)) & 0xffff) as f64 / 65535.0;
                    x += (hx * 2.0 - 1.0) * jitter * h;
                    y += (hy * 2.0 - 1.0) * jitter * h;
                }
                verts.push([x, y]);
                boundary.push(is_b);
            }
        }
        let vid = |i: usize, j: usize| i * n + j;
        let mut tris = Vec::new();
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                tris.push([vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]);
                tris.push([vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]);
            }
        }
        TriMesh { verts, tris, boundary }
    }
}

/// Solve `−∇²φ = S` on `mesh` with Dirichlet boundary values from `bc`, source from `source`.
/// Returns the vertex values. Assembles the P1 stiffness (box-scheme) operator and factors it with a
/// sparse Cholesky — the same machinery a general unstructured Poisson solve uses.
#[allow(clippy::needless_range_loop)] // vertex index v addresses several per-vertex arrays together
pub fn solve_poisson(mesh: &TriMesh, source: impl Fn(f64, f64) -> f64, bc: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let nv = mesh.verts.len();
    let mut k = vec![std::collections::BTreeMap::<usize, f64>::new(); nv];
    let mut b = vec![0.0f64; nv];

    for t in &mesh.tris {
        let p = [mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]];
        // 2·area (signed) and the barycentric-gradient coefficients (b_i, c_i).
        let two_a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let area = two_a.abs() * 0.5;
        let bc_ = [
            (p[1][1] - p[2][1], p[2][0] - p[1][0]),
            (p[2][1] - p[0][1], p[0][0] - p[2][0]),
            (p[0][1] - p[1][1], p[1][0] - p[0][0]),
        ];
        // element stiffness K_ij = (b_i b_j + c_i c_j) / (4·area); lumped load (area/3)·S(vertex_i).
        for a in 0..3 {
            for bb in 0..3 {
                let ke = (bc_[a].0 * bc_[bb].0 + bc_[a].1 * bc_[bb].1) / (4.0 * area);
                *k[t[a]].entry(t[bb]).or_insert(0.0) += ke;
            }
            b[t[a]] += area / 3.0 * source(p[a][0], p[a][1]);
        }
    }

    // Dirichlet elimination: move known boundary values to the RHS, then pin those rows.
    let mut phi_bc = vec![0.0f64; nv];
    for v in 0..nv {
        if mesh.boundary[v] {
            phi_bc[v] = bc(mesh.verts[v][0], mesh.verts[v][1]);
        }
    }
    for v in 0..nv {
        if mesh.boundary[v] {
            continue;
        }
        // subtract K[v][d]·φ_d for boundary neighbours d
        let row: Vec<(usize, f64)> = k[v].iter().map(|(&c, &val)| (c, val)).collect();
        for (c, val) in row {
            if mesh.boundary[c] {
                b[v] -= val * phi_bc[c];
            }
        }
    }

    // Build the interior SPD system with a compact interior numbering.
    let mut interior = Vec::new();
    let mut idx_of = vec![usize::MAX; nv];
    for v in 0..nv {
        if !mesh.boundary[v] {
            idx_of[v] = interior.len();
            interior.push(v);
        }
    }
    let ni = interior.len();
    let mut trips: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for (ri, &v) in interior.iter().enumerate() {
        for (&c, &val) in &k[v] {
            if !mesh.boundary[c] {
                let ci = idx_of[c];
                if ci <= ri {
                    trips.push(Triplet::new(ri, ci, val)); // lower triangle
                }
            }
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(ni, ni, &trips).expect("assemble");
    let llt = mat.sp_cholesky(Side::Lower).expect("SPD");
    let mut rhs = Mat::<f64>::zeros(ni, 1);
    for (ri, &v) in interior.iter().enumerate() {
        rhs[(ri, 0)] = b[v];
    }
    llt.solve_in_place(rhs.as_mut());

    let mut phi = phi_bc;
    for (ri, &v) in interior.iter().enumerate() {
        phi[v] = rhs[(ri, 0)];
    }
    phi
}

#[cfg(test)]
mod verification {
    use super::*;
    use std::f64::consts::PI;

    /// L2 error of the manufactured solution `φ* = sin(πx)sin(πy)` on an `n×n` triangulation with a
    /// given jitter. Source `S = −∇²φ* = 2π²φ*`, Dirichlet BC from `φ*`.
    fn mms_error(n: usize, jitter: f64) -> f64 {
        let mesh = TriMesh::unit_square(n, jitter);
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        let phi = solve_poisson(&mesh, |x, y| 2.0 * PI * PI * exact(x, y), exact);
        let mut se = 0.0;
        for v in 0..mesh.verts.len() {
            let (x, y) = (mesh.verts[v][0], mesh.verts[v][1]);
            se += (phi[v] - exact(x, y)).powi(2);
        }
        (se / mesh.verts.len() as f64).sqrt()
    }

    /// Second-order convergence on a regular triangulation (MMS, the gold-standard CFD check).
    #[test]
    fn mms_second_order_on_regular_mesh() {
        let e17 = mms_error(17, 0.0);
        let e33 = mms_error(33, 0.0);
        let order = (e17 / e33).log2();
        eprintln!("unstructured FVM MMS (regular): e17 {e17:.3e}  e33 {e33:.3e}  order {order:.2}");
        assert!(order > 1.8, "not 2nd order on a regular mesh: {order}");
    }

    /// Second order survives on a JITTERED, genuinely irregular mesh — the solver handles arbitrary
    /// triangles, not just a disguised grid.
    #[test]
    fn mms_second_order_on_irregular_mesh() {
        let e17 = mms_error(17, 0.25);
        let e33 = mms_error(33, 0.25);
        let order = (e17 / e33).log2();
        eprintln!("unstructured FVM MMS (jittered): e17 {e17:.3e}  e33 {e33:.3e}  order {order:.2}");
        assert!(e17 < 5e-3, "irregular-mesh error too large: {e17}");
        assert!(order > 1.6, "convergence lost on an irregular mesh: {order}");
    }
}
