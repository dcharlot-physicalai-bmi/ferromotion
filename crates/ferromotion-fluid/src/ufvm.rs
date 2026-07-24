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

/// The 3×3 element stiffness matrix of a triangle (geometry only; scaled by conductivity later).
fn element_ke(p: [[f64; 2]; 3]) -> [[f64; 3]; 3] {
    let two_a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
    let area = two_a.abs() * 0.5;
    let bc = [
        (p[1][1] - p[2][1], p[2][0] - p[1][0]),
        (p[2][1] - p[0][1], p[0][0] - p[2][0]),
        (p[0][1] - p[1][1], p[1][0] - p[0][0]),
    ];
    let mut ke = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            ke[a][b] = (bc[a].0 * bc[b].0 + bc[a].1 * bc[b].1) / (4.0 * area);
        }
    }
    ke
}

/// **Discrete adjoint on the unstructured mesh.** For `−∇·(k∇φ) = S` with homogeneous Dirichlet
/// boundaries and a per-element conductivity `k`, and objective `J = ½‖φ − target‖²` over interior
/// vertices, return `(J, ∂J/∂k)`. The operator `K(k)` is SPD and self-adjoint, so the adjoint solve
/// `K λ = (φ − target)` reuses the *same* factored Cholesky — one extra solve for the whole gradient
/// field, exactly as the MAC pressure-projection adjoint reused its factor.
#[allow(clippy::needless_range_loop)]
pub fn conductivity_gradient(mesh: &TriMesh, k_elem: &[f64], source: impl Fn(f64, f64) -> f64, target: &[f64]) -> (f64, Vec<f64>) {
    let nv = mesh.verts.len();
    let kes: Vec<[[f64; 3]; 3]> = mesh.tris.iter().map(|t| element_ke([mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]])).collect();

    // interior numbering
    let mut interior = Vec::new();
    let mut idx_of = vec![usize::MAX; nv];
    for v in 0..nv {
        if !mesh.boundary[v] {
            idx_of[v] = interior.len();
            interior.push(v);
        }
    }
    let ni = interior.len();

    // assemble K(k) over interior dofs + source load b
    use std::collections::BTreeMap;
    let mut rows = vec![BTreeMap::<usize, f64>::new(); ni];
    let mut b = vec![0.0f64; ni];
    for (e, t) in mesh.tris.iter().enumerate() {
        for a in 0..3 {
            if mesh.boundary[t[a]] {
                continue;
            }
            let ra = idx_of[t[a]];
            for bb in 0..3 {
                if mesh.boundary[t[bb]] {
                    continue; // homogeneous Dirichlet: boundary φ = 0, no elimination term
                }
                *rows[ra].entry(idx_of[t[bb]]).or_insert(0.0) += k_elem[e] * kes[e][a][bb];
            }
        }
    }
    for v in 0..nv {
        if !mesh.boundary[v] {
            // lumped source split across the element vertices happens per-element below
        }
    }
    for (e, t) in mesh.tris.iter().enumerate() {
        let p = [mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]];
        let two_a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let area = two_a.abs() * 0.5;
        let _ = e;
        for a in 0..3 {
            if !mesh.boundary[t[a]] {
                b[idx_of[t[a]]] += area / 3.0 * source(p[a][0], p[a][1]);
            }
        }
    }

    let mut trips: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for ri in 0..ni {
        for (&ci, &val) in &rows[ri] {
            if ci <= ri {
                trips.push(Triplet::new(ri, ci, val));
            }
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(ni, ni, &trips).expect("assemble");
    let llt = mat.sp_cholesky(Side::Lower).expect("SPD");

    // forward solve φ
    let mut rhs = Mat::<f64>::zeros(ni, 1);
    for ri in 0..ni {
        rhs[(ri, 0)] = b[ri];
    }
    llt.solve_in_place(rhs.as_mut());
    let mut phi = vec![0.0f64; nv];
    for ri in 0..ni {
        phi[interior[ri]] = rhs[(ri, 0)];
    }

    // objective + adjoint RHS (φ − target) on interior; self-adjoint ⇒ same factor
    let mut j = 0.0;
    let mut ar = Mat::<f64>::zeros(ni, 1);
    for ri in 0..ni {
        let d = phi[interior[ri]] - target[interior[ri]];
        j += 0.5 * d * d;
        ar[(ri, 0)] = d;
    }
    llt.solve_in_place(ar.as_mut());
    let mut lam = vec![0.0f64; nv];
    for ri in 0..ni {
        lam[interior[ri]] = ar[(ri, 0)];
    }

    // gradient: ∂J/∂k[e] = −λᵀ (∂K/∂k[e]) φ = −Σ_ab λ[t_a] Ke[a][b] φ[t_b]  (boundary λ=φ=0)
    let mut grad = vec![0.0f64; mesh.tris.len()];
    for (e, t) in mesh.tris.iter().enumerate() {
        let mut g = 0.0;
        for a in 0..3 {
            for bb in 0..3 {
                g += lam[t[a]] * kes[e][a][bb] * phi[t[bb]];
            }
        }
        grad[e] = -g;
    }
    (j, grad)
}

/// Solve the steady **advection–diffusion** transport equation `u·∇φ − D∇²φ = S` on the
/// unstructured mesh, with a divergence-free velocity field `vel` and Dirichlet boundaries. This is
/// the scalar-transport workhorse that unstructured incompressible NS (SIMPLE/PISO) is built on: the
/// Galerkin convection matrix `∫(u·∇λ_j)λ_i` is added to the diffusion stiffness, giving a
/// NON-symmetric system solved with a sparse LU (Cholesky no longer applies once advection enters).
/// Stable and 2nd-order in the resolved (cell-Péclet ≲ 1) regime.
#[allow(clippy::needless_range_loop)]
pub fn solve_advection_diffusion(
    mesh: &TriMesh,
    vel: impl Fn(f64, f64) -> (f64, f64),
    d: f64,
    source: impl Fn(f64, f64) -> f64,
    bc: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let nv = mesh.verts.len();
    use std::collections::BTreeMap;
    let mut rows = vec![BTreeMap::<usize, f64>::new(); nv];
    let mut b = vec![0.0f64; nv];

    for t in &mesh.tris {
        let p = [mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]];
        let two_a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let area = two_a.abs() * 0.5;
        // barycentric gradients ∇λ_j = (bj, cj)/(2A)
        let bc_ = [
            (p[1][1] - p[2][1], p[2][0] - p[1][0]),
            (p[2][1] - p[0][1], p[0][0] - p[2][0]),
            (p[0][1] - p[1][1], p[1][0] - p[0][0]),
        ];
        let grad = |j: usize| (bc_[j].0 / two_a, bc_[j].1 / two_a); // signed 2A cancels the abs sign
        let cen = [(p[0][0] + p[1][0] + p[2][0]) / 3.0, (p[0][1] + p[1][1] + p[2][1]) / 3.0];
        let (uc, vc) = vel(cen[0], cen[1]);
        for a in 0..3 {
            for j in 0..3 {
                let (gjx, gjy) = grad(j);
                let (gix, giy) = grad(a);
                // diffusion D·∇λ_a·∇λ_j·area  +  convection (u·∇λ_j)·∫λ_a  (∫λ_a = area/3)
                let diff = d * (gix * gjx + giy * gjy) * area;
                let conv = (uc * gjx + vc * gjy) * (area / 3.0);
                *rows[t[a]].entry(t[j]).or_insert(0.0) += diff + conv;
            }
            b[t[a]] += area / 3.0 * source(p[a][0], p[a][1]);
        }
    }

    // Dirichlet: pin boundary rows to the exact value, move their columns to the RHS.
    let mut phi = vec![0.0f64; nv];
    for v in 0..nv {
        if mesh.boundary[v] {
            phi[v] = bc(mesh.verts[v][0], mesh.verts[v][1]);
        }
    }
    for v in 0..nv {
        if mesh.boundary[v] {
            continue;
        }
        let cols: Vec<(usize, f64)> = rows[v].iter().map(|(&c, &val)| (c, val)).collect();
        for (c, val) in cols {
            if mesh.boundary[c] {
                b[v] -= val * phi[c];
            }
        }
    }
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
        for (&c, &val) in &rows[v] {
            if !mesh.boundary[c] {
                trips.push(Triplet::new(ri, idx_of[c], val)); // full (non-symmetric) matrix
            }
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(ni, ni, &trips).expect("assemble");
    let lu = mat.sp_lu().expect("LU");
    let mut rhs = Mat::<f64>::zeros(ni, 1);
    for (ri, &v) in interior.iter().enumerate() {
        rhs[(ri, 0)] = b[v];
    }
    lu.solve_in_place(rhs.as_mut());
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

    /// The discrete adjoint gradient of the objective w.r.t. the per-element conductivity, checked
    /// against central finite differences on a jittered mesh — the exact gradient through the
    /// assembled sparse solve, at O(1) solves regardless of the number of conductivities.
    #[test]
    fn conductivity_adjoint_matches_fd() {
        let mesh = TriMesh::unit_square(13, 0.2);
        let ne = mesh.tris.len();
        let k: Vec<f64> = (0..ne).map(|e| 1.0 + 0.3 * ((e % 5) as f64)).collect(); // heterogeneous
        let target = vec![0.0; mesh.verts.len()]; // J = ½‖φ‖²
        let src = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();

        let (_, grad) = conductivity_gradient(&mesh, &k, src, &target);

        // FD on a handful of elements spread across the mesh.
        let eps = 1e-6;
        let mut worst = 0.0f64;
        for &e in &[0usize, ne / 3, ne / 2, 2 * ne / 3, ne - 1] {
            let mut kp = k.clone();
            kp[e] += eps;
            let mut km = k.clone();
            km[e] -= eps;
            let jp = conductivity_gradient(&mesh, &kp, src, &target).0;
            let jm = conductivity_gradient(&mesh, &km, src, &target).0;
            let fd = (jp - jm) / (2.0 * eps);
            let rel = (grad[e] - fd).abs() / fd.abs().max(1e-9);
            worst = worst.max(rel);
        }
        eprintln!("unstructured adjoint dJ/dk vs FD: worst rel {worst:.2e}");
        assert!(worst < 1e-5, "discrete adjoint gradient off: {worst}");
    }

    /// Steady advection–diffusion on the unstructured mesh, MMS-verified at 2nd order with a
    /// divergence-free swirl velocity — the transport equation SIMPLE/PISO builds on, on triangles.
    #[test]
    fn advection_diffusion_mms_second_order() {
        let d = 0.1;
        // divergence-free velocity from ψ = sin(πx)sin(πy): u = ∂ψ/∂y, v = −∂ψ/∂x.
        let vel = |x: f64, y: f64| (PI * (PI * x).sin() * (PI * y).cos(), -PI * (PI * x).cos() * (PI * y).sin());
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();
        // S = u·∇φ* − D∇²φ* ; ∇²φ* = −2π²φ* ; ∇φ* = (π cos(πx)sin(πy), π sin(πx)cos(πy))
        let source = |x: f64, y: f64| {
            let (u, v) = vel(x, y);
            let gx = PI * (PI * x).cos() * (PI * y).sin();
            let gy = PI * (PI * x).sin() * (PI * y).cos();
            (u * gx + v * gy) + d * 2.0 * PI * PI * exact(x, y)
        };
        let err = |n: usize| -> f64 {
            let mesh = TriMesh::unit_square(n, 0.15);
            let phi = solve_advection_diffusion(&mesh, vel, d, source, exact);
            let mut se = 0.0;
            for v in 0..mesh.verts.len() {
                se += (phi[v] - exact(mesh.verts[v][0], mesh.verts[v][1])).powi(2);
            }
            (se / mesh.verts.len() as f64).sqrt()
        };
        let e17 = err(17);
        let e33 = err(33);
        let order = (e17 / e33).log2();
        eprintln!("unstructured advection–diffusion MMS: e17 {e17:.3e}  e33 {e33:.3e}  order {order:.2}");
        assert!(order > 1.7, "advection–diffusion not 2nd order: {order}");
    }
}
