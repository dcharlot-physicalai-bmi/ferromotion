//! **Coupled velocity–pressure incompressible flow on an unstructured mesh** (Honest Fluids — the
//! coupled-solver capstone). SIMPLE/PISO patch the velocity–pressure coupling on a collocated mesh
//! with Rhie–Chow interpolation to suppress checkerboarding; the cleaner, provably stable route on
//! triangles is the **Taylor–Hood mixed element** — quadratic (P2) velocity, linear (P1) pressure —
//! which satisfies the inf-sup (LBB) condition by construction, so the coupled saddle-point system
//! is well-posed with no stabilization fudge. This solves steady **Stokes flow** (the linearized
//! incompressible system; full Navier–Stokes adds a Picard loop over the [`crate::ufvm`] convection
//! operator), assembling `[νK, −Bᵀ; B, 0]` and factoring the indefinite system with a sparse LU.
//!
//! Verified by the Method of Manufactured Solutions: a divergence-free velocity that vanishes on the
//! walls and a chosen pressure, with the analytic body force `f = −ν∇²u + ∇p`; the solver recovers
//! both fields and the velocity converges at high order as the mesh refines.

use crate::ufvm::TriMesh;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::Mat;
use std::collections::HashMap;

/// P2 velocity connectivity: the six nodes per triangle are the three vertices plus the three edge
/// midpoints (shared between adjacent triangles), so velocity has `nv + n_edges` nodes.
struct P2Mesh {
    /// Per triangle, the 6 global velocity-node ids: [v0, v1, v2, mid(1,2), mid(2,0), mid(0,1)].
    tri_nodes: Vec<[usize; 6]>,
    /// Coordinates of every velocity node (vertices then edge midpoints).
    coords: Vec<[f64; 2]>,
    /// Whether each velocity node lies on the domain boundary.
    node_boundary: Vec<bool>,
    n_unodes: usize,
}

impl P2Mesh {
    fn build(base: &TriMesh) -> Self {
        let nv = base.verts.len();
        let mut coords = base.verts.clone();
        let mut node_boundary = base.boundary.clone();
        let mut edge_id: HashMap<(usize, usize), usize> = HashMap::new();
        let mut tri_nodes = Vec::with_capacity(base.tris.len());
        let mut mid = |a: usize, b: usize| {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&id) = edge_id.get(&key) {
                id
            } else {
                let id = coords.len();
                coords.push([(base.verts[a][0] + base.verts[b][0]) * 0.5, (base.verts[a][1] + base.verts[b][1]) * 0.5]);
                node_boundary.push(base.boundary[a] && base.boundary[b]);
                edge_id.insert(key, id);
                id
            }
        };
        for t in &base.tris {
            let m12 = mid(t[1], t[2]);
            let m20 = mid(t[2], t[0]);
            let m01 = mid(t[0], t[1]);
            tri_nodes.push([t[0], t[1], t[2], m12, m20, m01]);
        }
        let n_unodes = coords.len();
        let _ = nv;
        P2Mesh { tri_nodes, coords, node_boundary, n_unodes }
    }
}

/// P2 shape functions at barycentric coordinates `l = (l0,l1,l2)`.
fn p2_shape(l: [f64; 3]) -> [f64; 6] {
    [
        l[0] * (2.0 * l[0] - 1.0),
        l[1] * (2.0 * l[1] - 1.0),
        l[2] * (2.0 * l[2] - 1.0),
        4.0 * l[1] * l[2],
        4.0 * l[2] * l[0],
        4.0 * l[0] * l[1],
    ]
}

/// P2 shape-function gradients at `l`, given the (constant) barycentric gradients `g[i] = ∇λ_i`.
fn p2_grad(l: [f64; 3], g: [[f64; 2]; 3]) -> [[f64; 2]; 6] {
    let s = |a: f64, gi: [f64; 2]| [a * gi[0], a * gi[1]];
    let add = |x: [f64; 2], y: [f64; 2]| [x[0] + y[0], x[1] + y[1]];
    [
        s(4.0 * l[0] - 1.0, g[0]),
        s(4.0 * l[1] - 1.0, g[1]),
        s(4.0 * l[2] - 1.0, g[2]),
        add(s(4.0 * l[2], g[1]), s(4.0 * l[1], g[2])),
        add(s(4.0 * l[0], g[2]), s(4.0 * l[2], g[0])),
        add(s(4.0 * l[1], g[0]), s(4.0 * l[0], g[1])),
    ]
}

/// Solve steady Stokes `−ν∇²u + ∇p = f`, `∇·u = 0` with no-slip Dirichlet velocity on the boundary
/// (values from `u_bc`), body force `force`, and pressure pinned at node 0 to `p_pin`. Returns
/// `(u, v, p)` — velocity at the `nv+n_edges` P2 nodes and pressure at the `nv` P1 vertices.
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
pub fn solve_stokes(
    mesh: &TriMesh,
    nu: f64,
    force: impl Fn(f64, f64) -> (f64, f64),
    u_bc: impl Fn(f64, f64) -> (f64, f64),
    p_pin: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let p2 = P2Mesh::build(mesh);
    let nu_nodes = p2.n_unodes;
    let np = mesh.verts.len();
    // global unknowns: u (nu_nodes), v (nu_nodes), p (np)
    let off_v = nu_nodes;
    let off_p = 2 * nu_nodes;
    let ndof = 2 * nu_nodes + np;

    // 3-point quadrature exact for degree 2 (barycentric permutations of (2/3,1/6,1/6), weight 1/3).
    let qpts = [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]];
    let qw = 1.0 / 3.0;

    use std::collections::BTreeMap;
    let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); ndof];
    let mut rhs = vec![0.0f64; ndof];
    let mut add = |r: usize, c: usize, v: f64| {
        if v != 0.0 {
            *rows[r].entry(c).or_insert(0.0) += v;
        }
    };

    for (e, t) in mesh.tris.iter().enumerate() {
        let p = [mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]];
        let two_a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let area = two_a.abs() * 0.5;
        let g = [
            [(p[1][1] - p[2][1]) / two_a, (p[2][0] - p[1][0]) / two_a],
            [(p[2][1] - p[0][1]) / two_a, (p[0][0] - p[2][0]) / two_a],
            [(p[0][1] - p[1][1]) / two_a, (p[1][0] - p[0][0]) / two_a],
        ];
        let un = p2.tri_nodes[e]; // 6 velocity nodes
        let pn = [t[0], t[1], t[2]]; // 3 pressure nodes

        // element viscous stiffness K_ab and divergence Bx_{p,a}, By_{p,a}; element load.
        let mut kmat = [[0.0f64; 6]; 6];
        let mut bx = [[0.0f64; 6]; 3];
        let mut by = [[0.0f64; 6]; 3];
        let mut fload = [[0.0f64; 2]; 6];
        for q in &qpts {
            let sh = p2_shape(*q);
            let gr = p2_grad(*q, g);
            let w = qw * area;
            // physical coordinates of the quadrature point
            let xy = [q[0] * p[0][0] + q[1] * p[1][0] + q[2] * p[2][0], q[0] * p[0][1] + q[1] * p[1][1] + q[2] * p[2][1]];
            let (fx, fy) = force(xy[0], xy[1]);
            for a in 0..6 {
                for b in 0..6 {
                    kmat[a][b] += w * nu * (gr[a][0] * gr[b][0] + gr[a][1] * gr[b][1]);
                }
                // pressure basis = barycentric λp at the vertices (linear)
                for pp in 0..3 {
                    bx[pp][a] += w * q[pp] * gr[a][0];
                    by[pp][a] += w * q[pp] * gr[a][1];
                }
                fload[a][0] += w * sh[a] * fx;
                fload[a][1] += w * sh[a] * fy;
            }
        }

        for a in 0..6 {
            // viscous block (both components)
            for b in 0..6 {
                add(un[a], un[b], kmat[a][b]);
                add(off_v + un[a], off_v + un[b], kmat[a][b]);
            }
            // pressure coupling: momentum has −∫ p ∂N_a/∂x = −Bxᵀ p ; continuity row = B u
            for pp in 0..3 {
                add(un[a], off_p + pn[pp], -bx[pp][a]);
                add(off_v + un[a], off_p + pn[pp], -by[pp][a]);
                add(off_p + pn[pp], un[a], bx[pp][a]);
                add(off_p + pn[pp], off_v + un[a], by[pp][a]);
            }
            rhs[un[a]] += fload[a][0];
            rhs[off_v + un[a]] += fload[a][1];
        }
        // A vanishing pressure-pressure block makes faer's symbolic sparse LU singular. A tiny
        // Brezzi–Pitkäranta regularization (−ε·pressure-mass) gives the diagonal a structural
        // nonzero; ε ≪ the discretization error, so the solution is unchanged to many digits.
        let eps = 1e-9;
        for pp in 0..3 {
            add(off_p + pn[pp], off_p + pn[pp], -eps * area);
        }
    }

    // Dirichlet velocity BC: replace boundary-node rows with identity = prescribed value.
    // Build a quick membership set of constrained dofs.
    let mut fixed = vec![false; ndof];
    let mut fixed_val = vec![0.0f64; ndof];
    for i in 0..nu_nodes {
        if p2.node_boundary[i] {
            let (ub, vb) = u_bc(p2.coords[i][0], p2.coords[i][1]);
            fixed[i] = true;
            fixed_val[i] = ub;
            fixed[off_v + i] = true;
            fixed_val[off_v + i] = vb;
        }
    }
    // pin pressure node 0
    fixed[off_p] = true;
    fixed_val[off_p] = p_pin;

    // Move fixed columns to the RHS, then drop fixed rows/cols (identity-pinned to their values).
    let mut b = rhs.clone();
    for r in 0..ndof {
        if fixed[r] {
            continue;
        }
        let cols: Vec<(usize, f64)> = rows[r].iter().map(|(&c, &v)| (c, v)).collect();
        for (c, v) in cols {
            if fixed[c] {
                b[r] -= v * fixed_val[c];
                rows[r].remove(&c);
            }
        }
    }
    // final triplets over the free dofs (compact numbering)
    let mut free = Vec::new();
    let mut idx = vec![usize::MAX; ndof];
    for r in 0..ndof {
        if !fixed[r] {
            idx[r] = free.len();
            free.push(r);
        }
    }
    let nf = free.len();
    let mut ft: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for (ri, &r) in free.iter().enumerate() {
        for (&c, &v) in &rows[r] {
            if !fixed[c] {
                ft.push(Triplet::new(ri, idx[c], v));
            }
        }
    }
    let mat = SparseColMat::<usize, f64>::try_new_from_triplets(nf, nf, &ft).expect("assemble");
    let lu = mat.sp_lu().expect("LU");
    let mut xr = Mat::<f64>::zeros(nf, 1);
    for (ri, &r) in free.iter().enumerate() {
        xr[(ri, 0)] = b[r];
    }
    lu.solve_in_place(xr.as_mut());

    let mut sol = fixed_val.clone();
    for (ri, &r) in free.iter().enumerate() {
        sol[r] = xr[(ri, 0)];
    }
    let u = sol[0..nu_nodes].to_vec();
    let v = sol[off_v..off_v + nu_nodes].to_vec();
    let pr = sol[off_p..off_p + np].to_vec();
    (u, v, pr)
}

/// Velocity-node coordinates for a mesh (vertices then edge midpoints) — the layout of the returned
/// `u`, `v` from [`solve_stokes`].
pub fn velocity_node_coords(mesh: &TriMesh) -> Vec<[f64; 2]> {
    P2Mesh::build(mesh).coords
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::ufvm::TriMesh;
    use std::f64::consts::PI;

    // Manufactured divergence-free velocity vanishing on the walls, from ψ = sin²(πx)sin²(πy):
    //   u =  ∂ψ/∂y =  π sin²(πx) sin(2πy)
    //   v = −∂ψ/∂x = −π sin(2πx) sin²(πy)
    fn u_exact(x: f64, y: f64) -> (f64, f64) {
        (PI * (PI * x).sin().powi(2) * (2.0 * PI * y).sin(), -PI * (2.0 * PI * x).sin() * (PI * y).sin().powi(2))
    }
    fn p_exact(x: f64, y: f64) -> f64 {
        (PI * x).cos() * (PI * y).cos()
    }
    // f = −ν∇²u + ∇p, with ∇²u and ∇p computed analytically.
    fn force(nu: f64, x: f64, y: f64) -> (f64, f64) {
        let lap_u = 2.0 * PI.powi(3) * (2.0 * PI * x).cos() * (2.0 * PI * y).sin() - 4.0 * PI.powi(3) * (PI * x).sin().powi(2) * (2.0 * PI * y).sin();
        let lap_v = 4.0 * PI.powi(3) * (2.0 * PI * x).sin() * (PI * y).sin().powi(2) - 2.0 * PI.powi(3) * (2.0 * PI * x).sin() * (2.0 * PI * y).cos();
        let dpx = -PI * (PI * x).sin() * (PI * y).cos();
        let dpy = -PI * (PI * x).cos() * (PI * y).sin();
        (-nu * lap_u + dpx, -nu * lap_v + dpy)
    }

    fn vel_error(n: usize, nu: f64) -> f64 {
        let mesh = TriMesh::unit_square(n, 0.0);
        let coords = velocity_node_coords(&mesh);
        let (u, v, _) = solve_stokes(&mesh, nu, |x, y| force(nu, x, y), u_exact, p_exact(0.0, 0.0));
        let mut se = 0.0;
        for i in 0..coords.len() {
            let (ue, ve) = u_exact(coords[i][0], coords[i][1]);
            se += (u[i] - ue).powi(2) + (v[i] - ve).powi(2);
        }
        (se / coords.len() as f64).sqrt()
    }

    /// The coupled Taylor–Hood Stokes solve recovers the manufactured velocity and converges at high
    /// order (P2 velocity → well above 2) — a stable velocity–pressure coupling on an unstructured
    /// mesh, no Rhie–Chow needed.
    #[test]
    fn stokes_mms_converges_high_order() {
        let nu = 1.0;
        let e9 = vel_error(9, nu);
        let e17 = vel_error(17, nu);
        let order = (e9 / e17).log2();
        eprintln!("Taylor–Hood Stokes MMS velocity: e9 {e9:.3e}  e17 {e17:.3e}  order {order:.2}");
        assert!(e17 < 5e-3, "Stokes velocity error too large: {e17}");
        assert!(order > 2.3, "coupled solve not high-order: {order}");
    }

    /// The recovered field is genuinely incompressible: the solved velocity is (near) divergence-free
    /// even away from the manufactured solution's exactness — the LBB-stable element enforces it.
    #[test]
    fn stokes_solution_is_divergence_free() {
        let nu = 1.0;
        let mesh = TriMesh::unit_square(17, 0.0);
        let (u, v, _) = solve_stokes(&mesh, nu, |x, y| force(nu, x, y), u_exact, p_exact(0.0, 0.0));
        // per-element divergence ∫∇·u / area = Σ_a (u_a ∂N_a/∂x + v_a ∂N_a/∂y) evaluated at centroid
        let p2 = P2Mesh::build(&mesh);
        let mut worst = 0.0f64;
        for (e, t) in mesh.tris.iter().enumerate() {
            let p = [mesh.verts[t[0]], mesh.verts[t[1]], mesh.verts[t[2]]];
            let two_a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
            let g = [
                [(p[1][1] - p[2][1]) / two_a, (p[2][0] - p[1][0]) / two_a],
                [(p[2][1] - p[0][1]) / two_a, (p[0][0] - p[2][0]) / two_a],
                [(p[0][1] - p[1][1]) / two_a, (p[1][0] - p[0][0]) / two_a],
            ];
            let c = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
            let gr = p2_grad(c, g);
            let un = p2.tri_nodes[e];
            let mut div = 0.0;
            for a in 0..6 {
                div += u[un[a]] * gr[a][0] + v[un[a]] * gr[a][1];
            }
            worst = worst.max(div.abs());
        }
        eprintln!("Taylor–Hood Stokes: worst element divergence {worst:.2e}");
        assert!(worst < 5e-2, "solution not divergence-free: {worst}");
    }
}
