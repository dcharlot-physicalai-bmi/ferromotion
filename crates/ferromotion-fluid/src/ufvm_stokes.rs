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
                // Boundary flag deferred: see the incidence pass below. "Both endpoints are boundary vertices"
                // is NOT the same test as "this edge lies on the boundary".
                node_boundary.push(false);
                edge_id.insert(key, id);
                id
            }
        };
        let mut incidence: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &base.tris {
            let m12 = mid(t[1], t[2]);
            let m20 = mid(t[2], t[0]);
            let m01 = mid(t[0], t[1]);
            tri_nodes.push([t[0], t[1], t[2], m12, m20, m01]);
            for (a, b) in [(t[1], t[2]), (t[2], t[0]), (t[0], t[1])] {
                let key = if a < b { (a, b) } else { (b, a) };
                *incidence.entry(key).or_insert(0) += 1;
            }
        }
        // **A mid-edge node is on the boundary iff its edge belongs to exactly ONE triangle (2026-08-15).**
        //
        // This used to be `base.boundary[a] && base.boundary[b]` — both endpoints boundary vertices — which is a
        // different and strictly weaker condition. A diagonal that cuts a corner has both endpoints on the
        // boundary while lying in the interior, and on every `TriMesh::unit_square` mesh there are exactly two of
        // them: the corner-cutting diagonals `(0, 1-h)-(h, 1)` and `(1-h, 0)-(1, h)`. Their midpoints were flagged
        // boundary and then hard Dirichlet-pinned to `u_bc` evaluated at an interior point — `(0, 0)` under the
        // cavity BC — so two interior velocity dofs were *set* rather than solved, giving an `O(1)` velocity error
        // and a spurious element divergence at the two nodes nearest the top-left and bottom-right corners.
        //
        // Triangle incidence is the definition rather than a heuristic: an interior edge is shared by two
        // triangles, a boundary edge by one. It needs no geometry, no tolerance, and holds for any triangulation
        // rather than only for meshes whose diagonals happen to avoid the corners.
        for (key, count) in &incidence {
            if let Some(&id) = edge_id.get(key) {
                node_boundary[id] = *count == 1;
            }
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
#[allow(clippy::needless_range_loop, clippy::type_complexity, clippy::too_many_arguments)]
pub fn solve_oseen(
    mesh: &TriMesh,
    nu: f64,
    force: impl Fn(f64, f64) -> (f64, f64),
    u_bc: impl Fn(f64, f64) -> (f64, f64),
    p_pin: f64,
    adv_u: &[f64],
    adv_v: &[f64],
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

        // element viscous stiffness K_ab, Oseen convection C_ab (with the current advecting
        // velocity), divergence Bx/By, and element load.
        let mut kmat = [[0.0f64; 6]; 6];
        let mut cmat = [[0.0f64; 6]; 6];
        let mut bx = [[0.0f64; 6]; 3];
        let mut by = [[0.0f64; 6]; 3];
        let mut fload = [[0.0f64; 2]; 6];
        let advect = !adv_u.is_empty();
        for q in &qpts {
            let sh = p2_shape(*q);
            let gr = p2_grad(*q, g);
            let w = qw * area;
            // physical coordinates of the quadrature point
            let xy = [q[0] * p[0][0] + q[1] * p[1][0] + q[2] * p[2][0], q[0] * p[0][1] + q[1] * p[1][1] + q[2] * p[2][1]];
            let (fx, fy) = force(xy[0], xy[1]);
            // interpolate the advecting velocity at this quadrature point (P2)
            let (mut au, mut av) = (0.0, 0.0);
            if advect {
                for c in 0..6 {
                    au += adv_u[un[c]] * sh[c];
                    av += adv_v[un[c]] * sh[c];
                }
            }
            for a in 0..6 {
                for b in 0..6 {
                    kmat[a][b] += w * nu * (gr[a][0] * gr[b][0] + gr[a][1] * gr[b][1]);
                    if advect {
                        // C_ab = ∫ (u^k·∇N_b) N_a  — the Oseen (Picard-linearized) convection
                        cmat[a][b] += w * (au * gr[b][0] + av * gr[b][1]) * sh[a];
                    }
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
            // viscous + convection block (both components share the same scalar operator)
            for b in 0..6 {
                add(un[a], un[b], kmat[a][b] + cmat[a][b]);
                add(off_v + un[a], off_v + un[b], kmat[a][b] + cmat[a][b]);
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

/// Steady **Stokes** flow — the coupled solve with no advection (`solve_oseen` with an empty
/// advecting field). Velocity at P2 nodes, pressure at P1 vertices.
pub fn solve_stokes(
    mesh: &TriMesh,
    nu: f64,
    force: impl Fn(f64, f64) -> (f64, f64),
    u_bc: impl Fn(f64, f64) -> (f64, f64),
    p_pin: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    solve_oseen(mesh, nu, force, u_bc, p_pin, &[], &[])
}

/// Steady **incompressible Navier–Stokes** on the unstructured mesh by a Picard (Oseen) iteration:
/// start from the Stokes solution, then repeatedly freeze the advecting velocity at the current
/// iterate and re-solve the linear Oseen system, until the velocity update falls below `tol` (or
/// `max_iter` is hit). Returns `(u, v, p, iters)`. Convergent for moderate Reynolds numbers.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn solve_navier_stokes(
    mesh: &TriMesh,
    nu: f64,
    force: impl Fn(f64, f64) -> (f64, f64) + Copy,
    u_bc: impl Fn(f64, f64) -> (f64, f64) + Copy,
    p_pin: f64,
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let (mut u, mut v, mut p) = solve_stokes(mesh, nu, force, u_bc, p_pin);
    for it in 1..=max_iter {
        let (un, vn, pn) = solve_oseen(mesh, nu, force, u_bc, p_pin, &u, &v);
        let du: f64 = un.iter().zip(&u).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let scale: f64 = un.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        u = un;
        v = vn;
        p = pn;
        if du / scale < tol {
            return (u, v, p, it);
        }
    }
    (u, v, p, max_iter)
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

    /// **A mid-edge node is on the boundary iff its edge belongs to one triangle, not iff both endpoints are
    /// boundary vertices.** The two conditions differ on exactly the corner-cutting diagonals, and the weaker
    /// test pinned two strictly interior velocity dofs to a boundary value.
    #[test]
    fn only_edges_with_one_incident_triangle_are_boundary_nodes() {
        for n in [5, 9, 17] {
            let base = TriMesh::unit_square(n, 0.0);
            let p2 = P2Mesh::build(&base);
            let h = 1.0 / (n - 1) as f64;

            // Recount incidence independently of build(), so this is a check rather than a restatement.
            let mut inc: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
            for t in &base.tris {
                for (a, b) in [(t[1], t[2]), (t[2], t[0]), (t[0], t[1])] {
                    *inc.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
                }
            }
            let n_boundary_edges = inc.values().filter(|c| **c == 1).count();
            // A closed triangulated square has one boundary edge per boundary segment: 4*(n-1).
            assert_eq!(n_boundary_edges, 4 * (n - 1), "n={n}: boundary edge count");

            // The OLD rule — both endpoints boundary — would additionally flag the corner-cutting diagonals.
            let old_rule = inc
                .keys()
                .filter(|(a, b)| base.boundary[*a] && base.boundary[*b])
                .count();
            assert_eq!(
                old_rule,
                n_boundary_edges + 2,
                "n={n}: the old rule should over-count by exactly the two corner diagonals"
            );

            // Every flagged mid-edge node must lie ON a wall. This is the property that was violated: the two
            // over-counted midpoints sit at (h/2, 1-h/2) and (1-h/2, h/2), a half-cell inside the domain.
            let mut flagged_interior = Vec::new();
            for id in base.verts.len()..p2.n_unodes {
                if p2.node_boundary[id] {
                    let [x, y] = p2.coords[id];
                    let on_wall = x.abs() < 1e-12 || (x - 1.0).abs() < 1e-12 || y.abs() < 1e-12 || (y - 1.0).abs() < 1e-12;
                    if !on_wall {
                        flagged_interior.push([x, y]);
                    }
                }
            }
            assert!(
                flagged_interior.is_empty(),
                "n={n}: {} interior mid-edge node(s) flagged as boundary: {flagged_interior:?} — the corner \
                 diagonals are at ({:.5}, {:.5}) and ({:.5}, {:.5})",
                flagged_interior.len(),
                h / 2.0,
                1.0 - h / 2.0,
                1.0 - h / 2.0,
                h / 2.0
            );
        }
    }

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

    // Analytic velocity gradients of the manufactured u* (for the nonlinear convection source).
    fn grad_u(x: f64, y: f64) -> (f64, f64, f64, f64) {
        let dudx = PI * PI * (2.0 * PI * x).sin() * (2.0 * PI * y).sin();
        let dudy = 2.0 * PI * PI * (PI * x).sin().powi(2) * (2.0 * PI * y).cos();
        let dvdx = -2.0 * PI * PI * (2.0 * PI * x).cos() * (PI * y).sin().powi(2);
        let dvdy = -PI * PI * (2.0 * PI * x).sin() * (2.0 * PI * y).sin();
        (dudx, dudy, dvdx, dvdy)
    }
    // Full Navier–Stokes source: (u·∇)u − ν∇²u + ∇p.
    fn ns_force(nu: f64, x: f64, y: f64) -> (f64, f64) {
        let (u, v) = u_exact(x, y);
        let (dudx, dudy, dvdx, dvdy) = grad_u(x, y);
        let (sx, sy) = force(nu, x, y); // −ν∇²u + ∇p from the Stokes case
        (sx + u * dudx + v * dudy, sy + u * dvdx + v * dvdy)
    }

    /// **The full incompressible Navier–Stokes solve on the unstructured mesh.** The Picard/Oseen
    /// iteration converges from the Stokes solution to the manufactured NS solution, recovering the
    /// velocity at high order — the coupled Taylor–Hood solver is now a genuine NS solver, not just
    /// Stokes.
    #[test]
    fn navier_stokes_mms_converges() {
        let nu = 1.0; // moderate Reynolds ( |u*| ~ π, so Re ~ few ) — Picard converges
        let solve = |n: usize| -> (f64, usize) {
            let mesh = TriMesh::unit_square(n, 0.0);
            let coords = velocity_node_coords(&mesh);
            let (u, v, _, iters) = solve_navier_stokes(&mesh, nu, |x, y| ns_force(nu, x, y), u_exact, p_exact(0.0, 0.0), 1e-10, 30);
            let mut se = 0.0;
            for i in 0..coords.len() {
                let (ue, ve) = u_exact(coords[i][0], coords[i][1]);
                se += (u[i] - ue).powi(2) + (v[i] - ve).powi(2);
            }
            ((se / coords.len() as f64).sqrt(), iters)
        };
        let (e9, it9) = solve(9);
        let (e17, _) = solve(17);
        let order = (e9 / e17).log2();
        eprintln!("Navier–Stokes MMS velocity: e9 {e9:.3e} ({it9} Picard iters)  e17 {e17:.3e}  order {order:.2}");
        assert!(it9 < 30, "Picard did not converge: {it9} iters");
        assert!(e17 < 5e-3, "NS velocity error too large: {e17}");
        assert!(order > 2.3, "NS solve not high-order: {order}");
    }
}
