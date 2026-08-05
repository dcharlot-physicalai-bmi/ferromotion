//! ferromotion-cloth — a **differentiable FEM thin-shell cloth** solver (StVK membrane + bending),
//! the third material domain alongside the Aquarium fluid and MPM crates (in the spirit of
//! Diffclothai / DaXBench).
//!
//! The cloth is a triangle mesh. Each triangle is a **constant-strain membrane element** with a
//! St. Venant–Kirchhoff material: from the deformation gradient `F = Ds·Dm⁻¹` (3×2) come the Green
//! strain `E = ½(FᵀF − I)`, the 2nd-Piola–Kirchhoff stress `S = 2μE + λ tr(E) I`, the 1st-PK stress
//! `P = F·S`, and the exact nodal forces `H = −A₀·P·Dm⁻ᵀ`. Out-of-plane stiffness comes from bending
//! springs across quad diagonals. Everything is a smooth function of the vertex positions, so the
//! solver is differentiable — and because `S` is **linear in the Lamé parameters**, an outcome's
//! gradient w.r.t. the material stiffness `μ` is exact and available in closed form (verified against
//! finite differences to machine precision). Pure `nalgebra` → WASM-clean.

pub mod vbd;

use nalgebra::{Matrix2, Vector3};

/// Closest point on triangle `abc` to point `p`, with its barycentric weights `(wa, wb, wc)` (Ericson,
/// *Real-Time Collision Detection*). `wa·a + wb·b + wc·c` is the returned point; the weights let a
/// contact gradient distribute cleanly onto the three triangle vertices.
pub fn closest_point_on_triangle(p: Vector3<f64>, a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> (Vector3<f64>, (f64, f64, f64)) {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(&ap), ac.dot(&ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, (1.0, 0.0, 0.0));
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(&bp), ac.dot(&bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, (0.0, 1.0, 0.0));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + v * ab, (1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(&cp), ac.dot(&cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, (0.0, 0.0, 1.0));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + w * ac, (1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + w * (c - b), (0.0, 1.0 - w, w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, (1.0 - v - w, v, w))
}

/// Closest distance between segments `[p1,q1]` and `[p2,q2]` → `(dist, s, t)` with closest points
/// `c1 = p1+s(q1−p1)`, `c2 = p2+t(q2−p2)` — the edge-edge primitive for cloth self-collision.
pub fn segment_segment_closest(p1: Vector3<f64>, q1: Vector3<f64>, p2: Vector3<f64>, q2: Vector3<f64>) -> (f64, f64, f64) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(&d1), d2.dot(&d2), d2.dot(&r));
    const EPS: f64 = 1e-14;
    let (mut s, t);
    if a <= EPS && e <= EPS {
        return ((p1 - p2).norm(), 0.0, 0.0);
    } else if a <= EPS {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(&r);
        if e <= EPS {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(&d2);
            let denom = a * e - b * b;
            s = if denom.abs() > EPS { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut tt = (b * s + f) / e;
            if tt < 0.0 {
                tt = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if tt > 1.0 {
                tt = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            t = tt;
        }
    }
    let c1 = p1 + s * d1;
    let c2 = p2 + t * d2;
    ((c1 - c2).norm(), s, t)
}

/// A triangle-mesh cloth with a StVK membrane and bending springs.
#[derive(Clone)]
pub struct ClothSim {
    pub x: Vec<Vector3<f64>>,
    pub v: Vec<Vector3<f64>>,
    pub pinned: Vec<bool>,
    tris: Vec<[usize; 3]>,
    dm_inv: Vec<Matrix2<f64>>,
    area: Vec<f64>,
    /// Bending springs `(i, j, rest length)`.
    bends: Vec<(usize, usize, f64)>,
    pub mass: f64,
    pub mu: f64,
    pub lambda: f64,
    pub k_bend: f64,
    pub damping: f64,
    pub dt: f64,
    pub gravity: Vector3<f64>,
}

impl ClothSim {
    /// A flat rectangular cloth of `nx × ny` vertices with spacing `h`, in the x–y plane.
    pub fn grid(nx: usize, ny: usize, h: f64, mass: f64, mu: f64, lambda: f64, k_bend: f64, dt: f64) -> Self {
        let idx = |i: usize, j: usize| j * nx + i;
        let mut x = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                x.push(Vector3::new(i as f64 * h, j as f64 * h, 0.0));
            }
        }
        let n = x.len();
        let mut tris = Vec::new();
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                tris.push([idx(i, j), idx(i + 1, j), idx(i, j + 1)]);
                tris.push([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        // Rest membrane matrices (material coords = the flat rest positions' x,y).
        let (mut dm_inv, mut area) = (Vec::new(), Vec::new());
        for t in &tris {
            let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
            let dm = Matrix2::new(b.x - a.x, c.x - a.x, b.y - a.y, c.y - a.y);
            area.push(0.5 * dm.determinant().abs());
            dm_inv.push(dm.try_inverse().expect("degenerate rest triangle"));
        }
        // Bending springs across quad diagonals (vertices two apart).
        let mut bends = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                if i + 2 < nx {
                    bends.push((idx(i, j), idx(i + 2, j), 2.0 * h));
                }
                if j + 2 < ny {
                    bends.push((idx(i, j), idx(i, j + 2), 2.0 * h));
                }
            }
        }
        Self { x, v: vec![Vector3::zeros(); n], pinned: vec![false; n], tris, dm_inv, area, bends, mass, mu, lambda, k_bend, damping: 0.0, dt, gravity: Vector3::new(0.0, 0.0, -9.81) }
    }

    /// Elastic (membrane + bending) potential energy at positions `x`.
    pub fn elastic_energy(&self, x: &[Vector3<f64>]) -> f64 {
        let mut w = 0.0;
        for (ti, t) in self.tris.iter().enumerate() {
            let e = self.green_strain(x, ti, t);
            let tr = e[(0, 0)] + e[(1, 1)];
            let fro = e.iter().map(|v| v * v).sum::<f64>();
            w += self.area[ti] * (self.mu * fro + 0.5 * self.lambda * tr * tr);
        }
        for &(i, j, l0) in &self.bends {
            let d = (x[i] - x[j]).norm() - l0;
            w += 0.5 * self.k_bend * d * d;
        }
        w
    }

    fn green_strain(&self, x: &[Vector3<f64>], ti: usize, t: &[usize; 3]) -> Matrix2<f64> {
        let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
        let ds = nalgebra::Matrix3x2::from_columns(&[b - a, c - a]);
        let f = ds * self.dm_inv[ti]; // 3×2 deformation gradient
        0.5 * (f.transpose() * f - Matrix2::identity())
    }

    /// Elastic forces (membrane + bending) at positions `x`. If `d_dmu`, also returns `∂force/∂μ`.
    fn elastic_forces(&self, x: &[Vector3<f64>], d_dmu: bool) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
        let n = x.len();
        let (mut f, mut df) = (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
        for (ti, t) in self.tris.iter().enumerate() {
            let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
            let ds = nalgebra::Matrix3x2::from_columns(&[b - a, c - a]);
            let fdef = ds * self.dm_inv[ti];
            let e = 0.5 * (fdef.transpose() * fdef - Matrix2::identity());
            let tr = e[(0, 0)] + e[(1, 1)];
            let s = 2.0 * self.mu * e + self.lambda * tr * Matrix2::identity(); // 2nd PK
            let p = fdef * s; // 1st PK (3×2)
            let h = -self.area[ti] * p * self.dm_inv[ti].transpose(); // 3×2 → forces on b, c
            let (f1, f2) = (h.column(0).into_owned(), h.column(1).into_owned());
            f[t[1]] += f1;
            f[t[2]] += f2;
            f[t[0]] -= f1 + f2;
            if d_dmu {
                // S is linear in μ: ∂S/∂μ = 2E ⇒ ∂P/∂μ = F·2E ⇒ ∂force/∂μ from ∂H/∂μ.
                let dp = fdef * (2.0 * e);
                let dh = -self.area[ti] * dp * self.dm_inv[ti].transpose();
                let (d1, d2) = (dh.column(0).into_owned(), dh.column(1).into_owned());
                df[t[1]] += d1;
                df[t[2]] += d2;
                df[t[0]] -= d1 + d2;
            }
        }
        for &(i, j, l0) in &self.bends {
            let d = x[i] - x[j];
            let len = d.norm();
            if len > 1e-12 {
                let fb = -self.k_bend * (len - l0) * (d / len);
                f[i] += fb;
                f[j] -= fb;
            }
        }
        (f, df)
    }

    /// Total force per vertex (elastic + gravity + damping).
    fn total_force(&self, x: &[Vector3<f64>], v: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let (mut f, _) = self.elastic_forces(x, false);
        for i in 0..x.len() {
            f[i] += self.mass * self.gravity - self.damping * v[i];
        }
        f
    }

    /// Advance one timestep (semi-implicit Euler); pinned vertices are held fixed.
    pub fn step(&mut self) {
        let f = self.total_force(&self.x, &self.v);
        for i in 0..self.x.len() {
            if self.pinned[i] {
                self.v[i] = Vector3::zeros();
                continue;
            }
            self.v[i] += self.dt * f[i] / self.mass;
            self.x[i] += self.dt * self.v[i];
        }
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.v.iter().map(|v| v.norm_squared()).sum::<f64>()
    }

    // --------------------------------------------------------------------------------------------
    // Self-collision — the physics of a cloth that folds without passing through itself. Two
    // proximity modes: a vertex near a non-incident triangle (VT), and two non-adjacent edges near
    // each other (EE). Each is a one-sided penalty ½k(h−d)² within cloth thickness h, differentiable
    // (force = −∇E), split to the involved vertices by the closest-point weights. Same as the rod's
    // self-contact, lifted to a triangle mesh.
    // --------------------------------------------------------------------------------------------

    /// The mesh triangles (each `[i, j, k]` vertex indices).
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.tris
    }

    /// Each vertex's 1-ring (mesh-adjacent vertices). Self-collision skips topological neighbors so a
    /// FLAT sheet does not spuriously collide with itself: coplanar non-incident elements within a
    /// vertex's neighborhood sit at ~zero distance and would otherwise fire the proximity penalty.
    fn vertex_neighbors(&self) -> Vec<std::collections::HashSet<usize>> {
        let n = self.x.len();
        let mut nbr = vec![std::collections::HashSet::new(); n];
        for t in &self.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                nbr[a].insert(b);
                nbr[b].insert(a);
            }
        }
        nbr
    }

    /// Unique undirected mesh edges `(i, j)` with `i < j`.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        use std::collections::HashSet;
        let mut set: HashSet<(usize, usize)> = HashSet::new();
        for t in &self.tris {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                set.insert((a.min(b), a.max(b)));
            }
        }
        let mut e: Vec<_> = set.into_iter().collect();
        e.sort_unstable();
        e
    }

    /// One-sided self-collision energy at `x`: `½k(h−d)²` over every vertex/non-incident-triangle pair
    /// and every non-adjacent edge/edge pair closer than the thickness `h`. Zero when the sheet is clear.
    pub fn self_collision_energy(&self, x: &[Vector3<f64>], thickness: f64, k: f64) -> f64 {
        let nbr = self.vertex_neighbors();
        let vt_far = |p: usize, t: &[usize; 3]| !t.contains(&p) && !t.iter().any(|&v| nbr[p].contains(&v));
        let ee_far = |e0: (usize, usize), e1: (usize, usize)| {
            let vs = [e0.0, e0.1, e1.0, e1.1];
            e0.0 != e1.0 && e0.0 != e1.1 && e0.1 != e1.0 && e0.1 != e1.1 && !nbr[e0.0].contains(&e1.0) && !nbr[e0.0].contains(&e1.1) && !nbr[e0.1].contains(&e1.0) && !nbr[e0.1].contains(&e1.1) && vs.iter().all(|v| *v < x.len())
        };
        let mut w = 0.0;
        for (p, &pt) in x.iter().enumerate() {
            for t in &self.tris {
                if !vt_far(p, t) {
                    continue;
                }
                let (cp, _) = closest_point_on_triangle(pt, x[t[0]], x[t[1]], x[t[2]]);
                let d = (pt - cp).norm();
                if d < thickness {
                    w += 0.5 * k * (thickness - d).powi(2);
                }
            }
        }
        let edges = self.edges();
        for a in 0..edges.len() {
            for b in (a + 1)..edges.len() {
                if !ee_far(edges[a], edges[b]) {
                    continue;
                }
                let (e0, e1) = (edges[a], edges[b]);
                let (d, ..) = segment_segment_closest(x[e0.0], x[e0.1], x[e1.0], x[e1.1]);
                if d < thickness {
                    w += 0.5 * k * (thickness - d).powi(2);
                }
            }
        }
        w
    }

    /// Add self-collision repulsion to `force`: the exact `−∇` of [`self_collision_energy`], with each
    /// contact force `k(h−d)·n̂` split to the involved vertices by the closest-point barycentric weights.
    pub fn accumulate_self_collision_forces(&self, x: &[Vector3<f64>], thickness: f64, k: f64, force: &mut [Vector3<f64>]) {
        let nbr = self.vertex_neighbors();
        let vt_far = |p: usize, t: &[usize; 3]| !t.contains(&p) && !t.iter().any(|&v| nbr[p].contains(&v));
        let ee_far = |e0: (usize, usize), e1: (usize, usize)| {
            e0.0 != e1.0 && e0.0 != e1.1 && e0.1 != e1.0 && e0.1 != e1.1 && !nbr[e0.0].contains(&e1.0) && !nbr[e0.0].contains(&e1.1) && !nbr[e0.1].contains(&e1.0) && !nbr[e0.1].contains(&e1.1)
        };
        // vertex ↔ triangle
        for (p, &pt) in x.iter().enumerate() {
            for t in &self.tris {
                if !vt_far(p, t) {
                    continue;
                }
                let (cp, (wa, wb, wc)) = closest_point_on_triangle(pt, x[t[0]], x[t[1]], x[t[2]]);
                let d = (pt - cp).norm();
                if d < thickness && d > 1e-12 {
                    let n = (pt - cp) / d; // from triangle toward the vertex
                    let f = k * (thickness - d) * n;
                    force[p] += f;
                    force[t[0]] -= wa * f;
                    force[t[1]] -= wb * f;
                    force[t[2]] -= wc * f;
                }
            }
        }
        // edge ↔ edge
        let edges = self.edges();
        for a in 0..edges.len() {
            for b in (a + 1)..edges.len() {
                let (e0, e1) = (edges[a], edges[b]);
                if !ee_far(e0, e1) {
                    continue;
                }
                let (d, s, t) = segment_segment_closest(x[e0.0], x[e0.1], x[e1.0], x[e1.1]);
                if d < thickness && d > 1e-12 {
                    let c1 = x[e0.0] + s * (x[e0.1] - x[e0.0]);
                    let c2 = x[e1.0] + t * (x[e1.1] - x[e1.0]);
                    let n = (c1 - c2) / d;
                    let f = k * (thickness - d) * n;
                    force[e0.0] += (1.0 - s) * f;
                    force[e0.1] += s * f;
                    force[e1.0] -= (1.0 - t) * f;
                    force[e1.1] -= t * f;
                }
            }
        }
    }

    /// Broadphase: candidate `(vertex, triangle)` and `(edge_a, edge_b)` index pairs whose thickness-
    /// inflated AABBs overlap — via a uniform spatial hash of the triangles (`cell` ≥ `thickness`). The
    /// narrow-phase penalty then filters these; verified to miss no true contact.
    #[allow(clippy::type_complexity)]
    pub fn self_collision_candidates(&self, x: &[Vector3<f64>], thickness: f64, cell: f64) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        use std::collections::{HashMap, HashSet};
        let nbr = self.vertex_neighbors();
        let vt_far = |p: usize, t: &[usize; 3]| !t.contains(&p) && !t.iter().any(|&v| nbr[p].contains(&v));
        let ee_far = |e0: (usize, usize), e1: (usize, usize)| {
            e0.0 != e1.0 && e0.0 != e1.1 && e0.1 != e1.0 && e0.1 != e1.1 && !nbr[e0.0].contains(&e1.0) && !nbr[e0.0].contains(&e1.1) && !nbr[e0.1].contains(&e1.0) && !nbr[e0.1].contains(&e1.1)
        };
        let key = |p: Vector3<f64>| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
        let tri_aabb = |t: &[usize; 3]| {
            let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
            (a.inf(&b).inf(&c) - Vector3::repeat(thickness), a.sup(&b).sup(&c) + Vector3::repeat(thickness))
        };
        // bucket triangles into every cell their inflated AABB touches
        let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (ti, t) in self.tris.iter().enumerate() {
            let (lo, hi) = tri_aabb(t);
            let (kl, kh) = (key(lo), key(hi));
            for cx in kl.0..=kh.0 {
                for cy in kl.1..=kh.1 {
                    for cz in kl.2..=kh.2 {
                        grid.entry((cx, cy, cz)).or_default().push(ti);
                    }
                }
            }
        }
        // vertex→cell lookup: a vertex is a candidate against triangles sharing any of its cells
        let mut vt: HashSet<(usize, usize)> = HashSet::new();
        for (p, &pt) in x.iter().enumerate() {
            if let Some(tris) = grid.get(&key(pt)) {
                for &ti in tris {
                    if vt_far(p, &self.tris[ti]) {
                        vt.insert((p, ti));
                    }
                }
            }
        }
        // edges of triangles co-located in a cell are edge-edge candidates
        let edges = self.edges();
        let mut edge_of: HashMap<(usize, usize), usize> = HashMap::new();
        for (i, &e) in edges.iter().enumerate() {
            edge_of.insert(e, i);
        }
        let tri_edges = |t: &[usize; 3]| [(t[0].min(t[1]), t[0].max(t[1])), (t[1].min(t[2]), t[1].max(t[2])), (t[2].min(t[0]), t[2].max(t[0]))];
        let mut ee: HashSet<(usize, usize)> = HashSet::new();
        for tris in grid.values() {
            for a in 0..tris.len() {
                for b in (a + 1)..tris.len() {
                    for ea in tri_edges(&self.tris[tris[a]]) {
                        for eb in tri_edges(&self.tris[tris[b]]) {
                            if !ee_far(ea, eb) {
                                continue;
                            }
                            let (ia, ib) = (edge_of[&ea], edge_of[&eb]);
                            ee.insert((ia.min(ib), ia.max(ib)));
                        }
                    }
                }
            }
        }
        let mut vt: Vec<_> = vt.into_iter().collect();
        let mut ee: Vec<_> = ee.into_iter().collect();
        vt.sort_unstable();
        ee.sort_unstable();
        (vt, ee)
    }

    /// One semi-implicit step with self-collision folded into the forces (thickness `h`, stiffness `k`).
    pub fn step_self_collision(&mut self, thickness: f64, k: f64) {
        let mut f = self.total_force(&self.x, &self.v);
        self.accumulate_self_collision_forces(&self.x, thickness, k, &mut f);
        for i in 0..self.x.len() {
            if self.pinned[i] {
                self.v[i] = Vector3::zeros();
                continue;
            }
            self.v[i] += self.dt * f[i] / self.mass;
            self.x[i] += self.dt * self.v[i];
        }
    }

    /// Total kinetic energy after one step, and its exact `∂/∂μ` (the elastic stress is linear in μ).
    pub fn ke_and_dke_dmu(&self) -> (f64, f64) {
        let (fe, dfe) = self.elastic_forces(&self.x, true);
        let (mut ke, mut dke) = (0.0, 0.0);
        for i in 0..self.x.len() {
            if self.pinned[i] {
                continue;
            }
            let f = fe[i] + self.mass * self.gravity - self.damping * self.v[i];
            let vn = self.v[i] + self.dt * f / self.mass;
            let dvn = self.dt * dfe[i] / self.mass; // gravity/damping are μ-independent
            ke += 0.5 * self.mass * vn.norm_squared();
            dke += self.mass * vn.dot(&dvn);
        }
        (ke, dke)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cloth() -> ClothSim {
        ClothSim::grid(4, 4, 0.1, 0.02, 500.0, 300.0, 20.0, 1e-3)
    }

    #[test]
    fn membrane_force_is_the_exact_energy_gradient() {
        // Perturb the cloth out of plane, then check force == −∂(elastic energy)/∂x by finite diff.
        let mut c = small_cloth();
        for (k, xi) in c.x.iter_mut().enumerate() {
            xi.z += 0.01 * ((k * 7 % 5) as f64 - 2.0);
            xi.x += 0.005 * ((k * 3 % 4) as f64 - 1.5);
        }
        let (f, _) = c.elastic_forces(&c.x, false);
        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for i in 0..c.x.len() {
            for d in 0..3 {
                let mut xp = c.x.clone();
                let mut xm = c.x.clone();
                xp[i][d] += eps;
                xm[i][d] -= eps;
                let fd = -(c.elastic_energy(&xp) - c.elastic_energy(&xm)) / (2.0 * eps);
                max_err = max_err.max((f[i][d] - fd).abs());
            }
        }
        assert!(max_err < 1e-4, "force ≠ −∇energy: max err {max_err:.2e}");
    }

    #[test]
    fn pinned_cloth_drapes_to_equilibrium() {
        // Pin the two top corners; the cloth sags under gravity and (with damping) settles.
        let mut c = small_cloth();
        c.damping = 0.05;
        let (nx, ny) = (4, 4);
        c.pinned[(ny - 1) * nx] = true; // top-left
        c.pinned[(ny - 1) * nx + nx - 1] = true; // top-right
        let z_top = c.x[(ny - 1) * nx].z;
        let mut prev_energy = f64::INFINITY;
        for step in 0..4000 {
            c.step();
            if step % 500 == 499 {
                let e = c.elastic_energy(&c.x) - c.mass * c.gravity.z * c.x.iter().map(|p| p.z).sum::<f64>();
                assert!(e <= prev_energy + 1e-6, "total energy increased");
                prev_energy = e;
            }
        }
        // Free bottom-middle vertex hangs well below the pinned corners; motion has settled.
        assert!(c.x[0].z < z_top - 0.02, "cloth did not sag: z = {}", c.x[0].z);
        assert!(c.kinetic_energy() < 1e-4, "cloth did not settle: KE = {}", c.kinetic_energy());
    }

    // a cloth folded into two overlapping layers within thickness: top rows are a strain-free reflected
    // copy of the bottom rows, `gap` above them (so the two layers interpenetrate).
    fn folded_cloth(nx: usize, ny: usize, h: f64, gap: f64) -> ClothSim {
        // soft material + small dt so the (necessarily distorted) fold elements stay stable and the
        // test isolates the self-collision response rather than a membrane blow-up
        let mut c = ClothSim::grid(nx, ny, h, 0.02, 5.0, 2.0, 0.5, 5e-4);
        let idx = |i: usize, j: usize| j * nx + i;
        for j in 0..ny {
            for i in 0..nx {
                let k = idx(i, j);
                if j < ny / 2 {
                    c.x[k] = Vector3::new(i as f64 * h, j as f64 * h, 0.0); // bottom layer, flat
                } else {
                    let jb = ny - 1 - j; // reflect over the fold → strain-free copy above the bottom
                    c.x[k] = Vector3::new(i as f64 * h, jb as f64 * h, gap);
                }
            }
        }
        c.v = vec![Vector3::zeros(); c.x.len()];
        c
    }

    #[test]
    fn point_triangle_distance_matches_analytic() {
        let (a, b, cc) = (Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
        // above the interior: distance is the height, closest point the projection
        let (cp, _) = closest_point_on_triangle(Vector3::new(0.25, 0.25, 0.5), a, b, cc);
        assert!(((Vector3::new(0.25, 0.25, 0.5) - cp).norm() - 0.5).abs() < 1e-12, "interior: {cp:?}");
        // beyond a vertex: closest point is that vertex
        let (cp2, _) = closest_point_on_triangle(Vector3::new(-0.5, -0.5, 0.0), a, b, cc);
        assert!((cp2 - a).norm() < 1e-12, "vertex region: {cp2:?}");
        // beyond an edge: closest point on that edge
        let (cp3, _) = closest_point_on_triangle(Vector3::new(0.5, -0.4, 0.0), a, b, cc);
        assert!((cp3 - Vector3::new(0.5, 0.0, 0.0)).norm() < 1e-12, "edge region: {cp3:?}");
    }

    #[test]
    fn self_collision_force_is_the_exact_energy_gradient() {
        let (thickness, k) = (0.05, 4000.0);
        // two overlapping layers, top offset so its vertices project into bottom-triangle interiors and
        // its edges cross bottom edges (interior closest points → no barycentric-boundary kinks)
        let (nx, ny, h) = (4, 4, 0.1);
        let mut c = ClothSim::grid(nx, ny, h, 0.02, 200.0, 100.0, 5.0, 1e-3);
        let idx = |i: usize, j: usize| j * nx + i;
        let gap = 0.6 * thickness;
        for j in 0..ny {
            for i in 0..nx {
                let k = idx(i, j);
                if j >= ny / 2 {
                    let jb = ny - 1 - j;
                    c.x[k] = Vector3::new(i as f64 * h + 0.3 * h, jb as f64 * h + 0.3 * h, gap);
                }
            }
        }
        let mut f = vec![Vector3::zeros(); c.x.len()];
        c.accumulate_self_collision_forces(&c.x, thickness, k, &mut f);
        assert!(c.self_collision_energy(&c.x, thickness, k) > 0.0, "config has no active self-collisions");
        let eps = 1e-7;
        let mut worst = 0.0f64;
        for i in 0..c.x.len() {
            for d in 0..3 {
                let (mut xp, mut xm) = (c.x.clone(), c.x.clone());
                xp[i][d] += eps;
                xm[i][d] -= eps;
                let fd = -(c.self_collision_energy(&xp, thickness, k) - c.self_collision_energy(&xm, thickness, k)) / (2.0 * eps);
                worst = worst.max((f[i][d] - fd).abs());
            }
        }
        eprintln!("cloth self-collision force vs finite-diff: worst |Δ| {worst:.3e}");
        assert!(worst < 2e-3, "self-collision force is not the energy gradient: {worst}");
    }

    #[test]
    fn self_collision_broadphase_finds_every_true_contact() {
        let (thickness, cell) = (0.05, 0.11);
        let c = folded_cloth(6, 6, 0.1, 0.6 * thickness);
        // neighbor sets (mirror the crate's 1-ring exclusion) from the public triangle list
        use std::collections::{HashMap, HashSet};
        let mut nbr: HashMap<usize, HashSet<usize>> = HashMap::new();
        for t in c.triangles() {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                nbr.entry(a).or_default().insert(b);
                nbr.entry(b).or_default().insert(a);
            }
        }
        let nb = |a: usize, b: usize| nbr.get(&a).is_some_and(|s| s.contains(&b));
        let vt_far = |p: usize, t: &[usize; 3]| !t.contains(&p) && !t.iter().any(|&v| nb(p, v));
        let ee_far = |e0: (usize, usize), e1: (usize, usize)| e0.0 != e1.0 && e0.0 != e1.1 && e0.1 != e1.0 && e0.1 != e1.1 && !nb(e0.0, e1.0) && !nb(e0.0, e1.1) && !nb(e0.1, e1.0) && !nb(e0.1, e1.1);
        // ground truth: all active (topologically-far) VT and EE pairs by exhaustive search
        let mut vt_truth: HashSet<(usize, usize)> = HashSet::new();
        for (p, &pt) in c.x.iter().enumerate() {
            for (ti, t) in c.triangles().iter().enumerate() {
                if !vt_far(p, t) {
                    continue;
                }
                let (cp, _) = closest_point_on_triangle(pt, c.x[t[0]], c.x[t[1]], c.x[t[2]]);
                if (pt - cp).norm() < thickness {
                    vt_truth.insert((p, ti));
                }
            }
        }
        let edges = c.edges();
        let mut ee_truth: HashSet<(usize, usize)> = HashSet::new();
        for a in 0..edges.len() {
            for b in (a + 1)..edges.len() {
                let (e0, e1) = (edges[a], edges[b]);
                if !ee_far(e0, e1) {
                    continue;
                }
                if segment_segment_closest(c.x[e0.0], c.x[e0.1], c.x[e1.0], c.x[e1.1]).0 < thickness {
                    ee_truth.insert((a.min(b), a.max(b)));
                }
            }
        }
        let (vt_c, ee_c) = c.self_collision_candidates(&c.x, thickness, cell);
        let (vt_c, ee_c): (HashSet<_>, HashSet<_>) = (vt_c.into_iter().collect(), ee_c.into_iter().collect());
        let vt_missed: Vec<_> = vt_truth.difference(&vt_c).collect();
        let ee_missed: Vec<_> = ee_truth.difference(&ee_c).collect();
        eprintln!("cloth broadphase: VT {}/{} true, {} missed; EE {}/{} true, {} missed", vt_c.len(), vt_truth.len(), vt_missed.len(), ee_c.len(), ee_truth.len(), ee_missed.len());
        assert!(vt_missed.is_empty(), "broadphase missed vertex-triangle contacts: {vt_missed:?}");
        assert!(ee_missed.is_empty(), "broadphase missed edge-edge contacts: {ee_missed:?}");
        assert!(!vt_truth.is_empty(), "test config had no self-collisions");
    }

    // build a ClothSim by hand from an explicit vertex list and triangle list (in-crate access to the
    // private mesh fields), so a self-collision test can use a config with no confounding elastic coupling.
    fn mesh(x: Vec<Vector3<f64>>, tris: Vec<[usize; 3]>, pinned: Vec<bool>, mass: f64, dt: f64) -> ClothSim {
        let n = x.len();
        let (mut dm_inv, mut area) = (Vec::new(), Vec::new());
        for t in &tris {
            let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
            let dm = Matrix2::new(b.x - a.x, c.x - a.x, b.y - a.y, c.y - a.y);
            area.push(0.5 * dm.determinant().abs());
            dm_inv.push(dm.try_inverse().unwrap_or_else(Matrix2::identity));
        }
        ClothSim { x, v: vec![Vector3::zeros(); n], pinned, tris, dm_inv, area, bends: vec![], mass, mu: 0.0, lambda: 0.0, k_bend: 0.0, damping: 0.3, dt, gravity: Vector3::new(0.0, 0.0, -9.81) }
    }

    #[test]
    fn self_collision_stops_a_vertex_hitting_a_triangle() {
        // The cleanest dynamic check: a fixed triangle in the z=0 plane and a single free vertex directly
        // above it (topologically disjoint — no shared vertex, no elastic tie), falling under gravity.
        // The self-collision penalty must catch it at ~one thickness and never let it cross to z<0.
        // Without the penalty it would fall straight through. Isolates the collision response entirely.
        let (thickness, k) = (0.06, 8.0e3);
        let (a, b, cc) = (Vector3::new(0.0, 0.0, 0.0), Vector3::new(0.4, 0.0, 0.0), Vector3::new(0.0, 0.4, 0.0));
        let p0 = Vector3::new(0.13, 0.13, 1.5 * thickness); // above the triangle interior, clear
        let mut c = mesh(vec![a, b, cc, p0], vec![[0, 1, 2]], vec![true, true, true, false], 0.02, 5e-4);

        // control: gravity, no collision → the vertex falls straight through the triangle plane
        let mut ctrl = c.clone();
        for _ in 0..2000 {
            ctrl.step();
        }
        let ctrl_z = ctrl.x[3].z;

        // with collision → caught and held above the triangle, never crossing
        let mut min_z = c.x[3].z;
        for _ in 0..6000 {
            c.step_self_collision(thickness, k);
            min_z = min_z.min(c.x[3].z);
        }
        let end_z = c.x[3].z;
        eprintln!("vertex-vs-triangle: start z {:.4}, no-collision z {:.4}, with-collision end z {:.4} (min ever {:.4}), thickness {:.4}", p0.z, ctrl_z, end_z, min_z, thickness);
        assert!(c.x[3].iter().all(|v| v.is_finite()), "blew up");
        assert!(ctrl_z < 0.0, "control should fall through the triangle: z {ctrl_z}");
        assert!(min_z > 0.5 * thickness, "vertex tunneled toward/through the triangle: min z {min_z}");
        assert!(end_z > 0.7 * thickness && end_z < 1.3 * thickness, "vertex did not rest at ~one thickness: z {end_z}");
    }

    #[test]
    fn material_gradient_matches_finite_difference() {
        // Analytic ∂KE/∂μ of one step vs central FD — the differentiable-cloth check.
        let mut c = small_cloth();
        for (k, xi) in c.x.iter_mut().enumerate() {
            xi.z += 0.02 * ((k * 5 % 7) as f64 - 3.0);
        }
        for (k, vi) in c.v.iter_mut().enumerate() {
            *vi = Vector3::new(0.1 * ((k % 3) as f64 - 1.0), 0.05, -0.1);
        }
        let (_, dke) = c.ke_and_dke_dmu();
        let eps = 1e-2;
        let ke_at = |mu: f64| {
            let mut s = c.clone();
            s.mu = mu;
            s.ke_and_dke_dmu().0
        };
        let fd = (ke_at(c.mu + eps) - ke_at(c.mu - eps)) / (2.0 * eps);
        let rel = (dke - fd).abs() / fd.abs().max(1e-12);
        eprintln!("cloth ∂KE/∂μ: analytic={dke:.6e}, fd={fd:.6e}, rel_err={rel:.2e}");
        assert!(dke.abs() > 1e-9, "gradient trivially zero");
        assert!(rel < 1e-6, "material gradient wrong: analytic {dke} vs fd {fd}");
    }
}
