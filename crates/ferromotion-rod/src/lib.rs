//! ferromotion-rod — **Discrete Elastic Rods** (Bergou et al., SIGGRAPH 2008) for cables, tendons, and
//! continuum/soft robots — the 1D-deformable companion to the cloth (2D) and MPM (volumetric) solvers.
//!
//! A rod is a polyline centerline `x₀…x_n`. Two elastic energies act on it: **stretch** (edge springs
//! to rest length) and **isotropic bending**, the discrete curvature living on the curvature binormal
//! `κb_i = 2 e_{i-1}×e_i / (‖e_{i-1}‖‖e_i‖ + e_{i-1}·e_i)` at each interior vertex, with energy
//! `E_i = (EI / 2ℓ_i)‖κb_i‖²` — the discretization that reproduces the continuum `½∫EI κ² ds`. Nodal
//! forces are the exact analytic gradient of the energy (verified against finite differences), so the
//! rod is differentiable. Validated against the analytic **Euler-Bernoulli cantilever**. (Twist via
//! material frames is a further extension.) Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// A discrete elastic rod.
#[derive(Clone, Debug)]
pub struct Rod {
    pub x: Vec<Vector3<f64>>,
    pub v: Vec<Vector3<f64>>,
    pub rest_len: Vec<f64>,
    /// Mass per vertex.
    pub mass: f64,
    /// Stretch stiffness.
    pub ks: f64,
    /// Bending modulus `EI`.
    pub ei: f64,
    pub gravity: Vector3<f64>,
    /// Fixed (clamped) vertices — their forces are zeroed.
    pub clamped: Vec<bool>,
}

impl Rod {
    /// A straight rod of `nseg` segments along +x, length `len`, clamped at the first two vertices.
    pub fn straight(nseg: usize, len: f64, mass: f64, ks: f64, ei: f64, gravity: Vector3<f64>) -> Self {
        let h = len / nseg as f64;
        let x: Vec<_> = (0..=nseg).map(|i| Vector3::new(h * i as f64, 0.0, 0.0)).collect();
        let rest_len = vec![h; nseg];
        let mut clamped = vec![false; nseg + 1];
        clamped[0] = true;
        clamped[1] = true;
        Self { x, v: vec![Vector3::zeros(); nseg + 1], rest_len, mass, ks, ei, gravity, clamped }
    }

    /// Total potential energy (stretch + bending + gravity) of a configuration.
    pub fn energy(&self, x: &[Vector3<f64>]) -> f64 {
        let mut e = 0.0;
        for j in 0..self.rest_len.len() {
            let len = (x[j + 1] - x[j]).norm();
            e += 0.5 * self.ks * (len - self.rest_len[j]).powi(2);
        }
        for i in 1..x.len() - 1 {
            let (e0, e1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
            let l_i = 0.5 * (self.rest_len[i - 1] + self.rest_len[i]); // rest Voronoi length (const weight)
            e += self.ei / (2.0 * l_i) * kb(e0, e1).norm_squared();
        }
        for xi in x {
            e -= self.mass * self.gravity.dot(xi); // gravity PE = −m g·x
        }
        e
    }

    /// Net force on every vertex — the exact analytic `−∇energy`.
    pub fn forces(&self, x: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = x.len();
        let mut g = vec![Vector3::zeros(); n]; // ∇energy
        // Stretch.
        for j in 0..self.rest_len.len() {
            let e = x[j + 1] - x[j];
            let len = e.norm();
            let dir = e / len;
            let s = self.ks * (len - self.rest_len[j]);
            g[j] -= s * dir;
            g[j + 1] += s * dir;
        }
        // Bending.
        for i in 1..n - 1 {
            let (e0, e1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
            let l_i = 0.5 * (self.rest_len[i - 1] + self.rest_len[i]); // rest Voronoi length (const weight)
            let kbv = kb(e0, e1);
            let g_kb = (self.ei / l_i) * kbv; // ∂E_i/∂κb
            let (dde0, dde1) = kb_grads(e0, e1);
            // ∂x_{i-1} = −∂e0, ∂x_{i+1} = ∂e1, ∂x_i = ∂e0 − ∂e1.
            g[i - 1] += (-dde0).transpose() * g_kb;
            g[i + 1] += dde1.transpose() * g_kb;
            g[i] += (dde0 - dde1).transpose() * g_kb;
        }
        // Gravity.
        for gi in g.iter_mut() {
            *gi -= self.mass * self.gravity;
        }
        // Force = −∇energy, zeroed on clamped vertices.
        (0..n).map(|i| if self.clamped[i] { Vector3::zeros() } else { -g[i] }).collect()
    }

    /// One damped semi-implicit step (for dynamics / relaxation).
    pub fn step(&mut self, dt: f64, damping: f64) {
        let f = self.forces(&self.x);
        for i in 0..self.x.len() {
            if self.clamped[i] {
                continue;
            }
            self.v[i] += dt * f[i] / self.mass;
            self.v[i] *= damping;
            self.x[i] += dt * self.v[i];
        }
    }

    /// Relax to static equilibrium (heavy damping); returns the max residual force.
    pub fn relax(&mut self, iters: usize, dt: f64) -> f64 {
        for _ in 0..iters {
            self.step(dt, 0.9);
        }
        self.forces(&self.x).iter().map(|f| f.norm()).fold(0.0, f64::max)
    }

    /// Kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.v.iter().map(|v| v.norm_squared()).sum::<f64>()
    }
}

/// Curvature binormal at a vertex with incoming/outgoing edges `e0, e1`.
fn kb(e0: Vector3<f64>, e1: Vector3<f64>) -> Vector3<f64> {
    let d = e0.norm() * e1.norm() + e0.dot(&e1);
    2.0 * e0.cross(&e1) / d
}

/// `(∂κb/∂e0, ∂κb/∂e1)` as 3×3 matrices.
fn kb_grads(e0: Vector3<f64>, e1: Vector3<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let (n0, n1) = (e0.norm(), e1.norm());
    let cr = e0.cross(&e1);
    let d = n0 * n1 + e0.dot(&e1);
    let dd_de0 = (n1 / n0) * e0 + e1; // ∂d/∂e0
    let dd_de1 = (n0 / n1) * e1 + e0; // ∂d/∂e1
    let dde0 = (2.0 / d) * (-skew(e1)) - (2.0 / (d * d)) * cr * dd_de0.transpose();
    let dde1 = (2.0 / d) * skew(e0) - (2.0 / (d * d)) * cr * dd_de1.transpose();
    (dde0, dde1)
}

// ------------------------------------------------------------------------------------------------
// Self-contact & rod-rod contact — the physics of knots, cable routing, and bundles. A rod is a
// polyline of segments; two non-adjacent segments must not pass through each other. Contact is a
// one-sided penalty on the segment-segment distance (energy ½k(2r−d)² when closer than two radii),
// so it is differentiable (force = −∇E) and priced in the same energy currency as the elastic modes.
// ------------------------------------------------------------------------------------------------

/// Closest distance between segments `[p1,q1]` and `[p2,q2]` → `(dist, s, t, c1, c2)`, where the
/// closest points are `c1 = p1 + s(q1−p1)` and `c2 = p2 + t(q2−p2)` (Ericson, *Real-Time Collision
/// Detection*, clamped closest-point-of-two-segments).
pub fn segment_segment_closest(p1: Vector3<f64>, q1: Vector3<f64>, p2: Vector3<f64>, q2: Vector3<f64>) -> (f64, f64, f64, Vector3<f64>, Vector3<f64>) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(&d1), d2.dot(&d2), d2.dot(&r));
    const EPS: f64 = 1e-14;
    let (mut s, t);
    if a <= EPS && e <= EPS {
        s = 0.0;
        t = 0.0;
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
    ((c1 - c2).norm(), s, t, c1, c2)
}

/// Segment index pairs `(i, j)` (`i < j`, never adjacent) whose radius-inflated AABBs overlap — the
/// broadphase that keeps self-contact near-linear instead of O(n²). `cell` must be ≥ `2·radius`.
pub fn contact_candidate_pairs(x: &[Vector3<f64>], radius: f64, cell: f64) -> Vec<(usize, usize)> {
    use std::collections::{HashMap, HashSet};
    let nseg = x.len().saturating_sub(1);
    let key = |p: Vector3<f64>| (( (p.x / cell).floor() as i64), ((p.y / cell).floor() as i64), ((p.z / cell).floor() as i64));
    // insert each segment into every grid cell its inflated AABB touches
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for i in 0..nseg {
        let (a, b) = (x[i], x[i + 1]);
        let lo = a.inf(&b) - Vector3::repeat(radius);
        let hi = a.sup(&b) + Vector3::repeat(radius);
        let (kl, kh) = (key(lo), key(hi));
        for cx in kl.0..=kh.0 {
            for cy in kl.1..=kh.1 {
                for cz in kl.2..=kh.2 {
                    grid.entry((cx, cy, cz)).or_default().push(i);
                }
            }
        }
    }
    let mut pairs: HashSet<(usize, usize)> = HashSet::new();
    for seg in grid.values() {
        for a in 0..seg.len() {
            for b in (a + 1)..seg.len() {
                let (i, j) = (seg[a].min(seg[b]), seg[a].max(seg[b]));
                if j > i + 1 {
                    pairs.insert((i, j)); // skip adjacent (share a vertex)
                }
            }
        }
    }
    let mut out: Vec<_> = pairs.into_iter().collect();
    out.sort_unstable();
    out
}

/// One-sided penalty self-contact energy of a rod centerline: `½k(2r−d)²` summed over every
/// non-adjacent segment pair closer than `2·radius`. Zero when the rod is clear of itself.
pub fn self_contact_energy(x: &[Vector3<f64>], radius: f64, k: f64) -> f64 {
    let nseg = x.len().saturating_sub(1);
    let two_r = 2.0 * radius;
    let mut e = 0.0;
    for i in 0..nseg {
        for j in (i + 2)..nseg {
            let (d, ..) = segment_segment_closest(x[i], x[i + 1], x[j], x[j + 1]);
            if d < two_r {
                e += 0.5 * k * (two_r - d).powi(2);
            }
        }
    }
    e
}

/// Add the self-contact repulsion to `force` (a per-vertex accumulator). Force on each closest point
/// is `k(2r−d)·n̂` along the separation direction, split to the two segment endpoints by the closest-
/// point barycentric weights — the exact `−∇` of [`self_contact_energy`] (closest points stationary).
pub fn accumulate_self_contact_forces(x: &[Vector3<f64>], radius: f64, k: f64, force: &mut [Vector3<f64>]) {
    let nseg = x.len().saturating_sub(1);
    let two_r = 2.0 * radius;
    for i in 0..nseg {
        for j in (i + 2)..nseg {
            let (d, s, t, c1, c2) = segment_segment_closest(x[i], x[i + 1], x[j], x[j + 1]);
            if d < two_r && d > 1e-12 {
                let n = (c1 - c2) / d; // points from segment j toward segment i
                let mag = k * (two_r - d);
                let f1 = mag * n; // repels segment i's contact point
                force[i] += (1.0 - s) * f1;
                force[i + 1] += s * f1;
                force[j] -= (1.0 - t) * f1;
                force[j + 1] -= t * f1;
            }
        }
    }
}

/// Self-contact repulsion as a fresh per-vertex force vector.
pub fn self_contact_forces(x: &[Vector3<f64>], radius: f64, k: f64) -> Vec<Vector3<f64>> {
    let mut f = vec![Vector3::zeros(); x.len()];
    accumulate_self_contact_forces(x, radius, k, &mut f);
    f
}

/// Penalty contact between two distinct rods → `(forces on a, forces on b)`. Every segment of `a`
/// against every segment of `b` (all non-adjacent since they are different rods): the cable-bundle /
/// rod-on-rod primitive, the same energy as self-contact.
pub fn rod_rod_contact_forces(xa: &[Vector3<f64>], xb: &[Vector3<f64>], radius: f64, k: f64) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let (na, nb) = (xa.len().saturating_sub(1), xb.len().saturating_sub(1));
    let two_r = 2.0 * radius;
    let (mut fa, mut fb) = (vec![Vector3::zeros(); xa.len()], vec![Vector3::zeros(); xb.len()]);
    for i in 0..na {
        for j in 0..nb {
            let (d, s, t, c1, c2) = segment_segment_closest(xa[i], xa[i + 1], xb[j], xb[j + 1]);
            if d < two_r && d > 1e-12 {
                let n = (c1 - c2) / d;
                let f1 = k * (two_r - d) * n;
                fa[i] += (1.0 - s) * f1;
                fa[i + 1] += s * f1;
                fb[j] -= (1.0 - t) * f1;
                fb[j + 1] -= t * f1;
            }
        }
    }
    (fa, fb)
}

impl Rod {
    /// One damped step with self-contact: elastic forces plus the self-contact repulsion (radius `r`,
    /// stiffness `k_contact`). The step that lets a rod be tied into a knot without passing through itself.
    pub fn step_self_contact(&mut self, dt: f64, damping: f64, radius: f64, k_contact: f64) {
        let mut f = self.forces(&self.x);
        accumulate_self_contact_forces(&self.x, radius, k_contact, &mut f);
        for i in 0..self.x.len() {
            if self.clamped[i] {
                continue;
            }
            self.v[i] += dt * f[i] / self.mass;
            self.v[i] *= damping;
            self.x[i] += dt * self.v[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forces_are_the_exact_energy_gradient() {
        // A wiggly rod: analytic forces must equal −∇energy (finite differences).
        let mut rod = Rod::straight(8, 1.0, 0.01, 500.0, 0.1, Vector3::new(0.0, 0.0, -9.81));
        for (i, xi) in rod.x.iter_mut().enumerate() {
            xi.z += 0.05 * (i as f64 * 0.9).sin();
            xi.y += 0.03 * (i as f64 * 0.6).cos();
        }
        let x = rod.x.clone();
        let f = rod.forces(&x);
        let eps = 1e-6;
        for i in 0..x.len() {
            if rod.clamped[i] {
                continue;
            }
            for d in 0..3 {
                let (mut xp, mut xm) = (x.clone(), x.clone());
                xp[i][d] += eps;
                xm[i][d] -= eps;
                let fd = -(rod.energy(&xp) - rod.energy(&xm)) / (2.0 * eps);
                assert!((f[i][d] - fd).abs() < 1e-4, "force[{i}][{d}]: analytic {} vs fd {fd}", f[i][d]);
            }
        }
    }

    #[test]
    fn cantilever_matches_euler_bernoulli() {
        // A clamped rod sagging under self-weight; small-deflection tip deflection δ = w·L⁴/(8·EI).
        let (nseg, len, ei) = (20, 1.0, 0.5);
        let m = 0.001; // mass per vertex
        let g = 9.81;
        // ks/dt chosen for explicit stability: the bending mode ω ~ √(EI/(m·ℓ³)) is the stiff limiter.
        let mut rod = Rod::straight(nseg, len, m, 200.0, ei, Vector3::new(0.0, 0.0, -g));
        let residual = rod.relax(200000, 1e-4);
        assert!(residual < 1e-4, "did not reach equilibrium: residual {residual}");

        let w = m * nseg as f64 * g / len; // weight per unit length
        let expected = w * len.powi(4) / (8.0 * ei);
        let tip = -rod.x[nseg].z; // downward deflection
        let rel = (tip - expected).abs() / expected;
        eprintln!("cantilever: tip={tip:.5}, Euler-Bernoulli={expected:.5}, rel_err={rel:.3}");
        assert!(rel < 0.08, "cantilever deflection off Euler-Bernoulli by {rel:.3} (tip {tip}, EB {expected})");
    }

    #[test]
    fn segment_distance_matches_analytic_cases() {
        // two orthogonal segments crossing at height h: distance = h, closest points at the crossing
        let (d, s, t, _, _) = segment_segment_closest(
            Vector3::new(-1.0, 0.0, 0.5),
            Vector3::new(1.0, 0.0, 0.5),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        assert!((d - 0.5).abs() < 1e-12 && (s - 0.5).abs() < 1e-12 && (t - 0.5).abs() < 1e-12, "crossing: d={d}, s={s}, t={t}");
        // parallel segments offset by 1 in y: distance 1
        let (d2, ..) = segment_segment_closest(Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0), Vector3::new(0.5, 1.0, 0.0), Vector3::new(1.5, 1.0, 0.0));
        assert!((d2 - 1.0).abs() < 1e-12, "parallel: d={d2}");
        // endpoint-to-endpoint (segments point away from each other): distance = gap between near ends
        let (d3, ..) = segment_segment_closest(Vector3::new(0.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0));
        assert!((d3 - 1.0).abs() < 1e-12, "endpoints: d={d3}");
    }

    #[test]
    fn self_contact_force_is_the_exact_energy_gradient() {
        // a hairpin: the rod doubles back so two non-adjacent segments sit within 2r (interior closest
        // points, no endpoint clamping) — the analytic repulsion must equal −∇(self-contact energy).
        let (radius, k) = (0.08, 3000.0);
        let x = vec![
            Vector3::new(0.00, 0.0, 0.06),
            Vector3::new(0.20, 0.0, 0.06),
            Vector3::new(0.40, 0.0, 0.06),
            Vector3::new(0.45, 0.0, 0.00), // fold
            Vector3::new(0.30, 0.0, -0.05),
            Vector3::new(0.10, 0.0, -0.05),
        ];
        let f = self_contact_forces(&x, radius, k);
        let eps = 1e-7;
        let mut worst = 0.0f64;
        for i in 0..x.len() {
            for d in 0..3 {
                let (mut xp, mut xm) = (x.clone(), x.clone());
                xp[i][d] += eps;
                xm[i][d] -= eps;
                let fd = -(self_contact_energy(&xp, radius, k) - self_contact_energy(&xm, radius, k)) / (2.0 * eps);
                worst = worst.max((f[i][d] - fd).abs());
            }
        }
        eprintln!("self-contact force vs finite-diff: worst |Δ| {worst:.3e}");
        assert!(worst < 1e-3, "self-contact force is not the energy gradient: {worst}");
    }

    #[test]
    fn broadphase_finds_every_true_contact() {
        // a wiggly rod folded on itself; the spatial-hash broadphase must not miss any pair that the
        // exhaustive all-pairs check finds within 2r (a missed contact = a tunnel).
        let (radius, cell) = (0.05, 0.12);
        let n = 30;
        let x: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                let u = i as f64 / (n as f64 - 1.0);
                Vector3::new((u * 6.28).sin() * 0.3, 0.02 * (i as f64 * 1.3).cos(), u * 0.2 - 0.1)
            })
            .collect();
        let nseg = x.len() - 1;
        // ground truth: all non-adjacent pairs within 2r
        use std::collections::HashSet;
        let mut truth: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..nseg {
            for j in (i + 2)..nseg {
                let (d, ..) = segment_segment_closest(x[i], x[i + 1], x[j], x[j + 1]);
                if d < 2.0 * radius {
                    truth.insert((i, j));
                }
            }
        }
        let cand: HashSet<(usize, usize)> = contact_candidate_pairs(&x, radius, cell).into_iter().collect();
        let missed: Vec<_> = truth.difference(&cand).collect();
        eprintln!("broadphase: {} candidates, {} true contacts, {} missed", cand.len(), truth.len(), missed.len());
        assert!(missed.is_empty(), "broadphase missed true contacts: {missed:?}");
        assert!(!truth.is_empty(), "test config had no self-contacts to find");
    }

    #[test]
    fn self_contact_resolves_an_interpenetrating_rod() {
        // A hairpin whose two arms START overlapping (gap 0.6·2r — the strands are through each other).
        // Contact stiffness dominates the (weak) bending, so relaxation must PUSH the arms apart to at
        // least ~2r: penetration is resolved, not tolerated. Without self-contact they would stay crossed.
        let (radius, k) = (0.05, 3.0e4);
        let two_r = 2.0 * radius;
        let mut rod = Rod::straight(9, 1.0, 0.02, 1200.0, 0.005, Vector3::zeros());
        let gap = 0.6 * two_r; // arms start closer than 2r → interpenetrating
        rod.x = vec![
            Vector3::new(0.00, 0.0, gap / 2.0),
            Vector3::new(0.15, 0.0, gap / 2.0),
            Vector3::new(0.30, 0.0, gap / 2.0),
            Vector3::new(0.45, 0.0, gap / 2.0),
            Vector3::new(0.55, 0.0, 0.0), // fold
            Vector3::new(0.45, 0.0, -gap / 2.0),
            Vector3::new(0.30, 0.0, -gap / 2.0),
            Vector3::new(0.15, 0.0, -gap / 2.0),
            Vector3::new(0.00, 0.0, -gap / 2.0),
            Vector3::new(-0.05, 0.0, -gap / 2.0),
        ];
        rod.v = vec![Vector3::zeros(); rod.x.len()];
        rod.clamped = vec![false; rod.x.len()];
        let min_dist = |x: &[Vector3<f64>]| {
            let nseg = x.len() - 1;
            let mut m = f64::INFINITY;
            for i in 0..nseg {
                for j in (i + 2)..nseg {
                    m = m.min(segment_segment_closest(x[i], x[i + 1], x[j], x[j + 1]).0);
                }
            }
            m
        };
        let start_min = min_dist(&rod.x);
        for _ in 0..6000 {
            rod.step_self_contact(5e-4, 0.9, radius, k);
        }
        let end_min = min_dist(&rod.x);
        eprintln!("self-contact resolve: start min-dist {:.4} (2r {:.4}), end min-dist {:.4}", start_min, two_r, end_min);
        assert!(rod.x.iter().all(|p| p.iter().all(|c| c.is_finite())), "rod blew up");
        assert!(start_min < two_r, "test must start interpenetrating: {start_min} vs 2r {two_r}");
        assert!(end_min > 0.9 * two_r, "self-contact failed to separate the strands: end {end_min} vs 2r {two_r}");
    }

    #[test]
    fn free_vibration_conserves_energy() {
        // Undamped, no gravity: a plucked rod's total energy stays bounded/constant.
        let mut rod = Rod::straight(10, 1.0, 0.02, 200.0, 0.05, Vector3::zeros());
        // Clamp only the first vertex (free elsewhere).
        rod.clamped = vec![false; rod.x.len()];
        rod.clamped[0] = true;
        // Pluck: displace the free end.
        let last = rod.x.len() - 1;
        rod.x[last].z += 0.1;
        let e0 = rod.energy(&rod.x) + rod.kinetic_energy();
        let (dt, mut worst) = (2e-4, 0.0f64);
        for _ in 0..3000 {
            rod.step(dt, 1.0); // no damping
            let e = rod.energy(&rod.x) + rod.kinetic_energy();
            worst = worst.max((e - e0).abs() / e0.abs().max(1e-9));
        }
        assert!(worst < 5e-2, "energy not conserved: worst relative drift {worst}");
    }
}
