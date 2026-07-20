//! **Zonotopes & reachability analysis** — the SDP-free way to compute *guaranteed* forward-reachable sets
//! of a linear(ized) system under bounded disturbances/inputs. A **zonotope** `Z = { c + Σ βᵢ gᵢ : βᵢ ∈
//! [−1,1] }` (a centre plus generator columns) is closed under the two operations reachability needs —
//! **linear maps** (`A·Z`) and **Minkowski sums** (`Z₁ ⊕ Z₂`) — so the reachable set of `x⁺ = A x + B u`
//! with `u ∈ U` propagates in closed form: `R_{k+1} = A R_k ⊕ B U`. This is the set the crate's safety
//! controllers ([`crate::CbfQp`], tube MPC) *reference* but never *compute*; zonotopes give a deterministic,
//! wasm-clean certificate (unlike ellipsoidal/SOS reachability, which needs SDP).
//!
//! Everything reduces to the **support function** `ρ_Z(d) = c·d + Σ|gᵢ·d|` — the set's extent in direction
//! `d` — from which containment, bounding boxes, and over-approximation checks follow. Verified: the support
//! function matches brute-force vertex enumeration; it is additive under Minkowski sum and transforms
//! correctly under linear maps; and a computed reachable set provably contains every simulated trajectory of
//! the system under admissible inputs. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// A zonotope: centre `c` and generator matrix `G` (each **column** is a generator `gᵢ`).
#[derive(Clone, Debug)]
pub struct Zonotope {
    pub center: DVector<f64>,
    pub generators: DMatrix<f64>,
}

impl Zonotope {
    /// A zonotope from a centre and generator columns.
    pub fn new(center: DVector<f64>, generators: DMatrix<f64>) -> Self {
        Zonotope { center, generators }
    }

    /// A single point (a zero-generator zonotope).
    pub fn point(center: DVector<f64>) -> Self {
        let n = center.len();
        Zonotope { center, generators: DMatrix::zeros(n, 0) }
    }

    /// An axis-aligned box `[lo, hi]` as a zonotope (diagonal generators of half-width).
    pub fn from_interval(lo: &DVector<f64>, hi: &DVector<f64>) -> Self {
        let n = lo.len();
        let center = (lo + hi) * 0.5;
        let mut g = DMatrix::zeros(n, n);
        for i in 0..n {
            g[(i, i)] = (hi[i] - lo[i]) * 0.5;
        }
        Zonotope { center, generators: g }
    }

    /// The number of generators (the zonotope's *order* × dimension).
    pub fn num_generators(&self) -> usize {
        self.generators.ncols()
    }

    /// The support function `ρ_Z(d) = c·d + Σ|gᵢ·d|` — the maximum of `x·d` over `x ∈ Z`.
    pub fn support(&self, dir: &DVector<f64>) -> f64 {
        let gt_d = self.generators.transpose() * dir; // (gᵢ·d) per generator
        self.center.dot(dir) + gt_d.iter().map(|v| v.abs()).sum::<f64>()
    }

    /// The Minkowski sum `Z ⊕ other` (same dimension): centres add, generators concatenate.
    pub fn minkowski_sum(&self, other: &Zonotope) -> Zonotope {
        let n = self.center.len();
        let (p, q) = (self.num_generators(), other.num_generators());
        let mut g = DMatrix::zeros(n, p + q);
        g.view_mut((0, 0), (n, p)).copy_from(&self.generators);
        g.view_mut((0, p), (n, q)).copy_from(&other.generators);
        Zonotope { center: &self.center + &other.center, generators: g }
    }

    /// The image `A·Z` under a linear map.
    pub fn linear_map(&self, a: &DMatrix<f64>) -> Zonotope {
        Zonotope { center: a * &self.center, generators: a * &self.generators }
    }

    /// The axis-aligned bounding box `(lo, hi)` (the interval hull).
    pub fn interval_hull(&self) -> (DVector<f64>, DVector<f64>) {
        let n = self.center.len();
        let mut r = DVector::zeros(n);
        for i in 0..n {
            r[i] = (0..self.num_generators()).map(|j| self.generators[(i, j)].abs()).sum();
        }
        (&self.center - &r, &self.center + &r)
    }

    /// Whether `point` lies inside the zonotope, tested by the supporting-hyperplane condition over a set of
    /// probe directions (exact as the probe set → all directions; use the box normals plus any extras).
    pub fn contains(&self, point: &DVector<f64>, dirs: &[DVector<f64>], tol: f64) -> bool {
        dirs.iter().all(|d| point.dot(d) <= self.support(d) + tol)
    }
}

/// Forward reachable sets of `x⁺ = A x + B u` with `u ∈ U` (a zonotope), starting from `x0` (a zonotope):
/// `R₀ = x0`, `R_{k+1} = A R_k ⊕ B U`. Returns `R₀ … R_steps`.
pub fn reach_linear(a: &DMatrix<f64>, b: &DMatrix<f64>, x0: &Zonotope, u: &Zonotope, steps: usize) -> Vec<Zonotope> {
    let bu = u.linear_map(b);
    let mut out = vec![x0.clone()];
    for _ in 0..steps {
        let next = out.last().unwrap().linear_map(a).minkowski_sum(&bu);
        out.push(next);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    #[test]
    fn the_support_function_matches_brute_force_vertices() {
        // THE ORACLE. ρ_Z(d) = max over vertices of x·d, where vertices are c + Σ ±gᵢ (all sign combos).
        let c = dv(&[1.0, -0.5]);
        let g = DMatrix::from_row_slice(2, 3, &[1.0, 0.2, -0.3, 0.0, 0.5, 0.4]);
        let z = Zonotope::new(c.clone(), g.clone());
        for d in [dv(&[1.0, 0.0]), dv(&[0.0, 1.0]), dv(&[1.0, 1.0]), dv(&[-0.7, 0.3])] {
            // brute force over 2³ sign patterns
            let mut best = f64::NEG_INFINITY;
            for mask in 0..(1u32 << 3) {
                let mut x = c.clone();
                for j in 0..3 {
                    let s = if mask & (1 << j) != 0 { 1.0 } else { -1.0 };
                    x += g.column(j) * s;
                }
                best = best.max(x.dot(&d));
            }
            assert!((z.support(&d) - best).abs() < 1e-12, "support {} vs brute {best}", z.support(&d));
        }
    }

    #[test]
    fn minkowski_sum_and_linear_map_transform_the_support_correctly() {
        let z1 = Zonotope::new(dv(&[1.0, 0.0]), DMatrix::from_row_slice(2, 1, &[0.5, 0.2]));
        let z2 = Zonotope::new(dv(&[-0.5, 1.0]), DMatrix::from_row_slice(2, 1, &[0.1, 0.4]));
        let d = dv(&[0.8, -0.6]);
        // ρ_{Z1⊕Z2}(d) = ρ_{Z1}(d) + ρ_{Z2}(d)
        let sum = z1.minkowski_sum(&z2);
        assert!((sum.support(&d) - (z1.support(&d) + z2.support(&d))).abs() < 1e-12, "Minkowski additivity");
        // ρ_{A·Z}(d) = ρ_Z(Aᵀ d)
        let a = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, -0.2, 0.8]);
        let az = z1.linear_map(&a);
        assert!((az.support(&d) - z1.support(&(a.transpose() * &d))).abs() < 1e-12, "linear-map support");
    }

    #[test]
    fn the_interval_hull_bounds_the_box_zonotope_exactly() {
        let z = Zonotope::from_interval(&dv(&[-1.0, 2.0]), &dv(&[3.0, 5.0]));
        let (lo, hi) = z.interval_hull();
        assert!((lo - dv(&[-1.0, 2.0])).norm() < 1e-12 && (hi - dv(&[3.0, 5.0])).norm() < 1e-12, "box hull");
    }

    #[test]
    fn the_reachable_set_contains_every_admissible_trajectory() {
        // THE HEADLINE. Propagate the reachable set of a stable 2-D system with a box input set, then check
        // that many random admissible input sequences stay inside the computed sets at every step.
        let a = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.0, 0.85]);
        let b = DMatrix::identity(2, 2);
        let x0 = Zonotope::point(dv(&[0.0, 0.0]));
        let u = Zonotope::from_interval(&dv(&[-0.1, -0.1]), &dv(&[0.1, 0.1]));
        let steps = 15;
        let reach = reach_linear(&a, &b, &x0, &u, steps);
        // probe directions for the containment test (box normals + diagonals)
        let dirs = [dv(&[1.0, 0.0]), dv(&[-1.0, 0.0]), dv(&[0.0, 1.0]), dv(&[0.0, -1.0]), dv(&[1.0, 1.0]), dv(&[1.0, -1.0]), dv(&[-1.0, 1.0]), dv(&[-1.0, -1.0])];

        // deterministic pseudo-random admissible inputs
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 0.2 // in [-0.1, 0.1]
        };
        for _ in 0..200 {
            let mut x = dv(&[0.0, 0.0]);
            for (k, rk) in reach.iter().enumerate() {
                assert!(rk.contains(&x, &dirs, 1e-9), "state {x} escaped reachable set at step {k}");
                let ui = dv(&[rnd(), rnd()]);
                x = &a * &x + &b * &ui;
            }
        }
    }
}
