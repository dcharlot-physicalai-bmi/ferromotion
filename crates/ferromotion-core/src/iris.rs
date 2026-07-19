//! **IRIS — Iterative Regional Inflation** (Deits & Tedrake, WAFR 2014): grow a large obstacle-free
//! **convex region** around a seed point — the enabling primitive under safe-corridor planning, graphs-of-
//! convex-sets trajectory optimization, and composite configuration-space fields. IRIS alternates two
//! steps until the region stops growing:
//!
//! 1. **separating hyperplanes** — for each obstacle, place a half-space that excludes it, tangent to the
//!    current inscribed ellipsoid (normal `M(v−c)` through the obstacle point `v`); their intersection is
//!    an obstacle-free polytope;
//! 2. **inscribed ellipsoid** — fit a large ellipsoid inside that polytope; its anisotropy is what lets
//!    the region stretch down corridors a ball never could.
//!
//! The exact IRIS ellipsoid step is a max-volume (log-det) SDP. This implementation uses the **Dikin
//! ellipsoid** at the polytope's analytic center — a genuine inscribed ellipsoid computed with plain
//! Newton iterations, preserving IRIS's alternating anisotropic growth without an SDP solver (a documented
//! simplification of the volume-maximal step). Obstacles are points (use a convex obstacle's vertices).
//! Verified: the region is obstacle-free, contains the seed, grows monotonically, and elongates along a
//! corridor. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector};

/// An obstacle-free convex region: the polytope `{x : A x ≤ b}` plus the inscribed ellipsoid
/// `{c + C u : ‖u‖ ≤ 1}` (with `C = M^{-1/2}`) that IRIS grew inside it.
#[derive(Clone, Debug)]
pub struct ConvexRegion {
    pub a: DMatrix<f64>,
    pub b: DVector<f64>,
    pub center: DVector<f64>,
    /// The ellipsoid metric `M` (so the ellipsoid is `{x : (x−c)ᵀM(x−c) ≤ 1}`).
    pub metric: DMatrix<f64>,
}

impl ConvexRegion {
    /// Is `x` strictly inside the polytope? (Obstacle points sit on their own separating plane, so
    /// strictness is what makes them count as *excluded*.)
    pub fn contains(&self, x: &DVector<f64>) -> bool {
        (&self.a * x - &self.b).iter().all(|&r| r < -1e-9)
    }
    /// Ellipsoid volume up to the constant unit-ball factor: `det(C) = det(M)^{-1/2}`.
    pub fn ellipsoid_volume(&self) -> f64 {
        self.metric.clone().determinant().powf(-0.5)
    }
    /// The ellipsoid's semi-axis length along coordinate `k` (`1/√M_kk`-ish; here from `M⁻¹` diagonal).
    pub fn extent(&self, k: usize) -> f64 {
        self.metric.clone().try_inverse().map(|mi| mi[(k, k)].sqrt()).unwrap_or(0.0)
    }
}

/// An IRIS problem: point obstacles inside an axis-aligned domain box `[lo, hi]`.
#[derive(Clone, Debug)]
pub struct Iris {
    pub obstacles: Vec<DVector<f64>>,
    pub lo: DVector<f64>,
    pub hi: DVector<f64>,
    pub iters: usize,
}

impl Iris {
    fn dim(&self) -> usize {
        self.lo.len()
    }

    /// Separating half-spaces excluding every obstacle, tangent to the ellipsoid `(c, M)`: for obstacle
    /// `v`, normal `a = M(v−c)` and offset `b = aᵀv` (so `aᵀx ≤ b` keeps `c`, excludes `v`).
    fn obstacle_planes(&self, c: &DVector<f64>, m: &DMatrix<f64>) -> (Vec<DVector<f64>>, Vec<f64>) {
        let mut aa = Vec::new();
        let mut bb = Vec::new();
        for v in &self.obstacles {
            let a = m * (v - c);
            let na = a.norm();
            if na < 1e-12 {
                continue;
            }
            let a = a / na;
            aa.push(a.clone());
            bb.push(a.dot(v));
        }
        (aa, bb)
    }

    /// Stack the obstacle planes and the domain-box faces into one `A x ≤ b`.
    fn full_polytope(&self, c: &DVector<f64>, m: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
        let n = self.dim();
        let (mut aa, mut bb) = self.obstacle_planes(c, m);
        for k in 0..n {
            let mut e = DVector::zeros(n);
            e[k] = 1.0;
            aa.push(e.clone());
            bb.push(self.hi[k]); // x_k ≤ hi_k
            aa.push(-e.clone());
            bb.push(-self.lo[k]); // −x_k ≤ −lo_k
        }
        let m_rows = aa.len();
        let mut a = DMatrix::zeros(m_rows, n);
        for (i, ai) in aa.iter().enumerate() {
            a.row_mut(i).copy_from(&ai.transpose());
        }
        (a, DVector::from_vec(bb))
    }

    /// The analytic center of `{A x ≤ b}` (minimizer of `−Σ log(bᵢ − aᵢᵀx)`) by damped Newton, and the
    /// **Dikin** Hessian there `H = Σ aᵢaᵢᵀ/sᵢ²` — the metric of an inscribed ellipsoid.
    fn analytic_center(&self, a: &DMatrix<f64>, b: &DVector<f64>, x0: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        let n = self.dim();
        let mut x = x0.clone();
        for _ in 0..60 {
            let s = b - a * &x; // slacks
            let mut grad = DVector::zeros(n);
            let mut h = DMatrix::zeros(n, n);
            for i in 0..a.nrows() {
                let ai = a.row(i).transpose();
                let si = s[i].max(1e-9);
                grad += &ai / si;
                h += &ai * ai.transpose() / (si * si);
            }
            let Some(hinv) = h.clone().try_inverse() else { break };
            let mut step = -(&hinv * &grad); // Newton step to MINIMIZE −Σlog(sᵢ): x −= H⁻¹∇φ
            // damped/backtracking to keep strictly feasible
            let mut t = 1.0;
            for _ in 0..40 {
                let xn = &x + &step * t;
                if (b - a * &xn).iter().all(|&si| si > 1e-9) {
                    break;
                }
                t *= 0.5;
            }
            step *= t;
            let dec = step.norm();
            x += step;
            if dec < 1e-10 {
                break;
            }
        }
        // Dikin Hessian at the center
        let s = b - a * &x;
        let mut h = DMatrix::zeros(n, n);
        for i in 0..a.nrows() {
            let ai = a.row(i).transpose();
            let si = s[i].max(1e-9);
            h += &ai * ai.transpose() / (si * si);
        }
        (x, h)
    }

    /// Inflate an obstacle-free convex region around `seed`.
    pub fn inflate(&self, seed: &DVector<f64>) -> ConvexRegion {
        self.inflate_trace(seed).0
    }

    /// Like [`Iris::inflate`], but also returns the inscribed-ellipsoid volume after each iteration
    /// (`volumes[0]` is the seed ball) — the monotone-growth trace.
    pub fn inflate_trace(&self, seed: &DVector<f64>) -> (ConvexRegion, Vec<f64>) {
        let n = self.dim();
        let mut c = seed.clone();
        // seed metric: a small isotropic ball
        let r0 = 1e-2_f64;
        let mut m: DMatrix<f64> = DMatrix::identity(n, n) / (r0 * r0);
        let mut last = (DMatrix::zeros(0, n), DVector::zeros(0));
        let mut volumes = vec![m.clone().determinant().powf(-0.5)];

        for _ in 0..self.iters {
            let (a, b) = self.full_polytope(&c, &m);
            let (new_c, h) = self.analytic_center(&a, &b, &c);
            c = new_c;
            m = h;
            last = (a, b);
            volumes.push(m.clone().determinant().powf(-0.5));
        }
        (ConvexRegion { a: last.0, b: last.1, center: c, metric: m }, volumes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    #[test]
    fn the_region_is_obstacle_free_and_contains_the_seed() {
        // Obstacles scattered around a seed; the inflated polytope must exclude every obstacle and keep the
        // seed strictly inside.
        let obstacles = vec![dv(&[1.0, 0.0]), dv(&[-1.0, 0.2]), dv(&[0.1, 1.0]), dv(&[0.0, -1.0])];
        let iris = Iris { obstacles: obstacles.clone(), lo: dv(&[-3.0, -3.0]), hi: dv(&[3.0, 3.0]), iters: 6 };
        let region = iris.inflate(&dv(&[0.0, 0.0]));
        assert!(region.contains(&dv(&[0.0, 0.0])), "seed must be inside the region");
        for o in &obstacles {
            assert!(!region.contains(o), "obstacle {o:?} must be excluded");
        }
    }

    #[test]
    fn the_region_grows_monotonically() {
        // THE INVARIANT. As IRIS alternates, the inscribed-ellipsoid volume is non-decreasing.
        let obstacles = vec![dv(&[1.5, 0.0]), dv(&[-1.5, 0.0]), dv(&[0.0, 1.5]), dv(&[0.0, -1.5])];
        let iris = Iris { obstacles, lo: dv(&[-4.0, -4.0]), hi: dv(&[4.0, 4.0]), iters: 6 };
        let (_, vols) = iris.inflate_trace(&dv(&[0.0, 0.0]));
        // the inscribed ellipsoid never shrinks across the alternation …
        for w in vols.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "region volume shrank: {} → {}", w[0], w[1]);
        }
        // … and inflates massively from the tiny seed ball to fill the free space.
        assert!(vols.last().unwrap() > &(vols[0] * 100.0), "the region should inflate from the seed: {vols:?}");
    }

    #[test]
    fn the_region_elongates_along_a_corridor() {
        // THE HEADLINE. Two walls of obstacles form a corridor along x (obstacles above and below y=±0.5);
        // the inscribed ellipsoid must stretch along x far more than y — the anisotropy a ball can't have.
        let mut obstacles = Vec::new();
        let mut xx = -3.0;
        while xx <= 3.0 {
            obstacles.push(dv(&[xx, 0.7]));
            obstacles.push(dv(&[xx, -0.7]));
            xx += 0.5;
        }
        let iris = Iris { obstacles, lo: dv(&[-4.0, -4.0]), hi: dv(&[4.0, 4.0]), iters: 8 };
        let region = iris.inflate(&dv(&[0.0, 0.0]));
        let (ex, ey) = (region.extent(0), region.extent(1));
        assert!(ex > 2.0 * ey, "corridor region should elongate along x: extent_x {ex} vs extent_y {ey}");
    }

    #[test]
    fn interior_points_are_collision_free() {
        // Any point in the region is farther from every obstacle than a small margin (sampled).
        let obstacles = vec![dv(&[1.2, 0.3]), dv(&[-1.0, -0.4]), dv(&[0.2, 1.1])];
        let iris = Iris { obstacles: obstacles.clone(), lo: dv(&[-3.0, -3.0]), hi: dv(&[3.0, 3.0]), iters: 6 };
        let region = iris.inflate(&dv(&[0.0, 0.0]));
        // sample the ellipsoid interior: c + 0.5·C·(axis directions)
        let mi = region.metric.clone().try_inverse().unwrap();
        for k in 0..2 {
            for &s in &[-0.5, 0.5] {
                let mut d = DVector::zeros(2);
                d[k] = s * mi[(k, k)].sqrt();
                let p = &region.center + d;
                if region.contains(&p) {
                    for o in &obstacles {
                        assert!((&p - o).norm() > 0.05, "interior sample too close to an obstacle: {p:?}");
                    }
                }
            }
        }
    }
}
