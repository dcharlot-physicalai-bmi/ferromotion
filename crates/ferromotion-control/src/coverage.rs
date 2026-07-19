//! **Multi-robot coverage control — Lloyd's algorithm** (Cortés, Martínez, Karataş & Bullo, IEEE T-RA
//! 2004): deploy a team of robots to optimally *cover* a region under a density `φ` that weights where
//! sensing matters. Each robot owns its **Voronoi cell** (the points it is the closest robot to), and the
//! team minimizes the locational-optimization cost `H(p) = Σᵢ ∫_{Vᵢ} ‖q − pᵢ‖² φ(q) dq`. Lloyd's gradient
//! descent is beautifully simple: **move each robot to the centroid of its Voronoi cell**. The fixed points
//! are the **centroidal Voronoi tessellations**, the locally-optimal sensor deployments.
//!
//! This adds the *coverage/deployment* layer to the crate's multi-robot suite, next to consensus/formation
//! ([`crate::swarm`]) and optimal allocation ([`crate::hungarian`]). Cells and centroids are computed by
//! nearest-generator assignment over a density grid (the standard practical implementation). Verified: the
//! coverage cost decreases monotonically every iteration; a converged configuration is a fixed point (each
//! robot sits at its cell centroid); and on a uniform density the robots spread out rather than collapse.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::Vector2;

/// A Lloyd coverage controller over the axis-aligned box `[lo, hi]`, with the density integrated on a
/// `grid × grid` lattice.
#[derive(Clone, Debug)]
pub struct LloydCoverage {
    pub lo: Vector2<f64>,
    pub hi: Vector2<f64>,
    pub grid: usize,
}

impl LloydCoverage {
    fn grid_points(&self) -> impl Iterator<Item = (Vector2<f64>, f64)> + '_ {
        let (nx, ny) = (self.grid, self.grid);
        let dx = (self.hi.x - self.lo.x) / nx as f64;
        let dy = (self.hi.y - self.lo.y) / ny as f64;
        let cell_area = dx * dy;
        (0..nx).flat_map(move |i| {
            (0..ny).map(move |j| {
                let q = Vector2::new(self.lo.x + (i as f64 + 0.5) * dx, self.lo.y + (j as f64 + 0.5) * dy);
                (q, cell_area)
            })
        })
    }

    fn nearest(generators: &[Vector2<f64>], q: &Vector2<f64>) -> usize {
        let mut best = 0;
        let mut bd = f64::INFINITY;
        for (i, p) in generators.iter().enumerate() {
            let d = (p - q).norm_squared();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best
    }

    /// The locational-optimization cost `H(p) = Σᵢ ∫_{Vᵢ} ‖q − pᵢ‖² φ(q) dq`, integrated over the grid.
    pub fn coverage_cost(&self, generators: &[Vector2<f64>], density: impl Fn(&Vector2<f64>) -> f64) -> f64 {
        let mut h = 0.0;
        for (q, da) in self.grid_points() {
            let i = Self::nearest(generators, &q);
            h += (generators[i] - q).norm_squared() * density(&q) * da;
        }
        h
    }

    /// One Lloyd iteration: move each robot to the density-weighted centroid of its Voronoi cell. Cells with
    /// no captured mass keep their generator in place.
    pub fn step(&self, generators: &[Vector2<f64>], density: impl Fn(&Vector2<f64>) -> f64) -> Vec<Vector2<f64>> {
        let n = generators.len();
        let mut num = vec![Vector2::zeros(); n];
        let mut den = vec![0.0f64; n];
        for (q, da) in self.grid_points() {
            let i = Self::nearest(generators, &q);
            let w = density(&q) * da;
            num[i] += q * w;
            den[i] += w;
        }
        (0..n).map(|i| if den[i] > 1e-12 { num[i] / den[i] } else { generators[i] }).collect()
    }

    /// Run Lloyd's algorithm for `iters` iterations. Returns the final generators and the cost after each
    /// iteration (including the initial cost as element 0) — a monotonically non-increasing curve.
    pub fn converge(&self, generators: &[Vector2<f64>], density: impl Fn(&Vector2<f64>) -> f64 + Copy, iters: usize) -> (Vec<Vector2<f64>>, Vec<f64>) {
        let mut g = generators.to_vec();
        let mut hist = vec![self.coverage_cost(&g, density)];
        for _ in 0..iters {
            g = self.step(&g, density);
            hist.push(self.coverage_cost(&g, density));
        }
        (g, hist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> LloydCoverage {
        LloydCoverage { lo: Vector2::new(0.0, 0.0), hi: Vector2::new(1.0, 1.0), grid: 80 }
    }

    #[test]
    fn the_coverage_cost_decreases_monotonically() {
        // THE ORACLE. Lloyd's algorithm is gradient descent on H; every iteration weakly lowers the cost.
        let cov = unit_square();
        // a non-trivial density (Gaussian bump off-centre) makes the descent interesting
        let density = |q: &Vector2<f64>| 1.0 + 3.0 * (-(((q.x - 0.7).powi(2) + (q.y - 0.3).powi(2)) / 0.05)).exp();
        let init = vec![
            Vector2::new(0.2, 0.2),
            Vector2::new(0.8, 0.2),
            Vector2::new(0.5, 0.8),
            Vector2::new(0.5, 0.5),
        ];
        let (_g, hist) = cov.converge(&init, density, 25);
        for w in hist.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "coverage cost must not increase: {} → {}", w[0], w[1]);
        }
        assert!(hist.last().unwrap() < &(0.95 * hist[0]), "the deployment should measurably improve: {} → {}", hist[0], hist.last().unwrap());
    }

    #[test]
    fn a_converged_configuration_is_a_centroidal_voronoi_fixed_point() {
        // At a centroidal Voronoi tessellation each robot IS its cell centroid, so a further step barely
        // moves anyone.
        let cov = unit_square();
        let density = |_q: &Vector2<f64>| 1.0;
        let init = vec![Vector2::new(0.25, 0.25), Vector2::new(0.75, 0.25), Vector2::new(0.25, 0.75), Vector2::new(0.75, 0.75)];
        let (g, _hist) = cov.converge(&init, density, 60);
        let g2 = cov.step(&g, density);
        let max_move = g.iter().zip(&g2).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        assert!(max_move < 1e-3, "a converged CVT should be a fixed point: max move {max_move}");
    }

    #[test]
    fn robots_spread_out_under_uniform_density() {
        // THE HEADLINE. Four robots that start bunched near one corner must SPREAD to cover a uniform square,
        // not collapse together — each ends up near the centre of a quadrant.
        let cov = unit_square();
        let density = |_q: &Vector2<f64>| 1.0;
        let init = vec![Vector2::new(0.1, 0.1), Vector2::new(0.15, 0.12), Vector2::new(0.12, 0.16), Vector2::new(0.09, 0.14)];
        let start_spread = pairwise_min(&init);
        let (g, _hist) = cov.converge(&init, density, 80);
        let end_spread = pairwise_min(&g);
        assert!(end_spread > start_spread + 0.2, "robots should spread apart: {start_spread} → {end_spread}");
        // for a uniform unit square the CVT is the 2×2 grid of quadrant centres (0.25/0.75); every robot
        // should land near one such centre
        for p in &g {
            let near_quadrant = [0.25f64, 0.75].iter().flat_map(|&x| [0.25f64, 0.75].map(move |y| Vector2::new(x, y))).map(|c| (p - c).norm()).fold(f64::INFINITY, f64::min);
            assert!(near_quadrant < 0.08, "robot {p:?} should settle near a quadrant centre: dist {near_quadrant}");
        }
    }

    fn pairwise_min(g: &[Vector2<f64>]) -> f64 {
        let mut m = f64::INFINITY;
        for i in 0..g.len() {
            for j in (i + 1)..g.len() {
                m = m.min((g[i] - g[j]).norm());
            }
        }
        m
    }
}
