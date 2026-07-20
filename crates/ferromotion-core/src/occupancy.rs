//! **Occupancy-grid mapping** — build a 2-D map of free vs. occupied space from range-sensor (LiDAR/sonar)
//! beams. The grid stores each cell's **log-odds** of being occupied; integrating a beam **ray-casts**
//! (Bresenham) from the sensor to the measured endpoint, nudging every traversed cell toward *free* and the
//! hit cell toward *occupied*, with a clamp so the map stays correctable. Accumulating many beams turns
//! noisy individual measurements into a confident map — the classic front-end of grid SLAM (gmapping,
//! Cartographer) and the world model reactive planners and [`crate::bit_star`]/[`crate::hybrid_astar`] query
//! for collision.
//!
//! Log-odds are additive (Bayes with an inverse-sensor-model), so updates are cheap and order-independent.
//! Verified: a single beam frees the cells along its ray and marks the endpoint occupied; a max-range (no-
//! hit) beam frees the endpoint too; and repeated beams drive the log-odds toward (clamped) certainty. Pure
//! Rust → WASM-clean.

/// A 2-D occupancy grid over `[origin, origin + size·resolution)` storing per-cell log-odds.
#[derive(Clone, Debug)]
pub struct OccupancyGrid {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    log_odds: Vec<f64>,
    l_occ: f64,
    l_free: f64,
    clamp: f64,
}

impl OccupancyGrid {
    /// A grid of `width × height` cells of size `resolution`, with lower-left corner at `(origin_x,
    /// origin_y)` and every cell unknown (log-odds 0).
    pub fn new(width: usize, height: usize, resolution: f64, origin_x: f64, origin_y: f64) -> Self {
        OccupancyGrid { width, height, resolution, origin_x, origin_y, log_odds: vec![0.0; width * height], l_occ: 0.85, l_free: -0.4, clamp: 8.0 }
    }

    /// World coordinates → integer cell `(i, j)` (may be outside the grid).
    pub fn world_to_cell(&self, x: f64, y: f64) -> (i64, i64) {
        (((x - self.origin_x) / self.resolution).floor() as i64, ((y - self.origin_y) / self.resolution).floor() as i64)
    }

    fn in_bounds(&self, i: i64, j: i64) -> bool {
        i >= 0 && j >= 0 && (i as usize) < self.width && (j as usize) < self.height
    }

    /// Log-odds of a cell (0 = unknown).
    pub fn log_odds(&self, i: i64, j: i64) -> f64 {
        if self.in_bounds(i, j) {
            self.log_odds[j as usize * self.width + i as usize]
        } else {
            0.0
        }
    }

    /// Occupancy probability of a cell, `1 / (1 + e^{−logodds})`.
    pub fn probability(&self, i: i64, j: i64) -> f64 {
        1.0 / (1.0 + (-self.log_odds(i, j)).exp())
    }

    fn add(&mut self, i: i64, j: i64, delta: f64) {
        if self.in_bounds(i, j) {
            let k = j as usize * self.width + i as usize;
            self.log_odds[k] = (self.log_odds[k] + delta).clamp(-self.clamp, self.clamp);
        }
    }

    /// Integrate one beam from sensor position `(ox, oy)` to endpoint `(ex, ey)`. If `hit`, the endpoint is a
    /// real obstacle (marked occupied); otherwise it is a max-range reading (marked free).
    pub fn integrate_beam(&mut self, ox: f64, oy: f64, ex: f64, ey: f64, hit: bool) {
        let (x0, y0) = self.world_to_cell(ox, oy);
        let (x1, y1) = self.world_to_cell(ex, ey);
        // Bresenham line from (x0,y0) to (x1,y1); every cell before the last is free
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);
        loop {
            if x == x1 && y == y1 {
                break;
            }
            self.add(x, y, self.l_free);
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
        // the endpoint cell: occupied on a hit, free on a max-range reading
        self.add(x1, y1, if hit { self.l_occ } else { self.l_free });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_beam_frees_its_ray_and_occupies_the_endpoint() {
        // THE ORACLE. A horizontal beam from (0.5,0.5) to (4.5,0.5): the cells it passes through become free
        // (prob < 0.5) and the endpoint cell occupied (prob > 0.5).
        let mut g = OccupancyGrid::new(10, 10, 1.0, 0.0, 0.0);
        g.integrate_beam(0.5, 0.5, 4.5, 0.5, true);
        for i in 0..4 {
            assert!(g.probability(i, 0) < 0.5, "cell ({i},0) should be free: {}", g.probability(i, 0));
        }
        assert!(g.probability(4, 0) > 0.5, "endpoint should be occupied: {}", g.probability(4, 0));
    }

    #[test]
    fn a_max_range_beam_frees_the_endpoint_too() {
        let mut g = OccupancyGrid::new(10, 10, 1.0, 0.0, 0.0);
        g.integrate_beam(0.5, 0.5, 4.5, 0.5, false); // no hit
        assert!(g.probability(4, 0) < 0.5, "no-hit endpoint should be free: {}", g.probability(4, 0));
    }

    #[test]
    fn repeated_beams_converge_toward_certainty() {
        // Many consistent beams push the endpoint toward occupied and the ray toward free, up to the clamp.
        let mut g = OccupancyGrid::new(10, 10, 1.0, 0.0, 0.0);
        for _ in 0..50 {
            g.integrate_beam(0.5, 0.5, 3.5, 0.5, true);
        }
        assert!(g.probability(3, 0) > 0.99, "endpoint should be near-certain occupied: {}", g.probability(3, 0));
        assert!(g.probability(1, 0) < 0.01, "ray should be near-certain free: {}", g.probability(1, 0));
        // and the log-odds are clamped, not unbounded
        assert!(g.log_odds(3, 0) <= 8.0 + 1e-9, "log-odds should be clamped");
    }

    #[test]
    fn a_diagonal_beam_traces_a_connected_ray() {
        // Bresenham sanity: a diagonal beam frees a connected staircase of cells to the occupied endpoint.
        let mut g = OccupancyGrid::new(10, 10, 1.0, 0.0, 0.0);
        g.integrate_beam(0.5, 0.5, 5.5, 5.5, true);
        assert!(g.probability(0, 0) < 0.5 && g.probability(5, 5) > 0.5, "diagonal ray free, endpoint occupied");
    }
}
