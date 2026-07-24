//! **Environment-scale flow + flow-aware planning** (Honest Fluids — stage 9). Deployed robots
//! don't run CFD in the loop, but they *do* consume precomputed environment flow: urban wind fields
//! for drone routing, currents for AUVs. This is that pattern — a divergence-free wind field with
//! O(1) querying, and a Zermelo min-time planner that routes a vehicle to ride tailwinds and dodge
//! headwinds. The field is built from a streamfunction so it is incompressible by construction
//! (confirmed by the same [`crate::harness`] divergence receipt), and the planner is a Dijkstra
//! shortest-*time* search whose edge cost is the wind-adjusted traversal time.
//!
//! Verified: the flow-aware route beats the wind-naive straight line, by a margin that grows with
//! the wind strength — the planner genuinely exploits the flow, and the exploitation scales.

/// A steady 2-D wind field on the unit square: a uniform component plus gyres from a periodic
/// streamfunction `ψ = amp·sin(2πx)sin(2πy)` (so `u = ∂ψ/∂y`, `v = −∂ψ/∂x` — divergence-free by
/// construction, and periodic so a periodic divergence receipt applies cleanly).
#[derive(Clone, Copy)]
pub struct Wind {
    pub ux: f64,
    pub uy: f64,
    pub amp: f64,
}

impl Wind {
    /// Wind velocity at `(x, y)`.
    pub fn at(&self, x: f64, y: f64) -> (f64, f64) {
        use std::f64::consts::PI;
        let k = 2.0 * PI;
        let u = self.ux + self.amp * k * (k * x).sin() * (k * y).cos();
        let v = self.uy - self.amp * k * (k * x).cos() * (k * y).sin();
        (u, v)
    }
}

/// A circular no-fly obstacle.
#[derive(Clone, Copy)]
pub struct Obstacle {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
}

/// A min-time route through a wind field on an `n × n` grid, respecting obstacles.
pub struct Planner {
    pub n: usize,
    pub wind: Wind,
    pub obstacles: Vec<Obstacle>,
    pub airspeed: f64,
}

impl Planner {
    fn blocked(&self, i: usize, j: usize) -> bool {
        let (x, y) = ((i as f64 + 0.5) / self.n as f64, (j as f64 + 0.5) / self.n as f64);
        self.obstacles.iter().any(|o| (x - o.cx).hypot(y - o.cy) <= o.r)
    }

    /// Wind-adjusted traversal time for a step of length `ds` in unit direction `(dx, dy)` at the
    /// midpoint `(x, y)`: ground speed ≈ airspeed + wind·direction (Zermelo small-angle form).
    fn edge_time(&self, x: f64, y: f64, dx: f64, dy: f64, ds: f64) -> f64 {
        let (wx, wy) = self.wind.at(x, y);
        let ground = self.airspeed + wx * dx + wy * dy;
        if ground <= 1e-3 {
            f64::INFINITY // can't make headway
        } else {
            ds / ground
        }
    }

    /// Dijkstra min-time path from grid cell `start` to `goal` (8-connected). Returns
    /// `(total_time, path_cells)`, or `None` if unreachable.
    pub fn plan(&self, start: (usize, usize), goal: (usize, usize)) -> Option<(f64, Vec<(usize, usize)>)> {
        let n = self.n;
        let idx = |i: usize, j: usize| i * n + j;
        let mut dist = vec![f64::INFINITY; n * n];
        let mut prev = vec![usize::MAX; n * n];
        let mut visited = vec![false; n * n];
        dist[idx(start.0, start.1)] = 0.0;
        let h = 1.0 / n as f64;
        let neigh: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];
        // simple O(N²) Dijkstra (dense; verification grids are small).
        for _ in 0..n * n {
            let mut u = usize::MAX;
            let mut best = f64::INFINITY;
            for k in 0..n * n {
                if !visited[k] && dist[k] < best {
                    best = dist[k];
                    u = k;
                }
            }
            if u == usize::MAX {
                break;
            }
            visited[u] = true;
            let (ui, uj) = (u / n, u % n);
            if (ui, uj) == goal {
                break;
            }
            for &(di, dj) in &neigh {
                let (vi, vj) = (ui as i32 + di, uj as i32 + dj);
                if vi < 0 || vi >= n as i32 || vj < 0 || vj >= n as i32 {
                    continue;
                }
                let (vi, vj) = (vi as usize, vj as usize);
                if self.blocked(vi, vj) {
                    continue;
                }
                let len = ((di * di + dj * dj) as f64).sqrt() * h;
                let (mx, my) = ((ui as f64 + 0.5 + di as f64 * 0.5) / n as f64, (uj as f64 + 0.5 + dj as f64 * 0.5) / n as f64);
                let (ndx, ndy) = (di as f64, dj as f64);
                let norm = (ndx * ndx + ndy * ndy).sqrt();
                let t = self.edge_time(mx, my, ndx / norm, ndy / norm, len);
                let alt = dist[u] + t;
                if alt < dist[idx(vi, vj)] {
                    dist[idx(vi, vj)] = alt;
                    prev[idx(vi, vj)] = u;
                }
            }
        }
        let g = idx(goal.0, goal.1);
        if !dist[g].is_finite() {
            return None;
        }
        let mut path = vec![goal];
        let mut cur = g;
        while prev[cur] != usize::MAX {
            cur = prev[cur];
            path.push((cur / n, cur % n));
        }
        path.reverse();
        Some((dist[g], path))
    }

    /// Traversal time actually EXPERIENCED on a naive straight-line route from `start` to `goal`
    /// (the wind-blind planner still flies through the real wind). The honest baseline.
    pub fn straight_line_time(&self, start: (usize, usize), goal: (usize, usize), samples: usize) -> f64 {
        let n = self.n as f64;
        let (sx, sy) = ((start.0 as f64 + 0.5) / n, (start.1 as f64 + 0.5) / n);
        let (gx, gy) = ((goal.0 as f64 + 0.5) / n, (goal.1 as f64 + 0.5) / n);
        let (dx, dy) = (gx - sx, gy - sy);
        let total = dx.hypot(dy);
        let (ux, uy) = (dx / total, dy / total);
        let ds = total / samples as f64;
        let mut t = 0.0;
        for k in 0..samples {
            let f = (k as f64 + 0.5) / samples as f64;
            let (x, y) = (sx + dx * f, sy + dy * f);
            t += self.edge_time(x, y, ux, uy, ds);
        }
        t
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::harness::{audit, FlowField};

    /// The streamfunction wind field is divergence-free — confirmed by the harness receipt.
    #[test]
    fn wind_field_is_divergence_free() {
        let wind = Wind { ux: 0.3, uy: 0.0, amp: 0.4 };
        let f = FlowField::sample(64, |x, y| wind.at(x, y));
        let r = audit(&f);
        eprintln!("env wind divergence receipt: {:.2e}", r.divergence_rms);
        assert!(r.divergence_rms < 1e-6, "wind field not divergence-free: {}", r.divergence_rms);
    }

    /// Flow-aware routing beats the wind-naive straight line, and the advantage GROWS with wind
    /// strength — the planner exploits the flow, and the exploitation scales.
    #[test]
    fn flow_aware_routing_beats_naive_and_scales() {
        let n = 48;
        let start = (4, n / 2);
        let goal = (n - 5, n / 2);
        // Airspeed above the peak wind (amp·2π ≈ 2.8) so the straight line stays feasible and both
        // times are finite — a clean scaling comparison rather than "naive becomes impossible".
        let advantage = |amp: f64| -> f64 {
            let p = Planner { n, wind: Wind { ux: 0.0, uy: 0.0, amp }, obstacles: vec![], airspeed: 3.0 };
            let (opt, _) = p.plan(start, goal).expect("reachable");
            let naive = p.straight_line_time(start, goal, 200);
            naive / opt // > 1 ⇒ the optimal route is faster than flying straight through the wind
        };
        let weak = advantage(0.15);
        let strong = advantage(0.45);
        eprintln!("routing advantage (naive/optimal): weak wind {weak:.3}, strong wind {strong:.3}");
        // weak ≈ 1 (little spatial structure to exploit at high airspeed; ~1.0 up to grid-vs-continuous
        // discretization); strong ≫ 1 (the planner routes around the headwind pockets). Clear scaling.
        assert!(weak >= 0.98, "optimal should not be slower than straight line: {weak}");
        assert!(strong > 1.5, "no meaningful exploitation of a strong flow: {strong}");
        assert!(strong > weak + 0.5, "exploitation did not scale with wind strength: {weak} → {strong}");
    }

    /// The planner routes around a no-fly obstacle blocking the direct line.
    #[test]
    fn routes_around_obstacles() {
        let n = 48;
        let p = Planner {
            n,
            wind: Wind { ux: 0.2, uy: 0.0, amp: 0.2 },
            obstacles: vec![Obstacle { cx: 0.5, cy: 0.5, r: 0.12 }],
            airspeed: 1.0,
        };
        let (_, path) = p.plan((4, n / 2), (n - 5, n / 2)).expect("a route around exists");
        // no path cell may lie inside the obstacle
        let hit = path.iter().any(|&(i, j)| {
            let (x, y) = ((i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64);
            (x - 0.5).hypot(y - 0.5) <= 0.12
        });
        eprintln!("obstacle route length: {} cells, clear = {}", path.len(), !hit);
        assert!(!hit, "path passed through the obstacle");
    }
}
