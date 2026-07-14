//! **RRT\*** — asymptotically-optimal sampling-based motion planning (Karaman & Frazzoli, 2011). The
//! complement to the trajectory *optimizers* in the stack: instead of refining a fixed-topology
//! guess, RRT\* grows a tree of collision-free motions through the free space, converging almost
//! surely to the optimal path as samples accumulate. Each new sample is steered from its nearest
//! node, connected to the least-cost parent in a shrinking neighborhood, and the neighborhood is then
//! **rewired** through it when that lowers a node's cost.
//!
//! The planner is generic over a collision predicate `edge_free(a, b)`, so it composes directly with
//! the [`crate::SdfScene`] collision world (or any robot collision model). Deterministic (seeded LCG)
//! and dependency-free. Pure Rust → WASM-clean.

/// RRT\* configuration.
#[derive(Clone, Copy, Debug)]
pub struct RrtStar {
    pub dim: usize,
    /// Max steering extension per sample.
    pub step: f64,
    /// Probability of sampling the goal directly.
    pub goal_bias: f64,
    /// Neighborhood-radius constant (radius = `gamma·(ln n / n)^{1/dim}`, capped at `step`).
    pub gamma: f64,
    pub max_iters: usize,
    pub seed: u64,
}

/// A found plan.
#[derive(Clone, Debug)]
pub struct RrtResult {
    pub path: Vec<Vec<f64>>,
    pub cost: f64,
    pub nodes: usize,
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

impl RrtStar {
    fn steer(&self, from: &[f64], to: &[f64]) -> Vec<f64> {
        let d = dist(from, to);
        if d <= self.step {
            to.to_vec()
        } else {
            (0..self.dim).map(|i| from[i] + self.step * (to[i] - from[i]) / d).collect()
        }
    }

    /// Plan from `start` to `goal` within box bounds `[lo, hi]`, given an edge collision-free test.
    pub fn plan(&self, start: &[f64], goal: &[f64], lo: &[f64], hi: &[f64], edge_free: impl Fn(&[f64], &[f64]) -> bool) -> Option<RrtResult> {
        // nodes: (config, parent, cost-from-root)
        let mut cfg: Vec<Vec<f64>> = vec![start.to_vec()];
        let mut parent: Vec<usize> = vec![usize::MAX];
        let mut cost: Vec<f64> = vec![0.0];

        let mut rng = self.seed | 1;
        let mut u = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f64) / ((1u64 << 31) as f64)
        };

        let goal_tol = self.step;
        let (mut best_goal, mut best_goal_cost) = (usize::MAX, f64::INFINITY);

        for _ in 0..self.max_iters {
            // Sample (goal-biased).
            let x_rand: Vec<f64> = if u() < self.goal_bias {
                goal.to_vec()
            } else {
                (0..self.dim).map(|i| lo[i] + (hi[i] - lo[i]) * u()).collect()
            };
            // Nearest node.
            let nearest = (0..cfg.len()).min_by(|&a, &b| dist(&cfg[a], &x_rand).partial_cmp(&dist(&cfg[b], &x_rand)).unwrap()).unwrap();
            let x_new = self.steer(&cfg[nearest], &x_rand);
            if !edge_free(&cfg[nearest], &x_new) {
                continue;
            }
            // Neighborhood.
            let nn = cfg.len() as f64;
            let radius = (self.gamma * (nn.ln() / nn).powf(1.0 / self.dim as f64)).min(self.step);
            let near: Vec<usize> = (0..cfg.len()).filter(|&i| dist(&cfg[i], &x_new) <= radius).collect();
            // Choose the least-cost collision-free parent.
            let (mut bp, mut bc) = (nearest, cost[nearest] + dist(&cfg[nearest], &x_new));
            for &i in &near {
                let c = cost[i] + dist(&cfg[i], &x_new);
                if c < bc && edge_free(&cfg[i], &x_new) {
                    bp = i;
                    bc = c;
                }
            }
            let new_idx = cfg.len();
            cfg.push(x_new.clone());
            parent.push(bp);
            cost.push(bc);
            // Rewire the neighborhood through the new node.
            for &i in &near {
                let c = bc + dist(&x_new, &cfg[i]);
                if c < cost[i] && edge_free(&x_new, &cfg[i]) {
                    parent[i] = new_idx;
                    cost[i] = c;
                }
            }
            // Goal connection.
            if dist(&x_new, goal) < goal_tol && edge_free(&x_new, goal) {
                let gc = bc + dist(&x_new, goal);
                if gc < best_goal_cost {
                    best_goal_cost = gc;
                    best_goal = new_idx;
                }
            }
        }

        if best_goal == usize::MAX {
            return None;
        }
        // Backtrack (recomputing the true cost from edge lengths, robust to stale rewire costs).
        let mut chain = vec![best_goal];
        let mut cur = best_goal;
        while parent[cur] != usize::MAX {
            cur = parent[cur];
            chain.push(cur);
        }
        chain.reverse();
        let mut path: Vec<Vec<f64>> = chain.iter().map(|&i| cfg[i].clone()).collect();
        path.push(goal.to_vec());
        let total: f64 = path.windows(2).map(|w| dist(&w[0], &w[1])).sum();
        Some(RrtResult { path, cost: total, nodes: cfg.len() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sdf, SdfScene};
    use nalgebra::Vector3;

    // Edge collision test: sample the segment and require SDF clearance > robot radius.
    fn make_edge_free(scene: SdfScene, r: f64) -> impl Fn(&[f64], &[f64]) -> bool {
        move |a: &[f64], b: &[f64]| {
            let steps = ((dist(a, b) / 0.03).ceil() as usize).max(1);
            (0..=steps).all(|k| {
                let t = k as f64 / steps as f64;
                let p = Vector3::new(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]));
                scene.distance(&p) > r
            })
        }
    }

    #[test]
    fn plans_around_an_obstacle() {
        // A sphere blocks the straight line from start to goal.
        let scene = SdfScene { prims: vec![Sdf::Sphere { center: Vector3::new(1.0, 0.0, 0.0), radius: 0.4 }] };
        let ef = make_edge_free(scene, 0.1);
        let planner = RrtStar { dim: 3, step: 0.25, goal_bias: 0.1, gamma: 3.0, max_iters: 5000, seed: 7 };
        let (lo, hi) = ([-0.5, -1.5, -1.5], [2.5, 1.5, 1.5]);
        let res = planner.plan(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &lo, &hi, &ef).expect("should find a path");

        // Path endpoints are correct and every segment is collision-free.
        assert!(dist(&res.path[0], &[0.0, 0.0, 0.0]) < 1e-9);
        assert!(dist(res.path.last().unwrap(), &[2.0, 0.0, 0.0]) < 1e-9);
        for w in res.path.windows(2) {
            assert!(ef(&w[0], &w[1]), "path segment collides");
        }
        // It routes around the sphere (longer than the blocked straight line of 2.0) but stays sane.
        assert!(res.cost > 2.0 && res.cost < 3.2, "path cost unreasonable: {}", res.cost);
    }

    #[test]
    fn asymptotically_approaches_the_straight_line_in_free_space() {
        // No obstacles: RRT* should converge toward the straight-line optimum (length 2.0).
        let ef = make_edge_free(SdfScene::default(), 0.1);
        let plan_n = |iters: usize| {
            RrtStar { dim: 3, step: 0.3, goal_bias: 0.05, gamma: 3.0, max_iters: iters, seed: 3 }
                .plan(&[0.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &[-0.5, -1.0, -1.0], &[2.5, 1.0, 1.0], &ef)
                .unwrap()
                .cost
        };
        let (coarse, fine) = (plan_n(1200), plan_n(7000));
        assert!(fine <= coarse + 1e-9, "more samples should not worsen the path: {fine} vs {coarse}");
        assert!(fine < 2.0 * 1.05, "did not approach the straight-line optimum: {fine}");
    }
}
