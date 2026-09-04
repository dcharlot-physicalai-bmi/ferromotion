//! **Probabilistic Roadmap (k-PRM\*)** — the multi-query motion planner the tree lacked (it had
//! single-query RRT\*/BIT\*). A roadmap of collision-free samples is built *once*; then any number of
//! start→goal queries are answered by connecting the endpoints and running a graph search — the
//! multi-query advantage in a static environment (repeated pick-and-place, re-planning to new
//! goals). The `k`-nearest connection rule `k = ⌈e(1+1/d)·ln n⌉` is the PRM\* schedule that makes the
//! roadmap asymptotically optimal (Karaman & Frazzoli 2011). Deterministic sampling (splitmix64) →
//! reproducible. Pure `nalgebra`-free — plain slices → WASM-clean.

/// A built roadmap: free samples and their collision-free adjacency (index, edge cost).
pub struct Roadmap {
    pub nodes: Vec<Vec<f64>>,
    pub adj: Vec<Vec<(usize, f64)>>,
    dim: usize,
}

/// PRM\* configuration.
pub struct PrmStar {
    pub n_samples: usize,
    pub seed: u64,
    /// Max edge length to attempt (a segment longer than this is skipped even if within k-nearest).
    pub max_edge: f64,
}

impl Default for PrmStar {
    fn default() -> Self {
        PrmStar { n_samples: 400, seed: 0x2718, max_edge: f64::INFINITY }
    }
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

impl PrmStar {
    /// Build the roadmap in the box `[lo, hi]`. `point_free(x)` is true for a collision-free
    /// configuration; `edge_free(a, b)` for a collision-free straight segment.
    /// `None` when the bounds are malformed: different lengths, empty, or a non-finite coordinate.
    /// Non-finite bounds used to produce non-finite samples and fault inside the nearest-neighbour
    /// sort, the same way [`Roadmap::query`] did with a non-finite endpoint.
    pub fn build(
        &self,
        lo: &[f64],
        hi: &[f64],
        point_free: impl Fn(&[f64]) -> bool,
        edge_free: impl Fn(&[f64], &[f64]) -> bool,
    ) -> Option<Roadmap> {
        if lo.is_empty() || lo.len() != hi.len() || !lo.iter().chain(hi).all(|x| x.is_finite()) {
            return None;
        }
        let dim = lo.len();
        let mut s = self.seed;
        let mut rand = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) as f64) / (u64::MAX as f64)
        };
        // rejection-sample collision-free nodes
        let mut nodes: Vec<Vec<f64>> = Vec::with_capacity(self.n_samples);
        let mut tries = 0;
        while nodes.len() < self.n_samples && tries < self.n_samples * 100 {
            tries += 1;
            let x: Vec<f64> = (0..dim).map(|d| lo[d] + rand() * (hi[d] - lo[d])).collect();
            if point_free(&x) {
                nodes.push(x);
            }
        }
        let n = nodes.len();
        // k-PRM* connection count
        let k = ((std::f64::consts::E * (1.0 + 1.0 / dim as f64)) * (n.max(2) as f64).ln()).ceil() as usize;
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            // k nearest neighbours by distance
            let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            order.sort_by(|&a, &b| dist(&nodes[i], &nodes[a]).total_cmp(&dist(&nodes[i], &nodes[b])));
            for &j in order.iter().take(k) {
                let d = dist(&nodes[i], &nodes[j]);
                if d <= self.max_edge && edge_free(&nodes[i], &nodes[j]) {
                    adj[i].push((j, d));
                }
            }
        }
        Some(Roadmap { nodes, adj, dim })
    }
}

impl Roadmap {
    /// Answer a start→goal query by connecting each endpoint to the roadmap and running Dijkstra.
    /// Returns the collision-free waypoint path (including start and goal) or `None`.
    ///
    /// `None` also when an endpoint is malformed: a length other than the roadmap's dimension, or a
    /// non-finite coordinate. Both used to fault inside the nearest-neighbour sort rather than being
    /// reported, one by indexing past the slice and one by unwrapping a `NaN` comparison.
    pub fn query(&self, start: &[f64], goal: &[f64], edge_free: impl Fn(&[f64], &[f64]) -> bool) -> Option<Vec<Vec<f64>>> {
        if start.len() != self.dim || goal.len() != self.dim || !start.iter().chain(goal).all(|x| x.is_finite()) {
            return None;
        }
        let n = self.nodes.len();
        // virtual start = index n, goal = index n+1
        let mut adj: Vec<Vec<(usize, f64)>> = self.adj.clone();
        adj.push(Vec::new()); // start
        adj.push(Vec::new()); // goal
        let (si, gi) = (n, n + 1);
        let mut connect = |from: usize, p: &[f64]| {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| dist(p, &self.nodes[a]).total_cmp(&dist(p, &self.nodes[b])));
            let k = ((std::f64::consts::E * (1.0 + 1.0 / self.dim as f64)) * (n.max(2) as f64).ln()).ceil() as usize;
            for &j in order.iter().take(k) {
                if edge_free(p, &self.nodes[j]) {
                    let d = dist(p, &self.nodes[j]);
                    adj[from].push((j, d));
                    adj[j].push((from, d));
                }
            }
        };
        connect(si, start);
        connect(gi, goal);

        // Dijkstra
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        #[derive(PartialEq)]
        struct St(f64, usize);
        impl Eq for St {}
        impl PartialOrd for St {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for St {
            fn cmp(&self, o: &Self) -> Ordering {
                o.0.partial_cmp(&self.0).unwrap() // min-heap on cost
            }
        }
        let total = n + 2;
        let mut d = vec![f64::INFINITY; total];
        let mut prev = vec![usize::MAX; total];
        let mut heap = BinaryHeap::new();
        d[si] = 0.0;
        heap.push(St(0.0, si));
        while let Some(St(du, u)) = heap.pop() {
            if du > d[u] {
                continue;
            }
            if u == gi {
                break;
            }
            for &(v, w) in &adj[u] {
                if du + w < d[v] {
                    d[v] = du + w;
                    prev[v] = u;
                    heap.push(St(d[v], v));
                }
            }
        }
        if !d[gi].is_finite() {
            return None;
        }
        // reconstruct
        let mut path_idx = vec![gi];
        let mut cur = gi;
        while prev[cur] != usize::MAX {
            cur = prev[cur];
            path_idx.push(cur);
        }
        path_idx.reverse();
        let coord = |i: usize| -> Vec<f64> {
            if i == si {
                start.to_vec()
            } else if i == gi {
                goal.to_vec()
            } else {
                self.nodes[i].clone()
            }
        };
        Some(path_idx.into_iter().map(coord).collect())
    }
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::sdf::{Sdf, SdfScene};
    use nalgebra::Vector3;

    /// A 2-D environment with a circular obstacle; edge/point freedom from an SDF clearance.
    fn env() -> (SdfScene, impl Fn(&[f64]) -> bool, impl Fn(&[f64], &[f64]) -> bool) {
        let scene = SdfScene { prims: vec![Sdf::Sphere { center: Vector3::new(0.5, 0.5, 0.0), radius: 0.25 }] };
        let clr = 0.03;
        let sc1 = scene.clone();
        let point_free = move |p: &[f64]| sc1.distance(&Vector3::new(p[0], p[1], 0.0)) > clr;
        let sc2 = scene.clone();
        let edge_free = move |a: &[f64], b: &[f64]| {
            let steps = 40;
            (0..=steps).all(|k| {
                let t = k as f64 / steps as f64;
                let x = a[0] + t * (b[0] - a[0]);
                let y = a[1] + t * (b[1] - a[1]);
                sc2.distance(&Vector3::new(x, y, 0.0)) > clr
            })
        };
        (scene, point_free, edge_free)
    }

    /// The roadmap, built once, answers multiple start→goal queries with collision-free paths that
    /// route around the obstacle (the direct line is blocked).
    #[test]
    fn prm_multi_query_finds_free_paths() {
        let (scene, point_free, edge_free) = env();
        let prm = PrmStar { n_samples: 500, seed: 0x51, max_edge: 0.35 };
        let roadmap = prm.build(&[0.0, 0.0], &[1.0, 1.0], &point_free, &edge_free).expect("valid bounds");
        assert!(roadmap.nodes.len() > 400, "too few free samples: {}", roadmap.nodes.len());

        // several queries against the SAME roadmap (the multi-query advantage)
        let queries = [
            ([0.05, 0.05], [0.95, 0.95]), // diagonal, straight line blocked by the obstacle
            ([0.05, 0.5], [0.95, 0.5]),   // horizontal through the middle, blocked
            ([0.5, 0.05], [0.5, 0.95]),   // vertical, blocked
        ];
        for (s, g) in queries {
            // the straight line IS blocked (confirms the query is nontrivial)
            assert!(!edge_free(&s, &g), "test setup: straight line should be blocked for {s:?}->{g:?}");
            let path = roadmap.query(&s, &g, &edge_free).expect("no path found");
            // every segment collision-free, endpoints correct
            assert_eq!(path.first().unwrap(), &s.to_vec());
            assert_eq!(path.last().unwrap(), &g.to_vec());
            let mut clears = true;
            for w in path.windows(2) {
                clears &= edge_free(&w[0], &w[1]);
            }
            let mn = path.iter().map(|p| scene.distance(&Vector3::new(p[0], p[1], 0.0))).fold(f64::INFINITY, f64::min);
            eprintln!("PRM query {s:?}->{g:?}: {} waypoints, min clearance {mn:.3}", path.len());
            assert!(clears, "path has a colliding segment");
            assert!(mn > 0.02, "path grazes the obstacle: {mn}");
        }
    }

    /// **A malformed endpoint is refused, not a panic from inside the nearest-neighbour sort.**
    ///
    /// `query` sorts the roadmap by distance to the endpoint. A non-finite coordinate makes every
    /// distance NaN, and the comparison used to be unwrapped, so the library panicked on the caller's
    /// data; a wrong-length endpoint indexed past the end of the slice instead. Both are now `None`.
    #[test]
    fn query_refuses_a_non_finite_or_wrong_length_endpoint() {
        let (_scene, point_free, edge_free) = env();
        let prm = PrmStar { n_samples: 120, seed: 0x51, max_edge: 0.35 };
        let roadmap = prm.build(&[0.0, 0.0], &[1.0, 1.0], &point_free, &edge_free).expect("valid bounds");
        assert!(roadmap.nodes.len() > 50, "fixture needs a usable roadmap: {}", roadmap.nodes.len());
        let ok = roadmap.query(&[0.05, 0.05], &[0.95, 0.95], &edge_free);
        assert!(ok.is_some(), "the control query must still succeed");
        assert!(roadmap.query(&[f64::NAN, 0.05], &[0.95, 0.95], &edge_free).is_none(), "NaN start");
        assert!(roadmap.query(&[0.05, 0.05], &[0.95, f64::INFINITY], &edge_free).is_none(), "infinite goal");
        assert!(roadmap.query(&[0.05], &[0.95, 0.95], &edge_free).is_none(), "short start");
        assert!(roadmap.query(&[0.05, 0.05], &[0.9, 0.9, 0.9], &edge_free).is_none(), "long goal");
    }

    /// `build` takes the same caller bounds `query` validates, and faulted the same way: measured
    /// pre-fix as a panic at the nearest-neighbour sort, `prm.rs:70`.
    #[test]
    fn build_refuses_malformed_bounds() {
        let (_scene, point_free, edge_free) = env();
        let prm = PrmStar { n_samples: 60, seed: 0x51, max_edge: 0.35 };
        assert!(prm.build(&[0.0, 0.0], &[1.0, 1.0], &point_free, &edge_free).is_some(), "control: valid bounds still build");
        assert!(prm.build(&[0.0, f64::NAN], &[1.0, 1.0], &point_free, &edge_free).is_none(), "non-finite lo");
        assert!(prm.build(&[0.0, 0.0], &[1.0, f64::INFINITY], &point_free, &edge_free).is_none(), "non-finite hi");
        assert!(prm.build(&[0.0, 0.0], &[1.0], &point_free, &edge_free).is_none(), "length mismatch");
        assert!(prm.build(&[], &[], &point_free, &edge_free).is_none(), "empty bounds");
    }
}
