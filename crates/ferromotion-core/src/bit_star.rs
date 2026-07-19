//! **BIT\* — Batch Informed Trees** (Gammell, Srinivasa & Barfoot, ICRA 2015): anytime,
//! asymptotically-optimal sampling-based motion planning. BIT\* processes samples in **batches** and grows a
//! tree by expanding edges in order of an **admissible cost heuristic** `g(v) + ĉ(v,x) + ĥ(x)` — a
//! lazy-shortest-path search on a growing random geometric graph, with collision checks deferred until an
//! edge is actually pulled from the queue. Once a first solution of cost `c_best` is found, all further
//! sampling is drawn from the **informed set** — the hyper-ellipse `{x : ‖x−start‖ + ‖x−goal‖ ≤ c_best}`
//! that provably contains every state that could improve the path — so the estimate marches toward the
//! optimum instead of sampling the whole space. The queue's **key-pruning** (stop once the best remaining
//! edge cannot beat `c_best`) bounds the work per batch.
//!
//! Planar (2-D) here. Uses [`crate::spatial::KdTree`] for the radius-limited neighbour lookups (embedding
//! points at `z = 0`), so it is a consumer of the shared spatial index. Verified: in free space the returned
//! cost converges to the straight-line optimum; the cost is monotonically non-increasing across batches
//! (the anytime property); and around a wall-with-a-gap it returns a collision-free detour longer than the
//! blocked straight line. Deterministic (seeded) → WASM-clean.

use crate::spatial::KdTree;
use nalgebra::{Matrix2, Vector2, Vector3};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A total order over `f64` for heap keys (no NaNs in play).
#[derive(PartialEq)]
struct Ord64(f64);
impl Eq for Ord64 {}
impl PartialOrd for Ord64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ord64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A tiny deterministic SplitMix64, so planning is reproducible.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A BIT\* planner over an axis-aligned 2-D box `[lo, hi]`, adding `batch` samples per batch and connecting
/// states within `radius` (the RGG connection radius).
#[derive(Clone, Debug)]
pub struct BitStar {
    pub lo: Vector2<f64>,
    pub hi: Vector2<f64>,
    pub batch: usize,
    pub radius: f64,
}

struct Node {
    p: Vector2<f64>,
    g: f64,
    parent: Option<usize>,
    in_tree: bool,
}

/// Push every outgoing edge of vertex `v` that could still improve its neighbour and beat `c_best`, keyed
/// by the admissible heuristic `g(v) + ĉ(v,x) + ĥ(x)`. A free fn (not a closure) so it never holds a borrow
/// of `nodes` across the tree mutations in the search loop.
#[allow(clippy::too_many_arguments)]
fn expand(kd: &KdTree, radius: f64, nodes: &[Node], v: usize, c_best: f64, goal: Vector2<f64>, heap: &mut BinaryHeap<Reverse<(Ord64, usize, usize)>>) {
    let pv = nodes[v].p;
    for x in kd.within_radius(&Vector3::new(pv.x, pv.y, 0.0), radius) {
        if x == v {
            continue;
        }
        let c_edge = (pv - nodes[x].p).norm();
        let h = (goal - nodes[x].p).norm();
        if nodes[v].g + c_edge < nodes[x].g && nodes[v].g + c_edge + h < c_best {
            heap.push(Reverse((Ord64(nodes[v].g + c_edge + h), v, x)));
        }
    }
}

impl BitStar {
    /// Plan from `start` to `goal` in a space where `is_free(a, b)` reports whether the straight segment
    /// `a→b` is collision-free. Runs `batches` batches. Returns `(path, cost)` — the sequence of waypoints
    /// (start … goal) and its length — or `None` if no solution was found. `cost_history` collects `c_best`
    /// after each batch (the anytime curve). Deterministic in `seed`.
    pub fn plan(
        &self,
        start: Vector2<f64>,
        goal: Vector2<f64>,
        is_free: impl Fn(&Vector2<f64>, &Vector2<f64>) -> bool,
        batches: usize,
        seed: u64,
        cost_history: &mut Vec<f64>,
    ) -> Option<(Vec<Vector2<f64>>, f64)> {
        let mut rng = Lcg::new(seed);
        let mut nodes: Vec<Node> = vec![
            Node { p: start, g: 0.0, parent: None, in_tree: true }, // 0 = start
            Node { p: goal, g: f64::INFINITY, parent: None, in_tree: false }, // 1 = goal (a sample)
        ];
        const GOAL: usize = 1;
        let g_hat = |p: &Vector2<f64>| (p - start).norm();
        let h_hat = |p: &Vector2<f64>| (goal - p).norm();
        let c_min = (goal - start).norm();
        let centre = 0.5 * (start + goal);
        // rotation aligning the ellipse major axis with start→goal (2-D)
        let rot = if c_min > 1e-12 {
            let a1 = (goal - start) / c_min;
            Matrix2::new(a1.x, -a1.y, a1.y, a1.x)
        } else {
            Matrix2::identity()
        };
        let mut c_best = f64::INFINITY;

        for _ in 0..batches {
            // --- prune samples that cannot improve, then draw a fresh batch ---
            nodes.retain(|n| n.in_tree || g_hat(&n.p) + h_hat(&n.p) < c_best);
            let mut added = 0;
            let mut guard = 0;
            while added < self.batch && guard < self.batch * 200 {
                guard += 1;
                let p = if c_best.is_finite() {
                    // sample the informed hyper-ellipse
                    let r1 = c_best / 2.0;
                    let r2 = (c_best * c_best - c_min * c_min).max(0.0).sqrt() / 2.0;
                    let (mut x, mut y);
                    loop {
                        x = 2.0 * rng.unif() - 1.0;
                        y = 2.0 * rng.unif() - 1.0;
                        if x * x + y * y <= 1.0 {
                            break;
                        }
                    }
                    centre + rot * Vector2::new(r1 * x, r2 * y)
                } else {
                    Vector2::new(self.lo.x + (self.hi.x - self.lo.x) * rng.unif(), self.lo.y + (self.hi.y - self.lo.y) * rng.unif())
                };
                if p.x < self.lo.x || p.x > self.hi.x || p.y < self.lo.y || p.y > self.hi.y {
                    continue;
                }
                nodes.push(Node { p, g: f64::INFINITY, parent: None, in_tree: false });
                added += 1;
            }

            // spatial index over all node positions (z = 0), rebuilt per batch (positions are fixed within it)
            let kd = KdTree::build(nodes.iter().map(|n| Vector3::new(n.p.x, n.p.y, 0.0)).collect());

            // --- lazy edge-queue search: process edges by heuristic key g(v)+ĉ(v,x)+ĥ(x) ---
            // seed the queue with the outgoing edges of every in-tree vertex
            let mut heap: BinaryHeap<Reverse<(Ord64, usize, usize)>> = BinaryHeap::new();
            for v in 0..nodes.len() {
                if nodes[v].in_tree {
                    expand(&kd, self.radius, &nodes, v, c_best, goal, &mut heap);
                }
            }

            while let Some(Reverse((Ord64(key), v, x))) = heap.pop() {
                // key-pruning: the best remaining edge can't beat the incumbent ⇒ this batch is done
                if key >= c_best {
                    break;
                }
                let c_edge = (nodes[v].p - nodes[x].p).norm();
                // still an improvement with current costs (lazy staleness re-check)?
                if nodes[v].g + c_edge < nodes[x].g && nodes[v].g + c_edge + h_hat(&nodes[x].p) < c_best {
                    // defer the collision check until now
                    if is_free(&nodes[v].p, &nodes[x].p) {
                        nodes[x].g = nodes[v].g + c_edge;
                        nodes[x].parent = Some(v);
                        nodes[x].in_tree = true;
                        if x == GOAL {
                            c_best = nodes[GOAL].g;
                        } else {
                            expand(&kd, self.radius, &nodes, x, c_best, goal, &mut heap); // expand the newly-improved vertex
                        }
                    }
                }
            }
            cost_history.push(c_best);
        }

        if !nodes[GOAL].g.is_finite() {
            return None;
        }
        // reconstruct the path
        let mut path = Vec::new();
        let mut cur = Some(GOAL);
        while let Some(i) = cur {
            path.push(nodes[i].p);
            cur = nodes[i].parent;
        }
        path.reverse();
        Some((path, nodes[GOAL].g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_free_space_the_cost_converges_to_the_straight_line() {
        // THE ORACLE. With no obstacles the optimal path is the straight segment; BIT* should approach its
        // length ‖goal−start‖ as batches accumulate.
        let planner = BitStar { lo: Vector2::new(-1.0, -1.0), hi: Vector2::new(6.0, 6.0), batch: 100, radius: 1.6 };
        let start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(5.0, 5.0);
        let straight = (goal - start).norm();
        let mut hist = Vec::new();
        let (_path, cost) = planner.plan(start, goal, |_, _| true, 14, 7, &mut hist).expect("should find a path");
        assert!(cost >= straight - 1e-9, "cannot beat the straight line: {cost} < {straight}");
        assert!(cost < straight * 1.10, "should converge near the straight-line optimum: {cost} vs {straight}");
    }

    #[test]
    fn the_cost_is_monotonically_non_increasing_across_batches() {
        // THE ANYTIME PROPERTY. Each batch can only refine the incumbent solution, never worsen it.
        let planner = BitStar { lo: Vector2::new(-1.0, -1.0), hi: Vector2::new(6.0, 6.0), batch: 80, radius: 1.6 };
        let mut hist = Vec::new();
        planner.plan(Vector2::new(0.0, 0.0), Vector2::new(5.0, 5.0), |_, _| true, 14, 3, &mut hist);
        for w in hist.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "cost must not increase batch-to-batch: {} → {}", w[0], w[1]);
        }
        assert!(hist.last().unwrap().is_finite(), "a solution should be found");
    }

    #[test]
    fn it_routes_around_a_wall_through_the_gap() {
        // THE HEADLINE. A vertical wall at x≈2.5 blocks the straight line except through a gap at y≥4. BIT*
        // must return a collision-free detour longer than the blocked straight line, with every segment
        // clearing the wall.
        let wall_x = 2.5;
        let gap_lo = 4.0;
        let is_free = move |a: &Vector2<f64>, b: &Vector2<f64>| -> bool {
            if (a.x - wall_x) * (b.x - wall_x) > 0.0 {
                return true; // same side, no crossing
            }
            if (a.x - b.x).abs() < 1e-12 {
                return true;
            }
            let t = (wall_x - a.x) / (b.x - a.x);
            if !(0.0..=1.0).contains(&t) {
                return true;
            }
            let y = a.y + t * (b.y - a.y);
            y >= gap_lo // crossing allowed only through the gap
        };
        let planner = BitStar { lo: Vector2::new(-0.5, -1.0), hi: Vector2::new(5.5, 5.5), batch: 160, radius: 1.6 };
        let start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(5.0, 0.0);
        let mut hist = Vec::new();
        let (path, cost) = planner.plan(start, goal, is_free, 22, 11, &mut hist).expect("should find a detour");
        for w in path.windows(2) {
            assert!(is_free(&w[0], &w[1]), "path segment {:?}→{:?} hits the wall", w[0], w[1]);
        }
        assert!(cost > (goal - start).norm() + 0.5, "a real detour should exceed the straight line: {cost}");
    }
}
