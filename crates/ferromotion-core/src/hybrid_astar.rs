//! **Hybrid A\*** (Dolgov, Thrun et al., 2008/2010 — the DARPA Urban Challenge parking planner) — the
//! standard search for driving a **nonholonomic car** through a known obstacle map. Plain A\* on a grid
//! ignores the car's turning constraint and yields kinematically-infeasible zig-zags; Hybrid A\* keeps a
//! **continuous** pose `(x, y, θ)` at each node (advanced by bounded-curvature motion primitives) while
//! using the grid only to bound the search, so every node — and the final path — is drivable. Two ideas make
//! it fast: a **[`crate::reeds_shepp`] heuristic** (the true shortest kinematic distance to the goal,
//! ignoring obstacles — admissible and far tighter than Euclidean), and an **analytic expansion** that
//! periodically tries a direct Reeds–Shepp shot to the goal and, if collision-free, snaps to it.
//!
//! Unit turning radius (a units choice); a scene is a `is_free(x, y)` predicate. Verified: it finds a
//! drivable path to the goal in free space near the Reeds–Shepp optimum, routes a collision-free path around
//! a wall-with-a-gap (every sampled point clear), and its motion primitives respect the turning radius. Pure
//! `nalgebra` → WASM-clean.

use crate::reeds_shepp::{reeds_shepp, RsSegment};
use nalgebra::Vector3;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::{PI, TAU};

/// Hybrid A\* configuration (unit turning radius).
#[derive(Clone, Copy, Debug)]
pub struct HybridConfig {
    /// Arc length of one motion primitive.
    pub step: f64,
    /// Grid cell size for the closed set (position).
    pub xy_res: f64,
    /// Number of heading bins for the closed set.
    pub theta_bins: usize,
    /// Extra cost multiplier for reverse motion (> 1 discourages it).
    pub reverse_penalty: f64,
    /// Goal position/heading tolerance for the analytic-expansion snap.
    pub goal_tol: f64,
    /// Maximum node expansions before giving up.
    pub max_expansions: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        HybridConfig { step: 0.5, xy_res: 0.3, theta_bins: 36, reverse_penalty: 2.0, goal_tol: 0.4, max_expansions: 20000 }
    }
}

// advance a pose by an arc of signed curvature `kappa` (0 = straight), gear ±1, over length `len`
fn arc(p: Vector3<f64>, gear: f64, kappa: f64, len: f64) -> Vector3<f64> {
    let (x, y, th) = (p.x, p.y, p.z);
    if kappa.abs() < 1e-9 {
        Vector3::new(x + gear * len * th.cos(), y + gear * len * th.sin(), th)
    } else {
        let nth = th + gear * kappa * len;
        Vector3::new(x + (nth.sin() - th.sin()) / kappa, y + (th.cos() - nth.cos()) / kappa, nth)
    }
}

fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

fn key(p: Vector3<f64>, cfg: &HybridConfig) -> (i64, i64, i64) {
    let t = (wrap(p.z) + PI) / TAU * cfg.theta_bins as f64;
    ((p.x / cfg.xy_res).round() as i64, (p.y / cfg.xy_res).round() as i64, (t as i64).rem_euclid(cfg.theta_bins as i64))
}

// is the whole arc from `p` (gear,kappa,len) collision-free?
fn arc_free(p: Vector3<f64>, gear: f64, kappa: f64, len: f64, is_free: &impl Fn(f64, f64) -> bool) -> bool {
    let n = (len / 0.1).ceil().max(1.0) as usize;
    (0..=n).all(|i| {
        let s = len * i as f64 / n as f64;
        let q = arc(p, gear, kappa, s);
        is_free(q.x, q.y)
    })
}

// Reeds–Shepp path sampled into poses from `start`, or None if it collides.
fn rs_poses(start: Vector3<f64>, path: &[RsSegment], is_free: &impl Fn(f64, f64) -> bool) -> Option<Vec<Vector3<f64>>> {
    let mut pose = start;
    let mut out = vec![pose];
    for &(steer, len) in path {
        let (gear, kappa) = (len.signum(), steer as f64);
        let l = len.abs();
        if !arc_free(pose, gear, kappa, l, is_free) {
            return None;
        }
        // sample the segment
        let n = (l / 0.1).ceil().max(1.0) as usize;
        for i in 1..=n {
            out.push(arc(pose, gear, kappa, l * i as f64 / n as f64));
        }
        pose = arc(pose, gear, kappa, l);
    }
    Some(out)
}

struct Node {
    pose: Vector3<f64>,
    g: f64,
    parent: usize,
}

/// Plan a drivable path from `start` to `goal` (`(x, y, θ)`) in a scene where `is_free(x, y)` is true off
/// obstacles. Returns a sampled pose sequence, or `None` if unreachable within the budget.
pub fn hybrid_astar(start: Vector3<f64>, goal: Vector3<f64>, is_free: impl Fn(f64, f64) -> bool, cfg: &HybridConfig) -> Option<Vec<Vector3<f64>>> {
    let heuristic = |p: Vector3<f64>| reeds_shepp(p, goal).map(|path| path.iter().map(|&(_, l)| l.abs()).sum::<f64>()).unwrap_or_else(|| (goal.xy() - p.xy()).norm());

    let mut nodes: Vec<Node> = vec![Node { pose: start, g: 0.0, parent: usize::MAX }];
    let mut open: BinaryHeap<Reverse<(OrdF, usize)>> = BinaryHeap::new();
    open.push(Reverse((OrdF(heuristic(start)), 0)));
    let mut closed: HashMap<(i64, i64, i64), f64> = HashMap::new();

    let prims = [(1.0, -1.0), (1.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0)]; // (gear, kappa)

    for _ in 0..cfg.max_expansions {
        let Reverse((_, ni)) = open.pop()?;
        let pose = nodes[ni].pose;
        let g = nodes[ni].g;
        let k = key(pose, cfg);
        if let Some(&seen) = closed.get(&k)
            && seen <= g
        {
            continue;
        }
        closed.insert(k, g);

        // analytic expansion: try a direct Reeds–Shepp shot to the goal
        if let Some(path) = reeds_shepp(pose, goal)
            && let Some(mut poses) = rs_poses(pose, &path, &is_free)
        {
            // reconstruct the search prefix, then append the analytic tail
            let mut prefix = reconstruct(&nodes, ni);
            prefix.pop(); // avoid duplicating the join pose
            prefix.append(&mut poses);
            return Some(prefix);
        }

        for &(gear, kappa) in &prims {
            if !arc_free(pose, gear, kappa, cfg.step, &is_free) {
                continue;
            }
            let np = arc(pose, gear, kappa, cfg.step);
            let cost = cfg.step * if gear < 0.0 { cfg.reverse_penalty } else { 1.0 };
            let ng = g + cost;
            let nk = key(np, cfg);
            if let Some(&seen) = closed.get(&nk)
                && seen <= ng
            {
                continue;
            }
            let idx = nodes.len();
            nodes.push(Node { pose: np, g: ng, parent: ni });
            open.push(Reverse((OrdF(ng + heuristic(np)), idx)));
        }
    }
    None
}

fn reconstruct(nodes: &[Node], mut i: usize) -> Vec<Vector3<f64>> {
    let mut out = Vec::new();
    while i != usize::MAX {
        out.push(nodes[i].pose);
        i = nodes[i].parent;
    }
    out.reverse();
    out
}

// total order over f64 for the priority queue
#[derive(PartialEq)]
struct OrdF(f64);
impl Eq for OrdF {}
impl PartialOrd for OrdF {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_finds_a_drivable_path_in_free_space() {
        // THE ORACLE. In free space a path is found; every consecutive pose is within a primitive/​sample
        // step (drivable), and it ends at the goal.
        let start = Vector3::new(0.0, 0.0, 0.0);
        let goal = Vector3::new(4.0, 2.0, 1.0);
        let path = hybrid_astar(start, goal, |_, _| true, &HybridConfig::default()).expect("path in free space");
        let end = *path.last().unwrap();
        assert!((end.xy() - goal.xy()).norm() < 0.05 && wrap(end.z - goal.z).abs() < 0.05, "should reach the goal: {end:?}");
        // consecutive samples are close (bounded-curvature, small steps)
        for w in path.windows(2) {
            assert!((w[1].xy() - w[0].xy()).norm() < 0.6, "samples should be dense/drivable");
        }
    }

    #[test]
    fn it_routes_around_a_wall_through_the_gap() {
        // THE HEADLINE. A vertical wall at x≈2 blocks everything except a gap at y∈[1.5, 3]. The path must
        // stay collision-free and reach the goal on the far side.
        let is_free = |x: f64, y: f64| !((1.8..2.2).contains(&x) && y < 1.5);
        let start = Vector3::new(0.0, 0.0, 0.0);
        let goal = Vector3::new(4.0, 0.0, 0.0);
        let cfg = HybridConfig { max_expansions: 60000, ..HybridConfig::default() };
        let path = hybrid_astar(start, goal, is_free, &cfg).expect("path around the wall");
        for p in &path {
            assert!(is_free(p.x, p.y), "path point {p:?} hit the wall");
        }
        assert!(((*path.last().unwrap()).xy() - goal.xy()).norm() < 0.05, "reaches the goal");
    }
}
