//! **Grid A\* shortest-path search** — the workhorse 2-D planner over an occupancy / cost grid. Where
//! [`crate::bit_star`] samples an optimal continuous path and [`crate::hybrid_astar`] respects a car's
//! turning constraint, plain grid A\* finds the shortest **cell** path for a holonomic agent — exactly what
//! a mobile robot's cost-map navigator (and the query side of an [`crate::OccupancyGrid`]) needs. It is A\*
//! with an **8-connected** neighbourhood and the **octile** heuristic (the exact obstacle-free distance with
//! diagonal moves), which is admissible and consistent, so the path returned is optimal. Setting the
//! heuristic aside recovers Dijkstra.
//!
//! Verified: in open space the path length equals the octile distance (optimal); it routes a collision-free
//! path around a wall; and a fully-blocked goal returns `None`. Pure Rust → WASM-clean.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

const SQRT2: f64 = std::f64::consts::SQRT_2;

/// The octile distance between two cells (exact shortest 8-connected distance ignoring obstacles).
pub fn octile(a: (i32, i32), b: (i32, i32)) -> f64 {
    let dx = (a.0 - b.0).abs() as f64;
    let dy = (a.1 - b.1).abs() as f64;
    (dx - dy).abs() + SQRT2 * dx.min(dy)
}

/// A* over a `width × height` grid where `is_free(i, j)` is true off obstacles. 8-connected with the octile
/// heuristic; diagonal moves cost `√2` and may not cut a blocked corner. Returns the optimal cell path from
/// `start` to `goal` (inclusive), or `None` if unreachable.
pub fn astar_grid(width: usize, height: usize, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32)) -> Option<Vec<(i32, i32)>> {
    let idx = |i: i32, j: i32| (j as usize) * width + (i as usize);
    let in_bounds = |i: i32, j: i32| i >= 0 && j >= 0 && (i as usize) < width && (j as usize) < height;
    if !in_bounds(start.0, start.1) || !in_bounds(goal.0, goal.1) || !is_free(start.0, start.1) || !is_free(goal.0, goal.1) {
        return None;
    }
    let n = width * height;
    let mut g = vec![f64::INFINITY; n];
    let mut came: Vec<i32> = vec![-1; n]; // parent cell index, −1 = none
    let mut closed = vec![false; n];
    g[idx(start.0, start.1)] = 0.0;
    let mut open: BinaryHeap<Reverse<(OrdF, i32, i32)>> = BinaryHeap::new();
    open.push(Reverse((OrdF(octile(start, goal)), start.0, start.1)));

    let neighbours = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)];

    while let Some(Reverse((_, ci, cj))) = open.pop() {
        let cur = idx(ci, cj);
        if closed[cur] {
            continue;
        }
        closed[cur] = true;
        if (ci, cj) == goal {
            // reconstruct
            let mut path = vec![(ci, cj)];
            let mut k = cur as i32;
            while came[k as usize] >= 0 {
                k = came[k as usize];
                path.push(((k as usize % width) as i32, (k as usize / width) as i32));
            }
            path.reverse();
            return Some(path);
        }
        for &(di, dj, cost) in &neighbours {
            let (ni, nj) = (ci + di, cj + dj);
            if !in_bounds(ni, nj) || !is_free(ni, nj) {
                continue;
            }
            // no cutting blocked corners on a diagonal move
            if di != 0 && dj != 0 && (!is_free(ci + di, cj) || !is_free(ci, cj + dj)) {
                continue;
            }
            let ng = g[cur] + cost;
            let nidx = idx(ni, nj);
            if ng < g[nidx] {
                g[nidx] = ng;
                came[nidx] = cur as i32;
                open.push(Reverse((OrdF(ng + octile((ni, nj), goal)), ni, nj)));
            }
        }
    }
    None
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

/// The total length of a cell path (orthogonal steps cost 1, diagonal `√2`).
pub fn path_length(path: &[(i32, i32)]) -> f64 {
    path.windows(2).map(|w| octile(w[0], w[1])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_space_gives_the_optimal_octile_path() {
        // THE ORACLE. In an empty grid the shortest path length equals the octile distance.
        let path = astar_grid(10, 10, |_, _| true, (0, 0), (9, 6)).unwrap();
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(9, 6)));
        assert!((path_length(&path) - octile((0, 0), (9, 6))).abs() < 1e-9, "should be optimal: {} vs {}", path_length(&path), octile((0, 0), (9, 6)));
    }

    #[test]
    fn it_routes_around_a_wall() {
        // THE HEADLINE. A vertical wall at i=5 spanning j=0..8 (gap at j=8,9) forces a detour; the path stays
        // off the wall and reaches the goal.
        let is_free = |i: i32, j: i32| !(i == 5 && j < 8);
        let path = astar_grid(12, 12, is_free, (0, 0), (11, 0)).unwrap();
        for &(i, j) in &path {
            assert!(is_free(i, j), "path cell ({i},{j}) hit the wall");
        }
        assert_eq!(path.last(), Some(&(11, 0)));
        // the detour is longer than the blocked straight line
        assert!(path_length(&path) > 11.0, "detour should exceed the direct distance: {}", path_length(&path));
    }

    #[test]
    fn a_blocked_goal_returns_none() {
        // Wall out the whole right half; a goal on the far side is unreachable.
        let is_free = |i: i32, _j: i32| i != 6;
        assert!(astar_grid(12, 12, is_free, (0, 0), (11, 5)).is_none(), "sealed goal should be unreachable");
    }

    #[test]
    fn start_equals_goal_is_a_single_cell() {
        let path = astar_grid(5, 5, |_, _| true, (2, 2), (2, 2)).unwrap();
        assert_eq!(path, vec![(2, 2)]);
    }
}
