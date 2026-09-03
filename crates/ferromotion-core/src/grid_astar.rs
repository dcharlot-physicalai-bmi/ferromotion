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

/// Manhattan distance `|Δi| + |Δj|`: the exact obstacle-free cost under [`Connectivity::Four`], and
/// inadmissible under [`Connectivity::Eight`], where it overestimates a diagonal. Pair it with the
/// connectivity it belongs to.
pub fn manhattan(a: (i32, i32), b: (i32, i32)) -> f64 {
    ((a.0 - b.0).abs() + (a.1 - b.1).abs()) as f64
}

/// How cells connect: four orthogonal moves of cost 1, or those plus four diagonals of cost `√2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connectivity {
    Four,
    Eight,
}

impl Connectivity {
    /// The step set `(di, dj, cost)`. Shared by every grid planner in the crate so the costs cannot drift.
    pub fn steps(self) -> &'static [(i32, i32, f64)] {
        const FOUR: [(i32, i32, f64); 4] = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)];
        const EIGHT: [(i32, i32, f64); 8] = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)];
        match self {
            Connectivity::Four => &FOUR,
            Connectivity::Eight => &EIGHT,
        }
    }

    /// The admissible, consistent heuristic for this step set.
    pub fn heuristic(self, a: (i32, i32), b: (i32, i32)) -> f64 {
        match self {
            Connectivity::Four => manhattan(a, b),
            Connectivity::Eight => octile(a, b),
        }
    }
}

/// Whether a step `(di, dj)` from `(ci, cj)` is allowed: the destination is free, and a diagonal may not
/// **cut a blocked corner** — both orthogonal neighbours it passes between must be free too.
///
/// One function, used by every grid planner here, because the corner rule is the pitfall every planner
/// spec names and four private copies of it would drift. `is_free` must return `false` out of bounds.
pub fn can_step(is_free: &impl Fn(i32, i32) -> bool, ci: i32, cj: i32, di: i32, dj: i32) -> bool {
    let (ni, nj) = (ci + di, cj + dj);
    if !is_free(ni, nj) {
        return false;
    }
    if di != 0 && dj != 0 && (!is_free(ci + di, cj) || !is_free(ci, cj + dj)) {
        return false;
    }
    true
}

/// A* over a `width × height` grid where `is_free(i, j)` is true off obstacles. 8-connected with the octile
/// heuristic; diagonal moves cost `√2` and may not cut a blocked corner. Returns the optimal cell path from
/// `start` to `goal` (inclusive), or `None` if unreachable.
pub fn astar_grid(width: usize, height: usize, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32)) -> Option<Vec<(i32, i32)>> {
    astar_grid_conn(width, height, Connectivity::Eight, is_free, start, goal)
}

/// [`astar_grid`] with the step set chosen: the same search, with the heuristic that matches it.
///
/// This is the baseline the crate's other grid planners are tested against, so it is the one place the
/// neighbourhood, costs and corner rule are defined — through [`Connectivity`] and [`can_step`].
pub fn astar_grid_conn(width: usize, height: usize, conn: Connectivity, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32)) -> Option<Vec<(i32, i32)>> {
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
    open.push(Reverse((OrdF(conn.heuristic(start, goal)), start.0, start.1)));
    // bounds live inside the freedom test so `can_step`'s corner rule sees the map edge as blocked
    let free = |i: i32, j: i32| in_bounds(i, j) && is_free(i, j);

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
        for &(di, dj, cost) in conn.steps() {
            if !can_step(&free, ci, cj, di, dj) {
                continue;
            }
            let (ni, nj) = (ci + di, cj + dj);
            let ng = g[cur] + cost;
            let nidx = idx(ni, nj);
            if ng < g[nidx] {
                g[nidx] = ng;
                came[nidx] = cur as i32;
                open.push(Reverse((OrdF(ng + conn.heuristic((ni, nj), goal)), ni, nj)));
            }
        }
    }
    None
}

/// Total order over `f64` for a priority queue: the tie-break every grid planner here shares.
#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) struct OrdF(pub(crate) f64);
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

    /// 4-connected: corner to corner of an empty 5x5 is 8 unit steps (Manhattan), where 8-connected is
    /// `4·√2 ≈ 5.657`. The two must differ, or the connectivity argument is decorative.
    #[test]
    fn four_connected_costs_manhattan_and_differs_from_eight() {
        let free = |i: i32, j: i32| (0..5).contains(&i) && (0..5).contains(&j);
        let p4 = astar_grid_conn(5, 5, Connectivity::Four, free, (0, 0), (4, 4)).expect("open grid");
        let p8 = astar_grid_conn(5, 5, Connectivity::Eight, free, (0, 0), (4, 4)).expect("open grid");
        assert!((path_length(&p4) - 8.0).abs() < 1e-12, "4-conn cost {}", path_length(&p4));
        assert!((path_length(&p8) - 4.0 * SQRT2).abs() < 1e-12, "8-conn cost {}", path_length(&p8));
        // every 4-connected step is orthogonal
        assert!(p4.windows(2).all(|w| (w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs() == 1));
    }

    /// The corner rule through the shared helper: a diagonal between two blocked orthogonals is refused,
    /// and so is any step off the map, because the freedom test carries the bounds.
    #[test]
    fn can_step_refuses_corner_cuts_and_the_map_edge() {
        // 3x3 with (1,0) and (0,1) blocked: the diagonal (0,0)->(1,1) would squeeze between them
        let free = |i: i32, j: i32| (0..3).contains(&i) && (0..3).contains(&j) && !((i, j) == (1, 0) || (i, j) == (0, 1));
        assert!(!can_step(&free, 0, 0, 1, 1), "diagonal between two blocked orthogonals must be refused");
        assert!(can_step(&free, 1, 1, 1, 1), "an open diagonal is fine");
        assert!(!can_step(&free, 2, 2, 1, 0), "stepping off the map is refused when is_free carries bounds");
        assert!(astar_grid_conn(3, 3, Connectivity::Eight, free, (0, 0), (2, 2)).is_none(), "with both orthogonals blocked the start is sealed");
    }

}
