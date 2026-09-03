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

    // ---- Spec fixtures A1-A4 (Hart, Nilsson & Raphael 1968, doi:10.1109/TSSC.1968.300136, on the grid) ----
    //
    // The drawings, start/goal cells, connectivities and expected costs are the six `test_fixtures` of the
    // "A* on a grid (baseline)" planner specification; each expected cost is a hand computation the spec
    // states (Manhattan or octile through the gap) that the spec says it confirmed by Dijkstra / BFS.
    // Fixture coordinates are `(column, row-from-top)`; [`crate::OccupancyGrid::from_rows`] puts the top
    // drawn row at `j = height - 1`, so `spec_cell` flips the row before the planner sees it, and the
    // planner is handed `is_free = !grid.blocked(i, j)`, which carries the map bounds as blocked.

    use crate::OccupancyGrid;

    /// The drawing as a grid, unit cells at the origin, so cell indices are the drawing's columns/rows.
    fn spec_grid(rows: &[&str]) -> OccupancyGrid {
        OccupancyGrid::from_rows(rows, 1.0, 0.0, 0.0).expect("the spec drawings are rectangular and non-empty")
    }

    /// A spec `(column, row-from-top)` pair as the grid's `(i, j)`, whose `j` counts up from the bottom.
    fn spec_cell(grid: &OccupancyGrid, xy: (i32, i32)) -> (i32, i32) {
        (xy.0, grid.height as i32 - 1 - xy.1)
    }

    /// Plan on a spec grid between spec cells, through `OccupancyGrid::blocked` as the freedom test.
    fn spec_plan(grid: &OccupancyGrid, conn: Connectivity, start_xy: (i32, i32), goal_xy: (i32, i32)) -> Option<Vec<(i32, i32)>> {
        let is_free = |i: i32, j: i32| !grid.blocked(i as i64, j as i64);
        astar_grid_conn(grid.width, grid.height, conn, is_free, spec_cell(grid, start_xy), spec_cell(grid, goal_xy))
    }

    /// Every fixture's path must start and end where asked and run only on free cells (spec pitfall:
    /// "grid_path_length(path) must equal the returned cost"; here the path IS the returned object, so
    /// its length is the cost and a parent-pointer bug shows up as a wrong or broken path).
    fn assert_well_formed(grid: &OccupancyGrid, path: &[(i32, i32)], start: (i32, i32), goal: (i32, i32)) {
        assert_eq!(path.first(), Some(&start), "path must begin at the start");
        assert_eq!(path.last(), Some(&goal), "path must end at the goal");
        for &(i, j) in path {
            assert!(!grid.blocked(i as i64, j as i64), "path cell ({i},{j}) is blocked or off the map");
        }
        // consecutive cells are one king move apart, so `path_length` is summing real steps
        assert!(path.windows(2).all(|w| (w[1].0 - w[0].0).abs() <= 1 && (w[1].1 - w[0].1).abs() <= 1 && w[0] != w[1]), "path has a non-adjacent step: {path:?}");
    }

    const EMPTY_5X5: [&str; 5] = [".....", ".....", ".....", ".....", "....."];

    /// The 7x7 wall in column 3 with its single gap at (3,6), the bottom drawn row.
    const WALL_7X7: [&str; 7] = ["...#...", "...#...", "...#...", "...#...", "...#...", "...#...", "......."];

    /// A1-4: 5x5 empty, corner to corner, 4-connected. Expected 8 = Manhattan 4+4 (spec hand computation).
    #[test]
    fn spec_a1_4_empty_grid_four_connected_costs_manhattan() {
        let grid = spec_grid(&EMPTY_5X5);
        // non-vacuous: the grid really is empty and the endpoints really are 8 Manhattan apart
        assert!((0..5).all(|i| (0..5).all(|j| !grid.blocked(i, j))), "fixture must be empty");
        assert_eq!(manhattan(spec_cell(&grid, (0, 0)), spec_cell(&grid, (4, 4))), 8.0);
        let path = spec_plan(&grid, Connectivity::Four, (0, 0), (4, 4)).expect("A1-4 is reachable");
        assert_well_formed(&grid, &path, spec_cell(&grid, (0, 0)), spec_cell(&grid, (4, 4)));
        assert!((path_length(&path) - 8.0).abs() < 1e-9, "A1-4 cost {} vs 8", path_length(&path));
    }

    /// A1-8: the same corners, 8-connected. Expected 4·√2 = 5.656854249492381 (spec hand computation: four
    /// diagonal moves, the octile distance with dx = dy = 4). It must differ from A1-4's 8, or the
    /// connectivity argument is decorative.
    #[test]
    fn spec_a1_8_empty_grid_eight_connected_costs_octile() {
        let grid = spec_grid(&EMPTY_5X5);
        let expected = 5.656_854_249_492_381_f64;
        // non-vacuous: the 4- and 8-connected answers are different numbers, and the constant is 4√2
        assert!((expected - 4.0 * SQRT2).abs() < 1e-12);
        assert!((expected - 8.0).abs() > 2.0, "4- and 8-connected costs must differ");
        let path = spec_plan(&grid, Connectivity::Eight, (0, 0), (4, 4)).expect("A1-8 is reachable");
        assert_well_formed(&grid, &path, spec_cell(&grid, (0, 0)), spec_cell(&grid, (4, 4)));
        assert!((path_length(&path) - expected).abs() < 1e-9, "A1-8 cost {} vs {expected}", path_length(&path));
    }

    /// A2-4: 7x7 wall in column 3 with one gap at (3,6), 4-connected, (0,3) to (6,3). Expected 12 =
    /// Manhattan 6 to the gap plus 6 from it (spec hand computation, confirmed by BFS in the spec).
    #[test]
    fn spec_a2_4_wall_with_a_gap_four_connected() {
        let grid = spec_grid(&WALL_7X7);
        let (start, goal) = (spec_cell(&grid, (0, 3)), spec_cell(&grid, (6, 3)));
        // non-vacuous: the wall stands between start and goal (the direct Manhattan route, 6, is cut) and
        // the gap cell is open, so 12 can only come from the detour
        let (wi, wj) = spec_cell(&grid, (3, 3));
        assert!(grid.blocked(wi as i64, wj as i64), "wall cell (3,3) must be blocked");
        let (gi, gj) = spec_cell(&grid, (3, 6));
        assert!(!grid.blocked(gi as i64, gj as i64), "gap cell (3,6) must be free");
        assert_eq!(manhattan(start, goal), 6.0);
        let path = spec_plan(&grid, Connectivity::Four, (0, 3), (6, 3)).expect("A2-4 is reachable through the gap");
        assert_well_formed(&grid, &path, start, goal);
        assert!(path.contains(&(gi, gj)), "the only way across is the gap");
        assert!((path_length(&path) - 12.0).abs() < 1e-9, "A2-4 cost {} vs 12", path_length(&path));
    }

    /// A2-8: the same wall, 8-connected. Expected 4 + 4·√2 = 9.65685424949238 (spec hand computation: two
    /// diagonals to (2,5), three orthogonals through (2,6),(3,6),(4,6), two diagonals to (6,4), one
    /// orthogonal to (6,3)). The corner rule is load-bearing: (2,5)→(3,6) and (3,6)→(4,5) both pass the
    /// blocked (3,5), and an implementation that allows them returns 6·√2 = 8.485281374238571 instead,
    /// via (0,3),(1,4),(2,5),(3,6),(4,5),(5,4),(6,3) (measured by running this fixture with the corner
    /// test in `can_step` disabled; the spec's own "1 + 6√2 = 9.485" is not what a corner-cutting search
    /// returns, six diagonals suffice). Both the cost and the per-step corner check below catch it.
    #[test]
    fn spec_a2_8_wall_with_a_gap_eight_connected_respects_the_corner_rule() {
        let grid = spec_grid(&WALL_7X7);
        let (start, goal) = (spec_cell(&grid, (0, 3)), spec_cell(&grid, (6, 3)));
        let expected = 9.656_854_249_492_38_f64;
        assert!((expected - (4.0 + 4.0 * SQRT2)).abs() < 1e-12);
        // non-vacuous: the corner cell is blocked, the two cells the forbidden diagonals connect are free,
        // and the corner-cut answer is a different number than the expected one
        let corner = spec_cell(&grid, (3, 5));
        let (a, gap, b) = (spec_cell(&grid, (2, 5)), spec_cell(&grid, (3, 6)), spec_cell(&grid, (4, 5)));
        assert!(grid.blocked(corner.0 as i64, corner.1 as i64), "corner cell (3,5) must be blocked");
        for c in [a, gap, b] {
            assert!(!grid.blocked(c.0 as i64, c.1 as i64), "cell {c:?} must be free");
        }
        assert!((6.0 * SQRT2 - expected).abs() > 1.0, "corner-cut cost must be distinguishable");
        let path = spec_plan(&grid, Connectivity::Eight, (0, 3), (6, 3)).expect("A2-8 is reachable through the gap");
        assert_well_formed(&grid, &path, start, goal);
        assert!(path.contains(&gap), "the only way across is the gap");
        // the corner cell is not on the path, and neither forbidden diagonal is taken
        assert!(!path.contains(&corner), "the blocked corner (3,5) is on the path: {path:?}");
        for w in path.windows(2) {
            assert!(!((w[0] == a && w[1] == gap) || (w[0] == gap && w[1] == b)), "corner-cutting step {:?}->{:?} taken", w[0], w[1]);
            // and, generally, every diagonal on the path has both orthogonal neighbours free
            let (di, dj) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            if di != 0 && dj != 0 {
                assert!(!grid.blocked((w[0].0 + di) as i64, w[0].1 as i64) && !grid.blocked(w[0].0 as i64, (w[0].1 + dj) as i64), "diagonal {:?}->{:?} cuts a blocked corner", w[0], w[1]);
            }
        }
        assert!((path_length(&path) - expected).abs() < 1e-9, "A2-8 cost {} vs {expected}", path_length(&path));
    }

    /// A3: 5x5 with a full wall on row 2; rows 0-1 are sealed from rows 3-4 under either connectivity
    /// (spec: no diagonal can pass a full row). Expected `None` for both.
    #[test]
    fn spec_a3_full_wall_is_unreachable_under_both_connectivities() {
        let grid = spec_grid(&[".....", ".....", "#####", ".....", "....."]);
        let (start, goal) = (spec_cell(&grid, (0, 0)), spec_cell(&grid, (4, 4)));
        // non-vacuous: both endpoints are free, so a `None` is the wall's doing and not a sealed endpoint,
        // and the wall really spans the width
        assert!(!grid.blocked(start.0 as i64, start.1 as i64) && !grid.blocked(goal.0 as i64, goal.1 as i64));
        let wj = spec_cell(&grid, (0, 2)).1 as i64;
        assert!((0..5).all(|i| grid.blocked(i, wj)), "row 2 must be blocked across the full width");
        assert!(spec_plan(&grid, Connectivity::Eight, (0, 0), (4, 4)).is_none(), "A3 8-connected must be unreachable");
        assert!(spec_plan(&grid, Connectivity::Four, (0, 0), (4, 4)).is_none(), "A3 4-connected must be unreachable");
        // control: removing one wall cell makes the same query reachable, so the `None` is not a bug
        let opened = spec_grid(&[".....", ".....", "##.##", ".....", "....."]);
        assert!(spec_plan(&opened, Connectivity::Four, (0, 0), (4, 4)).is_some(), "a gap must restore reachability");
    }

    /// A4: start == goal on a free cell is a one-cell path of cost 0 (definition: zero moves); a start on
    /// `#` is `None`; a goal out of bounds is `None` without a panic.
    #[test]
    fn spec_a4_degenerate_queries() {
        let grid = spec_grid(&[".....", ".....", "..#..", ".....", "....."]);
        let path = spec_plan(&grid, Connectivity::Four, (0, 0), (0, 0)).expect("start == goal on a free cell is reachable");
        assert_eq!(path, vec![spec_cell(&grid, (0, 0))]);
        assert!((path_length(&path) - 0.0).abs() < 1e-9);
        // non-vacuous: the `#` really is blocked, and the out-of-bounds goal really is outside a 5x5
        let blocked = spec_cell(&grid, (2, 2));
        assert!(grid.blocked(blocked.0 as i64, blocked.1 as i64), "(2,2) must be blocked");
        assert!(spec_plan(&grid, Connectivity::Four, (2, 2), (0, 0)).is_none(), "a start on '#' must be None");
        assert!(spec_plan(&grid, Connectivity::Eight, (0, 0), (2, 2)).is_none(), "a goal on '#' must be None");
        assert!(grid.blocked(5, 0) && grid.blocked(0, -1), "out-of-bounds cells must read as blocked");
        assert!(spec_plan(&grid, Connectivity::Four, (0, 0), (5, 0)).is_none(), "a goal off the right edge must be None");
        assert!(spec_plan(&grid, Connectivity::Eight, (0, 0), (0, -1)).is_none(), "a goal below the drawing must be None");
        assert!(spec_plan(&grid, Connectivity::Eight, (0, 5), (0, 0)).is_none(), "a start above the drawing must be None");
    }
}
