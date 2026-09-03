//! **Distance-transform planner** — a wavefront swept outward from the goal, then steepest descent from
//! any start. One sweep serves every start for a fixed map and goal, so it is the multi-query
//! counterpart of [`crate::astar_grid_conn`], which searches once per start.
//!
//! The transform is a single-source shortest-path field rooted at the goal (Jarvis, *Collision-free
//! trajectory planning using distance transforms*, Mech. Eng. Trans. IEAust ME10(3):187–191, 1985; Jarvis,
//! *On distance transform based collision-free path planning*, in Recent Trends in Mobile Robots, World
//! Scientific 1994, pp. 3–31). Under [`Connectivity::Four`] every step costs 1 and a FIFO breadth-first
//! wavefront is exact, which is Jarvis's 1985 form. Under [`Connectivity::Eight`] a diagonal costs `√2`,
//! the first visit is no longer the shortest, and the sweep is a priority-queue (Dijkstra) sweep instead.
//! The integer chamfer transforms of Jarvis and of Borgefors (*Distance transformations in digital
//! images*, CVGIP 34(3):344–371, 1986, doi:10.1016/S0734-189X(86)80047-0) value a diagonal as 1 or `4/3`
//! and are **not** used: the field here is the exact grid metric, so it can double as an exact heuristic
//! for A\* and D\* Lite on the same map. Corke, *Robotics, Vision and Control* 2nd ed., Sec. 5.2.1, was
//! read for the description only.
//!
//! The neighbourhood, costs and the corner rule (a diagonal may not pass between two cells of which
//! either is blocked) are [`Connectivity::steps`] and [`can_step`], shared with every grid planner in the
//! crate; the sweep visits an edge `u→v` exactly when `can_step` allows the forward step `v→u`, which is
//! the same test because both orthogonal corner cells are common to the two directions.
//!
//! **Verified** (`tests` below): on the five specification fixtures DT1-4, DT1-8, DT2, DT3-4 and DT3-8 the
//! planned cost matches the hand-computed value (8, `4+2√2`, unreachable, 12, `4+4√2`) to `1e-9`, and the
//! oracle is [`crate::astar_grid_conn`] on the same fixture: the descent path's length equals the A\*
//! path's length, and on DT1-8 the field at **every** free cell equals the A\* cost from that cell. Three
//! further hand-computed fixtures pin what the specification's fixtures turned out not to (measured, see
//! their tests): a 3×5 grid where a FIFO wavefront under eight-connectivity reads `3+2√2` for a cell whose
//! cost is 5; a 3×6 grid where a heap sweep that keeps the first value pushed instead of relaxing on
//! `dv < d[v]` reads `2+3√2` for a cell whose cost is 6 (measured: that mutation passes all five
//! specification fixtures and the whole-field DT1-8 oracle, and fails only the 5×6 and 3×6 fixtures); and
//! a 5×6 grid where descending by the field value alone would take a `2+3√2` path for a cell whose cost is
//! 6. A blocked or out-of-bounds goal yields an all-`INFINITY` field; a blocked start
//! yields `None`; a field from a different map or connectivity is detected by the Bellman consistency
//! check and refused.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::grid_astar::{Connectivity, OrdF, can_step};

/// The cost-to-goal field over a `width × height` grid, row-major (`index = j * width + i`, the
/// [`crate::OccupancyGrid`] convention). `INFINITY` on blocked cells and on free cells from which the
/// goal cannot be reached; `0` at the goal. If `goal` is out of bounds or blocked the whole field is
/// `INFINITY`.
///
/// `is_free(i, j)` need not carry the bounds; they are applied here. Under [`Connectivity::Four`] this is
/// a FIFO wavefront (exact for unit costs); under [`Connectivity::Eight`] it is a heap sweep (Dijkstra),
/// because a FIFO wavefront under `√2` diagonals is wrong by up to `2-√2` per diagonal. The 3×5 fixture in
/// `a_fifo_wavefront_would_be_wrong_here_and_the_heap_is_not` pins that (DT1-8 does not: measured, see the
/// module doc). The heap branch relaxes on `dv < d[v]`, not on first push: the 3×6 fixture in
/// `a_first_push_wins_heap_would_be_wrong_here` pins that (the five specification fixtures do not:
/// measured, see the module doc).
pub fn distance_transform(width: usize, height: usize, conn: Connectivity, is_free: impl Fn(i32, i32) -> bool, goal: (i32, i32)) -> Vec<f64> {
    let n = width * height;
    let mut d = vec![f64::INFINITY; n];
    let in_bounds = |i: i32, j: i32| i >= 0 && j >= 0 && (i as usize) < width && (j as usize) < height;
    let free = |i: i32, j: i32| in_bounds(i, j) && is_free(i, j);
    if !free(goal.0, goal.1) {
        return d;
    }
    let idx = |i: i32, j: i32| (j as usize) * width + (i as usize);
    d[idx(goal.0, goal.1)] = 0.0;

    match conn {
        Connectivity::Four => {
            // unit costs: the first visit of a cell is its shortest, so breadth-first order is exact
            let mut queue = VecDeque::from([goal]);
            while let Some((ci, cj)) = queue.pop_front() {
                let du = d[idx(ci, cj)];
                for &(di, dj, cost) in conn.steps() {
                    if !can_step(&free, ci, cj, di, dj) {
                        continue;
                    }
                    let v = idx(ci + di, cj + dj);
                    if d[v].is_infinite() {
                        d[v] = du + cost;
                        queue.push_back((ci + di, cj + dj));
                    }
                }
            }
        }
        Connectivity::Eight => {
            let mut heap: BinaryHeap<Reverse<(OrdF, i32, i32)>> = BinaryHeap::new();
            heap.push(Reverse((OrdF(0.0), goal.0, goal.1)));
            while let Some(Reverse((OrdF(du), ci, cj))) = heap.pop() {
                if du > d[idx(ci, cj)] {
                    continue; // stale entry: this cell was relaxed again after being pushed
                }
                for &(di, dj, cost) in conn.steps() {
                    if !can_step(&free, ci, cj, di, dj) {
                        continue;
                    }
                    let v = idx(ci + di, cj + dj);
                    let dv = du + cost;
                    if dv < d[v] {
                        d[v] = dv;
                        heap.push(Reverse((OrdF(dv), ci + di, cj + dj)));
                    }
                }
            }
        }
    }
    d
}

/// Steepest descent of a field from [`distance_transform`]: from `start`, repeatedly step to the
/// neighbour `v` minimising `dt[v] + c(s, v)` (ties broken by [`Connectivity::steps`] order) until the
/// goal (`dt == 0`) is reached. Returns the cell path, `start` and goal inclusive, or `None` when `start`
/// is out of bounds, blocked or has no path (`dt[start]` infinite).
///
/// The choice is by `dt[v] + c(s, v)`, not by `dt[v]` alone: under [`Connectivity::Eight`] the latter
/// prefers a diagonal whose field value is only marginally lower (by less than `√2 − 1` short of a full
/// unit) and the path it produces is longer than `dt[start]`. On the five specification fixtures the two
/// rules happen to agree (measured: identical descents); the 5×6 fixture in
/// `descending_by_the_field_alone_would_be_longer` is one where they do not, by `(2+3√2) − 6 ≈ 0.243`.
///
/// Every step must satisfy the Bellman relation `dt[s] = min_v (dt[v] + c(s, v))` to `1e-9`, which holds
/// by construction for a field computed on this map with this connectivity. A field that fails it (one
/// cached across a map edit, or computed with the other connectivity) is refused with `None` rather than
/// followed into an obstacle. The descent terminates because every step lowers `dt` by at least 1.
pub fn descend(dt: &[f64], width: usize, height: usize, conn: Connectivity, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32)) -> Option<Vec<(i32, i32)>> {
    let n = width * height;
    if dt.len() != n {
        return None;
    }
    let in_bounds = |i: i32, j: i32| i >= 0 && j >= 0 && (i as usize) < width && (j as usize) < height;
    let free = |i: i32, j: i32| in_bounds(i, j) && is_free(i, j);
    let idx = |i: i32, j: i32| (j as usize) * width + (i as usize);
    if !free(start.0, start.1) || !dt[idx(start.0, start.1)].is_finite() {
        return None;
    }
    let mut path = vec![start];
    let (mut ci, mut cj) = start;
    let mut steps = 0usize;
    while dt[idx(ci, cj)] > 0.0 {
        let mut best: Option<((i32, i32), f64)> = None;
        for &(di, dj, cost) in conn.steps() {
            if !can_step(&free, ci, cj, di, dj) {
                continue;
            }
            let (ni, nj) = (ci + di, cj + dj);
            let value = dt[idx(ni, nj)] + cost;
            if value.is_finite() && best.is_none_or(|(_, b)| value < b) {
                best = Some(((ni, nj), value));
            }
        }
        let ((ni, nj), value) = best?;
        if (value - dt[idx(ci, cj)]).abs() > 1e-9 {
            return None; // the field is not this map's field
        }
        (ci, cj) = (ni, nj);
        path.push((ci, cj));
        steps += 1;
        if steps > n {
            return None; // unreachable if the consistency check held; defensive against a corrupt field
        }
    }
    Some(path)
}

/// [`distance_transform`] then [`descend`]: the cost `dt[start]` and the cell path from `start` to
/// `goal`, or `None` if either cell is blocked, out of bounds, or no path exists. Single-query
/// convenience; a caller with many starts should keep the field and call [`descend`] per start.
pub fn plan(width: usize, height: usize, conn: Connectivity, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32)) -> Option<(f64, Vec<(i32, i32)>)> {
    let dt = distance_transform(width, height, conn, &is_free, goal);
    // `descend` has already refused an out-of-bounds or blocked start, so the index is valid here
    let path = descend(&dt, width, height, conn, &is_free, start)?;
    Some((dt[(start.1 as usize) * width + (start.0 as usize)], path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_astar::path_length;
    use crate::{OccupancyGrid, astar_grid_conn};

    const SQRT2: f64 = std::f64::consts::SQRT_2;

    /// A fixture: the drawing, its start and goal in the specification's `(column, row-from-top)`
    /// coordinates, the connectivity, and the hand-computed expected cost (`None` = unreachable).
    struct Fixture {
        name: &'static str,
        rows: &'static [&'static str],
        start: (i32, i32),
        goal: (i32, i32),
        conn: Connectivity,
        expected: Option<f64>,
    }

    /// The five fixtures of the "Distance-transform planner" specification entry. Expected costs are the
    /// specification's hand computations: DT1-4 Manhattan `4+4`; DT1-8 two diagonals plus four
    /// orthogonals because `(2,2)` blocks the main diagonal and its corners; DT2 a sealed ring; DT3-4
    /// `3 down + 6 across + 3 up` through the gap at the bottom row; DT3-8 four diagonals plus four
    /// orthogonals, the diagonal `(2,5)→(3,6)` refused because it would cut the wall cell `(3,5)`.
    const FIXTURES: [Fixture; 5] = [
        Fixture { name: "DT1-4", rows: &[".....", ".....", "..#..", ".....", "....."], start: (0, 0), goal: (4, 4), conn: Connectivity::Four, expected: Some(8.0) },
        Fixture { name: "DT1-8", rows: &[".....", ".....", "..#..", ".....", "....."], start: (0, 0), goal: (4, 4), conn: Connectivity::Eight, expected: Some(4.0 + 2.0 * SQRT2) },
        Fixture { name: "DT2", rows: &[".......", ".......", "..###..", "..#.#..", "..###..", ".......", "......."], start: (0, 0), goal: (3, 3), conn: Connectivity::Eight, expected: None },
        Fixture { name: "DT3-4", rows: &["...#...", "...#...", "...#...", "...#...", "...#...", "...#...", "......."], start: (0, 3), goal: (6, 3), conn: Connectivity::Four, expected: Some(12.0) },
        Fixture { name: "DT3-8", rows: &["...#...", "...#...", "...#...", "...#...", "...#...", "...#...", "......."], start: (0, 3), goal: (6, 3), conn: Connectivity::Eight, expected: Some(4.0 + 4.0 * SQRT2) },
    ];

    /// The grid, and the fixture's `(column, row-from-top)` cells mapped to the grid's `(i, j)`:
    /// `from_rows` puts the top drawn row at the highest `j`, so `j = height - 1 - row`.
    fn build(f: &Fixture) -> (OccupancyGrid, (i32, i32), (i32, i32)) {
        let g = OccupancyGrid::from_rows(f.rows, 1.0, 0.0, 0.0).expect("rectangular drawing");
        let h = g.height as i32;
        (g, (f.start.0, h - 1 - f.start.1), (f.goal.0, h - 1 - f.goal.1))
    }

    fn free_of(g: &OccupancyGrid) -> impl Fn(i32, i32) -> bool + '_ {
        move |i, j| !g.blocked(i as i64, j as i64)
    }

    /// The specification's expected costs match the spec's fixtures' constants (a typo in the constant
    /// table above would otherwise be invisible).
    #[test]
    fn the_constant_table_matches_the_specification_decimals() {
        assert!((FIXTURES[1].expected.unwrap() - 6.82842712474619).abs() < 1e-12);
        assert!((FIXTURES[4].expected.unwrap() - 9.65685424949238).abs() < 1e-12);
    }

    /// Every fixture: the planned cost equals the hand-computed value to `1e-9`; the path starts at
    /// `start`, ends at `goal`, uses only legal steps (`can_step`, so no obstacle and no corner cut), and
    /// its length equals the reported cost. Non-vacuity: the field varies (0 at the goal, larger
    /// elsewhere, infinite on the drawn obstacles).
    #[test]
    fn every_fixture_matches_its_hand_computed_cost() {
        for f in &FIXTURES {
            let (g, start, goal) = build(f);
            let free = free_of(&g);
            let dt = distance_transform(g.width, g.height, f.conn, &free, goal);
            let gi = (goal.1 as usize) * g.width + goal.0 as usize;
            assert_eq!(dt[gi], 0.0, "{}: goal has zero cost", f.name);
            let finite_max = dt.iter().copied().filter(|v| v.is_finite()).fold(0.0, f64::max);
            // DT2's field is finite only at the goal; its non-vacuity is pinned by
            // `a_sealed_goal_leaves_the_outside_infinite`
            assert!(finite_max > 0.0 || f.expected.is_none(), "{}: the field must vary", f.name);
            let blocked_cells = (0..g.height).flat_map(|j| (0..g.width).map(move |i| (i, j))).filter(|&(i, j)| g.blocked(i as i64, j as i64)).count();
            assert!(blocked_cells > 0, "{}: the drawing has an obstacle", f.name);
            assert!(dt.iter().enumerate().all(|(k, v)| !g.blocked((k % g.width) as i64, (k / g.width) as i64) || v.is_infinite()), "{}: blocked cells are INF", f.name);

            let result = plan(g.width, g.height, f.conn, &free, start, goal);
            match f.expected {
                None => assert!(result.is_none(), "{}: expected unreachable, got {:?}", f.name, result),
                Some(expected) => {
                    let (cost, path) = result.unwrap_or_else(|| panic!("{}: expected reachable", f.name));
                    assert!((cost - expected).abs() < 1e-9, "{}: cost {cost} vs expected {expected}", f.name);
                    assert_eq!(path.first(), Some(&start), "{}: path starts at start", f.name);
                    assert_eq!(path.last(), Some(&goal), "{}: path ends at goal", f.name);
                    for w in path.windows(2) {
                        assert!(can_step(&free, w[0].0, w[0].1, w[1].0 - w[0].0, w[1].1 - w[0].1), "{}: illegal step {:?}->{:?}", f.name, w[0], w[1]);
                        if f.conn == Connectivity::Four {
                            assert_eq!((w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs(), 1, "{}: 4-connected steps are orthogonal", f.name);
                        }
                    }
                    assert!((path_length(&path) - cost).abs() < 1e-9, "{}: path length {} vs field {cost}", f.name, path_length(&path));
                }
            }
        }
    }

    /// ORACLE: on every fixture the cost equals the length of the `astar_grid_conn` path with the same
    /// connectivity, and reachability agrees. The two planners share the neighbourhood but not the
    /// search, so this is an independent computation of the same quantity.
    #[test]
    fn cost_equals_astar_grid_conn_on_every_fixture() {
        for f in &FIXTURES {
            let (g, start, goal) = build(f);
            let free = free_of(&g);
            let ours = plan(g.width, g.height, f.conn, &free, start, goal);
            let theirs = astar_grid_conn(g.width, g.height, f.conn, &free, start, goal);
            match (ours, theirs) {
                (None, None) => assert!(f.expected.is_none(), "{}: both unreachable but the fixture expects a path", f.name),
                (Some((cost, _)), Some(p)) => assert!((cost - path_length(&p)).abs() < 1e-9, "{}: field {cost} vs A* {}", f.name, path_length(&p)),
                (a, b) => panic!("{}: reachability disagrees: dt {:?} vs A* {:?}", f.name, a.map(|x| x.0), b.map(|p| path_length(&p))),
            }
        }
    }

    /// ORACLE over the whole field: on DT1-8 the value at every free cell equals the A\* cost from that
    /// cell to the goal, which is what makes the field an exact heuristic. Non-vacuity: at least 20 cells
    /// are compared and their values are not all equal.
    #[test]
    fn the_field_is_the_astar_cost_at_every_free_cell() {
        let f = &FIXTURES[1];
        let (g, _, goal) = build(f);
        let free = free_of(&g);
        let dt = distance_transform(g.width, g.height, f.conn, &free, goal);
        let mut compared = 0;
        let mut distinct = std::collections::BTreeSet::new();
        for j in 0..g.height as i32 {
            for i in 0..g.width as i32 {
                if !free(i, j) {
                    continue;
                }
                let p = astar_grid_conn(g.width, g.height, f.conn, &free, (i, j), goal).expect("open grid cell reaches the goal");
                let v = dt[(j as usize) * g.width + i as usize];
                assert!((v - path_length(&p)).abs() < 1e-9, "cell ({i},{j}): field {v} vs A* {}", path_length(&p));
                compared += 1;
                distinct.insert((v * 1e6).round() as i64);
            }
        }
        assert_eq!(compared, 24, "5x5 minus one obstacle");
        assert!(distinct.len() > 5, "the field takes many values: {distinct:?}");
    }

    /// Four- and eight-connectivity must give different costs on DT1 and DT3 (8 vs `4+2√2`, 12 vs
    /// `4+4√2`), or the connectivity argument is decorative.
    #[test]
    fn four_and_eight_connectivity_differ() {
        for (a, b) in [(0, 1), (3, 4)] {
            let (ga, sa, goa) = build(&FIXTURES[a]);
            let (gb, sb, gob) = build(&FIXTURES[b]);
            let ca = plan(ga.width, ga.height, Connectivity::Four, free_of(&ga), sa, goa).unwrap().0;
            let cb = plan(gb.width, gb.height, Connectivity::Eight, free_of(&gb), sb, gob).unwrap().0;
            assert!(ca - cb > 1.0, "{} vs {}: {ca} vs {cb}", FIXTURES[a].name, FIXTURES[b].name);
        }
    }

    /// DT2: the sealed goal. Every cell outside the ring is INF under both connectivities, the sweep
    /// terminates, and the goal itself still reads 0 (the field is valid, just unreachable from outside).
    #[test]
    fn a_sealed_goal_leaves_the_outside_infinite() {
        let f = &FIXTURES[2];
        let (g, start, goal) = build(f);
        let free = free_of(&g);
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let dt = distance_transform(g.width, g.height, conn, &free, goal);
            let finite: Vec<usize> = dt.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(k, _)| k).collect();
            assert_eq!(finite, vec![(goal.1 as usize) * g.width + goal.0 as usize], "{conn:?}: only the goal is finite");
            assert!(descend(&dt, g.width, g.height, conn, &free, start).is_none(), "{conn:?}: no descent from outside");
        }
        // the ring is really there: its 8 cells are blocked and the interior is free
        let ring = (2..=4).flat_map(|i| (2..=4).map(move |j| (i, j))).filter(|&c| c != (3, 3)).count();
        assert_eq!(ring, 8);
        assert!((2..=4).flat_map(|i| (2..=4).map(move |j| (i, j))).filter(|&c| c != (3, 3)).all(|(i, j)| g.blocked(i, j)));
        assert!(!g.blocked(3, 3));
    }

    /// The corner rule, hand computation: a 3x3 with `(1,0)` and `(0,1)` blocked seals `(0,0)` under
    /// eight-connectivity because the only exit is the diagonal that passes between the two blocked
    /// cells. With the rule removed the field at `(0,0)` would be `2√2` and the path would cut the corner.
    /// DT2 cannot detect this: its ring is eight-connected solid, so there is no corner to cut.
    #[test]
    fn the_corner_rule_seals_a_diagonal_gap() {
        let free = |i: i32, j: i32| (0..3).contains(&i) && (0..3).contains(&j) && !((i, j) == (1, 0) || (i, j) == (0, 1));
        let dt = distance_transform(3, 3, Connectivity::Eight, free, (2, 2));
        assert!(dt[0].is_infinite(), "(0,0) is sealed by the corner rule: {}", dt[0]);
        assert!((dt[4] - SQRT2).abs() < 1e-12, "(1,1) is one diagonal from the goal");
        assert!(plan(3, 3, Connectivity::Eight, free, (0, 0), (2, 2)).is_none());
        // and the gap is real: without the two blocked cells the corner cell reaches the goal at 2√2
        let open = |i: i32, j: i32| (0..3).contains(&i) && (0..3).contains(&j);
        assert!((plan(3, 3, Connectivity::Eight, open, (0, 0), (2, 2)).unwrap().0 - 2.0 * SQRT2).abs() < 1e-12);
    }

    /// A FIFO wavefront is not exact under eight-connectivity, hand computation. The specification names
    /// DT1-8 for this, but on DT1-8 (and DT3-8) every minimum-hop path has exactly as many diagonals as the
    /// minimum-cost path, so a breadth-first sweep in the crate's step order lands on the right value there
    /// (measured: field discrepancy 0 on both). This 3-wide, 5-high grid with `(1,3)` blocked and the goal
    /// at `(1,4)` is the smallest grid the search in the scratchpad script found where it does not: from
    /// the goal, `(2,4)` is queued before `(0,4)`, so `(2,2)` (hop 3, cost 3) expands before `(0,2)` and
    /// first-visits `(1,1)` at `3+√2`; `(1,1)` then first-visits `(0,0)` at `3+2√2 ≈ 5.828`, but the
    /// orthogonal column `(0,0)→(0,4)→(1,4)` costs 5. Non-vacuity: the two candidate paths are asserted
    /// to differ by more than `0.8`.
    #[test]
    fn a_fifo_wavefront_would_be_wrong_here_and_the_heap_is_not() {
        let g = OccupancyGrid::from_rows(&["...", ".#.", "...", "...", "..."], 1.0, 0.0, 0.0).unwrap();
        assert!(g.blocked(1, 3) && !g.blocked(1, 4), "the drawing's hash is at (1,3), just below the goal");
        let free = free_of(&g);
        let dt = distance_transform(3, 5, Connectivity::Eight, &free, (1, 4));
        let fifo_value = 3.0 + 2.0 * SQRT2;
        assert!(fifo_value - 5.0 > 0.8, "the two candidate paths differ");
        assert!((dt[0] - 5.0).abs() < 1e-9, "(0,0) should read 5, the orthogonal column; got {}", dt[0]);
        assert!((dt[gidx(&g, (1, 1))] - (3.0 + SQRT2)).abs() < 1e-9, "(1,1) reads 3+√2 under both sweeps");
        let (cost, path) = plan(3, 5, Connectivity::Eight, &free, (0, 0), (1, 4)).unwrap();
        assert!((cost - 5.0).abs() < 1e-9 && (path_length(&path) - 5.0).abs() < 1e-9);
        let oracle = astar_grid_conn(3, 5, Connectivity::Eight, &free, (0, 0), (1, 4)).unwrap();
        assert!((path_length(&oracle) - 5.0).abs() < 1e-9, "A* agrees: {}", path_length(&oracle));
    }

    /// A heap sweep must relax on `dv < d[v]`, not keep the first value pushed, hand computation. With a
    /// heap the first push of a cell comes from its neighbour of smallest field value, and when that
    /// neighbour is diagonal a later-popped orthogonal neighbour can still be cheaper: `d[u] = 2√2` from
    /// `u` diagonally gives `3√2 ≈ 4.243`, `d[u'] = 3` from `u'` orthogonally gives 4. The five specification
    /// fixtures do not exercise this (measured: the first-push rule passes all of them and the whole-field
    /// DT1-8 oracle); the search in the scratchpad script found this 3-wide, 6-high grid as the smallest
    /// with two obstacles where it fails. Goal `(0,5)`, obstacles `(0,0)` and `(1,2)`. The column
    /// `(0,5)→(0,1)` costs 4, `(1,1)` reads 5 from `(0,1)` (its diagonal from `(0,2)` is refused by the
    /// corner `(1,2)`), and `(1,0)` reads 6 from `(1,1)` (its diagonal from `(0,1)` is refused by the corner
    /// `(0,0)`). The other route `(0,5)→(1,4)→(2,3)→(2,2)→(2,1)` costs `2+2√2 ≈ 4.828`, so `(2,1)` pops
    /// before `(1,1)` and pushes `(1,0)` first at `2+3√2 ≈ 6.243`; only the later relaxation from `(1,1)`
    /// brings it to 6. Non-vacuity: `dt[(2,1)] < dt[(1,1)]` and the `0.243` excess are asserted before the
    /// field value is checked.
    #[test]
    fn a_first_push_wins_heap_would_be_wrong_here() {
        let g = OccupancyGrid::from_rows(&["...", "...", "...", ".#.", "...", "#.."], 1.0, 0.0, 0.0).unwrap();
        assert!(g.blocked(0, 0) && g.blocked(1, 2) && !g.blocked(0, 5), "the drawing's hashes are at (0,0) and (1,2)");
        let free = free_of(&g);
        let (start, goal) = ((1, 0), (0, 5));
        let dt = distance_transform(3, 6, Connectivity::Eight, &free, goal);
        let (d_orth, d_diag) = (dt[gidx(&g, (1, 1))], dt[gidx(&g, (2, 1))]);
        assert!((d_orth - 5.0).abs() < 1e-9 && (d_diag - (2.0 + 2.0 * SQRT2)).abs() < 1e-9, "field {d_orth} {d_diag}");
        assert!(d_diag < d_orth, "the diagonal neighbour pops first");
        let first_push = d_diag + SQRT2;
        assert!(first_push - 6.0 > 0.2, "and its push is not the cheapest: excess {}", first_push - 6.0);
        assert!((dt[gidx(&g, start)] - 6.0).abs() < 1e-9, "(1,0) should read 6, the relaxed value; got {}", dt[gidx(&g, start)]);
        let (cost, path) = plan(3, 6, Connectivity::Eight, &free, start, goal).unwrap();
        assert!((cost - 6.0).abs() < 1e-9 && (path_length(&path) - 6.0).abs() < 1e-9, "cost {cost} path {}", path_length(&path));
        assert_eq!(path[1], (1, 1), "the first step is the orthogonal one: {path:?}");
        let oracle = astar_grid_conn(3, 6, Connectivity::Eight, &free, start, goal).unwrap();
        assert!((path_length(&oracle) - 6.0).abs() < 1e-9, "A* agrees: {}", path_length(&oracle));
    }

    /// Descending by `dt[v]` alone is wrong, hand computation. On the five specification fixtures the
    /// `dt[v]` and `dt[v] + c` rules choose the same neighbour at every step (measured with the scratchpad
    /// script, which also found no 4×4 grid with up to four obstacles where they differ), so this 5×6 grid
    /// pins the rule. Obstacles `(2,5)` and `(3,3)`, goal `(2,0)`, start `(3,5)`. The diagonal
    /// `(3,5)→(2,4)` is refused by the corner `(2,5)` and `(3,4)→(2,3)` by the corner `(3,3)`, so the
    /// optimum is the six orthogonal steps `(3,5)→(3,4)→(2,4)→(2,3)→(2,2)→(2,1)→(2,0)`, cost 6, with
    /// `dt[(3,4)] = 5`. The diagonal neighbour `(4,4)` reads `2+2√2 ≈ 4.828` (`(4,4)→(4,3)→(4,2)` then two
    /// diagonals to `(2,0)`), which is *lower* than 5, so the field-only rule would step there at cost `√2`
    /// and finish at `2+3√2 ≈ 6.243`. Non-vacuity: both `dt[(4,4)] < dt[(3,4)]` and the `0.243` excess
    /// are asserted before the path is checked.
    #[test]
    fn descending_by_the_field_alone_would_be_longer() {
        let g = OccupancyGrid::from_rows(&["..#..", ".....", "...#.", ".....", ".....", "....."], 1.0, 0.0, 0.0).unwrap();
        assert!(g.blocked(2, 5) && g.blocked(3, 3), "the two corners are where the drawing puts them");
        let free = free_of(&g);
        let (start, goal) = ((3, 5), (2, 0));
        let dt = distance_transform(5, 6, Connectivity::Eight, &free, goal);
        let (d_start, d_orth, d_diag) = (dt[gidx(&g, start)], dt[gidx(&g, (3, 4))], dt[gidx(&g, (4, 4))]);
        assert!((d_start - 6.0).abs() < 1e-9 && (d_orth - 5.0).abs() < 1e-9 && (d_diag - (2.0 + 2.0 * SQRT2)).abs() < 1e-9, "field {d_start} {d_orth} {d_diag}");
        assert!(d_diag < d_orth, "the field-only rule would prefer the diagonal");
        assert!(d_diag + SQRT2 - d_start > 0.2, "and that step is not on a shortest path: excess {}", d_diag + SQRT2 - d_start);
        let path = descend(&dt, 5, 6, Connectivity::Eight, &free, start).unwrap();
        assert_eq!(path[1], (3, 4), "the first step is the orthogonal one: {path:?}");
        assert!((path_length(&path) - 6.0).abs() < 1e-9, "path cost {}", path_length(&path));
        let oracle = astar_grid_conn(5, 6, Connectivity::Eight, &free, start, goal).unwrap();
        assert!((path_length(&oracle) - 6.0).abs() < 1e-9, "A* agrees: {}", path_length(&oracle));
    }

    /// Degenerate inputs: a goal on a blocked cell or off the map gives an all-INF field (no panic); a
    /// blocked start is `None` even though the field is valid; start equal to goal is the single-cell
    /// path at cost 0; a field of the wrong length is refused.
    #[test]
    fn degenerate_goals_and_starts() {
        let f = &FIXTURES[0];
        let (g, start, goal) = build(f);
        let free = free_of(&g);
        assert!(distance_transform(g.width, g.height, Connectivity::Four, &free, (2, 2)).iter().all(|v| v.is_infinite()), "blocked goal");
        assert!(distance_transform(g.width, g.height, Connectivity::Eight, &free, (5, 0)).iter().all(|v| v.is_infinite()), "goal off the map");
        assert!(distance_transform(g.width, g.height, Connectivity::Eight, &free, (-1, 0)).iter().all(|v| v.is_infinite()), "negative goal");
        let dt = distance_transform(g.width, g.height, Connectivity::Eight, &free, goal);
        assert!(dt[gidx(&g, goal)] == 0.0 && dt[gidx(&g, start)].is_finite(), "the field is valid");
        assert!(descend(&dt, g.width, g.height, Connectivity::Eight, &free, (2, 2)).is_none(), "blocked start");
        assert!(descend(&dt, g.width, g.height, Connectivity::Eight, &free, (0, 5)).is_none(), "start off the map");
        assert_eq!(plan(g.width, g.height, Connectivity::Eight, &free, goal, goal), Some((0.0, vec![goal])));
        assert!(descend(&dt[..3], g.width, g.height, Connectivity::Eight, &free, start).is_none(), "wrong-size field");
    }

    /// A field computed for the other connectivity fails the Bellman check and is refused rather than
    /// followed: the DT3-4 (four-connected) field descended with eight-connected steps is inconsistent at
    /// the first cell where a diagonal beats two orthogonals. Non-vacuity: the two fields differ.
    #[test]
    fn a_field_from_the_other_connectivity_is_refused() {
        let f = &FIXTURES[3];
        let (g, start, goal) = build(f);
        let free = free_of(&g);
        let dt4 = distance_transform(g.width, g.height, Connectivity::Four, &free, goal);
        let dt8 = distance_transform(g.width, g.height, Connectivity::Eight, &free, goal);
        assert!(dt4.iter().zip(&dt8).any(|(a, b)| (a - b).abs() > 0.1), "the fields differ");
        assert!(descend(&dt4, g.width, g.height, Connectivity::Eight, &free, start).is_none(), "an inconsistent field is refused");
        assert!(descend(&dt4, g.width, g.height, Connectivity::Four, &free, start).is_some(), "the matching connectivity descends");
    }

    fn gidx(g: &OccupancyGrid, c: (i32, i32)) -> usize {
        (c.1 as usize) * g.width + c.0 as usize
    }
}
