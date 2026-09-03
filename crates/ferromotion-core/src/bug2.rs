//! **Bug2 on a grid** — the sensor-based reactive planner of Lumelsky & Stepanov, discretised to a
//! four-connected occupancy grid. Where [`crate::astar_grid_conn`] reads the whole map and returns the
//! shortest cell path, Bug2 reads only the cells adjacent to the robot as it moves, so it runs on a map that
//! is unknown a priori (an [`crate::OccupancyGrid`] still being built). It returns the sequence of cells the
//! robot actually traversed, which is complete but not shortest.
//!
//! The algorithm (Lumelsky & Stepanov, *Path-planning strategies for a point mobile automaton moving amidst
//! unknown obstacles of arbitrary shape*, Algorithmica 2:403–430, 1987, doi:10.1007/BF01840369; the earlier
//! form is IEEE Trans. Automatic Control 31(11):1058–1063, 1986, doi:10.1109/TAC.1986.1104175):
//!
//! 1. **Motion-to-goal** along the *m-line*, the straight segment start→goal. On the grid the m-line is the
//!    ordered cell list `M[0]=start … M[n]=goal` rasterised with the **same connectivity as the motion**:
//!    Bresenham's line (J. E. Bresenham, IBM Systems Journal 4(1):25–30, 1965) with the orthogonal
//!    intermediate cell inserted at every diagonal step, so consecutive m-line cells share an edge
//!    ([`m_line`]). "Closer to the goal along the m-line" is "larger index".
//! 2. **Hit point** `H`: the m-line cell at which the next m-line cell is blocked. The robot turns away from
//!    the wall ([`Turn::Left`] keeps the obstacle on its right, [`Turn::Right`] on its left) and
//! 3. **follows the boundary** — wall following through free cells, trying at each step the direction toward
//!    the wall side first, then straight, then away, then back. The map edge is an obstacle.
//! 4. **Leave condition**, checked after every following step: the robot stands **on an m-line cell** whose
//!    index is **strictly greater than** `H`'s (strictly closer to the goal along the m-line, Lumelsky's
//!    `d(L, T) < d(H, T)`). Motion-to-goal resumes from there; if the next m-line cell is blocked it hits
//!    again at once, at a strictly larger m-line index than the last hit point, so the number of hits is
//!    bounded by `|M|`. Lumelsky's second clause — the line `(L, T)` does not cross the *current* obstacle
//!    at `L` — is what stops the robot leaving at `H` itself, and the index inequality already does that.
//!    The specification's pseudocode approximates the clause as "the next m-line cell is free", which a
//!    local sensor cannot tell apart from "blocked by a *different* obstacle": on a 5×5 grid with the two
//!    single cells `(1,2)` and `(3,2)` blocked, start `(0,2)`, goal `(4,2)`, that approximation circles
//!    `(1,2)` without leaving at `(2,2)` and returns `None` for a goal A\* reaches in 6 moves (measured; the
//!    fixture is a test below). Leaving and re-hitting reaches it in 8.
//! 5. **Unreachable test**: returning to `H` **with the same heading** as the one it set off with, without
//!    having left, is a full circuit of the obstacle (the follower is a deterministic map on
//!    `(cell, heading)`, so recurrence of that state is a cycle). Comparing the cell alone would declare
//!    "unreachable" early on an obstacle whose boundary passes `H` twice (fixture B5 below exhibits this:
//!    cell-only detection returns `None` on a reachable goal).
//!
//! **Completeness, not optimality.** Lumelsky & Stepanov's Bug2 theorem: with finitely many obstacles of
//! finite perimeter the procedure terminates; a reachable target is reached and the return to a hit point
//! without leaving proves the target unreachable. Their length bound is `P ≤ D + Σᵢ nᵢ pᵢ / 2` with `D` the
//! start–goal distance, `pᵢ` obstacle `i`'s perimeter and `nᵢ` the number of times the m-line crosses it.
//! The grid preconditions for the proof to transfer are that motion is 4-connected while obstacle boundaries
//! are 8-connected (so the follower cannot leak through a diagonal gap), that the m-line is rasterised with
//! the motion connectivity, and that the map edge is an obstacle (its perimeter `2(W+H)` then enters the
//! bound — fixture B4 turned left circumnavigates the map edge). `max_steps` is a hard bound on executed
//! moves for the case the preconditions are violated; the specification's practical cap is
//! `4·W·H + |M|` ([`default_step_cap`]), and every fixture here terminates well under it.
//!
//! **Verified**, against hand traces of the four specification fixtures (each reproduced by an independent
//! simulator written from the same pseudocode) and against [`crate::astar_grid_conn`] with
//! [`crate::Connectivity::Four`] as the optimal-cost oracle:
//! - B1 (9×9, 3×6 block asymmetric about the m-line): turned left 16 steps, turned right 14, A\* optimum
//!   14. Bug2's executed cost is 16/14 = **1.143×** the optimum on this fixture — the measured non-optimality.
//! - B2 (goal sealed in a ring): `None`, after exactly 17 moves; A\* also finds no path. (The specification's
//!   trace calls `(5,3)` "m-line index 5"; the m-line is the segment start→goal and ends at the goal `(3,3)`,
//!   so `(5,3)` carries no index — the executed path is the same either way.)
//! - B3 (empty 5×5, horizontal m-line): 4 = the Manhattan optimum, no hit.
//! - B4 (wall with a gap at the map edge): turned right 12 = optimum; turned left 22 (the follower rounds the
//!   map edge and reaches the goal cell from below on the east edge). The specification text quotes 32 for
//!   this trace, which is the count if the robot walks *through* the goal without stopping; Lumelsky's
//!   procedure stops on reaching the target, and the pseudocode's leave test `j + 1 == |M|` does too.
//! - Two single-cell obstacles one free cell apart on the m-line (above): 8 moves against an optimum of 6.
//! - A diagonal m-line on an empty 5×5 costs 8 = Manhattan, with every step orthogonal (the 4-connected
//!   rasterisation); the same m-line with one cell blocked on it costs 10 against an optimum of 8.
//! - Every `is_free` query the planner makes lies within one cell of a traversed cell (the sensor-based
//!   property).
//!
//! Pure Rust, no allocation beyond the path and the m-line index → WASM-clean.

use std::collections::HashMap;

use crate::can_step;

/// Which way the robot turns at a hit point. [`Turn::Left`] keeps the obstacle on the robot's **right**
/// hand and is the convention the specification's fixtures B1–B3 use; [`Turn::Right`] mirrors it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    Left,
    Right,
}

/// Screen-frame unit steps, `E, S, W, N`: `y` grows downward, so `+1` on the index is a clockwise turn
/// (turn right) and `−1` is counter-clockwise (turn left).
const DIRS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

fn cw(h: usize) -> usize {
    (h + 1) % 4
}
fn ccw(h: usize) -> usize {
    (h + 3) % 4
}
fn back(h: usize) -> usize {
    (h + 2) % 4
}

/// The step cap the specification recommends, `4·W·H + |M|`, with `|M| ≤ W + H` for any m-line on the grid.
/// Exceeding it is a bug in the map's preconditions, not a planner outcome.
pub fn default_step_cap(width: usize, height: usize) -> usize {
    4 * width * height + width + height
}

/// The 4-connected rasterisation of the segment `start → goal`: Bresenham's line with the orthogonal
/// intermediate cell inserted at each diagonal step, major axis first, so consecutive cells share an edge and
/// the list has exactly `|Δx| + |Δy| + 1` cells. The cells are the m-line `M[0]=start … M[n]=goal`.
pub fn m_line(start: (i32, i32), goal: (i32, i32)) -> Vec<(i32, i32)> {
    let (mut x, mut y) = start;
    let (x1, y1) = goal;
    let dx = (x1 - x).abs();
    let dy = -(y1 - y).abs();
    let sx = if x1 > x { 1 } else { -1 };
    let sy = if y1 > y { 1 } else { -1 };
    let mut err = dx + dy;
    let mut out = Vec::with_capacity((dx - dy + 1) as usize);
    out.push((x, y));
    while (x, y) != (x1, y1) {
        let e2 = 2 * err;
        let mut moved_x = false;
        let mut moved_y = false;
        if e2 >= dy {
            err += dy;
            x += sx;
            moved_x = true;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
            moved_y = true;
        }
        if moved_x && moved_y {
            // a diagonal Bresenham step: take the major axis first, then the minor
            if dx >= -dy {
                out.push((x, y - sy));
            } else {
                out.push((x - sx, y));
            }
        }
        out.push((x, y));
    }
    out
}

/// What an execution ended as; the public entry points collapse it to `Option`.
#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    /// The goal was reached; the executed cell path.
    Reached(Vec<(i32, i32)>),
    /// A full circuit of an obstacle with no leave point, or start/goal blocked, or an isolated cell; the
    /// number of moves executed before giving up.
    Unreachable { steps: usize },
    /// `max_steps` moves were executed without a verdict.
    StepCap,
}

fn run(width: usize, height: usize, is_free: &impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32), turn: Turn, max_steps: usize) -> Outcome {
    let in_bounds = |i: i32, j: i32| i >= 0 && j >= 0 && (i as usize) < width && (j as usize) < height;
    // the map edge is an obstacle; every query below goes through `can_step` on the robot's own cell
    let free = |i: i32, j: i32| in_bounds(i, j) && is_free(i, j);
    if !free(start.0, start.1) || !free(goal.0, goal.1) {
        return Outcome::Unreachable { steps: 0 };
    }
    let m = m_line(start, goal);
    let idx: HashMap<(i32, i32), usize> = m.iter().enumerate().map(|(k, &c)| (c, k)).collect();
    let step_ok = |cur: (i32, i32), d: usize| can_step(&free, cur.0, cur.1, DIRS[d].0, DIRS[d].1);
    let dir_of = |from: (i32, i32), to: (i32, i32)| DIRS.iter().position(|&d| d == (to.0 - from.0, to.1 - from.1)).expect("consecutive m-line cells share an edge");

    let mut path = vec![start];
    let mut cur = start;
    let mut i = 0usize;
    let mut steps = 0usize;
    while cur != goal {
        // --- motion-to-goal along the m-line ---
        let nxt = m[i + 1];
        let toward = dir_of(cur, nxt);
        if step_ok(cur, toward) {
            if steps == max_steps {
                return Outcome::StepCap;
            }
            cur = nxt;
            i += 1;
            steps += 1;
            path.push(cur);
            continue;
        }
        // --- hit: turn away from the wall and follow the boundary ---
        let hit = cur;
        let i_hit = i;
        let mut heading = match turn {
            Turn::Left => ccw(toward),
            Turn::Right => cw(toward),
        };
        let heading_hit = heading;
        let mut first_step = true;
        loop {
            if !first_step && cur == hit && heading == heading_hit {
                return Outcome::Unreachable { steps }; // full circuit, same state: no leave point exists
            }
            first_step = false;
            // wall on the right (turned left): try right, straight, left, back; mirrored for the other turn
            let order = match turn {
                Turn::Left => [cw(heading), heading, ccw(heading), back(heading)],
                Turn::Right => [ccw(heading), heading, cw(heading), back(heading)],
            };
            let Some(&d) = order.iter().find(|&&d| step_ok(cur, d)) else {
                return Outcome::Unreachable { steps }; // isolated cell: all four neighbours blocked
            };
            if steps == max_steps {
                return Outcome::StepCap;
            }
            cur = (cur.0 + DIRS[d].0, cur.1 + DIRS[d].1);
            heading = d;
            steps += 1;
            path.push(cur);
            if let Some(&j) = idx.get(&cur) {
                // leave: on the m-line AND strictly closer to the goal along it than the hit point. If the
                // next m-line cell is blocked, motion-to-goal hits again right here, at a strictly larger
                // m-line index than the last hit point (see the module doc, "Leave condition").
                if j > i_hit {
                    i = j;
                    break;
                }
            }
        }
    }
    Outcome::Reached(path)
}

/// Bug2 with the turn convention chosen. `is_free(i, j)` is queried only for cells adjacent to the robot's
/// current cell (plus `start` and `goal` once); it must return `false` off the map, and the planner treats
/// the map edge as an obstacle regardless. Returns the executed cell path from `start` to `goal` inclusive,
/// or `None` when the goal is proven unreachable, when `start` or `goal` is blocked, or when `max_steps`
/// moves were executed without a verdict.
pub fn bug2_turn(width: usize, height: usize, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32), turn: Turn, max_steps: usize) -> Option<Vec<(i32, i32)>> {
    match run(width, height, &is_free, start, goal, turn, max_steps) {
        Outcome::Reached(p) => Some(p),
        Outcome::Unreachable { .. } | Outcome::StepCap => None,
    }
}

/// Bug2 on a `width × height` four-connected grid, turning **left** at each hit point (obstacle kept on the
/// right; the specification's default convention). See [`bug2_turn`] for the contract and the module
/// documentation for the algorithm and what was verified.
pub fn bug2(width: usize, height: usize, is_free: impl Fn(i32, i32) -> bool, start: (i32, i32), goal: (i32, i32), max_steps: usize) -> Option<Vec<(i32, i32)>> {
    bug2_turn(width, height, is_free, start, goal, Turn::Left, max_steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{astar_grid_conn, Connectivity};
    use std::cell::RefCell;

    /// A fixture from the specification's row strings: `'#'` blocked, `'.'` free, `(i, j)` = (column,
    /// row-from-top). Out of bounds is blocked.
    struct Grid {
        rows: Vec<Vec<u8>>,
    }
    impl Grid {
        fn new(rows: &[&str]) -> Self {
            Grid { rows: rows.iter().map(|r| r.as_bytes().to_vec()).collect() }
        }
        fn width(&self) -> usize {
            self.rows[0].len()
        }
        fn height(&self) -> usize {
            self.rows.len()
        }
        fn free(&self, i: i32, j: i32) -> bool {
            i >= 0 && j >= 0 && (j as usize) < self.height() && (i as usize) < self.width() && self.rows[j as usize][i as usize] == b'.'
        }
        fn cap(&self) -> usize {
            default_step_cap(self.width(), self.height())
        }
        fn bug2(&self, start: (i32, i32), goal: (i32, i32), turn: Turn) -> Option<Vec<(i32, i32)>> {
            bug2_turn(self.width(), self.height(), |i, j| self.free(i, j), start, goal, turn, self.cap())
        }
        fn astar_cost(&self, start: (i32, i32), goal: (i32, i32)) -> Option<usize> {
            astar_grid_conn(self.width(), self.height(), Connectivity::Four, |i, j| self.free(i, j), start, goal).map(|p| p.len() - 1)
        }
        /// The invariants every executed path must satisfy: endpoints, every cell free, every step orthogonal
        /// and of length one. Returns the number of moves.
        fn check_path(&self, path: &[(i32, i32)], start: (i32, i32), goal: (i32, i32)) -> usize {
            assert_eq!(path.first(), Some(&start), "path must start at start");
            assert_eq!(path.last(), Some(&goal), "path must end at goal");
            for &(i, j) in path {
                assert!(self.free(i, j), "path cell ({i},{j}) is blocked");
            }
            for w in path.windows(2) {
                assert_eq!((w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs(), 1, "step {:?}->{:?} is not 4-connected", w[0], w[1]);
            }
            path.len() - 1
        }
    }

    // Specification fixtures (planners_grid.json, "Bug2 on a grid"), coordinates (column, row-from-top).
    fn b1() -> Grid {
        Grid::new(&[".........", "...###...", "...###...", "...###...", "...###...", "...###...", "...###...", ".........", "........."])
    }
    fn b2() -> Grid {
        Grid::new(&[".......", ".......", "..###..", "..#.#..", "..###..", ".......", "......."])
    }
    fn b3() -> Grid {
        Grid::new(&[".....", ".....", ".....", ".....", "....."])
    }
    fn b4() -> Grid {
        Grid::new(&["...#...", "...#...", "...#...", "...#...", "...#...", "...#...", "......."])
    }
    /// B5 (this module's addition): B4's wall plus a hook `(1,0)(1,1)(1,2)(2,0)` 8-connected to it through
    /// `(2,0)–(3,1)`, so the left-turn follower goes up column 2 into a dead end, comes back **through the
    /// hit point (2,3) heading south**, and only then rounds the map edge to the goal.
    fn b5() -> Grid {
        Grid::new(&[".###...", ".#.#...", ".#.#...", "...#...", "...#...", "...#...", "......."])
    }

    /// B1, turn left. ORACLE: the specification's hand trace, 2 (to H=(2,4)) + 4 up + 4 across + 4 down +
    /// 2 = 16, listing every cell; reproduced by an independent simulator.
    #[test]
    fn b1_turn_left_executes_the_traced_16_step_path() {
        let g = b1();
        let path = g.bug2((0, 4), (8, 4), Turn::Left).expect("B1 is reachable");
        let expected = [(0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 4), (8, 4)];
        assert_eq!(g.check_path(&path, (0, 4), (8, 4)), 16);
        assert_eq!(path, expected.to_vec());
    }

    /// B1, turn right. ORACLE: the specification's hand trace, 2 + 3 + 4 + 3 + 2 = 14. The two conventions
    /// must differ on this asymmetric obstacle, or `Turn` is decorative.
    #[test]
    fn b1_turn_right_executes_14_and_differs_from_left() {
        let g = b1();
        let left = g.bug2((0, 4), (8, 4), Turn::Left).unwrap();
        let right = g.bug2((0, 4), (8, 4), Turn::Right).unwrap();
        assert_eq!(g.check_path(&right, (0, 4), (8, 4)), 14);
        assert_ne!(left, right, "the obstacle is asymmetric about the m-line, so the two turns must trace different paths");
        assert!(right.contains(&(2, 7)) && !left.contains(&(2, 7)), "the right turn goes below the block");
    }

    /// Bug2 is complete but not optimal: on B1 its executed cost is at least A*'s. Measured: 16 vs 14,
    /// ratio 16/14 = 1.1429 (recorded in the module doc). The inequality is the property; the fixture is
    /// non-vacuous because the two differ.
    #[test]
    fn b1_bug2_is_no_shorter_than_astar_and_here_strictly_longer() {
        let g = b1();
        let bug = g.bug2((0, 4), (8, 4), Turn::Left).unwrap().len() - 1;
        let opt = g.astar_cost((0, 4), (8, 4)).expect("A* finds B1 reachable");
        assert!(bug >= opt, "Bug2 {bug} must not beat the optimum {opt}");
        assert_eq!(opt, 14, "spec: Dijkstra optimum on B1 is 14");
        assert_eq!(bug, 16);
        let ratio = bug as f64 / opt as f64;
        assert!((ratio - 16.0 / 14.0).abs() < 1e-12, "measured ratio {ratio}");
    }

    /// B2: goal sealed in a ring. ORACLE: the specification's hand trace — H=(1,3), 15 following moves
    /// around the ring, (5,3) is met (index 5 > 1) but M[6]=(4,3) is blocked so no leave, return to H →
    /// unreachable after exactly 17 moves. A* agrees the goal is sealed.
    #[test]
    fn b2_sealed_goal_is_unreachable_after_17_moves() {
        let g = b2();
        assert!(g.astar_cost((0, 3), (3, 3)).is_none(), "the ring must actually seal the goal");
        assert_eq!(g.bug2((0, 3), (3, 3), Turn::Left), None);
        let out = run(g.width(), g.height(), &|i, j| g.free(i, j), (0, 3), (3, 3), Turn::Left, g.cap());
        assert_eq!(out, Outcome::Unreachable { steps: 17 }, "must be a proven circuit, not the step cap");
        // the verdict is reached with the cap set exactly at the trace length, and NOT one below it
        assert_eq!(run(g.width(), g.height(), &|i, j| g.free(i, j), (0, 3), (3, 3), Turn::Left, 17), Outcome::Unreachable { steps: 17 });
        assert_eq!(run(g.width(), g.height(), &|i, j| g.free(i, j), (0, 3), (3, 3), Turn::Left, 16), Outcome::StepCap);
    }

    /// B3: empty grid, horizontal m-line, never hits. ORACLE: the m-line itself, 4 moves = Manhattan.
    #[test]
    fn b3_open_grid_walks_the_m_line() {
        let g = b3();
        let path = g.bug2((0, 2), (4, 2), Turn::Left).unwrap();
        assert_eq!(g.check_path(&path, (0, 2), (4, 2)), 4);
        assert_eq!(path, m_line((0, 2), (4, 2)));
        assert_eq!(g.astar_cost((0, 2), (4, 2)), Some(4));
    }

    /// B4, turn right. ORACLE: the specification's hand trace 2 + 3 + 2 + 3 + 2 = 12, which is also the
    /// A* optimum (equality is the spec's claim for this fixture).
    #[test]
    fn b4_turn_right_threads_the_gap_in_12_equal_to_optimal() {
        let g = b4();
        let path = g.bug2((0, 3), (6, 3), Turn::Right).unwrap();
        assert_eq!(g.check_path(&path, (0, 3), (6, 3)), 12);
        assert_eq!(path, vec![(0, 3), (1, 3), (2, 3), (2, 4), (2, 5), (2, 6), (3, 6), (4, 6), (4, 5), (4, 4), (4, 3), (5, 3), (6, 3)]);
        assert_eq!(g.astar_cost((0, 3), (6, 3)), Some(12));
    }

    /// B4, turn left: the follower reaches the map edge at (2,0) and treats it as an obstacle, rounding the
    /// west, south and east edges. ORACLE: hand trace 2 + 3 (up) + 2 (west) + 6 (south) + 6 (east) + 3
    /// (north to the goal cell) = 22, confirmed by the independent simulator. The specification text quotes
    /// 32, the count if the robot walks through the goal without stopping; Lumelsky's procedure stops on
    /// reaching the target and so does this one. The path passes the start cell (index 0 ≤ index(H)=2) and
    /// must not leave there.
    #[test]
    fn b4_turn_left_rounds_the_map_edge_in_22_and_ignores_the_start_cell() {
        let g = b4();
        let path = g.bug2((0, 3), (6, 3), Turn::Left).unwrap();
        assert_eq!(g.check_path(&path, (0, 3), (6, 3)), 22);
        assert_eq!(path[2], (2, 3), "hit point");
        assert_eq!(path[5], (2, 0), "meets the north edge");
        assert!(path[8..].starts_with(&[(0, 1), (0, 2), (0, 3), (0, 4)]), "passes the start cell going south without leaving: {:?}", &path[8..12]);
        assert_eq!(path[path.len() - 4..], [(6, 6), (6, 5), (6, 4), (6, 3)], "arrives at the goal up the east edge");
        // the cap is exactly what the spec says makes this trace legal: it fits under 4WH + |M| with room
        assert!(22 < g.cap());
        assert_eq!(bug2_turn(g.width(), g.height(), |i, j| g.free(i, j), (0, 3), (6, 3), Turn::Left, 21), None, "one move short of the trace must hit the cap");
    }

    /// B5: the boundary passes the hit point twice. ORACLE: hand trace — H=(2,3) heading N; up to (2,1),
    /// dead end, back to (2,3) heading S (same cell, other heading: NOT a circuit); west to (0,3), up the
    /// west edge to (0,0), back down, along the south edge through the gap column (3,6), up the east edge to
    /// the goal: 2 + 2 + 2 + 2 + 3 + 6 + 6 + 3 = 26. Confirmed by the independent simulator, which with
    /// cell-only circuit detection returns unreachable after 8 moves on this reachable goal.
    #[test]
    fn b5_passing_the_hit_point_with_another_heading_is_not_a_circuit() {
        let g = b5();
        let opt = g.astar_cost((0, 3), (6, 3)).expect("B5 is reachable");
        let path = g.bug2((0, 3), (6, 3), Turn::Left).expect("cell-only circuit detection would give None here");
        assert_eq!(g.check_path(&path, (0, 3), (6, 3)), 26);
        assert_eq!(&path[2..7], &[(2, 3), (2, 2), (2, 1), (2, 2), (2, 3)], "the follower re-crosses the hit point heading south");
        assert!(26 >= opt);
        assert_eq!(g.bug2((0, 3), (6, 3), Turn::Right).map(|p| p.len() - 1), Some(12), "the right turn is B4's");
    }

    /// The reactive baseline is never shorter than the optimum on any reachable fixture, and equals it where
    /// the spec says (B3, B4-right).
    #[test]
    fn executed_cost_is_at_least_the_astar_optimum_on_every_reachable_fixture() {
        let cases: [(Grid, (i32, i32), (i32, i32), Turn, usize); 6] =
            [(b1(), (0, 4), (8, 4), Turn::Left, 16), (b1(), (0, 4), (8, 4), Turn::Right, 14), (b3(), (0, 2), (4, 2), Turn::Left, 4), (b4(), (0, 3), (6, 3), Turn::Right, 12), (b4(), (0, 3), (6, 3), Turn::Left, 22), (b5(), (0, 3), (6, 3), Turn::Left, 26)];
        let mut strict = 0;
        for (g, s, t, turn, expected) in cases {
            let bug = g.check_path(&g.bug2(s, t, turn).unwrap(), s, t);
            let opt = g.astar_cost(s, t).unwrap();
            assert_eq!(bug, expected);
            assert!(bug >= opt, "{bug} < {opt}");
            if bug > opt {
                strict += 1;
            }
        }
        assert_eq!(strict, 3, "B1-left, B4-left and B5 are the strictly longer ones");
    }

    /// The m-line is rasterised 4-connected: on an empty grid a diagonal start→goal costs Manhattan 8 with
    /// every step orthogonal. ORACLE: hand rasterisation (0,0)(1,0)(1,1)(2,1)(2,2)(3,2)(3,3)(4,3)(4,4).
    /// With an 8-connected Bresenham (the spec's pitfall) the path would jump (0,0)→(1,1).
    #[test]
    fn diagonal_m_line_is_four_connected() {
        let g = b3();
        let path = g.bug2((0, 0), (4, 4), Turn::Left).unwrap();
        assert_eq!(g.check_path(&path, (0, 0), (4, 4)), 8);
        assert_eq!(path, vec![(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4)]);
        assert_eq!(g.astar_cost((0, 0), (4, 4)), Some(8));
    }

    /// A diagonal m-line with one m-line cell (2,1) blocked. ORACLE: hand trace — H=(1,1) heading E→N;
    /// (1,0) is M[1] (index 1 < 2, no leave), (2,0)(3,0)(3,1)(3,2): M[5] with M[6]=(3,3) free → leave;
    /// (3,3)(4,3)(4,4): 10 moves against an optimum of 8. Confirmed by the independent simulator.
    #[test]
    fn diagonal_m_line_with_a_blocked_cell_hits_and_leaves() {
        let g = Grid::new(&[".....", "..#..", ".....", ".....", "....."]);
        assert!(!g.free(2, 1) && m_line((0, 0), (4, 4)).contains(&(2, 1)), "the blocked cell must lie on the m-line");
        let path = g.bug2((0, 0), (4, 4), Turn::Left).unwrap();
        assert_eq!(g.check_path(&path, (0, 0), (4, 4)), 10);
        assert_eq!(path, vec![(0, 0), (1, 0), (1, 1), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (4, 4)]);
        assert_eq!(g.astar_cost((0, 0), (4, 4)), Some(8));
    }

    /// Two single-cell obstacles on the m-line with one free m-line cell between them. ORACLE: hand trace —
    /// H=(0,2) heading E→N; (0,1)(1,1)(2,1)(2,2): index 2 > 0 → leave; (3,2) blocked → hit again at (2,2)
    /// with i_hit = 2, heading N; (2,1)(3,1)(4,1)(4,2) = goal: 8 moves. A* optimum 6 via row 1. With the
    /// specification pseudocode's extra "next m-line cell free" clause the robot does not leave at (2,2),
    /// circles (1,2) back to (0,2) heading N and returns `None` (reproduced by the independent simulator).
    #[test]
    fn two_obstacles_one_cell_apart_on_the_m_line_are_both_passed() {
        let g = Grid::new(&[".....", ".....", ".#.#.", ".....", "....."]);
        assert_eq!(g.astar_cost((0, 2), (4, 2)), Some(6), "the goal is reachable");
        let m = m_line((0, 2), (4, 2));
        assert!(!g.free(1, 2) && !g.free(3, 2) && m[2] == (2, 2), "both obstacles and the free cell between them lie on the m-line");
        let path = g.bug2((0, 2), (4, 2), Turn::Left).expect("a reachable goal must be reached");
        assert_eq!(g.check_path(&path, (0, 2), (4, 2)), 8);
        assert_eq!(path, vec![(0, 2), (0, 1), (1, 1), (2, 1), (2, 2), (2, 1), (3, 1), (4, 1), (4, 2)]);
        let out = run(5, 5, &|i, j| g.free(i, j), (0, 2), (4, 2), Turn::Left, g.cap());
        assert!(matches!(out, Outcome::Reached(_)));
    }

    /// `m_line` in every octant: endpoints, edge-adjacent consecutive cells, no repeats, exactly
    /// `|Δx|+|Δy|+1` cells, and each cell within one unit of the continuous segment in the minor coordinate.
    #[test]
    fn m_line_is_edge_connected_monotone_and_hugs_the_segment() {
        let ends = [(7, 3), (3, 7), (-7, 3), (-3, 7), (7, -3), (3, -7), (-7, -3), (-3, -7), (5, 5), (-5, 5), (6, 0), (0, -6), (0, 0)];
        let mut n_diagonal_lines = 0;
        for &(x1, y1) in &ends {
            let m = m_line((0, 0), (x1, y1));
            assert_eq!(m.first(), Some(&(0, 0)));
            assert_eq!(m.last(), Some(&(x1, y1)));
            assert_eq!(m.len(), (x1.abs() + y1.abs() + 1) as usize, "{:?}: {:?}", (x1, y1), m);
            for w in m.windows(2) {
                assert_eq!((w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs(), 1);
            }
            let mut seen = m.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), m.len(), "repeated cell");
            if x1 != 0 && y1 != 0 {
                n_diagonal_lines += 1;
                for &(x, y) in &m {
                    let (x, y, x1, y1) = (x as f64, y as f64, x1 as f64, y1 as f64);
                    let dev = if x1.abs() >= y1.abs() { y - y1 * x / x1 } else { x - x1 * y / y1 };
                    assert!(dev.abs() <= 1.0 + 1e-12, "cell ({x},{y}) deviates {dev} from the segment to ({x1},{y1})");
                }
            }
        }
        assert_eq!(n_diagonal_lines, 10, "the octant sweep must actually contain diagonal lines");
    }

    /// Sensor-based: every `is_free` query lies within one cell of some traversed cell (plus start/goal).
    /// Non-vacuous: the B1 run makes more than one query per path cell and the map has cells two away from
    /// the path that are never queried.
    #[test]
    fn every_map_query_is_adjacent_to_the_executed_path() {
        let g = b1();
        let queries = RefCell::new(Vec::new());
        let path = bug2(
            g.width(),
            g.height(),
            |i, j| {
                queries.borrow_mut().push((i, j));
                g.free(i, j)
            },
            (0, 4),
            (8, 4),
            g.cap(),
        )
        .unwrap();
        let queries = queries.into_inner();
        assert!(queries.len() > path.len(), "{} queries for {} cells", queries.len(), path.len());
        for &(qi, qj) in &queries {
            assert!(path.iter().any(|&(pi, pj)| (pi - qi).abs() + (pj - qj).abs() <= 1), "query ({qi},{qj}) is not adjacent to the executed path");
        }
        assert!(!queries.contains(&(4, 8)), "a cell two rows from the path was never sensed");
    }

    /// Start or goal blocked, start equal to goal, and an isolated start.
    #[test]
    fn degenerate_inputs() {
        let g = b1();
        assert_eq!(g.bug2((3, 3), (8, 4), Turn::Left), None, "start inside the obstacle");
        assert_eq!(g.bug2((0, 4), (4, 4), Turn::Left), None, "goal inside the obstacle");
        assert_eq!(g.bug2((0, 4), (9, 4), Turn::Left), None, "goal off the map");
        assert_eq!(g.bug2((0, 4), (0, 4), Turn::Left), Some(vec![(0, 4)]), "start == goal");
        let boxed = Grid::new(&["#.#", ".#.", "..."]);
        assert_eq!(boxed.bug2((1, 0), (2, 2), Turn::Left), None, "start with all four neighbours blocked");
        assert_eq!(run(3, 3, &|i, j| boxed.free(i, j), (1, 0), (2, 2), Turn::Left, 100), Outcome::Unreachable { steps: 0 });
    }

    /// The hard bound is on moves: B1's 16-move trace succeeds with `max_steps = 16` and is cut at 15.
    #[test]
    fn max_steps_bounds_the_executed_moves_exactly() {
        let g = b1();
        assert!(bug2(g.width(), g.height(), |i, j| g.free(i, j), (0, 4), (8, 4), 16).is_some());
        assert_eq!(bug2(g.width(), g.height(), |i, j| g.free(i, j), (0, 4), (8, 4), 15), None);
        assert_eq!(run(g.width(), g.height(), &|i, j| g.free(i, j), (0, 4), (8, 4), Turn::Left, 15), Outcome::StepCap);
        assert_eq!(bug2(g.width(), g.height(), |i, j| g.free(i, j), (0, 4), (8, 4), 0), None, "zero moves allowed, goal not at start");
    }
}
