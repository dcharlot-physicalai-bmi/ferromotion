//! **D\* Lite** — incremental shortest-path replanning on a grid whose cells flip between free and blocked
//! while the robot moves. Where [`crate::astar_grid_conn`] answers one query from scratch, D\* Lite keeps
//! its search rooted at the **goal** (so the values it stores are costs-to-goal and the start may move) and
//! after a map change re-expands only the vertices whose cost-to-goal actually changed. It is the
//! optimized variant of Koenig & Likhachev, "Fast Replanning for Navigation in Unknown Terrain", IEEE
//! Trans. Robotics 21(3):354–363, 2005, Figure 4 (the `k_m` offset, the `k_old < k_new` re-sort, and the
//! `g_old` test in the raise branch), first published at AAAI 2002 pp. 476–483; the problem statement —
//! a partially known environment sensed as the robot moves — is Stentz, ICRA 1994 pp. 3310–3317.
//!
//! The neighbourhood, the step costs (1 orthogonal, √2 diagonal) and the corner rule are not defined here:
//! they come from [`Connectivity`] and [`can_step`], the same functions [`crate::astar_grid_conn`] uses,
//! which is what makes that planner a valid oracle for this one.
//!
//! Verified against `astar_grid_conn` on the same grid: the five fixtures of the planner specification
//! (DL1a/DL1b 7×7 eight-connected, DL2a/DL2b 5×5 four-connected, DL3 sealed) return the hand-computed
//! costs `6√2`, `4 + 4√2`, `4`, `8` (and `7` after the robot moves one cell), and `None`, each equal to
//! A\* on the post-change grid to 1e-9; the greedy path's length equals the reported cost; re-planning
//! with no change expands zero vertices; a scripted 12×12 walk with four sensed changes matches A\* at
//! every replan (87 expansions over its five plans); and 4000 random grids (5–9 cells a side, 25% blocked,
//! both connectivities) walked for up to 8 replans each with 1–4 random cell flips before every replan
//! agree with A\* at all 26,000-odd replans. That last test is what found the one deviation from the
//! specification's transcription: keys are compared with a tie tolerance ([`key_cmp`]) rather than
//! exactly, because with exact `f64` comparison 11 of those 4000 trials returned a cost below the optimum
//! — a real tie in `k1` rounded an ulp the wrong way. Pure Rust → WASM-clean.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::grid_astar::{can_step, Connectivity, OrdF};

/// A priority-queue key `[k1; k2]`. Stale-entry detection compares keys bit-exactly (the same stored
/// floats); ordering goes through [`key_cmp`]. The `rhs(s) == c + g_old` test in the raise branch is a
/// separate, bit-exact comparison of the same expression that produced `rhs`.
type Key = (OrdF, OrdF);

const INF: f64 = f64::INFINITY;
const INF_KEY: Key = (OrdF(INF), OrdF(INF));

/// Tolerance under which two key components are the same real number (see [`key_cmp`]).
const KEY_EPS: f64 = 1e-6;

/// Order of one key component: equal when the two `f64`s are within [`KEY_EPS`] (or identical, which
/// covers `INF`), otherwise the total order of [`OrdF`].
fn comp_cmp(x: OrdF, y: OrdF) -> Ordering {
    if x == y || (x.0 - y.0).abs() <= KEY_EPS { Ordering::Equal } else { x.cmp(&y) }
}

/// The key order of Figure 4 — lexicographic on `(k1, k2)` — with components closer than [`KEY_EPS`]
/// treated as equal so that `k2` breaks a `k1` tie the way the paper intends. Used for the queue order,
/// the loop test `U.TopKey() < CalculateKey(s_start)` and the re-sort test `k_old < k_new`.
///
/// Every `g`, `rhs`, `h` and `k_m` on this grid is a real number `a + b·√2` with integers `a, b`, and the
/// algorithm's correctness rests on **exact ties** in `k1`: a vertex whose stale `g` still promises the
/// start's current `rhs` has `g(u) + h(s_start, u) = rhs(s_start)` exactly, and only `k2 = g(u) <
/// rhs(s_start)` orders it before the start (and before stale entries with the same `k1`). In `f64` the two
/// sides are computed by different additions and can land an ulp apart in either direction, which an exact
/// comparison turns into a wrong verdict. Two measured cases, each a regression test below:
/// `(1 + √2) + octile((0,4), (2,7))` is `6.242640687119286` while `1 + (√2 + (√2 + (√2 + 1)))` is
/// `6.242640687119285`, so an exact loop test stopped with `(2,7)` inconsistent and returned `2 + 3√2`
/// where the optimum is `4 + 2√2`; and `(2 + √2) + 2√2 + 1` is `7.242640687119286` while
/// `√2 + (√2 + (√2 + 2)) + 1` is `7.242640687119285`, so an exactly ordered queue placed a vertex with
/// `k2 = 2 + √2` *below* a stale entry with the same real `k1` and `k2 = 3 + 3√2`, and the loop test on
/// that stale top stopped the search. The spec's instruction to compare keys exactly is what fails here.
///
/// Why `1e-6` separates ties from real differences: two distinct values `a + b√2` satisfy
/// `|a + b√2| · |a − b√2| = |a² − 2b²| ≥ 1`, so a key difference with `|Δa|, |Δb| ≤ 2·10⁴` is at least
/// `1 / (2·10⁴·(1 + √2)) ≈ 2.1e-5`, while a component accumulated over at most `10⁴` additions of values
/// below `1.5·10⁴` (ulp `1.8e-12`) carries rounding error below `10⁴ · 0.9e-12 ≈ 1e-8` per side, plus one
/// ulp of the final sum (`1.9e-9` at `10⁷`). Both are more than an order of magnitude from `1e-6`, so on
/// grids of up to `10⁴` cells (paths and `g` differences bounded by that) the comparison equals the
/// real-number order and is therefore transitive, which is what [`BinaryHeap`] requires. Beyond that —
/// paths over `10⁴` steps, or `k_m` beyond `10⁷` — the margin shrinks and the guarantee lapses.
fn key_cmp(a: Key, b: Key) -> Ordering {
    comp_cmp(a.0, b.0).then_with(|| comp_cmp(a.1, b.1))
}

/// A queue entry, ordered by [`key_cmp`] and then by cell index so the order is total.
#[derive(Clone, Copy, Debug)]
struct Entry {
    key: Key,
    cell: usize,
}
impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        key_cmp(self.key, o.key).then_with(|| self.cell.cmp(&o.cell))
    }
}

/// Incremental grid planner: owns the blocked map, the `g`/`rhs` tables, the priority queue and `k_m`.
///
/// Cells are `(i, j)` with `i` the column and `j` the row, stored row-major at `j * width + i` — the
/// layout of [`crate::OccupancyGrid`]. Out-of-bounds cells are blocked. A blocked start or goal is not
/// an error at construction (the robot may be standing where the map now says it cannot); [`Self::plan`]
/// then reports `None`.
#[derive(Clone, Debug)]
pub struct DStarLite {
    width: usize,
    height: usize,
    conn: Connectivity,
    blocked: Vec<bool>,
    /// Cost-to-goal estimate per cell, `INF` until expanded.
    g: Vec<f64>,
    /// One-step lookahead `min_{s'} c(s, s') + g(s')`, `0` at the goal.
    rhs: Vec<f64>,
    /// Current key of each cell in the queue, `None` when it is not queued. The heap is lazily deleted:
    /// an entry whose key differs from this table is stale and is discarded on pop.
    key_of: Vec<Option<Key>>,
    heap: BinaryHeap<Reverse<Entry>>,
    km: f64,
    start: (i32, i32),
    /// Where the robot was when `km` was last increased (`s_last` in the paper).
    last: (i32, i32),
    goal: (i32, i32),
    expansions: usize,
}

impl DStarLite {
    /// A planner over a `width × height` map (`blocked.len()` must equal `width * height`, row-major) from
    /// `start` to `goal`. `None` when either cell is out of bounds or the map has the wrong size.
    ///
    /// This is `Initialize()` of the paper: every `g` and `rhs` is `INF`, `rhs(goal) = 0`, and the goal is
    /// queued with key `(h(start, goal), 0)`.
    pub fn new(width: usize, height: usize, conn: Connectivity, blocked: Vec<bool>, start: (i32, i32), goal: (i32, i32)) -> Option<Self> {
        if blocked.len() != width * height {
            return None;
        }
        let n = width * height;
        let mut p = DStarLite {
            width,
            height,
            conn,
            blocked,
            g: vec![INF; n],
            rhs: vec![INF; n],
            key_of: vec![None; n],
            heap: BinaryHeap::new(),
            km: 0.0,
            start,
            last: start,
            goal,
            expansions: 0,
        };
        if !p.in_bounds(start.0, start.1) || !p.in_bounds(goal.0, goal.1) {
            return None;
        }
        let gi = p.idx(goal);
        p.rhs[gi] = 0.0;
        let k = p.calculate_key(gi);
        p.push(gi, k);
        Some(p)
    }

    /// [`Self::new`] over an occupancy grid, taking every cell `grid.blocked(i, j)` reports (out of bounds
    /// counts as blocked there too, so the map edge is a wall).
    pub fn from_grid(grid: &crate::OccupancyGrid, conn: Connectivity, start: (i32, i32), goal: (i32, i32)) -> Option<Self> {
        let (w, h) = (grid.width, grid.height);
        let blocked = (0..h).flat_map(|j| (0..w).map(move |i| grid.blocked(i as i64, j as i64))).collect();
        DStarLite::new(w, h, conn, blocked, start, goal)
    }

    /// The robot's current cell.
    pub fn start(&self) -> (i32, i32) {
        self.start
    }

    /// The goal cell.
    pub fn goal(&self) -> (i32, i32) {
        self.goal
    }

    /// The `k_m` offset accumulated over the robot's moves.
    pub fn km(&self) -> f64 {
        self.km
    }

    /// Whether `(i, j)` is blocked (out of bounds counts as blocked).
    pub fn is_blocked(&self, i: i32, j: i32) -> bool {
        !self.in_bounds(i, j) || self.blocked[self.idx((i, j))]
    }

    /// Total vertices expanded (lower or raise) over the planner's life. Re-planning after no change must
    /// add zero, which is the property that makes the planner incremental rather than merely correct.
    pub fn expansions(&self) -> usize {
        self.expansions
    }

    /// The robot moved to `s`. `k_m += h(s_last, s)` and `s_last = s`, so keys already in the queue stay
    /// lower bounds of the keys they would have under the new start (the paper's lines {30"}–{31"}). The
    /// paper adds the increment only when a change is sensed after the move; adding it on every move
    /// is also valid, because `h` satisfies the triangle inequality so a sum of per-move increments is at
    /// least the one-shot increment. An out-of-bounds `s` is ignored.
    pub fn set_start(&mut self, s: (i32, i32)) {
        if !self.in_bounds(s.0, s.1) {
            return;
        }
        self.km += self.conn.heuristic(self.last, s);
        self.last = s;
        self.start = s;
    }

    /// A sensed change: cell `(i, j)` is now `blocked` (or free). Applies the edge-cost update of the
    /// paper's lines {33"}–{40"} to every directed edge whose cost changed:
    /// the edges into and out of the cell, and under [`Connectivity::Eight`] the diagonal edges between
    /// each pair of its orthogonal neighbours, whose corner rule the cell participates in.
    ///
    /// The old cost of each edge is read **before** the flip and the new cost after, as the specification
    /// requires; a flip to the state the cell already has is a no-op. Several changes may be applied before
    /// one [`Self::plan`]; nothing is searched here.
    pub fn update_cell(&mut self, i: i32, j: i32, blocked: bool) {
        if !self.in_bounds(i, j) {
            return;
        }
        let c = (i, j);
        let ci = self.idx(c);
        if self.blocked[ci] == blocked {
            return;
        }
        // every directed edge whose cost can depend on this cell, with its cost BEFORE the flip
        let mut edges: Vec<((i32, i32), (i32, i32), f64)> = Vec::with_capacity(24);
        for &(di, dj, _) in self.conn.steps() {
            let n = (i + di, j + dj);
            if self.in_bounds(n.0, n.1) {
                edges.push((n, c, self.cost(n, c)));
                edges.push((c, n, self.cost(c, n)));
            }
        }
        if self.conn == Connectivity::Eight {
            // the four diagonals that pass this cell as a corner: (i-1,j)<->(i,j-1), (i,j-1)<->(i+1,j),
            // (i+1,j)<->(i,j+1), (i,j+1)<->(i-1,j)
            let ring = [(i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)];
            for w in ring.windows(2) {
                let (a, b) = (w[0], w[1]);
                if self.in_bounds(a.0, a.1) && self.in_bounds(b.0, b.1) {
                    edges.push((a, b, self.cost(a, b)));
                    edges.push((b, a, self.cost(b, a)));
                }
            }
        }
        self.blocked[ci] = blocked;
        for (u, v, c_old) in edges {
            let c_new = self.cost(u, v);
            if c_old == c_new {
                continue;
            }
            let (ui, vi) = (self.idx(u), self.idx(v));
            if c_old > c_new {
                if u != self.goal {
                    self.rhs[ui] = self.rhs[ui].min(c_new + self.g[vi]);
                }
            } else if u != self.goal && self.rhs[ui] == c_old + self.g[vi] {
                self.rhs[ui] = self.min_rhs(u);
            }
            self.update_vertex(ui);
        }
    }

    /// Run `ComputeShortestPath()` from the current start and return `(cost, path)`: `cost = rhs(start)`
    /// and the path from start to goal (inclusive) obtained by greedily stepping to the successor
    /// minimising `c(s, s') + g(s')`, ties broken by the fixed order of [`Connectivity::steps`]. `None`
    /// when the goal is unreachable (`rhs(start) = INF` after the queue drains), the start is blocked, or
    /// the greedy walk fails to reach the goal within `width * height` steps (which cannot happen when
    /// the `g` table is consistent, and is guarded against so a path extraction can never loop).
    ///
    /// The cost is `rhs(start)`, not `g(start)`, because Figure 4's loop condition `rhs(s_start) >
    /// g(s_start)` stops as soon as the start is at the top of the queue over-consistent (`rhs < g`)
    /// without expanding it, and its line {29"} tests `rhs(s_start)` for reachability. `rhs(start)` is by
    /// definition the first greedy step's `c + g`, so it is exactly the length of the returned path,
    /// which the tests assert.
    pub fn plan(&mut self) -> Option<(f64, Vec<(i32, i32)>)> {
        self.compute_shortest_path();
        let si = self.idx(self.start);
        let cost = self.rhs[si];
        if !cost.is_finite() {
            return None;
        }
        let mut path = vec![self.start];
        let mut s = self.start;
        let limit = self.width * self.height;
        while s != self.goal {
            if path.len() > limit {
                return None;
            }
            let mut best: Option<((i32, i32), f64)> = None;
            for &(di, dj, _) in self.conn.steps() {
                let n = (s.0 + di, s.1 + dj);
                if !self.in_bounds(n.0, n.1) {
                    continue;
                }
                let v = self.cost(s, n) + self.g[self.idx(n)];
                if v.is_finite() && best.is_none_or(|(_, b)| v < b) {
                    best = Some((n, v));
                }
            }
            s = best?.0;
            path.push(s);
        }
        Some((cost, path))
    }

    // ---- internals -------------------------------------------------------------------------------

    fn in_bounds(&self, i: i32, j: i32) -> bool {
        i >= 0 && j >= 0 && (i as usize) < self.width && (j as usize) < self.height
    }

    fn idx(&self, s: (i32, i32)) -> usize {
        (s.1 as usize) * self.width + (s.0 as usize)
    }

    fn cell(&self, k: usize) -> (i32, i32) {
        ((k % self.width) as i32, (k / self.width) as i32)
    }

    /// Edge cost `c(u, v)`: the step cost from [`Connectivity::steps`] when `u` is free and the step passes
    /// [`can_step`] (destination free, corner rule), `INF` otherwise. Symmetric, because the step set and the
    /// corner rule are.
    fn cost(&self, u: (i32, i32), v: (i32, i32)) -> f64 {
        let free = |i: i32, j: i32| !self.is_blocked(i, j);
        if !free(u.0, u.1) {
            return INF;
        }
        let (di, dj) = (v.0 - u.0, v.1 - u.1);
        match self.conn.steps().iter().find(|&&(si, sj, _)| si == di && sj == dj) {
            Some(&(_, _, c)) if can_step(&free, u.0, u.1, di, dj) => c,
            _ => INF,
        }
    }

    /// `min_{s' ∈ Succ(u)} c(u, s') + g(s')`.
    fn min_rhs(&self, u: (i32, i32)) -> f64 {
        let mut m = INF;
        for &(di, dj, _) in self.conn.steps() {
            let n = (u.0 + di, u.1 + dj);
            if self.in_bounds(n.0, n.1) {
                m = m.min(self.cost(u, n) + self.g[self.idx(n)]);
            }
        }
        m
    }

    /// `CalculateKey(s) = [min(g, rhs) + h(s_start, s) + k_m; min(g, rhs)]` — the heuristic runs from the
    /// **start** to the vertex, since the search is goal-rooted.
    fn calculate_key(&self, k: usize) -> Key {
        let m = self.g[k].min(self.rhs[k]);
        (OrdF(m + self.conn.heuristic(self.start, self.cell(k)) + self.km), OrdF(m))
    }

    fn push(&mut self, k: usize, key: Key) {
        self.key_of[k] = Some(key);
        self.heap.push(Reverse(Entry { key, cell: k }));
    }

    /// `UpdateVertex(u)`: queued iff locally inconsistent (`g ≠ rhs`), always with a fresh key.
    fn update_vertex(&mut self, k: usize) {
        if self.g[k] != self.rhs[k] {
            let key = self.calculate_key(k);
            self.push(k, key);
        } else {
            self.key_of[k] = None;
        }
    }

    /// The queue's top after discarding stale entries: `(cell, key)`, or `None` when the queue is empty.
    fn top(&mut self) -> Option<(usize, Key)> {
        while let Some(Reverse(e)) = self.heap.peek().copied() {
            if self.key_of[e.cell] == Some(e.key) {
                return Some((e.cell, e.key));
            }
            self.heap.pop();
        }
        None
    }

    /// `ComputeShortestPath()` of Figure 4: pop while the top key is below the start's key or the start is
    /// inconsistent; re-sort a vertex whose key grew (`k_m` moved), lower it when `g > rhs`, raise it
    /// otherwise. Each expansion is counted in [`Self::expansions`].
    fn compute_shortest_path(&mut self) {
        let si = self.idx(self.start);
        loop {
            let top = self.top();
            let k_old = top.map_or(INF_KEY, |t| t.1);
            let k_start = self.calculate_key(si);
            if !(key_cmp(k_old, k_start) == Ordering::Less || self.rhs[si] > self.g[si]) {
                return;
            }
            let Some((u, _)) = top else {
                return; // queue drained: the start is unreachable (rhs(start) = INF)
            };
            let k_new = self.calculate_key(u);
            let uc = self.cell(u);
            if key_cmp(k_old, k_new) == Ordering::Less {
                // the stored key predates a k_m increase: re-sort, do not expand
                self.push(u, k_new);
            } else if self.g[u] > self.rhs[u] {
                // LOWER: u became cheaper
                self.g[u] = self.rhs[u];
                self.key_of[u] = None;
                self.expansions += 1;
                for &(di, dj, _) in self.conn.steps() {
                    let s = (uc.0 + di, uc.1 + dj);
                    if !self.in_bounds(s.0, s.1) {
                        continue;
                    }
                    let k = self.idx(s);
                    if s != self.goal {
                        self.rhs[k] = self.rhs[k].min(self.cost(s, uc) + self.g[u]);
                    }
                    self.update_vertex(k);
                }
            } else {
                // RAISE: u became dearer; every predecessor that reached the goal through u re-evaluates
                let g_old = self.g[u];
                self.g[u] = INF;
                self.expansions += 1;
                for &(di, dj, _) in self.conn.steps() {
                    let s = (uc.0 + di, uc.1 + dj);
                    if !self.in_bounds(s.0, s.1) {
                        continue;
                    }
                    let k = self.idx(s);
                    if self.rhs[k] == self.cost(s, uc) + g_old && s != self.goal {
                        self.rhs[k] = self.min_rhs(s);
                    }
                    self.update_vertex(k);
                }
                self.update_vertex(u);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_astar::{astar_grid_conn, path_length};
    use crate::OccupancyGrid;

    const SQRT2: f64 = std::f64::consts::SQRT_2;

    /// Spec fixtures give `(x, y)` with `y` the row from the TOP; `OccupancyGrid::from_rows` puts the top row
    /// at `j = height - 1`. Costs are invariant under the flip.
    fn cell(grid: &OccupancyGrid, xy: (i32, i32)) -> (i32, i32) {
        (xy.0, grid.height as i32 - 1 - xy.1)
    }

    fn grid(rows: &[&str]) -> OccupancyGrid {
        OccupancyGrid::from_rows(rows, 1.0, 0.0, 0.0).expect("well-formed drawing")
    }

    /// THE ORACLE: from-scratch A* on the same grid through the same `Connectivity` / `can_step`.
    fn astar_cost(grid: &OccupancyGrid, conn: Connectivity, start: (i32, i32), goal: (i32, i32)) -> Option<f64> {
        let free = |i: i32, j: i32| !grid.blocked(i as i64, j as i64);
        astar_grid_conn(grid.width, grid.height, conn, free, start, goal).map(|p| path_length(&p))
    }

    /// The path must be walkable on the planner's own map: every step passes the corner rule, and its
    /// length is the cost the planner reported (the spec's cycle guard, stated as an equality).
    fn assert_path_consistent(p: &DStarLite, cost: f64, path: &[(i32, i32)]) {
        assert_eq!(path.first(), Some(&p.start()));
        assert_eq!(path.last(), Some(&p.goal()));
        let free = |i: i32, j: i32| !p.is_blocked(i, j);
        for w in path.windows(2) {
            assert!(can_step(&free, w[0].0, w[0].1, w[1].0 - w[0].0, w[1].1 - w[0].1), "step {:?}->{:?} is not allowed", w[0], w[1]);
        }
        assert!((path_length(path) - cost).abs() < 1e-9, "path length {} vs reported cost {}", path_length(path), cost);
    }

    const DL1A: [&str; 7] = [".......", ".......", ".......", ".......", ".......", ".......", "......."];
    const DL1B: [&str; 7] = [".......", ".......", ".......", "...#...", ".......", ".......", "......."];
    const DL2A: [&str; 5] = ["..#..", "..#..", ".....", "..#..", "....."];
    const DL2B: [&str; 5] = ["..#..", "..#..", "..#..", "..#..", "....."];
    const DL3: [&str; 5] = ["..#..", "..#..", "..#..", "..#..", "..#.."];

    /// DL1a (spec): empty 7x7, eight-connected, corner to corner. By hand: six diagonal moves, `6·√2`.
    #[test]
    fn dl1a_open_grid_costs_six_diagonals() {
        let g = grid(&DL1A);
        let (s, t) = (cell(&g, (0, 0)), cell(&g, (6, 6)));
        let mut p = DStarLite::from_grid(&g, Connectivity::Eight, s, t).unwrap();
        let (cost, path) = p.plan().expect("open grid is reachable");
        assert!(cost > 0.0 && path.len() == 7, "non-vacuous: a real path of six steps, got {} cells", path.len());
        assert!((cost - 6.0 * SQRT2).abs() < 1e-9, "cost {cost}");
        assert!((cost - astar_cost(&g, Connectivity::Eight, s, t).unwrap()).abs() < 1e-9, "must equal A*");
        assert_path_consistent(&p, cost, &path);
    }

    /// DL1b (spec): after DL1a, the sensor reports (3,3) blocked before the robot moves (`k_m` stays 0).
    /// By hand: the diagonal is broken and no step may cut the corner of (3,3), so the detour from (2,2) to
    /// (4,4) costs 4 orthogonal moves instead of `2√2`: `4 + 4√2 = 9.6569`. With the corner rule missing
    /// the cost would be `2 + 5√2 = 9.07`, so the fixture separates the two.
    #[test]
    fn dl1b_blocking_the_diagonal_cell_replans_around_its_corners() {
        let g0 = grid(&DL1A);
        let (s, t) = (cell(&g0, (0, 0)), cell(&g0, (6, 6)));
        let mut p = DStarLite::from_grid(&g0, Connectivity::Eight, s, t).unwrap();
        let (c0, _) = p.plan().unwrap();
        let b = cell(&g0, (3, 3));
        p.update_cell(b.0, b.1, true);
        assert_eq!(p.km(), 0.0, "no move, so k_m stays 0");
        let (cost, path) = p.plan().expect("still reachable");
        assert!(cost > c0 + 1.0, "non-vacuous: the block must cost something, {c0} -> {cost}");
        assert!((cost - (4.0 + 4.0 * SQRT2)).abs() < 1e-9, "cost {cost}");
        let g1 = grid(&DL1B);
        assert!((cost - astar_cost(&g1, Connectivity::Eight, s, t).unwrap()).abs() < 1e-9, "must equal from-scratch A* on the changed grid");
        assert!(!path.contains(&b), "path passes through the blocked cell");
        assert_path_consistent(&p, cost, &path);
    }

    /// DL2a (spec): 5x5 four-connected, wall in column 2 with gaps at rows 2 and 4, straight through the
    /// row-2 gap: 4 unit moves, and 4 is the Manhattan lower bound.
    #[test]
    fn dl2a_four_connected_through_the_gap() {
        let g = grid(&DL2A);
        let (s, t) = (cell(&g, (0, 2)), cell(&g, (4, 2)));
        let mut p = DStarLite::from_grid(&g, Connectivity::Four, s, t).unwrap();
        let (cost, path) = p.plan().unwrap();
        assert_eq!(path.len(), 5, "non-vacuous: four steps");
        assert!((cost - 4.0).abs() < 1e-9, "cost {cost}");
        assert!((cost - astar_cost(&g, Connectivity::Four, s, t).unwrap()).abs() < 1e-9, "must equal A*");
        assert_path_consistent(&p, cost, &path);
    }

    /// DL2b(i) (spec): the robot is still at (0,2) when (2,2) is reported blocked. By hand: the only gap is
    /// (2,4): down 2, across 3, up 2, right 1 = 8.
    #[test]
    fn dl2b_i_blocking_the_gap_from_a_standing_robot() {
        let g0 = grid(&DL2A);
        let (s, t) = (cell(&g0, (0, 2)), cell(&g0, (4, 2)));
        let mut p = DStarLite::from_grid(&g0, Connectivity::Four, s, t).unwrap();
        let (c0, _) = p.plan().unwrap();
        let b = cell(&g0, (2, 2));
        p.update_cell(b.0, b.1, true);
        let (cost, path) = p.plan().expect("still reachable through (2,4)");
        assert!(cost > c0, "non-vacuous: {c0} -> {cost}");
        assert!((cost - 8.0).abs() < 1e-9, "cost {cost}");
        assert!((cost - astar_cost(&grid(&DL2B), Connectivity::Four, s, t).unwrap()).abs() < 1e-9, "must equal A* on the changed grid");
        assert_path_consistent(&p, cost, &path);
    }

    /// DL2b(ii) (spec): the robot first moves one step to (1,2), THEN senses (2,2) blocked, so
    /// `k_m += h((0,2),(1,2)) = 1`. By hand from (1,2): 2 + 2 + 2 + 1 = 7.
    #[test]
    fn dl2b_ii_robot_moves_then_senses_the_block() {
        let g0 = grid(&DL2A);
        let (s, t) = (cell(&g0, (0, 2)), cell(&g0, (4, 2)));
        let mut p = DStarLite::from_grid(&g0, Connectivity::Four, s, t).unwrap();
        let (c0, path0) = p.plan().unwrap();
        assert_eq!(path0[1], cell(&g0, (1, 2)), "the first plan steps right along row 2");
        p.set_start(path0[1]);
        assert!((p.km() - 1.0).abs() < 1e-12, "k_m = h((0,2),(1,2)) = 1, got {}", p.km());
        let b = cell(&g0, (2, 2));
        p.update_cell(b.0, b.1, true);
        let (cost, path) = p.plan().expect("still reachable");
        assert!(cost > c0, "non-vacuous: {c0} -> {cost}");
        assert!((cost - 7.0).abs() < 1e-9, "cost {cost}");
        assert!((cost - astar_cost(&grid(&DL2B), Connectivity::Four, path0[1], t).unwrap()).abs() < 1e-9, "must equal A* from the new start");
        assert_path_consistent(&p, cost, &path);
    }

    /// DL2b reverse (spec): from the DL2b grid, free (2,2) again and require 4 — the edge-cost DECREASE
    /// branch of the update, which the blocking fixtures never exercise.
    #[test]
    fn dl2b_reverse_freeing_the_gap_restores_the_straight_path() {
        let g0 = grid(&DL2B);
        let (s, t) = (cell(&g0, (0, 2)), cell(&g0, (4, 2)));
        let mut p = DStarLite::from_grid(&g0, Connectivity::Four, s, t).unwrap();
        let (c0, _) = p.plan().unwrap();
        assert!((c0 - 8.0).abs() < 1e-9, "non-vacuous: starts at the detour cost, got {c0}");
        let b = cell(&g0, (2, 2));
        p.update_cell(b.0, b.1, false);
        let (cost, path) = p.plan().unwrap();
        assert!((cost - 4.0).abs() < 1e-9, "cost {cost}");
        assert!((cost - astar_cost(&grid(&DL2A), Connectivity::Four, s, t).unwrap()).abs() < 1e-9, "must equal A* on the freed grid");
        assert!(path.contains(&b), "the straight path uses the freed gap");
        assert_path_consistent(&p, cost, &path);
    }

    /// DL3 (spec): column 2 blocked in every row splits the map; `None` under both connectivities (there is
    /// no gap to cut through and the corner rule forbids it anyway), the queue drains and the loop ends.
    /// Then freeing one wall cell makes the goal reachable again, so the `None` is not a permanently
    /// sealed planner.
    #[test]
    fn dl3_sealed_wall_returns_none_and_terminates() {
        let g = grid(&DL3);
        let (s, t) = (cell(&g, (0, 2)), cell(&g, (4, 2)));
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let mut p = DStarLite::from_grid(&g, conn, s, t).unwrap();
            assert!(p.plan().is_none(), "{conn:?}: sealed");
            assert_eq!(astar_cost(&g, conn, s, t), None, "{conn:?}: A* agrees the goal is sealed");
            assert!(p.expansions() > 0, "non-vacuous: the search actually ran before draining");
            let b = cell(&g, (2, 2));
            p.update_cell(b.0, b.1, false);
            let (cost, path) = p.plan().expect("freed gap");
            assert!((cost - 4.0).abs() < 1e-9, "{conn:?}: cost {cost}");
            assert_path_consistent(&p, cost, &path);
        }
    }

    /// Spec pitfall: a planner can be correct and still not incremental. Planning again with no change
    /// must expand zero vertices.
    #[test]
    fn replanning_without_changes_expands_nothing() {
        let g = grid(&DL1B);
        let (s, t) = (cell(&g, (0, 0)), cell(&g, (6, 6)));
        let mut p = DStarLite::from_grid(&g, Connectivity::Eight, s, t).unwrap();
        let (c0, _) = p.plan().unwrap();
        let e0 = p.expansions();
        assert!(e0 > 0, "non-vacuous: the first plan expanded something");
        let (c1, _) = p.plan().unwrap();
        assert_eq!(p.expansions(), e0, "second plan with no change must expand nothing");
        assert_eq!(c0, c1);
    }

    /// A scripted walk on a 12x12 eight-connected grid: the robot follows its plan several cells at a time,
    /// senses a batch of changes (walls appear ahead, one earlier wall cell opens), and replans. At every
    /// replan the incremental cost equals from-scratch A* on the same map from the same cell. This is the
    /// fixture where `k_m` is large (the robot has moved far from where the queue's keys were computed)
    /// and where the re-sort branch does real work: it is the only test that moves when that branch is
    /// dropped (the costs still agree, since a stale key is a lower bound and expanding it early is
    /// merely extra work — Figure 3 of the paper has no such branch — but the walk then expands 90 vertices
    /// instead of the measured 87, which the last assertion pins).
    #[test]
    fn scripted_walk_matches_astar_at_every_replan() {
        let w = 12;
        let mut blocked = vec![false; w * w];
        let (s0, t) = ((0, 0), (11, 11));
        let mut p = DStarLite::new(w, w, Connectivity::Eight, blocked.clone(), s0, t).unwrap();
        let oracle = |blocked: &Vec<bool>, s: (i32, i32)| {
            let free = |i: i32, j: i32| i >= 0 && j >= 0 && (i as usize) < w && (j as usize) < w && !blocked[(j as usize) * w + i as usize];
            astar_grid_conn(w, w, Connectivity::Eight, free, s, t).map(|q| path_length(&q))
        };
        // each batch: (cells to flip to blocked, cells to flip to free), applied after moving 3 steps
        let batches: [(&[(i32, i32)], &[(i32, i32)]); 4] = [
            (&[(4, 4), (5, 4), (6, 4), (4, 5), (4, 6)], &[]),
            (&[(7, 7), (7, 8), (7, 9), (8, 7), (9, 7)], &[]),
            (&[(10, 10), (9, 10), (10, 9), (9, 11)], &[(4, 4)]),
            (&[(6, 9), (6, 10), (6, 11)], &[(7, 8)]),
        ];
        let (mut cost, mut path) = p.plan().unwrap();
        assert!((cost - oracle(&blocked, s0).unwrap()).abs() < 1e-9);
        let mut costs = vec![cost];
        for (blk, fre) in batches {
            let next = path[3.min(path.len() - 1)];
            p.set_start(next);
            for &(i, j) in blk {
                blocked[(j as usize) * w + i as usize] = true;
                p.update_cell(i, j, true);
            }
            for &(i, j) in fre {
                blocked[(j as usize) * w + i as usize] = false;
                p.update_cell(i, j, false);
            }
            let ref_cost = oracle(&blocked, next).expect("the scripted walls never seal the goal (A* on the same map)");
            let (c, q) = p.plan().expect("D* Lite must agree the goal is reachable");
            assert!((c - ref_cost).abs() < 1e-9, "from {next:?}: D* Lite {c} vs A* {ref_cost}");
            assert_path_consistent(&p, c, &q);
            cost = c;
            path = q;
            costs.push(cost);
        }
        assert!(p.km() > 5.0, "non-vacuous: the robot moved far enough for k_m to matter, k_m = {}", p.km());
        // measured: 87 expansions over the five plans with the re-sort branch, 90 without it (a vertex
        // whose key predates a k_m increase is then expanded instead of re-sorted); costs agree either way
        assert!(p.expansions() <= 87, "expansions over the walk grew to {}", p.expansions());
        assert!(costs.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-9), "non-vacuous: the changes altered the cost, {costs:?}");
    }

    /// Construction refuses a map of the wrong size and an out-of-bounds start or goal; a blocked start
    /// plans to `None` rather than through itself; start equal to goal costs 0.
    #[test]
    fn construction_and_degenerate_queries() {
        assert!(DStarLite::new(3, 3, Connectivity::Four, vec![false; 8], (0, 0), (2, 2)).is_none(), "wrong map size");
        assert!(DStarLite::new(3, 3, Connectivity::Four, vec![false; 9], (3, 0), (2, 2)).is_none(), "start off the map");
        assert!(DStarLite::new(3, 3, Connectivity::Four, vec![false; 9], (0, 0), (0, -1)).is_none(), "goal off the map");
        let mut blocked = vec![false; 9];
        blocked[0] = true;
        let mut p = DStarLite::new(3, 3, Connectivity::Four, blocked, (0, 0), (2, 2)).unwrap();
        assert!(p.plan().is_none(), "a blocked start has no outgoing edge");
        let mut p = DStarLite::new(3, 3, Connectivity::Eight, vec![false; 9], (1, 1), (1, 1)).unwrap();
        assert_eq!(p.plan(), Some((0.0, vec![(1, 1)])));
    }


    /// Trial 268 of the differential test, by hand. 5x9 eight-connected, start (0,0), goal (4,8); the robot
    /// walks to (0,1) as `(0,8)` is blocked, then to (0,4) as the batch `(3,8)` blocked, `(1,0)` freed,
    /// `(4,4)` blocked arrives. Under an exactly compared loop test the search stopped with `(2,7)` still
    /// holding the stale `g = 1 + √2` it had through `(3,8)`: its key `k1 = (1 + √2) + octile((0,4),(2,7))`
    /// and the start's `rhs = 1 + (√2 + (√2 + (√2 + 1)))` are the same real `2 + 3√2`, and only
    /// `k2 = 1 + √2 < 2 + 3√2` orders `(2,7)` first — but the left sum rounds one ulp above the right, which
    /// the test asserts on those two expressions before the planner runs. Measured before the fix: D* Lite
    /// returned `2 + 3√2 = 6.2426` for a path its own map forbids. Replanned cost by hand:
    /// (0,4)→(0,5)→(1,6)→(2,6)→(3,6)→(4,7)→(4,8), four orthogonal and two diagonal steps, `4 + 2√2 = 6.8284`,
    /// equal to A*. (Which vertex holds the tie mid-search depends on the expansion order, so the test pins
    /// the rounding fact and the final answer, not the queue's intermediate state.)
    #[test]
    fn a_stale_g_whose_key_ties_the_start_on_k1_is_still_expanded() {
        let lhs = (1.0 + SQRT2) + Connectivity::Eight.heuristic((0, 4), (2, 7));
        let rhs = 1.0 + (SQRT2 + (SQRT2 + (SQRT2 + 1.0)));
        assert!(lhs > rhs && lhs - rhs < 1e-14, "non-vacuous: the same real 2 + 3√2 rounds apart, {lhs} vs {rhs}");
        assert_eq!(key_cmp((OrdF(lhs), OrdF(1.0 + SQRT2)), (OrdF(rhs), OrdF(rhs))), Ordering::Less, "k2 must break the tie");
        let rows = [".....", ".....", ".....", "..#..", ".....", "....#", "..##.", ".#..#", ".##.."];
        let g0 = grid(&rows);
        let goal = (4, 8);
        let mut p = DStarLite::from_grid(&g0, Connectivity::Eight, (0, 0), goal).unwrap();
        p.plan().unwrap();
        p.set_start((0, 1));
        p.update_cell(0, 8, true);
        p.plan().unwrap();
        p.set_start((0, 4));
        p.update_cell(3, 8, true);
        p.update_cell(1, 0, false);
        p.update_cell(4, 4, true);
        let (cost, path) = p.plan().unwrap();
        let mut g1 = [false; 45];
        for (k, b) in g1.iter_mut().enumerate() { *b = p.is_blocked((k % 5) as i32, (k / 5) as i32); }
        let free = |i: i32, j: i32| i >= 0 && j >= 0 && i < 5 && j < 9 && !g1[(j * 5 + i) as usize];
        let oracle = path_length(&astar_grid_conn(5, 9, Connectivity::Eight, free, (0, 4), goal).unwrap());
        assert!((cost - (4.0 + 2.0 * SQRT2)).abs() < 1e-9, "cost {cost}");
        assert!((cost - oracle).abs() < 1e-9, "must equal A* on the changed grid, {oracle}");
        assert_path_consistent(&p, cost, &path);
    }

    /// Trial 450 of the differential test, by hand. 5x6 eight-connected, start (0,0), goal (4,5); the robot
    /// moves to (1,0) (`k_m = 1`) and `(4,2)` is reported blocked, so `(3,2)`'s `g = 2 + √2` goes stale
    /// (`rhs = 4`). Its key `k1 = (2 + √2) + 2√2 + 1 = 3 + 3√2` ties the start's `rhs + k_m = 3 + 3√2` and
    /// also the **stale** key the old start (0,0) still holds, `(3 + 3√2, 3 + 3√2)`. In `f64` `(3,2)`'s `k1`
    /// rounds one ulp above the other two, so an exactly ordered queue puts `(3,2)` (`k2 = 2 + √2`) below
    /// `(0,0)` (`k2 = 3 + 3√2`), and the loop test on that stale top stops the search before `(3,2)` is
    /// raised. The queue order itself must therefore tolerate the tie, not only the loop test. Replanned
    /// cost by hand: (1,0)→(2,1)→(3,2)→(3,3)→(4,3)→(4,4)→(4,5), two diagonal and four orthogonal steps,
    /// `4 + 2√2`, equal to A*.
    #[test]
    fn a_tied_k1_one_ulp_high_still_sorts_ahead_of_a_stale_entry_with_larger_k2() {
        let g0 = grid(&["..#..", "...#.", "#.#..", ".....", ".....", "....."]);
        let goal = (4, 5);
        let mut p = DStarLite::from_grid(&g0, Connectivity::Eight, (0, 0), goal).unwrap();
        let (c0, _) = p.plan().unwrap();
        assert!((c0 - (3.0 + 3.0 * SQRT2)).abs() < 1e-9, "first plan {c0}");
        p.set_start((1, 0));
        p.update_cell(4, 2, true);
        let (si, ui, oi) = (p.idx((1, 0)), p.idx((3, 2)), p.idx((0, 0)));
        assert!(p.g[ui] < p.rhs[ui], "non-vacuous: (3,2) holds a stale low g");
        let (ks, ku, ko) = (p.calculate_key(si), p.key_of[ui].expect("(3,2) queued"), p.key_of[oi].expect("(0,0) still queued with its old key"));
        assert!(ko.0 == ks.0 && ku.0 > ks.0 && (ku.0 .0 - ks.0 .0).abs() < 1e-12, "non-vacuous: (0,0) ties the start exactly, (3,2) one ulp above: {ko:?} {ku:?} {ks:?}");
        assert!((ku, ui) > (ko, oi), "exact tuple order would put (3,2) below the stale (0,0) entry");
        assert!(Entry { key: ku, cell: ui } < Entry { key: ko, cell: oi }, "the tolerant order puts (3,2) first");
        let (cost, path) = p.plan().unwrap();
        let mut g1 = [false; 30];
        for (k, b) in g1.iter_mut().enumerate() { *b = p.is_blocked((k % 5) as i32, (k / 5) as i32); }
        let free = |i: i32, j: i32| i >= 0 && j >= 0 && i < 5 && j < 6 && !g1[(j * 5 + i) as usize];
        let oracle = path_length(&astar_grid_conn(5, 6, Connectivity::Eight, free, (1, 0), goal).unwrap());
        assert!((cost - (4.0 + 2.0 * SQRT2)).abs() < 1e-9, "cost {cost}");
        assert!((cost - oracle).abs() < 1e-9, "must equal A*, {oracle}");
        assert_path_consistent(&p, cost, &path);
    }

    /// Differential test against the oracle: 4000 random grids (5–9 on a side, 25% blocked, both
    /// connectivities), each walked for up to 8 replans; before every replan the robot moves 1–3 cells
    /// along its own path and 1–4 random cells flip. At every replan D* Lite and from-scratch A* on the same
    /// map must agree on the cost (or on unreachability), and the returned path must be walkable with the
    /// reported length. This is the test that found the floating-point tie (see [`key_cmp`]) at trials
    /// 268 and 450 of this sequence; the two regression tests above are those trials by hand.
    #[test]
    fn replanning_agrees_with_astar_on_random_grids_and_flips() {
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        let (mut replans, mut cost_changes, mut unreachable, mut eight) = (0usize, 0usize, 0usize, 0usize);
        for trial in 0..4000u32 {
            let w = 5 + (next() % 5) as usize;
            let h = 5 + (next() % 5) as usize;
            let conn = if next() % 2 == 0 { Connectivity::Four } else { Connectivity::Eight };
            let mut blocked: Vec<bool> = (0..w * h).map(|_| next() % 100 < 25).collect();
            let start = (0i32, 0i32);
            let goal = (w as i32 - 1, h as i32 - 1);
            blocked[0] = false;
            blocked[w * h - 1] = false;
            let rows0: Vec<String> = (0..h).rev().map(|j| (0..w).map(|i| if blocked[j * w + i] { '#' } else { '.' }).collect()).collect();
            let mut p = DStarLite::new(w, h, conn, blocked.clone(), start, goal).unwrap();
            let mut log: Vec<String> = vec![];
            let mut cur = start;
            let mut prev_cost: Option<f64> = None;
            eight += (conn == Connectivity::Eight) as usize;
            for _step in 0..8 {
                let free = |i: i32, j: i32| i >= 0 && j >= 0 && (i as usize) < w && (j as usize) < h && !blocked[(j as usize) * w + i as usize];
                let oracle = astar_grid_conn(w, h, conn, free, cur, goal).map(|q| path_length(&q));
                let res = p.plan();
                replans += 1;
                let same = match (&res, oracle) { (None, None) => true, (Some((c, _)), Some(o)) => (c - o).abs() < 1e-9, _ => false };
                if !same {
                    panic!("DIVERGENCE trial {trial} conn {conn:?} w {w} h {h} from {cur:?}: dstar {:?} vs astar {oracle:?}\nrows (top first, j=h-1):\n{}\nlog: {}", res.as_ref().map(|r| r.0), rows0.join("\n"), log.join("; "));
                }
                let Some((cost, path)) = res else { unreachable += 1; break };
                assert_path_consistent(&p, cost, &path);
                // a flip changed the cost-to-goal from the SAME cell only if the robot did not move; count
                // replans whose cost differs from the previous one by other than the distance walked
                if let Some(pc) = prev_cost
                    && (pc - cost).abs() > 1e-9
                {
                    cost_changes += 1;
                }
                prev_cost = Some(cost);
                let k = (1 + next() % 3) as usize;
                let k = k.min(path.len() - 1);
                cur = path[k];
                p.set_start(cur);
                log.push(format!("move {cur:?}"));
                for _ in 0..(1 + next() % 4) {
                    let i = (next() % w as u64) as i32;
                    let j = (next() % h as u64) as i32;
                    if (i, j) == cur || (i, j) == goal { continue; }
                    let ix = (j as usize) * w + i as usize;
                    blocked[ix] = !blocked[ix];
                    p.update_cell(i, j, blocked[ix]);
                    log.push(format!("flip ({i},{j}) -> {}", if blocked[ix] { "blocked" } else { "free" }));
                }
            }
        }
        assert!(replans > 4000 && cost_changes > 1000 && unreachable > 100 && eight > 1000 && eight < 3000, "non-vacuous: {replans} replans, {cost_changes} cost changes, {unreachable} unreachable, {eight} eight-connected trials");
    }
}
