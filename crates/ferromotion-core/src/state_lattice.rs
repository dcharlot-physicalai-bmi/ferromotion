//! **State-lattice planner** — A\* over a graph whose nodes are `(cell i, cell j, heading index)` and whose
//! edges are **motion primitives** that start on one node and end **exactly** on another. The
//! construction follows Pivtoraiko & Kelly, *Efficient Constrained Path Planning via Search in State
//! Lattices*, i-SAIRAS 2005, and Pivtoraiko, Knepper & Kelly, CMU-RI-TR-07-15 (2007; the journal form is
//! J. Field Robotics 26(3):308–333, 2009). Where [`crate::hybrid_astar`] carries a continuous pose and
//! rounds it into a closed set, a lattice primitive's endpoint is *defined* by integers `(di, dj, h_to)`,
//! so the closed set is exact, the same node is re-entered by many primitives, and the graph is a regular
//! one that an incremental search can repair.
//!
//! # Conventions
//!
//! * Node `(i, j, h)` sits at the **centre** of cell `(i, j)`: world `(origin_x + (i + ½)·dl,
//!   origin_y + (j + ½)·dl)`, heading `headings[h]`. Cell `(i, j)` covers `[i·dl, (i+1)·dl) × [j·dl,
//!   (j+1)·dl)` relative to the origin, which is [`OccupancyGrid::world_to_cell`]'s floor convention.
//! * Primitives are **geometric** (speed-free), in the arc-length form of the unicycle (LaValle,
//!   *Planning Algorithms*, §13.1.2.3): `dx/ds = dir·cos θ`, `dy/ds = dir·sin θ`, `dθ/ds = dir·κ(s)`, with
//!   `dir = ±1` for forward/reverse. The same primitive drives a differential drive or a car, since only
//!   the realisation `(v, ω)` / `(u_l, u_r)` / `φ = atan(L·κ)` changes.
//! * A constant-curvature primitive ends on a node exactly when its radius is `m·dl` and its turn a
//!   multiple of 90°; [`Lattice::unicycle_primitive`] checks the closed-form endpoint against the nearest
//!   node to `1e-9·dl` and refuses anything that misses, then snaps the last sample to the node so no
//!   error can accumulate along a path (TR-07-15 §4.3's snapping rule, with tolerance zero because these
//!   endpoints are exact in closed form).
//! * Each primitive carries its **swath**: the sorted, unique cell offsets a point robot covers along the
//!   samples (TR-07-15 §2.3), sampled every `dl/10` (the point-robot spacing the specification prescribes
//!   against tunnelling). An edge is free iff no swath cell is [`OccupancyGrid::blocked`], which also
//!   makes leaving the map a collision.
//! * The heuristic is the Euclidean distance between node centres, admissible because every primitive is
//!   at least as long as its chord and consistent (TR-07-15 §5.4).
//!
//! # Verified
//!
//! Against hand computation from the specification: the four-heading control set's endpoints
//! (including the reverse arcs' heading-change sign), the point-robot swaths `S → {(0,0),(1,0)}`,
//! `L → {(0,0),(1,0),(1,1)}`, `R → {(0,0),(1,0),(1,−1)}`, the open-grid optimum `3π/2` (`L,R,L`), the
//! corridor forcing `S,S,L,S,S` at `4 + π/2`, the no-fit corridor returning `None`, and the reverse-only
//! corridor at `2·w_rev`. Against [`crate::astar_grid_conn`] under [`crate::Connectivity::Four`]: a
//! lattice of unit straights plus zero-cost point turns reproduces the 4-connected grid distance on an
//! open grid and around a wall. Every returned sample lies in a free cell. [`DStarLite`] repairs a
//! plan on the same lattice when cells change (Koenig & Likhachev, AAAI-02, Figure 3): its first search
//! returns A\*'s cost, and each repair is checked against a fresh [`lattice_astar`]. Pure Rust →
//! WASM-clean.

use crate::grid_astar::OrdF;
use crate::OccupancyGrid;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// A lattice node `(i, j, h)`: cell column, cell row, heading index into [`Lattice::headings`].
pub type LatticeNode = (i32, i32, usize);

/// The shape of a primitive's curvature profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimitiveKind {
    /// Zero curvature.
    Straight,
    /// Constant curvature `kappa` (signed: positive turns left).
    Arc {
        /// Curvature in 1/length units.
        kappa: f64,
    },
    /// Rotation in place; zero displacement.
    PointTurn,
}

/// One motion primitive of the control set, expressed relative to its start node.
#[derive(Clone, Debug)]
pub struct Primitive {
    /// Heading index at the start node.
    pub h_from: usize,
    /// Heading index at the end node.
    pub h_to: usize,
    /// Integer cell offset of the end node from the start node.
    pub di: i32,
    /// Integer cell offset of the end node from the start node.
    pub dj: i32,
    /// Unsigned arc length (zero for a point turn).
    pub length: f64,
    /// `+1` forward, `−1` reverse (`+1` for a point turn).
    pub dir: i8,
    /// Curvature profile.
    pub kind: PrimitiveKind,
    /// Signed heading change over the primitive.
    pub dtheta: f64,
    /// Poses `(x, y, θ)` relative to the start node's centre, `θ` wrapped to `[0, 2π)`. The first is
    /// `(0, 0, headings[h_from])` and the last is exactly `(di·dl, dj·dl, headings[h_to])`.
    pub samples: Vec<(f64, f64, f64)>,
    /// Sorted, unique cell offsets relative to the start cell that the samples pass through.
    pub swath: Vec<(i32, i32)>,
}

/// Cost weights applied at plan time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatticeWeights {
    /// Multiplier on the length of a reverse primitive (`1` = no penalty).
    pub w_rev: f64,
    /// Cost per radian of a point turn.
    pub w_turn: f64,
}

impl Default for LatticeWeights {
    /// Unit weights: cost is path length, and a point turn costs its angle in radians.
    fn default() -> Self {
        LatticeWeights { w_rev: 1.0, w_turn: 1.0 }
    }
}

impl Primitive {
    /// Edge cost under `w`: `length` (forward), `length·w_rev` (reverse), or `w_turn·|Δθ|` (point turn).
    pub fn cost(&self, w: &LatticeWeights) -> f64 {
        match self.kind {
            PrimitiveKind::PointTurn => w.w_turn * self.dtheta.abs(),
            _ if self.dir < 0 => self.length * w.w_rev,
            _ => self.length,
        }
    }
}

/// Wrap an angle to `(−π, π]`.
fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

/// Closed-form endpoint of a constant-curvature motion: from pose `(0, 0, theta0)`, along signed arc
/// length `s_signed` (negative = reverse) at curvature `kappa`, returns `(x, y, θ)`.
///
/// Integrating `dθ/ds = κ` gives `Δθ = κ·s`; in the start frame `x' = sin Δθ / κ`, `y' = (1 − cos Δθ) / κ`,
/// rotated by `theta0`. For `|κ| < 1e-12` the straight-line limit is used. Verified against the values
/// the specification states for the four-heading set: a forward quarter turn of radius 1 from heading 0
/// ends at `(1, ±1, ±π/2)` and its reverse copy at `(−1, +1, −π/2)` (left) / `(−1, −1, +π/2)` (right).
pub fn arc_endpoint(theta0: f64, kappa: f64, s_signed: f64) -> (f64, f64, f64) {
    if kappa.abs() < 1e-12 {
        return (s_signed * theta0.cos(), s_signed * theta0.sin(), theta0);
    }
    let dth = kappa * s_signed;
    let dx = dth.sin() / kappa;
    let dy = (1.0 - dth.cos()) / kappa;
    let (c, s) = (theta0.cos(), theta0.sin());
    (dx * c - dy * s, dx * s + dy * c, theta0 + dth)
}

/// The cells a point robot occupies along `samples` (poses relative to a node centre): the sample is
/// shifted by `+½dl` into the start cell's corner frame and floored — `floor`, not `as i32`, so a
/// negative offset lands in the cell below rather than being truncated toward zero.
fn point_swath(samples: &[(f64, f64, f64)], dl: f64) -> Vec<(i32, i32)> {
    let mut cells = BTreeSet::new();
    for &(x, y, _) in samples {
        cells.insert((((x + 0.5 * dl) / dl).floor() as i32, ((y + 0.5 * dl) / dl).floor() as i32));
    }
    cells.into_iter().collect()
}

/// A heading table plus a control set of primitives, with per-heading successor and predecessor lists.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Cell size. Must equal the [`OccupancyGrid::resolution`] it is planned on.
    pub dl: f64,
    /// Headings in radians, wrapped to `[0, 2π)`; nodes store the index, never the float.
    pub headings: Vec<f64>,
    /// The control set.
    pub prims: Vec<Primitive>,
    /// `out[h]`: ids of primitives with `h_from == h`.
    pub out: Vec<Vec<usize>>,
    /// `inp[h]`: ids of primitives with `h_to == h`.
    pub inp: Vec<Vec<usize>>,
}

impl Lattice {
    /// An empty lattice over `headings` (wrapped to `[0, 2π)`) with cell size `dl`.
    pub fn new(dl: f64, headings: Vec<f64>) -> Lattice {
        let n = headings.len();
        Lattice { dl, headings: headings.into_iter().map(|h| h.rem_euclid(TAU)).collect(), prims: Vec::new(), out: vec![Vec::new(); n], inp: vec![Vec::new(); n] }
    }

    /// Add a primitive to the control set and return its id. Panics if a heading index is out of range,
    /// since a primitive that names a heading the table does not have cannot be an edge of this graph.
    pub fn push(&mut self, p: Primitive) -> usize {
        assert!(p.h_from < self.headings.len() && p.h_to < self.headings.len(), "primitive headings ({}, {}) exceed the {}-entry table", p.h_from, p.h_to, self.headings.len());
        let id = self.prims.len();
        self.out[p.h_from].push(id);
        self.inp[p.h_to].push(id);
        self.prims.push(p);
        id
    }

    /// A constant-curvature unicycle primitive from heading `h_from`, or `None` if it does not end
    /// exactly on a lattice node.
    ///
    /// The endpoint comes from [`arc_endpoint`] with `s = dir·length`; it must lie within `1e-9·dl` of
    /// `(di·dl, dj·dl)` for integer `(di, dj)` and within `1e-9` rad of a table heading, else `None`.
    /// The samples are spaced at most `dl/10` apart along the arc, the first is forced to
    /// `(0, 0, headings[h_from])`, the last is snapped to the exact node, and the swath is the point-robot
    /// cell set along them. Verified: every primitive of [`Lattice::four_heading`] passes; a quarter arc
    /// of radius `1.5·dl`, a 45° arc, and a straight of `1.5·dl` are refused.
    pub fn unicycle_primitive(&self, h_from: usize, kappa: f64, dir: i8, length: f64) -> Option<Primitive> {
        let theta0 = *self.headings.get(h_from)?;
        if !length.is_finite() || length <= 0.0 || !kappa.is_finite() || dir == 0 || !self.dl.is_finite() || self.dl <= 0.0 {
            return None;
        }
        let dir = dir.signum();
        let s_end = f64::from(dir) * length;
        let (xe, ye, te) = arc_endpoint(theta0, kappa, s_end);
        let tol = 1e-9 * self.dl;
        let (di, dj) = ((xe / self.dl).round(), (ye / self.dl).round());
        if (xe - di * self.dl).abs() > tol || (ye - dj * self.dl).abs() > tol {
            return None;
        }
        let h_to = self.headings.iter().position(|&h| wrap(te - h).abs() <= 1e-9)?;
        let n = (length / (self.dl / 10.0)).ceil().max(1.0) as usize;
        let mut samples: Vec<(f64, f64, f64)> = (0..=n)
            .map(|k| {
                let (x, y, t) = arc_endpoint(theta0, kappa, s_end * k as f64 / n as f64);
                (x, y, t.rem_euclid(TAU))
            })
            .collect();
        samples[0] = (0.0, 0.0, theta0);
        *samples.last_mut().expect("n >= 1 samples") = (di * self.dl, dj * self.dl, self.headings[h_to]);
        let swath = point_swath(&samples, self.dl);
        let kind = if kappa.abs() < 1e-12 { PrimitiveKind::Straight } else { PrimitiveKind::Arc { kappa } };
        Some(Primitive { h_from, h_to, di: di as i32, dj: dj as i32, length, dir, kind, dtheta: kappa * s_end, samples, swath })
    }

    /// A rotation in place from heading `h_from` to `h_to` (shortest signed angle), swath `{(0, 0)}`.
    /// `None` if either index is out of range or they are equal.
    pub fn point_turn(&self, h_from: usize, h_to: usize) -> Option<Primitive> {
        if h_from == h_to {
            return None;
        }
        let (a, b) = (*self.headings.get(h_from)?, *self.headings.get(h_to)?);
        let dtheta = wrap(b - a);
        Some(Primitive { h_from, h_to, di: 0, dj: 0, length: 0.0, dir: 1, kind: PrimitiveKind::PointTurn, dtheta, samples: vec![(0.0, 0.0, a), (0.0, 0.0, b)], swath: vec![(0, 0)] })
    }

    /// The exact four-heading control set (TR-07-15 Fig. 3's "carefully chosen length" Reeds–Shepp
    /// set): headings `{0, π/2, π, 3π/2}`; per heading a straight of one cell (`S`, length `dl`) and left
    /// and right quarter arcs of radius `m·dl` (`L`, `R`, length `π·m·dl/2`, ending `(m, ±m)` cells away
    /// with the heading turned by `±90°`); with `reverse`, the same three driven backwards; with
    /// `point_turns`, `±90°` rotations in place. `None` for `m == 0` or a non-positive `dl`.
    ///
    /// Out-degree per heading: 3, 6 with reverse, plus 2 with point turns.
    pub fn four_heading(dl: f64, m: u32, reverse: bool, point_turns: bool) -> Option<Lattice> {
        if m == 0 || !dl.is_finite() || dl <= 0.0 {
            return None;
        }
        let mut lat = Lattice::new(dl, vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
        let r = f64::from(m) * dl;
        let quarter = FRAC_PI_2 * r;
        let dirs: &[i8] = if reverse { &[1, -1] } else { &[1] };
        for h in 0..4 {
            for &(kappa, len) in &[(0.0, dl), (1.0 / r, quarter), (-1.0 / r, quarter)] {
                for &dir in dirs {
                    let p = lat.unicycle_primitive(h, kappa, dir, len)?;
                    lat.push(p);
                }
            }
            if point_turns {
                for q in [1, 3] {
                    let p = lat.point_turn(h, (h + q) % 4)?;
                    lat.push(p);
                }
            }
        }
        Some(lat)
    }
}

/// Whether primitive `prim` driven from `node` is collision-free: no swath cell is
/// [`OccupancyGrid::blocked`] (which counts out-of-bounds as blocked).
pub fn edge_free(grid: &OccupancyGrid, node: LatticeNode, prim: &Primitive) -> bool {
    prim.swath.iter().all(|&(di, dj)| !grid.blocked(i64::from(node.0 + di), i64::from(node.1 + dj)))
}

/// A plan on the lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct LatticePath {
    /// Sum of the primitive costs under the weights the plan was made with.
    pub cost: f64,
    /// Nodes from start to goal inclusive; `nodes.len() == primitives.len() + 1`.
    pub nodes: Vec<LatticeNode>,
    /// Primitive ids into [`Lattice::prims`], one per edge.
    pub primitives: Vec<usize>,
    /// World-frame `(x, y, θ)` samples along the whole path, junction poses de-duplicated.
    pub samples: Vec<(f64, f64, f64)>,
}

/// World-frame pose of a node's centre.
fn node_pose(grid: &OccupancyGrid, lattice: &Lattice, n: LatticeNode) -> (f64, f64, f64) {
    (grid.origin_x + (f64::from(n.0) + 0.5) * lattice.dl, grid.origin_y + (f64::from(n.1) + 0.5) * lattice.dl, lattice.headings[n.2])
}

/// A\* over the lattice from `start` to `goal` (both integer nodes). Returns the cost-optimal primitive
/// sequence within the lattice graph, or `None` when no primitive sequence connects them without a swath
/// cell being blocked or off the map, when either endpoint is off the map or on a blocked cell, or when
/// `lattice.dl` differs from `grid.resolution` (the swath offsets assume they agree).
///
/// Heuristic: Euclidean distance between node centres, admissible (every primitive is at least its
/// chord) and consistent, so a node is closed once. Ties on `f` prefer the larger `g`. Duplicate heap
/// entries stand in for decrease-key and are skipped on pop when the node is already closed.
pub fn lattice_astar(grid: &OccupancyGrid, lattice: &Lattice, start: LatticeNode, goal: LatticeNode, weights: LatticeWeights) -> Option<LatticePath> {
    let (w, hgt, n_h) = (grid.width, grid.height, lattice.headings.len());
    if n_h == 0 || (lattice.dl - grid.resolution).abs() > 1e-12 * grid.resolution.abs() {
        return None;
    }
    let in_bounds = |n: LatticeNode| n.0 >= 0 && n.1 >= 0 && (n.0 as usize) < w && (n.1 as usize) < hgt && n.2 < n_h;
    if !in_bounds(start) || !in_bounds(goal) || grid.blocked(i64::from(start.0), i64::from(start.1)) || grid.blocked(i64::from(goal.0), i64::from(goal.1)) {
        return None;
    }
    if start == goal {
        return Some(LatticePath { cost: 0.0, nodes: vec![start], primitives: Vec::new(), samples: vec![node_pose(grid, lattice, start)] });
    }
    let idx = |n: LatticeNode| (n.1 as usize * w + n.0 as usize) * n_h + n.2;
    let node_of = |k: usize| -> LatticeNode {
        let c = k / n_h;
        ((c % w) as i32, (c / w) as i32, k % n_h)
    };
    let heuristic = |n: LatticeNode| lattice.dl * f64::from(goal.0 - n.0).hypot(f64::from(goal.1 - n.1));

    let n_states = w * hgt * n_h;
    let mut g = vec![f64::INFINITY; n_states];
    let mut parent: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); n_states]; // (node index, primitive id)
    let mut closed = vec![false; n_states];
    let (s_idx, goal_idx) = (idx(start), idx(goal));
    g[s_idx] = 0.0;
    let mut open: BinaryHeap<Reverse<(OrdF, OrdF, usize)>> = BinaryHeap::new();
    open.push(Reverse((OrdF(heuristic(start)), OrdF(0.0), s_idx)));

    while let Some(Reverse((_, _, u_idx))) = open.pop() {
        if closed[u_idx] {
            continue;
        }
        closed[u_idx] = true;
        if u_idx == goal_idx {
            break;
        }
        let u = node_of(u_idx);
        for &pid in &lattice.out[u.2] {
            let p = &lattice.prims[pid];
            let v = (u.0 + p.di, u.1 + p.dj, p.h_to);
            if !in_bounds(v) {
                continue;
            }
            let v_idx = idx(v);
            if closed[v_idx] || !edge_free(grid, u, p) {
                continue;
            }
            let ng = g[u_idx] + p.cost(&weights);
            if ng < g[v_idx] {
                g[v_idx] = ng;
                parent[v_idx] = (u_idx, pid);
                open.push(Reverse((OrdF(ng + heuristic(v)), OrdF(-ng), v_idx)));
            }
        }
    }
    if !closed[goal_idx] {
        return None;
    }
    // reconstruct: walk parents back to the start, then replay forward for nodes and samples
    let mut primitives = Vec::new();
    let mut cur = goal_idx;
    while cur != s_idx {
        let (prev, pid) = parent[cur];
        primitives.push(pid);
        cur = prev;
    }
    primitives.reverse();
    Some(assemble_path(grid, lattice, start, primitives, g[goal_idx]))
}

/// Replay a primitive sequence from `start` into nodes and world-frame samples. Shared by the A\* and
/// D\* Lite paths so the two cannot disagree on where a sample lands.
fn assemble_path(grid: &OccupancyGrid, lattice: &Lattice, start: LatticeNode, primitives: Vec<usize>, cost: f64) -> LatticePath {
    let mut nodes = vec![start];
    let mut samples = Vec::new();
    for (k, &pid) in primitives.iter().enumerate() {
        let u = *nodes.last().expect("start is present");
        let p = &lattice.prims[pid];
        let (cx, cy, _) = node_pose(grid, lattice, u);
        // the first sample of every primitive after the first is the previous primitive's snapped endpoint
        for &(sx, sy, th) in p.samples.iter().skip(usize::from(k > 0)) {
            samples.push((cx + sx, cy + sy, th));
        }
        nodes.push((u.0 + p.di, u.1 + p.dj, p.h_to));
    }
    LatticePath { cost, nodes, primitives, samples }
}

// ------------------------------------------------------------------------------------------------
// D* Lite on the same lattice
// ------------------------------------------------------------------------------------------------

/// Incremental replanning over the lattice: Koenig & Likhachev, *D\* Lite*, AAAI-02 pp. 476–483,
/// Figure 3, lines {01'}–{35'}, with the lattice's primitives as edges. The search runs **backward**
/// from the goal (`g` is cost-to-goal), so it uses [`Lattice::inp`] for predecessors, and a changed
/// cell is mapped to the edges whose swath covers it: every `(node, primitive)` with `node = cell −
/// offset` for each offset in the primitive's swath, over all headings.
///
/// The occupancy grid is passed to each call rather than stored, so the caller can replace or mutate
/// it between calls; the lattice, dimensions and goal are fixed at construction. Keys are compared
/// lexicographically as `(k1, k2)`; the priority queue is a heap with a per-node version counter for
/// lazy deletion, so `Remove(u)` is a version bump. The heuristic is the Euclidean distance between
/// node centres, which is consistent for arc-length costs as the paper requires.
///
/// Verified: the first [`DStarLite::compute_shortest_path`] gives the same cost as [`lattice_astar`]
/// on the open-grid fixture (`3π/2`, `L,R,L`); blocking cell `(1,2)` afterwards repairs the plan to
/// `S,S,L,S,S` at `4 + π/2` while expanding fewer than all 64 states; blocking `(2,0)` in a corridor
/// leaves `g(start) = ∞`; and after the robot advances and a cell on its remaining path is blocked
/// (so `k_m > 0`), the repaired cost equals a fresh `lattice_astar` from the new start.
#[derive(Clone, Debug)]
pub struct DStarLite<'a> {
    lattice: &'a Lattice,
    weights: LatticeWeights,
    width: usize,
    height: usize,
    g: Vec<f64>,
    rhs: Vec<f64>,
    /// `(k1, k2, version, node)`; an entry whose version is not the node's current one is stale.
    queue: BinaryHeap<Reverse<(OrdF, OrdF, u64, usize)>>,
    version: Vec<u64>,
    next_version: u64,
    km: f64,
    start: LatticeNode,
    last: LatticeNode,
    goal: LatticeNode,
    expansions: usize,
}

impl<'a> DStarLite<'a> {
    /// Set up for `grid`'s dimensions with the robot at `start` and a fixed `goal`, then run the first
    /// search (`initialize` + `ComputeShortestPath`, lines {21'}–{24'}). `None` if either node is off the
    /// map, the heading table is empty, or `lattice.dl` differs from `grid.resolution`.
    pub fn new(grid: &OccupancyGrid, lattice: &'a Lattice, start: LatticeNode, goal: LatticeNode, weights: LatticeWeights) -> Option<Self> {
        let n_h = lattice.headings.len();
        if n_h == 0 || (lattice.dl - grid.resolution).abs() > 1e-12 * grid.resolution.abs() {
            return None;
        }
        let n = grid.width * grid.height * n_h;
        let mut d = DStarLite { lattice, weights, width: grid.width, height: grid.height, g: vec![f64::INFINITY; n], rhs: vec![f64::INFINITY; n], queue: BinaryHeap::new(), version: vec![0; n], next_version: 1, km: 0.0, start, last: start, goal, expansions: 0 };
        if !d.in_bounds(start) || !d.in_bounds(goal) {
            return None;
        }
        let gi = d.idx(goal);
        d.rhs[gi] = 0.0;
        d.insert(gi);
        d.compute_shortest_path(grid);
        Some(d)
    }

    fn in_bounds(&self, n: LatticeNode) -> bool {
        n.0 >= 0 && n.1 >= 0 && (n.0 as usize) < self.width && (n.1 as usize) < self.height && n.2 < self.lattice.headings.len()
    }

    fn idx(&self, n: LatticeNode) -> usize {
        (n.1 as usize * self.width + n.0 as usize) * self.lattice.headings.len() + n.2
    }

    fn node_of(&self, k: usize) -> LatticeNode {
        let n_h = self.lattice.headings.len();
        let c = k / n_h;
        ((c % self.width) as i32, (c / self.width) as i32, k % n_h)
    }

    fn heuristic(&self, a: LatticeNode, b: LatticeNode) -> f64 {
        self.lattice.dl * f64::from(a.0 - b.0).hypot(f64::from(a.1 - b.1))
    }

    /// `CalculateKey(s)`, line {01'}: `(min(g, rhs) + h(start, s) + k_m, min(g, rhs))`.
    fn key(&self, k: usize) -> (OrdF, OrdF) {
        let m = self.g[k].min(self.rhs[k]);
        (OrdF(m + self.heuristic(self.start, self.node_of(k)) + self.km), OrdF(m))
    }

    fn insert(&mut self, k: usize) {
        let v = self.next_version;
        self.next_version += 1;
        self.version[k] = v;
        let key = self.key(k);
        self.queue.push(Reverse((key.0, key.1, v, k)));
    }

    /// Cost of the edge that drives primitive `pid` from `v`: `∞` if `v` is off the map or the swath
    /// is blocked, else the primitive's weighted cost.
    fn edge_cost(&self, grid: &OccupancyGrid, v: LatticeNode, pid: usize) -> f64 {
        let p = &self.lattice.prims[pid];
        if !self.in_bounds(v) || !edge_free(grid, v, p) { f64::INFINITY } else { p.cost(&self.weights) }
    }

    /// The successor of `u` with the least `c(u, s') + g(s')`, as `(cost, primitive id, s')`.
    fn best_successor(&self, grid: &OccupancyGrid, u: LatticeNode) -> Option<(f64, usize, LatticeNode)> {
        let mut best: Option<(f64, usize, LatticeNode)> = None;
        for &pid in &self.lattice.out[u.2] {
            let p = &self.lattice.prims[pid];
            let s = (u.0 + p.di, u.1 + p.dj, p.h_to);
            if !self.in_bounds(s) {
                continue;
            }
            let c = self.edge_cost(grid, u, pid) + self.g[self.idx(s)];
            if best.is_none_or(|b| c < b.0) {
                best = Some((c, pid, s));
            }
        }
        best
    }

    /// `UpdateVertex(u)`, lines {07'}–{09'}.
    fn update_vertex(&mut self, grid: &OccupancyGrid, k: usize) {
        let u = self.node_of(k);
        if u != self.goal {
            self.rhs[k] = self.best_successor(grid, u).map_or(f64::INFINITY, |b| b.0);
        }
        self.version[k] = 0; // Remove(u): any queued entry is now stale
        if self.g[k] != self.rhs[k] {
            self.insert(k);
        }
    }

    /// Drop stale heap entries and return the top key, or `None` for an empty queue.
    fn top_key(&mut self) -> Option<(OrdF, OrdF)> {
        while let Some(Reverse((k1, k2, v, k))) = self.queue.peek().copied() {
            if self.version[k] == v {
                return Some((k1, k2));
            }
            self.queue.pop();
        }
        None
    }

    /// `ComputeShortestPath`, lines {10'}–{20'}. Returns whether a path is known afterwards
    /// (`g(start) < ∞`).
    pub fn compute_shortest_path(&mut self, grid: &OccupancyGrid) -> bool {
        let si = self.idx(self.start);
        loop {
            let Some(k_old) = self.top_key() else { break };
            if !(k_old < self.key(si) || self.rhs[si] != self.g[si]) {
                break;
            }
            let Reverse((_, _, _, k)) = self.queue.pop().expect("top_key found a live entry");
            self.version[k] = 0;
            self.expansions += 1;
            let k_new = self.key(k);
            if k_old < k_new {
                self.insert(k); // {13'}–{14'}: the key went stale as k_m grew
                continue;
            }
            let u = self.node_of(k);
            let preds: Vec<usize> = self.lattice.inp[u.2]
                .iter()
                .map(|&pid| {
                    let p = &self.lattice.prims[pid];
                    (u.0 - p.di, u.1 - p.dj, p.h_from)
                })
                .filter(|&v| self.in_bounds(v))
                .map(|v| self.idx(v))
                .collect();
            if self.g[k] > self.rhs[k] {
                self.g[k] = self.rhs[k]; // {15'}–{17'}: overconsistent
                for v in preds {
                    self.update_vertex(grid, v);
                }
            } else {
                self.g[k] = f64::INFINITY; // {18'}–{20'}: underconsistent
                for v in preds.into_iter().chain([k]) {
                    self.update_vertex(grid, v);
                }
            }
        }
        self.g[si].is_finite()
    }

    /// Lines {28'}–{35'}: `cells` have changed in `grid` since the last call. Every edge whose swath
    /// covers one of them is re-evaluated at its tail, `k_m` absorbs the start's motion since the last
    /// change, and the path is repaired. Returns whether a path is known afterwards.
    pub fn cells_changed(&mut self, grid: &OccupancyGrid, cells: &[(i32, i32)]) -> bool {
        if cells.is_empty() {
            return self.g[self.idx(self.start)].is_finite();
        }
        self.km += self.heuristic(self.last, self.start);
        self.last = self.start;
        let mut tails = BTreeSet::new();
        for &(ci, cj) in cells {
            for p in &self.lattice.prims {
                for &(di, dj) in &p.swath {
                    let u = (ci - di, cj - dj, p.h_from);
                    if self.in_bounds(u) {
                        tails.insert(self.idx(u));
                    }
                }
            }
        }
        for u in tails {
            self.update_vertex(grid, u);
        }
        self.compute_shortest_path(grid)
    }

    /// Lines {25'}–{27'}: drive one whole primitive along the current plan. Returns the primitive id
    /// and the node the robot is now on, or `None` when no path is known or the robot is at the goal.
    pub fn advance(&mut self, grid: &OccupancyGrid) -> Option<(usize, LatticeNode)> {
        if self.start == self.goal || !self.g[self.idx(self.start)].is_finite() {
            return None;
        }
        let (c, pid, s) = self.best_successor(grid, self.start)?;
        if !c.is_finite() {
            return None;
        }
        self.start = s;
        Some((pid, s))
    }

    /// The robot's current node.
    pub fn start(&self) -> LatticeNode {
        self.start
    }

    /// `g(start)`: the cost-to-goal of the current plan, `∞` if none is known.
    pub fn cost_to_goal(&self) -> f64 {
        self.g[self.idx(self.start)]
    }

    /// The `k_m` offset accumulated so far (line {30'}).
    pub fn km(&self) -> f64 {
        self.km
    }

    /// Number of queue pops that did work, cumulative over every search.
    pub fn expansions(&self) -> usize {
        self.expansions
    }

    /// The current plan, traced by the greedy successor rule from the start without moving the robot,
    /// or `None` if no path is known. Its cost is `g(start)`.
    pub fn path(&self, grid: &OccupancyGrid) -> Option<LatticePath> {
        let cost = self.cost_to_goal();
        if !cost.is_finite() {
            return None;
        }
        let mut primitives = Vec::new();
        let mut cur = self.start;
        while cur != self.goal {
            let (c, pid, s) = self.best_successor(grid, cur)?;
            if !c.is_finite() || primitives.len() >= self.g.len() {
                return None;
            }
            primitives.push(pid);
            cur = s;
        }
        Some(assemble_path(grid, self.lattice, self.start, primitives, cost))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_astar::{astar_grid_conn, path_length, Connectivity};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The forward-only exact set of the specification's fixtures: `S, L, R`, `m = 1`, `dl = 1`.
    fn slr() -> Lattice {
        Lattice::four_heading(1.0, 1, false, false).expect("exact set")
    }

    /// Four unit straights and eight `±90°` point turns, no arcs: the lattice that IS the 4-connected grid
    /// once turns are free.
    fn straights_and_turns() -> Lattice {
        let mut lat = Lattice::new(1.0, vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
        for h in 0..4 {
            let s = lat.unicycle_primitive(h, 0.0, 1, 1.0).unwrap();
            lat.push(s);
            for q in [1, 3] {
                let t = lat.point_turn(h, (h + q) % 4).unwrap();
                lat.push(t);
            }
        }
        lat
    }

    /// Whether every world-frame sample of a plan lies in a free cell of `grid`.
    fn samples_in_free_cells(grid: &OccupancyGrid, path: &LatticePath) -> bool {
        path.samples.iter().all(|&(x, y, _)| {
            let (i, j) = grid.world_to_cell(x, y);
            !grid.blocked(i, j)
        })
    }

    /// The kinds of a plan's primitives as letters, so a fixture can name the sequence it forces.
    fn letters(lat: &Lattice, path: &LatticePath) -> String {
        path.primitives
            .iter()
            .map(|&pid| {
                let p = &lat.prims[pid];
                let c = match p.kind {
                    PrimitiveKind::Straight => 'S',
                    PrimitiveKind::Arc { kappa } if kappa > 0.0 => 'L',
                    PrimitiveKind::Arc { .. } => 'R',
                    PrimitiveKind::PointTurn => 'T',
                };
                if p.dir < 0 { c.to_ascii_lowercase() } else { c }
            })
            .collect()
    }

    /// **The closed-form endpoint, against the specification's hand values.** From `(0, 0, 0)` with
    /// radius 1: forward `L` ends at `(1, 1, +π/2)`, forward `R` at `(1, −1, −π/2)`; reverse `L`
    /// (`s = −π/2`, `κ = +1`) at `(−1, +1, −π/2)` and reverse `R` at `(−1, −1, +π/2)` — the sign the
    /// specification's fixture 4 pins. A straight of length 3 at heading π/2 ends at `(0, 3, π/2)`.
    #[test]
    fn arc_endpoint_matches_the_hand_values_including_reverse_signs() {
        let q = FRAC_PI_2;
        let cases = [
            ((0.0, 1.0, q), (1.0, 1.0, q)),
            ((0.0, -1.0, q), (1.0, -1.0, -q)),
            ((0.0, 1.0, -q), (-1.0, 1.0, -q)),
            ((0.0, -1.0, -q), (-1.0, -1.0, q)),
            ((q, 0.0, 3.0), (0.0, 3.0, q)),
            ((q, 0.5, PI), (-2.0, 2.0, PI)), // radius 2, quarter turn left from north
        ];
        // non-vacuous: the four quarter-turn cases land on four distinct corners
        let ends: BTreeSet<(i32, i32)> = cases[..4].iter().map(|c| ((c.1).0 as i32, (c.1).1 as i32)).collect();
        assert_eq!(ends.len(), 4);
        for ((th0, kappa, s), (x, y, th)) in cases {
            let (ex, ey, eth) = arc_endpoint(th0, kappa, s);
            assert!(close(ex, x, 1e-12) && close(ey, y, 1e-12) && close(wrap(eth - th), 0.0, 1e-12), "arc_endpoint({th0}, {kappa}, {s}) = ({ex}, {ey}, {eth}), want ({x}, {y}, {th})");
        }
    }

    /// **Point-robot swaths, by hand (specification §1).** With `m = 1`, `dl = 1`, from heading east:
    /// `S → {(0,0),(1,0)}`, `L → {(0,0),(1,0),(1,1)}`, `R → {(0,0),(1,0),(1,−1)}`. The left arc's centre
    /// is `(½, 3/2)` in the cell-corner frame; it crosses `x = 1` at `t = π/6` where `y = 0.634` (row 0)
    /// and `y = 1` at `t = π/3` where `x = 1.366` (column 1), so it never enters `(0, 1)`. The reverse
    /// copies mirror through the origin: `s → {(−1,0),(0,0)}`, `l → {(−1,1),(−1,0),(0,0)}`.
    #[test]
    fn point_robot_swaths_match_the_hand_computation() {
        let lat = Lattice::four_heading(1.0, 1, true, false).unwrap();
        let find = |kappa: f64, dir: i8| lat.prims.iter().find(|p| p.h_from == 0 && p.dir == dir && close(kappa, match p.kind { PrimitiveKind::Arc { kappa } => kappa, _ => 0.0 }, 1e-12)).expect("primitive exists");
        assert_eq!(find(0.0, 1).swath, vec![(0, 0), (1, 0)], "S");
        assert_eq!(find(1.0, 1).swath, vec![(0, 0), (1, 0), (1, 1)], "L");
        assert_eq!(find(-1.0, 1).swath, vec![(0, 0), (1, -1), (1, 0)], "R");
        assert_eq!(find(0.0, -1).swath, vec![(-1, 0), (0, 0)], "reverse S");
        assert_eq!(find(1.0, -1).swath, vec![(-1, 0), (-1, 1), (0, 0)], "reverse L");
        assert_eq!(find(-1.0, -1).swath, vec![(-1, -1), (-1, 0), (0, 0)], "reverse R");
        // the rotated copy from north: L goes to (−1, 1) via cells (0,0),(0,1),(−1,1)
        let l_north = lat.prims.iter().find(|p| p.h_from == 1 && p.dir == 1 && matches!(p.kind, PrimitiveKind::Arc { kappa } if kappa > 0.0)).unwrap();
        assert_eq!((l_north.di, l_north.dj, l_north.h_to), (-1, 1, 2));
        assert_eq!(l_north.swath, vec![(-1, 1), (0, 0), (0, 1)]);
    }

    /// **Every primitive of the four-heading set is exact.** The last sample equals `(di·dl, dj·dl,
    /// headings[h_to])` bitwise, the first is `(0, 0, headings[h_from])`, consecutive samples are at
    /// most `dl/10` apart in position, the swath contains both end cells, and `out`/`inp` index every
    /// primitive exactly once. Checked at `dl = 0.25`, `m = 2`, with reverse and point turns (out-degree
    /// 8 per heading, 32 primitives), so the tolerances scale with `dl` rather than assuming `dl = 1`.
    #[test]
    fn every_four_heading_primitive_ends_exactly_on_a_node() {
        let dl = 0.25;
        let lat = Lattice::four_heading(dl, 2, true, true).unwrap();
        assert_eq!(lat.prims.len(), 32);
        assert!(lat.out.iter().all(|o| o.len() == 8) && lat.inp.iter().all(|o| o.len() == 8));
        let mut seen = vec![0usize; 32];
        for o in lat.out.iter().chain(lat.inp.iter()) {
            for &id in o {
                seen[id] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 2), "each primitive appears once in out and once in inp");
        // non-vacuous: arcs reach (±2, ±2) cells, so the endpoints are not all trivially at the origin
        assert!(lat.prims.iter().any(|p| p.di.abs() == 2 && p.dj.abs() == 2));
        for p in &lat.prims {
            let last = *p.samples.last().unwrap();
            assert_eq!(last, (f64::from(p.di) * dl, f64::from(p.dj) * dl, lat.headings[p.h_to]), "snapped endpoint of {p:?}");
            assert_eq!(p.samples[0], (0.0, 0.0, lat.headings[p.h_from]));
            for w in p.samples.windows(2) {
                let d = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
                assert!(d <= dl / 10.0 + 1e-12, "sample spacing {d} exceeds dl/10 = {}", dl / 10.0);
            }
            assert!(p.swath.contains(&(0, 0)) && p.swath.contains(&(p.di, p.dj)), "swath {:?} must cover both end cells of {p:?}", p.swath);
            let expect_len = match p.kind {
                PrimitiveKind::Straight => dl,
                PrimitiveKind::Arc { .. } => FRAC_PI_2 * 2.0 * dl,
                PrimitiveKind::PointTurn => 0.0,
            };
            assert!(close(p.length, expect_len, 1e-15), "length of {p:?}");
        }
    }

    /// **The generator refuses non-lattice endpoints.** A quarter arc of radius `1.5·dl` ends at
    /// `(1.5, 1.5)`; a 45° arc of radius `dl` ends at `(1/√2, 1 − 1/√2)`; a straight of `1.5·dl` ends
    /// mid-cell; a quarter arc from a heading table without the turned heading has no `h_to`. All are
    /// `None`, and a radius-`dl` quarter arc is `Some` under the same call.
    #[test]
    fn the_generator_refuses_a_primitive_that_misses_the_lattice() {
        let lat = slr();
        assert!(lat.unicycle_primitive(0, 1.0, 1, FRAC_PI_2).is_some(), "the exact arc is accepted");
        assert!(lat.unicycle_primitive(0, 1.0 / 1.5, 1, FRAC_PI_2 * 1.5).is_none(), "radius 1.5 lands mid-cell");
        assert!(lat.unicycle_primitive(0, 1.0, 1, FRAC_PI_2 / 2.0).is_none(), "a 45 degree arc endpoint is irrational");
        assert!(lat.unicycle_primitive(0, 0.0, 1, 1.5).is_none(), "a 1.5-cell straight lands mid-cell");
        assert!(lat.unicycle_primitive(0, 0.0, 1, 0.0).is_none() && lat.unicycle_primitive(0, 0.0, 0, 1.0).is_none() && lat.unicycle_primitive(7, 0.0, 1, 1.0).is_none(), "degenerate inputs");
        let two = Lattice::new(1.0, vec![0.0, PI]);
        assert!(two.unicycle_primitive(0, 1.0, 1, FRAC_PI_2).is_none(), "no heading at pi/2 in the table");
        assert!(two.unicycle_primitive(0, 1.0, 1, PI).is_some(), "a half turn of radius 1 ends at (0, 2, pi), which the table has");
    }

    /// **Fixture 1: open grid, by hand.** From east `S` adds `(1,0)`, `L` adds `(1,1)` turning to north;
    /// from north `S` adds `(0,1)`, `R` adds `(1,1)` turning to east. East→north needs an odd number of
    /// turns `k`, `k + #S_east = 3`, `k + #S_north = 3`: `k = 1` gives `S,S,L,S,S = 4 + π/2 = 5.5708`,
    /// `k = 3` gives `L,R,L = 3π/2 = 4.71239`, `k ≥ 5` is impossible. The optimum is `3π/2`, its swath
    /// `(0,0),(1,0),(1,1),(1,2),(2,2),(3,2),(3,3)`.
    #[test]
    fn open_grid_optimum_is_l_r_l_at_three_half_pi() {
        let grid = OccupancyGrid::from_rows(&["....", "....", "....", "...."], 1.0, 0.0, 0.0).unwrap();
        let lat = slr();
        let path = lattice_astar(&grid, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).expect("reachable");
        // non-vacuous: the runner-up S,S,L,S,S is itself feasible on this grid and costs strictly more,
        // so the optimum is a choice between two real candidates
        let mut node = (0, 0, 0);
        let mut runner_up = 0.0;
        for want in "SSLSS".chars() {
            let pid = lat.out[node.2].iter().copied().find(|&pid| letters(&lat, &LatticePath { cost: 0.0, nodes: vec![], primitives: vec![pid], samples: vec![] }) == want.to_string()).unwrap();
            let p = &lat.prims[pid];
            assert!(edge_free(&grid, node, p));
            runner_up += p.cost(&LatticeWeights::default());
            node = (node.0 + p.di, node.1 + p.dj, p.h_to);
        }
        assert_eq!(node, (3, 3, 1));
        assert!(close(runner_up, 4.0 + FRAC_PI_2, 1e-12) && runner_up > path.cost, "runner-up {runner_up} vs optimum {}", path.cost);
        assert!(close(path.cost, 3.0 * FRAC_PI_2, 1e-9), "cost {}", path.cost);
        assert_eq!(letters(&lat, &path), "LRL");
        assert_eq!(path.nodes, vec![(0, 0, 0), (1, 1, 1), (2, 2, 0), (3, 3, 1)]);
        let swath: BTreeSet<(i64, i64)> = path.samples.iter().map(|&(x, y, _)| grid.world_to_cell(x, y)).collect();
        assert_eq!(swath.into_iter().collect::<Vec<_>>(), vec![(0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (3, 3)]);
        assert!(samples_in_free_cells(&grid, &path));
        // the cost is the sum of the chosen primitives' costs
        let sum: f64 = path.primitives.iter().map(|&p| lat.prims[p].cost(&LatticeWeights::default())).sum();
        assert!(close(sum, path.cost, 1e-12));
    }

    /// **Fixture 2: the corridor forces `S,S,L,S,S`, by hand.** Free cells are the bottom row and column
    /// 3. The `L` from `(2,0,E)` sweeps `(2,0),(3,0),(3,1)`, all free; an earlier `L` needs `(1,1)` or
    /// `(2,1)` (blocked) and any `R` needs row `−1` (off the map). Cost `1+1+π/2+1+1 = 5.570796`. On the
    /// open grid the same query costs `3π/2`, so the corridor changed the answer (non-vacuous), and it
    /// also pins the node-at-cell-centre convention: nodes on corners give the arc a different swath.
    #[test]
    fn a_corridor_forces_the_one_primitive_sequence_that_fits() {
        let grid = OccupancyGrid::from_rows(&["###.", "###.", "###.", "...."], 1.0, 0.0, 0.0).unwrap();
        let lat = slr();
        let open = OccupancyGrid::from_rows(&["....", "....", "....", "...."], 1.0, 0.0, 0.0).unwrap();
        let free = lattice_astar(&open, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).unwrap();
        let path = lattice_astar(&grid, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).expect("reachable");
        assert!(free.cost < path.cost, "the corridor must cost more than the open grid: {} vs {}", free.cost, path.cost);
        assert!(close(path.cost, 4.0 + FRAC_PI_2, 1e-9), "cost {}", path.cost);
        assert_eq!(letters(&lat, &path), "SSLSS");
        assert_eq!(path.nodes, vec![(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 1, 1), (3, 2, 1), (3, 3, 1)]);
        assert!(samples_in_free_cells(&grid, &path), "a sample entered a wall cell");
    }

    /// **Fixture 3: no primitive sequence fits, by hand.** A width-1 corridor, goal heading north. The
    /// only primitives ending at north are `L` from east and `R` from west, both needing a start node one
    /// row below the goal — row `−1`, off the map — so `(3,0,N)` has no predecessor and A\* exhausts the
    /// open list. The same query with goal heading east is reachable at cost 3 (non-vacuous), and adding
    /// `±90°` point turns at `w_turn = 2` makes the north goal reachable at `3 + 2·π/2`.
    #[test]
    fn a_goal_heading_no_primitive_can_produce_returns_none() {
        let grid = OccupancyGrid::from_rows(&["...."], 1.0, 0.0, 0.0).unwrap();
        let lat = slr();
        let east = lattice_astar(&grid, &lat, (0, 0, 0), (3, 0, 0), LatticeWeights::default()).expect("east goal reachable");
        assert!(close(east.cost, 3.0, 1e-12) && letters(&lat, &east) == "SSS");
        assert!(lattice_astar(&grid, &lat, (0, 0, 0), (3, 0, 1), LatticeWeights::default()).is_none(), "north goal has no predecessor");
        let turning = Lattice::four_heading(1.0, 1, false, true).unwrap();
        let w = LatticeWeights { w_rev: 1.0, w_turn: 2.0 };
        let path = lattice_astar(&grid, &turning, (0, 0, 0), (3, 0, 1), w).expect("point turns make it reachable");
        assert!(close(path.cost, 3.0 + 2.0 * FRAC_PI_2, 1e-12), "cost {}", path.cost);
        assert_eq!(letters(&turning, &path).chars().filter(|&c| c == 'T').count(), 1);
        assert_eq!(path.nodes.last(), Some(&(3, 0, 1)));
        assert!(samples_in_free_cells(&grid, &path));
    }

    /// **Fixture 4: reverse primitives and their weight, by hand.** A width-1 corridor of three cells,
    /// start `(2,0,E)`, goal `(0,0,E)`. Every arc's swath has a cell in row `±1` (off the map) and forward
    /// `S` moves `+x`, so the only sequence is `s,s` (reverse straights): cost `2·w_rev = 4` at
    /// `w_rev = 2`, and `2` at `w_rev = 1` (non-vacuous: the weight is applied).
    #[test]
    fn reverse_straights_carry_the_reverse_weight() {
        let grid = OccupancyGrid::from_rows(&["..."], 1.0, 0.0, 0.0).unwrap();
        let lat = Lattice::four_heading(1.0, 1, true, false).unwrap();
        let heavy = lattice_astar(&grid, &lat, (2, 0, 0), (0, 0, 0), LatticeWeights { w_rev: 2.0, w_turn: 1.0 }).expect("reachable in reverse");
        let unit = lattice_astar(&grid, &lat, (2, 0, 0), (0, 0, 0), LatticeWeights::default()).unwrap();
        assert!(close(heavy.cost, 4.0, 1e-12) && close(unit.cost, 2.0, 1e-12), "costs {} / {}", heavy.cost, unit.cost);
        assert_eq!(letters(&lat, &heavy), "ss");
        assert_eq!(heavy.nodes, vec![(2, 0, 0), (1, 0, 0), (0, 0, 0)]);
        assert!(samples_in_free_cells(&grid, &heavy));
        // without reverse primitives the same query has no solution
        assert!(lattice_astar(&grid, &slr(), (2, 0, 0), (0, 0, 0), LatticeWeights::default()).is_none());
    }

    /// **THE ORACLE: unit straights plus free point turns are the 4-connected grid.** With `w_turn = 0`
    /// a heading change costs nothing, so the lattice cost from `(0,0,E)` to any goal heading is the
    /// Manhattan path length [`astar_grid_conn`] returns under [`Connectivity::Four`], on an open `7×5`
    /// grid for every goal cell and around a wall with one gap. The straight's swath is its two end
    /// cells, which is exactly the 4-connected step's collision test. Non-vacuous: the costs vary with
    /// the goal, and the wall makes at least one goal cost more than its Manhattan distance. A
    /// straight-only lattice (no turns at all) is checked too: along its own row it matches the grid
    /// distance and it cannot leave the row.
    #[test]
    fn straights_with_free_turns_reproduce_four_connected_grid_astar() {
        let open = OccupancyGrid::from_rows(&[".......", ".......", ".......", ".......", "......."], 1.0, 0.0, 0.0).unwrap();
        let walled = OccupancyGrid::from_rows(&["...#...", "...#...", "...#...", ".......", "...#..."], 1.0, 0.0, 0.0).unwrap();
        let lat = straights_and_turns();
        let w = LatticeWeights { w_rev: 1.0, w_turn: 0.0 };
        for grid in [&open, &walled] {
            let free = |i: i32, j: i32| !grid.blocked(i64::from(i), i64::from(j));
            let mut costs = Vec::new();
            let mut detours = 0;
            for j in 0..5 {
                for i in 0..7 {
                    if !free(i, j) {
                        continue;
                    }
                    let want = astar_grid_conn(7, 5, Connectivity::Four, free, (0, 0), (i, j)).map(|p| path_length(&p));
                    for h in 0..4 {
                        let got = lattice_astar(grid, &lat, (0, 0, 0), (i, j, h), w).map(|p| {
                            assert!(samples_in_free_cells(grid, &p));
                            p.cost
                        });
                        match (want, got) {
                            (Some(a), Some(b)) => assert!(close(a, b, 1e-9), "goal ({i},{j},{h}): grid {a} vs lattice {b}"),
                            (None, None) => {}
                            _ => panic!("goal ({i},{j},{h}): grid {want:?} vs lattice {got:?}"),
                        }
                    }
                    if let Some(c) = want {
                        if c > f64::from(i + j) {
                            detours += 1;
                        }
                        costs.push(c as i64);
                    }
                }
            }
            costs.sort_unstable();
            costs.dedup();
            assert!(costs.len() > 5, "costs must vary across goals: {costs:?}");
            if std::ptr::eq(grid, &walled) {
                assert!(detours > 0, "the wall must lengthen at least one path");
            }
        }
        // straight-only: no turns, so only the start row is reachable, at the grid distance
        let mut straight = Lattice::new(1.0, vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
        for h in 0..4 {
            let p = straight.unicycle_primitive(h, 0.0, 1, 1.0).unwrap();
            straight.push(p);
        }
        assert!(lat.prims.len() == 12 && straight.prims.len() == 4, "the oracle lattices carry no arcs");
        let want = astar_grid_conn(7, 5, Connectivity::Four, |i, j| !open.blocked(i64::from(i), i64::from(j)), (1, 2), (6, 2)).map(|p| path_length(&p)).unwrap();
        let got = lattice_astar(&open, &straight, (1, 2, 0), (6, 2, 0), w).unwrap();
        assert!(close(want, 5.0, 1e-12) && close(got.cost, want, 1e-12), "straight-only along a row: {} vs {want}", got.cost);
        assert!(lattice_astar(&open, &straight, (1, 2, 0), (6, 3, 0), w).is_none(), "straight-only cannot change row");
    }

    /// **A trap for an inadmissible heuristic, by hand.** Start `(0,0)`, goal `(6,0)`, walls at
    /// `(1,0),(3,1),(5,0),(5,1)`; unit straights with free point turns, so this is the 4-connected grid.
    /// The optimum goes over row 2: `(0,0)→(0,1)→(2,1)→(2,2)→(6,2)→(6,0)`, 10 steps. The pocket
    /// `(2..4, 0)` lies nearer the goal but is sealed by `(5,0),(5,1)`, so a path through it must climb
    /// back to row 2: `(0,0)→(0,1)→(2,1)→(2,0)→(4,0)→(4,2)→(6,2)→(6,0)`, 12 steps. With `3×` Euclidean
    /// the search closes `(4,2)` through the pocket at `g = 8` before the row-2 route (at `(2,2)`,
    /// `g = 4`, `f = 4 + 3·√20 ≈ 17.4`) is expanded, and returns 12; with the Euclidean heuristic it
    /// returns 10, which is also what [`astar_grid_conn`] returns. Non-vacuous: both routes are walked
    /// through the lattice and are feasible.
    #[test]
    fn a_pocket_nearer_the_goal_does_not_lure_the_search_off_the_optimum() {
        let grid = OccupancyGrid::from_rows(&[".......", ".......", "...#.#.", ".#...#."], 1.0, 0.0, 0.0).unwrap();
        let lat = straights_and_turns();
        let w = LatticeWeights { w_rev: 1.0, w_turn: 0.0 };
        let walk = |cells: &[(i32, i32)]| -> f64 {
            // drive the cell path with straights and free turns, asserting each edge is free
            let mut cost = 0.0;
            let mut node = (cells[0].0, cells[0].1, 0usize);
            for pair in cells.windows(2) {
                let (di, dj) = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
                let h = match (di, dj) {
                    (1, 0) => 0,
                    (0, 1) => 1,
                    (-1, 0) => 2,
                    (0, -1) => 3,
                    _ => panic!("not a unit step"),
                };
                if node.2 != h {
                    let pid = lat.out[node.2].iter().copied().find(|&pid| lat.prims[pid].h_to == h && lat.prims[pid].kind == PrimitiveKind::PointTurn).unwrap_or_else(|| {
                        // a 180 degree turn is two quarter turns; both cost zero here
                        let mid = (node.2 + 1) % 4;
                        node.2 = mid;
                        lat.out[mid].iter().copied().find(|&pid| lat.prims[pid].h_to == h && lat.prims[pid].kind == PrimitiveKind::PointTurn).unwrap()
                    });
                    node.2 = lat.prims[pid].h_to;
                }
                let pid = lat.out[node.2].iter().copied().find(|&pid| lat.prims[pid].kind == PrimitiveKind::Straight).unwrap();
                assert!(edge_free(&grid, node, &lat.prims[pid]), "step {pair:?} is not free");
                cost += lat.prims[pid].cost(&w);
                node = (pair[1].0, pair[1].1, node.2);
            }
            cost
        };
        let over = walk(&[(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 1), (6, 0)]);
        let pocket = walk(&[(0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (5, 2), (6, 2), (6, 1), (6, 0)]);
        assert!(close(over, 10.0, 1e-12) && close(pocket, 12.0, 1e-12), "hand routes: {over} / {pocket}");
        let grid_cost = astar_grid_conn(7, 4, Connectivity::Four, |i, j| !grid.blocked(i64::from(i), i64::from(j)), (0, 0), (6, 0)).map(|p| path_length(&p)).unwrap();
        assert!(close(grid_cost, 10.0, 1e-12));
        for h in 0..4 {
            let path = lattice_astar(&grid, &lat, (0, 0, 0), (6, 0, h), w).expect("reachable");
            assert!(close(path.cost, 10.0, 1e-9), "goal heading {h}: cost {} (the pocket route costs 12)", path.cost);
            assert!(samples_in_free_cells(&grid, &path));
        }
    }

    /// **Samples are a continuous trajectory from the start pose to the goal pose.** Position steps are
    /// at most `dl/10`, heading steps (wrapped) at most `dl/10·κ_max = 0.1` rad plus the snap, the first
    /// sample is the start node's centre and the last the goal's. Checked at `dl = 0.5` with a non-zero
    /// origin so the world transform is exercised, on the `L,R,L` open-grid path.
    #[test]
    fn samples_form_a_continuous_trajectory_between_the_node_centres() {
        let grid = OccupancyGrid::from_rows(&["....", "....", "....", "...."], 0.5, -3.0, 2.0).unwrap();
        let lat = Lattice::four_heading(0.5, 1, false, false).unwrap();
        let path = lattice_astar(&grid, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).unwrap();
        assert!(close(path.cost, 0.5 * 3.0 * FRAC_PI_2, 1e-12), "cost scales with dl: {}", path.cost);
        assert!(path.samples.len() > 40, "three arcs at dl/10 spacing give more than 40 samples: {}", path.samples.len());
        assert_eq!(path.samples[0], (-3.0 + 0.25, 2.0 + 0.25, 0.0));
        assert_eq!(*path.samples.last().unwrap(), (-3.0 + 1.75, 2.0 + 1.75, FRAC_PI_2));
        for w in path.samples.windows(2) {
            let d = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
            let dth = wrap(w[1].2 - w[0].2).abs();
            assert!(d <= 0.05 + 1e-12 && dth <= 0.1 + 1e-9, "step ({d}, {dth}) between {:?} and {:?}", w[0], w[1]);
        }
        // a junction pose appears once: 3 primitives x 16 arc samples + 1
        assert_eq!(path.samples.len(), 3 * 16 + 1);
        assert!(samples_in_free_cells(&grid, &path));
    }

    /// Edge cases: start equals goal (cost 0, one sample), a blocked or off-map endpoint, a lattice whose
    /// cell size differs from the grid's, and an empty heading table all return as documented.
    #[test]
    fn degenerate_queries_return_as_documented() {
        let grid = OccupancyGrid::from_rows(&["..#", "...", "..."], 1.0, 0.0, 0.0).unwrap();
        let lat = slr();
        let same = lattice_astar(&grid, &lat, (1, 1, 2), (1, 1, 2), LatticeWeights::default()).unwrap();
        assert_eq!(same, LatticePath { cost: 0.0, nodes: vec![(1, 1, 2)], primitives: vec![], samples: vec![(1.5, 1.5, PI)] });
        assert!(lattice_astar(&grid, &lat, (0, 0, 0), (2, 2, 0), LatticeWeights::default()).is_none(), "goal cell is a wall");
        assert!(lattice_astar(&grid, &lat, (2, 2, 0), (0, 0, 0), LatticeWeights::default()).is_none(), "start cell is a wall");
        assert!(lattice_astar(&grid, &lat, (0, 0, 0), (3, 0, 0), LatticeWeights::default()).is_none(), "goal off the map");
        assert!(lattice_astar(&grid, &lat, (0, 0, 4), (1, 0, 0), LatticeWeights::default()).is_none(), "heading index off the table");
        assert!(lattice_astar(&grid, &Lattice::four_heading(0.5, 1, false, false).unwrap(), (0, 0, 0), (1, 0, 0), LatticeWeights::default()).is_none(), "dl must match the grid resolution");
        assert!(lattice_astar(&grid, &Lattice::new(1.0, vec![]), (0, 0, 0), (1, 0, 0), LatticeWeights::default()).is_none(), "no headings");
        assert!(Lattice::four_heading(1.0, 0, false, false).is_none() && Lattice::four_heading(0.0, 1, false, false).is_none());
        // and a valid query on the same grid still succeeds, so the refusals above are not a dead planner
        assert!(lattice_astar(&grid, &lat, (0, 0, 0), (1, 0, 0), LatticeWeights::default()).is_some());
    }

    /// **D\* Lite's first search equals A\*.** Koenig & Likhachev, Theorem 4 (via LPA\*): the first
    /// `ComputeShortestPath` expands the same vertices as A\*, so on the open-grid fixture `g(start)` must
    /// be bitwise `3π/2` and the greedy path `L,R,L`. Across the straights-and-turns oracle grids, every
    /// goal's cost matches [`lattice_astar`] (both finite and equal, or both absent). Non-vacuous: the
    /// costs differ between goals.
    #[test]
    fn dstar_lite_first_search_equals_astar() {
        let grid = OccupancyGrid::from_rows(&["....", "....", "....", "...."], 1.0, 0.0, 0.0).unwrap();
        let lat = slr();
        let d = DStarLite::new(&grid, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).unwrap();
        let a = lattice_astar(&grid, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).unwrap();
        assert_eq!(d.cost_to_goal(), a.cost, "bitwise: the same edge costs summed");
        assert_eq!(d.cost_to_goal(), 3.0 * FRAC_PI_2);
        let p = d.path(&grid).unwrap();
        assert_eq!(letters(&lat, &p), "LRL");
        assert_eq!((p.nodes, p.samples, p.cost), (a.nodes, a.samples, a.cost));
        assert!(d.expansions() > 0 && d.expansions() <= 64, "expansions {}", d.expansions());

        let walled = OccupancyGrid::from_rows(&["...#...", "...#...", "...#...", ".......", "...#..."], 1.0, 0.0, 0.0).unwrap();
        let turns = straights_and_turns();
        let w = LatticeWeights { w_rev: 1.0, w_turn: 0.0 };
        let mut costs = BTreeSet::new();
        for (i, j, h) in [(6, 4, 0), (6, 0, 2), (2, 0, 1), (0, 4, 3), (3, 1, 0), (5, 2, 1)] {
            let want = lattice_astar(&walled, &turns, (0, 0, 0), (i, j, h), w).map(|p| p.cost);
            let got = DStarLite::new(&walled, &turns, (0, 0, 0), (i, j, h), w).unwrap();
            match want {
                Some(c) => {
                    assert!(close(got.cost_to_goal(), c, 1e-9), "goal ({i},{j},{h}): {} vs A* {c}", got.cost_to_goal());
                    let p = got.path(&walled).unwrap();
                    assert!(close(p.cost, c, 1e-9) && samples_in_free_cells(&walled, &p));
                    costs.insert(c as i64);
                }
                None => assert!(!got.cost_to_goal().is_finite() && got.path(&walled).is_none(), "goal ({i},{j},{h}) is unreachable for A*"),
            }
        }
        assert!(costs.len() >= 3, "costs must vary: {costs:?}");
    }

    /// **Replan after a cell becomes blocked, by hand (D\* Lite fixture 2).** The `L,R,L` plan's swath
    /// is `(0,0),(1,0),(1,1),(1,2),(2,2),(3,2),(3,3)`; the `R` from `(1,1,N)` passes `(1.79, 2.21)` at
    /// `t = π/4`, which is cell `(1,2)`. Blocking it costs that edge `∞`; the only other feasible
    /// sequence is `S,S,L,S,S` with swath `(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3)`, all free, at
    /// `4 + π/2`. The robot did not move, so `k_m` stays 0. The critical edge is the north-heading `R`
    /// primitive (offsets `(0,0),(0,1),(1,1)`, tail `(1,1,N)`): a tail enumeration that forgot the
    /// rotated copies would leave the old plan in place. Fewer than all 64 states are re-expanded.
    #[test]
    fn dstar_lite_repairs_the_plan_when_a_cell_on_it_becomes_blocked() {
        let open = OccupancyGrid::from_rows(&["....", "....", "....", "...."], 1.0, 0.0, 0.0).unwrap();
        let after = OccupancyGrid::from_rows(&["....", ".#..", "....", "...."], 1.0, 0.0, 0.0).unwrap();
        assert!(after.blocked(1, 2) && !open.blocked(1, 2), "the drawing blocks (1,2)");
        let lat = slr();
        let mut d = DStarLite::new(&open, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).unwrap();
        assert!(close(d.cost_to_goal(), 3.0 * FRAC_PI_2, 1e-12));
        let before = d.expansions();
        assert!(d.cells_changed(&after, &[(1, 2)]), "a repaired path exists");
        let repaired = d.expansions() - before;
        assert!(repaired > 0 && repaired < 64, "re-expansions {repaired} must be some but not all 64 states");
        assert!(close(d.cost_to_goal(), 4.0 + FRAC_PI_2, 1e-9), "repaired cost {}", d.cost_to_goal());
        assert_eq!(d.km(), 0.0);
        let p = d.path(&after).unwrap();
        assert_eq!(letters(&lat, &p), "SSLSS");
        assert!(samples_in_free_cells(&after, &p));
        // oracle: a fresh A* on the changed grid agrees
        let a = lattice_astar(&after, &lat, (0, 0, 0), (3, 3, 1), LatticeWeights::default()).unwrap();
        assert!(close(a.cost, d.cost_to_goal(), 1e-9) && a.nodes == p.nodes);
        // and a change that touches no swath on the plan leaves the cost alone
        assert!(d.cells_changed(&after, &[(0, 3)]));
        assert!(close(d.cost_to_goal(), 4.0 + FRAC_PI_2, 1e-9));
    }

    /// **No known path after a change (D\* Lite fixture 3).** A width-1 corridor, `S,S,S` at cost 3;
    /// blocking `(2,0)` makes every edge into `(3,0,E)` cost `∞` (arcs need rows `±1`, off the map),
    /// `rhs(2,0,E)` becomes `∞`, the queue drains, and `g(start) = ∞`: [`DStarLite::advance`] refuses to
    /// drive (line {25'}). Non-vacuous: before the change the robot could advance.
    #[test]
    fn dstar_lite_reports_no_known_path_after_the_corridor_closes() {
        let open = OccupancyGrid::from_rows(&["...."], 1.0, 0.0, 0.0).unwrap();
        let closed = OccupancyGrid::from_rows(&["..#."], 1.0, 0.0, 0.0).unwrap();
        let lat = slr();
        let mut d = DStarLite::new(&open, &lat, (0, 0, 0), (3, 0, 0), LatticeWeights::default()).unwrap();
        assert!(close(d.cost_to_goal(), 3.0, 1e-12) && letters(&lat, &d.path(&open).unwrap()) == "SSS");
        let mut probe = d.clone();
        assert!(probe.advance(&open).is_some(), "before the change the robot can drive");
        assert!(!d.cells_changed(&closed, &[(2, 0)]), "no path after the corridor closes");
        assert!(!d.cost_to_goal().is_finite() && d.path(&closed).is_none() && d.advance(&closed).is_none());
    }

    /// **The robot moves, then the map changes (`k_m > 0`).** On an open 6×6 grid the plan from
    /// `(0,0,E)` to `(5,5,N)` is `L,R,L,R,L` (`5π/2`; `k = 5` turns, no straights, by the fixture-1
    /// enumeration). After one primitive the robot is at `(1,1,N)`; blocking the next node's cell
    /// `(2,2)` raises `k_m` by `h((0,0),(1,1)) = √2` and forces a repair whose cost must equal a fresh
    /// [`lattice_astar`] from `(1,1,N)` on the changed grid (by hand: `S,R,S,S,L,S = 4 + π`, which is
    /// more than the `2π` remaining before the change). Driving the repaired plan to the goal spends
    /// exactly that cost, one primitive at a time.
    #[test]
    fn dstar_lite_replans_with_a_moving_start() {
        let rows = ["......"; 6];
        let open = OccupancyGrid::from_rows(&rows, 1.0, 0.0, 0.0).unwrap();
        let after = OccupancyGrid::from_rows(&["......", "......", "......", "..#...", "......", "......"], 1.0, 0.0, 0.0).unwrap();
        assert!(after.blocked(2, 2));
        let lat = slr();
        let w = LatticeWeights::default();
        let mut d = DStarLite::new(&open, &lat, (0, 0, 0), (5, 5, 1), w).unwrap();
        assert!(close(d.cost_to_goal(), 5.0 * FRAC_PI_2, 1e-12) && letters(&lat, &d.path(&open).unwrap()) == "LRLRL");
        let (pid, at) = d.advance(&open).unwrap();
        assert_eq!(at, (1, 1, 1));
        assert!(close(lat.prims[pid].cost(&w), FRAC_PI_2, 1e-15));
        let remaining_before = d.cost_to_goal();
        assert!(close(remaining_before, 2.0 * PI, 1e-12));
        assert!(d.cells_changed(&after, &[(2, 2)]), "a repair exists");
        assert!(close(d.km(), std::f64::consts::SQRT_2, 1e-12), "k_m = h(last, start): {}", d.km());
        let fresh = lattice_astar(&after, &lat, (1, 1, 1), (5, 5, 1), w).unwrap();
        assert!(close(fresh.cost, 4.0 + PI, 1e-9), "hand value for the repair: {}", fresh.cost);
        assert!(close(d.cost_to_goal(), fresh.cost, 1e-9), "repaired {} vs fresh A* {}", d.cost_to_goal(), fresh.cost);
        assert!(d.cost_to_goal() > remaining_before, "the block must make the rest dearer");
        let p = d.path(&after).unwrap();
        assert!(samples_in_free_cells(&after, &p) && p.nodes.first() == Some(&(1, 1, 1)) && p.nodes.last() == Some(&(5, 5, 1)));
        // drive it home
        let mut spent = 0.0;
        let mut steps = 0;
        while let Some((pid, _)) = d.advance(&after) {
            spent += lat.prims[pid].cost(&w);
            steps += 1;
            assert!(steps <= 12, "the drive must terminate");
        }
        assert_eq!(d.start(), (5, 5, 1));
        assert!(close(spent, fresh.cost, 1e-9), "driven {spent} vs planned {}", fresh.cost);
    }
}
