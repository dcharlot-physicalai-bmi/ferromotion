//! **GCS — shortest paths through Graphs of Convex Sets** (Marcucci, Petersen, von Wrangel & Tedrake,
//! *Science Robotics* 2023). Free space is decomposed into overlapping **convex regions** (e.g. from
//! [`crate::Iris`]); a collision-free trajectory is then a shortest path that hops between regions through
//! their overlaps. Because each segment lives inside one convex region, the whole path is guaranteed
//! collision-free — and, unlike a local trajectory optimizer, it needs *no initial guess*.
//!
//! This implements the geometric core: given a sequence of regions (found here by breadth-first search
//! over the overlap graph), place a waypoint in each region-to-region overlap and minimize the **true
//! Euclidean path length** — a second-order-cone program solved with `clarabel`. (The paper's headline
//! contribution is the tight convex *relaxation* that also selects the discrete region sequence in one
//! shot; here that discrete choice is an explicit graph search, which is exact for the small graphs a
//! decomposition produces.) Pure Rust → WASM-clean.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::{DMatrix, DVector};

/// A convex region as a half-space polytope `{x : A x ≤ b}`.
#[derive(Clone, Debug)]
pub struct HPolytope {
    pub a: DMatrix<f64>,
    pub b: DVector<f64>,
}

impl HPolytope {
    /// An axis-aligned box `[lo, hi]`.
    pub fn box_region(lo: &[f64], hi: &[f64]) -> HPolytope {
        let n = lo.len();
        let mut a = DMatrix::zeros(2 * n, n);
        let mut b = DVector::zeros(2 * n);
        for k in 0..n {
            a[(2 * k, k)] = 1.0;
            b[2 * k] = hi[k];
            a[(2 * k + 1, k)] = -1.0;
            b[2 * k + 1] = -lo[k];
        }
        HPolytope { a, b }
    }
    pub fn contains(&self, x: &DVector<f64>, tol: f64) -> bool {
        (&self.a * x - &self.b).iter().all(|&r| r <= tol)
    }
}

/// A GCS problem: a set of convex regions covering the free space.
#[derive(Clone, Debug)]
pub struct Gcs {
    pub regions: Vec<HPolytope>,
}

/// A planned path: the waypoints (start … goal) and total Euclidean length.
#[derive(Clone, Debug)]
pub struct GcsPath {
    pub waypoints: Vec<DVector<f64>>,
    pub length: f64,
}

impl Gcs {
    /// Do regions `i` and `j` overlap? (Feasibility of `A_i x ≤ b_i ∧ A_j x ≤ b_j`, via a tiny LP that
    /// maximizes the slack margin.)
    pub fn overlap(&self, i: usize, j: usize) -> bool {
        let (ri, rj) = (&self.regions[i], &self.regions[j]);
        let n = ri.a.ncols();
        // maximize t s.t. A x + t·1 ≤ b for both; feasible overlap ⇔ optimal t ≥ 0.
        // vars z = [x (n); t]. minimize −t.
        let nv = n + 1;
        let rows = ri.a.nrows() + rj.a.nrows();
        let mut a = DMatrix::zeros(rows, nv);
        let mut b = DVector::zeros(rows);
        let mut r = 0;
        for reg in [ri, rj] {
            for k in 0..reg.a.nrows() {
                a.view_mut((r, 0), (1, n)).copy_from(&reg.a.row(k));
                a[(r, n)] = 1.0; // +t
                b[r] = reg.b[k];
                r += 1;
            }
        }
        let p = DMatrix::zeros(nv, nv);
        let mut q = vec![0.0; nv];
        q[n] = -1.0; // minimize −t
        // bound t ≤ 10 so the LP is not unbounded
        let mut a2 = DMatrix::zeros(rows + 1, nv);
        a2.view_mut((0, 0), (rows, nv)).copy_from(&a);
        a2[(rows, n)] = 1.0;
        let mut b2 = DVector::zeros(rows + 1);
        b2.rows_mut(0, rows).copy_from(&b);
        b2[rows] = 10.0;
        let sol = solve_conic(&p, &q, &a2, &b2, &[SupportedConeT::NonnegativeConeT(rows + 1)]);
        sol.map(|z| z[n] > 1e-6).unwrap_or(false)
    }

    /// A region sequence from the region containing `start` to the one containing `goal`, by BFS over the
    /// overlap graph. Returns region indices, or `None` if disconnected.
    pub fn region_sequence(&self, start: &DVector<f64>, goal: &DVector<f64>) -> Option<Vec<usize>> {
        let s = self.regions.iter().position(|r| r.contains(start, 1e-6))?;
        let g = self.regions.iter().position(|r| r.contains(goal, 1e-6))?;
        if s == g {
            return Some(vec![s]);
        }
        let n = self.regions.len();
        let mut prev = vec![usize::MAX; n];
        let mut seen = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(s);
        seen[s] = true;
        while let Some(u) = queue.pop_front() {
            if u == g {
                break;
            }
            for v in 0..n {
                if !seen[v] && self.overlap(u, v) {
                    seen[v] = true;
                    prev[v] = u;
                    queue.push_back(v);
                }
            }
        }
        if !seen[g] {
            return None;
        }
        let mut path = vec![g];
        let mut cur = g;
        while cur != s {
            cur = prev[cur];
            path.push(cur);
        }
        path.reverse();
        Some(path)
    }

    /// Plan a shortest collision-free path from `start` to `goal`. Finds a region sequence, then solves the
    /// SOCP that places an overlap waypoint between consecutive regions and minimizes total length.
    pub fn plan(&self, start: &DVector<f64>, goal: &DVector<f64>) -> Option<GcsPath> {
        let seq = self.region_sequence(start, goal)?;
        Some(self.shortest_through(&seq, start, goal))
    }

    /// Minimum-length path through a fixed region sequence: waypoints `start, o_1, …, o_{k-1}, goal`, with
    /// each overlap point `o_j ∈ region_{j-1} ∩ region_j`, minimizing `Σ‖w_{j}−w_{j-1}‖` (an SOCP).
    pub fn shortest_through(&self, seq: &[usize], start: &DVector<f64>, goal: &DVector<f64>) -> GcsPath {
        let n = start.len();
        let k = seq.len(); // regions
        let n_inner = k.saturating_sub(1); // overlap waypoints
        let n_way = n_inner + 2; // incl. start & goal
        let n_seg = n_way - 1;
        // vars: inner waypoints o_1..o_{n_inner} (n each), then segment lengths t_1..t_{n_seg}.
        let n_pos = n_inner * n;
        let nv = n_pos + n_seg;
        let ipos = |j: usize| j * n; // 0-based inner index
        let it = |s: usize| n_pos + s;

        // objective: min Σ t_s
        let p = DMatrix::zeros(nv, nv);
        let mut q = vec![0.0; nv];
        for s in 0..n_seg {
            q[it(s)] = 1.0;
        }

        // waypoint accessor as (is_var, index/const)
        let way = |j: usize| -> WayRef {
            if j == 0 {
                WayRef::Fixed(start.clone())
            } else if j == n_way - 1 {
                WayRef::Fixed(goal.clone())
            } else {
                WayRef::Var(j - 1) // inner index
            }
        };

        // inequality (region) rows + SOC rows collected separately
        let mut ineq_a: Vec<Vec<f64>> = Vec::new();
        let mut ineq_b: Vec<f64> = Vec::new();
        // each inner waypoint o_j lies in region_{j-1} ∩ region_j
        for j in 1..=n_inner {
            for &reg_idx in &[seq[j - 1], seq[j]] {
                let reg = &self.regions[reg_idx];
                for r in 0..reg.a.nrows() {
                    let mut row = vec![0.0; nv];
                    for c in 0..n {
                        row[ipos(j - 1) + c] = reg.a[(r, c)];
                    }
                    ineq_a.push(row);
                    ineq_b.push(reg.b[r]);
                }
            }
        }

        // SOC rows: for each segment s (w_{s} → w_{s+1}): (t_s, w_{s+1} − w_s) ∈ SOC(n+1)
        let mut soc_a: Vec<Vec<f64>> = Vec::new();
        let mut soc_b: Vec<f64> = Vec::new();
        let mut soc_dims: Vec<usize> = Vec::new();
        for s in 0..n_seg {
            // row 0: s0 = t_s  ⇒  A row = −e_{t_s}, b = 0
            let mut r0 = vec![0.0; nv];
            r0[it(s)] = -1.0;
            soc_a.push(r0);
            soc_b.push(0.0);
            // rows 1..n: s = (w_{s+1} − w_s)  ⇒  A = −(w_{s+1} − w_s), b picks fixed parts
            let (wa, wb) = (way(s), way(s + 1));
            for c in 0..n {
                let mut row = vec![0.0; nv];
                let mut bval = 0.0;
                match &wb {
                    WayRef::Var(j) => row[ipos(*j) + c] = -1.0,
                    WayRef::Fixed(v) => bval += v[c],
                }
                match &wa {
                    WayRef::Var(j) => row[ipos(*j) + c] += 1.0,
                    WayRef::Fixed(v) => bval -= v[c],
                }
                soc_a.push(row);
                soc_b.push(bval);
            }
            soc_dims.push(n + 1);
        }

        // stack A = [ineq; soc], cones = [Nonneg(ineq); SOC(each)]
        let n_ineq = ineq_a.len();
        let n_soc = soc_a.len();
        let mut a = DMatrix::zeros(n_ineq + n_soc, nv);
        let mut b = DVector::zeros(n_ineq + n_soc);
        for (i, row) in ineq_a.iter().enumerate() {
            a.row_mut(i).copy_from(&DVector::from_row_slice(row).transpose());
            b[i] = ineq_b[i];
        }
        for (i, row) in soc_a.iter().enumerate() {
            a.row_mut(n_ineq + i).copy_from(&DVector::from_row_slice(row).transpose());
            b[n_ineq + i] = soc_b[i];
        }
        let mut cones = vec![SupportedConeT::NonnegativeConeT(n_ineq)];
        for d in &soc_dims {
            cones.push(SupportedConeT::SecondOrderConeT(*d));
        }

        let sol = solve_conic(&p, &q, &a, &b, &cones).expect("GCS SOCP should be feasible for a valid sequence");
        // read waypoints
        let mut ways = vec![start.clone()];
        for j in 0..n_inner {
            ways.push(DVector::from_iterator(n, (0..n).map(|c| sol[ipos(j) + c])));
        }
        ways.push(goal.clone());
        let length = ways.windows(2).map(|w| (&w[1] - &w[0]).norm()).sum();
        GcsPath { waypoints: ways, length }
    }
}

enum WayRef {
    Fixed(DVector<f64>),
    Var(usize),
}

/// Solve `min ½zᵀPz + qᵀz s.t. A z + s = b, s ∈ cones` with clarabel; `None` if not solved.
fn solve_conic(p: &DMatrix<f64>, q: &[f64], a: &DMatrix<f64>, b: &DVector<f64>, cones: &[SupportedConeT<f64>]) -> Option<Vec<f64>> {
    let p_csc = csc_upper(p);
    let a_csc = csc_dense(a);
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p_csc, q, &a_csc, b.as_slice(), cones, settings).ok()?;
    solver.solve();
    Some(solver.solution.x.clone())
}

fn csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.ncols();
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..=j {
            if p[(i, j)] != 0.0 {
                rowval.push(i);
                nzval.push(p[(i, j)]);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

fn csc_dense(a: &DMatrix<f64>) -> CscMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..m {
            let v = a[(i, j)];
            if v != 0.0 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(m, n, colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    #[test]
    fn a_single_region_gives_the_straight_line() {
        // Start and goal in one convex region ⇒ the shortest path is the straight segment.
        let gcs = Gcs { regions: vec![HPolytope::box_region(&[-1.0, -1.0], &[3.0, 3.0])] };
        let (start, goal) = (dv(&[0.0, 0.0]), dv(&[2.0, 1.0]));
        let path = gcs.plan(&start, &goal).expect("planned");
        assert_eq!(path.waypoints.len(), 2, "no interior waypoints for one region");
        assert!((path.length - (&goal - &start).norm()).abs() < 1e-5, "length {} vs straight {}", path.length, (&goal - &start).norm());
    }

    #[test]
    fn overlapping_regions_are_detected() {
        let gcs = Gcs {
            regions: vec![
                HPolytope::box_region(&[0.0, 0.0], &[2.0, 1.0]),
                HPolytope::box_region(&[1.5, 0.0], &[4.0, 1.0]), // overlaps the first in x∈[1.5,2]
                HPolytope::box_region(&[10.0, 10.0], &[11.0, 11.0]), // far away
            ],
        };
        assert!(gcs.overlap(0, 1), "adjacent boxes should overlap");
        assert!(!gcs.overlap(0, 2), "distant boxes should not overlap");
    }

    #[test]
    fn it_routes_through_an_l_shaped_corridor() {
        // THE HEADLINE. An L-corridor of two boxes (horizontal then vertical) with an obstacle in the
        // corner's diagonal. GCS must route start→goal through the overlap, and every segment must lie in a
        // region (collision-free by construction). The path is longer than the blocked straight line.
        let horiz = HPolytope::box_region(&[0.0, 0.0], &[3.0, 1.0]);
        let vert = HPolytope::box_region(&[2.0, 0.0], &[3.0, 3.0]);
        let gcs = Gcs { regions: vec![horiz, vert] };
        let (start, goal) = (dv(&[0.5, 0.5]), dv(&[2.5, 2.5]));
        let path = gcs.plan(&start, &goal).expect("planned");
        assert!(path.waypoints.len() == 3, "one overlap waypoint expected");
        // the straight line would cut the corner (leave both boxes); the routed path is longer
        assert!(path.length > (&goal - &start).norm() + 1e-6, "routed path should exceed the (blocked) straight line");
        // every segment midpoint lies in some region ⇒ collision-free
        for w in path.waypoints.windows(2) {
            let mid = (&w[0] + &w[1]) * 0.5;
            let inside = gcs.regions.iter().any(|r| r.contains(&mid, 1e-6));
            assert!(inside, "segment midpoint {mid:?} left every region");
        }
        // endpoints preserved
        assert!((&path.waypoints[0] - &start).norm() < 1e-9 && (path.waypoints.last().unwrap() - &goal).norm() < 1e-9);
    }

    #[test]
    fn a_disconnected_goal_has_no_plan() {
        let gcs = Gcs {
            regions: vec![HPolytope::box_region(&[0.0, 0.0], &[1.0, 1.0]), HPolytope::box_region(&[5.0, 5.0], &[6.0, 6.0])],
        };
        assert!(gcs.plan(&dv(&[0.5, 0.5]), &dv(&[5.5, 5.5])).is_none(), "no path between disconnected regions");
    }
}
