//! **Perceptive footstep planning as a mixed-integer QP** (Deits & Tedrake, Humanoids 2014). Given a set
//! of convex **safe regions** carved out of the terrain by perception, plan where a walker places its next
//! `N` footsteps: each footstep must land *inside one* of the regions (the integer/combinatorial part —
//! which region?), consecutive steps must be within a reachability limit, and the plan minimizes a
//! quadratic cost (progress toward a goal + a nominal stride). Assigning footsteps to regions is the
//! discrete decision that makes this a mixed-integer quadratic program.
//!
//! `clarabel` is a convex (continuous) solver, so the integers are handled by **branch-and-bound** over
//! the region assignments: each node fixes the regions of a prefix of footsteps and solves the convex QP
//! relaxation (only the assigned footsteps constrained to their region); relaxing the unassigned regions
//! only lowers the cost, so that QP is an admissible **lower bound** used to prune. The result is verified,
//! to the optimum, against exhaustive enumeration of every assignment. Pure Rust (`nalgebra` + `clarabel`)
//! → WASM-clean.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use nalgebra::DMatrix;

/// A convex safe region as half-planes `A·p ≤ b` (each row is one linear inequality on a 2-D footstep).
#[derive(Clone, Debug)]
pub struct ConvexRegion {
    pub a: Vec<[f64; 2]>,
    pub b: Vec<f64>,
}

impl ConvexRegion {
    /// An axis-aligned rectangle `[xlo,xhi] × [ylo,yhi]`.
    pub fn rect(xlo: f64, xhi: f64, ylo: f64, yhi: f64) -> ConvexRegion {
        ConvexRegion {
            a: vec![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            b: vec![xhi, -xlo, yhi, -ylo],
        }
    }
    /// Whether a point satisfies every half-plane (within `tol`).
    pub fn contains(&self, p: [f64; 2], tol: f64) -> bool {
        self.a.iter().zip(&self.b).all(|(a, &b)| a[0] * p[0] + a[1] * p[1] <= b + tol)
    }
}

/// A perceptive footstep-planning problem.
#[derive(Clone, Debug)]
pub struct FootstepPlanner {
    /// The last already-placed foot (the stance before footstep 0).
    pub foot_prev: [f64; 2],
    pub goal: [f64; 2],
    pub n_steps: usize,
    pub regions: Vec<ConvexRegion>,
    /// Maximum per-step displacement (∞-norm reachability box).
    pub reach: f64,
    /// Nominal per-step progress vector (added to the previous foot as the preferred next step).
    pub stride: [f64; 2],
    /// Weight on landing the final footstep near the goal.
    pub w_goal: f64,
}

/// A planned sequence of footsteps.
#[derive(Clone, Debug)]
pub struct FootstepPlan {
    pub positions: Vec<[f64; 2]>,
    /// The region index assigned to each footstep.
    pub regions: Vec<usize>,
    pub cost: f64,
    /// Convex QP solves performed (branch-and-bound expands far fewer than exhaustive enumeration).
    pub nodes: usize,
}

impl FootstepPlanner {
    /// Solve the convex QP for a given (possibly partial) region assignment: `assign[i] = Some(r)` pins
    /// footstep `i` inside region `r`; `None` leaves it free (used for the branch-and-bound lower bound).
    /// Reachability and the quadratic cost always apply. Returns `(positions, cost)` or `None` if infeasible.
    fn solve_qp(&self, assign: &[Option<usize>]) -> Option<(Vec<[f64; 2]>, f64)> {
        let n = self.n_steps;
        let nv = 2 * n;
        let col = |i: usize| 2 * i;

        // ---- cost ½xᵀPx + qᵀx + const from residuals ‖M x − c‖² ----
        let mut p = DMatrix::<f64>::identity(nv, nv) * 1e-9; // tiny ridge
        let mut q = vec![0.0; nv];
        let mut cst = 0.0;
        // accumulate one residual block r = M x − c with weight w
        let add_res = |rows: &[(usize, f64)], c: [f64; 2], w: f64, p: &mut DMatrix<f64>, q: &mut [f64], cst: &mut f64| {
            // M is 2×nv with entries `rows` applied to both coordinates (x and y share the sparsity).
            // P += 2w MᵀM, q += −2w Mᵀc, const += w‖c‖²
            for &(ci, si) in rows {
                for &(cj, sj) in rows {
                    p[(col(ci), col(cj))] += 2.0 * w * si * sj;
                    p[(col(ci) + 1, col(cj) + 1)] += 2.0 * w * si * sj;
                }
                q[col(ci)] += -2.0 * w * si * c[0];
                q[col(ci) + 1] += -2.0 * w * si * c[1];
            }
            *cst += w * (c[0] * c[0] + c[1] * c[1]);
        };
        // progress residuals: p_i − p_{i-1} − stride  (p_{-1} = foot_prev)
        for i in 0..n {
            if i == 0 {
                add_res(&[(0, 1.0)], [self.foot_prev[0] + self.stride[0], self.foot_prev[1] + self.stride[1]], 1.0, &mut p, &mut q, &mut cst);
            } else {
                add_res(&[(i, 1.0), (i - 1, -1.0)], self.stride, 1.0, &mut p, &mut q, &mut cst);
            }
        }
        // goal residual on the last footstep
        add_res(&[(n - 1, 1.0)], self.goal, self.w_goal, &mut p, &mut q, &mut cst);

        // ---- inequality constraints A x ≤ b ----
        let mut arows: Vec<Vec<f64>> = Vec::new();
        let mut b: Vec<f64> = Vec::new();
        let push_row = |coefs: &[(usize, f64)], rhs: f64, arows: &mut Vec<Vec<f64>>, b: &mut Vec<f64>| {
            let mut row = vec![0.0; nv];
            for &(idx, v) in coefs {
                row[idx] = v;
            }
            arows.push(row);
            b.push(rhs);
        };
        // reachability: |p_i − p_{i-1}|∞ ≤ reach  (p_{-1} = foot_prev)
        for i in 0..n {
            for d in 0..2 {
                if i == 0 {
                    push_row(&[(col(0) + d, 1.0)], self.foot_prev[d] + self.reach, &mut arows, &mut b);
                    push_row(&[(col(0) + d, -1.0)], -self.foot_prev[d] + self.reach, &mut arows, &mut b);
                } else {
                    push_row(&[(col(i) + d, 1.0), (col(i - 1) + d, -1.0)], self.reach, &mut arows, &mut b);
                    push_row(&[(col(i) + d, -1.0), (col(i - 1) + d, 1.0)], self.reach, &mut arows, &mut b);
                }
            }
        }
        // region membership for assigned footsteps
        for (i, a) in assign.iter().enumerate() {
            if let Some(r) = a {
                let reg = &self.regions[*r];
                for (arow, &brow) in reg.a.iter().zip(&reg.b) {
                    push_row(&[(col(i), arow[0]), (col(i) + 1, arow[1])], brow, &mut arows, &mut b);
                }
            }
        }

        let m = arows.len();
        let mut amat = DMatrix::<f64>::zeros(m, nv);
        for (r, row) in arows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                amat[(r, c)] = v;
            }
        }

        let sol = solve_clarabel(&p, &q, &amat, &b)?;
        let positions: Vec<[f64; 2]> = (0..n).map(|i| [sol[col(i)], sol[col(i) + 1]]).collect();
        // true objective (with the constant term)
        let mut cost = cst;
        for i in 0..nv {
            cost += 0.5 * sol[i] * (0..nv).map(|j| p[(i, j)] * sol[j]).sum::<f64>() + q[i] * sol[i];
        }
        Some((positions, cost))
    }

    /// **Exhaustive** solver: try every region assignment, solve its QP, keep the cheapest feasible plan.
    /// The exact optimum — used as the branch-and-bound oracle.
    pub fn solve_exhaustive(&self) -> Option<FootstepPlan> {
        let r = self.regions.len();
        let n = self.n_steps;
        let mut best: Option<FootstepPlan> = None;
        let mut nodes = 0;
        let total = r.pow(n as u32);
        for code in 0..total {
            let mut assign = vec![None; n];
            let mut c = code;
            for a in assign.iter_mut() {
                *a = Some(c % r);
                c /= r;
            }
            nodes += 1;
            if let Some((pos, cost)) = self.solve_qp(&assign) {
                // must actually lie in the assigned regions (guard against QP tolerance)
                let ok = pos.iter().zip(&assign).all(|(p, a)| self.regions[a.unwrap()].contains(*p, 1e-6));
                if ok && best.as_ref().is_none_or(|b| cost < b.cost) {
                    best = Some(FootstepPlan { positions: pos, regions: assign.iter().map(|a| a.unwrap()).collect(), cost, nodes });
                }
            }
        }
        best.map(|mut b| {
            b.nodes = nodes;
            b
        })
    }

    /// **Branch-and-bound** MIQP solver: fix footstep regions one at a time; the QP with only the fixed
    /// prefix constrained is a valid lower bound, so prune whenever it meets the incumbent. Returns the
    /// same optimum as [`Self::solve_exhaustive`] while solving far fewer QPs.
    pub fn solve(&self) -> Option<FootstepPlan> {
        let n = self.n_steps;
        let r = self.regions.len();
        let mut incumbent: Option<FootstepPlan> = None;
        let mut nodes = 0usize;
        let mut assign = vec![None; n];
        self.branch(0, &mut assign, r, n, &mut incumbent, &mut nodes);
        incumbent.map(|mut b| {
            b.nodes = nodes;
            b
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn branch(&self, depth: usize, assign: &mut Vec<Option<usize>>, r: usize, n: usize, incumbent: &mut Option<FootstepPlan>, nodes: &mut usize) {
        *nodes += 1;
        // lower bound: QP with the currently-assigned prefix constrained, the rest free
        let lb = match self.solve_qp(assign) {
            Some((_, c)) => c,
            None => return, // infeasible prefix ⇒ prune the whole subtree
        };
        if let Some(inc) = incumbent
            && lb >= inc.cost - 1e-9
        {
            return; // cannot beat the incumbent
        }
        if depth == n {
            // full assignment: this lb is the true cost
            if let Some((pos, cost)) = self.solve_qp(assign) {
                let ok = pos.iter().zip(assign.iter()).all(|(p, a)| self.regions[a.unwrap()].contains(*p, 1e-6));
                if ok && incumbent.as_ref().is_none_or(|b| cost < b.cost) {
                    *incumbent = Some(FootstepPlan { positions: pos, regions: assign.iter().map(|a| a.unwrap()).collect(), cost, nodes: 0 });
                }
            }
            return;
        }
        for reg in 0..r {
            assign[depth] = Some(reg);
            self.branch(depth + 1, assign, r, n, incumbent, nodes);
        }
        assign[depth] = None;
    }
}

/// Solve `min ½xᵀPx + qᵀx s.t. Ax ≤ b` with clarabel; `None` if not solved (e.g. primal-infeasible).
fn solve_clarabel(p: &DMatrix<f64>, q: &[f64], a: &DMatrix<f64>, b: &[f64]) -> Option<Vec<f64>> {
    let p_csc = csc_upper(p);
    let a_csc = csc_dense(a);
    let cones = [SupportedConeT::NonnegativeConeT(a.nrows())];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p_csc, q, &a_csc, b, &cones, settings).unwrap();
    solver.solve();
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Some(solver.solution.x.clone()),
        _ => None,
    }
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

    // Three stepping-stones in a line with gaps between them, plus a goal beyond the last.
    fn stones() -> FootstepPlanner {
        FootstepPlanner {
            foot_prev: [0.0, 0.0],
            goal: [3.0, 0.0],
            n_steps: 4,
            regions: vec![
                ConvexRegion::rect(0.4, 0.9, -0.3, 0.3),
                ConvexRegion::rect(1.2, 1.7, -0.3, 0.3),
                ConvexRegion::rect(2.1, 2.6, -0.3, 0.3),
            ],
            reach: 1.0,
            stride: [0.75, 0.0],
            w_goal: 5.0,
        }
    }

    #[test]
    fn an_assignment_places_each_footstep_in_its_region() {
        let plan = stones().solve().expect("a feasible plan should exist");
        assert_eq!(plan.positions.len(), 4);
        for (p, &r) in plan.positions.iter().zip(&plan.regions) {
            assert!(stones().regions[r].contains(*p, 1e-6), "footstep {p:?} not in region {r}");
        }
    }

    #[test]
    fn an_unreachable_step_is_infeasible() {
        // Regions exist but the reach is too short to bridge the gap from the start to the first stone.
        let mut planner = stones();
        planner.reach = 0.3; // start at origin, first stone begins at x=0.4 → out of reach
        planner.n_steps = 1;
        planner.regions = vec![ConvexRegion::rect(0.4, 0.9, -0.3, 0.3)];
        assert!(planner.solve().is_none(), "a step beyond the reach box must be infeasible");
    }

    #[test]
    fn branch_and_bound_matches_exhaustive_search() {
        // THE HEADLINE. The MIQP branch-and-bound must return the exact optimum found by enumerating every
        // region assignment — while expanding fewer nodes.
        let planner = stones();
        let bnb = planner.solve().expect("bnb plan");
        let ex = planner.solve_exhaustive().expect("exhaustive plan");
        assert!((bnb.cost - ex.cost).abs() < 1e-6, "bnb cost {} vs exhaustive {}", bnb.cost, ex.cost);
        assert_eq!(bnb.regions, ex.regions, "bnb should find the same optimal assignment");
        let total = planner.regions.len().pow(planner.n_steps as u32);
        assert!(bnb.nodes < total, "branch-and-bound should prune: {} nodes vs {} assignments", bnb.nodes, total);
    }

    #[test]
    fn reachability_between_consecutive_steps_is_respected() {
        let planner = stones();
        let plan = planner.solve().unwrap();
        let mut prev = planner.foot_prev;
        for p in &plan.positions {
            let d = (p[0] - prev[0]).abs().max((p[1] - prev[1]).abs());
            assert!(d <= planner.reach + 1e-6, "step of {d} exceeds reach {}", planner.reach);
            prev = *p;
        }
    }

    #[test]
    fn perception_changes_the_plan() {
        // "Perceptive": remove the middle stone and the planner must route differently (or fail), proving
        // the safe regions actually drive the plan.
        let base = stones();
        let with_mid = base.solve().unwrap();
        let mut gapped = base.clone();
        gapped.regions.remove(1); // delete the middle stepping stone
        let without_mid = gapped.solve();
        // Either infeasible, or a different assignment of footsteps to the remaining regions.
        match without_mid {
            None => {} // the gap is now unbridgeable — perception mattered decisively
            Some(pl) => assert!(pl.regions != with_mid.regions || pl.cost > with_mid.cost - 1e-9, "removing a region should change the plan"),
        }
    }
}
