//! **Distributed Potential iLQR** (Williams, Chen & Mehr, ICRA 2023): scalable *game-theoretic* multi-
//! agent trajectory planning. Where a general dynamic game (see [`crate::algames`]) needs a centralized
//! solve of coupled optimal-control problems, this exploits the structure of a **dynamic potential game**:
//! when every agent's interaction cost is *symmetric*, there exists a single scalar **potential** `Φ`
//! whose gradient with respect to agent `i`'s controls equals the gradient of `i`'s own cost. The Nash
//! equilibrium of the game is then simply a *minimizer of that one potential* — so the game can be solved
//! by ordinary (single-objective) trajectory optimization, and, crucially, **distributed**: each agent
//! repeatedly best-responds to the others' current plans (block-coordinate descent on `Φ`), which
//! converges to the potential's minimizer. That is what makes it scale to many interacting robots.
//!
//! Here the agents are 2-D double integrators. Each minimizes a tracking-plus-effort cost plus a symmetric
//! quadratic **formation** coupling `k‖(pᵢ−pⱼ)−(fᵢ−fⱼ)‖²` to every other agent. Because the dynamics are
//! linear and the costs quadratic, each agent's iLQR step is one exact LQ solve (a condensed least-
//! squares), and the potential `Φ` is convex — so the distributed block-coordinate solution is verified,
//! bit-close, against the analytic centralized minimizer, and shown to be a Nash equilibrium. Pure
//! `nalgebra`, deterministic → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector2, Vector4};

/// One agent of the game: where it starts, where it wants to go, its nominal formation slot (the coupling
/// pulls pairwise *relative* positions toward `fᵢ − fⱼ`), and how it weights tracking vs. effort.
#[derive(Clone, Debug)]
pub struct PiAgent {
    pub x0: Vector4<f64>, // [px, py, vx, vy]
    pub goal: Vector2<f64>,
    pub formation: Vector2<f64>,
    pub w_goal: f64,
    pub r_ctrl: f64,
}

/// A dynamic potential game over `agents`, coupled by a symmetric quadratic formation term of weight
/// `k_couple`, planned over `horizon` control steps of duration `dt`.
#[derive(Clone, Debug)]
pub struct PotentialGame {
    pub agents: Vec<PiAgent>,
    pub k_couple: f64,
    pub dt: f64,
    pub horizon: usize,
}

/// The outcome of a distributed solve.
#[derive(Clone, Debug)]
pub struct DpilqrResult {
    /// Each agent's control sequence (`horizon` stacked 2-vectors).
    pub controls: Vec<DVector<f64>>,
    /// Each agent's resulting position trajectory (`horizon` stacked 2-vectors).
    pub positions: Vec<DVector<f64>>,
    pub potential: f64,
    pub rounds: usize,
    /// The potential after each block-coordinate round (monotone non-increasing).
    pub potential_history: Vec<f64>,
}

impl PotentialGame {
    fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// The position-from-control map `p = Su·u + Pfree(x0)` for one agent (identical dynamics for all).
    /// Returns `Su` (2T×2T). Double-integrator dynamics `x⁺ = A x + B u`.
    fn condense(&self) -> DMatrix<f64> {
        let (dt, t) = (self.dt, self.horizon);
        let a = DMatrix::from_row_slice(4, 4, &[1., 0., dt, 0., 0., 1., 0., dt, 0., 0., 1., 0., 0., 0., 0., 1.]);
        let b = DMatrix::from_row_slice(4, 2, &[0., 0., 0., 0., dt, 0., 0., dt]);
        let p = DMatrix::from_row_slice(2, 4, &[1., 0., 0., 0., 0., 1., 0., 0.]);
        // powers of A: apow[k] = A^k
        let mut apow = vec![DMatrix::<f64>::identity(4, 4)];
        for k in 1..=t {
            apow.push(&apow[k - 1] * &a);
        }
        let mut su = DMatrix::zeros(2 * t, 2 * t);
        for tt in 1..=t {
            for s in 0..tt {
                let blk = &p * &apow[tt - 1 - s] * &b; // 2×2
                su.view_mut((2 * (tt - 1), 2 * s), (2, 2)).copy_from(&blk);
            }
        }
        su
    }

    /// Free response `Pfree = [P·A^t·x0]_{t=1..T}` (2T) for an agent's initial state.
    fn free_response(&self, x0: &Vector4<f64>) -> DVector<f64> {
        let (dt, t) = (self.dt, self.horizon);
        let a = DMatrix::from_row_slice(4, 4, &[1., 0., dt, 0., 0., 1., 0., dt, 0., 0., 1., 0., 0., 0., 0., 1.]);
        let mut pf = DVector::zeros(2 * t);
        let mut ax: DVector<f64> = DVector::from_iterator(4, x0.iter().cloned());
        for tt in 1..=t {
            ax = &a * ax;
            pf[2 * (tt - 1)] = ax[0];
            pf[2 * (tt - 1) + 1] = ax[1];
        }
        pf
    }

    /// `goal_i` repeated across the horizon (2T).
    fn stacked(&self, v: &Vector2<f64>) -> DVector<f64> {
        let mut s = DVector::zeros(2 * self.horizon);
        for tt in 0..self.horizon {
            s[2 * tt] = v.x;
            s[2 * tt + 1] = v.y;
        }
        s
    }

    fn positions_of(&self, su: &DMatrix<f64>, pfree: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
        su * u + pfree
    }

    /// Agent `i`'s own cost given every other agent's current positions.
    fn agent_cost(&self, i: usize, su: &DMatrix<f64>, pfree: &[DVector<f64>], goals: &[DVector<f64>], u_i: &DVector<f64>, pos: &[DVector<f64>]) -> f64 {
        let ag = &self.agents[i];
        let p_i = self.positions_of(su, &pfree[i], u_i);
        let mut c = ag.w_goal * (&p_i - &goals[i]).norm_squared() + ag.r_ctrl * u_i.norm_squared();
        for (j, pj) in pos.iter().enumerate() {
            if j == i {
                continue;
            }
            let dij = self.stacked(&(self.agents[i].formation - self.agents[j].formation));
            c += self.k_couple * (&p_i - pj - &dij).norm_squared();
        }
        c
    }

    /// Agent `i`'s own cost under a full control profile (its tracking + effort + coupling to the others) —
    /// the quantity each agent unilaterally minimizes; at the Nash equilibrium it is at a stationary point.
    pub fn agent_cost_of(&self, i: usize, u: &[DVector<f64>]) -> f64 {
        let su = self.condense();
        let pfree: Vec<DVector<f64>> = self.agents.iter().map(|a| self.free_response(&a.x0)).collect();
        let goals: Vec<DVector<f64>> = self.agents.iter().map(|a| self.stacked(&a.goal)).collect();
        let pos: Vec<DVector<f64>> = (0..self.n_agents()).map(|j| self.positions_of(&su, &pfree[j], &u[j])).collect();
        self.agent_cost(i, &su, &pfree, &goals, &u[i], &pos)
    }

    /// Agent `i`'s exact iLQR/LQ best response to the others' fixed positions: minimize its own cost over
    /// its own controls. (For linear dynamics + quadratic cost the iLQR backward–forward pass is this one
    /// condensed linear solve.)
    fn best_response(&self, i: usize, sutsu: &DMatrix<f64>, sut: &DMatrix<f64>, pfree: &[DVector<f64>], goals: &[DVector<f64>], pos: &[DVector<f64>]) -> DVector<f64> {
        let ag = &self.agents[i];
        let n = self.n_agents();
        let deg = (n - 1) as f64;
        let m = 2 * self.horizon;
        // H = (w + deg·k)·SuᵀSu + r·I
        let h = sutsu * (ag.w_goal + deg * self.k_couple) + DMatrix::identity(m, m) * ag.r_ctrl;
        // c = −Suᵀ[ w·a_i + k·Σ_{j≠i}(Pfree_i − p_j − D_ij) ] , a_i = Pfree_i − goal_i
        let a_i = &pfree[i] - &goals[i];
        let mut rhs_inner = &a_i * ag.w_goal;
        for (j, pj) in pos.iter().enumerate() {
            if j == i {
                continue;
            }
            let dij = self.stacked(&(self.agents[i].formation - self.agents[j].formation));
            rhs_inner += (&pfree[i] - pj - &dij) * self.k_couple;
        }
        let c = -(sut * rhs_inner);
        h.lu().solve(&c).expect("agent LQ subproblem is SPD (r>0)")
    }

    /// **Distributed** solve: block-coordinate (Gauss–Seidel) best response until the plans stop changing.
    /// Converges to the minimizer of the potential `Φ` — the Nash equilibrium of the potential game.
    pub fn solve_distributed(&self, max_rounds: usize, tol: f64) -> DpilqrResult {
        let n = self.n_agents();
        let m = 2 * self.horizon;
        let su = self.condense();
        let sut = su.transpose();
        let sutsu = &sut * &su;
        let pfree: Vec<DVector<f64>> = self.agents.iter().map(|a| self.free_response(&a.x0)).collect();
        let goals: Vec<DVector<f64>> = self.agents.iter().map(|a| self.stacked(&a.goal)).collect();

        let mut u: Vec<DVector<f64>> = vec![DVector::zeros(m); n];
        let mut pos: Vec<DVector<f64>> = (0..n).map(|i| self.positions_of(&su, &pfree[i], &u[i])).collect();
        let mut potential_history = vec![self.potential(&su, &pfree, &goals, &u)];
        let mut rounds = 0;

        for _ in 0..max_rounds {
            rounds += 1;
            let mut max_change = 0.0_f64;
            for i in 0..n {
                let ui_new = self.best_response(i, &sutsu, &sut, &pfree, &goals, &pos);
                max_change = max_change.max((&ui_new - &u[i]).amax());
                u[i] = ui_new;
                pos[i] = self.positions_of(&su, &pfree[i], &u[i]); // Gauss–Seidel: others see the update
            }
            potential_history.push(self.potential(&su, &pfree, &goals, &u));
            if max_change < tol {
                break;
            }
        }

        DpilqrResult {
            potential: *potential_history.last().unwrap(),
            positions: pos,
            controls: u,
            rounds,
            potential_history,
        }
    }

    /// The scalar potential `Φ = Σᵢ[w‖pᵢ−goalᵢ‖² + r‖uᵢ‖²] + Σ_{i<j} k‖(pᵢ−pⱼ)−(fᵢ−fⱼ)‖²`.
    fn potential(&self, su: &DMatrix<f64>, pfree: &[DVector<f64>], goals: &[DVector<f64>], u: &[DVector<f64>]) -> f64 {
        let n = self.n_agents();
        let pos: Vec<DVector<f64>> = (0..n).map(|i| self.positions_of(su, &pfree[i], &u[i])).collect();
        let mut phi = 0.0;
        for i in 0..n {
            phi += self.agents[i].w_goal * (&pos[i] - &goals[i]).norm_squared() + self.agents[i].r_ctrl * u[i].norm_squared();
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let dij = self.stacked(&(self.agents[i].formation - self.agents[j].formation));
                phi += self.k_couple * (&pos[i] - &pos[j] - &dij).norm_squared();
            }
        }
        phi
    }

    /// Public convenience: the potential of a full control profile.
    pub fn potential_of(&self, u: &[DVector<f64>]) -> f64 {
        let su = self.condense();
        let pfree: Vec<DVector<f64>> = self.agents.iter().map(|a| self.free_response(&a.x0)).collect();
        let goals: Vec<DVector<f64>> = self.agents.iter().map(|a| self.stacked(&a.goal)).collect();
        self.potential(&su, &pfree, &goals, u)
    }

    /// The **centralized** minimizer of the potential `Φ` — one joint convex least-squares solve over all
    /// agents' controls at once. The distributed solve must converge to this (a correctness oracle).
    pub fn solve_centralized(&self) -> Vec<DVector<f64>> {
        let n = self.n_agents();
        let m = 2 * self.horizon;
        let nv = n * m;
        let su = self.condense();
        let pfree: Vec<DVector<f64>> = self.agents.iter().map(|a| self.free_response(&a.x0)).collect();
        let goals: Vec<DVector<f64>> = self.agents.iter().map(|a| self.stacked(&a.goal)).collect();

        // Stack weighted linear residuals ‖A·U − b‖² and solve the normal equations AᵀA U = Aᵀb.
        let mut rows: Vec<DMatrix<f64>> = Vec::new();
        let mut bs: Vec<DVector<f64>> = Vec::new();
        let col = |i: usize| i * m;
        for i in 0..n {
            let w = self.agents[i].w_goal.sqrt();
            // tracking: √w·(Su u_i + (Pfree_i − goal_i))  ⇒ A=√w Su on i, b = −√w(Pfree_i−goal_i)
            let mut ai = DMatrix::zeros(m, nv);
            ai.view_mut((0, col(i)), (m, m)).copy_from(&(&su * w));
            rows.push(ai);
            bs.push(-((&pfree[i] - &goals[i]) * w));
            // control: √r·u_i
            let r = self.agents[i].r_ctrl.sqrt();
            let mut ci = DMatrix::zeros(m, nv);
            ci.view_mut((0, col(i)), (m, m)).copy_from(&(DMatrix::identity(m, m) * r));
            rows.push(ci);
            bs.push(DVector::zeros(m));
        }
        let kc = self.k_couple.sqrt();
        for i in 0..n {
            for j in (i + 1)..n {
                let dij = self.stacked(&(self.agents[i].formation - self.agents[j].formation));
                let mut aij = DMatrix::zeros(m, nv);
                aij.view_mut((0, col(i)), (m, m)).copy_from(&(&su * kc));
                aij.view_mut((0, col(j)), (m, m)).copy_from(&(&su * (-kc)));
                rows.push(aij);
                bs.push(-((&pfree[i] - &pfree[j] - &dij) * kc));
            }
        }
        let total: usize = rows.iter().map(|r| r.nrows()).sum();
        let mut a = DMatrix::zeros(total, nv);
        let mut b = DVector::zeros(total);
        let mut off = 0;
        for (ai, bi) in rows.iter().zip(bs.iter()) {
            let h = ai.nrows();
            a.view_mut((off, 0), (h, nv)).copy_from(ai);
            b.rows_mut(off, h).copy_from(bi);
            off += h;
        }
        let ata = a.transpose() * &a;
        let atb = a.transpose() * b;
        let u_all = ata.lu().solve(&atb).expect("centralized normal equations are SPD");
        (0..n).map(|i| u_all.rows(col(i), m).into_owned()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v2(x: f64, y: f64) -> Vector2<f64> {
        Vector2::new(x, y)
    }
    fn x0(px: f64, py: f64) -> Vector4<f64> {
        Vector4::new(px, py, 0.0, 0.0)
    }

    /// A small crossing game: agents want to swap sides while a formation coupling keeps them apart.
    fn game(n: usize) -> PotentialGame {
        let ring = |i: usize| {
            let th = std::f64::consts::TAU * i as f64 / n as f64;
            (th.cos(), th.sin())
        };
        let agents = (0..n)
            .map(|i| {
                let (cx, cy) = ring(i);
                PiAgent {
                    x0: x0(2.0 * cx, 2.0 * cy),
                    goal: v2(-2.0 * cx, -2.0 * cy), // cross to the opposite side
                    formation: v2(cx, cy),
                    w_goal: 2.0,
                    r_ctrl: 0.1,
                }
            })
            .collect();
        PotentialGame { agents, k_couple: 0.5, dt: 0.2, horizon: 15 }
    }

    #[test]
    fn the_condensation_matches_a_forward_rollout() {
        let g = game(2);
        let su = g.condense();
        let pf = g.free_response(&g.agents[0].x0);
        // random-ish control, roll it out by hand and compare positions
        let m = 2 * g.horizon;
        let mut u = DVector::zeros(m);
        for k in 0..m {
            u[k] = 0.1 * ((k as f64 * 0.7).sin());
        }
        let p_cond = &su * &u + &pf;
        // manual rollout
        let (dt, t) = (g.dt, g.horizon);
        let mut x = g.agents[0].x0;
        for tt in 0..t {
            let ax = x[0] + dt * x[2];
            let ay = x[1] + dt * x[3];
            let avx = x[2] + dt * u[2 * tt];
            let avy = x[3] + dt * u[2 * tt + 1];
            x = Vector4::new(ax, ay, avx, avy);
            assert!((p_cond[2 * tt] - x[0]).abs() < 1e-10 && (p_cond[2 * tt + 1] - x[1]).abs() < 1e-10, "condensation mismatch at {tt}");
        }
    }

    #[test]
    fn it_is_an_exact_potential_game() {
        // THE DEFINING PROPERTY. ∂J_i/∂u_i must equal ∂Φ/∂u_i at any point — this is what makes the
        // Nash equilibrium a minimizer of the single potential. Checked by finite differences.
        let g = game(3);
        let su = g.condense();
        let pfree: Vec<DVector<f64>> = g.agents.iter().map(|a| g.free_response(&a.x0)).collect();
        let goals: Vec<DVector<f64>> = g.agents.iter().map(|a| g.stacked(&a.goal)).collect();
        let m = 2 * g.horizon;
        // a nontrivial profile
        let mut u: Vec<DVector<f64>> = (0..3).map(|i| DVector::from_iterator(m, (0..m).map(|k| 0.05 * ((k + i) as f64).cos()))).collect();
        let i = 1;
        let eps = 1e-6;
        for k in 0..m {
            let mut up = u.clone();
            let mut um = u.clone();
            up[i][k] += eps;
            um[i][k] -= eps;
            // ∂Φ/∂u_i,k
            let dphi = (g.potential(&su, &pfree, &goals, &up) - g.potential(&su, &pfree, &goals, &um)) / (2.0 * eps);
            // ∂J_i/∂u_i,k (others fixed)
            let pos = |uu: &Vec<DVector<f64>>| (0..3).map(|j| g.positions_of(&su, &pfree[j], &uu[j])).collect::<Vec<_>>();
            let dji = (g.agent_cost(i, &su, &pfree, &goals, &up[i], &pos(&up)) - g.agent_cost(i, &su, &pfree, &goals, &um[i], &pos(&um))) / (2.0 * eps);
            assert!((dphi - dji).abs() < 1e-6, "potential identity broken at k={k}: ∂Φ {dphi} vs ∂J_i {dji}");
        }
    }

    #[test]
    fn distributed_converges_to_the_centralized_potential_minimizer() {
        // THE HEADLINE. The decentralized block-coordinate solve reaches the same joint plan as one
        // centralized minimization of the potential — the Nash equilibrium — without ever solving the
        // coupled problem jointly.
        let g = game(4);
        let dist = g.solve_distributed(200, 1e-10);
        let cent = g.solve_centralized();
        for i in 0..g.n_agents() {
            let diff = (&dist.controls[i] - &cent[i]).amax();
            assert!(diff < 1e-6, "agent {i}: distributed vs centralized diff {diff}");
        }
        // both realize the same potential value
        assert!((dist.potential - g.potential_of(&cent)).abs() < 1e-8, "potential mismatch");
    }

    #[test]
    fn the_potential_decreases_monotonically() {
        let g = game(4);
        let res = g.solve_distributed(200, 1e-10);
        for w in res.potential_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "potential must not increase: {} → {}", w[0], w[1]);
        }
        assert!(res.potential_history.last().unwrap() < &(res.potential_history[0] * 0.9), "the game should make real progress");
    }

    #[test]
    fn the_solution_is_a_nash_equilibrium() {
        // At the solution no agent can lower its OWN cost by unilaterally re-planning: its current plan is
        // already its best response, and any perturbation raises its cost.
        let g = game(3);
        let su = g.condense();
        let pfree: Vec<DVector<f64>> = g.agents.iter().map(|a| g.free_response(&a.x0)).collect();
        let goals: Vec<DVector<f64>> = g.agents.iter().map(|a| g.stacked(&a.goal)).collect();
        let res = g.solve_distributed(300, 1e-12);
        let pos = &res.positions;
        for i in 0..g.n_agents() {
            let ci = g.agent_cost(i, &su, &pfree, &goals, &res.controls[i], pos);
            // best response equals the held plan (already optimal)
            let br = g.best_response(i, &(su.transpose() * &su), &su.transpose(), &pfree, &goals, pos);
            assert!((&br - &res.controls[i]).amax() < 1e-5, "agent {i} is not at its best response");
            // a deliberate perturbation can only raise its cost
            let mut bad = res.controls[i].clone();
            bad[0] += 0.3;
            let cbad = g.agent_cost(i, &su, &pfree, &goals, &bad, pos);
            assert!(cbad > ci - 1e-9, "perturbing agent {i} should not reduce its cost: {cbad} vs {ci}");
        }
    }
}
