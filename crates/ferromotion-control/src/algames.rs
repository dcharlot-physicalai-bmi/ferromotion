//! ALGAMES — game-theoretic trajectory optimization (Le Cleac'h, Schwager, Manchester).
//!
//! We solve a constrained dynamic game for a generalized Nash equilibrium among `N` players. Each
//! player is a 2-D double-integrator point robot (state `[px,py,vx,vy]`, control `[ax,ay]`) that
//! minimizes its own tracking-plus-effort cost `Σₜ w‖pₜ−goal‖² + Σₜ r‖uₜ‖²` over the horizon,
//! subject to *shared* pairwise collision-avoidance inequalities `‖pᵢ−pⱼ‖ ≥ d_safe`.
//!
//! The equilibrium is the stationary point of every player's own (augmented) Lagrangian w.r.t. its
//! own controls simultaneously: `∇_{Uᵢ} Lᵢ = 0 ∀i`. Because the dynamics are linear we *condense*
//! each player's trajectory to a function of its control sequence alone (`p = D·U + p_free`), so
//! the decision vector is just the stacked controls. An inner Levenberg-damped Newton root-find
//! drives the stacked stationarity residual to zero (game Jacobian by central finite differences),
//! and an outer augmented-Lagrangian loop lifts the collision penalty `ρ` and updates the
//! multipliers `λ ← max(0, λ + ρ·c)` until the constraints are satisfied. Pure `nalgebra`, no
//! randomness in the solve → WASM-clean and bit-for-bit reproducible.

use nalgebra::{DMatrix, DVector};

/// One player: where it starts, where it wants to go, and how it weights the two cost terms.
#[derive(Clone, Debug)]
pub struct Player {
    /// Initial state `[px, py, vx, vy]`.
    pub start: [f64; 4],
    /// Goal position `[gx, gy]` (velocity target is zero, encoded only via effort cost).
    pub goal: [f64; 2],
    /// Weight on per-step position error `‖pₜ − goal‖²`.
    pub w: f64,
    /// Weight on per-step control effort `‖uₜ‖²`.
    pub r: f64,
}

/// An ALGAMES problem: the players, the discretization, the safety radius, and the AL schedule.
#[derive(Clone, Debug)]
pub struct AlGames {
    pub players: Vec<Player>,
    /// Number of control steps `N` (states run `x₀ … x_N`).
    pub horizon: usize,
    pub dt: f64,
    /// Minimum allowed inter-player distance `‖pᵢ − pⱼ‖ ≥ d_safe`.
    pub d_safe: f64,
    /// Outer augmented-Lagrangian iterations.
    pub max_outer: usize,
    /// Inner Newton iterations per outer step.
    pub max_inner: usize,
    /// Initial collision penalty.
    pub rho0: f64,
    /// Penalty ceiling.
    pub rho_max: f64,
    /// Penalty growth factor between outer iterations.
    pub rho_scale: f64,
    /// Convergence tolerance on the max constraint violation (`d_safe − dist`).
    pub tol: f64,
}

/// Result of an ALGAMES solve: per-player plans plus equilibrium diagnostics.
#[derive(Clone, Debug)]
pub struct AlGamesResult {
    /// Per-player control trajectories, `controls[i][t] = [ax, ay]` (length `horizon`).
    pub controls: Vec<Vec<[f64; 2]>>,
    /// Per-player state trajectories, `states[i][t] = [px,py,vx,vy]` (length `horizon + 1`, incl. start).
    pub states: Vec<Vec<[f64; 4]>>,
    pub converged: bool,
    pub outer_iters: usize,
    /// Final max constraint violation over all pairs/timesteps (`≤ 0` ⇒ satisfied).
    pub max_violation: f64,
    /// Final stacked stationarity residual `‖∇L‖`.
    pub residual: f64,
    /// Collision multipliers used in the final inner solve (`pairs × horizon`, pair-major).
    pub multipliers: Vec<f64>,
    /// Collision penalty used in the final inner solve.
    pub rho: f64,
}

/// Precomputed, per-solve linear-algebra scratch shared by the gradient / Jacobian evaluations.
struct Work {
    np: usize,     // number of players
    n: usize,      // horizon
    m: usize,      // decision vars per player = 2·n
    d: DMatrix<f64>,           // position-from-control map, 2n × 2n (shared: identical dynamics)
    hg: Vec<DMatrix<f64>>,     // per-player quadratic Hessian of Jᵢ: 2w DᵀD + 2r I
    lin: Vec<DVector<f64>>,    // per-player linear term of ∇Jᵢ: 2w Dᵀ(p_free − G)
    p_free: Vec<DVector<f64>>, // per-player free (control-independent) position response, 2n
    pairs: Vec<(usize, usize)>,
    d_safe: f64,
}

impl Work {
    /// Positions of player `i` over `t=1..=n` for its control slice: `D·Uᵢ + p_freeᵢ` (length 2n).
    fn positions(&self, u: &DVector<f64>, i: usize) -> DVector<f64> {
        let ui = u.rows(i * self.m, self.m).into_owned();
        &self.d * &ui + &self.p_free[i]
    }

    /// Stacked stationarity residual `g = [∇_{U₀}L₀; …]` at controls `u`, multipliers `lam`,
    /// penalty `rho`.
    fn grad(&self, u: &DVector<f64>, lam: &[f64], rho: f64) -> DVector<f64> {
        let (np, n, m) = (self.np, self.n, self.m);
        let total = np * m;
        let mut g = DVector::zeros(total);

        // Per-player quadratic part ∇Jᵢ = Hgᵢ·Uᵢ + linᵢ, and cache positions for the coupling.
        let mut pos: Vec<DVector<f64>> = Vec::with_capacity(np);
        for i in 0..np {
            let ui = u.rows(i * m, m).into_owned();
            let gi = &self.hg[i] * &ui + &self.lin[i];
            g.rows_mut(i * m, m).copy_from(&gi);
            pos.push(&self.d * &ui + &self.p_free[i]);
        }

        // Shared pairwise collision penalty. The augmented-Lagrangian term for constraint
        // c = d_safe − ‖pᵢ−pⱼ‖ ≤ 0 appears in BOTH players' Lagrangians; each differentiates w.r.t.
        // its own controls. ∂c/∂pᵢ = −δ/‖δ‖ (δ = pᵢ−pⱼ), ∂c/∂pⱼ = +δ/‖δ‖. The active multiplier is
        // μ = max(0, λ + ρ·c); its gradient contribution is Dₜᵀ·(μ·∂c/∂p).
        for (pidx, &(i, j)) in self.pairs.iter().enumerate() {
            for t in 1..=n {
                let r0 = 2 * (t - 1);
                let dx = pos[i][r0] - pos[j][r0];
                let dy = pos[i][r0 + 1] - pos[j][r0 + 1];
                let dist = (dx * dx + dy * dy).sqrt().max(1e-9);
                let c = self.d_safe - dist;
                let mu = (lam[pidx * n + (t - 1)] + rho * c).max(0.0);
                if mu > 0.0 {
                    // μ·∂c/∂pᵢ = μ·(−δ/dist)
                    let cx = mu * (-dx / dist);
                    let cy = mu * (-dy / dist);
                    for col in 0..m {
                        let val = cx * self.d[(r0, col)] + cy * self.d[(r0 + 1, col)];
                        g[i * m + col] += val; // Dₜᵀ·(μ ∂c/∂pᵢ)
                        g[j * m + col] -= val; // ∂c/∂pⱼ = −∂c/∂pᵢ
                    }
                }
            }
        }
        g
    }

    /// Game Jacobian `H = ∂g/∂U` by central finite differences (the blocks are asymmetric across
    /// players in general — this is a game, not a single optimization).
    fn jac(&self, u: &DVector<f64>, lam: &[f64], rho: f64) -> DMatrix<f64> {
        let total = self.np * self.m;
        let eps = 1e-6;
        let mut h = DMatrix::zeros(total, total);
        for col in 0..total {
            let mut up = u.clone();
            let mut um = u.clone();
            up[col] += eps;
            um[col] -= eps;
            let dcol = (self.grad(&up, lam, rho) - self.grad(&um, lam, rho)) / (2.0 * eps);
            h.set_column(col, &dcol);
        }
        h
    }
}

impl AlGames {
    /// Build a problem with sensible augmented-Lagrangian defaults.
    pub fn new(players: Vec<Player>, horizon: usize, dt: f64, d_safe: f64) -> Self {
        Self {
            players,
            horizon,
            dt,
            d_safe,
            max_outer: 20,
            max_inner: 25,
            rho0: 1.0,
            rho_max: 1e7,
            rho_scale: 8.0,
            tol: 1e-3,
        }
    }

    fn pairs(&self) -> Vec<(usize, usize)> {
        let np = self.players.len();
        let mut v = Vec::new();
        for i in 0..np {
            for j in (i + 1)..np {
                v.push((i, j));
            }
        }
        v
    }

    /// Discrete double-integrator matrices `A` (4×4), `B` (4×2) for this `dt`.
    fn dynamics(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let dt = self.dt;
        let hh = 0.5 * dt * dt;
        let a = DMatrix::from_row_slice(
            4,
            4,
            &[1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        );
        let b = DMatrix::from_row_slice(4, 2, &[hh, 0.0, 0.0, hh, dt, 0.0, 0.0, dt]);
        (a, b)
    }

    /// Roll player `i`'s controls forward from its start; returns `horizon + 1` states.
    pub fn rollout(&self, i: usize, controls: &[[f64; 2]]) -> Vec<[f64; 4]> {
        let dt = self.dt;
        let mut x = self.players[i].start;
        let mut out = vec![x];
        for t in 0..self.horizon {
            let u = controls.get(t).copied().unwrap_or([0.0, 0.0]);
            let (ax, ay) = (u[0], u[1]);
            let nx = [
                x[0] + x[2] * dt + 0.5 * ax * dt * dt,
                x[1] + x[3] * dt + 0.5 * ay * dt * dt,
                x[2] + ax * dt,
                x[3] + ay * dt,
            ];
            x = nx;
            out.push(x);
        }
        out
    }

    /// Player `i`'s own tracking-plus-effort cost `Σ_{t=1..N} w‖pₜ−goal‖² + Σ_{t=0..N-1} r‖uₜ‖²`.
    pub fn player_cost(&self, i: usize, controls: &[[f64; 2]]) -> f64 {
        let p = &self.players[i];
        let states = self.rollout(i, controls);
        let mut cost = 0.0;
        for t in 1..=self.horizon {
            let dx = states[t][0] - p.goal[0];
            let dy = states[t][1] - p.goal[1];
            cost += p.w * (dx * dx + dy * dy);
        }
        for t in 0..self.horizon {
            let u = controls.get(t).copied().unwrap_or([0.0, 0.0]);
            cost += p.r * (u[0] * u[0] + u[1] * u[1]);
        }
        cost
    }

    /// Player `i`'s *augmented Lagrangian* value: its own cost plus its share of the collision
    /// penalty across every pair it participates in. This is the potential the equilibrium
    /// minimizes for player `i`, so `∇_{Uᵢ}` of it is what the solver zeroes.
    pub fn player_augmented_cost(
        &self,
        i: usize,
        controls: &[Vec<[f64; 2]>],
        multipliers: &[f64],
        rho: f64,
    ) -> f64 {
        let mut cost = self.player_cost(i, &controls[i]);
        let states: Vec<Vec<[f64; 4]>> =
            (0..self.players.len()).map(|p| self.rollout(p, &controls[p])).collect();
        for (pidx, &(a, b)) in self.pairs().iter().enumerate() {
            if a != i && b != i {
                continue;
            }
            for t in 1..=self.horizon {
                let dx = states[a][t][0] - states[b][t][0];
                let dy = states[a][t][1] - states[b][t][1];
                let dist = (dx * dx + dy * dy).sqrt().max(1e-9);
                let c = self.d_safe - dist;
                let l = multipliers[pidx * self.horizon + (t - 1)];
                let mu = (l + rho * c).max(0.0);
                cost += (mu * mu - l * l) / (2.0 * rho);
            }
        }
        cost
    }

    /// Assemble the condensed linear-algebra scratch (position-from-control map, per-player cost
    /// Hessians, linear terms, free responses).
    fn build_work(&self) -> Work {
        let np = self.players.len();
        let n = self.horizon;
        let m = 2 * n;
        let (a, b) = self.dynamics();

        // A powers 0..=n.
        let mut apow = vec![DMatrix::<f64>::identity(4, 4)];
        for _ in 0..n {
            let next = &a * apow.last().unwrap();
            apow.push(next);
        }

        // D: block (t,k) = position rows of A^{t-1-k}·B, for t=1..=n, k=0..t-1.
        let mut d = DMatrix::zeros(m, m);
        for t in 1..=n {
            for k in 0..t {
                let ab = &apow[t - 1 - k] * &b; // 4×2
                let r0 = 2 * (t - 1);
                let c0 = 2 * k;
                d[(r0, c0)] = ab[(0, 0)];
                d[(r0, c0 + 1)] = ab[(0, 1)];
                d[(r0 + 1, c0)] = ab[(1, 0)];
                d[(r0 + 1, c0 + 1)] = ab[(1, 1)];
            }
        }
        let dt_d = d.transpose() * &d; // DᵀD (m×m)

        let mut hg = Vec::with_capacity(np);
        let mut lin = Vec::with_capacity(np);
        let mut p_free = Vec::with_capacity(np);
        for pl in &self.players {
            // Free position response: positions of A^t·x₀ for t=1..=n.
            let x0 = DVector::from_row_slice(&pl.start);
            let mut pf = DVector::zeros(m);
            let mut g = DVector::zeros(m); // stacked goal
            for t in 1..=n {
                let xt = &apow[t] * &x0;
                pf[2 * (t - 1)] = xt[0];
                pf[2 * (t - 1) + 1] = xt[1];
                g[2 * (t - 1)] = pl.goal[0];
                g[2 * (t - 1) + 1] = pl.goal[1];
            }
            // ∇Jᵢ = (2w DᵀD + 2r I)·Uᵢ + 2w Dᵀ(p_free − G).
            let h = &dt_d * (2.0 * pl.w) + DMatrix::identity(m, m) * (2.0 * pl.r);
            let l = d.transpose() * (&pf - &g) * (2.0 * pl.w);
            hg.push(h);
            lin.push(l);
            p_free.push(pf);
        }

        Work { np, n, m, d, hg, lin, p_free, pairs: self.pairs(), d_safe: self.d_safe }
    }

    /// Solve the game. `initial_guess[i][t]` seeds player `i`'s controls (missing/short entries
    /// default to zero). Returns per-player trajectories and a converged flag.
    pub fn solve(&self, initial_guess: &[Vec<[f64; 2]>]) -> AlGamesResult {
        let np = self.players.len();
        let n = self.horizon;
        let m = 2 * n;
        let total = np * m;
        let work = self.build_work();
        let npairs = work.pairs.len();

        // Stack the initial controls.
        let mut u = DVector::zeros(total);
        for i in 0..np {
            for t in 0..n {
                let g = initial_guess.get(i).and_then(|s| s.get(t)).copied().unwrap_or([0.0, 0.0]);
                u[i * m + 2 * t] = g[0];
                u[i * m + 2 * t + 1] = g[1];
            }
        }

        let mut lam = vec![0.0f64; npairs * n];
        let mut rho = self.rho0;
        let mut used_lam = lam.clone();
        let mut used_rho = rho;
        let mut converged = false;
        let mut outer_iters = 0usize;
        let mut max_violation = f64::INFINITY;
        let inner_tol = 1e-7;

        for outer in 0..self.max_outer {
            outer_iters = outer + 1;
            used_lam = lam.clone();
            used_rho = rho;

            // Inner Levenberg-damped Newton on the stacked stationarity residual.
            for _ in 0..self.max_inner {
                let g = work.grad(&u, &lam, rho);
                let res = g.norm();
                if res < inner_tol {
                    break;
                }
                let h = work.jac(&u, &lam, rho);
                let mut reg = 1e-8;
                let mut stepped = false;
                for _ in 0..14 {
                    let hr = &h + DMatrix::identity(total, total) * reg;
                    if let Some(inv) = hr.try_inverse() {
                        let du = -(&inv * &g);
                        // Backtracking line search on the residual norm.
                        let mut alpha = 1.0;
                        for _ in 0..25 {
                            let un = &u + &du * alpha;
                            if work.grad(&un, &lam, rho).norm() < res {
                                u = un;
                                stepped = true;
                                break;
                            }
                            alpha *= 0.5;
                        }
                        if stepped {
                            break;
                        }
                    }
                    reg *= 10.0;
                }
                if !stepped {
                    break; // stuck for this penalty level; the outer loop will lift ρ
                }
            }

            // Evaluate constraints, update multipliers.
            max_violation = f64::NEG_INFINITY;
            for (pidx, &(i, j)) in work.pairs.iter().enumerate() {
                let pi = work.positions(&u, i);
                let pj = work.positions(&u, j);
                for t in 1..=n {
                    let r0 = 2 * (t - 1);
                    let dx = pi[r0] - pj[r0];
                    let dy = pi[r0 + 1] - pj[r0 + 1];
                    let dist = (dx * dx + dy * dy).sqrt();
                    let c = self.d_safe - dist;
                    if c > max_violation {
                        max_violation = c;
                    }
                    let idx = pidx * n + (t - 1);
                    lam[idx] = (lam[idx] + rho * c).max(0.0);
                }
            }

            if max_violation <= self.tol {
                converged = true;
                break;
            }
            rho = (rho * self.rho_scale).min(self.rho_max);
        }

        let residual = work.grad(&u, &used_lam, used_rho).norm();

        // Unpack.
        let mut controls = Vec::with_capacity(np);
        for i in 0..np {
            let ci: Vec<[f64; 2]> =
                (0..n).map(|t| [u[i * m + 2 * t], u[i * m + 2 * t + 1]]).collect();
            controls.push(ci);
        }
        let states: Vec<Vec<[f64; 4]>> =
            (0..np).map(|i| self.rollout(i, &controls[i])).collect();

        AlGamesResult {
            controls,
            states,
            converged,
            outer_iters,
            max_violation,
            residual,
            multipliers: used_lam,
            rho: used_rho,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimum inter-player distance over the whole solution (all timesteps, first pair).
    fn min_distance(res: &AlGamesResult) -> f64 {
        let mut md = f64::INFINITY;
        let (a, b) = (&res.states[0], &res.states[1]);
        for t in 0..a.len() {
            let dx = a[t][0] - b[t][0];
            let dy = a[t][1] - b[t][1];
            md = md.min((dx * dx + dy * dy).sqrt());
        }
        md
    }

    /// Deterministic LCG + Box–Muller (per repo policy — no `rand`).
    struct Lcg(u64);
    impl Lcg {
        fn u01(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.u01().max(1e-12);
            let u2 = self.u01();
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        }
    }

    fn swap_game() -> AlGames {
        // Two players swapping ends. Slightly offset lanes (±0.1 in y) seed the symmetry break so
        // the equilibrium is the clean "A passes high, B passes low" detour rather than a degenerate
        // head-on stall.
        let players = vec![
            Player { start: [-2.0, 0.1, 0.0, 0.0], goal: [2.0, 0.1], w: 1.0, r: 0.01 },
            Player { start: [2.0, -0.1, 0.0, 0.0], goal: [-2.0, -0.1], w: 1.0, r: 0.01 },
        ];
        AlGames::new(players, 15, 0.25, 0.7)
    }

    #[test]
    fn two_players_swap_and_avoid_collision() {
        let game = swap_game();
        let guess = vec![vec![[0.0, 0.0]; game.horizon]; 2];
        let res = game.solve(&guess);

        // Shapes.
        assert_eq!(res.controls.len(), 2);
        assert_eq!(res.states[0].len(), game.horizon + 1);

        // (1) Collision constraint satisfied along the whole solution.
        assert!(res.converged, "did not converge (max_violation {})", res.max_violation);
        let md = min_distance(&res);
        assert!(
            md >= game.d_safe - 0.05,
            "players got too close: min distance {md} < d_safe {}",
            game.d_safe
        );

        // (2) Each player ends near its goal (a 4 m swap; "near" = within 0.5 m).
        for i in 0..2 {
            let last = *res.states[i].last().unwrap();
            let g = game.players[i].goal;
            let e = ((last[0] - g[0]).powi(2) + (last[1] - g[1]).powi(2)).sqrt();
            assert!(e < 0.5, "player {i} ended {e} from goal (pos {last:?}, goal {g:?})");
        }

        // Sanity: the constraint is genuinely active (they actually conflicted), so this is a real
        // game, not two independent regulators.
        // If each flew straight (zero-detour), the lanes are only 0.2 apart at the crossing.
        let straight_min = 0.2;
        assert!(md > straight_min, "no avoidance happened (min dist {md})");

        // (3) Numerical Nash: player 0 cannot reduce its own cost with a feasible unilateral
        // deviation. We perturb only player 0's controls, keep player 1 fixed, and — for every
        // perturbation that stays collision-free — require the cost not to drop below equilibrium.
        let base = game.player_cost(0, &res.controls[0]);
        let fixed_b = &res.states[1];
        let mut rng = Lcg(0xC0FFEE_1234_5678);
        let mut feasible_tested = 0;
        for _ in 0..80 {
            let mut pert = res.controls[0].clone();
            for t in 0..game.horizon {
                pert[t][0] += 0.03 * rng.gauss();
                pert[t][1] += 0.03 * rng.gauss();
            }
            let sp = game.rollout(0, &pert);
            // Feasibility of the deviation against player 1's fixed trajectory.
            let mut md_dev = f64::INFINITY;
            for t in 0..sp.len() {
                let dx = sp[t][0] - fixed_b[t][0];
                let dy = sp[t][1] - fixed_b[t][1];
                md_dev = md_dev.min((dx * dx + dy * dy).sqrt());
            }
            if md_dev >= game.d_safe {
                feasible_tested += 1;
                let c = game.player_cost(0, &pert);
                assert!(
                    c >= base - 2e-3,
                    "feasible deviation reduced player-0 cost: {c} < {base}"
                );
            }
        }
        assert!(feasible_tested >= 5, "did not exercise enough feasible deviations ({feasible_tested})");
    }

    #[test]
    fn augmented_cost_matches_the_zeroed_gradient_potential() {
        // Cross-check the public AL-potential accessor against the equilibrium: it should be finite
        // and, at the solution, player 0's own tracking cost is a large fraction of it (the penalty
        // share is small once the constraint is satisfied and multipliers are moderate).
        let game = swap_game();
        let guess = vec![vec![[0.0, 0.0]; game.horizon]; 2];
        let res = game.solve(&guess);
        let al = game.player_augmented_cost(0, &res.controls, &res.multipliers, res.rho);
        let pure = game.player_cost(0, &res.controls[0]);
        assert!(al.is_finite() && pure.is_finite());
        // At a satisfied equilibrium the collision penalty share is near zero (complementarity:
        // λ>0 ⇒ c≈0 ⇒ (μ²−λ²)/2ρ ≈ 0), so the augmented cost tracks the pure cost closely.
        assert!((al - pure).abs() < 1.0, "AL/pure cost diverged: {al} vs {pure}");
    }

    #[test]
    fn solve_is_deterministic() {
        let game = swap_game();
        let guess = vec![vec![[0.0, 0.0]; game.horizon]; 2];
        let a = game.solve(&guess);
        let b = game.solve(&guess);
        assert_eq!(a.converged, b.converged);
        for i in 0..2 {
            for t in 0..game.horizon {
                assert_eq!(a.controls[i][t], b.controls[i][t], "nondeterministic solve");
            }
        }
    }

    #[test]
    fn no_conflict_players_reach_goals_independently() {
        // Well-separated parallel lanes: the collision constraint never activates, so each player
        // just tracks its own goal. Confirms the unconstrained game degenerates to independent
        // optimal control.
        let players = vec![
            Player { start: [-2.0, 2.0, 0.0, 0.0], goal: [2.0, 2.0], w: 1.0, r: 0.01 },
            Player { start: [-2.0, -2.0, 0.0, 0.0], goal: [2.0, -2.0], w: 1.0, r: 0.01 },
        ];
        let game = AlGames::new(players, 15, 0.25, 0.7);
        let guess = vec![vec![[0.0, 0.0]; game.horizon]; 2];
        let res = game.solve(&guess);
        assert!(res.converged);
        assert!(res.max_violation <= game.tol, "unexpected violation {}", res.max_violation);
        for i in 0..2 {
            let last = *res.states[i].last().unwrap();
            let g = game.players[i].goal;
            let e = ((last[0] - g[0]).powi(2) + (last[1] - g[1]).powi(2)).sqrt();
            assert!(e < 0.3, "player {i} ended {e} from goal");
        }
    }
}
