//! **Hamilton-Jacobi reachability** — formal safety/reachability verification by solving a PDE.
//!
//! Where [`crate::CbfFilter`] enforces safety through a hand-designed barrier, HJ reachability
//! *computes* the exact answer: represent the target/failure set implicitly as `{x : l(x) ≤ 0}`, then
//! the **backward reachable tube** — every state that can reach it within time `T` — is the
//! zero-sublevel set of a value function solving the HJ variational inequality
//!
//! ```text
//!   ∂V/∂τ = min(0, H(x, ∇V)),    V(x, 0) = l(x),    H(x, p) = min_u  p · f(x, u)
//! ```
//!
//! marched backward in time (`τ`). The `min(0, ·)` freezes the value once the set is reached, which is
//! what makes it a *tube* rather than a slice. `BRT(T) = {x : V(x, T) ≤ 0}`. The Hamiltonian is
//! discretized with the standard **Lax-Friedrichs** scheme. Pure Rust → WASM-clean.

/// A uniform Cartesian grid over an axis-aligned box.
#[derive(Clone, Debug)]
pub struct HjGrid {
    pub dims: Vec<usize>,
    pub lo: Vec<f64>,
    pub dx: Vec<f64>,
}

impl HjGrid {
    /// Grid spanning `[lo, hi]` with `dims` points per axis.
    pub fn new(lo: Vec<f64>, hi: Vec<f64>, dims: Vec<usize>) -> Self {
        let dx = (0..dims.len()).map(|i| (hi[i] - lo[i]) / (dims[i] - 1) as f64).collect();
        Self { dims, lo, dx }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    /// Multi-index → flat index (row-major).
    pub fn flat(&self, idx: &[usize]) -> usize {
        let mut f = 0;
        for i in 0..self.ndim() {
            f = f * self.dims[i] + idx[i];
        }
        f
    }

    /// Flat index → multi-index.
    pub fn multi(&self, mut f: usize) -> Vec<usize> {
        let mut m = vec![0; self.ndim()];
        for i in (0..self.ndim()).rev() {
            m[i] = f % self.dims[i];
            f /= self.dims[i];
        }
        m
    }

    /// Multi-index → state coordinates.
    pub fn coord(&self, idx: &[usize]) -> Vec<f64> {
        (0..self.ndim()).map(|i| self.lo[i] + idx[i] as f64 * self.dx[i]).collect()
    }
}

/// Solve for the backward reachable tube's value function, from initial values `l0` on `grid`.
///
/// * `h(x, p)` — the Hamiltonian `min_u p·f(x, u)`.
/// * `alpha(x)` — per-axis dissipation `max |∂H/∂p_i|` (the Lax-Friedrichs coefficients).
///
/// Returns `V(·, t_final)`; the BRT is `{x : V(x) ≤ 0}`.
pub fn solve_brt<H, A>(grid: &HjGrid, l0: &[f64], h: H, alpha: A, t_final: f64, dt: f64) -> Vec<f64>
where
    H: Fn(&[f64], &[f64]) -> f64,
    A: Fn(&[f64]) -> Vec<f64>,
{
    let (n, nd) = (grid.size(), grid.ndim());
    let mut v = l0.to_vec();
    let steps = (t_final / dt).ceil() as usize;
    let mut next = v.clone();

    for _ in 0..steps {
        for f in 0..n {
            let idx = grid.multi(f);
            let x = grid.coord(&idx);
            let (mut pm, mut pp) = (vec![0.0; nd], vec![0.0; nd]);
            for d in 0..nd {
                // One-sided differences, clamped at the boundary (Neumann-like).
                let mut lo_i = idx.clone();
                let mut hi_i = idx.clone();
                lo_i[d] = idx[d].saturating_sub(1);
                hi_i[d] = (idx[d] + 1).min(grid.dims[d] - 1);
                pm[d] = (v[f] - v[grid.flat(&lo_i)]) / grid.dx[d];
                pp[d] = (v[grid.flat(&hi_i)] - v[f]) / grid.dx[d];
            }
            // Lax-Friedrichs. The PDE is ∂V/∂τ + H̃(∇V) = 0 with H̃ = −min(0, H), so the LF viscosity
            // attaches to H̃ — i.e. it is *added* here. (Folding it inside the min(0,·) flips its sign
            // into anti-diffusion and the scheme blows up.)
            let p_avg: Vec<f64> = (0..nd).map(|d| 0.5 * (pm[d] + pp[d])).collect();
            let a = alpha(&x);
            let hval = h(&x, &p_avg).min(0.0); // the freeze: the tube only grows
            let diss: f64 = (0..nd).map(|d| 0.5 * a[d] * (pp[d] - pm[d])).sum();
            // Clamp to non-increasing: the tube's value can never rise (keeps the set monotone even
            // where the numerical viscosity would nudge it up at a kink).
            next[f] = (v[f] + dt * (hval + diss)).min(v[f]);
        }
        std::mem::swap(&mut v, &mut next);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_integrator_brt_matches_the_analytic_set() {
        // ẋ = u, |u| ≤ 1, target {|x| ≤ r}. Anything within r + T can reach it ⇒ BRT(T) = {|x| ≤ r+T}.
        let (r, t_final) = (0.2, 1.0);
        let grid = HjGrid::new(vec![-3.0], vec![3.0], vec![1201]);
        let l0: Vec<f64> = (0..grid.size()).map(|f| grid.coord(&grid.multi(f))[0].abs() - r).collect();
        // H(x,p) = min_{|u|≤1} p·u = −|p|.
        let v = solve_brt(&grid, &l0, |_x, p: &[f64]| -p[0].abs(), |_x| vec![1.0], t_final, 0.4 * grid.dx[0]);

        // Find the zero crossing on the positive side.
        let mut boundary = f64::NAN;
        for f in 0..grid.size() - 1 {
            let x = grid.coord(&grid.multi(f))[0];
            if x > 0.0 && v[f] <= 0.0 && v[f + 1] > 0.0 {
                boundary = x;
                break;
            }
        }
        let expected = r + t_final;
        assert!((boundary - expected).abs() < 0.02, "BRT boundary {boundary} vs analytic {expected}");
    }

    #[test]
    fn the_tube_only_grows_with_time() {
        let grid = HjGrid::new(vec![-3.0], vec![3.0], vec![601]);
        let l0: Vec<f64> = (0..grid.size()).map(|f| grid.coord(&grid.multi(f))[0].abs() - 0.2).collect();
        let dt = 0.4 * grid.dx[0];
        let short = solve_brt(&grid, &l0, |_x, p: &[f64]| -p[0].abs(), |_x| vec![1.0], 0.5, dt);
        let long = solve_brt(&grid, &l0, |_x, p: &[f64]| -p[0].abs(), |_x| vec![1.0], 1.0, dt);
        // More time can only lower the value ⇒ BRT(0.5) ⊆ BRT(1.0).
        for f in 0..grid.size() {
            assert!(long[f] <= short[f] + 1e-9, "value increased with time at {f}");
            assert!(short[f] <= l0[f] + 1e-9, "value increased above l0");
        }
    }

    #[test]
    fn double_integrator_brt_matches_the_bang_bang_reach_extent() {
        // ẋ₁ = x₂, ẋ₂ = u, |u| ≤ 1. From (x₁, 0) the minimum time to the origin is 2√|x₁|
        // (accelerate then brake) ⇒ at x₂ = 0 the BRT(T) reaches |x₁| ≈ T²/4.
        let (t_final, r) = (1.0, 0.05);
        let grid = HjGrid::new(vec![-1.0, -1.5], vec![1.0, 1.5], vec![101, 151]);
        let l0: Vec<f64> = (0..grid.size())
            .map(|f| {
                let x = grid.coord(&grid.multi(f));
                (x[0] * x[0] + x[1] * x[1]).sqrt() - r
            })
            .collect();
        // H(x,p) = min_{|u|≤1} (p₁·x₂ + p₂·u) = p₁·x₂ − |p₂|.
        let h = |x: &[f64], p: &[f64]| p[0] * x[1] - p[1].abs();
        let alpha = |x: &[f64]| vec![x[1].abs(), 1.0];
        let dt = 0.25 / (1.5 / grid.dx[0] + 1.0 / grid.dx[1]); // CFL
        let v = solve_brt(&grid, &l0, h, alpha, t_final, dt);

        // Walk along x₂ = 0 and find where the tube ends on the +x₁ side.
        let j0 = ((0.0 - grid.lo[1]) / grid.dx[1]).round() as usize;
        let mut extent = 0.0f64;
        for i in 0..grid.dims[0] {
            let x = grid.coord(&[i, j0]);
            if x[0] > 0.0 && v[grid.flat(&[i, j0])] <= 0.0 {
                extent = extent.max(x[0]);
            }
        }
        let analytic = t_final * t_final / 4.0; // = 0.25
        eprintln!("double-integrator BRT extent at x₂=0: {extent:.3} (analytic {analytic:.3} + target r={r})");
        assert!((extent - analytic).abs() < 0.08, "BRT extent {extent} vs analytic bang-bang {analytic}");
    }
}
