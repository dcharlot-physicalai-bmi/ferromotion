//! **Time-optimal lab** — the rig behind the textbook chapter on traversing a path as fast as physics
//! allows.
//!
//! You are handed a *shape* to follow — a path `q(s)` through the plane — and asked to go along it in
//! the least time your motors permit. The answer is never "constant speed": you must crawl through the
//! tight corners and floor it on the straights, and the exact schedule is a bang-bang law that is
//! almost always pinned against some limit. TOPP finds it by the change of variable `x = ṡ²`, which
//! makes every velocity and acceleration bound *linear* in `x` and the path acceleration `s̈`, so the
//! optimum can be computed by a backward reachability sweep and a greedy forward pass.
//!
//! The rig builds a 2-D path, runs the real [`ferromotion_control::topp`], and exposes the speed
//! profile against the maximum-velocity ceiling so the reader can watch the optimum ride the limits.

use ferromotion_control::{topp, ToppPath, ToppResult};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ToppLab {
    xs: Vec<f64>,
    ys: Vec<f64>,
    dq: Vec<Vec<f64>>,
    ddq: Vec<Vec<f64>>,
    ds: f64,
    res: Option<ToppResult>,
    vmax: Vec<f64>,
    amax: Vec<f64>,
}

#[wasm_bindgen]
impl ToppLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ToppLab {
        ToppLab { xs: vec![], ys: vec![], dq: vec![], ddq: vec![], ds: 1.0, res: None, vmax: vec![1.0, 1.0], amax: vec![1.0, 1.0] }
    }

    /// Set the 2-D geometric path from sampled points and precompute `q'`, `q''` by finite differences.
    pub fn set_path(&mut self, xs: &[f64], ys: &[f64]) {
        let n = xs.len();
        self.xs = xs.to_vec();
        self.ys = ys.to_vec();
        self.ds = 1.0 / (n - 1) as f64;
        let ds = self.ds;
        let mut dq = vec![vec![0.0; 2]; n];
        let mut ddq = vec![vec![0.0; 2]; n];
        for i in 0..n {
            let (im, ip) = (i.saturating_sub(1), (i + 1).min(n - 1));
            dq[i][0] = (xs[ip] - xs[im]) / ((ip - im) as f64 * ds);
            dq[i][1] = (ys[ip] - ys[im]) / ((ip - im) as f64 * ds);
        }
        for i in 0..n {
            let (im, ip) = (i.saturating_sub(1), (i + 1).min(n - 1));
            ddq[i][0] = (dq[ip][0] - dq[im][0]) / ((ip - im) as f64 * ds);
            ddq[i][1] = (dq[ip][1] - dq[im][1]) / ((ip - im) as f64 * ds);
        }
        self.dq = dq;
        self.ddq = ddq;
    }

    fn path(&self) -> ToppPath {
        ToppPath { dq: self.dq.clone(), ddq: self.ddq.clone(), ds: self.ds }
    }

    /// Run TOPP under per-axis velocity/acceleration limits. Returns whether the path is feasible.
    pub fn solve(&mut self, vx: f64, vy: f64, ax: f64, ay: f64) -> bool {
        self.vmax = vec![vx, vy];
        self.amax = vec![ax, ay];
        self.res = topp(&self.path(), &self.vmax, &self.amax);
        self.res.is_some()
    }

    /// The maximum-velocity ceiling in path speed `ṡ = √x` at each grid point (the fastest you could
    /// ever go there without exceeding a velocity limit or an acceleration limit through curvature).
    pub fn mvc_speed(&self) -> Vec<f64> {
        let n = self.xs.len();
        (0..n)
            .map(|i| {
                let mut x = f64::INFINITY;
                for j in 0..2 {
                    let (a, b) = (self.dq[i][j], self.ddq[i][j]);
                    if a.abs() > 1e-9 {
                        x = x.min((self.vmax[j] / a.abs()).powi(2));
                    } else if b.abs() > 1e-9 {
                        x = x.min(self.amax[j] / b.abs());
                    }
                }
                x.max(0.0).sqrt()
            })
            .collect()
    }

    /// The time-optimal path speed `ṡ = √x` at each grid point (rides the ceiling where it can).
    pub fn speed(&self) -> Vec<f64> {
        match &self.res {
            Some(r) => r.x.iter().map(|&x| x.max(0.0).sqrt()).collect(),
            None => vec![],
        }
    }

    pub fn duration(&self) -> f64 {
        self.res.as_ref().map(|r| r.duration).unwrap_or(f64::NAN)
    }

    /// Fraction of segments where the path acceleration is pinned to a limit **or** the speed is on the
    /// ceiling — the bang-bang signature of time-optimality (should be ≈1).
    pub fn saturated_fraction(&self) -> f64 {
        let Some(r) = &self.res else { return 0.0 };
        let n = r.u.len();
        let mvc = self.mvc_speed();
        let mut sat = 0;
        for i in 0..n {
            let (umin, umax) = self.u_bounds(i, r.x[i]);
            let at_accel = (r.u[i] - umax).abs() < 1e-3 * (1.0 + umax.abs()) || (r.u[i] - umin).abs() < 1e-3 * (1.0 + umin.abs());
            let on_ceiling = (r.x[i].max(0.0).sqrt() - mvc[i]).abs() < 1e-2 * (1.0 + mvc[i]);
            if at_accel || on_ceiling {
                sat += 1;
            }
        }
        sat as f64 / n as f64
    }

    fn u_bounds(&self, i: usize, x: f64) -> (f64, f64) {
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for j in 0..2 {
            let a = self.dq[i][j];
            let b = self.ddq[i][j] * x;
            if a.abs() > 1e-9 {
                let (mut l, mut h) = ((-self.amax[j] - b) / a, (self.amax[j] - b) / a);
                if l > h {
                    std::mem::swap(&mut l, &mut h);
                }
                lo = lo.max(l);
                hi = hi.min(h);
            }
        }
        (lo, hi)
    }

    /// The time an unvarying-speed planner would take: it must pick the single path speed slow enough
    /// for the *tightest* point, so its time is `1 / min(ceiling)`. TOPP beats this by speeding up
    /// wherever the path allows.
    pub fn naive_duration(&self) -> f64 {
        let mvc = self.mvc_speed();
        let slowest = mvc.iter().cloned().fold(f64::INFINITY, f64::min);
        if slowest < 1e-9 { f64::INFINITY } else { 1.0 / slowest }
    }

    /// Position `(x, y)` on the path at time `t` under the optimal schedule (for the moving dot).
    pub fn pos_at_time(&self, t: f64) -> Vec<f64> {
        let Some(r) = &self.res else { return vec![self.xs.first().copied().unwrap_or(0.0), self.ys.first().copied().unwrap_or(0.0)] };
        let n = self.xs.len();
        // find the segment containing t
        let mut i = 0;
        while i + 1 < n && r.time[i + 1] < t {
            i += 1;
        }
        if i + 1 >= n {
            return vec![self.xs[n - 1], self.ys[n - 1]];
        }
        let f = ((t - r.time[i]) / (r.time[i + 1] - r.time[i]).max(1e-12)).clamp(0.0, 1.0);
        vec![self.xs[i] + f * (self.xs[i + 1] - self.xs[i]), self.ys[i] + f * (self.ys[i + 1] - self.ys[i])]
    }

    pub fn path_xy(&self) -> Vec<f64> {
        let mut o = Vec::with_capacity(2 * self.xs.len());
        for i in 0..self.xs.len() {
            o.push(self.xs[i]);
            o.push(self.ys[i]);
        }
        o
    }
    pub fn n(&self) -> usize {
        self.xs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(l: f64, n: usize) -> ToppLab {
        let mut lab = ToppLab::new();
        let xs: Vec<f64> = (0..=n).map(|i| l * i as f64 / n as f64).collect();
        let ys = vec![0.0; n + 1];
        lab.set_path(&xs, &ys);
        lab
    }

    /// A path with a sharp corner: right along x, then a tight quarter turn upward.
    fn corner(n: usize) -> ToppLab {
        let mut lab = ToppLab::new();
        let (mut xs, mut ys) = (vec![], vec![]);
        for i in 0..=n {
            let s = i as f64 / n as f64;
            if s < 0.5 {
                xs.push(2.0 * s * 2.0); // 0 → 2
                ys.push(0.0);
            } else {
                let a = (s - 0.5) * 2.0 * std::f64::consts::FRAC_PI_2;
                xs.push(2.0 + 0.5 * a.sin()); // small-radius turn
                ys.push(0.5 * (1.0 - a.cos()));
            }
        }
        lab.set_path(&xs, &ys);
        lab
    }

    #[test]
    fn a_straight_move_matches_the_analytic_trapezoidal_optimum() {
        // THE CHECK against ground truth. Distance 2, vmax 1, amax 2 → T = L/v + v/a = 2.5.
        let mut lab = straight(2.0, 2000);
        assert!(lab.solve(1.0, 1.0, 2.0, 2.0));
        let expected = 2.0 / 1.0 + 1.0 / 2.0;
        assert!((lab.duration() - expected).abs() < 5e-3, "duration {} vs analytic {expected}", lab.duration());
    }

    #[test]
    fn a_short_move_matches_the_analytic_triangular_optimum() {
        let (l, amax) = (0.2, 2.0);
        let mut lab = straight(l, 2000);
        assert!(lab.solve(1.0, 1.0, amax, amax));
        let expected = 2.0 * (l / amax).sqrt();
        assert!((lab.duration() - expected).abs() < 5e-3, "duration {} vs analytic {expected}", lab.duration());
    }

    #[test]
    fn the_optimum_is_bang_bang_almost_everywhere() {
        // THE CHAPTER. Time-optimality means always pinned to a limit — accelerating flat out, braking
        // flat out, or riding the velocity ceiling. On a curved path that should hold nearly everywhere.
        let mut lab = corner(1200);
        assert!(lab.solve(1.2, 1.2, 2.5, 2.5));
        let frac = lab.saturated_fraction();
        assert!(frac > 0.95, "time-optimal profile should be saturated almost everywhere: {frac}");
    }

    #[test]
    fn varying_the_speed_beats_a_single_safe_speed() {
        // TOPP must be at least as fast as the unvarying-speed planner, and strictly faster when the
        // path has an easy stretch the constant-speed planner is forced to crawl through.
        let mut lab = corner(1200);
        assert!(lab.solve(1.2, 1.2, 2.5, 2.5));
        assert!(lab.duration() < lab.naive_duration(), "TOPP {} should beat constant-speed {}", lab.duration(), lab.naive_duration());
    }

    #[test]
    fn tighter_limits_take_longer() {
        let dur = |a: f64| {
            let mut lab = corner(1000);
            lab.solve(1.5, 1.5, a, a);
            lab.duration()
        };
        assert!(dur(1.0) > dur(4.0), "less acceleration should take longer: {} vs {}", dur(1.0), dur(4.0));
    }
}
