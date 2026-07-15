//! **TOPP — time-optimal path parameterization** (Bobrow; Pham & Pham's reachability form, TOPP-RA).
//! Given a *geometric path* `q(s)`, find the fastest way to traverse it subject to the robot's limits.
//! This is the missing half of trajectory generation here: `ferromotion-ruckig` solves jerk-limited
//! point-to-point motion, but says nothing about following a *given* path (e.g. one from
//! [`ferromotion_core::RrtStar`]) as fast as the hardware allows.
//!
//! The trick is the change of variable `x = ṡ²`. Along the path
//! `q̇ = q'(s)·ṡ` and `q̈ = q'(s)·s̈ + q''(s)·ṡ²`, so **every limit becomes linear in `(s̈, x)`**:
//! velocity bounds cap `x` (the maximum-velocity curve), while acceleration (and, with the same
//! structure, torque) bounds give an interval of admissible `s̈` at each `x`. TOPP-RA then computes
//! **controllable sets** backward from the goal and sweeps forward greedily; the result saturates a
//! constraint almost everywhere, which is exactly what time-optimality looks like.
//! Pure Rust → WASM-clean.

/// A geometric path sampled on a uniform `s`-grid, with its first and second derivatives w.r.t. `s`.
pub struct ToppPath {
    /// `q'(s)` at each of the `N+1` grid points.
    pub dq: Vec<Vec<f64>>,
    /// `q''(s)` at each grid point.
    pub ddq: Vec<Vec<f64>>,
    /// Grid spacing in `s`.
    pub ds: f64,
}

/// The time-optimal parameterization.
#[derive(Clone, Debug)]
pub struct ToppResult {
    /// Squared path velocity `x = ṡ²` at each grid point.
    pub x: Vec<f64>,
    /// Path acceleration `s̈` on each segment.
    pub u: Vec<f64>,
    /// Time at each grid point.
    pub time: Vec<f64>,
    pub duration: f64,
}

impl ToppPath {
    fn n(&self) -> usize {
        self.dq.len() - 1
    }

    /// Maximum-velocity curve: the largest `x = ṡ²` allowed at grid point `i` by the velocity limits
    /// (and by any acceleration row whose `q'` vanishes, which constrains `x` directly).
    fn mvc(&self, i: usize, vmax: &[f64], amax: &[f64]) -> f64 {
        let mut x = f64::INFINITY;
        for j in 0..vmax.len() {
            let (a, b) = (self.dq[i][j], self.ddq[i][j]);
            if a.abs() > 1e-9 {
                x = x.min((vmax[j] / a.abs()).powi(2));
            } else if b.abs() > 1e-9 {
                x = x.min(amax[j] / b.abs()); // |q''·x| ≤ amax when q' ≈ 0
            }
        }
        x
    }

    /// Admissible interval of path acceleration `s̈` at grid point `i` given `x`, from
    /// `|q'·s̈ + q''·x| ≤ amax`. `None` if infeasible at this `x`.
    fn u_bounds(&self, i: usize, x: f64, amax: &[f64]) -> Option<(f64, f64)> {
        let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
        for j in 0..amax.len() {
            let a = self.dq[i][j];
            let b = self.ddq[i][j] * x;
            if a.abs() > 1e-9 {
                let (mut l, mut h) = ((-amax[j] - b) / a, (amax[j] - b) / a);
                if l > h {
                    std::mem::swap(&mut l, &mut h);
                }
                lo = lo.max(l);
                hi = hi.min(h);
            } else if b.abs() > amax[j] + 1e-9 {
                return None; // x alone already violates this row
            }
        }
        if lo > hi { None } else { Some((lo, hi)) }
    }
}

/// Time-optimally parameterize `path` under joint velocity and acceleration limits, starting and
/// ending at rest. Returns `None` if the path is infeasible.
pub fn topp(path: &ToppPath, vmax: &[f64], amax: &[f64]) -> Option<ToppResult> {
    let (n, ds) = (path.n(), path.ds);

    // ---- backward pass: controllable sets K_i = [0, kmax_i] ----
    let mut kmax = vec![0.0f64; n + 1];
    kmax[n] = 0.0; // end at rest
    for i in (0..n).rev() {
        let ceiling = path.mvc(i, vmax, amax);
        // Feasible ⇔ from x we can decelerate hard enough to land inside K_{i+1}.
        let feasible = |x: f64| -> bool {
            match path.u_bounds(i, x, amax) {
                None => false,
                Some((umin, _)) => x + 2.0 * umin * ds <= kmax[i + 1] + 1e-12,
            }
        };
        if !feasible(0.0) {
            return None; // path not traversable
        }
        kmax[i] = if feasible(ceiling) {
            ceiling
        } else {
            // Feasibility is monotone in x — bisect for the largest admissible x.
            let (mut lo, mut hi) = (0.0, ceiling);
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if feasible(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
    }

    // ---- forward pass: accelerate as hard as the controllable sets allow ----
    let mut x = vec![0.0f64; n + 1];
    let mut u = vec![0.0f64; n];
    x[0] = 0.0; // start at rest
    for i in 0..n {
        let (_, umax_i) = path.u_bounds(i, x[i], amax)?;
        let x_free = x[i] + 2.0 * umax_i * ds;
        x[i + 1] = x_free.min(kmax[i + 1]).max(0.0);
        u[i] = (x[i + 1] - x[i]) / (2.0 * ds); // the s̈ actually taken on this segment
    }

    // ---- time: Δt = 2Δs / (√x_i + √x_{i+1}) ----
    let mut time = vec![0.0f64; n + 1];
    for i in 0..n {
        let denom = x[i].max(0.0).sqrt() + x[i + 1].max(0.0).sqrt();
        if denom < 1e-12 {
            return None; // stalled
        }
        time[i + 1] = time[i] + 2.0 * ds / denom;
    }
    let duration = time[n];
    Some(ToppResult { x, u, time, duration })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight-line path `q(s) = s·L` in one joint: `q' = L`, `q'' = 0`.
    fn line(l: f64, n: usize) -> ToppPath {
        ToppPath { dq: vec![vec![l]; n + 1], ddq: vec![vec![0.0]; n + 1], ds: 1.0 / n as f64 }
    }

    #[test]
    fn matches_the_analytic_trapezoidal_optimum() {
        // Distance 2, vmax 1, amax 2 → cruise is reached: T = L/vmax + vmax/amax = 2 + 0.5 = 2.5.
        let r = topp(&line(2.0, 2000), &[1.0], &[2.0]).unwrap();
        let expected = 2.0 / 1.0 + 1.0 / 2.0;
        assert!((r.duration - expected).abs() < 5e-3, "duration {} vs analytic {expected}", r.duration);
    }

    #[test]
    fn matches_the_analytic_triangular_optimum() {
        // Short move: never reaches vmax → T = 2√(L/amax).
        let (l, amax) = (0.2, 2.0);
        let r = topp(&line(l, 2000), &[1.0], &[amax]).unwrap();
        let expected = 2.0 * (l / amax).sqrt();
        assert!((r.duration - expected).abs() < 5e-3, "duration {} vs analytic {expected}", r.duration);
    }

    #[test]
    fn respects_limits_and_saturates_a_constraint_everywhere() {
        // A curved 2-joint path: q1 = s·1.5, q2 = sin(2πs)·0.4.
        let n = 800;
        let (mut dq, mut ddq) = (Vec::new(), Vec::new());
        let tau = std::f64::consts::TAU;
        for i in 0..=n {
            let s = i as f64 / n as f64;
            dq.push(vec![1.5, 0.4 * tau * (tau * s).cos()]);
            ddq.push(vec![0.0, -0.4 * tau * tau * (tau * s).sin()]);
        }
        let path = ToppPath { dq, ddq, ds: 1.0 / n as f64 };
        let (vmax, amax) = ([1.0, 1.2], [2.0, 3.0]);
        let r = topp(&path, &vmax, &amax).unwrap();

        let mut saturated = 0;
        for i in 0..n {
            let sdot = r.x[i].sqrt();
            for j in 0..2 {
                // Velocity and acceleration limits hold along the whole path.
                let qd = path.dq[i][j] * sdot;
                let qdd = path.dq[i][j] * r.u[i] + path.ddq[i][j] * r.x[i];
                assert!(qd.abs() <= vmax[j] + 1e-3, "velocity limit violated at {i}, joint {j}: {qd}");
                assert!(qdd.abs() <= amax[j] + 1e-3, "acceleration limit violated at {i}, joint {j}: {qdd}");
            }
            // Time-optimality: some constraint is active almost everywhere.
            let at_mvc = (r.x[i] - path.mvc(i, &vmax, &amax)).abs() < 1e-3;
            let acc_sat = (0..2).any(|j| (path.dq[i][j] * r.u[i] + path.ddq[i][j] * r.x[i]).abs() > amax[j] - 1e-2);
            if at_mvc || acc_sat {
                saturated += 1;
            }
        }
        let frac = saturated as f64 / n as f64;
        assert!(frac > 0.95, "only {frac:.2} of the path saturates a constraint — not time-optimal");
    }

    #[test]
    fn tighter_limits_take_longer() {
        let fast = topp(&line(1.0, 1000), &[1.0], &[2.0]).unwrap();
        let slow = topp(&line(1.0, 1000), &[0.5], &[2.0]).unwrap();
        assert!(slow.duration > fast.duration, "halving vmax must not be faster");
    }
}
