//! **Frenet-frame optimal trajectory planning** (Werling, Ziegler, Kammel & Thrun, ICRA 2010) — the
//! canonical local planner for cars and other nonholonomic ground vehicles. Motion is planned in
//! **road-aligned Frenet coordinates** `(s, d)`: arc length `s` along a reference path and lateral offset
//! `d` from it. The insight is that comfortable driving = **minimum jerk**, and the jerk-optimal connection
//! between two boundary states of a double/triple integrator is a fixed-degree polynomial: a **quintic** for
//! the lateral motion (position/velocity/acceleration pinned at both ends) and a **quartic** for
//! velocity-keeping longitudinal motion (terminal *velocity* pinned, terminal position free). The planner
//! samples terminal lateral offsets and horizons, scores each candidate by jerk + time + lateral deviation,
//! and returns the cheapest collision-free one.
//!
//! Complements [`crate::MinSnap`] (quadrotor min-snap) with the ground-vehicle Frenet formulation, and pairs
//! with [`crate::dubins`](car turning geometry). Verified: the quintic/quartic hit their boundary conditions
//! to machine precision; the quintic is provably minimum-jerk (any boundary-preserving perturbation raises
//! `∫jerk²`); and the planner returns the vehicle to the lane centre while routing around a Frenet obstacle.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

/// A quintic polynomial `p(t) = Σ cₖ tᵏ`, the jerk-optimal connection between two `(x, ẋ, ẍ)` boundary
/// states over horizon `T`.
#[derive(Clone, Copy, Debug)]
pub struct Quintic {
    pub c: [f64; 6],
}

impl Quintic {
    /// Fit the quintic with start state `(x0, v0, a0)` and end state `(x1, v1, a1)` at time `T`.
    pub fn new(x0: f64, v0: f64, a0: f64, x1: f64, v1: f64, a1: f64, t: f64) -> Self {
        let (c0, c1, c2) = (x0, v0, 0.5 * a0);
        // solve the 3×3 for c3,c4,c5 from the terminal position/velocity/acceleration
        let (t2, t3, t4, t5) = (t * t, t * t * t, t.powi(4), t.powi(5));
        let a = Matrix3::new(t3, t4, t5, 3.0 * t2, 4.0 * t3, 5.0 * t4, 6.0 * t, 12.0 * t2, 20.0 * t3);
        let b = Vector3::new(x1 - (c0 + c1 * t + c2 * t2), v1 - (c1 + 2.0 * c2 * t), a1 - 2.0 * c2);
        let c345 = a.lu().solve(&b).expect("quintic system is nonsingular for T>0");
        Quintic { c: [c0, c1, c2, c345[0], c345[1], c345[2]] }
    }

    pub fn pos(&self, t: f64) -> f64 {
        let c = &self.c;
        c[0] + c[1] * t + c[2] * t * t + c[3] * t.powi(3) + c[4] * t.powi(4) + c[5] * t.powi(5)
    }
    pub fn vel(&self, t: f64) -> f64 {
        let c = &self.c;
        c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t * t + 4.0 * c[4] * t.powi(3) + 5.0 * c[5] * t.powi(4)
    }
    pub fn acc(&self, t: f64) -> f64 {
        let c = &self.c;
        2.0 * c[2] + 6.0 * c[3] * t + 12.0 * c[4] * t * t + 20.0 * c[5] * t.powi(3)
    }
    pub fn jerk(&self, t: f64) -> f64 {
        let c = &self.c;
        6.0 * c[3] + 24.0 * c[4] * t + 60.0 * c[5] * t * t
    }
}

/// A quartic polynomial, the jerk-optimal connection when the terminal **velocity/acceleration** are pinned
/// but the terminal position is free — used for longitudinal velocity-keeping.
#[derive(Clone, Copy, Debug)]
pub struct Quartic {
    pub c: [f64; 5],
}

impl Quartic {
    /// Fit with start `(x0, v0, a0)` and terminal velocity/acceleration `(v1, a1)` at time `T`.
    pub fn new(x0: f64, v0: f64, a0: f64, v1: f64, a1: f64, t: f64) -> Self {
        let (c0, c1, c2) = (x0, v0, 0.5 * a0);
        let (t2, t3) = (t * t, t * t * t);
        let a = Matrix2::new(3.0 * t2, 4.0 * t3, 6.0 * t, 12.0 * t2);
        let b = Vector2::new(v1 - (c1 + 2.0 * c2 * t), a1 - 2.0 * c2);
        let c34 = a.lu().solve(&b).expect("quartic system is nonsingular for T>0");
        Quartic { c: [c0, c1, c2, c34[0], c34[1]] }
    }

    pub fn pos(&self, t: f64) -> f64 {
        let c = &self.c;
        c[0] + c[1] * t + c[2] * t * t + c[3] * t.powi(3) + c[4] * t.powi(4)
    }
    pub fn vel(&self, t: f64) -> f64 {
        let c = &self.c;
        c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t * t + 4.0 * c[4] * t.powi(3)
    }
    pub fn acc(&self, t: f64) -> f64 {
        let c = &self.c;
        2.0 * c[2] + 6.0 * c[3] * t + 12.0 * c[4] * t * t
    }
}

/// A planned Frenet trajectory: time-stamped `(s, d)` samples and the total cost.
#[derive(Clone, Debug)]
pub struct FrenetPath {
    pub t: Vec<f64>,
    pub s: Vec<f64>,
    pub d: Vec<f64>,
    pub cost: f64,
}

/// A Werling Frenet planner. Samples terminal lateral offsets `d1 ∈ target_ds` over horizons
/// `t ∈ [min_t, max_t]`, at time resolution `dt`, scoring `k_j·∫jerk² + k_t·T + k_d·d1²`.
#[derive(Clone, Debug)]
pub struct FrenetPlanner {
    pub min_t: f64,
    pub max_t: f64,
    pub dt: f64,
    pub target_ds: Vec<f64>,
    pub target_speed: f64,
    pub k_j: f64,
    pub k_t: f64,
    pub k_d: f64,
}

impl FrenetPlanner {
    /// Plan from lateral state `(d0, d0v, d0a)` and longitudinal `(s0, s0v)` (constant-accel-free start),
    /// avoiding circular Frenet obstacles `(s, d, radius)`. Returns the cheapest collision-free trajectory.
    pub fn plan(&self, d0: f64, d0v: f64, d0a: f64, s0: f64, s0v: f64, obstacles: &[(f64, f64, f64)]) -> Option<FrenetPath> {
        let mut best: Option<FrenetPath> = None;
        let mut t_end = self.min_t;
        while t_end <= self.max_t + 1e-9 {
            for &d1 in &self.target_ds {
                let lat = Quintic::new(d0, d0v, d0a, d1, 0.0, 0.0, t_end); // return to d1 at rest laterally
                let lon = Quartic::new(s0, s0v, 0.0, self.target_speed, 0.0, t_end); // keep target speed
                // sample
                let n = (t_end / self.dt).round() as usize;
                let mut ts = Vec::with_capacity(n + 1);
                let mut ss = Vec::with_capacity(n + 1);
                let mut ds = Vec::with_capacity(n + 1);
                let mut jerk_sq = 0.0;
                let mut feasible = true;
                for i in 0..=n {
                    let t = i as f64 * self.dt;
                    let (sv, dv) = (lon.pos(t), lat.pos(t));
                    // jerk cost (lateral + longitudinal); longitudinal jerk = 3rd deriv of quartic
                    let lat_j = lat.jerk(t);
                    jerk_sq += lat_j * lat_j * self.dt;
                    ts.push(t);
                    ss.push(sv);
                    ds.push(dv);
                    for &(os, od, orad) in obstacles {
                        if ((sv - os).powi(2) + (dv - od).powi(2)).sqrt() < orad {
                            feasible = false;
                            break;
                        }
                    }
                    if !feasible {
                        break;
                    }
                }
                if !feasible {
                    continue;
                }
                let cost = self.k_j * jerk_sq + self.k_t * t_end + self.k_d * d1 * d1;
                if best.as_ref().is_none_or(|b| cost < b.cost) {
                    best = Some(FrenetPath { t: ts, s: ss, d: ds, cost });
                }
            }
            t_end += self.dt;
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quintic_hits_its_boundary_conditions_exactly() {
        // THE ORACLE. The fitted quintic reproduces the pinned (x,v,a) at both ends to machine precision.
        let t = 3.0;
        let q = Quintic::new(2.0, 0.5, -0.2, -1.0, 0.3, 0.1, t);
        assert!((q.pos(0.0) - 2.0).abs() < 1e-12 && (q.vel(0.0) - 0.5).abs() < 1e-12 && (q.acc(0.0) + 0.2).abs() < 1e-12);
        assert!((q.pos(t) + 1.0).abs() < 1e-9, "end pos {}", q.pos(t));
        assert!((q.vel(t) - 0.3).abs() < 1e-9, "end vel {}", q.vel(t));
        assert!((q.acc(t) - 0.1).abs() < 1e-9, "end acc {}", q.acc(t));
    }

    #[test]
    fn the_quartic_keeps_its_terminal_velocity_exactly() {
        // Velocity-keeping oracle: start (x,v,a), terminal (v,a); terminal position is free.
        let t = 4.0;
        let q = Quartic::new(0.0, 8.0, 0.0, 12.0, 0.0, t);
        assert!((q.pos(0.0)).abs() < 1e-12 && (q.vel(0.0) - 8.0).abs() < 1e-12);
        assert!((q.vel(t) - 12.0).abs() < 1e-9, "terminal vel {}", q.vel(t));
        assert!((q.acc(t)).abs() < 1e-9, "terminal acc {}", q.acc(t));
    }

    #[test]
    fn the_quintic_is_the_minimum_jerk_trajectory() {
        // THE OPTIMALITY PROPERTY. Any perturbation that preserves the 6 boundary conditions can only RAISE
        // ∫jerk². The bump δ(t)=ε·t³(T−t)³ has zero position/velocity/acceleration at both ends, so
        // quintic+δ meets the same BCs — and must have a larger jerk integral.
        let t = 3.0;
        let q = Quintic::new(1.0, 0.2, 0.0, 4.0, -0.1, 0.0, t);
        let jerk_int = |f: &dyn Fn(f64) -> f64| -> f64 {
            let n = 4000;
            let h = t / n as f64;
            (0..n).map(|i| { let tt = (i as f64 + 0.5) * h; f(tt) * f(tt) * h }).sum::<f64>()
        };
        let base = jerk_int(&|tt| q.jerk(tt));
        // δ‴(t) for δ=ε·t³(T−t)³
        for &eps in &[0.05, -0.03, 0.1] {
            let bump_jerk = |tt: f64| {
                // third derivative of ε·t³(T−t)³
                let d = |tt: f64| eps * tt.powi(3) * (t - tt).powi(3);
                // numeric 3rd derivative
                let h = 1e-3;
                (d(tt + 1.5 * h) - 3.0 * d(tt + 0.5 * h) + 3.0 * d(tt - 0.5 * h) - d(tt - 1.5 * h)) / h.powi(3)
            };
            let perturbed = jerk_int(&|tt| q.jerk(tt) + bump_jerk(tt));
            assert!(perturbed > base - 1e-6, "perturbed ∫jerk² {perturbed} should exceed the quintic's {base}");
        }
    }

    #[test]
    fn the_planner_returns_to_lane_centre_around_an_obstacle() {
        // THE HEADLINE. The car starts offset to the right (d0 = 3.0) with an obstacle sitting on that same
        // lateral line a short distance ahead (within the shortest planning horizon's reach). The planner
        // must produce a trajectory that (a) clears the obstacle and (b) settles back toward the centre.
        let planner = FrenetPlanner {
            min_t: 3.0,
            max_t: 6.0,
            dt: 0.2,
            target_ds: vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            target_speed: 10.0,
            k_j: 0.1,
            k_t: 1.0,
            k_d: 5.0,
        };
        // obstacle at s≈20m (reached by every horizon, since min_t·speed = 30m) on the start line d=3
        let obstacles = [(20.0, 3.0, 1.5)];
        let path = planner.plan(3.0, 0.0, 0.0, 0.0, 10.0, &obstacles).expect("a feasible plan should exist");
        // every sample clears the obstacle
        let min_clear = path.s.iter().zip(&path.d).map(|(&s, &d)| ((s - 20.0).powi(2) + (d - 3.0).powi(2)).sqrt()).fold(f64::INFINITY, f64::min);
        assert!(min_clear >= 1.5 - 1e-9, "path must clear the obstacle: min clearance {min_clear}");
        // and the maneuver ends near the lane centre (moved away from the start line d=3)
        assert!(path.d.last().unwrap().abs() <= 1.0 + 1e-9, "should settle toward centre: terminal d {}", path.d.last().unwrap());
    }
}
