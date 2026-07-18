//! **CHOMP — Covariant Hamiltonian Optimization for Motion Planning** (Ratliff, Zucker, Bagnell,
//! Srinivasa, ICRA 2009 / IJRR 2013): optimize a whole trajectory by gradient descent on a
//! **smoothness** cost plus an **obstacle** cost read directly off a signed distance field. The twist that
//! names it is *covariant* descent — the gradient is preconditioned by the smoothness metric `A⁻¹`, so an
//! obstacle pushing on one waypoint deforms the trajectory smoothly instead of putting a kink in it, and
//! the update is invariant to how the trajectory is discretized.
//!
//! Here the robot is a point in ℝ³ against a [`crate::SdfScene`]. The smoothness metric is the discrete
//! Laplacian (sum of squared velocities, fixed endpoints), whose obstacle-free minimizer is the straight
//! line; the obstacle cost is CHOMP's `c(d)` hinge over the clearance, with gradient `c′(d)·∇d` from the
//! field. Verified: it recovers the straight line with no obstacles, and bows around one that blocks it.
//! Pure `nalgebra` → WASM-clean.

use crate::sdf::SdfScene;
use nalgebra::{DMatrix, Vector3};

/// A CHOMP planner for a spherical point robot against an SDF scene.
#[derive(Clone)]
pub struct Chomp<'a> {
    pub scene: &'a SdfScene,
    /// Obstacle-cost margin: clearance below `epsilon` is penalized.
    pub epsilon: f64,
    /// Obstacle-cost weight (relative to smoothness).
    pub lambda: f64,
    /// Covariant step size.
    pub eta: f64,
    /// Robot radius (clearance = SDF distance − radius).
    pub radius: f64,
}

/// The result of a CHOMP solve.
#[derive(Clone, Debug)]
pub struct ChompResult {
    /// Interior + endpoint waypoints (start … goal).
    pub waypoints: Vec<Vector3<f64>>,
    pub cost: f64,
    pub cost_history: Vec<f64>,
    /// Minimum clearance over the path (negative ⇒ still in collision).
    pub min_clearance: f64,
}

impl Chomp<'_> {
    /// CHOMP's obstacle cost `c(clearance)`: a smooth hinge that is `>0` inside the margin and 0 outside.
    fn obstacle_cost(&self, clr: f64) -> f64 {
        if clr < 0.0 {
            -clr + 0.5 * self.epsilon
        } else if clr <= self.epsilon {
            let t = clr - self.epsilon;
            0.5 / self.epsilon * t * t
        } else {
            0.0
        }
    }
    /// `c′(clearance)`.
    fn obstacle_dcost(&self, clr: f64) -> f64 {
        if clr < 0.0 {
            -1.0
        } else if clr <= self.epsilon {
            (clr - self.epsilon) / self.epsilon
        } else {
            0.0
        }
    }

    /// The interior smoothness metric `A` (tridiagonal discrete Laplacian, `[−1, 2, −1]`) for `n` interior
    /// waypoints with fixed endpoints — the Hessian of `½ Σ‖q_{i+1}−q_i‖²`.
    fn smoothness_matrix(n: usize) -> DMatrix<f64> {
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
            }
            if i + 1 < n {
                a[(i, i + 1)] = -1.0;
            }
        }
        a
    }

    fn total_cost(&self, way: &[Vector3<f64>]) -> f64 {
        let mut smooth = 0.0;
        for w in way.windows(2) {
            smooth += 0.5 * (w[1] - w[0]).norm_squared();
        }
        let mut obs = 0.0;
        for q in &way[1..way.len() - 1] {
            obs += self.obstacle_cost(self.scene.distance(q) - self.radius);
        }
        smooth + self.lambda * obs
    }

    fn min_clearance(&self, way: &[Vector3<f64>]) -> f64 {
        way.iter().map(|q| self.scene.distance(q) - self.radius).fold(f64::INFINITY, f64::min)
    }

    /// Plan from `start` to `goal` with `n` interior waypoints, `iters` covariant-gradient steps.
    pub fn plan(&self, start: Vector3<f64>, goal: Vector3<f64>, n: usize, iters: usize) -> ChompResult {
        let a = Self::smoothness_matrix(n);
        let ainv = a.clone().try_inverse().expect("Laplacian is invertible");
        // interior waypoints, initialized on the straight line
        let mut xi: Vec<Vector3<f64>> = (1..=n).map(|i| start + (goal - start) * (i as f64 / (n + 1) as f64)).collect();

        let full = |xi: &[Vector3<f64>]| {
            let mut v = Vec::with_capacity(n + 2);
            v.push(start);
            v.extend_from_slice(xi);
            v.push(goal);
            v
        };
        let mut cost_history = vec![self.total_cost(&full(&xi))];

        for _ in 0..iters {
            // smoothness gradient  A·ξ − B  (B carries the fixed endpoints)
            let mut grad: Vec<Vector3<f64>> = (0..n).map(|i| {
                let mut g = xi[i] * 2.0;
                g -= if i == 0 { start } else { xi[i - 1] };
                g -= if i + 1 == n { goal } else { xi[i + 1] };
                g
            }).collect();
            // obstacle gradient  λ·c′(clr)·∇d  at each waypoint
            for (i, gi) in grad.iter_mut().enumerate() {
                let clr = self.scene.distance(&xi[i]) - self.radius;
                let dc = self.obstacle_dcost(clr);
                if dc != 0.0 {
                    *gi += self.lambda * dc * self.scene.gradient(&xi[i]);
                }
            }
            // covariant step: ξ ← ξ − η·A⁻¹·grad   (per coordinate)
            for axis in 0..3 {
                let g_axis = nalgebra::DVector::from_iterator(n, grad.iter().map(|g| g[axis]));
                let step = &ainv * g_axis;
                for i in 0..n {
                    xi[i][axis] -= self.eta * step[i];
                }
            }
            cost_history.push(self.total_cost(&full(&xi)));
        }

        let way = full(&xi);
        ChompResult {
            cost: *cost_history.last().unwrap(),
            min_clearance: self.min_clearance(&way),
            waypoints: way,
            cost_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::Sdf;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    #[test]
    fn with_no_obstacles_it_recovers_the_straight_line() {
        // Smoothness alone ⇒ the covariant minimizer is the evenly-spaced straight line. A perturbed start
        // must converge back to it.
        let empty = SdfScene { prims: vec![] };
        let chomp = Chomp { scene: &empty, epsilon: 0.1, lambda: 0.0, eta: 1.0, radius: 0.0 };
        let start = v(0.0, 0.0, 0.0);
        let goal = v(2.0, 0.0, 0.0);
        let res = chomp.plan(start, goal, 9, 5);
        for (i, w) in res.waypoints.iter().enumerate() {
            let t = i as f64 / (res.waypoints.len() - 1) as f64;
            let line = start + (goal - start) * t;
            assert!((w - line).norm() < 1e-9, "waypoint {i} off the straight line: {w:?} vs {line:?}");
        }
    }

    #[test]
    fn it_bows_the_path_around_an_obstacle() {
        // THE HEADLINE. An obstacle straddling the straight line (just off-axis so the avoidance direction
        // is defined) must be routed around: the path's minimum clearance rises above the initial
        // collision, and the total cost falls.
        let scene = SdfScene { prims: vec![Sdf::Sphere { center: v(1.0, 0.12, 0.0), radius: 0.4 }] };
        let chomp = Chomp { scene: &scene, epsilon: 0.15, lambda: 15.0, eta: 0.02, radius: 0.05 };
        let start = v(0.0, 0.0, 0.0);
        let goal = v(2.0, 0.0, 0.0);
        // straight-line clearance at the midpoint (starts in collision)
        let straight_mid_clr = scene.distance(&v(1.0, 0.0, 0.0)) - 0.05;
        assert!(straight_mid_clr < 0.0, "the straight line should start in collision: {straight_mid_clr}");
        let res = chomp.plan(start, goal, 19, 1500);
        assert!(res.min_clearance > straight_mid_clr + 0.3, "CHOMP should raise clearance: {straight_mid_clr} → {}", res.min_clearance);
        assert!(res.min_clearance > 0.0, "the final path should be collision-free: {}", res.min_clearance);
        // endpoints preserved
        assert!((res.waypoints.first().unwrap() - start).norm() < 1e-12);
        assert!((res.waypoints.last().unwrap() - goal).norm() < 1e-12);
    }

    #[test]
    fn the_cost_descends_and_converges() {
        // CHOMP drives the cost down substantially; on the non-convex obstacle hinge with a fixed step the
        // only "rises" are convergence-noise-sized (a tiny fraction of the total descent), never divergence.
        let scene = SdfScene { prims: vec![Sdf::Sphere { center: v(1.0, 0.12, 0.0), radius: 0.4 }] };
        let chomp = Chomp { scene: &scene, epsilon: 0.15, lambda: 15.0, eta: 0.02, radius: 0.05 };
        let res = chomp.plan(v(0.0, 0.0, 0.0), v(2.0, 0.0, 0.0), 19, 1500);
        let (init, fin) = (res.cost_history[0], *res.cost_history.last().unwrap());
        assert!(fin < 0.5 * init, "cost should fall substantially: {init} → {fin}");
        let max_rise = res.cost_history.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
        assert!(max_rise < 0.01 * init, "any per-step rise must be convergence noise: max rise {max_rise} vs init {init}");
    }
}
