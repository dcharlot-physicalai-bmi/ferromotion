//! **2-D pose-graph SLAM** — the nonlinear back-end that turns noisy relative-motion measurements into a
//! globally-consistent trajectory. Each node is a robot pose `Tᵢ ∈ SE(2)`; each edge is a measured relative
//! transform `Z_ij` (from wheel odometry, scan-matching, or a **loop closure**). The optimal map minimizes
//! `Σ ‖Log(Z_ij⁻¹ · (Tᵢ⁻¹ Tⱼ))‖²_Ω` over all poses — an on-manifold Gauss–Newton solve. It is the batch
//! smoother behind GraphSLAM/g2o/GTSAM, and the back-end that pairs with the [`crate::KissIcp`] scan-matching
//! front-end (which produces the odometry edges).
//!
//! Built as a **consumer of [`crate::solve_factor_graph`]**: each pose contributes three scalar variables
//! `(x, y, θ)`, an SE(2) between-factor supplies the relative-pose residual with an analytic Jacobian, and a
//! prior factor anchors the gauge freedom at pose 0. Verified: with noise-free, loop-consistent measurements
//! it recovers the ground-truth trajectory to solver precision; a loop closure corrects the drift a
//! prior-free odometry chain cannot; and it reduces trajectory error under measurement noise. Pure
//! `nalgebra` → WASM-clean.

use crate::sparse::{solve_factor_graph, SparseFactor};
use crate::SolveOptions;
use nalgebra::{DMatrix, DVector};

/// A planar pose `(x, y, θ)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2 {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Pose2 { x, y, theta }
    }
    /// The relative transform `self⁻¹ ∘ other` — `other` expressed in `self`'s frame.
    pub fn between(&self, other: &Pose2) -> Pose2 {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        let (ex, ey) = (other.x - self.x, other.y - self.y);
        Pose2 { x: c * ex + s * ey, y: -s * ex + c * ey, theta: wrap(other.theta - self.theta) }
    }
}

/// Wrap an angle to `(−π, π]`.
fn wrap(a: f64) -> f64 {
    let mut a = a % std::f64::consts::TAU;
    if a > std::f64::consts::PI {
        a -= std::f64::consts::TAU;
    } else if a <= -std::f64::consts::PI {
        a += std::f64::consts::TAU;
    }
    a
}

/// An SE(2) between-factor: residual `Z_ij⁻¹ ∘ (Tᵢ⁻¹ Tⱼ)` in the minimal `(x, y, θ)` chart, scaled by a
/// scalar `sqrt_info` (isotropic information weight).
struct BetweenFactor {
    i: usize,
    j: usize,
    m: Pose2, // measured relative pose
    sqrt_info: f64,
}

impl SparseFactor for BetweenFactor {
    fn dim(&self) -> usize {
        3
    }
    fn vars(&self) -> Vec<usize> {
        vec![3 * self.i, 3 * self.i + 1, 3 * self.i + 2, 3 * self.j, 3 * self.j + 1, 3 * self.j + 2]
    }
    fn eval(&self, x: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        let (xi, yi, ti) = (x[3 * self.i], x[3 * self.i + 1], x[3 * self.i + 2]);
        let (xj, yj, tj) = (x[3 * self.j], x[3 * self.j + 1], x[3 * self.j + 2]);
        let (c, s) = (ti.cos(), ti.sin());
        let (ex, ey) = (xj - xi, yj - yi);
        let w = self.sqrt_info;
        // predicted Tᵢ⁻¹Tⱼ minus the measurement
        let r = DVector::from_row_slice(&[
            w * (c * ex + s * ey - self.m.x),
            w * (-s * ex + c * ey - self.m.y),
            w * wrap(tj - ti - self.m.theta),
        ]);
        // Jacobian 3×6 w.r.t. (xi,yi,ti, xj,yj,tj)
        let mut j = DMatrix::zeros(3, 6);
        // ∂r0
        j[(0, 0)] = w * -c;
        j[(0, 1)] = w * -s;
        j[(0, 2)] = w * (-s * ex + c * ey);
        j[(0, 3)] = w * c;
        j[(0, 4)] = w * s;
        // ∂r1
        j[(1, 0)] = w * s;
        j[(1, 1)] = w * -c;
        j[(1, 2)] = w * (-c * ex - s * ey);
        j[(1, 3)] = w * -s;
        j[(1, 4)] = w * c;
        // ∂r2
        j[(2, 2)] = -w;
        j[(2, 5)] = w;
        (r, j)
    }
}

/// A prior pinning one pose to a fixed value (fixes the gauge). Residual `sqrt_info·(Tᵢ − prior)`.
struct PriorFactor {
    i: usize,
    prior: Pose2,
    sqrt_info: f64,
}

impl SparseFactor for PriorFactor {
    fn dim(&self) -> usize {
        3
    }
    fn vars(&self) -> Vec<usize> {
        vec![3 * self.i, 3 * self.i + 1, 3 * self.i + 2]
    }
    fn eval(&self, x: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        let w = self.sqrt_info;
        let r = DVector::from_row_slice(&[
            w * (x[3 * self.i] - self.prior.x),
            w * (x[3 * self.i + 1] - self.prior.y),
            w * wrap(x[3 * self.i + 2] - self.prior.theta),
        ]);
        (r, DMatrix::identity(3, 3) * w)
    }
}

/// A 2-D pose-graph SLAM problem: poses (initial guesses) + between-edges + a gauge prior.
#[derive(Clone, Debug, Default)]
pub struct PoseGraph2D {
    poses: Vec<Pose2>,
    edges: Vec<(usize, usize, Pose2, f64)>,
    prior: Option<(usize, Pose2, f64)>,
}

impl PoseGraph2D {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pose (its initial estimate) and return its index.
    pub fn add_pose(&mut self, p: Pose2) -> usize {
        self.poses.push(p);
        self.poses.len() - 1
    }

    /// Add a relative-pose measurement `Tᵢ⁻¹Tⱼ ≈ measured`, with isotropic information weight `sqrt_info`.
    pub fn add_edge(&mut self, i: usize, j: usize, measured: Pose2, sqrt_info: f64) {
        self.edges.push((i, j, measured, sqrt_info));
    }

    /// Anchor the gauge: pin pose `i` to `prior` with a (large) information weight.
    pub fn set_prior(&mut self, i: usize, prior: Pose2, sqrt_info: f64) {
        self.prior = Some((i, prior, sqrt_info));
    }

    /// Optimize all poses by on-manifold Gauss–Newton over the factor graph. Returns the refined poses.
    pub fn optimize(&self, iters: usize) -> Vec<Pose2> {
        let n = self.poses.len();
        let mut factors: Vec<Box<dyn SparseFactor + '_>> = Vec::new();
        for &(i, j, m, w) in &self.edges {
            factors.push(Box::new(BetweenFactor { i, j, m, sqrt_info: w }));
        }
        if let Some((i, prior, w)) = self.prior {
            factors.push(Box::new(PriorFactor { i, prior, sqrt_info: w }));
        }
        let x0: Vec<f64> = self.poses.iter().flat_map(|p| [p.x, p.y, p.theta]).collect();
        let opts = SolveOptions { max_iters: iters, tol: 1e-12, lambda0: 1e-6 };
        let res = solve_factor_graph(3 * n, &factors, &x0, &opts);
        (0..n).map(|k| Pose2::new(res.x[3 * k], res.x[3 * k + 1], wrap(res.x[3 * k + 2]))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj_error(a: &[Pose2], b: &[Pose2]) -> f64 {
        a.iter().zip(b).map(|(p, q)| ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + wrap(p.theta - q.theta).powi(2)).sqrt()).sum::<f64>() / a.len() as f64
    }

    // A ground-truth square loop: 4 poses at the corners, heading tangent to the loop.
    fn ground_truth() -> Vec<Pose2> {
        vec![
            Pose2::new(0.0, 0.0, 0.0),
            Pose2::new(1.0, 0.0, std::f64::consts::FRAC_PI_2),
            Pose2::new(1.0, 1.0, std::f64::consts::PI),
            Pose2::new(0.0, 1.0, -std::f64::consts::FRAC_PI_2),
        ]
    }

    #[test]
    fn it_recovers_ground_truth_from_consistent_measurements() {
        // THE ORACLE. Odometry edges + a loop-closure edge, all computed from the true poses (noise-free),
        // must reconstruct the true trajectory exactly (up to the anchored gauge).
        let gt = ground_truth();
        let mut pg = PoseGraph2D::new();
        // start every estimate perturbed away from the truth
        for (k, p) in gt.iter().enumerate() {
            pg.add_pose(Pose2::new(p.x + 0.3 * (k as f64 - 1.5), p.y - 0.2, p.theta + 0.15));
        }
        // consecutive odometry edges
        for k in 0..3 {
            pg.add_edge(k, k + 1, gt[k].between(&gt[k + 1]), 1.0);
        }
        // loop closure 3 → 0
        pg.add_edge(3, 0, gt[3].between(&gt[0]), 1.0);
        pg.set_prior(0, gt[0], 1e3);

        let est = pg.optimize(50);
        assert!(traj_error(&est, &gt) < 1e-6, "should recover ground truth: err {}", traj_error(&est, &gt));
    }

    #[test]
    fn a_loop_closure_corrects_open_chain_drift() {
        // THE HEADLINE. Odometry alone (an open chain) leaves the last pose wherever accumulated error puts
        // it; adding a loop-closure constraint back to the start pulls the whole graph consistent. Here the
        // odometry measurements carry a small systematic bias; the loop closure (to the true relation) fixes
        // the endpoint.
        let gt = ground_truth();
        let mut pg = PoseGraph2D::new();
        for p in &gt {
            pg.add_pose(*p);
        }
        // biased odometry: each rotation under-measured, so the open chain won't close
        for k in 0..3 {
            let mut m = gt[k].between(&gt[k + 1]);
            m.theta = wrap(m.theta - 0.1); // systematic heading bias
            pg.add_edge(k, k + 1, m, 1.0);
        }
        // an accurate loop closure 3 → 0
        pg.add_edge(3, 0, gt[3].between(&gt[0]), 5.0);
        pg.set_prior(0, gt[0], 1e3);

        let est = pg.optimize(50);
        // the loop-closure residual after optimization should be small (graph is globally consistent)
        let lc = est[3].between(&est[0]);
        let target = gt[3].between(&gt[0]);
        let lc_res = ((lc.x - target.x).powi(2) + (lc.y - target.y).powi(2) + wrap(lc.theta - target.theta).powi(2)).sqrt();
        assert!(lc_res < 0.05, "loop closure should be satisfied after optimization: {lc_res}");
        // and pose 0 stays anchored at the origin
        assert!(est[0].x.abs() < 1e-3 && est[0].y.abs() < 1e-3, "gauge anchored at origin");
    }

    #[test]
    fn it_reduces_trajectory_error_under_noisy_measurements() {
        // With noisy edges, the optimized trajectory is closer to ground truth than the raw dead-reckoned
        // chain built by composing the noisy odometry.
        let gt = ground_truth();
        // deterministic noise
        let noise = [0.05, -0.04, 0.06, -0.03, 0.045, -0.05, 0.035, -0.055, 0.04];
        let mut pg = PoseGraph2D::new();
        // dead-reckon initial guess from noisy odometry
        let mut dr = vec![gt[0]];
        for k in 0..3 {
            let mut m = gt[k].between(&gt[k + 1]);
            m.x += noise[3 * k];
            m.y += noise[3 * k + 1];
            m.theta = wrap(m.theta + noise[3 * k + 2]);
            // compose onto the dead-reckoned chain
            let prev = dr[k];
            let (c, s) = (prev.theta.cos(), prev.theta.sin());
            dr.push(Pose2::new(prev.x + c * m.x - s * m.y, prev.y + s * m.x + c * m.y, wrap(prev.theta + m.theta)));
            pg.add_pose(dr[k + 1]);
        }
        // fix pose 0's estimate
        pg.poses.insert(0, gt[0]);
        // rebuild edges (noisy) + a noisy-but-informative loop closure
        for k in 0..3 {
            let mut m = gt[k].between(&gt[k + 1]);
            m.x += noise[3 * k];
            m.y += noise[3 * k + 1];
            m.theta = wrap(m.theta + noise[3 * k + 2]);
            pg.add_edge(k, k + 1, m, 1.0);
        }
        pg.add_edge(3, 0, gt[3].between(&gt[0]), 2.0);
        pg.set_prior(0, gt[0], 1e3);

        let dr_err = traj_error(&dr, &gt);
        let est = pg.optimize(50);
        let opt_err = traj_error(&est, &gt);
        assert!(opt_err < dr_err, "optimization should beat dead reckoning: {opt_err} vs {dr_err}");
    }
}
