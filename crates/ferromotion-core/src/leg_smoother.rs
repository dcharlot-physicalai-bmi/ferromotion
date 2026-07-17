//! **Factor-graph proprioceptive legged smoother** — a fixed-lag pose-graph estimator over a window of
//! base keyframes, the *smoother* counterpart to the invariant-EKF *filter*. Where a filter commits the
//! past and only updates the present, a smoother re-optimizes the whole window every step, so a fact
//! learned late (a contact revisited, a leg-odometry constraint) reaches back and corrects earlier poses.
//!
//! Each base pose is an `SE(2)` keyframe `(x, y, θ)`. Three factor types build the graph:
//! * a **prior** anchoring a keyframe;
//! * **IMU/odometry** relative-pose factors between consecutive keyframes;
//! * **contact / leg-odometry** relative-pose factors from the (independent) leg kinematics while a foot
//!   stays planted — the proprioceptive information that fights base drift.
//!
//! The nonlinear least-squares problem is solved by the sparse Gauss-Newton core
//! [`crate::solve_factor_graph`] (faer sparse Cholesky). Analytic `SE(2)` Jacobians. Pure Rust → WASM-clean.

use crate::{solve_factor_graph, SolveOptions, SparseFactor};
use nalgebra::{DMatrix, DVector};

fn wrap(a: f64) -> f64 {
    a.sin().atan2(a.cos())
}

/// A relative-pose factor between keyframes `i` and `j`: the measured transform `(dx, dy, dθ)` is the
/// pose of `j` expressed in `i`'s frame. Serves both IMU-odometry and contact/leg-odometry edges.
#[derive(Clone, Debug)]
pub struct RelPose2 {
    pub i: usize,
    pub j: usize,
    pub meas: [f64; 3],
    /// Square-root information weight (higher ⇒ more trusted).
    pub w: f64,
}

impl SparseFactor for RelPose2 {
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
        let (dx, dy) = (xj - xi, yj - yi);
        // predicted relative pose (j in i's frame)
        let px = c * dx + s * dy;
        let py = -s * dx + c * dy;
        let pt = wrap(tj - ti);
        let r = DVector::from_row_slice(&[
            self.w * (px - self.meas[0]),
            self.w * (py - self.meas[1]),
            self.w * wrap(pt - self.meas[2]),
        ]);
        // analytic Jacobian (3 × 6), columns [xi, yi, ti, xj, yj, tj]
        let mut j = DMatrix::zeros(3, 6);
        // ∂px
        j[(0, 0)] = -c;
        j[(0, 1)] = -s;
        j[(0, 2)] = -s * dx + c * dy; // ∂(c·dx + s·dy)/∂ti
        j[(0, 3)] = c;
        j[(0, 4)] = s;
        // ∂py
        j[(1, 0)] = s;
        j[(1, 1)] = -c;
        j[(1, 2)] = -c * dx - s * dy; // ∂(−s·dx + c·dy)/∂ti
        j[(1, 3)] = -s;
        j[(1, 4)] = c;
        // ∂pt
        j[(2, 2)] = -1.0;
        j[(2, 5)] = 1.0;
        (r, j * self.w)
    }
}

/// A prior on keyframe `i`, anchoring it to `(x, y, θ)`.
#[derive(Clone, Debug)]
pub struct PriorPose2 {
    pub i: usize,
    pub meas: [f64; 3],
    pub w: f64,
}

impl SparseFactor for PriorPose2 {
    fn dim(&self) -> usize {
        3
    }
    fn vars(&self) -> Vec<usize> {
        vec![3 * self.i, 3 * self.i + 1, 3 * self.i + 2]
    }
    fn eval(&self, x: &[f64]) -> (DVector<f64>, DMatrix<f64>) {
        let r = DVector::from_row_slice(&[
            self.w * (x[3 * self.i] - self.meas[0]),
            self.w * (x[3 * self.i + 1] - self.meas[1]),
            self.w * wrap(x[3 * self.i + 2] - self.meas[2]),
        ]);
        (r, DMatrix::identity(3, 3) * self.w)
    }
}

/// A fixed-lag pose-graph smoother over `n` base keyframes.
#[derive(Clone, Debug, Default)]
pub struct LegSmoother {
    pub n: usize,
    priors: Vec<PriorPose2>,
    edges: Vec<RelPose2>,
}

impl LegSmoother {
    pub fn new(n: usize) -> Self {
        LegSmoother { n, priors: vec![], edges: vec![] }
    }
    pub fn add_prior(&mut self, i: usize, pose: [f64; 3], w: f64) {
        self.priors.push(PriorPose2 { i, meas: pose, w });
    }
    /// An IMU/odometry edge (consecutive keyframes).
    pub fn add_odometry(&mut self, i: usize, j: usize, rel: [f64; 3], w: f64) {
        self.edges.push(RelPose2 { i, j, meas: rel, w });
    }
    /// A contact / leg-odometry edge — the independent proprioceptive constraint from a planted foot.
    pub fn add_contact(&mut self, i: usize, j: usize, rel: [f64; 3], w: f64) {
        self.edges.push(RelPose2 { i, j, meas: rel, w });
    }

    /// Solve the smoothing problem from an initial guess (`3n` scalars) → smoothed poses.
    pub fn solve(&self, x0: &[f64]) -> Vec<[f64; 3]> {
        let mut factors: Vec<Box<dyn SparseFactor>> = Vec::new();
        for p in &self.priors {
            factors.push(Box::new(p.clone()));
        }
        for e in &self.edges {
            factors.push(Box::new(e.clone()));
        }
        let opts = SolveOptions { max_iters: 60, tol: 1e-12, ..Default::default() };
        let res = solve_factor_graph(3 * self.n, &factors, x0, &opts);
        (0..self.n).map(|k| [res.x[3 * k], res.x[3 * k + 1], res.x[3 * k + 2]]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ground-truth base trajectory: a gentle left turn while advancing.
    fn ground_truth(n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                [0.3 * t, 0.05 * t * t, 0.04 * t] // x, y, heading
            })
            .collect()
    }

    /// The relative pose of `b` in `a`'s frame.
    fn rel(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        let (c, s) = (a[2].cos(), a[2].sin());
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        [c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])]
    }

    #[test]
    fn recovers_the_ground_truth_from_perfect_factors() {
        // THE HEADLINE. Perfect prior + odometry ⇒ the smoother reconstructs the trajectory to
        // machine precision, from a deliberately wrong initial guess.
        let gt = ground_truth(8);
        let mut sm = LegSmoother::new(8);
        sm.add_prior(0, gt[0], 100.0);
        for k in 0..7 {
            sm.add_odometry(k, k + 1, rel(gt[k], gt[k + 1]), 50.0);
        }
        let x0: Vec<f64> = (0..8).flat_map(|_| [0.0, 0.0, 0.0]).collect(); // all-zero guess
        let est = sm.solve(&x0);
        let worst = (0..8)
            .map(|k| (0..3).map(|c| (est[k][c] - gt[k][c]).abs()).fold(0.0, f64::max))
            .fold(0.0, f64::max);
        assert!(worst < 1e-6, "did not recover ground truth: worst error {worst}");
    }

    #[test]
    fn the_relative_pose_jacobian_matches_finite_differences() {
        let f = RelPose2 { i: 0, j: 1, meas: [0.2, -0.1, 0.05], w: 1.3 };
        let x = vec![0.4, -0.3, 0.2, 1.1, 0.6, 0.5];
        let (_, jac) = f.eval(&x);
        let eps = 1e-6;
        for col in 0..6 {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[col] += eps;
            xm[col] -= eps;
            let fd = (f.eval(&xp).0 - f.eval(&xm).0) / (2.0 * eps);
            for row in 0..3 {
                assert!((jac[(row, col)] - fd[row]).abs() < 1e-5, "J[{row},{col}] {} vs fd {}", jac[(row, col)], fd[row]);
            }
        }
    }

    #[test]
    fn contact_factors_cut_the_drift() {
        // Noisy odometry drifts; adding independent contact/leg-odometry edges pulls the estimate back.
        let gt = ground_truth(9);
        let build = |with_contact: bool| -> f64 {
            let mut sm = LegSmoother::new(9);
            sm.add_prior(0, gt[0], 100.0);
            for k in 0..8 {
                // biased odometry: a small systematic under-rotation + translation error
                let mut r = rel(gt[k], gt[k + 1]);
                r[0] += 0.03;
                r[2] -= 0.015;
                sm.add_odometry(k, k + 1, r, 20.0);
            }
            if with_contact {
                // a foot planted across k..k+2 gives an accurate 2-step relative constraint
                for k in (0..7).step_by(2) {
                    sm.add_contact(k, k + 2, rel(gt[k], gt[k + 2]), 40.0);
                }
            }
            let x0: Vec<f64> = (0..9).flat_map(|_| [0.0, 0.0, 0.0]).collect();
            let est = sm.solve(&x0);
            // final-keyframe position drift
            ((est[8][0] - gt[8][0]).powi(2) + (est[8][1] - gt[8][1]).powi(2)).sqrt()
        };
        let odom_only = build(false);
        let with_contact = build(true);
        assert!(with_contact < odom_only, "contact factors should reduce drift: {with_contact} vs {odom_only}");
        assert!(with_contact < 0.6 * odom_only, "contact factors should cut drift substantially");
    }

    #[test]
    fn a_late_factor_corrects_the_past_a_smoother_not_a_filter() {
        // THE DISTINCTION. Build a drifting odometry chain, then add ONE late loop-closure/contact
        // factor tying the last keyframe to its true relative pose from the first. A filter cannot
        // revise committed past states; the smoother pushes the correction back through the window,
        // so an *interior* keyframe moves closer to ground truth.
        let gt = ground_truth(7);
        let mut base = LegSmoother::new(7);
        base.add_prior(0, gt[0], 100.0);
        for k in 0..6 {
            let mut r = rel(gt[k], gt[k + 1]);
            r[0] += 0.04; // drift
            base.add_odometry(k, k + 1, r, 20.0);
        }
        let x0: Vec<f64> = (0..7).flat_map(|_| [0.0, 0.0, 0.0]).collect();
        let before = base.solve(&x0);
        let mid_err_before = ((before[3][0] - gt[3][0]).powi(2) + (before[3][1] - gt[3][1]).powi(2)).sqrt();

        // now add the late constraint (0 → 6 from an accurate long-baseline contact/loop closure)
        let mut with_closure = base.clone();
        with_closure.add_contact(0, 6, rel(gt[0], gt[6]), 60.0);
        let after = with_closure.solve(&x0);
        let mid_err_after = ((after[3][0] - gt[3][0]).powi(2) + (after[3][1] - gt[3][1]).powi(2)).sqrt();

        assert!(mid_err_after < mid_err_before, "the late factor should retro-correct the interior keyframe: {mid_err_after} vs {mid_err_before}");
    }
}
