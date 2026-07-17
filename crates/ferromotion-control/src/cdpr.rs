//! **Cable-driven parallel robot — tension distribution** (Pott's improved closed-form; Gouttefarde
//! et al.). A platform is held by `m` cables from fixed anchors; cables can only *pull*, so every cable
//! tension must stay within `[t_min, t_max]` with `t_min > 0`. When the robot is redundantly actuated
//! (`m > DOF`) the tensions that produce a required wrench are not unique, and the control problem is to
//! pick a continuous, strictly-positive one — the redundancy resolution at the heart of CDPR control.
//!
//! The closed form takes the mid-tension `t̄ = (t_min+t_max)/2` and projects it onto the wrench-
//! equilibrium affine set `W t = w`: `t = t̄ + W⁺ (w − W t̄)`, the tension nearest the middle of the
//! range that balances the wrench exactly (`W⁺` the pseudo-inverse). It is feasible throughout the
//! closed-form workspace; outside it, one or more cables would slacken or over-tension and no valid
//! distribution exists. Planar (3-DOF) here. Pure `nalgebra` → WASM-clean.

use nalgebra::{DMatrix, DVector, Vector2, Vector3};

/// A planar cable-driven parallel robot.
#[derive(Clone, Debug)]
pub struct Cdpr {
    /// Fixed anchor points (world).
    pub anchors: Vec<Vector2<f64>>,
    /// Cable attachment points on the platform, in the platform's local frame.
    pub attach: Vec<Vector2<f64>>,
    pub t_min: f64,
    pub t_max: f64,
}

/// The result of a tension-distribution query.
#[derive(Clone, Debug)]
pub struct TensionResult {
    pub tensions: DVector<f64>,
    /// Every cable within `[t_min, t_max]` (a valid pulling distribution)?
    pub feasible: bool,
    /// Residual `‖W t − w‖` — the wrench-equilibrium error (≈0 by construction).
    pub residual: f64,
}

impl Cdpr {
    fn m(&self) -> usize {
        self.anchors.len()
    }

    /// The `3×m` structure (wrench) matrix at platform pose `(x, y, θ)`: column `i` is the unit wrench
    /// `[û_i; r_i × û_i]` a unit tension in cable `i` applies to the platform, with `û_i` the cable
    /// direction (attachment → anchor) and `r_i` the moment arm from the platform origin.
    pub fn structure_matrix(&self, x: f64, y: f64, theta: f64) -> DMatrix<f64> {
        let (c, s) = (theta.cos(), theta.sin());
        let mut w = DMatrix::zeros(3, self.m());
        for i in 0..self.m() {
            let l = self.attach[i];
            let r = Vector2::new(c * l.x - s * l.y, s * l.x + c * l.y); // rotated moment arm
            let p = Vector2::new(x, y) + r; // world attachment
            let u = (self.anchors[i] - p).normalize(); // pull direction
            w[(0, i)] = u.x;
            w[(1, i)] = u.y;
            w[(2, i)] = r.x * u.y - r.y * u.x; // r × û (scalar torque)
        }
        w
    }

    /// Closed-form tension distribution to apply platform wrench `w_platform` = `[fx, fy, τ]` — i.e. to
    /// balance an external wrench `−w_platform`. Returns the mid-range-nearest feasible tensions.
    pub fn tension_distribution(&self, x: f64, y: f64, theta: f64, w_platform: Vector3<f64>) -> TensionResult {
        let mat = self.structure_matrix(x, y, theta);
        let w = DVector::from_row_slice(w_platform.as_slice());
        let t_mid = DVector::from_element(self.m(), 0.5 * (self.t_min + self.t_max));
        let pinv = mat.clone().pseudo_inverse(1e-12).unwrap_or_else(|_| DMatrix::zeros(self.m(), 3));
        let t = &t_mid + &pinv * (&w - &mat * &t_mid);
        let residual = (&mat * &t - &w).norm();
        let feasible = t.iter().all(|&ti| ti >= self.t_min - 1e-9 && ti <= self.t_max + 1e-9);
        TensionResult { tensions: t, feasible, residual }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A symmetric 4-cable planar CDPR: a square frame with the platform (a small square) in the middle.
    fn square_cdpr() -> Cdpr {
        Cdpr {
            anchors: vec![
                Vector2::new(-2.0, -2.0),
                Vector2::new(2.0, -2.0),
                Vector2::new(2.0, 2.0),
                Vector2::new(-2.0, 2.0),
            ],
            attach: vec![
                Vector2::new(-0.5, -0.5),
                Vector2::new(0.5, -0.5),
                Vector2::new(0.5, 0.5),
                Vector2::new(-0.5, 0.5),
            ],
            t_min: 1.0,
            t_max: 40.0,
        }
    }

    #[test]
    fn the_distribution_balances_the_wrench_exactly() {
        // THE INVARIANT. At a generic (non-singular) pose the structure matrix has full row rank, so
        // any wrench is reproduced exactly: W t = w. (A rotated, off-centre pose; the perfectly
        // centred config is torque-singular — see the dedicated test.)
        let r = square_cdpr();
        for w in [Vector3::new(0.0, 10.0, 0.0), Vector3::new(3.0, -5.0, 1.0), Vector3::zeros()] {
            let res = r.tension_distribution(0.3, -0.2, 0.15, w);
            assert!(res.residual < 1e-9, "wrench equilibrium violated: ‖Wt−w‖ = {}", res.residual);
        }
    }

    #[test]
    fn the_centred_symmetric_config_is_torque_singular() {
        // A real property worth pinning: at the perfectly centred, unrotated config every cable pulls
        // straight along its own moment arm, so the platform can apply forces but NO torque — a wrench
        // with τ ≠ 0 cannot be balanced there (residual = |τ|), and any rotation restores authority.
        let r = square_cdpr();
        let singular = r.tension_distribution(0.0, 0.0, 0.0, Vector3::new(0.0, 0.0, 1.0));
        assert!(singular.residual > 0.5, "centred config should be unable to make torque: {}", singular.residual);
        let rotated = r.tension_distribution(0.0, 0.0, 0.2, Vector3::new(0.0, 0.0, 1.0));
        assert!(rotated.residual < 1e-9, "a rotated config recovers torque authority: {}", rotated.residual);
    }

    #[test]
    fn a_centered_platform_holds_gravity_with_positive_symmetric_tensions() {
        // Hold up a weight (apply +y force) from the centre: all four cables pull, and by symmetry the
        // two lower and two upper cables pair up — every tension strictly positive and within range.
        let r = square_cdpr();
        let res = r.tension_distribution(0.0, 0.0, 0.0, Vector3::new(0.0, 10.0, 0.0));
        assert!(res.feasible, "centered hold should be feasible: {:?}", res.tensions);
        assert!(res.tensions.iter().all(|&t| t >= r.t_min - 1e-9), "all cables must stay taut (≥ t_min)");
        // symmetry: bottom pair equal, top pair equal, left/right mirror
        assert!((res.tensions[0] - res.tensions[1]).abs() < 1e-9, "bottom cables should match by symmetry");
        assert!((res.tensions[2] - res.tensions[3]).abs() < 1e-9, "top cables should match by symmetry");
    }

    #[test]
    fn the_solution_sits_nearest_the_mid_tension() {
        // The closed form is the wrench-feasible tension closest to the mid-range: the correction
        // t − t̄ lies in the row space of W (⊥ its null space), so no null-space tension is wasted.
        let r = square_cdpr();
        let res = r.tension_distribution(0.3, -0.2, 0.1, Vector3::new(2.0, 8.0, 0.5));
        let mat = r.structure_matrix(0.3, -0.2, 0.1);
        let t_mid = DVector::from_element(4, 0.5 * (r.t_min + r.t_max));
        let delta = &res.tensions - &t_mid;
        // min-norm ⟺ the correction lies in the row space of W (⊥ null(W)). Verify it is fixed by the
        // row-space projector P = W⁺W:  P·δ = δ. (nalgebra's SVD is thin, so we avoid the null basis.)
        let pinv = mat.clone().pseudo_inverse(1e-12).unwrap();
        let proj = &pinv * &mat; // 4×4 projector onto row(W)
        assert!((&proj * &delta - &delta).norm() < 1e-8, "correction must lie in row(W) (min-norm)");
    }

    #[test]
    fn an_out_of_range_wrench_is_flagged_infeasible() {
        // A wrench too large for the cable limits forces a tension past t_max (or a slack cable below
        // t_min) — the closed form still returns, but feasible=false warns it is not usable.
        let r = square_cdpr();
        let res = r.tension_distribution(0.0, 0.0, 0.0, Vector3::new(0.0, 500.0, 0.0)); // huge load
        assert!(!res.feasible, "an over-range wrench must be flagged infeasible");
        assert!(res.residual < 1e-9, "equilibrium still holds even when infeasible (bounds are the issue)");
    }
}
