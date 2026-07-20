//! **Uniform cubic B-spline trajectories** — the continuous-time curve representation behind modern
//! trajectory optimization (Ewok, `mav_trajectory_generation`) and continuous-time SLAM/VIO. A cubic
//! B-spline is defined by control points on a uniform time grid; the curve is **C²-continuous** everywhere
//! by construction (continuous position, velocity, and acceleration — so jerk is bounded), has **local
//! support** (moving one control point perturbs only a 4-span neighbourhood), and obeys the **convex-hull
//! property** (each curve point is a convex blend of four local control points, so it never overshoots their
//! hull). These are exactly the properties an optimizer wants: smoothness for free, sparse Jacobians, and
//! bound-preservation.
//!
//! This complements the crate's *piecewise-polynomial* generators — [`crate::Gpmp2`] (GP smoothing) and the
//! quadrotor min-snap / Frenet quintics — with the basis-spline formulation. Analytic position/velocity/
//! acceleration via the cubic basis (and its derivatives). Verified: the basis is a partition of unity;
//! the trajectory is C² across segment boundaries; the curve stays in the convex hull of the local control
//! points; and the analytic velocity matches a finite difference. Pure `nalgebra` → WASM-clean.

use nalgebra::DVector;

/// A uniform cubic B-spline over `k ≥ 4` control points, each segment spanning `dt` in time. The valid time
/// domain is `[0, (k − 3)·dt]` (`k − 3` segments).
#[derive(Clone, Debug)]
pub struct BSpline {
    pub ctrl: Vec<DVector<f64>>,
    pub dt: f64,
}

impl BSpline {
    /// Number of curve segments.
    pub fn segments(&self) -> usize {
        self.ctrl.len().saturating_sub(3)
    }

    fn dim(&self) -> usize {
        self.ctrl[0].len()
    }

    /// The four cubic B-spline basis weights `[b0, b1, b2, b3]` at local parameter `u ∈ [0, 1]`.
    fn basis(u: f64) -> [f64; 4] {
        let u2 = u * u;
        let u3 = u2 * u;
        [
            (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
            (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
            (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
            u3 / 6.0,
        ]
    }

    /// Basis first derivatives `db/du`.
    fn basis_d1(u: f64) -> [f64; 4] {
        let u2 = u * u;
        [-(1.0 - u) * (1.0 - u) / 2.0, (-12.0 * u + 9.0 * u2) / 6.0, (3.0 + 6.0 * u - 9.0 * u2) / 6.0, u2 / 2.0]
    }

    /// Basis second derivatives `d²b/du²`.
    fn basis_d2(u: f64) -> [f64; 4] {
        [1.0 - u, 3.0 * u - 2.0, 1.0 - 3.0 * u, u]
    }

    // locate the segment and local parameter for a global time `t`
    fn locate(&self, t: f64) -> (usize, f64) {
        let n = self.segments();
        let tt = t.clamp(0.0, n as f64 * self.dt - 1e-12);
        let seg = (tt / self.dt).floor() as usize;
        let seg = seg.min(n - 1);
        (seg, (tt - seg as f64 * self.dt) / self.dt)
    }

    fn blend(&self, seg: usize, w: &[f64; 4]) -> DVector<f64> {
        let mut out = DVector::zeros(self.dim());
        for (k, &wk) in w.iter().enumerate() {
            out += &self.ctrl[seg + k] * wk;
        }
        out
    }

    /// Position on the curve at time `t`.
    pub fn position(&self, t: f64) -> DVector<f64> {
        let (seg, u) = self.locate(t);
        self.blend(seg, &Self::basis(u))
    }

    /// Velocity `dp/dt` at time `t`.
    pub fn velocity(&self, t: f64) -> DVector<f64> {
        let (seg, u) = self.locate(t);
        self.blend(seg, &Self::basis_d1(u)) / self.dt
    }

    /// Acceleration `d²p/dt²` at time `t`.
    pub fn acceleration(&self, t: f64) -> DVector<f64> {
        let (seg, u) = self.locate(t);
        self.blend(seg, &Self::basis_d2(u)) / (self.dt * self.dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctrl_pts() -> Vec<DVector<f64>> {
        vec![
            DVector::from_row_slice(&[0.0, 0.0]),
            DVector::from_row_slice(&[1.0, 2.0]),
            DVector::from_row_slice(&[3.0, -1.0]),
            DVector::from_row_slice(&[4.0, 1.0]),
            DVector::from_row_slice(&[6.0, 0.5]),
            DVector::from_row_slice(&[7.0, -2.0]),
        ]
    }

    #[test]
    fn the_basis_is_a_partition_of_unity() {
        // THE ORACLE. The four cubic basis weights sum to 1 (and are non-negative) for every u ∈ [0,1] — the
        // property that makes the curve an affine/convex blend of its control points.
        for i in 0..=20 {
            let u = i as f64 / 20.0;
            let b = BSpline::basis(u);
            assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-12, "basis should sum to 1 at u={u}");
            assert!(b.iter().all(|&w| w >= -1e-12), "basis should be non-negative at u={u}");
            // derivative bases sum to 0
            assert!(BSpline::basis_d1(u).iter().sum::<f64>().abs() < 1e-12);
            assert!(BSpline::basis_d2(u).iter().sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn the_trajectory_is_c2_continuous_across_segment_boundaries() {
        // THE DEFINING PROPERTY. Position, velocity, AND acceleration match across each interior knot.
        let s = BSpline { ctrl: ctrl_pts(), dt: 0.5 };
        for seg in 1..s.segments() {
            let t = seg as f64 * s.dt;
            let eps = 1e-6;
            let (pl, pr) = (s.position(t - eps), s.position(t + eps));
            let (vl, vr) = (s.velocity(t - eps), s.velocity(t + eps));
            let (al, ar) = (s.acceleration(t - eps), s.acceleration(t + eps));
            assert!((pl - pr).norm() < 1e-4, "position discontinuous at knot {seg}");
            assert!((vl - vr).norm() < 1e-4, "velocity discontinuous at knot {seg}");
            assert!((al - ar).norm() < 1e-3, "acceleration discontinuous at knot {seg}");
        }
    }

    #[test]
    fn the_curve_stays_in_the_convex_hull_of_the_local_control_points() {
        // THE CONVEX-HULL PROPERTY. Every curve point lies within the axis-aligned bounds of its four
        // governing control points (a consequence of the non-negative partition-of-unity basis).
        let s = BSpline { ctrl: ctrl_pts(), dt: 1.0 };
        for i in 0..100 {
            let t = i as f64 / 100.0 * s.segments() as f64 * s.dt;
            let (seg, _u) = s.locate(t);
            let p = s.position(t);
            for d in 0..2 {
                let lo = (0..4).map(|k| s.ctrl[seg + k][d]).fold(f64::INFINITY, f64::min);
                let hi = (0..4).map(|k| s.ctrl[seg + k][d]).fold(f64::NEG_INFINITY, f64::max);
                assert!(p[d] >= lo - 1e-9 && p[d] <= hi + 1e-9, "curve escaped the hull at t={t}, dim {d}: {} not in [{lo},{hi}]", p[d]);
            }
        }
    }

    #[test]
    fn the_analytic_velocity_matches_a_finite_difference() {
        let s = BSpline { ctrl: ctrl_pts(), dt: 0.4 };
        let h = 1e-6;
        for i in 1..30 {
            let t = i as f64 / 30.0 * s.segments() as f64 * s.dt * 0.98;
            let fd = (s.position(t + h) - s.position(t - h)) / (2.0 * h);
            assert!((s.velocity(t) - fd).norm() < 1e-4, "velocity vs finite-diff at t={t}");
        }
    }
}
