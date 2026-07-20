//! **Catmull–Rom splines** — the *interpolating* cubic spline. Given a sequence of waypoints it produces a
//! smooth C¹ curve that passes **through every one of them**, with the tangent at each interior point set to
//! `(pᵢ₊₁ − pᵢ₋₁)/2`. This is the complement to the crate's [`crate::BSpline`], which *approximates* its
//! control points (the curve stays inside their hull but touches none): use a B-spline when you shape a path
//! by pulling control points, a Catmull–Rom when you must hit given waypoints exactly — smoothing a coarse
//! plan, threading a camera through key poses, or filleting a polyline of via-points.
//!
//! Uniform parameterization here; endpoints are handled by duplicating the first and last waypoint so the
//! curve spans the whole sequence. Analytic position and velocity. Verified: the curve interpolates every
//! waypoint, is C¹ (velocity continuous) across segments, matches the `(pᵢ₊₁ − pᵢ₋₁)/2` tangent, and
//! reduces to a straight line on collinear points. Pure `nalgebra` → WASM-clean.

use nalgebra::DVector;

/// A uniform Catmull–Rom spline through `m ≥ 2` waypoints; the parameter domain is `[0, m − 1]` (one unit
/// per segment).
#[derive(Clone, Debug)]
pub struct CatmullRom {
    pub pts: Vec<DVector<f64>>,
}

impl CatmullRom {
    /// Number of segments (`waypoints − 1`).
    pub fn segments(&self) -> usize {
        self.pts.len().saturating_sub(1)
    }

    // the four control points governing segment `seg` (endpoints duplicated)
    fn quad(&self, seg: usize) -> (&DVector<f64>, &DVector<f64>, &DVector<f64>, &DVector<f64>) {
        let m = self.pts.len();
        let p0 = &self.pts[seg.saturating_sub(1)];
        let p1 = &self.pts[seg];
        let p2 = &self.pts[seg + 1];
        let p3 = &self.pts[(seg + 2).min(m - 1)];
        (p0, p1, p2, p3)
    }

    fn locate(&self, t: f64) -> (usize, f64) {
        let n = self.segments();
        let tt = t.clamp(0.0, n as f64);
        let seg = (tt.floor() as usize).min(n - 1);
        (seg, tt - seg as f64)
    }

    /// Position at parameter `t ∈ [0, segments]`.
    pub fn position(&self, t: f64) -> DVector<f64> {
        let (seg, u) = self.locate(t);
        let (p0, p1, p2, p3) = self.quad(seg);
        // 0.5·[ 2p1 + (−p0+p2)u + (2p0−5p1+4p2−p3)u² + (−p0+3p1−3p2+p3)u³ ]
        let u2 = u * u;
        let u3 = u2 * u;
        (p1 * 2.0 + (-p0 + p2) * u + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * u2 + (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * u3) * 0.5
    }

    /// Velocity `dp/dt` at parameter `t`.
    pub fn velocity(&self, t: f64) -> DVector<f64> {
        let (seg, u) = self.locate(t);
        let (p0, p1, p2, p3) = self.quad(seg);
        let u2 = u * u;
        ((-p0 + p2) + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (2.0 * u) + (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * (3.0 * u2)) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dv(v: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(v)
    }

    fn waypoints() -> Vec<DVector<f64>> {
        vec![dv(&[0.0, 0.0]), dv(&[1.0, 2.0]), dv(&[3.0, -1.0]), dv(&[4.0, 1.0]), dv(&[6.0, 0.0])]
    }

    #[test]
    fn it_passes_through_every_waypoint() {
        // THE ORACLE. At integer parameters the curve equals the waypoints exactly (interpolation) — the
        // defining property that separates it from an approximating B-spline.
        let cr = CatmullRom { pts: waypoints() };
        for (i, w) in cr.pts.iter().enumerate() {
            let p = cr.position(i as f64);
            assert!((&p - w).norm() < 1e-12, "waypoint {i}: {p} vs {w}");
        }
    }

    #[test]
    fn it_is_c1_continuous_across_segments() {
        // Velocity matches from both sides of each interior knot.
        let cr = CatmullRom { pts: waypoints() };
        for seg in 1..cr.segments() {
            let t = seg as f64;
            let eps = 1e-6;
            let vl = cr.velocity(t - eps);
            let vr = cr.velocity(t + eps);
            assert!((&vl - &vr).norm() < 1e-4, "velocity discontinuous at knot {seg}: {vl} vs {vr}");
        }
    }

    #[test]
    fn the_interior_tangent_is_the_centered_difference() {
        // The Catmull–Rom tangent at an interior waypoint pᵢ is (pᵢ₊₁ − pᵢ₋₁)/2.
        let cr = CatmullRom { pts: waypoints() };
        for i in 1..cr.pts.len() - 1 {
            let tangent = cr.velocity(i as f64);
            let expected = (&cr.pts[i + 1] - &cr.pts[i - 1]) * 0.5;
            assert!((&tangent - &expected).norm() < 1e-9, "tangent at {i}: {tangent} vs {expected}");
        }
    }

    #[test]
    fn collinear_waypoints_give_a_straight_line() {
        // Points on a line ⇒ the spline is that line (every sample lies on it).
        let cr = CatmullRom { pts: (0..5).map(|i| dv(&[i as f64, 2.0 * i as f64])).collect() };
        for k in 0..40 {
            let t = k as f64 / 40.0 * cr.segments() as f64;
            let p = cr.position(t);
            assert!((p[1] - 2.0 * p[0]).abs() < 1e-9, "off the line at t={t}: {p}");
        }
    }
}
