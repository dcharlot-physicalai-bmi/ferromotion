//! **Polyline utilities** — the geometric post-processing a path needs after a planner produces it:
//! **Ramer–Douglas–Peucker** simplification (thin a dense trajectory to the few vertices that capture its
//! shape within a tolerance), **arc-length resampling** (re-space the vertices uniformly, which a
//! pure-pursuit/Stanley tracker or a spline fitter wants), and **length**. A sampling planner or a swept
//! Hybrid-A\* path comes out with hundreds of near-collinear points; RDP compresses it losslessly-within-ε,
//! and resampling turns an irregularly-spaced path into evenly-spaced way-points.
//!
//! Works in 3-D (planar paths just keep `z = 0`). Verified: a straight run of points simplifies to its two
//! endpoints; RDP keeps the endpoints and every dropped vertex stays within `ε` of the kept polyline, while
//! a sharp corner is preserved; resampling yields uniform spacing and preserves the total length and
//! endpoints. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector3;

/// Perpendicular distance from `p` to the segment `a`–`b`.
fn point_segment_distance(p: &Vector3<f64>, a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let ab = b - a;
    let len2 = ab.norm_squared();
    if len2 < 1e-18 {
        return (p - a).norm();
    }
    let t = ((p - a).dot(&ab) / len2).clamp(0.0, 1.0);
    (p - (a + t * ab)).norm()
}

/// The total arc length of a polyline.
pub fn polyline_length(points: &[Vector3<f64>]) -> f64 {
    points.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
}

/// **Ramer–Douglas–Peucker** simplification: return the subsequence of `points` (endpoints always kept)
/// such that every dropped vertex lies within `epsilon` of the retained polyline.
pub fn rdp_simplify(points: &[Vector3<f64>], epsilon: f64) -> Vec<Vector3<f64>> {
    if points.len() < 3 {
        return points.to_vec();
    }
    // farthest vertex from the chord (first → last)
    let (first, last) = (points[0], points[points.len() - 1]);
    let mut idx = 0;
    let mut dmax = 0.0;
    for (i, p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let d = point_segment_distance(p, &first, &last);
        if d > dmax {
            dmax = d;
            idx = i;
        }
    }
    if dmax > epsilon {
        // keep the farthest vertex and recurse on both halves
        let mut left = rdp_simplify(&points[..=idx], epsilon);
        let right = rdp_simplify(&points[idx..], epsilon);
        left.pop(); // avoid duplicating the shared vertex
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

/// **Arc-length resampling**: return points spaced (approximately) `spacing` apart along the polyline,
/// starting at the first vertex and ending at the last.
pub fn resample_uniform(points: &[Vector3<f64>], spacing: f64) -> Vec<Vector3<f64>> {
    if points.len() < 2 || spacing <= 0.0 {
        return points.to_vec();
    }
    let mut out = vec![points[0]];
    let mut carry = 0.0; // distance already covered since the last emitted point, within the current segment
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let seg = (b - a).norm();
        if seg < 1e-18 {
            continue;
        }
        let dir = (b - a) / seg;
        let mut dist = spacing - carry; // arc position of the next sample within this segment
        while dist <= seg + 1e-12 {
            out.push(a + dir * dist);
            dist += spacing;
        }
        carry = seg - (dist - spacing); // leftover into the next segment
    }
    // ensure the true endpoint is present
    let last = *points.last().unwrap();
    if (out.last().unwrap() - last).norm() > 1e-9 {
        out.push(last);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> Vector3<f64> {
        Vector3::new(x, y, 0.0)
    }

    #[test]
    fn a_straight_run_simplifies_to_its_endpoints() {
        // THE ORACLE. Many collinear points ⇒ RDP keeps only the two ends.
        let pts: Vec<Vector3<f64>> = (0..20).map(|i| v(i as f64 * 0.5, i as f64 * 0.5)).collect();
        let s = rdp_simplify(&pts, 1e-6);
        assert_eq!(s.len(), 2, "a straight line reduces to 2 points, got {}", s.len());
        assert!((s[0] - pts[0]).norm() < 1e-12 && (s[1] - *pts.last().unwrap()).norm() < 1e-12, "endpoints preserved");
    }

    #[test]
    fn rdp_keeps_a_corner_and_bounds_the_error() {
        // An L-shaped path with dense sampling: RDP must keep the corner, and every dropped vertex stays
        // within ε of the retained polyline.
        let mut pts = Vec::new();
        for i in 0..=10 {
            pts.push(v(i as f64, 0.0));
        }
        for i in 1..=10 {
            pts.push(v(10.0, i as f64));
        }
        let eps = 0.01;
        let s = rdp_simplify(&pts, eps);
        assert!(s.iter().any(|p| (p - v(10.0, 0.0)).norm() < 1e-9), "the corner should be kept");
        // every original vertex is within ε of some retained segment
        for p in &pts {
            let d = s.windows(2).map(|w| point_segment_distance(p, &w[0], &w[1])).fold(f64::INFINITY, f64::min);
            assert!(d < eps + 1e-9, "dropped vertex {p:?} exceeds ε: {d}");
        }
    }

    #[test]
    fn resampling_gives_uniform_spacing_and_preserves_length() {
        let pts = [v(0.0, 0.0), v(3.0, 0.0), v(3.0, 4.0)]; // total length 3 + 4 = 7
        let r = resample_uniform(&pts, 0.5);
        // interior gaps are ~uniform
        for w in r.windows(2).take(r.len() - 2) {
            assert!(((w[1] - w[0]).norm() - 0.5).abs() < 1e-9, "gap should be 0.5: {}", (w[1] - w[0]).norm());
        }
        assert!((r[0] - pts[0]).norm() < 1e-12 && (r.last().unwrap() - pts.last().unwrap()).norm() < 1e-12, "endpoints preserved");
        assert!((polyline_length(&r) - 7.0).abs() < 1e-9, "length preserved: {}", polyline_length(&r));
    }

    #[test]
    fn polyline_length_sums_the_segments() {
        assert!((polyline_length(&[v(0.0, 0.0), v(3.0, 4.0), v(3.0, 4.0 + 2.0)]) - 7.0).abs() < 1e-12);
    }
}
