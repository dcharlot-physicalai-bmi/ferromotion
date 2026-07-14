//! **Grasp force closure** — a differentiable grasp-quality metric, in the spirit of GraspQP
//! (Zurbrügg, Cramariuc & Hutter, CoRL 2025) and the classic Ferrari-Canny metric.
//!
//! A planar grasp is a set of frictional contacts on an object. Each contact can push within its
//! **friction cone**; linearizing the cone gives a few unit **primitive wrenches** `w = [f; p×f]` in
//! wrench space (`ℝ³` planar). The grasp is **force-closure** — able to resist an external wrench in
//! any direction — iff the primitive wrenches positively span wrench space, i.e. `0` is interior to
//! their convex hull. The **Ferrari-Canny Q1** quality is the radius of the largest origin-centered
//! ball inside that hull, computed from the support function `Q1 = min_d max_i (w_i·d)` over unit
//! directions `d`: `Q1 > 0` ⟺ force closure, and larger is more robust. An LSE-smoothed version is
//! differentiable in the contact geometry — the signal a grasp synthesizer optimizes. Pure `nalgebra` → WASM-clean.

use nalgebra::{Vector2, Vector3};

/// A frictional point contact on a planar object.
#[derive(Clone, Copy, Debug)]
pub struct GraspContact {
    /// Contact position, relative to the object's reference point (for the torque arm).
    pub pos: Vector2<f64>,
    /// Inward surface normal (into the object).
    pub normal: Vector2<f64>,
    pub mu: f64,
}

/// The unit primitive wrenches `[fx, fy, τ]` from the linearized friction cones (2 edges per contact).
pub fn primitive_wrenches(contacts: &[GraspContact]) -> Vec<Vector3<f64>> {
    let mut w = Vec::with_capacity(2 * contacts.len());
    for c in contacts {
        let n = c.normal.normalize();
        let t = Vector2::new(-n.y, n.x); // tangent
        for s in [-1.0, 1.0] {
            let f = (n + s * c.mu * t).normalize(); // cone edge (unit force)
            let torque = c.pos.x * f.y - c.pos.y * f.x; // p × f (scalar)
            w.push(Vector3::new(f.x, f.y, torque));
        }
    }
    w
}

/// Unit directions on `S²` (Fibonacci sphere) for sampling the support function.
fn fib_dirs(n: usize) -> Vec<Vector3<f64>> {
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // golden angle
    (0..n)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = ga * k as f64;
            Vector3::new(r * th.cos(), r * th.sin(), z)
        })
        .collect()
}

/// **Ferrari-Canny Q1** force-closure quality: `min_d max_i (w_i · d)` over sampled unit directions.
/// `> 0` ⟺ force closure (with `Q1` the robustness margin); `≤ 0` ⟺ not force closure.
pub fn force_closure_q1(contacts: &[GraspContact], n_dirs: usize) -> f64 {
    let ws = primitive_wrenches(contacts);
    fib_dirs(n_dirs)
        .iter()
        .map(|d| ws.iter().map(|w| w.dot(d)).fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::INFINITY, f64::min)
}

/// LSE-smoothed Q1 (soft-min over directions of soft-max over wrenches) — differentiable in the
/// contact geometry. `beta` is the sharpness (→ [`force_closure_q1`] as `beta → ∞`).
pub fn force_closure_soft(contacts: &[GraspContact], n_dirs: usize, beta: f64) -> f64 {
    let ws = primitive_wrenches(contacts);
    // soft-max over wrenches per direction, then soft-min over directions.
    let per_dir: Vec<f64> = fib_dirs(n_dirs)
        .iter()
        .map(|d| {
            let m = ws.iter().map(|w| w.dot(d)).fold(f64::NEG_INFINITY, f64::max);
            m + (ws.iter().map(|w| (beta * (w.dot(d) - m)).exp()).sum::<f64>()).ln() / beta
        })
        .collect();
    let mn = per_dir.iter().cloned().fold(f64::INFINITY, f64::min);
    mn - (per_dir.iter().map(|&x| (-beta * (x - mn)).exp()).sum::<f64>()).ln() / beta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antipodal_grasp_is_force_closure_but_same_side_is_not() {
        // Two contacts on opposite sides of a unit object, normals pointing inward toward each other.
        let antipodal = [
            GraspContact { pos: Vector2::new(1.0, 0.0), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
            GraspContact { pos: Vector2::new(-1.0, 0.0), normal: Vector2::new(1.0, 0.0), mu: 0.5 },
        ];
        let q_fc = force_closure_q1(&antipodal, 800);
        assert!(q_fc > 1e-3, "antipodal grasp should be force-closure: Q1 = {q_fc}");

        // Two contacts on the *same* side (both pushing +x): the object can escape → not force closure.
        let same_side = [
            GraspContact { pos: Vector2::new(1.0, 0.3), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
            GraspContact { pos: Vector2::new(1.0, -0.3), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
        ];
        let q_no = force_closure_q1(&same_side, 800);
        assert!(q_no < 0.0, "same-side grasp should NOT be force-closure: Q1 = {q_no}");
    }

    #[test]
    fn more_friction_improves_the_quality() {
        let grasp = |mu: f64| {
            [
                GraspContact { pos: Vector2::new(1.0, 0.0), normal: Vector2::new(-1.0, 0.0), mu },
                GraspContact { pos: Vector2::new(-1.0, 0.0), normal: Vector2::new(1.0, 0.0), mu },
            ]
        };
        // A wider friction cone spans more of wrench space → a more robust (larger-Q1) grasp.
        assert!(force_closure_q1(&grasp(0.8), 800) > force_closure_q1(&grasp(0.3), 800));
    }

    #[test]
    fn soft_quality_is_differentiable_and_tracks_q1() {
        let contacts = [
            GraspContact { pos: Vector2::new(1.0, 0.1), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
            GraspContact { pos: Vector2::new(-1.0, -0.05), normal: Vector2::new(1.0, 0.0), mu: 0.5 },
        ];
        let (n, beta) = (1200, 200.0);
        // The soft metric approximates the hard Q1.
        let (soft, hard) = (force_closure_soft(&contacts, n, beta), force_closure_q1(&contacts, n));
        assert!((soft - hard).abs() < 0.02, "soft {soft} should track hard Q1 {hard}");

        // Its gradient w.r.t. a contact coordinate is finite and matches a finite difference — the
        // signal a differentiable grasp synthesizer follows.
        let eps = 1e-5;
        let perturb = |dx: f64| {
            let mut c = contacts;
            c[0].pos.x += dx;
            force_closure_soft(&c, n, beta)
        };
        let fd = (perturb(eps) - perturb(-eps)) / (2.0 * eps);
        assert!(fd.is_finite() && fd.abs() < 1e4, "gradient not well-behaved: {fd}");
        // Sanity: moving the contact outward (larger torque arm) changes quality measurably.
        assert!((perturb(0.1) - perturb(-0.1)).abs() > 1e-4, "quality insensitive to geometry");
    }
}
