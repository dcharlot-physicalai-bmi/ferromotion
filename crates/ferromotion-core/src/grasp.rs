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
//!
//! **The `Q1 > 0 ⟺ force closure` biconditional needs the rank condition to hold, and sampling alone
//! cannot supply it (2026-08-14).** Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic
//! Manipulation*, Prop. 5.2 states force closure as "`G` surjective **and** a strictly internal force
//! exists", and Prop. 5.3 gives the geometric equivalents: the wrenches must *positively span* wrench
//! space, equivalently their hull must contain a *neighbourhood* of the origin. A rank-deficient set fails
//! both, so its true `Q1` is exactly `0` — but a minimum over finitely many directions never samples the
//! orthogonal direction that would show it, and returns a small positive number instead. So
//! [`force_closure_q1`] tests [`wrench_rank`] first. Note MLS Prop. 5.3 condition 4 also gives an *exact*
//! combinatorial test for frictionless point contacts, which needs no sampling at all.

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

/// **Numerical rank of the planar wrench set.** Below `3` the grasp cannot resist some wrench at any
/// magnitude, so the true `Q1` is exactly zero however comfortable a sampled estimate looks.
///
/// This is the surjectivity requirement of the definition, not a numerical nicety. Murray, Li & Sastry
/// (1994), *A Mathematical Introduction to Robotic Manipulation*, Proposition 5.2: a grasp is
/// force-closure **iff `G` is surjective** and there is a strictly internal force. Proposition 5.3 gives
/// the equivalent geometric statements — the columns of `G` must *positively span* `Rᵖ`, and their convex
/// hull must contain a *neighbourhood of the origin*. A rank-deficient set does neither.
pub fn wrench_rank(contacts: &[GraspContact]) -> usize {
    let ws = primitive_wrenches(contacts);
    if ws.is_empty() {
        return 0;
    }
    let g = nalgebra::DMatrix::from_fn(3, ws.len(), |r, c| ws[c][r]);
    let sv = g.svd(false, false).singular_values;
    let s_max = sv.iter().fold(0.0f64, |a, b| a.max(*b));
    if s_max <= 0.0 {
        return 0;
    }
    sv.iter().filter(|s| **s > 1e-9 * s_max).count()
}

/// **Ferrari-Canny Q1** force-closure quality: `min_d max_i (w_i · d)` over sampled unit directions.
///
/// Returns exactly `0.0` when [`wrench_rank`] is below `3` — see that function for why this is the
/// definition rather than a guard.
///
/// **The returned value is an UPPER bound on the true `Q1`, and the gate above is what makes it safe
/// (2026-08-14).** A minimum over a *finite* set of directions can only overestimate a minimum over all
/// directions. Where the wrench set is rank-deficient that overestimate is unbounded in the worst way: a
/// direction orthogonal to the spanned subspace gives `max_i (w_i · d) = 0` exactly, so the true `Q1` is
/// `0` and the grasp is **not** force closure — but no sampled direction is ever exactly orthogonal, so
/// the estimate came back positive and merely shrank as `~1/n_dirs`, never reaching zero. Measured before
/// the gate: three frictionless contacts at 120° on a disk, whose lines of action all pass through the
/// reference point so every torque is identically zero, returned `+0.0250` at 800 directions and `+0.0204`
/// at 1200 — a "robustness margin" for a grasp that cannot resist a pure moment at any magnitude. The
/// failure was worst exactly where it mattered most.
///
/// **A negative return is information, so it is NOT clamped.** When the wrench set is full rank but the
/// origin lies *outside* the hull, the sampled minimum is genuinely negative and its magnitude says how far
/// the grasp is from closure — the same-side two-contact grasp in the tests below measures `−0.858`. A first
/// version of this gate also clamped with `.max(0.0)`, copying the 6-D sibling, and that flattened
/// `−0.858` to `0`: it destroys the gradient a synthesiser descends on in exactly the region where
/// optimisation starts, and it conflates "cannot resist a wrench in some direction at any magnitude"
/// (rank-deficient, truly `0`) with "resists nothing yet, by this margin". The rank gate is the correct
/// place to return `0`; the sign of a full-rank result is not.
///
/// The 6-D sibling [`crate::grasp_spatial::force_closure_q1_spatial`] has gated on rank since it was
/// written — but it *does* carry that `.max(0.0)` clamp, and so loses the same signal.
pub fn force_closure_q1(contacts: &[GraspContact], n_dirs: usize) -> f64 {
    if contacts.is_empty() || wrench_rank(contacts) < 3 {
        return 0.0;
    }
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

    /// A rank-deficient grasp scores exactly zero, per Murray, Li & Sastry Prop. 5.2/5.3: force closure
    /// requires the grasp map to be surjective, equivalently that the primitive wrenches positively span
    /// wrench space. Direction sampling alone cannot deliver that answer.
    #[test]
    fn a_rank_deficient_grasp_scores_exactly_zero_not_a_sampling_artifact() {
        // Frictionless contacts on the unit disk with inward radial normals: every line of action passes
        // through the reference point, so every torque is identically zero and the wrench set lives in the
        // (fx, fy) plane. Rank 2 of 3 — the grasp cannot resist a pure moment at ANY magnitude.
        let radial = |deg: f64| {
            let a = deg.to_radians();
            let p = Vector2::new(a.cos(), a.sin());
            GraspContact { pos: p, normal: -p, mu: 0.0 }
        };
        let flat = [radial(0.0), radial(120.0), radial(240.0)];
        assert_eq!(wrench_rank(&flat), 2, "this fixture must be rank-deficient to test the gate");
        for n in [200, 800, 1200, 4000] {
            assert_eq!(
                force_closure_q1(&flat, n),
                0.0,
                "a rank-deficient grasp must be exactly 0 at every sampling density; the un-gated \
                 estimator returned +0.0250 at 800 directions and +0.0204 at 1200, decaying as ~1/n and \
                 never reaching zero"
            );
        }

        // Two antipodal frictionless contacts: wrench rank 1 (a single force line, no torque).
        let two = [radial(0.0), radial(180.0)];
        assert!(wrench_rank(&two) < 3, "antipodal frictionless is rank-deficient");
        assert_eq!(force_closure_q1(&two, 800), 0.0);

        // Friction restores the missing directions, so the SAME positions become force closure once mu > 0
        // — the gate must not swallow a genuinely good grasp.
        let gripped: Vec<GraspContact> =
            [0.0, 120.0, 240.0].iter().map(|d| GraspContact { mu: 0.6, ..radial(*d) }).collect();
        assert_eq!(wrench_rank(&gripped), 3, "friction should restore full rank");
        assert!(force_closure_q1(&gripped, 800) > 1e-3, "a frictional 3-contact grasp is force closure");
    }

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
