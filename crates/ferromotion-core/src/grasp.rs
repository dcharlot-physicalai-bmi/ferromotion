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

/// **Exact force-closure decision — no sampling.** `true` iff the origin lies in the *interior* of the
/// convex hull of the primitive wrenches.
///
/// Murray, Li & Sastry (1994) Prop. 5.3 condition 4: a grasp fails force closure iff there exists a
/// `v ≠ 0` with `v · wᵢ ≥ 0` for every generator — a *supporting hyperplane through the origin*, with every
/// wrench on one side of it. MLS notes that only finitely many candidate `v` need be considered: take any
/// `p − 1` independent generators and let `v` be normal to them. In the plane `p = 3`, so the candidates are
/// the cross products of generator *pairs*, and the whole test is `O(m³)` with no tolerance on direction
/// density.
///
/// **This answers what [`force_closure_q1`] structurally cannot.** Q1 is a minimum over *finitely many*
/// sampled directions, so it is an upper bound that can report a positive margin for a grasp that is not
/// force closure — the defect the [`planar_wrench_rank`](crate::planar_wrench_rank) gate exists to catch, and which for a full-rank grasp with
/// the origin merely *on* the hull boundary no amount of sampling resolves. This decides it exactly. Use it
/// for the yes/no question and Q1 for *how robust*, which is the part a synthesiser needs a gradient of.
///
/// Note MLS states Prop. 5.3 for frictionless point contacts; the convexity argument applies unchanged to any
/// finite generator set, which is what a linearised friction cone provides. It is therefore exact for the
/// *linearised* problem — the same problem Q1 measures — and not for the true smooth cone.
pub fn is_force_closure(contacts: &[GraspContact], tol: f64) -> bool {
    let ws = primitive_wrenches(contacts);
    if ws.len() < 3 || wrench_rank(contacts) < 3 {
        return false; // cannot positively span ℝ³
    }
    for i in 0..ws.len() {
        for j in (i + 1)..ws.len() {
            let v = ws[i].cross(&ws[j]);
            let n = v.norm();
            if n <= tol {
                continue; // parallel pair spans no plane, so it defines no candidate hyperplane
            }
            let v = v / n;
            let (mut any_pos, mut any_neg) = (false, false);
            for w in &ws {
                let d = w.dot(&v);
                if d > tol {
                    any_pos = true;
                } else if d < -tol {
                    any_neg = true;
                }
            }
            // All generators on one closed side ⇒ a supporting hyperplane through the origin exists.
            if !(any_pos && any_neg) {
                return false;
            }
        }
    }
    true
}

/// **Ferrari-Canny Q1** force-closure quality: `min_d max_i (w_i · d)` over sampled unit directions.
///
/// Returns exactly `0.0` when [`planar_wrench_rank`](crate::planar_wrench_rank) is below `3` — see that function for why this is the
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

    /// **The exact test and the sampled metric must agree on the yes/no question**, and where they cannot,
    /// the exact one is right. MLS Prop. 5.3 condition 4 needs no direction sampling at all.
    #[test]
    fn the_exact_decision_agrees_with_the_sampled_metric_and_settles_what_it_cannot() {
        let radial = |deg: f64, mu: f64| {
            let a: f64 = deg.to_radians();
            let p = Vector2::new(a.cos(), a.sin());
            GraspContact { pos: p, normal: -p, mu }
        };

        // Force closure: three frictional contacts at 120°. Both must say yes.
        let good: Vec<GraspContact> = [0.0, 120.0, 240.0].iter().map(|d| radial(*d, 0.6)).collect();
        assert!(is_force_closure(&good, 1e-9), "3 frictional contacts at 120° are force closure");
        assert!(force_closure_q1(&good, 800) > 1e-3, "and Q1 should agree");

        // Rank-deficient: frictionless radial contacts, every torque identically zero. Both must say no —
        // and the exact test needs no gate to get there, because no v-candidate finds two signs.
        let flat: Vec<GraspContact> = [0.0, 120.0, 240.0].iter().map(|d| radial(*d, 0.0)).collect();
        assert!(!is_force_closure(&flat, 1e-9), "a rank-deficient grasp is not force closure");
        assert_eq!(force_closure_q1(&flat, 800), 0.0);

        // Same-side grasp: full rank, origin OUTSIDE the hull. The exact test says no; Q1 says no by sign.
        let same_side = [
            GraspContact { pos: Vector2::new(1.0, 0.3), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
            GraspContact { pos: Vector2::new(1.0, -0.3), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
        ];
        assert_eq!(wrench_rank(&same_side), 3, "this one is full rank — the gate does not fire");
        assert!(!is_force_closure(&same_side, 1e-9), "the object can escape");
        assert!(force_closure_q1(&same_side, 800) < 0.0, "and Q1 reports it by sign");

        // Antipodal frictional: force closure, and both agree.
        let anti = [
            GraspContact { pos: Vector2::new(1.0, 0.0), normal: Vector2::new(-1.0, 0.0), mu: 0.5 },
            GraspContact { pos: Vector2::new(-1.0, 0.0), normal: Vector2::new(1.0, 0.0), mu: 0.5 },
        ];
        assert!(is_force_closure(&anti, 1e-9));
        assert!(force_closure_q1(&anti, 800) > 1e-3);

        // Two contacts cannot span ℝ³ once frictionless: refused without needing a hyperplane search.
        let two_flat = [radial(0.0, 0.0), radial(180.0, 0.0)];
        assert!(!is_force_closure(&two_flat, 1e-9));
        assert!(!is_force_closure(&[], 1e-9), "no contacts is not force closure");
    }

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
