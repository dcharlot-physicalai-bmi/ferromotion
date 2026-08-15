//! **Grasp quality in the full six-dimensional wrench space.**
//!
//! [`grasp`](crate::grasp) computes Ferrari-Canny `Q1` for a *planar* grasp: contact positions and normals in `R^2`,
//! wrenches `[fx, fy, tau]` in `R^3`. A real hand grasping a real object works in `R^6`, and the planar number
//! **overstates** the spatial one — measured on three contacts 120 degrees apart on a unit disk:
//!
//! ```text
//!    mu    rank   planar Q1   spatial Q1   ratio
//!   0.0       2    0.005000     0.000000       -
//!   0.2       6    0.202708     0.136095    1.49
//!   0.4       6    0.379231     0.239409    1.58
//!   1.0       6    0.666100     0.434308    1.53
//! ```
//!
//! **A correction worth stating, because it was the premise this module was written to demonstrate.** The expectation
//! was that coplanar contacts would be rank-deficient and score exactly zero in six dimensions. They are not, once
//! there is any friction at all. For a normal lying in the plane, `n x t1` points out of it, so the friction cone
//! contains out-of-plane forces; applied at in-plane positions those generate in-plane moments, and the wrench set
//! reaches full rank the moment `mu > 0`. Friction at coplanar contacts genuinely does resist out-of-plane moments.
//! The rank goes `2 -> 6` between `mu = 0` and `mu = 0.1`, and the honest finding is the consistent factor of about
//! `1.5` rather than a collapse to zero.
//!
//! Two things this module still treats as first-class because they decide correctness:
//!
//! **Rank before magnitude.** `Q1` is estimated as `min_d max_i (w_i . d)` over sampled unit directions, and sampling a
//! finite set of directions gives an *upper* bound. When the wrench set is rank-deficient there exists a direction
//! orthogonal to every generator, so the true `Q1` is zero — but a sampler that misses that direction reports a
//! comfortable positive number. The failure is worst exactly where it matters, so [`wrench_rank`] is computed first and
//! [`force_closure_q1_spatial`] returns a hard zero on deficiency rather than trusting the sample.
//!
//! **Which way each approximation errs.** More cone facets enlarge the inscribed polyhedral cone, so `Q1` rises toward
//! the smooth-cone value **from below**. More sampled directions can only lower the minimum, so the estimate falls
//! toward the true value **from above**. Both are reported and both are checked in the tests, because "converges" with
//! no direction attached is not a useful statement. The consequence is that monotonicity in `mu` — which must hold for
//! the true `Q1` — only appears once the direction sampling is fine enough: it fails at 4096 directions on the grasp
//! above and holds at 20000.

use nalgebra::{DMatrix, Vector3, Vector6};

/// A frictional point contact on a three-dimensional object, optionally soft-fingered.
#[derive(Clone, Copy, Debug)]
pub struct GraspContact3 {
    /// Contact position relative to the object's reference point — the moment arm.
    pub pos: Vector3<f64>,
    /// Inward surface normal (into the object).
    pub normal: Vector3<f64>,
    /// Tangential Coulomb friction coefficient.
    pub mu: f64,
    /// Torsional friction coefficient for a soft finger. `0` is a hard point contact, which can transmit no moment
    /// about its own normal.
    pub mu_torsion: f64,
}

impl GraspContact3 {
    /// A hard point contact.
    pub fn hard(pos: Vector3<f64>, normal: Vector3<f64>, mu: f64) -> GraspContact3 {
        GraspContact3 { pos, normal, mu, mu_torsion: 0.0 }
    }

    /// Two unit tangents completing a right-handed frame with the normal, chosen to avoid the degenerate axis.
    fn tangents(&self) -> (Vector3<f64>, Vector3<f64>) {
        let n = self.normal.normalize();
        let seed = if n.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
        let t1 = (seed - n * n.dot(&seed)).normalize();
        (t1, n.cross(&t1))
    }
}

/// **The primitive wrenches** of the grasp: one per friction-cone edge, plus torsional generators for a soft finger.
///
/// Each wrench is `[f; p x f]` for a **unit-magnitude** cone-edge force `f`. That normalisation is the convention this
/// module works in, and it fixes what the torsional generator may be scaled by.
///
/// **A correction (2026-08-06).** This previously emitted `[0; +/- mu_torsion * n]` and documented the normal force as
/// "unit by construction here". It is not. The cone-edge force is `(n + mu*t).normalize()`, so its NORMAL component is
///
/// ```text
///   f.n = 1 / sqrt(1 + mu^2)
/// ```
///
/// measured over `mu` in `[0, 2]` and matching that expression to `2.2e-16`, not `1`. Taking the tangential and the
/// torsional bounds from two different normalisations overstated the torsional generator by `sqrt(1 + mu^2)`: `1.03x`
/// at `mu = 0.25`, `1.12x` at `0.5`, `1.41x` at `1.0`, `1.80x` at `1.5`, `2.24x` at `2.0`. So the generator is now
/// scaled by the normal force this normalisation actually delivers.
///
/// **What that was worth, measured.** `Q1` is exactly FLAT in `mu_torsion` until torsion becomes the binding
/// constraint, and the threshold rises steeply with `mu`: on the four-contact non-coplanar fixture in
/// `examples/soft_finger_sensitivity.rs` it binds from `mu_torsion ~ 0.28` at `mu = 0.3`, `~ 0.38` at `0.5`, `~ 0.70`
/// at `1.0`, and **never** up to `4.0` at `mu = 1.5`. Below the threshold the old and new generators give bit-identical
/// `Q1`; above it the old one overstated the grasp:
///
/// ```text
///   mu = 0.5, mu_torsion = 1.0    Q1  0.542841048750 -> 0.492818002608    +10.2% overstated
///   mu = 1.0, mu_torsion = 1.0        0.598498599390 -> 0.566294829236     +5.7%
///   mu = 1.5, mu_torsion = 1.0        0.634002191339    unchanged          torsion never binds
/// ```
///
/// The largest premise error therefore sits where it costs nothing, and the cost appears at moderate `mu` where the
/// premise error is small. A first probe that sampled only `mu_torsion` in `{0.1, 0.3}` reported "no change, ratio
/// 1.0000" in all twelve cases — every sample was inside the flat region. A null result is a claim about the sweep, not
/// about the code.
///
/// Every hard contact (`mu_torsion = 0`) is bit-identical across this change, so no published hard-grasp number moves.
///
/// **What the polyhedral form gives up.** The true soft-finger limit surface couples the bounds to the instantaneous
/// normal force (`|f_t| <= mu*f_n` and `|m_n| <= mu_torsion*f_n` at the same `f_n`). Decoupling them into independent
/// generators is what makes the set a polytope, and it means the torsional bound can only be scaled by a
/// representative normal force rather than the one a particular combination happens to use. This uses the largest
/// normal force the normalised edge set can reach, which is the hull's centroid direction, `1/sqrt(1 + mu^2)`.
pub fn primitive_wrenches_spatial(contacts: &[GraspContact3], facets: usize) -> Vec<Vector6<f64>> {
    let facets = facets.max(3);
    let mut out = Vec::with_capacity(contacts.len() * (facets + 2));
    for c in contacts {
        let n = c.normal.normalize();
        let (t1, t2) = c.tangents();
        for k in 0..facets {
            let th = std::f64::consts::TAU * k as f64 / facets as f64;
            let f = (n + c.mu * (th.cos() * t1 + th.sin() * t2)).normalize();
            let m = c.pos.cross(&f);
            out.push(Vector6::new(f.x, f.y, f.z, m.x, m.y, m.z));
        }
        if c.mu_torsion > 0.0 {
            // the normal force a unit-magnitude cone-edge force delivers — NOT 1, see the correction above
            let f_normal = 1.0 / (1.0 + c.mu * c.mu).sqrt();
            for s in [-1.0, 1.0] {
                let m = s * c.mu_torsion * f_normal * n;
                out.push(Vector6::new(0.0, 0.0, 0.0, m.x, m.y, m.z));
            }
        }
    }
    out
}

/// The grasp matrix `G` in `R^{6 x k}`, columns being the primitive wrenches.
pub fn grasp_matrix(contacts: &[GraspContact3], facets: usize) -> DMatrix<f64> {
    let ws = primitive_wrenches_spatial(contacts, facets);
    DMatrix::from_fn(6, ws.len(), |r, c| ws[c][r])
}

/// **Numerical rank of the wrench set.** Below `6` the grasp cannot resist some wrench at any magnitude, so `Q1` is
/// exactly zero however comfortable a sampled estimate looks.
pub fn wrench_rank(contacts: &[GraspContact3], facets: usize) -> usize {
    let g = grasp_matrix(contacts, facets);
    if g.ncols() == 0 {
        return 0;
    }
    let sv = g.svd(false, false).singular_values;
    let s_max = sv.iter().fold(0.0f64, |a, b| a.max(*b));
    if s_max <= 0.0 {
        return 0;
    }
    sv.iter().filter(|s| **s > 1e-9 * s_max).count()
}

/// Deterministic near-uniform unit directions on `S^5`.
///
/// A Fibonacci lattice is a two-dimensional construction and does not extend to `S^5`, so this uses a Kronecker
/// (additive-recurrence) sequence with irrational multipliers pushed through Box-Muller: Gaussian coordinates
/// normalised to the sphere are exactly uniform, and a low-discrepancy source keeps the coverage even.
fn sphere5_dirs(n: usize) -> Vec<Vector6<f64>> {
    const ALPHA: [f64; 6] = [0.618_033_988_749_895, 0.324_717_957_244_746, 0.213_797_951_074_697, 0.161_802_638_412_437, 0.130_224_026_469_147, 0.108_378_726_432_100];
    let mut out = Vec::with_capacity(n);
    for k in 1..=n {
        let u: Vec<f64> = (0..6).map(|i| (k as f64 * ALPHA[i]).fract().clamp(1e-12, 1.0 - 1e-12)).collect();
        // three Box-Muller pairs -> six standard normals
        let mut v = Vector6::zeros();
        for p in 0..3 {
            let (a, b) = (u[2 * p], u[2 * p + 1]);
            let r = (-2.0 * a.ln()).sqrt();
            v[2 * p] = r * (std::f64::consts::TAU * b).cos();
            v[2 * p + 1] = r * (std::f64::consts::TAU * b).sin();
        }
        let norm = v.norm();
        if norm > 1e-12 {
            out.push(v / norm);
        }
    }
    out
}

/// **Ferrari-Canny `Q1` in six dimensions.**
///
/// Returns exactly `0.0` when [`wrench_rank`] is below `6`: a rank-deficient grasp has a wrench it cannot resist, and
/// reporting a sampled minimum there would be reporting the sampler's blind spot as a safety margin.
///
/// Otherwise returns `min_d max_i (w_i . d)` over `n_dirs` sampled directions — an **upper bound** on the true `Q1`
/// that decreases as `n_dirs` grows.
///
/// **The negative branch is no longer clamped (2026-08-14).** This used to end in `.max(0.0)`. When the wrench
/// set is full rank but the origin lies outside the hull, the sampled minimum is genuinely negative and its
/// magnitude says how far the grasp is from closure — clamping conflates that with the rank-deficient case,
/// which is truly `0`, and it flattens the gradient a synthesiser descends on exactly where optimisation
/// starts. The rank gate above is the right place to return zero; the sign of a full-rank result is not.
///
/// **`Q1` in six dimensions is frame- and scale-dependent, and no choice of units repairs that.** Murray, Li
/// & Sastry, Appendix A.3.2, is explicit: *there is no inner product structure on `se(3)` invariant under
/// change of coordinate frame*, and they name the "incorrect use of the inner product on `R⁶` as an inner
/// product in `se(3)`" as a known source of confusion in the robotics literature. Sampling unit directions on
/// the Euclidean sphere `S⁵` is exactly that use: the sphere mixes newtons with newton-metres, so the value
/// depends on the length unit, and because `Ad_g` does not preserve the `R⁶` dot product for `p ≠ 0`, it also
/// depends on where the reference point is placed. Measured, the same physical grasp expressed at different
/// object scales gives values differing by 1.5x-3.8x.
///
/// A characteristic length would make the metric dimensionally consistent, which is worth doing, but it would
/// **not** make it invariant — MLS proves no such metric exists. So `Q1` values are comparable only across
/// grasps sharing one frame and one length scale, and a `Q1` quoted without them is not a physical quantity.
/// This is a real limitation of the metric rather than of this implementation, and it is why the ordering of
/// grasps is the trustworthy output here rather than the magnitude.
pub fn force_closure_q1_spatial(contacts: &[GraspContact3], facets: usize, n_dirs: usize) -> f64 {
    if contacts.is_empty() || wrench_rank(contacts, facets) < 6 {
        return 0.0;
    }
    let ws = primitive_wrenches_spatial(contacts, facets);
    sphere5_dirs(n_dirs)
        .iter()
        .map(|d| ws.iter().map(|w| w.dot(d)).fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::INFINITY, f64::min)
}

/// LSE-smoothed spatial `Q1`, differentiable in the contact geometry — the signal a grasp synthesiser can descend on.
/// `beta` is the sharpness; the value approaches [`force_closure_q1_spatial`] as `beta` grows.
///
/// The rank gate is deliberately **not** applied here: a hard zero has no gradient, and a synthesiser needs a slope
/// that points out of a rank-deficient configuration. Use [`wrench_rank`] to decide whether the result is a quality or
/// merely a direction of improvement.
pub fn force_closure_soft_spatial(contacts: &[GraspContact3], facets: usize, n_dirs: usize, beta: f64) -> f64 {
    let ws = primitive_wrenches_spatial(contacts, facets);
    if ws.is_empty() || beta <= 0.0 {
        return 0.0;
    }
    let per_dir: Vec<f64> = sphere5_dirs(n_dirs)
        .iter()
        .map(|d| {
            let vals: Vec<f64> = ws.iter().map(|w| w.dot(d)).collect();
            let m = vals.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            m + (vals.iter().map(|v| (beta * (v - m)).exp()).sum::<f64>().ln()) / beta
        })
        .collect();
    let m = per_dir.iter().fold(f64::INFINITY, |a, b| a.min(*b));
    m - (per_dir.iter().map(|v| (-beta * (v - m)).exp()).sum::<f64>().ln()) / beta
}

/// **`Q1` restricted to the planar wrench subspace** `(fx, fy, mz)` — the three coordinates the planar metric uses.
///
/// This is what makes the comparison against [`grasp`](crate::grasp) meaningful: the planar metric is not wrong, it is
/// computing the quality of a different, smaller wrench space, and restricting the spatial computation to that subspace
/// recovers it.
pub fn force_closure_q1_planar_subspace(contacts: &[GraspContact3], facets: usize, n_dirs: usize) -> f64 {
    let ws = primitive_wrenches_spatial(contacts, facets);
    if ws.is_empty() {
        return 0.0;
    }
    // sample directions in the 3-dimensional subspace spanned by (fx, fy, mz)
    let ga = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n_dirs)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / n_dirs as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let th = ga * k as f64;
            Vector3::new(r * th.cos(), r * th.sin(), z)
        })
        .map(|d| ws.iter().map(|w| w[0] * d.x + w[1] * d.y + w[5] * d.z).fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::INFINITY, f64::min)
}

/// How the grasp's force capacity splits: wrenches it can apply to the object, against internal squeeze it can apply
/// to itself.
#[derive(Clone, Debug)]
pub struct GraspSplit {
    /// Rank of `G` — the dimension of the wrench space the grasp can command.
    pub rank: usize,
    /// Dimension of `null(G)`: contact-force combinations producing **no** net wrench. This is the squeeze a hand uses
    /// to hold an object without moving it, and it is invisible to any net-wrench metric.
    pub internal_dim: usize,
    /// An orthonormal basis of `null(G)`, one column per internal direction.
    pub internal_basis: DMatrix<f64>,
}

/// Split the grasp matrix into what it can command and what it can only squeeze with.
pub fn grasp_split(contacts: &[GraspContact3], facets: usize) -> GraspSplit {
    let g = grasp_matrix(contacts, facets);
    let k = g.ncols();
    if k == 0 {
        return GraspSplit { rank: 0, internal_dim: 0, internal_basis: DMatrix::zeros(0, 0) };
    }
    let svd = g.clone().svd(false, true);
    let s_max = svd.singular_values.iter().fold(0.0f64, |a, b| a.max(*b));
    let cut = 1e-9 * s_max.max(1e-300);
    let rank = svd.singular_values.iter().filter(|s| **s > cut).count();
    let v = svd.v_t.as_ref().map(|vt| vt.transpose());
    let null_cols: Vec<usize> = (0..svd.singular_values.len()).filter(|i| svd.singular_values[*i] <= cut).collect();
    // right-singular vectors past the rank, plus the columns the thin SVD does not return at all
    let internal_dim = k - rank;
    let internal_basis = match v {
        Some(v) => DMatrix::from_fn(k, null_cols.len(), |r, c| v[(r, null_cols[c])]),
        None => DMatrix::zeros(k, 0),
    };
    GraspSplit { rank, internal_dim, internal_basis }
}

/// Convenience: the net wrench produced by a set of non-negative generator magnitudes.
pub fn net_wrench(contacts: &[GraspContact3], facets: usize, magnitudes: &[f64]) -> Option<Vector6<f64>> {
    let ws = primitive_wrenches_spatial(contacts, facets);
    (ws.len() == magnitudes.len()).then(|| ws.iter().zip(magnitudes).map(|(w, a)| w * *a).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{force_closure_q1, GraspContact};
    use nalgebra::Vector2;

    /// Three contacts 120 degrees apart on a unit disk in the `z = 0` plane, normals pointing inward.
    fn three_coplanar(mu: f64) -> Vec<GraspContact3> {
        (0..3)
            .map(|k| {
                let th = std::f64::consts::TAU * k as f64 / 3.0;
                let pos = Vector3::new(th.cos(), th.sin(), 0.0);
                GraspContact3::hard(pos, -pos, mu)
            })
            .collect()
    }

    /// The same grasp in the planar formulation, for the comparison.
    fn three_coplanar_planar(mu: f64) -> Vec<GraspContact> {
        (0..3)
            .map(|k| {
                let th = std::f64::consts::TAU * k as f64 / 3.0;
                let pos = Vector2::new(th.cos(), th.sin());
                GraspContact { pos, normal: -pos, mu }
            })
            .collect()
    }

    /// **The correction.** The planar metric consistently overstates the spatial quality, by about `1.5x` here.
    #[test]
    fn the_planar_metric_overstates_the_spatial_quality() {
        eprintln!("{:>6}  {:>6}  {:>11}  {:>11}  {:>7}", "mu", "rank", "planar Q1", "spatial Q1", "ratio");
        let mut ratios = Vec::new();
        for mu in [0.2, 0.4, 0.6, 0.8, 1.0] {
            let cs = three_coplanar(mu);
            let (r, sp, pl) = (wrench_rank(&cs, 24), force_closure_q1_spatial(&cs, 24, 20_000), force_closure_q1(&three_coplanar_planar(mu), 20_000));
            eprintln!("{mu:>6.1}  {r:>6}  {pl:>11.6}  {sp:>11.6}  {:>7.2}", pl / sp);
            assert!(sp > 0.0 && pl > sp, "the planar metric is the larger number: {pl:.6} vs {sp:.6}");
            ratios.push(pl / sp);
        }
        let (lo, hi) = ratios.iter().fold((f64::INFINITY, 0.0f64), |(a, b), r| (a.min(*r), b.max(*r)));
        eprintln!("   overstatement ranges {lo:.2}x to {hi:.2}x across friction");
        assert!(lo > 1.3 && hi < 1.9, "a consistent factor, not a collapse: {lo:.2} to {hi:.2}");
    }

    /// **The premise this module was written to demonstrate was wrong, and this records why.**
    ///
    /// Coplanar contacts were expected to be rank-deficient. With any friction they are not: `n x t1` leaves the plane,
    /// so the cone contains out-of-plane forces, and those at in-plane positions produce in-plane moments. Rank goes
    /// `2 -> 6` as soon as `mu > 0`.
    #[test]
    fn friction_gives_coplanar_contacts_out_of_plane_capacity() {
        eprintln!("wrench rank of three coplanar contacts, against friction:");
        for mu in [0.0, 0.001, 0.1, 0.4] {
            let r = wrench_rank(&three_coplanar(mu), 24);
            eprintln!("   mu = {mu:<6}: rank {r} of 6");
            if mu == 0.0 {
                assert_eq!(r, 2, "frictionless radial forces on a disk span only (fx, fy)");
            } else {
                assert_eq!(r, 6, "any friction at all reaches full rank");
            }
        }
        // and the out-of-plane component is real, not a rounding artefact
        let ws = primitive_wrenches_spatial(&three_coplanar(0.4), 24);
        let max_fz = ws.iter().map(|w| w[2].abs()).fold(0.0, f64::max);
        eprintln!("   largest out-of-plane force component among the cone edges: {max_fz:.4}");
        assert!(max_fz > 0.3, "the friction cone genuinely leaves the plane: {max_fz:.4}");
    }

    /// **The rank gate earns its place** on the frictionless grasp: the true `Q1` is zero, and un-gated sampling reports
    /// a positive margin at every sample count because it never lands exactly on a null direction.
    #[test]
    fn the_rank_gate_catches_what_sampling_cannot() {
        let cs = three_coplanar(0.0);
        assert_eq!(wrench_rank(&cs, 24), 2, "rank 2 of 6");
        let ws = primitive_wrenches_spatial(&cs, 24);
        eprintln!("un-gated sampled minimum for a rank-2 grasp (true Q1 = 0):");
        for n in [64usize, 512, 4096, 20_000] {
            let raw = sphere5_dirs(n)
                .iter()
                .map(|d| ws.iter().map(|w| w.dot(d)).fold(f64::NEG_INFINITY, f64::max))
                .fold(f64::INFINITY, f64::min);
            eprintln!("   {n:>6} directions: {raw:.6}");
            assert!(raw > 0.0, "sampling never finds the null direction and claims {raw:.6} at {n} samples");
        }
        assert_eq!(force_closure_q1_spatial(&cs, 24, 20_000), 0.0, "the gated function returns exactly zero");
        eprintln!("   Rank has to be computed, not sampled for.");
    }

    /// A fourth contact off the plane keeps the grasp full rank and raises its quality.
    #[test]
    fn a_non_coplanar_contact_raises_the_quality() {
        let base = three_coplanar(0.5);
        let mut with_pole = base.clone();
        with_pole.push(GraspContact3::hard(Vector3::new(0.0, 0.0, 1.0), -Vector3::z(), 0.5));
        let (q3, q4) = (force_closure_q1_spatial(&base, 24, 20_000), force_closure_q1_spatial(&with_pole, 24, 20_000));
        eprintln!("three contacts: Q1 = {q3:.6}; adding a fourth on the pole: Q1 = {q4:.6}");
        assert_eq!(wrench_rank(&with_pole, 24), 6);
        assert!(q4 > q3, "an extra contact never reduces the quality: {q4:.6} vs {q3:.6}");
    }

    /// **Which way each approximation errs**, and the estimator consequence for monotonicity in friction.
    #[test]
    fn the_two_approximations_err_in_known_directions() {
        let cs = three_coplanar(0.6);
        eprintln!("facets (inscribed cone grows -> Q1 rises from below), 8192 directions:");
        let mut prev = -1.0;
        for f in [4usize, 8, 16, 32, 64] {
            let q = force_closure_q1_spatial(&cs, f, 8192);
            eprintln!("   {f:>3} facets: Q1 = {q:.6}");
            assert!(q >= prev - 1e-9, "more facets cannot shrink the inscribed cone: {q:.6} after {prev:.6}");
            prev = q;
        }
        eprintln!("directions (min over a larger set -> Q1 falls from above), 32 facets:");
        let mut prev = f64::INFINITY;
        for n in [256usize, 1024, 4096, 16_384] {
            let q = force_closure_q1_spatial(&cs, 32, n);
            eprintln!("   {n:>6} directions: Q1 = {q:.6}");
            assert!(q <= prev + 1e-9, "more directions cannot raise the minimum: {q:.6} after {prev:.6}");
            prev = q;
        }
    }

    /// Monotone in friction — but only once the direction sampling is fine enough, which is a property of the estimator
    /// and not of the quantity. At 4096 directions it fails; at 20000 it holds.
    #[test]
    fn monotonicity_in_friction_needs_enough_directions() {
        let q = |mu: f64, n: usize| force_closure_q1_spatial(&three_coplanar(mu), 24, n);
        eprintln!("Q1 against friction, at two sampling densities:");
        for n in [4096usize, 20_000] {
            let vals: Vec<f64> = [0.6, 0.8, 1.0].iter().map(|m| q(*m, n)).collect();
            let mono = vals.windows(2).all(|w| w[1] >= w[0] - 1e-12);
            eprintln!("   {n:>6} directions: {:.6}, {:.6}, {:.6}  monotone = {mono}", vals[0], vals[1], vals[2]);
            if n >= 20_000 {
                assert!(mono, "monotone once the sampling is fine enough: {vals:?}");
            }
        }
        // the coarse sampling genuinely violates it, which is why the density matters
        let coarse: Vec<f64> = [0.6, 0.8, 1.0].iter().map(|m| q(*m, 4096)).collect();
        assert!(coarse[2] < coarse[1], "at 4096 directions the estimate is not monotone: {coarse:?}");
    }

    /// **Soft-finger torsion adds rank but does not rescue a frictionless coplanar grasp**, which is worth knowing
    /// before reaching for it as a fix.
    #[test]
    fn soft_finger_torsion_adds_rank_without_rescuing_a_flat_grasp() {
        eprintln!("frictionless coplanar contacts, adding torsional capacity:");
        for mt in [0.0, 0.1, 0.3] {
            let cs: Vec<GraspContact3> = three_coplanar(0.0).iter().map(|c| GraspContact3 { mu_torsion: mt, ..*c }).collect();
            let (r, q) = (wrench_rank(&cs, 24), force_closure_q1_spatial(&cs, 24, 20_000));
            eprintln!("   mu_torsion = {mt:.1}: rank {r} of 6, Q1 = {q:.6}");
            if mt > 0.0 {
                assert_eq!(r, 4, "torsion adds two wrench dimensions");
            }
            assert_eq!(q, 0.0, "still rank-deficient, so still exactly zero");
        }
        eprintln!("   Rank 2 -> 4, but 4 is not 6: torsion alone cannot close a flat frictionless grasp.");
    }

    /// **The torsional generator is scaled by the normal force the normalisation actually delivers, not by 1.**
    ///
    /// The doc comment used to claim the normal force was "unit by construction". It is `1/sqrt(1 + mu^2)`. This pins
    /// both halves: the normal component of every cone edge, and the resulting torsional generator magnitude.
    #[test]
    fn the_cone_edge_normal_force_is_not_unit_and_the_torsion_is_scaled_by_it() {
        for mu in [0.0f64, 0.25, 0.5, 1.0, 1.5, 2.0] {
            let c = GraspContact3 { pos: Vector3::new(1.0, 0.0, 0.0), normal: -Vector3::x(), mu, mu_torsion: 0.7 };
            let n = c.normal.normalize();
            let ws = primitive_wrenches_spatial(std::slice::from_ref(&c), 24);
            let expected = 1.0 / (1.0 + mu * mu).sqrt();

            for w in &ws {
                let f = Vector3::new(w[0], w[1], w[2]);
                if f.norm() < 1e-12 {
                    // a torsional generator: pure moment about the normal, magnitude mu_torsion * f.n
                    let m = Vector3::new(w[3], w[4], w[5]);
                    assert!(
                        (m.norm() - 0.7 * expected).abs() < 1e-12,
                        "mu={mu}: torsional generator is {} but the normal force available is {expected}, so it must be {}",
                        m.norm(),
                        0.7 * expected
                    );
                    assert!(m.cross(&n).norm() < 1e-12, "the torsional moment must lie along the normal");
                } else {
                    assert!((f.norm() - 1.0).abs() < 1e-12, "cone-edge forces are unit magnitude by convention");
                    assert!(
                        (f.dot(&n) - expected).abs() < 1e-12,
                        "mu={mu}: cone-edge normal force is {} not the claimed 1.0 (expected {expected})",
                        f.dot(&n)
                    );
                }
            }
        }
    }

    /// **The sweep that catches it — past the point where torsion BINDS.**
    ///
    /// `Q1` is exactly flat in `mu_torsion` until torsion becomes the binding constraint, so a sweep that stays below
    /// the threshold reports "no change" no matter how wrong the generator is. That is how the mixed convention
    /// survived: a probe over `mu_torsion` in `{0.1, 0.3}` found ratio 1.0000 in twelve cases. This test asserts the
    /// fixture is past the threshold FIRST, then asserts the value.
    #[test]
    fn q1_responds_to_the_torsional_scaling_once_torsion_actually_binds() {
        // non-coplanar so Q1 is not identically zero, matching examples/soft_finger_sensitivity.rs
        let grasp = |mu: f64, mt: f64| -> Vec<GraspContact3> {
            [
                Vector3::new(1.0, 0.0, -0.35),
                Vector3::new(-0.5, 0.866, -0.35),
                Vector3::new(-0.5, -0.866, -0.35),
                Vector3::new(0.0, 0.0, 1.0),
            ]
            .iter()
            .map(|d| {
                let p = d.normalize();
                GraspContact3 { pos: p, normal: -p, mu, mu_torsion: mt }
            })
            .collect()
        };

        let (mu, mt, dirs) = (0.5f64, 1.0f64, 20_000);
        let base = force_closure_q1_spatial(&grasp(mu, 0.0), 24, dirs);
        let soft = force_closure_q1_spatial(&grasp(mu, mt), 24, dirs);

        // 1. the fixture must be PAST the binding threshold, or the rest of this test proves nothing
        assert!(
            soft > base * 1.01,
            "mu_torsion={mt} does not bind at mu={mu} (Q1 {base:.9} -> {soft:.9}); the sweep is inside the flat \
             region and would pass against any torsional scaling"
        );

        // 2. and the VALUE must be the scaled one. Comparing Q1 at two mu_torsion values does NOT test this — Q1 rises
        //    with generator magnitude either way, so that comparison passes against the unscaled generator too (it
        //    did). The only discriminating assertion is the absolute value, which the two conventions separate by 10%.
        //    This is a deterministic Kronecker sweep with no RNG, so 0.5% is a safe band and still rules the other out.
        let f_normal = 1.0 / (1.0 + mu * mu).sqrt();
        let scaled_expected = 0.492_818_002_608; // torsion scaled by f.n = 0.894427
        let unscaled_would_be = 0.542_841_048_750; // torsion scaled as though f.n = 1
        assert!(
            (soft / scaled_expected - 1.0).abs() < 5e-3,
            "Q1 is {soft:.12}; expected {scaled_expected:.12} with torsion scaled by f.n={f_normal:.6}. \
             {unscaled_would_be:.12} would mean the generator is still scaled as though the normal force were 1."
        );
        assert!(
            (soft / unscaled_would_be - 1.0).abs() > 5e-2,
            "Q1 {soft:.12} is indistinguishable from the unscaled convention's {unscaled_would_be:.12}"
        );
        eprintln!(
            "mu={mu} mu_torsion={mt}: Q1 {base:.9} (hard) -> {soft:.9} (torsion scaled by f.n={f_normal:.6}); \
             the unscaled convention gives {unscaled_would_be:.9}, +{:.1}% overstated",
            (unscaled_would_be / soft - 1.0) * 100.0
        );

        // 3. and a hard contact is untouched by the whole change
        for m in [0.0f64, 0.3, 0.5, 1.0, 1.5] {
            let hard: Vec<GraspContact3> =
                grasp(m, 0.0).iter().map(|c| GraspContact3 { mu_torsion: 0.0, ..*c }).collect();
            let ws = primitive_wrenches_spatial(&hard, 24);
            assert!(
                ws.iter().all(|w| Vector3::new(w[0], w[1], w[2]).norm() > 1e-12),
                "a hard contact must emit no torsional generators at all"
            );
        }
    }

    /// **The planar reduction.** Restricting the spatial computation to `(fx, fy, mz)` recovers the planar metric
    /// exactly, which is what shows the two are consistent rather than merely different.
    #[test]
    fn the_planar_subspace_recovers_the_planar_metric() {
        for mu in [0.2, 0.4, 0.7] {
            let planar = force_closure_q1(&three_coplanar_planar(mu), 20_000);
            let reduced = force_closure_q1_planar_subspace(&three_coplanar(mu), 256, 20_000);
            eprintln!("mu = {mu:.1}: planar {planar:.6}, spatial restricted to (fx, fy, mz) {reduced:.6}, difference {:.2e}", (planar - reduced).abs());
            assert!((planar - reduced).abs() < 5e-3, "the two must agree: {planar:.6} vs {reduced:.6}");
        }
    }

    /// The internal (squeeze) subspace is real and invisible to any net-wrench metric.
    #[test]
    fn the_internal_squeeze_subspace_produces_no_net_wrench() {
        let mut cs = three_coplanar(0.5);
        cs.push(GraspContact3::hard(Vector3::new(0.0, 0.0, 1.0), -Vector3::z(), 0.5));
        let split = grasp_split(&cs, 8);
        eprintln!("4 contacts x 8 facets = 32 generators: rank {}, internal dimension {}", split.rank, split.internal_dim);
        assert_eq!(split.rank, 6);
        assert!(split.internal_dim > 0, "a multi-finger grasp always has squeeze freedom");
        for c in 0..split.internal_basis.ncols().min(3) {
            let z: Vec<f64> = split.internal_basis.column(c).iter().copied().collect();
            let w = net_wrench(&cs, 8, &z).expect("width matches");
            eprintln!("   internal direction {c}: net wrench norm {:.2e}", w.norm());
            assert!(w.norm() < 1e-9, "an internal direction must produce no net wrench: {:.3e}", w.norm());
        }
    }

    /// **The smoothed metric is continuous where the gated one steps.**
    ///
    /// The rank gate makes the hard metric discontinuous at `mu = 0`: exactly zero below, immediately positive above.
    /// That is correct for a quality but useless as a descent signal. The smoothed form has no gate and varies through
    /// it, which is what a synthesiser needs.
    ///
    /// An earlier version of this test asserted that lifting a contact off the plane must *raise* the smoothed value.
    /// It does not (0.0435 -> 0.0407 at `dz = 0.05`), and there is no reason it should: with zero friction the forces
    /// stay in-plane and lifting a position changes only moment arms. The assertion was an assumption.
    #[test]
    fn the_smoothed_metric_is_continuous_where_the_gated_one_steps() {
        eprintln!("{:>8}  {:>12}  {:>12}", "mu", "gated Q1", "soft Q1");
        let mut soft_prev = f64::NEG_INFINITY;
        let mut saw_step = false;
        for mu in [0.0, 0.02, 0.1, 0.3, 0.6] {
            let cs = three_coplanar(mu);
            let hard = force_closure_q1_spatial(&cs, 24, 8192);
            let soft = force_closure_soft_spatial(&cs, 24, 2048, 40.0);
            eprintln!("{mu:>8.2}  {hard:>12.6}  {soft:>12.6}");
            if mu == 0.0 {
                assert_eq!(hard, 0.0, "the gate makes the hard metric exactly zero at mu = 0");
                assert!(soft.abs() > 1e-6, "the smoothed metric is still nonzero there, so it has a slope");
            } else {
                saw_step |= hard > 0.0;
            }
            assert!(soft > soft_prev, "the smoothed metric rises monotonically with friction: {soft:.6} after {soft_prev:.6}");
            soft_prev = soft;
        }
        assert!(saw_step, "and the hard metric does become positive, so the step is real");
        eprintln!("   The gated metric steps from 0 to positive at mu > 0; the smoothed one crosses it continuously.");
    }

    /// Degenerate input is handled rather than answered.
    #[test]
    fn degenerate_input_is_handled() {
        assert_eq!(force_closure_q1_spatial(&[], 16, 64), 0.0);
        assert_eq!(wrench_rank(&[], 16), 0);
        assert_eq!(grasp_split(&[], 16).rank, 0);
        assert!(net_wrench(&three_coplanar(0.4), 8, &[1.0]).is_none(), "a magnitude vector of the wrong width is refused");
        let one = [GraspContact3::hard(Vector3::x(), -Vector3::x(), 0.0)];
        assert_eq!(force_closure_q1_spatial(&one, 16, 64), 0.0, "a single frictionless contact spans one dimension");
    }
}
