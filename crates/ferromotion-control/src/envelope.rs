//! **Certified operating envelopes** — Q2: turning a *measured* reduced-vs-full discrepancy into an
//! *a-priori bounded* one, and inverting the E-ISS descent for the region where the guarantee holds.
//!
//! Theorem 6.4's embedding transfers a reduced-order guarantee to the full robot, degrading gracefully in a
//! bounded discrepancy `‖d_k‖ ≤ d̄`. Its one soft hypothesis is that `d̄` is *known*, and today it is
//! established empirically — which is exactly what the M2 certificate here did: it sampled the discrepancy and
//! took the worst value seen. That is an honest measurement and it is not a bound.
//!
//! Two pieces close the gap, and they are what this module supplies:
//!
//! * [`grid_max_bound`] — an enclosure of a function's maximum over a box, rather than its largest sampled
//!   value. Samples on a grid, then inflates by the Lipschitz constant times the cell's half-diagonal, so the
//!   space *between* the samples is accounted for.
//! * [`eiss_envelope`] and [`max_tolerable_discrepancy`] — the design rule. Given `δ²` and a discrepancy bound,
//!   the E-ISS ball is `√(σ/c₃)`; inverting that for the largest `d̄` whose ball still fits inside a stated
//!   region gives a **certified operating envelope**: the model errors for which the embedding holds.
//!
//! # What the enclosure does and does not guarantee
//!
//! Stated plainly, because the distinction is the whole point of Q2. A Lipschitz-grid enclosure is rigorous
//! **given a true upper bound on the Lipschitz constant**. [`grid_max_bound`] estimates that constant from the
//! same samples and inflates it by a stated factor, which is the standard method and carries the standard
//! caveat: it is sound if the function's steepest behaviour is visible at the grid's resolution, and a function
//! with a narrow spike between samples defeats it. So this is a certified bound *modulo* an assumption that is
//! named rather than hidden, and it is strictly stronger than the sampled maximum it replaces.
//!
//! The other half of Q2's risk is conservatism: a bound so loose that the envelope is empty. That is why
//! [`GridBound`] reports the inflation separately from the sampled maximum — the ratio between them is the
//! price of the guarantee, and it should be looked at rather than assumed small.

/// An enclosure of a function's maximum over a box.
#[derive(Clone, Debug)]
pub struct GridBound {
    /// The largest value actually seen on the grid.
    pub max_sampled: f64,
    /// The Lipschitz constant used, estimated from neighbouring samples and inflated.
    pub lipschitz: f64,
    /// Half the diagonal of one grid cell: the furthest any point can be from a sample.
    pub cell_reach: f64,
    /// `max_sampled + lipschitz · cell_reach`: the enclosure.
    pub bound: f64,
    pub samples: usize,
    /// How many sample points the function refused (outside its domain, or a failed step). A large count means
    /// the box reaches past where the model is defined and the enclosure covers less than it appears to.
    pub refused: usize,
}

impl GridBound {
    /// `bound / max_sampled`: the price of turning a measurement into an enclosure. Near one is tight; large
    /// means the grid is too coarse for the function's steepness and the envelope will be needlessly small.
    pub fn conservatism(&self) -> f64 {
        self.bound / self.max_sampled.abs().max(1e-300)
    }
}

/// **Enclose the maximum of `f` over the box `[lo, hi]`** by sampling a regular grid and inflating by a
/// Lipschitz estimate times the cell reach.
///
/// `per_axis` is the number of samples along each axis, so the cost is `per_axis^dim` — which is why this is
/// usable here and not on a humanoid. The dimension that matters is the *off-manifold* one, and a reduced-order
/// embedding is chosen precisely to make that small.
///
/// `inflation` multiplies the Lipschitz estimate; `1.0` trusts the grid's own slopes, and a larger value buys
/// margin against a steeper region between samples. `f` returns `None` where it is undefined, and those points
/// are counted rather than treated as zero.
pub fn grid_max_bound(f: &dyn Fn(&[f64]) -> Option<f64>, lo: &[f64], hi: &[f64], per_axis: usize, inflation: f64) -> Option<GridBound> {
    let dim = lo.len();
    if dim == 0 || hi.len() != dim || per_axis < 2 || inflation <= 0.0 {
        return None;
    }
    if lo.iter().zip(hi).any(|(a, b)| b < a) {
        return None;
    }
    let n = per_axis;
    let total = n.checked_pow(dim as u32)?;
    if total > 5_000_000 {
        return None; // the grid would be larger than this method is meant for
    }

    let point = |idx: usize| -> Vec<f64> {
        let mut rem = idx;
        (0..dim)
            .map(|d| {
                let i = rem % n;
                rem /= n;
                lo[d] + (hi[d] - lo[d]) * i as f64 / (n - 1) as f64
            })
            .collect()
    };

    let mut values: Vec<Option<f64>> = Vec::with_capacity(total);
    let mut refused = 0;
    for idx in 0..total {
        let v = f(&point(idx));
        if v.is_none() {
            refused += 1;
        }
        values.push(v);
    }
    let max_sampled = values.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_sampled.is_finite() {
        return None; // nothing evaluated
    }

    // Lipschitz estimate: the steepest slope between axis-adjacent samples.
    let mut lip = 0.0f64;
    let step: Vec<f64> = (0..dim).map(|d| (hi[d] - lo[d]) / (n - 1) as f64).collect();
    let mut stride = 1;
    for &h in &step {
        if h > 0.0 {
            for idx in 0..total {
                let i = (idx / stride) % n;
                if i + 1 < n
                    && let (Some(a), Some(b)) = (values[idx], values[idx + stride])
                {
                    lip = lip.max((b - a).abs() / h);
                }
            }
        }
        stride *= n;
    }
    let lipschitz = lip * inflation;
    // the furthest any point in the box can be from a grid sample
    let cell_reach = 0.5 * step.iter().map(|s| s * s).sum::<f64>().sqrt();
    Some(GridBound { max_sampled, lipschitz, cell_reach, bound: max_sampled + lipschitz * cell_reach, samples: total, refused })
}

/// The E-ISS operating envelope for an affine restricted map of multiplier `delta_sq`, a discrepancy bound
/// `d_bar`, a Young split `eta`, and a region radius the guarantee must fit inside.
#[derive(Clone, Copy, Debug)]
pub struct Envelope {
    /// `c₃ = 1 − δ⁴(1+η)`: the per-step contraction of `V` left after the Young split.
    pub c3: f64,
    /// `σ = (1 + 1/η)·d̄²`: the disturbance term.
    pub sigma: f64,
    /// `√(σ/c₃)`: the ball the certificate claims around the ideal gait.
    pub ball: f64,
    /// `region − ball`: positive when the guarantee fits inside the region it was stated for.
    pub margin: f64,
    /// Whether the envelope holds: `c₃ > 0` and the ball fits.
    pub valid: bool,
}

/// Build the envelope. `None` if the inputs are out of range.
pub fn eiss_envelope(delta_sq: f64, d_bar: f64, eta: f64, region: f64) -> Option<Envelope> {
    if !(0.0..1.0).contains(&delta_sq) || d_bar < 0.0 || eta <= 0.0 || region <= 0.0 {
        return None;
    }
    let c3 = 1.0 - delta_sq * delta_sq * (1.0 + eta);
    if c3 <= 0.0 {
        // The split has eaten the whole contraction: a valid statement, just not a useful one. Reported rather
        // than hidden, since it is the first thing to check when an envelope comes out empty.
        return Some(Envelope { c3, sigma: f64::INFINITY, ball: f64::INFINITY, margin: f64::NEG_INFINITY, valid: false });
    }
    let sigma = (1.0 + 1.0 / eta) * d_bar * d_bar;
    let ball = (sigma / c3).sqrt();
    Some(Envelope { c3, sigma, ball, margin: region - ball, valid: ball <= region })
}

/// **The design rule**: the largest discrepancy whose E-ISS ball still fits inside `region`.
///
/// Inverting `√(σ/c₃) ≤ region` gives `d̄ ≤ region·√(c₃/(1 + 1/η))`. This is the number Q2 asks for — it turns
/// "how accurate must the reduced model be?" from a judgement into an inequality, and it is what makes an
/// envelope *certified* rather than observed.
pub fn max_tolerable_discrepancy(delta_sq: f64, eta: f64, region: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&delta_sq) || eta <= 0.0 || region <= 0.0 {
        return None;
    }
    let c3 = 1.0 - delta_sq * delta_sq * (1.0 + eta);
    (c3 > 0.0).then(|| region * (c3 / (1.0 + 1.0 / eta)).sqrt())
}

/// **The optimal Young split, in closed form** — and the fact that it makes the bound *tight*.
///
/// Maximising `c₃/(1 + 1/η) = η(1 − δ⁴(1+η))/(1+η)` over `η` gives
///
/// ```text
/// η* = (1 − δ²)/δ²,     c₃ = 1 − δ²,     1 + 1/η* = 1/(1 − δ²),     d̄_max = region·(1 − δ²)
/// ```
///
/// The consequence is worth more than the convenience. At `η*` the certified ball is
///
/// ```text
/// √(σ/c₃) = √( d̄²/(1−δ²) / (1−δ²) ) = d̄/(1 − δ²)
/// ```
///
/// and `d̄/(1−δ²)` is **exactly** the steady deviation of the disturbed affine map, whose fixed point moves from
/// `ζ*` to `ζ* + w/(1−δ²)`. So the E-ISS certificate, with the split chosen properly, is not conservative at
/// all — it recovers the exact answer. Q2's named risk is that a certified bound is so loose the envelope is
/// useless; for an affine restricted map that risk is zero, and all the conservatism that remains lives in the
/// discrepancy enclosure rather than in the certificate.
///
/// Returns `(η*, c₃, d̄_max)`.
pub fn optimal_split(delta_sq: f64, region: f64) -> Option<(f64, f64, f64)> {
    if !(0.0..1.0).contains(&delta_sq) || delta_sq <= 0.0 || region <= 0.0 {
        return None;
    }
    let c3 = 1.0 - delta_sq;
    Some(((1.0 - delta_sq) / delta_sq, c3, region * c3))
}

/// **The funnel design rule**: the largest per-step disturbance a contact task can absorb without falling out
/// of its certified basin.
///
/// This is P1's sub-problem (b) made concrete. A contact-rich loop has no global contraction metric — the
/// continuous dynamics are non-smooth and the rate is negative on part of the state space — so the horizon-free
/// regret bound cannot be hung on a global `λ`. What it *can* be hung on is a **funnel**: a region the loop
/// provably returns to. On a section where the return map is affine with multiplier `δ²`, a per-step disturbance
/// `w` moves the fixed point from `ζ*` to `ζ* + w/(1 − δ²)`, so the loop stays inside the basin exactly while
///
/// ```text
/// w  <  (ζ* − ζ_stall)·(1 − δ²)
/// ```
///
/// Below that, regret is bounded and horizon-free; above it the robot leaves the funnel and falls, and there is
/// nothing horizon-free left to say. The threshold plays the role the `H∞` margin plays for a smooth loop, and
/// like it, it is computable **before any policy is trained** — from the gait and the geometry alone.
///
/// `None` unless the map contracts and the gait sits strictly inside its own basin.
pub fn max_disturbance_in_funnel(delta_sq: f64, zeta_star: f64, zeta_stall: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&delta_sq) || zeta_star <= zeta_stall || zeta_stall < 0.0 {
        return None;
    }
    Some((zeta_star - zeta_stall) * (1.0 - delta_sq))
}

/// The Young split that tolerates the **largest** discrepancy, found by a sweep. Kept alongside
/// [`optimal_split`] because a numerical sweep agreeing with a closed form is how the closed form was checked.
///
/// There is a real trade here and it is worth seeing rather than guessing at: a small `η` keeps `c₃` large but
/// blows up `σ` through the `1/η`, while a large `η` tames `σ` and eats the contraction. The optimum is
/// interior, and it moves with `δ²` — a gait that barely contracts has almost no room to give away.
pub fn best_eta(delta_sq: f64, region: f64, samples: usize) -> Option<(f64, f64)> {
    let n = samples.max(16);
    let mut best: Option<(f64, f64)> = None;
    for k in 1..n {
        // geometric sweep, since the useful range spans orders of magnitude
        let eta = 1e-3 * (1e6f64).powf(k as f64 / n as f64);
        if let Some(d) = max_tolerable_discrepancy(delta_sq, eta, region)
            && best.as_ref().is_none_or(|(_, b)| d > *b)
        {
            best = Some((eta, d));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The enclosure must **contain** the true maximum, which the sampled maximum need not. Tested on a function
    /// whose peak deliberately falls between grid points.
    #[test]
    fn the_enclosure_contains_a_maximum_the_grid_misses() {
        // peak at x = 0.5 exactly; an even sample count never lands on it
        let f = |x: &[f64]| Some(-((x[0] - 0.5) * (x[0] - 0.5)) + 1.0);
        let g = grid_max_bound(&f, &[0.0], &[1.0], 8, 1.0).unwrap();
        eprintln!("true max 1.0: sampled {:.6}, Lipschitz {:.4}, cell reach {:.4}, enclosure {:.6}", g.max_sampled, g.lipschitz, g.cell_reach, g.bound);
        assert!(g.max_sampled < 1.0, "the grid should miss the peak, got {}", g.max_sampled);
        assert!(g.bound >= 1.0, "the enclosure must contain the true maximum: {} < 1.0", g.bound);
        assert!(g.conservatism() > 1.0 && g.conservatism() < 1.2, "and not be absurdly loose: {}", g.conservatism());
    }

    /// Refining the grid tightens the enclosure toward the truth, which is what makes it a usable bound rather
    /// than a formality.
    #[test]
    fn refining_the_grid_tightens_the_enclosure() {
        let f = |x: &[f64]| Some((3.0 * x[0]).sin() + 0.5 * (7.0 * x[1]).cos());
        let mut prev = f64::INFINITY;
        for n in [5usize, 9, 17, 33] {
            let g = grid_max_bound(&f, &[0.0, 0.0], &[2.0, 2.0], n, 1.0).unwrap();
            eprintln!("   {n}x{n}: sampled {:.6}, enclosure {:.6}, conservatism {:.3}", g.max_sampled, g.bound, g.conservatism());
            assert!(g.bound < prev, "a finer grid must not loosen the enclosure");
            assert_eq!(g.samples, n * n);
            prev = g.bound;
        }
        // the true maximum is 1.5 (both terms at their peak somewhere in the box)
        assert!((1.4..1.9).contains(&prev), "the refined enclosure should be near the truth, got {prev}");
    }

    /// Points the function refuses are counted, not silently treated as zero. A box reaching outside a model's
    /// domain otherwise produces an enclosure that looks valid and covers nothing.
    #[test]
    fn refused_points_are_reported_rather_than_absorbed() {
        let f = |x: &[f64]| (x[0] < 0.6).then_some(x[0]);
        let g = grid_max_bound(&f, &[0.0], &[1.0], 11, 1.0).unwrap();
        eprintln!("11 samples over [0,1] with the top 40% refused: {} refused, sampled max {:.3}", g.refused, g.max_sampled);
        assert_eq!(g.refused, 5, "the five points at 0.6 and above must be counted");
        assert!(g.max_sampled < 0.6);
        // and a function defined nowhere yields no bound at all rather than a meaningless one
        assert!(grid_max_bound(&|_: &[f64]| None, &[0.0], &[1.0], 5, 1.0).is_none());
    }

    /// **The design rule is exact at its own boundary**: at the largest tolerable discrepancy the ball touches
    /// the region, and a hair more overflows it. That is what makes it a rule rather than a heuristic.
    #[test]
    fn the_design_rule_is_exact_at_the_boundary() {
        let (delta_sq, eta, region) = (0.83f64, 0.15f64, 0.5f64);
        let d_max = max_tolerable_discrepancy(delta_sq, eta, region).unwrap();
        let at = eiss_envelope(delta_sq, d_max, eta, region).unwrap();
        eprintln!("delta^2 {delta_sq}, eta {eta}, region {region}: largest tolerable d_bar = {d_max:.6}, ball {:.6}", at.ball);
        assert!((at.ball - region).abs() < 1e-9, "at the boundary the ball must equal the region: {} vs {region}", at.ball);
        assert!(at.valid && at.margin.abs() < 1e-9);

        let over = eiss_envelope(delta_sq, d_max * 1.001, eta, region).unwrap();
        assert!(!over.valid && over.margin < 0.0, "a hair over must not be certified");
        let under = eiss_envelope(delta_sq, d_max * 0.5, eta, region).unwrap();
        assert!(under.valid && under.margin > 0.0, "and comfortably under must be");
        // the tolerance scales with the region, which is the sanity check on the algebra
        let bigger = max_tolerable_discrepancy(delta_sq, eta, 2.0 * region).unwrap();
        assert!((bigger / d_max - 2.0).abs() < 1e-12, "linear in the region");
    }

    /// **The funnel rule is exact at its own boundary**, and it is the contact-task analogue of a stability
    /// margin: a disturbance just under it keeps the gait inside the basin forever, and just over it walks the
    /// fixed point out through the stall threshold.
    #[test]
    fn the_funnel_rule_is_exact_at_the_stall_boundary() {
        let (delta_sq, zstar, stall) = (0.82f64, 2.5f64, 0.6f64);
        let w_max = max_disturbance_in_funnel(delta_sq, zstar, stall).unwrap();
        eprintln!("delta^2 {delta_sq}, zeta* {zstar}, stall {stall}: largest tolerable disturbance {w_max:.6}");
        // the disturbed fixed point sits exactly at the stall threshold
        let moved = zstar - w_max / (1.0 - delta_sq);
        assert!((moved - stall).abs() < 1e-9, "at the boundary the fixed point lands on the stall threshold: {moved} vs {stall}");
        // just inside, the gait survives; just outside, it does not
        assert!(zstar - 0.99 * w_max / (1.0 - delta_sq) > stall);
        assert!(zstar - 1.01 * w_max / (1.0 - delta_sq) < stall);
        // a faster gait or a stronger contraction buys more room, both linearly
        assert!(max_disturbance_in_funnel(delta_sq, 2.0 * zstar - stall, stall).unwrap() > 1.9 * w_max);
        assert!(max_disturbance_in_funnel(0.5, zstar, stall).unwrap() > w_max);
        // and a gait already at its stall threshold has no room at all
        assert!(max_disturbance_in_funnel(delta_sq, stall, stall).is_none());
    }

    /// **The closed form against the sweep**, and then the claim that matters: at the optimal split the
    /// certified ball equals the *exact* steady deviation of the disturbed map, so the certificate costs nothing.
    #[test]
    fn the_optimal_split_makes_the_eiss_bound_exact() {
        let region = 1.0;
        for delta_sq in [0.3f64, 0.5, 0.7, 0.83, 0.95, 0.99] {
            let (eta_c, c3_c, d_c) = optimal_split(delta_sq, region).unwrap();
            let (eta_s, d_s) = best_eta(delta_sq, region, 4000).unwrap();
            assert!((d_c - d_s).abs() / d_c < 1e-3, "closed form and sweep must agree at delta^2 {delta_sq}: {d_c} vs {d_s}");
            assert!((eta_c - eta_s).abs() / eta_c < 0.02, "and on the split itself: {eta_c} vs {eta_s}");
            assert!((c3_c - (1.0 - delta_sq)).abs() < 1e-15, "c3 at the optimum is exactly 1 - delta^2");
            assert!((d_c - region * (1.0 - delta_sq)).abs() < 1e-15, "and the budget is region(1 - delta^2)");

            // The tightness claim: for a disturbance w the certified ball must equal w/(1 - delta^2), which is
            // where the disturbed affine map's fixed point actually sits.
            let w = 0.01;
            let ball = eiss_envelope(delta_sq, w, eta_c, 1e6).unwrap().ball;
            let exact = w / (1.0 - delta_sq);
            eprintln!("delta^2 {delta_sq}: eta* {eta_c:.5}, certified ball {ball:.8}, exact steady deviation {exact:.8}");
            assert!((ball - exact).abs() / exact < 1e-12, "the bound must be exact at the optimal split: {ball} vs {exact}");

            // ... and any other split is strictly worse, which is why the closed form is worth having
            let worse = eiss_envelope(delta_sq, w, eta_c * 3.0, 1e6).unwrap().ball;
            assert!(worse > ball, "a non-optimal split must be conservative: {worse} vs {ball}");
        }
    }

    /// A gait that barely contracts has almost no discrepancy budget, and one that contracts hard has a lot.
    /// This is the quantitative form of "how good does the reduced model have to be?", and the answer depends on
    /// the gait rather than on taste.
    #[test]
    fn the_discrepancy_budget_grows_with_the_gaits_contraction() {
        let region = 1.0;
        let mut last = 0.0;
        for delta_sq in [0.99f64, 0.9, 0.7, 0.4] {
            let (eta, d) = best_eta(delta_sq, region, 400).unwrap();
            eprintln!("   delta^2 {delta_sq}: best eta {eta:.4}, tolerable d_bar {d:.6}");
            assert!(d > last, "a stronger contraction must tolerate more error");
            last = d;
        }
        // and the optimal split is interior, not at either extreme
        let (eta, _) = best_eta(0.8, region, 400).unwrap();
        assert!(eta > 1e-3 && eta < 1e3, "the optimum should be interior, got {eta}");
        // a non-contracting map has no budget at all
        assert!(max_tolerable_discrepancy(0.999_999, 100.0, region).is_none());
        assert!(eiss_envelope(1.0, 0.1, 0.15, 1.0).is_none(), "delta^2 = 1 is not a contraction");
    }

    /// `c₃ ≤ 0` is a valid statement with no useful content, and it has to be reported that way rather than as a
    /// missing answer — it is the first thing to check when an envelope comes out empty.
    #[test]
    fn a_split_that_eats_the_contraction_is_reported_not_hidden() {
        // delta^2 = 0.95 with eta = 1.0 gives c3 = 1 - 0.9025*2 < 0
        let e = eiss_envelope(0.95, 0.01, 1.0, 1.0).unwrap();
        eprintln!("delta^2 0.95 with eta 1.0: c3 = {:.4}, valid = {}", e.c3, e.valid);
        assert!(e.c3 < 0.0 && !e.valid);
        assert!(e.ball.is_infinite(), "and the ball is infinite rather than a number that looks meaningful");
        // a smaller split rescues the same gait
        let (eta, d) = best_eta(0.95, 1.0, 400).unwrap();
        assert!(d > 0.0, "a better split gives this gait a real budget: eta {eta}, d_bar {d}");
        assert!(eiss_envelope(0.95, d * 0.5, eta, 1.0).unwrap().valid);
    }
}
