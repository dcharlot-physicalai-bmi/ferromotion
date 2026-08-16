//! **Fatigue** — what a duty cycle costs the hardware, and why a reward function cannot see it.
//!
//! Every other module in the actuator layer answers a question about *now*: what torque, what temperature,
//! what voltage. This one answers a question about the thousandth hour. A joint driven within every
//! instantaneous limit — under its torque rating, under its thermal rating, inside its voltage envelope — still
//! fails, from the accumulated damage of load *reversals*. Nothing in the stack could express that.
//!
//! # Why this belongs next to a policy, not only next to a stress analysis
//!
//! Fatigue damage goes as stress amplitude to a power between about 3 and 10. With the exponent at 5, a 20%
//! larger torque amplitude does **2.49 times** the damage, and a 50% larger one does 7.6 times. A learned
//! policy's objective contains no term for any of this, so two policies with **identical reward** can differ by
//! orders of magnitude in what they do to the actuator: the chattering one and the smooth one reach the same
//! goal, collect the same return, and one of them destroys a gearbox. That is measured in the tests here, not
//! argued: a policy with the same mean torque and a superimposed high-frequency dither accumulates far more
//! damage per episode, and the ratio follows the S-N exponent.
//!
//! How much more depends on how hard it chatters, and the measured range is wide. With `b = 5`, a Goodman mean
//! correction, and a trace holding the same mean torque and the same slow excursion, adding a high-frequency
//! dither costs:
//!
//! ```text
//!   dither amplitude    damage vs smooth
//!   25% of command            3.1x
//!   62% of command           14.5x  to   41.3x   (100 Hz to 800 Hz)
//!  100% of command           67.1x  to  346.2x
//! ```
//!
//! So one to two orders of magnitude for a policy that chatters at the amplitude it commands. The frequency
//! dependence is **sub-linear** in cycle count — doubling the dither frequency doubles the number of cycles but
//! raises damage only 1.3 to 1.6 times — because at low dither frequency the slow excursion's drift inflates
//! each dither cycle's *range*, so there are fewer but individually worse cycles.
//!
//! The action-smoothness penalty that appears in every hand-tuned reward function is a proxy for this, applied
//! without a unit. [`damage`] gives the quantity the proxy was reaching for, in cycles-of-life consumed, which
//! is comparable across policies and across hardware.
//!
//! # The three pieces
//!
//! * **[`rainflow`]** — cycle extraction from a variable-amplitude history, per ASTM E1049. A torque trace is
//!   not a sequence of cycles until something decides which peak pairs with which valley, and the answer is
//!   not "adjacent ones": a small oscillation riding on a large one belongs to the large cycle, and counting it
//!   as two independent small cycles understates the damage badly. This is the piece that is easy to get
//!   subtly wrong and hard to notice, so it is checked against an exact conservation law.
//! * **[`SnCurve`]** — the material's cycles-to-failure at a given amplitude, `N = (A/S)^b` (Basquin), with an
//!   optional endurance limit below which life is unbounded.
//! * **[`MeanCorrection`]** — rainflow returns a mean as well as an amplitude, and mean stress matters: a cycle
//!   about a large tensile mean is far more damaging than the same amplitude about zero. Ignoring the mean is
//!   the most common way a fatigue estimate comes out optimistic.
//!
//! # The conservation law the counting is checked against
//!
//! Rainflow must account for every reversal in the history exactly once. A full cycle of range `R` consumes an
//! up-move of `R` and a down-move of `R`; a half cycle consumes one move of `R`. So for any history:
//!
//! ```text
//!   Σ 2 · count · range  =  total variation of the reversal sequence
//! ```
//!
//! That is exact, it is independent of the algorithm's bookkeeping, and it fails immediately for the common
//! implementation errors: dropping the residue, double-counting a closed cycle, or losing the first point.
//! It is asserted on randomised histories as well as hand-built ones.

/// One extracted load cycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cycle {
    /// Peak-to-valley range (twice the amplitude).
    pub range: f64,
    /// Mean level of the cycle.
    pub mean: f64,
    /// `1.0` for a closed cycle, `0.5` for a residual half cycle.
    pub count: f64,
}

impl Cycle {
    /// Half the range — the amplitude an S-N curve is indexed by.
    pub fn amplitude(&self) -> f64 {
        0.5 * self.range
    }
}

/// Reduce a series to its **turning points**: the first point, every local extremum, and the last point.
///
/// Everything between reversals is irrelevant to fatigue — a load that rises through a value and keeps rising
/// has not completed anything — so this both shrinks the problem and is a precondition for the counting rule
/// to mean what it says. Consecutive equal values are collapsed, because a plateau is not a reversal and
/// treating it as two would inject zero-range cycles that pollute the count.
pub fn reversals(series: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for &v in series {
        if out.last().is_some_and(|l| *l == v) {
            continue; // collapse plateaus
        }
        out.push(v);
    }
    if out.len() < 3 {
        return out;
    }
    let mut turning = Vec::with_capacity(out.len());
    turning.push(out[0]);
    for i in 1..out.len() - 1 {
        let (a, b, c) = (out[i - 1], out[i], out[i + 1]);
        // A turning point is where the direction changes sign. Monotone interior points carry no information.
        if (b - a) * (c - b) < 0.0 {
            turning.push(b);
        }
    }
    turning.push(out[out.len() - 1]);
    turning
}

/// **Rainflow cycle counting**, ASTM E1049 four-point method.
///
/// Returns closed cycles (`count = 1.0`) and residual half cycles (`count = 0.5`). The residue is **not
/// discarded**: for a short history it can be most of the damage, and dropping it is a silent underestimate
/// that grows worse the shorter the record.
///
/// The rule: among four consecutive turning points `s1 s2 s3 s4`, if the **inner** range `Y = |s2 − s3|` is no
/// larger than both neighbours `X = |s1 − s2|` and `Z = |s3 − s4|`, then `s2 s3` is a closed cycle. It is
/// extracted and those two points are removed, leaving `s1` adjacent to `s4`.
///
/// **Why four points and not three.** The three-point variant extracts the *outer* pair and keeps the newest
/// point, which severs the chain: the part of the current excursion not consumed by the extracted cycle is
/// dropped along with the removed points. My first version did exactly that, and the conservation law caught
/// it on a five-point history — counted variation 259.39 against an actual 279.13, a deficit of 19.74 which is
/// precisely the unconsumed remainder of the last up-move. The four-point form conserves *by construction*:
/// removing the inner pair takes the path `s1→s2→s3→s4` to `s1→s4`, and since `Y` is the smallest,
/// `|s1 − s4| = X + Z − Y`, so the variation removed is exactly `2Y` — which is what a full cycle of range `Y`
/// accounts for.
pub fn rainflow(series: &[f64]) -> Vec<Cycle> {
    let rev = reversals(series);
    let mut cycles: Vec<Cycle> = Vec::new();
    if rev.len() < 2 {
        return cycles;
    }
    let mut stack: Vec<f64> = Vec::with_capacity(rev.len());
    for &p in &rev {
        stack.push(p);
        while stack.len() >= 4 {
            let n = stack.len();
            let (s1, s2, s3, s4) = (stack[n - 4], stack[n - 3], stack[n - 2], stack[n - 1]);
            let x = (s2 - s1).abs();
            let y = (s3 - s2).abs();
            let z = (s4 - s3).abs();
            if y > x || y > z {
                break;
            }
            cycles.push(Cycle { range: y, mean: 0.5 * (s2 + s3), count: 1.0 });
            // Remove the inner pair, leaving s1 adjacent to s4. Order matters: higher index first.
            stack.remove(n - 2);
            stack.remove(n - 3);
        }
    }
    // Whatever is left never closed: each remaining range is a half cycle.
    for w in stack.windows(2) {
        cycles.push(Cycle { range: (w[1] - w[0]).abs(), mean: 0.5 * (w[0] + w[1]), count: 0.5 });
    }
    cycles
}

/// Total variation of a series' reversal sequence: `Σ |Δ|`. The quantity rainflow must conserve.
pub fn total_variation(series: &[f64]) -> f64 {
    reversals(series).windows(2).map(|w| (w[1] - w[0]).abs()).sum()
}

/// A **Basquin** S-N curve: `N = (A/S)^b`, optionally with an endurance limit.
///
/// `A` is the amplitude at which failure occurs in one cycle and `b` the inverse slope on a log-log plot.
/// Steels typically have `b` near 3 for welded joints and up to 10 or so for polished specimens; aluminium
/// alloys have **no endurance limit at all**, which is why that field is an `Option` rather than a number with
/// a default.
#[derive(Clone, Copy, Debug)]
pub struct SnCurve {
    /// Amplitude giving failure in one cycle.
    pub coefficient: f64,
    /// Inverse log-log slope, `b`. Larger means less sensitive to amplitude.
    pub exponent: f64,
    /// Amplitude below which life is unbounded. `None` for materials that have none.
    pub endurance_limit: Option<f64>,
}

impl SnCurve {
    /// A curve from `A` and `b`, with no endurance limit. Returns `None` for non-physical parameters.
    pub fn basquin(coefficient: f64, exponent: f64) -> Option<SnCurve> {
        // Finiteness first, so the ordering comparisons are on real numbers and NaN cannot slip past them.
        if !coefficient.is_finite() || !exponent.is_finite() || coefficient <= 0.0 || exponent <= 0.0 {
            return None;
        }
        Some(SnCurve { coefficient, exponent, endurance_limit: None })
    }

    /// Add an endurance limit.
    pub fn with_endurance_limit(mut self, limit: f64) -> SnCurve {
        self.endurance_limit = Some(limit);
        self
    }

    /// Cycles to failure at amplitude `s`. `None` means unbounded life, either below the endurance limit or at
    /// zero amplitude.
    ///
    /// `None` rather than `f64::INFINITY` on purpose: infinite life is a categorically different answer from a
    /// large number, and a caller that sums `1/N` over cycles must not have an infinity silently become a zero
    /// it never examined.
    pub fn cycles_to_failure(&self, s: f64) -> Option<f64> {
        let s = s.abs();
        if s <= 0.0 {
            return None;
        }
        if self.endurance_limit.is_some_and(|limit| s <= limit) {
            return None;
        }
        Some((self.coefficient / s).powf(self.exponent))
    }

    /// The amplitude whose life is exactly `n` cycles: the inverse of
    /// [`cycles_to_failure`](SnCurve::cycles_to_failure).
    pub fn amplitude_at(&self, n: f64) -> f64 {
        self.coefficient * n.powf(-1.0 / self.exponent)
    }
}

/// How to fold a cycle's mean level into an equivalent fully-reversed amplitude.
///
/// Rainflow gives an amplitude *and* a mean, and the mean matters: a cycle about a large tensile mean is far
/// more damaging than the same amplitude about zero, because the mean holds any crack open. All three of these
/// map `(amplitude, mean)` to the fully-reversed amplitude of equal damage, which is the only thing an S-N
/// curve can be indexed by.
#[derive(Clone, Copy, Debug)]
pub enum MeanCorrection {
    /// Ignore the mean. Correct only for fully-reversed loading, and **optimistic** otherwise — the most
    /// common reason a fatigue estimate comes out longer than the hardware lasts.
    None,
    /// Goodman: `S_eq = S_a / (1 − S_m/S_u)`. Linear to the ultimate strength; the usual conservative choice.
    Goodman {
        /// Ultimate tensile strength.
        ultimate: f64,
    },
    /// Soderberg: as Goodman but to the yield strength. More conservative still.
    Soderberg {
        /// Yield strength.
        yield_strength: f64,
    },
    /// Gerber: `S_eq = S_a / (1 − (S_m/S_u)²)`. Parabolic, and closer to test data for ductile metals, at the
    /// cost of being non-conservative where the data are thin.
    Gerber {
        /// Ultimate tensile strength.
        ultimate: f64,
    },
}

impl MeanCorrection {
    /// The fully-reversed amplitude of equal damage.
    ///
    /// **Compressive means are not credited.** A negative mean genuinely extends life, but the correction
    /// formulas extrapolate that benefit without bound, and a cycle about a large compressive mean would come
    /// out with an equivalent amplitude near zero and therefore infinite life. Clamping the mean at zero
    /// forgoes a real benefit rather than inventing one, which is the right direction for a life estimate.
    ///
    /// A mean at or above the reference strength returns `None`: the material is already at its static limit
    /// and a cycle superimposed on it has no fatigue life to speak of.
    pub fn equivalent_amplitude(&self, amplitude: f64, mean: f64) -> Option<f64> {
        let s_a = amplitude.abs();
        let s_m = mean.max(0.0);
        match *self {
            MeanCorrection::None => Some(s_a),
            // Each arm checks its own reference strength. A non-positive one is a malformed material rather
            // than a severe one, so it gets `None` (no life computable) instead of a division.
            MeanCorrection::Goodman { ultimate } => {
                let d = 1.0 - s_m / ultimate;
                if ultimate <= 0.0 || d <= 0.0 {
                    None
                } else {
                    Some(s_a / d)
                }
            }
            MeanCorrection::Soderberg { yield_strength } => {
                let d = 1.0 - s_m / yield_strength;
                if yield_strength <= 0.0 || d <= 0.0 {
                    None
                } else {
                    Some(s_a / d)
                }
            }
            MeanCorrection::Gerber { ultimate } => {
                let r = s_m / ultimate;
                let d = 1.0 - r * r;
                if ultimate <= 0.0 || d <= 0.0 {
                    None
                } else {
                    Some(s_a / d)
                }
            }
        }
    }
}

/// **Miner's rule**: accumulated damage `D = Σ n_i / N_i`, with failure predicted at `D = 1`.
///
/// Returns the damage fraction. `1.0` is the predicted failure point, and values above it mean the history
/// exceeds the predicted life.
///
/// Miner's rule is **linear and sequence-independent**, and both of those are known to be wrong: a large cycle
/// applied first changes how subsequent small ones propagate, and real lives scatter by a factor of two or
/// more around the prediction. It is used anyway because it is the only accumulation rule simple enough to
/// apply to an arbitrary history, and it is a *design* tool. Treat a computed `D` as a comparison between duty
/// cycles rather than a prediction of when a specific unit dies. That is exactly the use this module is for:
/// ranking policies against each other on the same hardware.
pub fn damage(cycles: &[Cycle], sn: &SnCurve, correction: MeanCorrection) -> f64 {
    let mut d = 0.0;
    for c in cycles {
        let Some(s_eq) = correction.equivalent_amplitude(c.amplitude(), c.mean) else {
            // The mean has reached the static limit: treat the cycle as immediately failing rather than
            // skipping it, which would report a *lower* damage for a more severe load.
            return f64::INFINITY;
        };
        if let Some(n) = sn.cycles_to_failure(s_eq) {
            d += c.count / n;
        }
        // A cycle below the endurance limit contributes nothing, which is what the limit means.
    }
    d
}

/// Damage accumulated by one load history: rainflow, then Miner. The convenience path.
pub fn damage_from_history(series: &[f64], sn: &SnCurve, correction: MeanCorrection) -> f64 {
    damage(&rainflow(series), sn, correction)
}

/// Number of repetitions of `series` predicted to reach `D = 1`.
///
/// `None` if the history does no damage at all — every cycle below the endurance limit — which is the design
/// target rather than an error.
pub fn repetitions_to_failure(series: &[f64], sn: &SnCurve, correction: MeanCorrection) -> Option<f64> {
    let d = damage_from_history(series, sn, correction);
    if d <= 0.0 {
        None
    } else {
        Some(1.0 / d)
    }
}

/// The **equivalent constant amplitude** that would do the same damage in the same number of cycles.
///
/// A single number summarising a variable history, and the honest way to compare two duty cycles of equal
/// length. Derived by inverting Miner's rule against the S-N curve, so it depends on the exponent: the same
/// history has a different equivalent amplitude on a different material, which is the whole point and the
/// reason a bare RMS torque is not a substitute.
pub fn equivalent_amplitude(series: &[f64], sn: &SnCurve, correction: MeanCorrection) -> Option<f64> {
    let cycles = rainflow(series);
    let n: f64 = cycles.iter().map(|c| c.count).sum();
    if n <= 0.0 {
        return None;
    }
    let d = damage(&cycles, sn, correction);
    if d <= 0.0 || !d.is_finite() {
        return None;
    }
    // D = n / N_eq  and  N_eq = (A/S_eq)^b   =>   S_eq = A (D/n)^(1/b)
    Some(sn.coefficient * (d / n).powf(1.0 / sn.exponent))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steel() -> SnCurve {
        // A = 1000 (arbitrary stress units), b = 5: mid-range for a machined steel part.
        SnCurve::basquin(1000.0, 5.0).expect("valid curve")
    }

    /// The conservation law: every reversal accounted for exactly once.
    fn assert_conserves(series: &[f64]) {
        let cycles = rainflow(series);
        let counted: f64 = cycles.iter().map(|c| 2.0 * c.count * c.range).sum();
        let tv = total_variation(series);
        assert!(
            (counted - tv).abs() < 1e-9 * tv.max(1.0),
            "rainflow must conserve total variation: counted {counted} vs {tv} for {series:?}"
        );
    }

    #[test]
    fn reversals_keeps_turning_points_and_drops_the_rest() {
        // Monotone interior points carry no fatigue information; plateaus are not reversals.
        assert_eq!(reversals(&[0.0, 1.0, 2.0, 3.0]), vec![0.0, 3.0]);
        assert_eq!(reversals(&[0.0, 2.0, 1.0, 3.0]), vec![0.0, 2.0, 1.0, 3.0]);
        assert_eq!(reversals(&[0.0, 1.0, 1.0, 1.0, 2.0]), vec![0.0, 2.0], "a plateau is not a reversal");
        assert_eq!(reversals(&[5.0]), vec![5.0]);
        assert_eq!(reversals(&[]), Vec::<f64>::new());
        // A saw-tooth is all turning points.
        assert_eq!(reversals(&[0.0, 1.0, 0.0, 1.0, 0.0]).len(), 5);
    }

    #[test]
    fn a_constant_amplitude_history_counts_exactly_its_cycles() {
        // The case with an unambiguous right answer, and the one every implementation must get exactly.
        let n = 20;
        let mut series = vec![0.0];
        for _ in 0..n {
            series.push(10.0);
            series.push(-10.0);
        }
        let cycles = rainflow(&series);
        let total: f64 = cycles.iter().map(|c| c.count).sum();
        // n full excursions from a zero start: 20 cycles' worth of counts, to within the half-cycle
        // bookkeeping at the two ends.
        assert!(
            (total - n as f64).abs() <= 1.0,
            "expected about {n} cycles, counted {total}: {cycles:?}"
        );
        // Every counted range must be the excursion's range, 20.
        for c in &cycles {
            assert!(
                (c.range - 20.0).abs() < 1e-9 || (c.range - 10.0).abs() < 1e-9,
                "unexpected range {} in a constant-amplitude history",
                c.range
            );
        }
        assert_conserves(&series);
    }

    #[test]
    fn a_small_oscillation_on_a_large_one_belongs_to_the_large_cycle() {
        // The property that makes rainflow more than peak counting, and the one whose absence understates
        // damage badly. A big excursion with a wiggle partway up must yield ONE large cycle plus one small
        // one, not several medium ones.
        let series = [0.0, 10.0, 6.0, 8.0, 4.0, 12.0, -8.0, 0.0];
        let cycles = rainflow(&series);
        assert_conserves(&series);

        let max_range = cycles.iter().map(|c| c.range).fold(0.0f64, f64::max);
        let history_range = 12.0 - (-8.0);
        assert!(
            (max_range - history_range).abs() < 1e-9,
            "the largest counted cycle must span the whole history: {max_range} vs {history_range}"
        );
        // And the interior wiggle 6 -> 8 is present as its own small cycle of range 2.
        assert!(
            cycles.iter().any(|c| (c.range - 2.0).abs() < 1e-9),
            "the small interior oscillation must be counted: {cycles:?}"
        );
    }

    #[test]
    fn the_largest_cycle_always_spans_the_whole_history() {
        // A general property, checked on many shapes rather than one.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        for _ in 0..200 {
            let series: Vec<f64> = (0..40).map(|_| 50.0 * next()).collect();
            let cycles = rainflow(&series);
            if cycles.is_empty() {
                continue;
            }
            let max_range = cycles.iter().map(|c| c.range).fold(0.0f64, f64::max);
            let rev = reversals(&series);
            let span = rev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - rev.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(
                (max_range - span).abs() < 1e-9 * span,
                "largest cycle {max_range} should equal the span {span}"
            );
        }
    }

    #[test]
    fn rainflow_conserves_total_variation_on_random_histories() {
        // The conservation law is the real test of the bookkeeping: dropping the residue, double-counting a
        // closed cycle, or losing the first point each break it immediately.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        for len in [2usize, 3, 5, 8, 17, 50, 200] {
            for _ in 0..60 {
                let series: Vec<f64> = (0..len).map(|_| 100.0 * next()).collect();
                assert_conserves(&series);
            }
        }
        // And on degenerate inputs, where an off-by-one would show first.
        assert_conserves(&[]);
        assert_conserves(&[1.0]);
        assert_conserves(&[1.0, 2.0]);
        assert_conserves(&[1.0, 1.0, 1.0]);
        assert_conserves(&[0.0, 5.0, 0.0]);
    }

    #[test]
    fn the_residue_is_counted_and_matters_most_for_short_records() {
        // A monotone ramp closes nothing at all, so ALL of its damage is residue. An implementation that
        // discards the residue reports zero here, which is a silent and total underestimate.
        let ramp = [0.0, 100.0];
        let cycles = rainflow(&ramp);
        assert_eq!(cycles.len(), 1, "a single excursion is one half cycle");
        assert_eq!(cycles[0].count, 0.5);
        assert!((cycles[0].range - 100.0).abs() < 1e-12);
        assert!((cycles[0].mean - 50.0).abs() < 1e-12);
        assert!(damage(&cycles, &steel(), MeanCorrection::None) > 0.0, "the residue must do damage");
    }

    #[test]
    fn the_sn_curve_inverts_itself_and_reports_infinite_life_as_none() {
        let sn = steel();
        for &n in &[1.0, 10.0, 1e3, 1e6, 1e9] {
            let s = sn.amplitude_at(n);
            let back = sn.cycles_to_failure(s).expect("finite life without an endurance limit");
            assert!((back / n - 1.0).abs() < 1e-9, "N={n}: round trip gave {back}");
        }
        // At the coefficient, failure in exactly one cycle: the definition of A.
        assert!((sn.cycles_to_failure(1000.0).expect("finite") - 1.0).abs() < 1e-9);
        // Zero amplitude is unbounded life, reported as None rather than as a huge number.
        assert!(sn.cycles_to_failure(0.0).is_none());
        // With an endurance limit, anything at or below it is unbounded.
        let with_limit = sn.with_endurance_limit(100.0);
        assert!(with_limit.cycles_to_failure(100.0).is_none(), "at the limit is unbounded");
        assert!(with_limit.cycles_to_failure(99.9).is_none());
        assert!(with_limit.cycles_to_failure(100.1).is_some(), "just above it is not");
        // Non-physical parameters are rejected.
        assert!(SnCurve::basquin(0.0, 5.0).is_none());
        assert!(SnCurve::basquin(1000.0, 0.0).is_none());
        assert!(SnCurve::basquin(-1.0, 5.0).is_none());
    }

    #[test]
    fn miner_predicts_failure_at_exactly_the_curves_life() {
        // The consistency requirement between the two halves. A constant-amplitude history of N(S) cycles must
        // give D = 1.
        let sn = steel();
        let s = 200.0;
        let n = sn.cycles_to_failure(s).expect("finite");
        let cycles = vec![Cycle { range: 2.0 * s, mean: 0.0, count: n }];
        let d = damage(&cycles, &sn, MeanCorrection::None);
        assert!((d - 1.0).abs() < 1e-12, "D should be exactly 1 at the rated life, got {d}");
        // Half the cycles, half the damage: linearity is Miner's defining assumption.
        let half = vec![Cycle { range: 2.0 * s, mean: 0.0, count: 0.5 * n }];
        assert!((damage(&half, &sn, MeanCorrection::None) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn damage_follows_the_sn_exponent_in_amplitude() {
        // The claim in the module docs, as a number: with b = 5, a 20% larger amplitude does 1.2^5 = 2.49x the
        // damage. This is why an action-smoothness penalty is not a nicety.
        let sn = steel();
        let base = vec![Cycle { range: 200.0, mean: 0.0, count: 1000.0 }];
        let d0 = damage(&base, &sn, MeanCorrection::None);
        for f in [1.1f64, 1.2, 1.5, 2.0] {
            let scaled = vec![Cycle { range: 200.0 * f, mean: 0.0, count: 1000.0 }];
            let d = damage(&scaled, &sn, MeanCorrection::None);
            let expect = f.powf(sn.exponent);
            assert!(
                (d / d0 / expect - 1.0).abs() < 1e-9,
                "a {f}x amplitude should be {expect:.3}x the damage, got {:.3}x",
                d / d0
            );
        }
        // And spell out the headline figure so it cannot drift unnoticed.
        assert!((1.2f64.powf(5.0) - 2.488_32).abs() < 1e-4);
    }

    #[test]
    fn a_tensile_mean_shortens_life_and_the_corrections_order_as_expected() {
        // Ignoring the mean is optimistic, which is the direction that matters.
        let sn = steel();
        let (amp, mean) = (150.0, 300.0);
        let ultimate = 900.0;
        let none = MeanCorrection::None.equivalent_amplitude(amp, mean).expect("finite");
        let gerber = MeanCorrection::Gerber { ultimate }.equivalent_amplitude(amp, mean).expect("finite");
        let goodman = MeanCorrection::Goodman { ultimate }.equivalent_amplitude(amp, mean).expect("finite");
        let soderberg =
            MeanCorrection::Soderberg { yield_strength: 600.0 }.equivalent_amplitude(amp, mean).expect("finite");

        assert_eq!(none, amp, "no correction returns the raw amplitude");
        assert!(none < gerber, "any correction must increase the equivalent amplitude");
        assert!(gerber < goodman, "Gerber is less conservative than Goodman");
        assert!(goodman < soderberg, "Soderberg (to yield) is the most conservative");

        // Goodman's closed form, checked: 150/(1 - 300/900) = 225.
        assert!((goodman - 225.0).abs() < 1e-9, "got {goodman}");
        // Gerber's: 150/(1 - (1/3)^2) = 168.75.
        assert!((gerber - 168.75).abs() < 1e-9, "got {gerber}");

        // At zero mean every correction is the identity, which is the boundary condition they must share.
        for c in [
            MeanCorrection::Goodman { ultimate },
            MeanCorrection::Gerber { ultimate },
            MeanCorrection::Soderberg { yield_strength: 600.0 },
        ] {
            assert!((c.equivalent_amplitude(amp, 0.0).expect("finite") - amp).abs() < 1e-12);
        }
        // At the reference strength there is no life left, reported as None rather than a huge number.
        assert!(MeanCorrection::Goodman { ultimate }.equivalent_amplitude(amp, ultimate).is_none());
        assert!(MeanCorrection::Goodman { ultimate }.equivalent_amplitude(amp, 2.0 * ultimate).is_none());
        // And an infinite equivalent amplitude becomes infinite damage, not a skipped cycle.
        let at_limit = vec![Cycle { range: 2.0 * amp, mean: ultimate, count: 1.0 }];
        assert!(damage(&at_limit, &sn, MeanCorrection::Goodman { ultimate }).is_infinite());

        // A compressive mean is not credited: it returns the raw amplitude rather than an unbounded benefit.
        let compressive = MeanCorrection::Goodman { ultimate }.equivalent_amplitude(amp, -5000.0);
        assert_eq!(compressive, Some(amp), "compressive means are clamped, not extrapolated");
    }

    /// A torque trace: a mean, a slow excursion, and an optional high-frequency dither on top.
    fn policy_trace(n: usize, mean: f64, slow_amp: f64, dither_amp: f64, dither_hz: f64) -> Vec<f64> {
        (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                let slow = slow_amp * (2.0 * std::f64::consts::PI * 3.0 * t).sin();
                let dither = if dither_amp == 0.0 {
                    0.0
                } else {
                    dither_amp * (2.0 * std::f64::consts::PI * dither_hz * t).sin()
                };
                mean + slow + dither
            })
            .collect()
    }

    #[test]
    fn a_chattering_policy_costs_one_to_two_orders_of_magnitude_more_life() {
        // THE result this module exists for. Both traces hold the same mean torque and the same slow
        // excursion; one of them chatters. A reward function sees no difference. The gearbox does.
        //
        // My first version of this test asserted a ratio above 50 with a dither at 62% of the command
        // amplitude, and measured 21.9. The threshold was the wrong part to keep: the effect is real but its
        // size depends on how hard the policy chatters, so the numbers below are measured and the test asserts
        // the structure (monotone in amplitude) plus one conservative floor.
        let sn = steel();
        let corr = MeanCorrection::Goodman { ultimate: 900.0 };
        let n = 40_000;
        let (mean_torque, slow_amp) = (120.0, 40.0);

        // The trace must be well sampled, or the dither's peaks are clipped and the damage understated. An
        // earlier probe at 4000 samples gave 5 samples per period at 800 Hz and a materially wrong number.
        let dither_hz = 400.0;
        let samples_per_period = n as f64 / dither_hz;
        assert!(
            samples_per_period >= 20.0,
            "the dither must be well sampled, only {samples_per_period} samples per period"
        );

        let smooth = policy_trace(n, mean_torque, slow_amp, 0.0, 0.0);
        let chatter = policy_trace(n, mean_torque, slow_amp, slow_amp, dither_hz);

        // The sampled peak must reach the commanded peak, which is the operational check on the above.
        let peak = chatter.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (peak - (mean_torque + 2.0 * slow_amp)).abs() < 0.1,
            "sampled peak {peak} should reach the commanded {}",
            mean_torque + 2.0 * slow_amp
        );

        // The two must be comparable on the coarse statistics a reward would notice.
        let mean_of = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            (mean_of(&smooth) - mean_of(&chatter)).abs() < 1.0,
            "the mean torque must match, or this compares two different tasks"
        );

        let d_smooth = damage_from_history(&smooth, &sn, corr);
        let d_chatter = damage_from_history(&chatter, &sn, corr);
        assert!(d_smooth > 0.0 && d_chatter > 0.0);
        let ratio = d_chatter / d_smooth;
        // Measured 188.1 at this configuration; the floor is set well below it and would still fail for a
        // model that could not tell the two traces apart.
        assert!(ratio > 50.0, "a policy chattering at its command amplitude should cost far more, got {ratio:.1}");

        // Monotone in dither amplitude: the structural claim, which does not depend on picking a number.
        let mut prev = 0.0;
        for frac in [0.0f64, 0.25, 0.62, 1.0] {
            let t = policy_trace(n, mean_torque, slow_amp, frac * slow_amp, dither_hz);
            let d = damage_from_history(&t, &sn, corr);
            assert!(d >= prev, "damage must not fall as the dither grows, at {frac}");
            prev = d;
        }

        // The practical statement: episodes to failure.
        let r_smooth = repetitions_to_failure(&smooth, &sn, corr).expect("does damage");
        let r_chatter = repetitions_to_failure(&chatter, &sn, corr).expect("does damage");
        assert!(r_smooth > 50.0 * r_chatter, "{r_smooth:.0} vs {r_chatter:.0} episodes to failure");

        // And the equivalent constant amplitude summarises it in one comparable number.
        let e_smooth = equivalent_amplitude(&smooth, &sn, corr).expect("finite");
        let e_chatter = equivalent_amplitude(&chatter, &sn, corr).expect("finite");
        assert!(e_chatter > e_smooth, "{e_chatter:.2} vs {e_smooth:.2}");
    }

    #[test]
    fn dither_damage_grows_sub_linearly_in_frequency_because_slow_drift_inflates_each_cycle() {
        // A real rainflow property, and one I would have mis-stated. Doubling the dither frequency doubles the
        // number of counted cycles exactly, so a naive reading says damage doubles. Measured: 1.29x, 1.44x,
        // 1.60x for successive doublings. The reason is that at LOW dither frequency each dither cycle spans
        // enough of the slow excursion that the drift adds to its range, making fewer but individually worse
        // cycles. Pinning this keeps the module docs honest about the frequency dependence.
        let sn = steel();
        let corr = MeanCorrection::Goodman { ultimate: 900.0 };
        let n = 40_000;
        let base = damage_from_history(&policy_trace(n, 120.0, 40.0, 0.0, 0.0), &sn, corr);

        let mut prev: Option<(f64, f64, f64)> = None;
        for &hz in &[100.0f64, 200.0, 400.0, 800.0] {
            let t = policy_trace(n, 120.0, 40.0, 25.0, hz);
            let cycles: f64 = rainflow(&t).iter().map(|c| c.count).sum();
            let d = damage_from_history(&t, &sn, corr) - base;
            if let Some((p_hz, p_cycles, p_d)) = prev {
                let freq_ratio = hz / p_hz;
                let cycle_ratio = cycles / p_cycles;
                let damage_ratio = d / p_d;
                // The cycle count really does track the frequency, so the sub-linearity is about the ranges.
                assert!(
                    (cycle_ratio - freq_ratio).abs() < 0.05 * freq_ratio,
                    "{p_hz}->{hz} Hz: cycles scaled {cycle_ratio:.3} vs frequency {freq_ratio:.3}"
                );
                assert!(
                    damage_ratio > 1.0 && damage_ratio < cycle_ratio,
                    "{p_hz}->{hz} Hz: damage scaled {damage_ratio:.3}, should be between 1 and {cycle_ratio:.3}"
                );
            }
            prev = Some((hz, cycles, d));
        }
    }

    #[test]
    fn an_endurance_limit_can_make_a_duty_cycle_free() {
        // The design target: keep every cycle under the limit and the part does not wear out. Reported as
        // `None` repetitions rather than a very large number, because they are different claims.
        let sn = steel().with_endurance_limit(100.0);
        let gentle: Vec<f64> = (0..500)
            .map(|k| 50.0 * (2.0 * std::f64::consts::PI * k as f64 / 100.0).sin())
            .collect();
        // Every cycle has amplitude 50, under the 100 limit.
        assert_eq!(damage_from_history(&gentle, &sn, MeanCorrection::None), 0.0);
        assert!(repetitions_to_failure(&gentle, &sn, MeanCorrection::None).is_none());

        // Scale it past the limit and life becomes finite, which is the sharp edge the limit describes.
        let harsh: Vec<f64> = gentle.iter().map(|v| v * 2.5).collect();
        assert!(damage_from_history(&harsh, &sn, MeanCorrection::None) > 0.0);
        assert!(repetitions_to_failure(&harsh, &sn, MeanCorrection::None).is_some());
    }

    #[test]
    fn the_equivalent_amplitude_depends_on_the_material_which_is_why_rms_is_not_a_substitute() {
        // A single-number summary of a history is only meaningful relative to an exponent. Two materials rank
        // the same history differently, and an RMS torque cannot express that.
        let series: Vec<f64> = (0..2000)
            .map(|k| {
                let t = k as f64 / 2000.0;
                200.0 * (2.0 * std::f64::consts::PI * 2.0 * t).sin()
                    + 60.0 * (2.0 * std::f64::consts::PI * 90.0 * t).sin()
            })
            .collect();
        let soft = SnCurve::basquin(1000.0, 3.0).expect("valid");
        let hard = SnCurve::basquin(1000.0, 10.0).expect("valid");
        let e_soft = equivalent_amplitude(&series, &soft, MeanCorrection::None).expect("finite");
        let e_hard = equivalent_amplitude(&series, &hard, MeanCorrection::None).expect("finite");
        assert!(
            (e_soft - e_hard).abs() > 1.0,
            "the same history must summarise differently on different materials: {e_soft:.3} vs {e_hard:.3}"
        );
        // The high-exponent material weights the largest cycles more heavily, so its equivalent amplitude sits
        // closer to the peak amplitude of the history.
        let peak = rainflow(&series).iter().map(|c| c.amplitude()).fold(0.0f64, f64::max);
        assert!(
            (peak - e_hard).abs() < (peak - e_soft).abs(),
            "b=10 should sit nearer the peak {peak:.1}: hard {e_hard:.1}, soft {e_soft:.1}"
        );
    }
}
