//! **Signal Temporal Logic with quantitative robustness** — saying what a task *is*, in a form a
//! controller can be scored against and a verifier can check.
//!
//! A reward function says "more is better" and leaves the requirement implicit; a temporal-logic
//! formula states the requirement. "Stay above 5 cm for the whole run, and reach the target within two
//! seconds of the door opening" is one sentence in this logic and several hand-tuned reward terms
//! without it.
//!
//! What makes STL useful for control rather than only for verification is the **quantitative**
//! semantics: instead of a yes/no answer, [`Stl::robustness`] returns a real number whose sign is the
//! satisfaction and whose magnitude is the margin. Positive means satisfied with that much room to
//! spare, negative means violated by that much. That turns a specification into something
//! differentiable-ish and directly optimisable, and it turns "did it pass" into "by how much", which is
//! what a certificate needs.
//!
//! Robustness follows the standard semantics (Fainekos and Pappas; Donzé and Maler): conjunction is a
//! minimum, disjunction a maximum, `always` a minimum over the window, `eventually` a maximum. Time is
//! in sample indices over a discrete trace. Pure Rust, no allocation beyond the formula tree.

/// A specification over a discrete trace of vector-valued samples. Time bounds `[a, b]` are in samples
/// and relative to the evaluation instant.
#[derive(Clone, Debug, PartialEq)]
pub enum Stl {
    /// Trivially satisfied, with unbounded margin.
    True,
    /// Trivially violated.
    False,
    /// `signal[i] ≥ c`; robustness is `signal[i] − c`, so the margin is in the signal's own units.
    Ge { i: usize, c: f64 },
    /// `signal[i] ≤ c`; robustness is `c − signal[i]`.
    Le { i: usize, c: f64 },
    Not(Box<Stl>),
    /// All of them, robustness = the worst.
    And(Vec<Stl>),
    /// Any of them, robustness = the best.
    Or(Vec<Stl>),
    /// `G_[a,b] φ` — holds at every instant in the window.
    Always { a: usize, b: usize, phi: Box<Stl> },
    /// `F_[a,b] φ` — holds at some instant in the window.
    Eventually { a: usize, b: usize, phi: Box<Stl> },
    /// `φ U_[a,b] ψ` — `ψ` holds somewhere in the window, and `φ` holds at every instant until then.
    Until { a: usize, b: usize, phi: Box<Stl>, psi: Box<Stl> },
}

impl Stl {
    /// `signal[i] ≥ c`.
    pub fn ge(i: usize, c: f64) -> Stl {
        Stl::Ge { i, c }
    }
    /// `signal[i] ≤ c`.
    pub fn le(i: usize, c: f64) -> Stl {
        Stl::Le { i, c }
    }
    /// `G_[a,b] self`.
    pub fn always(self, a: usize, b: usize) -> Stl {
        Stl::Always { a, b, phi: Box::new(self) }
    }
    /// `F_[a,b] self`.
    pub fn eventually(self, a: usize, b: usize) -> Stl {
        Stl::Eventually { a, b, phi: Box::new(self) }
    }
    /// `self U_[a,b] psi`.
    pub fn until(self, psi: Stl, a: usize, b: usize) -> Stl {
        Stl::Until { a, b, phi: Box::new(self), psi: Box::new(psi) }
    }
    /// Negation. Named `negate` rather than `not` so it cannot be confused with `std::ops::Not`.
    pub fn negate(self) -> Stl {
        Stl::Not(Box::new(self))
    }

    /// **Quantitative robustness** of the formula on `trace` at sample `t`. Positive means satisfied
    /// with that margin, negative means violated by that much, and the units are the signal's own.
    ///
    /// A window that runs past the end of the trace is truncated; a window entirely past the end yields
    /// `−∞` for `always`-like and `+∞`-free behaviour for `eventually`-like operators, because nothing
    /// in the trace can witness the requirement. That is the conservative reading: an unfinished run
    /// does not get credit for a requirement it never had the chance to violate.
    pub fn robustness(&self, trace: &[Vec<f64>], t: usize) -> f64 {
        match self {
            Stl::True => f64::INFINITY,
            Stl::False => f64::NEG_INFINITY,
            Stl::Ge { i, c } => trace.get(t).and_then(|s| s.get(*i)).map(|v| v - c).unwrap_or(f64::NEG_INFINITY),
            Stl::Le { i, c } => trace.get(t).and_then(|s| s.get(*i)).map(|v| c - v).unwrap_or(f64::NEG_INFINITY),
            Stl::Not(p) => -p.robustness(trace, t),
            Stl::And(ps) => ps.iter().fold(f64::INFINITY, |m, p| m.min(p.robustness(trace, t))),
            Stl::Or(ps) => ps.iter().fold(f64::NEG_INFINITY, |m, p| m.max(p.robustness(trace, t))),
            Stl::Always { a, b, phi } => {
                let (lo, hi) = window(trace.len(), t, *a, *b);
                if lo > hi {
                    return f64::NEG_INFINITY;
                }
                (lo..=hi).fold(f64::INFINITY, |m, k| m.min(phi.robustness(trace, k)))
            }
            Stl::Eventually { a, b, phi } => {
                let (lo, hi) = window(trace.len(), t, *a, *b);
                if lo > hi {
                    return f64::NEG_INFINITY;
                }
                (lo..=hi).fold(f64::NEG_INFINITY, |m, k| m.max(phi.robustness(trace, k)))
            }
            Stl::Until { a, b, phi, psi } => {
                let (lo, hi) = window(trace.len(), t, *a, *b);
                if lo > hi {
                    return f64::NEG_INFINITY;
                }
                // best witness instant: psi holds there, and phi held all the way from t to it
                (lo..=hi).fold(f64::NEG_INFINITY, |best, k| {
                    let hold = (t..=k).fold(f64::INFINITY, |m, j| m.min(phi.robustness(trace, j)));
                    best.max(psi.robustness(trace, k).min(hold))
                })
            }
        }
    }

    /// Whether the trace satisfies the formula at `t`. Sound with respect to
    /// [`robustness`](Self::robustness): a strictly positive robustness implies satisfaction and a
    /// strictly negative one implies violation.
    pub fn satisfied(&self, trace: &[Vec<f64>], t: usize) -> bool {
        self.robustness(trace, t) > 0.0
    }
}

/// The evaluation window `[t+a, t+b]` clipped to the trace. Returns `(lo, hi)` with `lo > hi` when the
/// window lies entirely past the end.
fn window(len: usize, t: usize, a: usize, b: usize) -> (usize, usize) {
    let lo = t + a;
    let hi = (t + b).min(len.saturating_sub(1));
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-dimensional trace of heights.
    fn heights(v: &[f64]) -> Vec<Vec<f64>> {
        v.iter().map(|x| vec![*x]).collect()
    }

    /// Robustness of a predicate is the margin in the signal's own units, and its sign is satisfaction.
    #[test]
    fn predicate_robustness_is_the_margin() {
        let tr = heights(&[0.20, 0.15, 0.05, 0.30]);
        let clear = Stl::ge(0, 0.10);
        assert!((clear.robustness(&tr, 0) - 0.10).abs() < 1e-12, "0.20 clears 0.10 by 0.10");
        assert!((clear.robustness(&tr, 2) - (-0.05)).abs() < 1e-12, "0.05 misses 0.10 by 0.05");
        assert!(clear.satisfied(&tr, 0) && !clear.satisfied(&tr, 2));
    }

    /// `always` is the worst instant in the window, so a single dip sets the robustness and the sign.
    #[test]
    fn always_is_decided_by_the_worst_instant() {
        let tr = heights(&[0.20, 0.15, 0.05, 0.30]);
        let spec = Stl::ge(0, 0.10).always(0, 3);
        let r = spec.robustness(&tr, 0);
        eprintln!("always(h >= 0.10) over the run: robustness {r:+.3} (the 0.05 dip)");
        assert!((r - (-0.05)).abs() < 1e-12, "the dip to 0.05 must set it, got {r}");
        assert!(!spec.satisfied(&tr, 0), "one violation violates the whole window");
        // and with the dip removed it passes with the margin of the next-worst instant
        let ok = heights(&[0.20, 0.15, 0.12, 0.30]);
        assert!((spec.robustness(&ok, 0) - 0.02).abs() < 1e-12, "0.12 is now the tightest");
        assert!(spec.satisfied(&ok, 0));
    }

    /// `eventually` needs one witness, and its robustness is the best one available.
    #[test]
    fn eventually_is_decided_by_the_best_instant() {
        let tr = heights(&[0.0, 0.2, 0.9, 0.1]);
        let reach = Stl::ge(0, 0.5).eventually(0, 3);
        assert!((reach.robustness(&tr, 0) - 0.4).abs() < 1e-12, "the 0.9 sample witnesses it by 0.4");
        assert!(reach.satisfied(&tr, 0));
        // a deadline that excludes the witness fails
        let early = Stl::ge(0, 0.5).eventually(0, 1);
        assert!(!early.satisfied(&tr, 0), "the target is not reached inside the shorter deadline");
    }

    /// `until` requires the guard to hold right up to the witness, which is what separates it from a
    /// plain conjunction of `always` and `eventually`.
    #[test]
    fn until_requires_the_guard_to_hold_until_the_witness() {
        // safe = column 0, goal = column 1
        let tr: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let spec = Stl::ge(0, 0.5).until(Stl::ge(1, 0.5), 0, 2);
        assert!(spec.satisfied(&tr, 0), "guard holds throughout and the goal arrives");

        // break the guard before the goal: the same goal at the same time no longer counts
        let broken: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]];
        let r = spec.robustness(&broken, 0);
        eprintln!("until with the guard broken mid-way: robustness {r:+.3}");
        assert!(!spec.satisfied(&broken, 0), "a broken guard invalidates the later witness, got {r}");
    }

    /// Conjunction takes the worst, disjunction the best, and negation flips the sign — so a spec built
    /// from parts has the margin of its weakest part, which is what makes robustness usable as a score.
    #[test]
    fn boolean_operators_compose_the_margins() {
        let tr = heights(&[0.30]);
        let a = Stl::ge(0, 0.10); // margin +0.20
        let b = Stl::le(0, 0.25); // margin −0.05
        assert!((Stl::And(vec![a.clone(), b.clone()]).robustness(&tr, 0) - (-0.05)).abs() < 1e-12);
        assert!((Stl::Or(vec![a.clone(), b.clone()]).robustness(&tr, 0) - 0.20).abs() < 1e-12);
        assert!((a.clone().negate().robustness(&tr, 0) - (-0.20)).abs() < 1e-12);
    }

    /// The property that makes robustness a usable objective: pushing the signal further inside the
    /// requirement monotonically increases it. A score that plateaus cannot guide a controller.
    #[test]
    fn robustness_increases_with_the_margin() {
        let spec = Stl::ge(0, 0.10).always(0, 2);
        let mut last = f64::NEG_INFINITY;
        for lift in [0.0, 0.05, 0.10, 0.20] {
            let tr = heights(&[0.12 + lift, 0.15 + lift, 0.11 + lift]);
            let r = spec.robustness(&tr, 0);
            assert!(r > last, "robustness must rise with the margin: {r} after {last}");
            last = r;
        }
        eprintln!("robustness rose monotonically with the safety margin, ending at {last:+.3}");
    }

    /// A requirement whose window runs past the end of the trace is not credited. An unfinished run
    /// should not pass a deadline it never reached.
    #[test]
    fn a_window_past_the_end_is_not_credited() {
        let tr = heights(&[0.9, 0.9]);
        let late = Stl::ge(0, 0.5).eventually(5, 9);
        assert!(!late.satisfied(&tr, 0), "a deadline beyond the trace cannot be satisfied");
        assert!(late.robustness(&tr, 0).is_infinite() && late.robustness(&tr, 0) < 0.0);
    }
}
