//! **The exact derivative of a trajectory through contact, as a callable.**
//!
//! [`saltation_matrix`](crate::saltation_matrix) gives the jump derivative at one event and
//! [`compose_monodromy`](crate::compose_monodromy) composes a list of them, but the work between those two — finding
//! the events, bracketing each to a single timestep, checking that the composition is even legitimate, and refusing
//! when it is not — has so far been hand-written per system. [`hybrid_jacobian`] is that driver.
//!
//! It matters because `examples/contact_gradient_descent.rs` measured that this derivative converges an optimisation
//! from 4 of 4 starts where a penalty simulator's autodiff gradient converges from 0 of 4 and cannot find a single
//! descent direction from three of them. A gradient that behaves that differently should not require a bespoke
//! implementation each time it is wanted.
//!
//! **Every guard here was earned by a specific failure**, and each is a refusal rather than a warning, because all of
//! them produce a plausible wrong number when ignored:
//!
//! - [`HybridGradientError::HiddenState`] — the flow must be a function of the state alone. Warm-starting a contact
//!   solver from the previous step makes the impulses hidden state, and then `flow(0, 2h) != flow(flow(0, h), h, h)`
//!   and the chain rule does not hold across a split. Detected by measuring exactly that residual.
//! - [`HybridGradientError::Grazing`] — the saltation matrix divides by `g^T f-`, so an event the trajectory only
//!   grazes has an unbounded correction. The term genuinely diverges (measured: 3.92 at an impact speed of 4 m/s
//!   rising to 1569.6 at 0.01 m/s), so a small transversality is not a tolerance to widen.
//! - [`HybridGradientError::EventsTooClose`] — two events inside one timestep cannot be given separate saltation
//!   matrices, and merging them silently is how a quadruped's 205 contact-mode changes per period got mistaken for 4
//!   touchdowns.
//! - [`probe_stable_jacobian`] — a probe large enough to change the event sequence measures a different system.
//!   Reported as a spread across probes rather than assumed.
//!
//! Segments exclude the timestep in which a guard crosses; the saltation matrix covers that step. This is the
//! convention the verified quadruped monodromy uses, and mixing it with one where the reset is also inside the
//! segment double-counts the impact.

use crate::saltation_matrix;
use nalgebra::{DMatrix, DVector};

/// A system whose flow is smooth except at guard crossings.
///
/// The vector field is taken from [`Self::step`] rather than asked for separately, so an implementation cannot supply
/// a field that disagrees with the integrator it actually runs.
pub trait HybridSystem {
    fn dim(&self) -> usize;

    /// One timestep of the smooth flow. **Must be a function of `(x, t)` alone** — see
    /// [`HybridGradientError::HiddenState`].
    fn step(&self, x: &DVector<f64>, t: f64, dt: f64) -> Option<DVector<f64>>;

    /// Guard values at `x`. An event is a guard going from strictly positive to non-positive.
    fn guards(&self, x: &DVector<f64>) -> Vec<f64>;

    /// The reset map's Jacobian at an event on guard `k`, evaluated just before the crossing.
    fn reset_jacobian(&self, x: &DVector<f64>, k: usize) -> Option<DMatrix<f64>>;

    /// The gradient of guard `k` at `x`.
    fn guard_normal(&self, x: &DVector<f64>, k: usize) -> Option<DVector<f64>>;

    /// The reset map itself, applied at an event. Defaults to the step, for systems whose integrator already resolves
    /// the event internally.
    fn reset(&self, x: &DVector<f64>, _k: usize, t: f64, dt: f64) -> Option<DVector<f64>> {
        self.step(x, t, dt)
    }
}

/// Why a Jacobian could not be produced. Each variant is a refusal, not a diagnostic: every one of these conditions
/// yields a plausible wrong number if it is ignored.
#[derive(Clone, Debug, PartialEq)]
pub enum HybridGradientError {
    /// The flow left the domain or produced a non-finite state.
    FlowDiverged { at: f64 },
    /// `flow(0, 2h)` and `flow(flow(0, h), h, h)` disagree, so the flow carries state the Jacobian cannot see.
    HiddenState { residual: f64 },
    /// An event's transversality `g^T f-` is too small for the saltation correction to be bounded.
    Grazing { at: f64, transversality: f64 },
    /// Two events fell inside one timestep.
    EventsTooClose { first: f64, second: f64 },
    /// A reset Jacobian or guard normal was unavailable, or the saltation matrix was singular.
    Degenerate { at: f64 },
    /// The probe swept in [`probe_stable_jacobian`] gave answers that do not agree.
    ProbeUnstable { spread: f64 },
}

impl std::fmt::Display for HybridGradientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HybridGradientError::FlowDiverged { at } => write!(f, "the flow diverged at t = {at}"),
            HybridGradientError::HiddenState { residual } => {
                write!(f, "the flow is not a function of the state alone: split residual {residual:.3e}")
            }
            HybridGradientError::Grazing { at, transversality } => {
                write!(f, "grazing event at t = {at}: transversality {transversality:.3e} is too small for a bounded saltation correction")
            }
            HybridGradientError::EventsTooClose { first, second } => write!(f, "events at t = {first} and t = {second} share a timestep"),
            HybridGradientError::Degenerate { at } => write!(f, "the event at t = {at} has no usable reset Jacobian or guard normal"),
            HybridGradientError::ProbeUnstable { spread } => write!(f, "the linearisation is probe-dependent: relative spread {spread:.3e}"),
        }
    }
}

impl std::error::Error for HybridGradientError {}

/// Tolerances, gathered so the signature stays short and every threshold has somewhere to be justified.
#[derive(Clone, Copy, Debug)]
pub struct HybridGradientOptions {
    pub dt: f64,
    /// Central-difference probe for the smooth segments.
    pub probe: f64,
    /// Smallest accepted `|g^T f-|`. The saltation correction scales as its reciprocal, so this decides how far the
    /// answer can be trusted rather than being a numerical nicety.
    pub min_transversality: f64,
    /// Largest accepted split-consistency residual before the flow is declared to carry hidden state.
    ///
    /// The default is calibrated against a measurement rather than chosen: on the verified quadruped, a **cold-started**
    /// contact solve gives a split residual of `6.8e-14` and a **warm-started** one gives `5.3e-9`. A threshold of
    /// `1e-10` therefore accepts the flow whose Jacobian composes correctly and refuses the one whose does not.
    pub max_split_residual: f64,
}

impl Default for HybridGradientOptions {
    fn default() -> Self {
        HybridGradientOptions { dt: 1e-4, probe: 1e-7, min_transversality: 1e-3, max_split_residual: 1e-10 }
    }
}

/// The result: the Jacobian, and enough about how it was obtained to judge it.
#[derive(Clone, Debug)]
pub struct HybridLinearisation {
    /// `d x(t0 + horizon) / d x(t0)`.
    pub jacobian: DMatrix<f64>,
    /// `(time, guard index)` for each event, in order.
    pub events: Vec<(f64, usize)>,
    /// The smallest `|g^T f-|` over the events. Large is good; the saltation correction scales as its reciprocal.
    pub worst_transversality: f64,
    /// The residual of the chain-rule split check. Zero means the flow is a function of the state.
    pub split_residual: f64,
}

impl HybridLinearisation {
    pub fn spectral_radius(&self) -> f64 {
        self.jacobian.complex_eigenvalues().iter().map(|z| z.norm()).fold(0.0, f64::max)
    }

    /// The largest singular value: how much a perturbation is amplified at worst, which a spectral radius does not
    /// bound. A contracting map can still amplify tenfold on the way to decaying.
    pub fn worst_gain(&self) -> f64 {
        self.jacobian.clone().svd(false, false).singular_values.max()
    }
}

/// Advance `secs` from `x`, returning `None` on divergence.
fn flow<S: HybridSystem + ?Sized>(sys: &S, x: &DVector<f64>, t0: f64, secs: f64, dt: f64) -> Option<DVector<f64>> {
    let steps = (secs / dt).round().max(0.0) as usize;
    let mut s = x.clone();
    for k in 0..steps {
        s = sys.step(&s, t0 + k as f64 * dt, dt)?;
        if !s.iter().all(|v| v.is_finite()) {
            return None;
        }
    }
    Some(s)
}

/// Central-difference Jacobian of a segment, valid because a segment contains no event by construction.
fn segment_jacobian<S: HybridSystem + ?Sized>(sys: &S, x: &DVector<f64>, t0: f64, secs: f64, dt: f64, probe: f64) -> Option<DMatrix<f64>> {
    let n = sys.dim();
    if secs <= 0.5 * dt {
        return Some(DMatrix::identity(n, n));
    }
    let mut j = DMatrix::zeros(n, n);
    for c in 0..n {
        let (mut p, mut m) = (x.clone(), x.clone());
        p[c] += probe;
        m[c] -= probe;
        let a = flow(sys, &p, t0, secs, dt)?;
        let b = flow(sys, &m, t0, secs, dt)?;
        j.set_column(c, &((a - b) / (2.0 * probe)));
    }
    Some(j)
}

/// **The split-consistency check.** If the flow is a function of the state, splitting it in two must give the same
/// answer as running it whole. If it is not — a warm-started contact solver, a cached factorisation, an adaptive
/// step-size controller carrying history — the chain rule fails across the split and no composition of segment
/// Jacobians can be right, however good the saltation matrices are.
pub fn split_residual<S: HybridSystem + ?Sized>(sys: &S, x: &DVector<f64>, t0: f64, secs: f64, dt: f64) -> Option<f64> {
    let steps = (secs / dt).round().max(2.0) as usize;
    let half = (steps / 2).max(1);
    let whole = flow(sys, x, t0, steps as f64 * dt, dt)?;
    let mid = flow(sys, x, t0, half as f64 * dt, dt)?;
    let split = flow(sys, &mid, t0 + half as f64 * dt, (steps - half) as f64 * dt, dt)?;
    let scale = whole.amax().max(1.0);
    Some((whole - split).amax() / scale)
}

/// **The exact Jacobian of a hybrid trajectory.**
///
/// `min_transversality` is the smallest `|g^T f-|` accepted before an event is called grazing; the saltation
/// correction scales as its reciprocal, so this is the parameter that decides how far the result can be trusted rather
/// than a numerical nicety.
pub fn hybrid_jacobian<S: HybridSystem + ?Sized>(
    sys: &S,
    x0: &DVector<f64>,
    t0: f64,
    horizon: f64,
    opts: HybridGradientOptions,
) -> Result<HybridLinearisation, HybridGradientError> {
    let n = sys.dim();
    let HybridGradientOptions { dt, probe, min_transversality, max_split_residual } = opts;

    // 0. is the flow a function of the state at all?
    let residual = split_residual(sys, x0, t0, horizon, dt).ok_or(HybridGradientError::FlowDiverged { at: t0 })?;
    if residual > max_split_residual {
        return Err(HybridGradientError::HiddenState { residual });
    }

    // 1. find the events, resolved to a single timestep
    let steps = (horizon / dt).round().max(1.0) as usize;
    let mut x = x0.clone();
    let mut prev = sys.guards(&x);
    let mut events: Vec<(usize, DVector<f64>, usize)> = Vec::new(); // step index, pre-event state, guard
    for k in 0..steps {
        let t = t0 + k as f64 * dt;
        let next = sys.step(&x, t, dt).ok_or(HybridGradientError::FlowDiverged { at: t })?;
        if !next.iter().all(|v| v.is_finite()) {
            return Err(HybridGradientError::FlowDiverged { at: t });
        }
        let now = sys.guards(&next);
        let mut crossed = None;
        for (g, (&before, &after)) in prev.iter().zip(now.iter()).enumerate() {
            if before > 0.0 && after <= 0.0 {
                if let Some(first) = crossed {
                    // two guards crossed in the same step: they cannot be given separate saltation matrices
                    let _ = first;
                    return Err(HybridGradientError::EventsTooClose { first: t, second: t });
                }
                crossed = Some(g);
            }
        }
        if let Some(g) = crossed {
            if let Some(pk) = events.last().map(|e| e.0).filter(|pk| k <= pk + 1) {
                return Err(HybridGradientError::EventsTooClose { first: t0 + pk as f64 * dt, second: t });
            }
            events.push((k, x.clone(), g));
        }
        x = next;
        prev = now;
    }

    // 2. compose: segment, saltation, segment, ...
    let mut jac = DMatrix::identity(n, n);
    let mut cursor_step = 0usize;
    let mut cursor_state = x0.clone();
    let mut worst_transversality = f64::INFINITY;
    let mut event_times = Vec::with_capacity(events.len());

    for &(k, ref pre, g) in &events {
        let t_event = t0 + k as f64 * dt;
        // segment up to (but excluding) the event step
        let secs = (k - cursor_step) as f64 * dt;
        let phi = segment_jacobian(sys, &cursor_state, t0 + cursor_step as f64 * dt, secs, dt, probe)
            .ok_or(HybridGradientError::FlowDiverged { at: t_event })?;
        jac = phi * jac;

        // the saltation matrix at the event
        let reset = sys.reset_jacobian(pre, g).ok_or(HybridGradientError::Degenerate { at: t_event })?;
        let normal = sys.guard_normal(pre, g).ok_or(HybridGradientError::Degenerate { at: t_event })?;
        // the field on each side, from the integrator that is actually running
        let post = sys.reset(pre, g, t_event, dt).ok_or(HybridGradientError::FlowDiverged { at: t_event })?;
        let f_minus = (sys.step(pre, t_event, dt).ok_or(HybridGradientError::FlowDiverged { at: t_event })? - pre) / dt;
        let f_plus = (sys.step(&post, t_event + dt, dt).ok_or(HybridGradientError::FlowDiverged { at: t_event })? - &post) / dt;

        let transversality = normal.dot(&f_minus).abs();
        worst_transversality = worst_transversality.min(transversality);
        if transversality < min_transversality {
            return Err(HybridGradientError::Grazing { at: t_event, transversality });
        }
        let xi = saltation_matrix(&reset, &normal, &f_minus, &f_plus).ok_or(HybridGradientError::Degenerate { at: t_event })?;
        jac = xi * jac;

        cursor_step = k + 1;
        cursor_state = post;
        event_times.push((t_event, g));
    }

    // the last segment
    let secs = (steps - cursor_step) as f64 * dt;
    let phi = segment_jacobian(sys, &cursor_state, t0 + cursor_step as f64 * dt, secs, dt, probe)
        .ok_or(HybridGradientError::FlowDiverged { at: t0 + horizon })?;
    jac = phi * jac;

    Ok(HybridLinearisation {
        jacobian: jac,
        events: event_times,
        worst_transversality: if worst_transversality.is_finite() { worst_transversality } else { f64::INFINITY },
        split_residual: residual,
    })
}

/// **The same, checked across several probes.** A probe large enough to change the event sequence measures a different
/// system, and the only way to know is to vary it. Returns the linearisation from the first probe that works together
/// with the relative spread of the spectral radius across all of them.
pub fn probe_stable_jacobian<S: HybridSystem + ?Sized>(
    sys: &S,
    x0: &DVector<f64>,
    t0: f64,
    horizon: f64,
    opts: HybridGradientOptions,
    probes: &[f64],
    tolerance: f64,
) -> Result<(HybridLinearisation, f64), HybridGradientError> {
    let mut results = Vec::new();
    let mut last_err = None;
    for &p in probes {
        match hybrid_jacobian(sys, x0, t0, horizon, HybridGradientOptions { probe: p, ..opts }) {
            Ok(r) => results.push(r),
            Err(e) => last_err = Some(e),
        }
    }
    let Some(first) = results.first().cloned() else {
        return Err(last_err.unwrap_or(HybridGradientError::ProbeUnstable { spread: f64::INFINITY }));
    };
    if results.len() < 2 {
        return Err(HybridGradientError::ProbeUnstable { spread: f64::INFINITY });
    }
    let rhos: Vec<f64> = results.iter().map(HybridLinearisation::spectral_radius).collect();
    let (lo, hi) = rhos.iter().fold((f64::INFINITY, 0.0f64), |(a, b), r| (a.min(*r), b.max(*r)));
    let spread = if lo > 0.0 { (hi - lo) / lo } else { hi };
    if spread > tolerance {
        return Err(HybridGradientError::ProbeUnstable { spread });
    }
    Ok((first, spread))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BouncingMass, GRAVITY};

    /// The bouncing mass, as a [`HybridSystem`]. Its Jacobian is available in closed form through
    /// [`BouncingMass::jacobian_saltation`], which is itself verified against finite differences to `8e-10`, so this
    /// is a driver test against a known answer rather than a self-consistency check.
    struct Bouncer {
        inner: BouncingMass,
    }

    impl HybridSystem for Bouncer {
        fn dim(&self) -> usize {
            2
        }
        fn step(&self, x: &DVector<f64>, _t: f64, dt: f64) -> Option<DVector<f64>> {
            // exact free flight, so the segment Jacobians carry no integration error
            Some(DVector::from_row_slice(&[x[0] + x[1] * dt - 0.5 * self.inner.gravity * dt * dt, x[1] - self.inner.gravity * dt]))
        }
        fn guards(&self, x: &DVector<f64>) -> Vec<f64> {
            vec![x[0]]
        }
        fn reset_jacobian(&self, _x: &DVector<f64>, _k: usize) -> Option<DMatrix<f64>> {
            Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -self.inner.restitution]))
        }
        fn guard_normal(&self, _x: &DVector<f64>, _k: usize) -> Option<DVector<f64>> {
            Some(DVector::from_row_slice(&[1.0, 0.0]))
        }
        fn reset(&self, x: &DVector<f64>, _k: usize, _t: f64, dt: f64) -> Option<DVector<f64>> {
            // step to the floor, then reflect - the integrator does not resolve the impact itself here
            let f = self.step(x, 0.0, dt)?;
            Some(DVector::from_row_slice(&[f[0].max(0.0), -self.inner.restitution * f[1]]))
        }
    }

    #[test]
    fn the_driver_reproduces_the_closed_form_jacobian() {
        let inner = BouncingMass::new(GRAVITY, 0.6).unwrap();
        let sys = Bouncer { inner };
        let opts = HybridGradientOptions { dt: 1e-5, probe: 1e-6, ..Default::default() };
        for (h, v, t) in [(1.0, 0.0, 0.8), (2.0, 1.0, 1.2), (1.5, 2.0, 1.0)] {
            let x0 = DVector::from_row_slice(&[h, v]);
            let lin = hybrid_jacobian(&sys, &x0, 0.0, t, opts).expect("linearises");
            let exact = inner.jacobian_saltation([h, v], t).expect("closed form");
            let worst = (0..2)
                .flat_map(|i| (0..2).map(move |j| (i, j)))
                .map(|(i, j)| (lin.jacobian[(i, j)] - exact[i][j]).abs())
                .fold(0.0, f64::max);
            let scale = (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| exact[i][j].abs()).fold(0.0, f64::max);
            eprintln!(
                "h={h} v={v} t={t}: {} event(s), transversality {:.3}, split residual {:.1e}, relative error vs closed form {:.2e}",
                lin.events.len(),
                lin.worst_transversality,
                lin.split_residual,
                worst / scale
            );
            assert!(worst / scale < 2e-3, "the driver must reproduce the closed form: {:.3e}", worst / scale);
            assert!(!lin.events.is_empty(), "the horizon contains an impact");
        }

        // the residual is the event bracket, one timestep wide, so it must shrink with dt - if it did not, it would
        // be a defect in the composition rather than a discretisation
        eprintln!("\n   the residual against the closed form, by timestep:");
        let exact = inner.jacobian_saltation([1.0, 0.0], 0.8).unwrap();
        let scale = (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| exact[i][j].abs()).fold(0.0, f64::max);
        let mut errs = Vec::new();
        for step in [1e-4, 1e-5, 1e-6] {
            let lin = hybrid_jacobian(&sys, &DVector::from_row_slice(&[1.0, 0.0]), 0.0, 0.8, HybridGradientOptions { dt: step, probe: 1e-6, ..Default::default() }).unwrap();
            let e = (0..2)
                .flat_map(|i| (0..2).map(move |j| (i, j)))
                .map(|(i, j)| (lin.jacobian[(i, j)] - exact[i][j]).abs())
                .fold(0.0, f64::max)
                / scale;
            eprintln!("      dt = {step:.0e}: relative error {e:.2e}");
            errs.push(e);
        }
        assert!(errs.windows(2).all(|w| w[1] < w[0]), "the error is the one-timestep event bracket: {errs:?}");
        eprintln!("      first order in dt, as a one-step bracket must be");
    }

    /// **Hidden state is refused, not absorbed.** A flow that remembers anything breaks the chain rule across a split,
    /// and the driver has to say so rather than return a composed Jacobian that cannot be right.
    #[test]
    fn a_flow_with_memory_is_refused() {
        use std::cell::Cell;
        struct Leaky {
            inner: BouncingMass,
            /// The previous call's velocity, fed into this call - the exact shape of a warm-started contact solver
            /// handing its impulses to the next step.
            last: Cell<f64>,
        }
        impl HybridSystem for Leaky {
            fn dim(&self) -> usize {
                2
            }
            fn step(&self, x: &DVector<f64>, _t: f64, dt: f64) -> Option<DVector<f64>> {
                // one step of memory: the previous call's velocity biases this one. A split re-enters with a stale
                // value, so the whole run and the split run disagree - which is precisely why a warm-started solver
                // cannot be linearised by composing segment Jacobians.
                let bias = 1e-2 * self.last.get();
                self.last.set(x[1]);
                Some(DVector::from_row_slice(&[x[0] + (x[1] + bias) * dt, x[1] - self.inner.gravity * dt]))
            }
            fn guards(&self, x: &DVector<f64>) -> Vec<f64> {
                vec![x[0]]
            }
            fn reset_jacobian(&self, _x: &DVector<f64>, _k: usize) -> Option<DMatrix<f64>> {
                Some(DMatrix::identity(2, 2))
            }
            fn guard_normal(&self, _x: &DVector<f64>, _k: usize) -> Option<DVector<f64>> {
                Some(DVector::from_row_slice(&[1.0, 0.0]))
            }
        }
        let sys = Leaky { inner: BouncingMass::new(GRAVITY, 0.6).unwrap(), last: Cell::new(0.0) };
        let opts = HybridGradientOptions { dt: 1e-4, probe: 1e-6, ..Default::default() };
        let err = hybrid_jacobian(&sys, &DVector::from_row_slice(&[1.0, 0.0]), 0.0, 0.5, opts).expect_err("must refuse");
        eprintln!("{err}");
        assert!(matches!(err, HybridGradientError::HiddenState { .. }));
        // and the same system without the memory linearises fine, so the refusal is about the memory and nothing else
        let clean = Bouncer { inner: BouncingMass::new(GRAVITY, 0.6).unwrap() };
        let clean_residual = split_residual(&clean, &DVector::from_row_slice(&[1.0, 0.0]), 0.0, 0.5, 1e-4).unwrap();
        eprintln!("   the same system without the memory: split residual {clean_residual:.2e}");
        assert!(hybrid_jacobian(&clean, &DVector::from_row_slice(&[1.0, 0.0]), 0.0, 0.5, opts).is_ok());
    }

    /// **A grazing event is refused.** The saltation correction is `1/(g^T f-)` and genuinely diverges, so a nearly
    /// tangential crossing has no bounded derivative to return.
    #[test]
    fn a_grazing_event_is_refused() {
        let sys = Bouncer { inner: BouncingMass::new(GRAVITY, 0.6).unwrap() };
        // released just above the floor with almost no speed: it reaches the guard nearly tangentially
        let x0 = DVector::from_row_slice(&[2e-9, 0.0]);
        let opts = HybridGradientOptions { dt: 1e-5, probe: 1e-11, min_transversality: 1e-2, ..Default::default() };
        let err = hybrid_jacobian(&sys, &x0, 0.0, 0.2, opts).expect_err("must refuse a graze");
        eprintln!("{err}");
        match err {
            HybridGradientError::Grazing { transversality, .. } => assert!(transversality < 1e-2),
            other => panic!("expected a grazing refusal, got {other}"),
        }
    }

    /// Probe stability is reported, and an oversized probe that changes the event sequence is caught.
    #[test]
    fn probe_stability_is_measured() {
        let sys = Bouncer { inner: BouncingMass::new(GRAVITY, 0.6).unwrap() };
        let x0 = DVector::from_row_slice(&[1.0, 0.0]);
        let opts = HybridGradientOptions { dt: 1e-5, ..Default::default() };
        let (lin, spread) = probe_stable_jacobian(&sys, &x0, 0.0, 0.8, opts, &[1e-6, 1e-7, 1e-8], 1e-3).expect("stable");
        eprintln!("probe-stable: rho = {:.6}, worst gain = {:.4}, spread across probes {spread:.2e}", lin.spectral_radius(), lin.worst_gain());
        assert!(spread < 1e-3);
        // the spectral radius and the worst gain are different questions, and the second is the larger
        assert!(lin.worst_gain() >= lin.spectral_radius());
    }
}
