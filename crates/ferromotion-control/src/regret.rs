//! **Measuring closed-loop regret**, for any plant, with the variance-reduction that makes it measurable.
//!
//! [`lq_regret`](crate::lq_regret) gives the linear-quadratic answer in closed form. This module is the
//! instrument for checking whether that answer's *structure* survives on a nonlinear loop — which is P1's next
//! milestone and not a foregone conclusion, since the corrections there lean on expanding a smooth cost around
//! the expert's trajectory and a nonlinear loop's higher-order terms are exactly what that expansion drops.
//!
//! # Why common random numbers, and not more samples
//!
//! The quantity wanted is a *difference* of two average costs that are individually much larger than their gap.
//! On the reference linear system the regret is `0.0135` against costs of order one, so resolving it from
//! independent runs needs on the order of `10⁸` steps. Driving the expert and the learner with **the same process
//! noise** cancels the shared term exactly and leaves the difference, which turns an intractable measurement into
//! a cheap one. [`measure_regret`] does that by construction — it is not an option, and forgetting it is the
//! usual reason a regret measurement looks like noise.
//!
//! # The three error types, kept distinct
//!
//! [`ActionError`] covers the three cases that behave differently, because treating them as interchangeable is
//! the mistake the linear analysis exists to rule out: zero-mean sampling noise costs `Θ(η²/λ)`, a systematic
//! bias costs `Θ(η²/λ²)`, and a state-proportional error is safe only below an `H∞` margin and diverges above it.

use nalgebra::{DMatrix, DVector};

/// One step of a plant: `(state, input, disturbance) → next state`.
pub type Plant<'a> = &'a dyn Fn(&DVector<f64>, &DVector<f64>, &DVector<f64>) -> DVector<f64>;
/// A state-feedback policy.
pub type ExpertPolicy<'a> = &'a dyn Fn(&DVector<f64>) -> DVector<f64>;
/// A per-step cost, `(state, input) → cost`.
pub type StepCost<'a> = &'a dyn Fn(&DVector<f64>, &DVector<f64>) -> f64;

/// The residual by which a policy misses its expert. The three variants are the three cases of the linear
/// analysis, and they can be combined.
#[derive(Clone, Debug, Default)]
pub struct ActionError {
    /// Standard deviation of an i.i.d. zero-mean component, applied per input channel.
    pub sigma: f64,
    /// A constant offset that does not average out.
    pub bias: Option<DVector<f64>>,
    /// A state-proportional component `ΔK x` — what a systematically mis-fit score produces, and the one with a
    /// stability cliff rather than a cost.
    pub gain: Option<DMatrix<f64>>,
}

impl ActionError {
    /// Zero-mean sampling noise of the given per-channel standard deviation.
    pub fn noise(sigma: f64) -> Self {
        ActionError { sigma, ..Default::default() }
    }
    /// A constant action bias.
    pub fn constant(bias: DVector<f64>) -> Self {
        ActionError { bias: Some(bias), ..Default::default() }
    }
    /// A state-proportional error.
    pub fn state_dependent(gain: DMatrix<f64>) -> Self {
        ActionError { gain: Some(gain), ..Default::default() }
    }

    /// The per-step second moment `E‖e‖²`, which is the `η²` the agenda's bounds are stated in. The
    /// state-dependent part is excluded because its magnitude depends on the state, not on the policy alone —
    /// which is precisely why it is a different kind of error rather than a larger one.
    pub fn eta_squared(&self, inputs: usize) -> f64 {
        inputs as f64 * self.sigma * self.sigma + self.bias.as_ref().map(|b| b.norm_squared()).unwrap_or(0.0)
    }

    fn sample(&self, x: &DVector<f64>, inputs: usize, rng: &mut Xorshift) -> DVector<f64> {
        let mut e = DVector::zeros(inputs);
        if self.sigma != 0.0 {
            for i in 0..inputs {
                e[i] += self.sigma * rng.normal();
            }
        }
        if let Some(b) = &self.bias {
            e += b;
        }
        if let Some(g) = &self.gain {
            e += g * x;
        }
        e
    }
}

/// A deterministic Gaussian stream, so a regret measurement is reproducible and a test cannot flake.
pub struct Xorshift {
    state: u64,
}

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        Xorshift { state: seed | 1 }
    }
    /// A uniform on `(0, 1)`. Public because a Bernoulli trial needs a real uniform: deriving one by squashing a
    /// normal through a tanh is not uniform, and a reliability simulation built that way reported 0.51 where the
    /// closed form said 0.36.
    pub fn uniform(&mut self) -> f64 {
        self.next_u()
    }

    fn next_u(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A standard normal by Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.next_u().max(1e-12), self.next_u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The outcome of a regret measurement.
#[derive(Clone, Copy, Debug)]
pub struct RegretMeasurement {
    /// Average cost under the expert.
    pub expert_cost: f64,
    /// Average cost under the perturbed policy.
    pub policy_cost: f64,
    /// The difference — the quantity of interest, and much smaller than either term.
    pub regret: f64,
    /// Whether both runs stayed bounded. A diverged run makes the regret meaningless rather than large.
    pub bounded: bool,
    pub steps: usize,
}

/// **Measure closed-loop regret** of `expert + error` against `expert`, under shared process noise.
///
/// `plant` advances the state given `(x, u, w)`; `expert` is the nominal policy; `cost` is charged per step. The
/// two runs see identical `w` at every step, which is what makes a small regret resolvable.
///
/// `bound` is the state norm past which a run counts as diverged; the measurement then reports `bounded = false`
/// rather than a number, because an unstable loop has no average cost to compare.
#[allow(clippy::too_many_arguments)]
pub fn measure_regret(
    plant: Plant,
    expert: ExpertPolicy,
    cost: StepCost,
    error: &ActionError,
    x0: &DVector<f64>,
    inputs: usize,
    noise_sigma: f64,
    steps: usize,
    burn: usize,
    seed: u64,
) -> RegretMeasurement {
    let mut rng = Xorshift::new(seed);
    let n = x0.len();
    let (mut xe, mut xp) = (x0.clone(), x0.clone());
    let (mut ce, mut cp) = (0.0f64, 0.0f64);
    let mut counted = 0usize;
    let mut bounded = true;

    for t in 0..steps {
        // one shared noise draw, then one error draw: the expert never consumes the error stream, so the two
        // runs stay aligned on w even as the policy's own randomness advances
        let w = DVector::from_fn(n, |_, _| noise_sigma * rng.normal());
        let e = error.sample(&xp, inputs, &mut rng);

        let ue = expert(&xe);
        let up = expert(&xp) + &e;
        if t >= burn {
            ce += cost(&xe, &ue);
            cp += cost(&xp, &up);
            counted += 1;
        }
        xe = plant(&xe, &ue, &w);
        xp = plant(&xp, &up, &w);
        if !xe.iter().chain(xp.iter()).all(|v| v.is_finite()) || xp.norm() > 1e6 || xe.norm() > 1e6 {
            bounded = false;
            break;
        }
    }
    let d = counted.max(1) as f64;
    RegretMeasurement { expert_cost: ce / d, policy_cost: cp / d, regret: cp / d - ce / d, bounded: bounded && counted > 0, steps: counted }
}

/// Least-squares slope of `log y` against `log x` — the empirical exponent.
pub fn log_log_slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let (lx, ly): (Vec<f64>, Vec<f64>) = pts.iter().map(|(x, y)| (x.ln(), y.ln())).unzip();
    let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
    let num: f64 = lx.iter().zip(&ly).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LqLoop;

    /// A linear plant where the answer is known exactly, so the *instrument* can be checked before it is used on
    /// a nonlinear loop. Anything wrong with the coupling or the burn-in shows up here as a bias against the
    /// closed form.
    #[test]
    fn the_instrument_reproduces_the_closed_form_on_a_linear_loop() {
        let dt = 0.1;
        let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_row_slice(1, 1, &[0.1]);
        let k = crate::lqr_gain(&a, &b, &q, &r);
        let l = LqLoop { a: a.clone(), b: b.clone(), q: q.clone(), r: r.clone(), k: k.clone() };

        let eta = 0.3;
        let predicted = l.regret_variance(&DMatrix::from_row_slice(1, 1, &[eta * eta])).unwrap();
        let m = measure_regret(
            &|x, u, w| &a * x + &b * u + w,
            &|x| -(&k * x),
            &|x, u| (x.transpose() * &q * x)[0] + (u.transpose() * &r * u)[0],
            &ActionError::noise(eta),
            &DVector::zeros(2),
            1,
            0.05,
            2_000_000,
            20_000,
            0xC0FF_EE12_3456_789Du64,
        );
        let rel = (m.regret - predicted).abs() / predicted;
        eprintln!("linear check: closed form {predicted:.8}, measured {:.8} over {} steps ({:.2}%)", m.regret, m.steps, 100.0 * rel);
        assert!(m.bounded);
        assert!(rel < 0.06, "the instrument must reproduce the closed form: {predicted} vs {}", m.regret);
        eprintln!("   for scale: expert cost {:.5}, policy cost {:.5} - the regret is their difference", m.expert_cost, m.policy_cost);
    }

    /// The three error types are distinguished, and `eta_squared` is the second moment the bounds are stated in.
    #[test]
    fn the_error_types_are_kept_distinct() {
        assert!((ActionError::noise(0.3).eta_squared(2) - 2.0 * 0.09).abs() < 1e-12);
        assert!((ActionError::constant(DVector::from_row_slice(&[0.3, 0.4])).eta_squared(2) - 0.25).abs() < 1e-12);
        // a state-dependent error contributes nothing to eta^2, because its size is not a property of the policy
        assert_eq!(ActionError::state_dependent(DMatrix::identity(1, 2)).eta_squared(1), 0.0);
        // and they compose
        let both = ActionError { sigma: 0.3, bias: Some(DVector::from_row_slice(&[0.4])), gain: None };
        assert!((both.eta_squared(1) - (0.09 + 0.16)).abs() < 1e-12);
    }

    /// A diverging run is reported as unbounded rather than as a large regret — the distinction a state-dependent
    /// error past its margin makes necessary.
    #[test]
    fn a_diverged_run_is_reported_not_scored() {
        let a = DMatrix::from_row_slice(1, 1, &[1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let k = DMatrix::from_row_slice(1, 1, &[0.5]); // A_K = 0.5, stable
        let m = measure_regret(
            &|x, u, w| &a * x + &b * u + w,
            &|x| -(&k * x),
            &|x, u| x.norm_squared() + 0.1 * u.norm_squared(),
            // a state-proportional error large enough to destabilise: A_K + B*dK = 0.5 + 3 = 3.5
            &ActionError::state_dependent(DMatrix::from_row_slice(1, 1, &[3.0])),
            &DVector::from_row_slice(&[0.1]),
            1,
            0.01,
            10_000,
            100,
            7,
        );
        eprintln!("state-dependent error past the margin: bounded = {}", m.bounded);
        assert!(!m.bounded, "a destabilised loop must be reported as unbounded");
    }
}
