//! **Linear viscoelasticity** — the materials whose stiffness depends on how long you have been pushing.
//!
//! [`plasticity`](crate::plasticity) covers permanent deformation past a yield point. This covers the other
//! kind of inelasticity, the one that is fully recoverable and yet still not elastic: an elastomer foot that
//! sinks over the second a robot stands on it, a tendon that lengthens over a stride, a harmonic-drive
//! flexspline whose stiffness at 1 Hz is not its stiffness at 100 Hz. Nothing in the stack could express any
//! of that, and the consequences are the ones that look like sensor drift:
//!
//! * **A calibration measured quickly is wrong slowly.** Under held load an elastomer keeps deforming
//!   ([`creep`](Prony::creep_compliance)); under held deformation its stress decays
//!   ([`relaxation`](Prony::relaxation_modulus)). A gripper calibrated in a snapshot loses grip force over the
//!   minutes it holds an object, and no amount of position feedback recovers it because the *material* moved.
//! * **Stiffness is a function of frequency, so a single number is a choice of frequency.** The
//!   [`storage modulus`](Prony::storage_modulus) rises from the equilibrium value to the instantaneous one
//!   across the relaxation spectrum. A model with one `E` is right at one frequency.
//! * **The phase lag is where the energy goes.** [`loss_modulus`](Prony::loss_modulus) sets the hysteresis loop
//!   area, and that is real dissipated heat: `π ε₀² E''(ω)` per cycle, per unit volume. For a leg that is the
//!   damping it gets for free, and for a tyre it is the rolling resistance.
//!
//! # The model
//!
//! A **generalized Maxwell** model, whose relaxation modulus is a Prony series:
//!
//! ```text
//!   E(t) = E_∞ + Σ_i E_i exp(−t/τ_i)
//! ```
//!
//! One term is the standard linear solid (Zener). Physically it is a spring `E_∞` in parallel with `n`
//! spring-dashpot branches, and `E_i`, `τ_i` are that branch's stiffness and time constant.
//!
//! Integration uses **internal variables**, one per branch, rather than storing strain history:
//!
//! ```text
//!   σ = E_∞ ε + Σ q_i,        q̇_i = −q_i/τ_i + E_i ε̇
//! ```
//!
//! which is what makes it usable in a simulation at all: the alternative is a hereditary integral over the
//! whole past, and its cost grows without bound.
//!
//! The update is **exact for strain that is linear across the step**, at any `dt`:
//!
//! ```text
//!   q_i ← q_i e^{−dt/τ_i} + E_i τ_i (1 − e^{−dt/τ_i}) Δε/dt
//! ```
//!
//! That is asserted at timesteps spanning four orders of magnitude, and it matters because it means a
//! viscoelastic material adds no integration error of its own to a ramp — unusual, and worth relying on.
//!
//! # Three dimensions
//!
//! This module is scalar (uniaxial). The standard extension applies the Prony series to the **deviatoric**
//! response and keeps the volumetric response elastic, on the grounds that shape change is what the polymer
//! network relaxes and volume change is resisted by the far stiffer intermolecular repulsion. That is a
//! modelling assumption rather than a theorem, and it is stated here rather than buried, because a material
//! with genuinely viscoelastic bulk behaviour (a closed-cell foam, whose gas phase does relax) needs a second
//! Prony series and this module would silently give the wrong answer.
//!
//! # What the tests pin
//!
//! Every claim above is checked against something independent of the code path making it: the recursion
//! against the analytic ramp solution; a step-strain relaxation against `E(t)` exactly; a time-domain sinusoid
//! against the closed-form complex modulus in both amplitude and phase; the measured hysteresis loop area
//! against `π ε₀² E''(ω)`; the creep integrator against the relaxation modulus through the Laplace identity
//! `s² Ê(s) Ĵ(s) = 1`; and Boltzmann superposition, which is the defining property of *linear* viscoelasticity.

/// A generalized Maxwell (Prony series) relaxation spectrum.
#[derive(Clone, Debug)]
pub struct Prony {
    /// Long-time (equilibrium, fully relaxed) modulus. Must be `> 0` for a solid; `0` gives a fluid, whose
    /// creep is unbounded.
    pub e_inf: f64,
    /// One `(E_i, τ_i)` per Maxwell branch: branch stiffness and relaxation time.
    pub branches: Vec<(f64, f64)>,
}

impl Prony {
    /// A **standard linear solid** (Zener): one branch.
    ///
    /// `e_inf` is the relaxed modulus, `e_1` the extra stiffness present instantaneously, `tau` the relaxation
    /// time. Its relaxation modulus is `E_∞ + E_1 e^{−t/τ}` in closed form, which is why it is the case the
    /// tests check the general machinery against.
    pub fn standard_linear_solid(e_inf: f64, e_1: f64, tau: f64) -> Prony {
        Prony { e_inf, branches: vec![(e_1, tau)] }
    }

    /// A spectrum from explicit branches. Returns `None` if any stiffness is negative or any time constant is
    /// not strictly positive.
    ///
    /// **Negative `E_i` is rejected rather than tolerated.** A branch with negative stiffness gives a negative
    /// loss modulus over some frequency band, and a material with `E'' < 0` *generates* energy over a cycle.
    /// Fitting a Prony series to noisy data by unconstrained least squares produces exactly this, and it looks
    /// like a good fit right up to the point where a simulation gains energy.
    pub fn new(e_inf: f64, branches: &[(f64, f64)]) -> Option<Prony> {
        if e_inf < 0.0 || !e_inf.is_finite() {
            return None;
        }
        if branches.iter().any(|(e, t)| !e.is_finite() || !t.is_finite() || *e < 0.0 || *t <= 0.0) {
            return None;
        }
        if e_inf == 0.0 && branches.is_empty() {
            return None;
        }
        Some(Prony { e_inf, branches: branches.to_vec() })
    }

    /// The relaxation modulus `E(t) = E_∞ + Σ E_i e^{−t/τ_i}`: stress per unit of a strain step held since
    /// `t = 0`.
    pub fn relaxation_modulus(&self, t: f64) -> f64 {
        self.e_inf + self.branches.iter().map(|(e, tau)| e * (-t / tau).exp()).sum::<f64>()
    }

    /// `E(0)`: the modulus a fast load sees, `E_∞ + Σ E_i`.
    pub fn instantaneous_modulus(&self) -> f64 {
        self.e_inf + self.branches.iter().map(|(e, _)| *e).sum::<f64>()
    }

    /// `E(∞)`: the modulus a held load eventually sees.
    pub fn equilibrium_modulus(&self) -> f64 {
        self.e_inf
    }

    /// **Storage modulus** `E'(ω) = E_∞ + Σ E_i (ωτ_i)²/(1 + (ωτ_i)²)`: the in-phase, energy-storing part.
    ///
    /// Rises monotonically from `E_∞` at `ω = 0` to `E(0)` as `ω → ∞`, which is the frequency-domain statement
    /// of "a fast load sees a stiffer material".
    pub fn storage_modulus(&self, omega: f64) -> f64 {
        self.e_inf
            + self
                .branches
                .iter()
                .map(|(e, tau)| {
                    let wt = omega * tau;
                    e * wt * wt / (1.0 + wt * wt)
                })
                .sum::<f64>()
    }

    /// **Loss modulus** `E''(ω) = Σ E_i (ωτ_i)/(1 + (ωτ_i)²)`: the out-of-phase, dissipating part.
    ///
    /// Non-negative for every `ω` whenever the branch stiffnesses are, which is the passivity the second law
    /// requires and which [`Prony::new`] enforces at construction. Each branch peaks at `ω = 1/τ_i`, so the
    /// spectrum of time constants is directly visible as the shape of this curve.
    pub fn loss_modulus(&self, omega: f64) -> f64 {
        self.branches
            .iter()
            .map(|(e, tau)| {
                let wt = omega * tau;
                e * wt / (1.0 + wt * wt)
            })
            .sum::<f64>()
    }

    /// `tan δ = E''/E'`: the loss tangent, i.e. the phase lag of stress behind strain.
    pub fn loss_tangent(&self, omega: f64) -> f64 {
        self.loss_modulus(omega) / self.storage_modulus(omega)
    }

    /// `|E*(ω)| = √(E'² + E''²)`: the ratio of stress amplitude to strain amplitude under sinusoidal loading.
    pub fn dynamic_modulus(&self, omega: f64) -> f64 {
        let (ep, epp) = (self.storage_modulus(omega), self.loss_modulus(omega));
        (ep * ep + epp * epp).sqrt()
    }

    /// Energy dissipated per unit volume per cycle of sinusoidal strain of amplitude `strain_amplitude`:
    /// `π ε₀² E''(ω)`.
    ///
    /// This is the hysteresis loop area, and it is checked against one measured from a time-domain simulation
    /// rather than assumed.
    pub fn dissipation_per_cycle(&self, omega: f64, strain_amplitude: f64) -> f64 {
        std::f64::consts::PI * strain_amplitude * strain_amplitude * self.loss_modulus(omega)
    }

    /// The **algorithmic tangent modulus** for a step of size `dt`:
    /// `E_∞ + Σ E_i τ_i (1 − e^{−dt/τ_i})/dt`.
    ///
    /// This is `∂σ/∂ε` as the update actually computes it, which is what an implicit FEM assembly needs in its
    /// stiffness matrix. Using the instantaneous or the equilibrium modulus there instead is a common mistake,
    /// and it costs convergence rate rather than correctness: the residual is still right, so it converges,
    /// just slowly and with no indication why.
    ///
    /// It interpolates between the two limits: `→ E(0)` as `dt → 0`, `→ E_∞` as `dt → ∞`.
    pub fn tangent_modulus(&self, dt: f64) -> f64 {
        self.e_inf
            + self
                .branches
                .iter()
                .map(|(e, tau)| {
                    if dt <= 0.0 {
                        *e
                    } else {
                        // `-expm1(-x)`, not `1 - exp(-x)`. For dt/tau = 1e-11 the naive form loses about
                        // five digits to cancellation, which put this 40 Pa off a 4 MPa modulus — and small
                        // dt is exactly the regime an explicit step uses, so the error is worst where the
                        // quantity matters most.
                        e * tau * -(-dt / tau).exp_m1() / dt
                    }
                })
                .sum::<f64>()
    }

    /// A fresh, unstressed state for this material.
    pub fn state(&self) -> ViscoState {
        ViscoState { strain: 0.0, q: vec![0.0; self.branches.len()] }
    }

    /// The **creep compliance** `J(t)` — strain per unit of a stress step held since `t = 0` — sampled at
    /// `n + 1` points `t = 0, dt, 2dt, …, n·dt`.
    ///
    /// The `t = 0` sample is included deliberately. An earlier version started at `t = dt` and the test then
    /// compared its first element against `J(0+)`, which it is not: by `t = dt` the material has already
    /// crept. Returning the instant is cheaper than documenting its absence.
    ///
    /// Computed by integrating the stress-controlled form, not from a closed form, so that the Laplace
    /// identity `s² Ê(s) Ĵ(s) = 1` is a genuine cross-check between two independent paths rather than algebra
    /// restated. Returns `None` for a material with no equilibrium stiffness, whose creep is unbounded.
    pub fn creep_compliance(&self, dt: f64, n: usize) -> Option<Vec<f64>> {
        if self.e_inf <= 0.0 || dt <= 0.0 {
            return None;
        }
        let mut st = self.state();
        // The instantaneous response to a unit stress step is the instantaneous compliance.
        st.strain = 1.0 / self.instantaneous_modulus();
        for (i, (e, _)) in self.branches.iter().enumerate() {
            st.q[i] = e * st.strain;
        }
        let mut out = Vec::with_capacity(n + 1);
        out.push(st.strain); // t = 0: the instantaneous compliance
        for _ in 0..n {
            st.step_stress(self, dt, 1.0);
            out.push(st.strain);
        }
        Some(out)
    }

    /// The Laplace transform of the relaxation modulus, in closed form:
    /// `Ê(s) = E_∞/s + Σ E_i τ_i/(1 + s τ_i)`.
    pub fn relaxation_laplace(&self, s: f64) -> f64 {
        self.e_inf / s + self.branches.iter().map(|(e, tau)| e * tau / (1.0 + s * tau)).sum::<f64>()
    }
}

/// The internal state of a viscoelastic material point: current strain and one variable per Maxwell branch.
#[derive(Clone, Debug)]
pub struct ViscoState {
    /// Current total strain.
    pub strain: f64,
    /// Branch stresses `q_i`. The total stress is `E_∞ ε + Σ q_i`.
    pub q: Vec<f64>,
}

impl ViscoState {
    /// Current stress, `E_∞ ε + Σ q_i`.
    pub fn stress(&self, m: &Prony) -> f64 {
        m.e_inf * self.strain + self.q.iter().sum::<f64>()
    }

    /// Impose a strain **step** of `delta`, instantaneously.
    ///
    /// A step is not a limit of ramps here: it is its own case, because every branch responds elastically to
    /// an instantaneous change. Applying a step through [`step_strain`](ViscoState::step_strain) with a small
    /// `dt` approaches this but never reaches it, so relaxation tests use this to start exactly on `E(t)`.
    pub fn apply_strain_step(&mut self, m: &Prony, delta: f64) {
        self.strain += delta;
        for (i, (e, _)) in m.branches.iter().enumerate() {
            self.q[i] += e * delta;
        }
    }

    /// Advance by `dt` with the strain moving linearly to `new_strain`. Returns the stress after the step.
    ///
    /// **Exact** for strain linear across the step, at any `dt`: the branch ODE `q̇ = −q/τ + E ε̇` has the
    /// closed-form solution `q e^{−dt/τ} + E ε̇ τ (1 − e^{−dt/τ})` for constant `ε̇`, and that is what is
    /// evaluated. No stability bound, and no accuracy penalty for a large step during a ramp.
    pub fn step_strain(&mut self, m: &Prony, dt: f64, new_strain: f64) -> f64 {
        if dt <= 0.0 {
            self.apply_strain_step(m, new_strain - self.strain);
            return self.stress(m);
        }
        let rate = (new_strain - self.strain) / dt;
        for (i, (e, tau)) in m.branches.iter().enumerate() {
            let x = -dt / tau;
            // `one_minus_a` via expm1 rather than `1 - exp(x)`: see `Prony::tangent_modulus`.
            let a = x.exp();
            let one_minus_a = -x.exp_m1();
            self.q[i] = self.q[i] * a + e * rate * tau * one_minus_a;
        }
        self.strain = new_strain;
        self.stress(m)
    }

    /// Advance by `dt` holding the **stress** at `target_stress`, solving for the strain. Returns the strain.
    ///
    /// The implicit solve is closed-form, not iterative, because the update is linear in the new strain:
    ///
    /// ```text
    ///   σ = E_∞ ε' + Σ (a_i q_i + b_i (ε' − ε))      with a_i = e^{−dt/τ_i},  b_i = E_i τ_i (1 − a_i)/dt
    ///   ⇒ ε' = (σ + B ε − Σ a_i q_i) / (E_∞ + B),     B = Σ b_i
    /// ```
    ///
    /// Note `E_∞ + B` is exactly [`Prony::tangent_modulus`], which is the algorithmic tangent appearing where
    /// it should.
    pub fn step_stress(&mut self, m: &Prony, dt: f64, target_stress: f64) -> f64 {
        if dt <= 0.0 {
            return self.strain;
        }
        let mut decayed = 0.0;
        let mut b_sum = 0.0;
        for (i, (e, tau)) in m.branches.iter().enumerate() {
            let x = -dt / tau;
            decayed += x.exp() * self.q[i];
            b_sum += e * tau * -x.exp_m1() / dt;
        }
        let denom = m.e_inf + b_sum;
        let new_strain = (target_stress + b_sum * self.strain - decayed) / denom;
        self.step_strain(m, dt, new_strain);
        self.strain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn material() -> Prony {
        // Two decades apart, so the spectrum is genuinely broad and a single-branch bug would show.
        Prony::new(1.0e6, &[(2.0e6, 0.1), (1.0e6, 2.0)]).expect("valid spectrum")
    }

    #[test]
    fn the_standard_linear_solid_matches_its_closed_form() {
        let (e_inf, e_1, tau) = (1.5e6, 3.0e6, 0.25);
        let m = Prony::standard_linear_solid(e_inf, e_1, tau);
        assert_eq!(m.instantaneous_modulus(), e_inf + e_1);
        assert_eq!(m.equilibrium_modulus(), e_inf);
        for &t in &[0.0, 0.01, 0.25, 1.0, 5.0] {
            let expect = e_inf + e_1 * (-t / tau).exp();
            assert!((m.relaxation_modulus(t) - expect).abs() < 1e-9 * expect);
        }
        // At one time constant the branch has decayed to exactly 1/e of its initial value.
        let at_tau = m.relaxation_modulus(tau);
        assert!(((at_tau - e_inf) / e_1 - (-1.0f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn a_negative_branch_stiffness_is_rejected_because_it_would_generate_energy() {
        // An unconstrained least-squares fit to noisy data produces these, and the failure mode is a
        // simulation that gains energy rather than an obviously bad fit.
        assert!(Prony::new(1e6, &[(-1e6, 0.1)]).is_none(), "negative stiffness must be rejected");
        assert!(Prony::new(1e6, &[(1e6, 0.0)]).is_none(), "a zero time constant must be rejected");
        assert!(Prony::new(1e6, &[(1e6, -0.1)]).is_none(), "a negative time constant must be rejected");
        assert!(Prony::new(-1.0, &[]).is_none(), "negative equilibrium modulus must be rejected");
        assert!(Prony::new(0.0, &[]).is_none(), "a material with no stiffness at all must be rejected");
        // A fluid (no equilibrium stiffness but real branches) is legal, and its creep is unbounded, which is
        // reported by `creep_compliance` rather than by a wrong number.
        let fluid = Prony::new(0.0, &[(1e6, 0.5)]).expect("a Maxwell fluid is a valid material");
        assert!(fluid.creep_compliance(0.01, 10).is_none(), "unbounded creep must be reported, not computed");
    }

    #[test]
    fn the_loss_modulus_is_never_negative_and_peaks_at_one_over_tau() {
        // Passivity, and the reason the spectrum is readable off the loss curve.
        let m = material();
        for k in 0..2000 {
            let omega = 10f64.powf(-4.0 + 8.0 * k as f64 / 2000.0);
            assert!(m.loss_modulus(omega) >= 0.0, "E'' must never be negative, at omega {omega}");
            assert!(m.storage_modulus(omega) >= m.e_inf - 1e-9, "E' must never fall below E_inf");
        }
        // A single branch peaks at omega = 1/tau, with value exactly E_1/2.
        let sls = Prony::standard_linear_solid(1e6, 4e6, 0.3);
        let peak_omega = 1.0 / 0.3;
        let peak = sls.loss_modulus(peak_omega);
        assert!((peak - 4e6 / 2.0).abs() < 1e-6, "the peak should be E_1/2, got {peak}");
        for f in [0.2, 0.5, 2.0, 5.0] {
            assert!(sls.loss_modulus(peak_omega * f) < peak, "and it must BE a peak, failed at {f}x");
        }
    }

    #[test]
    fn the_storage_modulus_spans_exactly_the_two_static_limits() {
        let m = material();
        // Low frequency approaches the equilibrium modulus, high frequency the instantaneous one. Relative
        // to the modulus scale: an absolute bound on a quantity measured in MPa is not a tolerance, it is a
        // number that happens to be small compared to nothing in particular.
        let scale = m.instantaneous_modulus();
        assert!((m.storage_modulus(1e-9) - m.equilibrium_modulus()).abs() < 1e-12 * scale);
        assert!((m.storage_modulus(1e12) - m.instantaneous_modulus()).abs() < 1e-12 * scale);
        // Monotone in between, which is what makes "stiffer when loaded faster" a theorem here rather than a
        // property of the particular numbers.
        let mut prev = m.storage_modulus(1e-6);
        for k in 1..=3000 {
            let omega = 10f64.powf(-6.0 + 12.0 * k as f64 / 3000.0);
            let now = m.storage_modulus(omega);
            assert!(now >= prev - 1e-6, "E' must be non-decreasing in omega, at {omega}");
            prev = now;
        }
        // And the loss modulus vanishes at both ends: no dissipation from a static or an infinitely fast load.
        // Again relative: at omega = 1e-9 the loss modulus is 6e-3 Pa, which an absolute 1e-3 bound called a
        // failure and which is in fact 1.5e-9 of the modulus scale, i.e. zero for every purpose.
        assert!(m.loss_modulus(1e-9) < 1e-8 * scale, "got {}", m.loss_modulus(1e-9));
        assert!(m.loss_modulus(1e12) < 1e-8 * scale, "got {}", m.loss_modulus(1e12));
        // Each falls off as 1/omega and omega respectively, so check the SCALING rather than only a bound:
        // ten times lower frequency gives ten times less loss.
        let a = m.loss_modulus(1e-6);
        let b = m.loss_modulus(1e-7);
        assert!((a / b - 10.0).abs() < 1e-3, "E'' should be linear in omega at low frequency, ratio {}", a / b);
    }

    #[test]
    fn a_held_strain_step_relaxes_along_the_relaxation_modulus_exactly() {
        // The defining property, and the reason `apply_strain_step` exists as its own case.
        let m = material();
        let eps = 0.003;
        for &dt in &[1e-4, 1e-3, 1e-2, 0.1] {
            let mut st = m.state();
            st.apply_strain_step(&m, eps);
            assert!((st.stress(&m) - m.instantaneous_modulus() * eps).abs() < 1e-9 * st.stress(&m).abs());
            let mut t = 0.0;
            let mut worst = 0.0f64;
            while t < 10.0 {
                let sigma = st.step_strain(&m, dt, eps); // hold
                t += dt;
                let expect = m.relaxation_modulus(t) * eps;
                worst = worst.max((sigma - expect).abs() / expect.abs());
            }
            // Exact regardless of dt: holding is a linear ramp of zero slope, which the update solves exactly.
            assert!(worst < 1e-12, "dt {dt}: relaxation should be exact, worst relative error {worst:.3e}");
        }
    }

    #[test]
    fn the_ramp_response_is_exact_at_every_timestep() {
        // The strong claim in the module docs: no integration error for linear strain, at ANY dt. Checked
        // against the analytic ramp solution, and across four orders of magnitude of dt.
        let m = material();
        let rate = 0.01; // strain per second
        let t_end = 3.0;
        // Analytic: for eps = rate*t from rest, q_i(t) = E_i rate tau_i (1 - e^{-t/tau_i}).
        let analytic = |t: f64| -> f64 {
            m.e_inf * rate * t
                + m.branches
                    .iter()
                    .map(|(e, tau)| e * rate * tau * (1.0 - (-t / tau).exp()))
                    .sum::<f64>()
        };
        for &n in &[3usize, 30, 300, 3000, 30_000] {
            let dt = t_end / n as f64;
            let mut st = m.state();
            for k in 1..=n {
                st.step_strain(&m, dt, rate * dt * k as f64);
            }
            let got = st.stress(&m);
            let want = analytic(t_end);
            assert!(
                (got - want).abs() < 1e-11 * want.abs(),
                "n={n} (dt={dt:.2e}): got {got:.6e} want {want:.6e}"
            );
        }
    }

    #[test]
    fn a_time_domain_sinusoid_reproduces_the_complex_modulus_in_amplitude_and_phase() {
        // Two independent code paths: the incremental integrator and the closed-form E'(w), E''(w). If the
        // recursion's coefficients were wrong, this is where it would show, and a relaxation-only test would
        // not catch a phase error at all.
        let m = material();
        let eps0 = 0.002;
        for &f in &[0.05f64, 0.5, 5.0, 50.0] {
            let omega = 2.0 * PI * f;
            let period = 1.0 / f;
            let n_per = 2000;
            let dt = period / n_per as f64;
            let mut st = m.state();
            // Run several periods to reach steady state, discarding the transient.
            for k in 1..=(n_per * 12) {
                st.step_strain(&m, dt, eps0 * (omega * dt * k as f64).sin());
            }
            // Extract the fundamental by correlating one full period against sin and cos.
            let (mut c_sin, mut c_cos) = (0.0f64, 0.0f64);
            let t0 = dt * (n_per * 12) as f64;
            for k in 1..=n_per {
                let t = t0 + dt * k as f64;
                let sigma = st.step_strain(&m, dt, eps0 * (omega * t).sin());
                c_sin += sigma * (omega * t).sin();
                c_cos += sigma * (omega * t).cos();
            }
            // sigma = eps0 (E' sin(wt) + E'' cos(wt)), so the correlations give E' and E'' directly.
            let e_prime = 2.0 * c_sin / (n_per as f64 * eps0);
            let e_dprime = 2.0 * c_cos / (n_per as f64 * eps0);
            let want_p = m.storage_modulus(omega);
            let want_dp = m.loss_modulus(omega);
            assert!(
                (e_prime - want_p).abs() < 2e-3 * want_p,
                "f={f}: E' measured {e_prime:.4e} vs closed form {want_p:.4e}"
            );
            assert!(
                (e_dprime - want_dp).abs() < 2e-3 * want_dp,
                "f={f}: E'' measured {e_dprime:.4e} vs closed form {want_dp:.4e}"
            );
            // And the phase lag, which is the sign-sensitive part: stress must LEAD strain for a solid.
            let measured_delta = e_dprime.atan2(e_prime);
            let want_delta = m.loss_tangent(omega).atan();
            assert!((measured_delta - want_delta).abs() < 1e-3, "f={f}: phase {measured_delta} vs {want_delta}");
            assert!(measured_delta > 0.0, "stress must lead strain, got {measured_delta}");
        }
    }

    #[test]
    fn the_hysteresis_loop_area_is_the_dissipated_energy() {
        // pi eps0^2 E''(w) per cycle, measured by integrating sigma d(eps) around the loop. This is the claim
        // that makes viscoelastic damping quantitative rather than qualitative.
        let m = material();
        let eps0 = 0.002;
        for &f in &[0.1f64, 1.0, 10.0] {
            let omega = 2.0 * PI * f;
            let period = 1.0 / f;
            let n_per = 4000;
            let dt = period / n_per as f64;
            let mut st = m.state();
            for k in 1..=(n_per * 12) {
                st.step_strain(&m, dt, eps0 * (omega * dt * k as f64).sin());
            }
            let mut area = 0.0;
            let mut prev_eps = st.strain;
            let mut prev_sigma = st.stress(&m);
            let t0 = dt * (n_per * 12) as f64;
            for k in 1..=n_per {
                let t = t0 + dt * k as f64;
                let eps = eps0 * (omega * t).sin();
                let sigma = st.step_strain(&m, dt, eps);
                // A genuine trapezoid. The first version of this line was `sigma * (eps - prev_eps)`, a
                // right-endpoint rectangle rule, under a comment that said trapezoid; it left a 0.97% bias
                // that I nearly absorbed into the tolerance instead of fixing.
                area += 0.5 * (sigma + prev_sigma) * (eps - prev_eps);
                prev_eps = eps;
                prev_sigma = sigma;
            }
            let want = m.dissipation_per_cycle(omega, eps0);
            assert!(
                (area - want).abs() < 1e-3 * want,
                "f={f}: loop area {area:.6e} vs pi eps0^2 E'' = {want:.6e}"
            );
            assert!(area > 0.0, "the loop must dissipate, not generate");
        }
    }

    #[test]
    fn creep_and_relaxation_satisfy_the_laplace_identity() {
        // s^2 E-hat(s) J-hat(s) = 1. This ties the stress-controlled integrator to the relaxation modulus's
        // closed form, and it is the check that neither one can pass alone: a creep function that is wrong in
        // a way that still looks like creep fails here.
        let m = material();
        let dt = 1e-4;
        let t_end = 200.0; // many times the longest time constant
        let n = (t_end / dt) as usize;
        // n + 1 samples at t = 0, dt, ..., n*dt.
        let j = m.creep_compliance(dt, n).expect("a solid has bounded creep");
        assert_eq!(j.len(), n + 1);

        // Sanity on the endpoints first, or a bad J could pass the transform by accident.
        let j_0 = 1.0 / m.instantaneous_modulus();
        assert!(
            (j[0] - j_0).abs() < 1e-12 * j_0,
            "J(0) must be the instantaneous compliance: {} vs {j_0}",
            j[0]
        );
        let j_inf = 1.0 / m.equilibrium_modulus();
        assert!(
            (j[n] - j_inf).abs() < 1e-6 * j_inf,
            "J(inf) must be the equilibrium compliance: {} vs {j_inf}",
            j[n]
        );
        // Creep is monotone increasing for a passive material.
        for k in 1..=n {
            assert!(j[k] >= j[k - 1] - 1e-18, "creep must be monotone, broke at step {k}");
        }

        for &s in &[0.05f64, 0.2, 1.0, 5.0, 20.0] {
            // Numerical Laplace transform of J by the trapezoid rule, with the constant tail done exactly.
            let mut integral = 0.0;
            for k in 0..n {
                let t = dt * k as f64;
                let w0 = (-s * t).exp();
                let w1 = (-s * (t + dt)).exp();
                integral += 0.5 * dt * (j[k] * w0 + j[k + 1] * w1);
            }
            // Tail: J is flat at j_inf beyond t_end.
            integral += j_inf * (-s * t_end).exp() / s;

            let e_hat = m.relaxation_laplace(s);
            let product = s * s * e_hat * integral;
            assert!(
                (product - 1.0).abs() < 2e-3,
                "s={s}: s^2 E-hat J-hat should be 1, got {product:.6}"
            );
        }
    }

    #[test]
    fn boltzmann_superposition_holds() {
        // The defining property of LINEAR viscoelasticity: the response to a sum of histories is the sum of
        // the responses. A model that is subtly nonlinear (a rate-dependent coefficient, say) fails here
        // while passing every single-input test above.
        let m = material();
        let dt = 1e-3;
        let n = 4000;
        let h1 = |k: usize| 0.002 * (dt * k as f64 * 3.0).sin();
        let h2 = |k: usize| 0.001 * (dt * k as f64).min(1.0); // a ramp that saturates

        let run = |h: &dyn Fn(usize) -> f64| -> f64 {
            let mut st = m.state();
            for k in 1..=n {
                st.step_strain(&m, dt, h(k));
            }
            st.stress(&m)
        };
        let s1 = run(&h1);
        let s2 = run(&h2);
        let s_both = run(&|k| h1(k) + h2(k));
        assert!(
            (s_both - (s1 + s2)).abs() < 1e-9 * (s1.abs() + s2.abs()),
            "superposition: combined {s_both:.6e} vs sum {:.6e}",
            s1 + s2
        );
        // Scaling too, which is the other half of linearity.
        let s_scaled = run(&|k| 3.7 * h1(k));
        assert!((s_scaled - 3.7 * s1).abs() < 1e-9 * s_scaled.abs(), "homogeneity must hold");
        // And the two inputs are genuinely different, so the test is not comparing a thing to itself.
        assert!((s1 - s2).abs() > 1e-6 * s1.abs().max(s2.abs()), "the two histories must differ");
    }

    #[test]
    fn the_algorithmic_tangent_interpolates_the_two_static_limits() {
        // Using the instantaneous or equilibrium modulus in an implicit assembly costs convergence rate, not
        // correctness, so nothing fails loudly. Pinning the interpolation is the only way it stays right.
        let m = material();
        // These are tight on purpose: they are what caught `1 - exp(-x)` losing five digits to cancellation
        // at small dt. With `expm1` the limits are reached to near machine precision.
        let scale = m.instantaneous_modulus();
        assert!(
            (m.tangent_modulus(1e-12) - m.instantaneous_modulus()).abs() < 1e-10 * scale,
            "dt -> 0 must give the instantaneous modulus, off by {}",
            (m.tangent_modulus(1e-12) - m.instantaneous_modulus()).abs()
        );
        assert!((m.tangent_modulus(1e9) - m.equilibrium_modulus()).abs() < 1e-8 * scale);
        // Monotone decreasing in dt, and always between the limits.
        let mut prev = m.tangent_modulus(1e-9);
        for k in 1..=500 {
            let dt = 10f64.powf(-9.0 + 12.0 * k as f64 / 500.0);
            let now = m.tangent_modulus(dt);
            assert!(now <= prev + 1e-6, "the tangent must not increase with dt, at {dt}");
            assert!(now >= m.equilibrium_modulus() - 1e-6 && now <= m.instantaneous_modulus() + 1e-6);
            prev = now;
        }

        // And it IS the derivative the update realises: perturb the strain increment and difference it.
        let dt = 0.05;
        let base = {
            let mut st = m.state();
            st.step_strain(&m, dt, 0.001)
        };
        let up = {
            let mut st = m.state();
            st.step_strain(&m, dt, 0.001 + 1e-9)
        };
        let numeric = (up - base) / 1e-9;
        assert!(
            (numeric - m.tangent_modulus(dt)).abs() < 1e-3 * m.tangent_modulus(dt),
            "tangent {} vs differenced {numeric}",
            m.tangent_modulus(dt)
        );
    }

    #[test]
    fn a_gripper_loses_force_over_a_hold_and_the_amount_is_predictable() {
        // The engineering consequence, stated as a number. This is the failure that reads as sensor drift: the
        // position is unchanged, the commanded force is unchanged, and the grip is quietly going away.
        let m = material();
        let eps = 0.01; // 1% strain in the pad
        let mut st = m.state();
        st.apply_strain_step(&m, eps);
        let initial = st.stress(&m);

        let dt = 1e-3;
        let mut t = 0.0;
        let mut after_1s = None;
        while t < 30.0 {
            st.step_strain(&m, dt, eps);
            t += dt;
            if after_1s.is_none() && t >= 1.0 {
                after_1s = Some(st.stress(&m));
            }
        }
        let one = after_1s.expect("one second elapsed");
        let thirty = st.stress(&m);

        // The retained fractions follow E(t)/E(0) exactly, so they are predictable rather than surprising.
        assert!((one / initial - m.relaxation_modulus(1.0) / m.instantaneous_modulus()).abs() < 1e-9);
        // For this spectrum: instantaneous 4 MPa, equilibrium 1 MPa, so three quarters of the grip is
        // transient. After 30 s (15 long time constants) essentially all of it is gone.
        assert!(one < 0.55 * initial, "a second of holding should cost most of it, retained {}", one / initial);
        assert!(
            (thirty / initial - 0.25).abs() < 0.01,
            "and it settles at E_inf/E_0 = 0.25, got {}",
            thirty / initial
        );
    }
}
