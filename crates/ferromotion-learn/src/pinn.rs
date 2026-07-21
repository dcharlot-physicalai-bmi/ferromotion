//! **Physics-informed neural network (PINN)** — the first way to inject physics: in the **loss**. Instead of
//! fitting data, a PINN fits the *governing equation*. A network `u_θ(t)` is trained so that its own
//! derivatives satisfy a differential equation — the loss is the equation's residual driven to zero at a cloud
//! of collocation points, plus the initial/boundary conditions. No solution data is needed; the physics is the
//! supervision (Raissi et al.).
//!
//! The interesting part is that the loss contains derivatives of the network w.r.t. its **input** (`u'(t)`,
//! `u''(t)`), and we still need the gradient of that loss w.r.t. the **parameters**. This module composes the
//! two exactly: it propagates a second-order **jet** `(u, u', u'')` through the network where every component
//! is a reverse-mode [`Var`] on the tape. The jet gives the exact input-derivatives (forward-mode chain rule),
//! and one `backward` pass then gives the exact parameter gradient of the physics loss (reverse mode) — no
//! finite differences anywhere. This is the "use the network's gradient as an output" idea that defines PINNs.
//!
//! Verified against the closed form: trained to solve the harmonic oscillator `u'' + ω²u = 0`, `u(0)=1`,
//! `u'(0)=0`, the network reproduces `cos(ωt)` across the domain.

use crate::autodiff::{Tape, Var};

/// A trained PINN for the harmonic oscillator `u'' + ω² u = 0` on `[0, t_max]` with `u(0)=1, u'(0)=0`.
pub struct Pinn {
    sizes: Vec<usize>,
    params: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    omega: f64,
    colloc: Vec<f64>,
    ic_weight: f64,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// A second-order jet in the input t: (value, d/dt, d²/dt²), each a tape Var.
type Jet<'t> = (Var<'t>, Var<'t>, Var<'t>);

fn jet_add<'t>(a: Jet<'t>, b: Jet<'t>) -> Jet<'t> {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}
// scale a jet by a parameter Var w that does not depend on t (dt = dtt = 0)
fn jet_scale<'t>(w: Var<'t>, a: Jet<'t>) -> Jet<'t> {
    (w * a.0, w * a.1, w * a.2)
}
// tanh of a jet: f(g(t)) with f=tanh, f'=1−tanh², f''=−2·tanh·(1−tanh²)
fn jet_tanh<'t>(g: Jet<'t>) -> Jet<'t> {
    let f0 = g.0.tanh();
    let fp = f0 * f0 * (-1.0) + 1.0; // 1 − tanh²
    let fpp = (f0 * fp) * (-2.0); // −2·tanh·(1−tanh²)
    (f0, fp * g.1, fpp * (g.1 * g.1) + fp * g.2)
}

impl Pinn {
    /// A PINN with the given hidden width, angular frequency `omega`, domain `[0, t_max]`, and number of
    /// interior collocation points.
    pub fn new(hidden: usize, omega: f64, t_max: f64, n_colloc: usize, seed: u64) -> Self {
        let sizes = vec![1, hidden, hidden, 1];
        let mut state = seed ^ 0x1234_5678_9ABC_DEF0;
        let mut params = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (ind, outd) = (sizes[l], sizes[l + 1]);
            let r = (6.0 / (ind + outd) as f64).sqrt();
            for _ in 0..ind * outd {
                let u = (splitmix64(&mut state) as f64 / u64::MAX as f64) * 2.0 - 1.0;
                params.push(u * r);
            }
            params.extend(std::iter::repeat_n(0.0, outd));
        }
        let n = params.len();
        let colloc: Vec<f64> = (0..n_colloc).map(|i| t_max * (i as f64 + 0.5) / n_colloc as f64).collect();
        Pinn { sizes, params, m: vec![0.0; n], v: vec![0.0; n], t: 0, omega, colloc, ic_weight: 12.0 }
    }

    /// The closed-form solution `cos(ω t)` (for verification / plotting the truth).
    pub fn exact(&self, t: f64) -> f64 {
        (self.omega * t).cos()
    }

    /// Plain-`f64` inference: the network's `u(t)`.
    pub fn forward(&self, t: f64) -> f64 {
        let mut a = vec![t];
        let mut off = 0;
        let layers = self.sizes.len() - 1;
        for l in 0..layers {
            let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
            let mut z = vec![0.0; outd];
            for (o, zo) in z.iter_mut().enumerate() {
                let mut s = self.params[off + ind * outd + o];
                for (i, &ai) in a.iter().enumerate() {
                    s += self.params[off + o * ind + i] * ai;
                }
                *zo = if l + 1 < layers { s.tanh() } else { s };
            }
            off += ind * outd + outd;
            a = z;
        }
        a[0]
    }

    // Forward the network as a second-order jet in t; returns (u, u', u'') as tape Vars.
    fn forward_jet<'t>(&self, tape: &'t Tape, pv: &[Var<'t>], t: f64, zero: Var<'t>, one: Var<'t>) -> Jet<'t> {
        let mut a: Vec<Jet<'t>> = vec![(tape.constant(t), one, zero)];
        let mut off = 0;
        let layers = self.sizes.len() - 1;
        for l in 0..layers {
            let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
            let mut z: Vec<Jet<'t>> = Vec::with_capacity(outd);
            for o in 0..outd {
                let mut s: Jet<'t> = (pv[off + ind * outd + o], zero, zero); // bias (const in t)
                for (i, &ai) in a.iter().enumerate() {
                    s = jet_add(s, jet_scale(pv[off + o * ind + i], ai));
                }
                z.push(if l + 1 < layers { jet_tanh(s) } else { s });
            }
            off += ind * outd + outd;
            a = z;
        }
        a[0]
    }

    fn loss_and_grad(&self) -> (f64, Vec<f64>) {
        let tape = Tape::new();
        let pv: Vec<Var> = self.params.iter().map(|&p| tape.var(p)).collect();
        let zero = tape.constant(0.0);
        let one = tape.constant(1.0);
        let w2 = self.omega * self.omega;

        // physics residual: u'' + ω² u = 0 at each collocation point
        let mut loss = tape.constant(0.0);
        for &t in &self.colloc {
            let (u, _ut, utt) = self.forward_jet(&tape, &pv, t, zero, one);
            let r = utt + u * w2;
            loss = loss + r * r;
        }
        loss = loss * (1.0 / self.colloc.len() as f64);

        // initial conditions: u(0) = 1, u'(0) = 0
        let (u0, ut0, _) = self.forward_jet(&tape, &pv, 0.0, zero, one);
        let e1 = u0 - 1.0;
        loss = loss + (e1 * e1 + ut0 * ut0) * self.ic_weight;

        let g = loss.backward();
        (loss.value(), pv.iter().map(|&p| g.wrt(p)).collect())
    }

    /// One Adam step; returns the loss before the step.
    pub fn train_step(&mut self, lr: f64) -> f64 {
        let (loss, grad) = self.loss_and_grad();
        self.t += 1;
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (i, &gi) in grad.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            self.params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + eps);
        }
        loss
    }

    /// Train for `epochs` steps; returns the final loss.
    pub fn train(&mut self, epochs: usize, lr: f64) -> f64 {
        let mut l = f64::INFINITY;
        for _ in 0..epochs {
            l = self.train_step(lr);
        }
        l
    }

    /// Max absolute error against the closed form on a fine grid — the verification metric.
    pub fn max_error(&self) -> f64 {
        let n = 100;
        (0..=n)
            .map(|i| {
                let t = self.colloc.last().copied().unwrap_or(1.0) * i as f64 / n as f64;
                (self.forward(t) - self.exact(t)).abs()
            })
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pinn_solves_the_harmonic_oscillator_from_physics_alone() {
        // THE HEADLINE. With NO solution data — only the ODE residual and the initial conditions in the loss —
        // the network reconstructs cos(ω t). ω=2 on [0,2] (past the first turning point).
        let mut pinn = Pinn::new(16, 2.0, 2.0, 30, 7);
        pinn.train(3000, 6e-3);
        let err = pinn.max_error();
        assert!(err < 0.08, "PINN should match cos(2t) from physics alone: max error {err}");
    }

    #[test]
    fn an_untrained_pinn_does_not_satisfy_the_equation() {
        // Control: before training, the network is nowhere near the solution.
        let pinn = Pinn::new(24, 2.0, 2.0, 48, 7);
        assert!(pinn.max_error() > 0.1, "an untrained net should not solve the ODE");
    }
}
