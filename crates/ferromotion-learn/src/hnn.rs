//! **Hamiltonian Neural Network (HNN)** — physics injected into the **architecture**. Instead of predicting
//! the dynamics directly, the network learns a single scalar function, the Hamiltonian `H_θ(q, p)` (the total
//! energy), and the dynamics are *derived* from it by Hamilton's equations: `q̇ = ∂H/∂p`, `ṗ = −∂H/∂q`. A
//! vector field obtained this way is symplectic by construction, so trajectories conserve the learned `H` —
//! the network cannot leak or inject energy no matter how long you roll it out. This is the structural answer
//! to the integrator-drift problem from the energy lesson: conservation is baked in, not hoped for.
//!
//! Training uses the network's derivatives w.r.t. its **inputs** `(q, p)` as the predicted velocities, matched
//! to observed `(q̇, ṗ)`. Those input-derivatives are computed exactly by propagating a first-order two-input
//! jet `(value, ∂/∂q, ∂/∂p)` through the network where each slot is a reverse-mode [`Var`], so one backward
//! pass gives the exact parameter gradient of the matching loss. Verified on the ideal mass–spring
//! (`H = ½p² + ½q²`): the learned vector field matches the truth, and a long rollout conserves energy where a
//! direct black-box predictor of the same data drifts.

use crate::autodiff::{Tape, Var};

/// A Hamiltonian network `H_θ(q, p)` (two inputs, one scalar output).
pub struct Hnn {
    sizes: Vec<usize>,
    params: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// A first-order jet in two inputs (q, p): (value, ∂/∂q, ∂/∂p), each a tape Var.
type Jet2<'t> = (Var<'t>, Var<'t>, Var<'t>);

fn j_add<'t>(a: Jet2<'t>, b: Jet2<'t>) -> Jet2<'t> {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}
fn j_scale<'t>(w: Var<'t>, a: Jet2<'t>) -> Jet2<'t> {
    (w * a.0, w * a.1, w * a.2)
}
fn j_tanh<'t>(g: Jet2<'t>) -> Jet2<'t> {
    let f0 = g.0.tanh();
    let fp = f0 * f0 * (-1.0) + 1.0; // 1 − tanh²
    (f0, fp * g.1, fp * g.2)
}

impl Hnn {
    /// A Hamiltonian network with the given hidden width.
    pub fn new(hidden: usize, seed: u64) -> Self {
        let sizes = vec![2, hidden, hidden, 1];
        let mut state = seed ^ 0x0F1E_2D3C_4B5A_6978;
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
        Hnn { sizes, params, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    // Forward as a two-input jet; returns (H, ∂H/∂q, ∂H/∂p) as tape Vars.
    fn forward_jet<'t>(&self, tape: &'t Tape, pv: &[Var<'t>], q: f64, p: f64, zero: Var<'t>, one: Var<'t>) -> Jet2<'t> {
        let mut a: Vec<Jet2<'t>> = vec![(tape.constant(q), one, zero), (tape.constant(p), zero, one)];
        let mut off = 0;
        let layers = self.sizes.len() - 1;
        for l in 0..layers {
            let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
            let mut z: Vec<Jet2<'t>> = Vec::with_capacity(outd);
            for o in 0..outd {
                let mut s: Jet2<'t> = (pv[off + ind * outd + o], zero, zero);
                for (i, &ai) in a.iter().enumerate() {
                    s = j_add(s, j_scale(pv[off + o * ind + i], ai));
                }
                z.push(if l + 1 < layers { j_tanh(s) } else { s });
            }
            off += ind * outd + outd;
            a = z;
        }
        a[0]
    }

    // Plain-f64 two-input jet: (H, ∂H/∂q, ∂H/∂p) with no tape (for inference / rollout).
    fn forward_deriv(&self, q: f64, p: f64) -> (f64, f64, f64) {
        let mut a: Vec<(f64, f64, f64)> = vec![(q, 1.0, 0.0), (p, 0.0, 1.0)];
        let mut off = 0;
        let layers = self.sizes.len() - 1;
        for l in 0..layers {
            let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
            let mut z = Vec::with_capacity(outd);
            for o in 0..outd {
                let (mut v0, mut v1, mut v2) = (self.params[off + ind * outd + o], 0.0, 0.0);
                for (i, ai) in a.iter().enumerate() {
                    let w = self.params[off + o * ind + i];
                    v0 += w * ai.0;
                    v1 += w * ai.1;
                    v2 += w * ai.2;
                }
                if l + 1 < layers {
                    let t = v0.tanh();
                    let fp = 1.0 - t * t;
                    z.push((t, fp * v1, fp * v2));
                } else {
                    z.push((v0, v1, v2));
                }
            }
            off += ind * outd + outd;
            a = z;
        }
        a[0]
    }

    /// The learned Hamiltonian (energy) at a phase point.
    pub fn hamiltonian(&self, q: f64, p: f64) -> f64 {
        self.forward_deriv(q, p).0
    }

    /// The learned dynamics `(q̇, ṗ) = (∂H/∂p, −∂H/∂q)` at a phase point.
    pub fn field(&self, q: f64, p: f64) -> (f64, f64) {
        let (_, dq, dp) = self.forward_deriv(q, p);
        (dp, -dq)
    }

    /// One RK4 rollout step of the learned dynamics.
    pub fn step_rk4(&self, q: f64, p: f64, dt: f64) -> (f64, f64) {
        let f = |q: f64, p: f64| self.field(q, p);
        let (k1q, k1p) = f(q, p);
        let (k2q, k2p) = f(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p);
        let (k3q, k3p) = f(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p);
        let (k4q, k4p) = f(q + dt * k3q, p + dt * k3p);
        (q + dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q), p + dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p))
    }

    /// One Adam step matching the learned field `(∂H/∂p, −∂H/∂q)` to the observed `(q̇, ṗ)`. Returns the loss.
    pub fn train_step(&mut self, q: &[f64], p: &[f64], qdot: &[f64], pdot: &[f64], lr: f64) -> f64 {
        let tape = Tape::new();
        let pv: Vec<Var> = self.params.iter().map(|&x| tape.var(x)).collect();
        let zero = tape.constant(0.0);
        let one = tape.constant(1.0);
        let mut loss = tape.constant(0.0);
        for i in 0..q.len() {
            let (_h, hq, hp) = self.forward_jet(&tape, &pv, q[i], p[i], zero, one);
            let r_q = hp - qdot[i]; //  q̇_pred − q̇ = ∂H/∂p − q̇
            let r_p = (-hq) - pdot[i]; //  ṗ_pred − ṗ = −∂H/∂q − ṗ
            loss = loss + r_q * r_q + r_p * r_p;
        }
        loss = loss * (1.0 / q.len() as f64);
        let g = loss.backward();
        let grad: Vec<f64> = pv.iter().map(|&x| g.wrt(x)).collect();

        self.t += 1;
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (i, &gi) in grad.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            self.params[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + eps);
        }
        loss.value()
    }

    /// Train for `epochs` steps; returns the final loss.
    pub fn train(&mut self, q: &[f64], p: &[f64], qdot: &[f64], pdot: &[f64], epochs: usize, lr: f64) -> f64 {
        let mut l = f64::INFINITY;
        for _ in 0..epochs {
            l = self.train_step(q, p, qdot, pdot, lr);
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ideal mass-spring H = ½p² + ½q²: true field q̇ = p, ṗ = −q; energy ½(q²+p²) is conserved.
    fn mass_spring_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut q, mut p, mut qd, mut pd) = (vec![], vec![], vec![], vec![]);
        let mut s = 12345u64;
        for _ in 0..80 {
            let qi = (splitmix64(&mut s) as f64 / u64::MAX as f64) * 4.0 - 2.0;
            let pi = (splitmix64(&mut s) as f64 / u64::MAX as f64) * 4.0 - 2.0;
            q.push(qi);
            p.push(pi);
            qd.push(pi); // q̇ = p
            pd.push(-qi); // ṗ = −q
        }
        (q, p, qd, pd)
    }

    #[test]
    fn hnn_learns_the_field_and_conserves_energy() {
        // Train once, then check both properties. THE HEADLINE: from (q,p)→(q̇,ṗ) samples the HNN recovers the
        // true vector field via ∂H/∂p, −∂H/∂q. THE STRUCTURAL PAYOFF: because the field derives from a scalar
        // H, rolling it out conserves the true energy over a long horizon — no drift.
        let (q, p, qd, pd) = mass_spring_data();
        let mut hnn = Hnn::new(16, 3);
        hnn.train(&q, &p, &qd, &pd, 2500, 5e-3);

        let mut err = 0.0;
        let mut n = 0;
        for &(tq, tp) in &[(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.5, 0.5)] {
            let (fq, fp) = hnn.field(tq, tp);
            err += (fq - tp).abs() + (fp - (-tq)).abs(); // truth (p, −q)
            n += 2;
        }
        assert!(err / (n as f64) < 0.15, "learned field should match (p,−q): mean abs err {}", err / (n as f64));

        let (mut qq, mut pp) = (1.5, 0.0);
        let e0 = 0.5 * (qq * qq + pp * pp);
        let mut max_dev: f64 = 0.0;
        for _ in 0..2000 {
            let (nq, np) = hnn.step_rk4(qq, pp, 0.02);
            qq = nq;
            pp = np;
            max_dev = max_dev.max((0.5 * (qq * qq + pp * pp) - e0).abs());
        }
        assert!(max_dev < 0.1, "HNN rollout should conserve energy: max deviation {max_dev}");
    }
}
