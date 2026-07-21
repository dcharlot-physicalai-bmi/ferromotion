//! **HNN lab** — the rig behind the "conserve energy by construction" lesson. It stages the classic showdown
//! on the ideal mass–spring: a [`ferromotion_learn::Hnn`] (learns a scalar Hamiltonian, derives the dynamics
//! as its symplectic gradient) against a baseline [`ferromotion_learn::Mlp`] (predicts `(q̇, ṗ)` directly),
//! both trained on the *same* `(q,p)→(q̇,ṗ)` samples. Rolled out from the same start, the HNN traces a clean
//! closed orbit and holds its energy flat, while the black-box baseline spirals — energy leaking or growing —
//! because nothing constrains its vector field to be conservative. Structure in the architecture buys a
//! conservation law the data never explicitly taught.

use ferromotion_learn::{Hnn, Mlp};
use wasm_bindgen::prelude::*;

const HIST: usize = 400;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[wasm_bindgen]
pub struct HnnLab {
    hnn: Hnn,
    base: Mlp,
    // training data: mass-spring H = ½(q²+p²), field (q̇,ṗ) = (p,−q)
    q: Vec<f64>,
    p: Vec<f64>,
    qd: Vec<f64>,
    pd: Vec<f64>,
    xs: Vec<Vec<f64>>,
    ys: Vec<Vec<f64>>,
    epochs: u32,
    // rollout states
    qh: f64,
    ph: f64,
    qb: f64,
    pb: f64,
    e0: f64,
    hist_h: Vec<f64>,
    hist_b: Vec<f64>,
}

fn energy(q: f64, p: f64) -> f64 {
    0.5 * (q * q + p * p)
}

#[wasm_bindgen]
impl HnnLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> HnnLab {
        // Sparse + slightly noisy samples: this is where structure pays off. With few, noisy examples the
        // black-box baseline learns a wrong vector field off-data and spirals, while the HNN's conservative-
        // by-construction field still holds energy.
        let mut s = 12345u64;
        let noise = |st: &mut u64| ((splitmix64(st) as f64 / u64::MAX as f64) * 2.0 - 1.0) * 0.12;
        let (mut q, mut p, mut qd, mut pd) = (vec![], vec![], vec![], vec![]);
        let (mut xs, mut ys) = (vec![], vec![]);
        for _ in 0..28 {
            let qi = (splitmix64(&mut s) as f64 / u64::MAX as f64) * 4.0 - 2.0;
            let pi = (splitmix64(&mut s) as f64 / u64::MAX as f64) * 4.0 - 2.0;
            let (tqd, tpd) = (pi + noise(&mut s), -qi + noise(&mut s));
            q.push(qi);
            p.push(pi);
            qd.push(tqd);
            pd.push(tpd);
            xs.push(vec![qi, pi]);
            ys.push(vec![tqd, tpd]);
        }
        let start = (1.6, 0.0);
        HnnLab {
            hnn: Hnn::new(16, 3),
            base: Mlp::new(&[2, 16, 16, 2], 3),
            q,
            p,
            qd,
            pd,
            xs,
            ys,
            epochs: 0,
            qh: start.0,
            ph: start.1,
            qb: start.0,
            pb: start.1,
            e0: energy(start.0, start.1),
            hist_h: vec![energy(start.0, start.1)],
            hist_b: vec![energy(start.0, start.1)],
        }
    }

    /// Train BOTH models for `n` epochs on the same data.
    pub fn train(&mut self, n: u32) {
        self.hnn.train(&self.q, &self.p, &self.qd, &self.pd, n as usize, 5e-3);
        self.base.train(&self.xs, &self.ys, n as usize, 5e-3);
        self.epochs += n;
    }

    pub fn epochs(&self) -> u32 {
        self.epochs
    }

    /// Reset both rollouts to the same start; keeps the trained weights.
    pub fn reset_rollout(&mut self) {
        let (q0, p0) = (1.6, 0.0);
        self.qh = q0;
        self.ph = p0;
        self.qb = q0;
        self.pb = p0;
        self.hist_h = vec![self.e0];
        self.hist_b = vec![self.e0];
    }

    /// Advance both rollouts one RK4 step; record each model's true energy.
    pub fn step(&mut self, dt: f64) {
        let (nqh, nph) = self.hnn.step_rk4(self.qh, self.ph, dt);
        self.qh = nqh;
        self.ph = nph;
        // baseline: RK4 on the directly-predicted field
        let f = |q: f64, p: f64| {
            let o = self.base.forward(&[q, p]);
            (o[0], o[1])
        };
        let (k1q, k1p) = f(self.qb, self.pb);
        let (k2q, k2p) = f(self.qb + 0.5 * dt * k1q, self.pb + 0.5 * dt * k1p);
        let (k3q, k3p) = f(self.qb + 0.5 * dt * k2q, self.pb + 0.5 * dt * k2p);
        let (k4q, k4p) = f(self.qb + dt * k3q, self.pb + dt * k3p);
        self.qb += dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
        self.pb += dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);

        push_capped(&mut self.hist_h, energy(self.qh, self.ph));
        push_capped(&mut self.hist_b, energy(self.qb, self.pb));
    }

    pub fn qh(&self) -> f64 { self.qh }
    pub fn ph(&self) -> f64 { self.ph }
    pub fn qb(&self) -> f64 { self.qb }
    pub fn pb(&self) -> f64 { self.pb }
    pub fn e0(&self) -> f64 { self.e0 }
    pub fn energy_hnn(&self) -> f64 { energy(self.qh, self.ph) }
    pub fn energy_base(&self) -> f64 { energy(self.qb, self.pb) }
    pub fn n_hist(&self) -> usize { self.hist_h.len() }
    pub fn hist_hnn(&self, i: usize) -> f64 { self.hist_h.get(i).copied().unwrap_or(self.e0) }
    pub fn hist_base(&self, i: usize) -> f64 { self.hist_b.get(i).copied().unwrap_or(self.e0) }
}

fn push_capped(v: &mut Vec<f64>, x: f64) {
    v.push(x);
    if v.len() > HIST {
        v.remove(0);
    }
}

impl Default for HnnLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_hnn_conserves_energy_better_than_the_baseline_on_rollout() {
        let mut lab = HnnLab::new();
        lab.train(2000);
        lab.reset_rollout();
        let e0 = lab.e0();
        let (mut dev_h, mut dev_b): (f64, f64) = (0.0, 0.0);
        for _ in 0..1500 {
            lab.step(0.02);
            dev_h = dev_h.max((lab.energy_hnn() - e0).abs());
            dev_b = dev_b.max((lab.energy_base() - e0).abs());
        }
        assert!(dev_h < 0.15, "HNN rollout should hold energy: {dev_h}");
        assert!(dev_b > dev_h, "baseline should drift more than the HNN: base {dev_b} vs hnn {dev_h}");
    }
}
