//! **PINN lab** — the rig behind the "physics in the loss" lesson. It trains a real
//! [`ferromotion_learn::Pinn`] to solve the harmonic oscillator `u'' + ω² u = 0`, `u(0)=1`, `u'(0)=0` — with
//! **no solution data**. The only supervision is the equation's residual (the network's own second derivative,
//! computed exactly by the tape jet) driven to zero at collocation points, plus the initial conditions. The
//! reader presses *Train* and watches the network's curve rise out of nothing and lock onto `cos(ω t)` — a
//! solution learned purely from physics.

use ferromotion_learn::Pinn;
use wasm_bindgen::prelude::*;

const OMEGA: f64 = 2.0;
const TMAX: f64 = 2.0;

#[wasm_bindgen]
pub struct PinnLab {
    pinn: Pinn,
    epochs: u32,
    loss: f64,
}

#[wasm_bindgen]
impl PinnLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PinnLab {
        PinnLab { pinn: Pinn::new(14, OMEGA, TMAX, 28, 7), epochs: 0, loss: f64::INFINITY }
    }

    /// Train for `n` Adam epochs on the physics loss; returns the current loss.
    pub fn train(&mut self, n: u32) -> f64 {
        self.loss = self.pinn.train(n as usize, 6e-3);
        self.epochs += n;
        self.loss
    }

    /// Re-initialize the network (clears training).
    pub fn reset(&mut self) {
        self.pinn = Pinn::new(14, OMEGA, TMAX, 28, 7 + self.epochs as u64);
        self.epochs = 0;
        self.loss = f64::INFINITY;
    }

    /// The network's learned solution `u(t)`.
    pub fn predict(&self, t: f64) -> f64 {
        self.pinn.forward(t)
    }
    /// The closed-form truth `cos(ω t)`.
    pub fn exact(&self, t: f64) -> f64 {
        self.pinn.exact(t)
    }
    /// Max absolute error vs the closed form across the domain.
    pub fn max_error(&self) -> f64 {
        self.pinn.max_error()
    }
    pub fn t_max(&self) -> f64 {
        TMAX
    }
    pub fn epochs(&self) -> u32 {
        self.epochs
    }
    pub fn loss(&self) -> f64 {
        self.loss
    }
}

impl Default for PinnLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_the_pinn_reduces_the_error_toward_the_true_solution() {
        let mut lab = PinnLab::new();
        let e_start = lab.max_error();
        lab.train(3000);
        let e_end = lab.max_error();
        assert!(e_end < e_start, "training should reduce the error: {e_start} → {e_end}");
        assert!(e_end < 0.12, "PINN should approach cos(2t): max error {e_end}");
    }
}
