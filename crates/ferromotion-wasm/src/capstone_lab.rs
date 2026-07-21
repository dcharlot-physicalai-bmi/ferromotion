//! **Capstone lab** — the rig behind the final "learn in imagination" lesson. It drives a real
//! [`ferromotion_learn::ModelBasedControl`]: identify an unknown damped-point-mass plant from data, tune a PID
//! by differentiating through the *learned* model, then deploy it on the *true* plant. The reader sees all
//! three stages — the learned coefficients matching the truth, a controller tuned entirely in the model, and
//! the same controller settling the real plant on its setpoint, having never been tuned against it.

use ferromotion_learn::ModelBasedControl;
use wasm_bindgen::prelude::*;

const SETPOINT: f64 = 1.0;

#[wasm_bindgen]
pub struct CapstoneLab {
    mbc: ModelBasedControl,
    epochs: u32,
}

#[wasm_bindgen]
impl CapstoneLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CapstoneLab {
        let mut mbc = ModelBasedControl::new(1.0, 0.6, 2.0, SETPOINT);
        mbc.identify(); // stage 1 runs immediately
        CapstoneLab { mbc, epochs: 0 }
    }

    /// Stage 2 — train the controller inside the learned model for `n` steps.
    pub fn train(&mut self, n: u32) -> f64 {
        let c = self.mbc.train(n as usize, 0.05);
        self.epochs += n;
        c
    }
    pub fn reset(&mut self) {
        self.mbc = ModelBasedControl::new(1.0, 0.6, 2.0, SETPOINT);
        self.mbc.identify();
        self.epochs = 0;
    }

    pub fn epochs(&self) -> u32 {
        self.epochs
    }
    pub fn setpoint(&self) -> f64 {
        SETPOINT
    }
    pub fn learned_coeff(&self, i: usize) -> f64 {
        self.mbc.learned_coeffs()[i]
    }
    pub fn true_coeff(&self, i: usize) -> f64 {
        self.mbc.true_coeffs()[i]
    }
    /// Max abs error between learned and true model coefficients.
    pub fn id_error(&self) -> f64 {
        (0..4).map(|i| (self.learned_coeff(i) - self.true_coeff(i)).abs()).fold(0.0, f64::max)
    }
    pub fn kp(&self) -> f64 {
        self.mbc.gains()[0]
    }
    pub fn ki(&self) -> f64 {
        self.mbc.gains()[1]
    }
    pub fn kd(&self) -> f64 {
        self.mbc.gains()[2]
    }

    /// Stage 2 trajectory — the response in the learned model (the "imagination").
    pub fn imagine(&self) -> Vec<f64> {
        self.mbc.imagine()
    }
    /// Stage 3 trajectory — the response on the true plant (deployment).
    pub fn deploy(&self) -> Vec<f64> {
        self.mbc.deploy()
    }
    /// Steady-state error on the true plant.
    pub fn deploy_error(&self) -> f64 {
        self.mbc.deploy_error()
    }
}

impl Default for CapstoneLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_the_plant_and_controls_it_through_the_learned_model() {
        let mut lab = CapstoneLab::new();
        assert!(lab.id_error() < 1e-6, "stage 1: identification accurate, {}", lab.id_error());
        lab.train(800);
        assert!(lab.deploy_error() < 0.06, "stage 3: real plant settles, {}", lab.deploy_error());
        assert!(lab.ki() > 0.1, "discovered integral action: Ki={}", lab.ki());
    }
}
