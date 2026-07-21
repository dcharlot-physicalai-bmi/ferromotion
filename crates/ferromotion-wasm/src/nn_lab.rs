//! **Neural-network lab** — the rig behind the "universal approximation" lesson. It trains a real
//! [`ferromotion_learn::Mlp`] (through the reverse-mode autodiff tape, with Adam) to fit a target curve the
//! reader picks — a sine, a kink, or a step — so they can press *Train* and watch a plain black-box network
//! bend itself onto any continuous shape as the loss falls. No physics in it yet: that is the point, and the
//! baseline every later module improves on.

use ferromotion_learn::Mlp;
use wasm_bindgen::prelude::*;

fn target(kind: u32, x: f64) -> f64 {
    match kind {
        1 => x.abs() - 0.4,        // a kink (non-smooth)
        2 => 0.7 * x.signum(),     // a step (discontinuous)
        _ => (3.0 * x).sin(),      // a smooth wave
    }
}

#[wasm_bindgen]
pub struct NnLab {
    net: Mlp,
    kind: u32,
    xs: Vec<Vec<f64>>,
    ys: Vec<Vec<f64>>,
    mse: f64,
    epochs: u32,
    hidden: usize,
}

#[wasm_bindgen]
impl NnLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> NnLab {
        let hidden = 20;
        let mut lab = NnLab {
            net: Mlp::new(&[1, hidden, hidden, 1], 12),
            kind: 0,
            xs: vec![],
            ys: vec![],
            mse: f64::INFINITY,
            epochs: 0,
            hidden,
        };
        lab.rebuild();
        lab
    }

    fn rebuild(&mut self) {
        self.xs = (0..40).map(|i| vec![-1.0 + 2.0 * i as f64 / 39.0]).collect();
        self.ys = self.xs.iter().map(|x| vec![target(self.kind, x[0])]).collect();
    }

    /// Choose the target curve: 0 = wave sin(3x), 1 = kink |x|, 2 = step.
    pub fn set_target(&mut self, kind: u32) {
        self.kind = kind;
        self.rebuild();
        self.reset();
    }

    /// Re-initialize the network weights and clear training progress (keeps the current target).
    pub fn reset(&mut self) {
        self.net = Mlp::new(&[1, self.hidden, self.hidden, 1], 12 + self.epochs as u64);
        self.mse = f64::INFINITY;
        self.epochs = 0;
    }

    /// Train for `n` Adam epochs; returns the current MSE.
    pub fn train(&mut self, n: u32) -> f64 {
        self.mse = self.net.train(&self.xs, &self.ys, n as usize, 0.02);
        self.epochs += n;
        self.mse
    }

    /// The network's prediction at `x` (for drawing the learned curve).
    pub fn predict(&self, x: f64) -> f64 {
        self.net.forward(&[x])[0]
    }
    /// The target value at `x` (for drawing the ground-truth curve).
    pub fn target_at(&self, x: f64) -> f64 {
        target(self.kind, x)
    }

    pub fn n_train(&self) -> usize {
        self.xs.len()
    }
    pub fn train_x(&self, i: usize) -> f64 {
        self.xs[i][0]
    }
    pub fn train_y(&self, i: usize) -> f64 {
        self.ys[i][0]
    }
    pub fn mse(&self) -> f64 {
        self.mse
    }
    pub fn epochs(&self) -> u32 {
        self.epochs
    }
    pub fn n_params(&self) -> usize {
        self.net.n_params()
    }
}

impl Default for NnLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_reduces_the_loss_and_fits_the_wave() {
        let mut lab = NnLab::new();
        let start = lab.train(50);
        let end = lab.train(1200);
        assert!(end < start, "loss should fall: {start} → {end}");
        assert!(end < 5e-3, "should fit sin(3x): MSE {end}");
        assert!((lab.predict(0.3) - lab.target_at(0.3)).abs() < 0.1, "prediction tracks target");
    }

    #[test]
    fn switching_target_resets_progress() {
        let mut lab = NnLab::new();
        lab.train(100);
        lab.set_target(2);
        assert_eq!(lab.epochs(), 0, "changing target resets training");
        assert_eq!(lab.kind, 2);
    }
}
