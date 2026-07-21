//! **Neural ODE lab** — the rig behind the "learn the vector field, predict the future" lesson. A
//! [`ferromotion_learn::NeuralOde`] is trained on the *first part* of one trajectory of a damped oscillator
//! (by backprop through the RK4 solver), then rolled forward. Because it learned the continuous vector field
//! rather than memorizing points, its rollout keeps tracking the true trajectory *past the training window* —
//! it predicts the future. The learned field is drawn as arrows so the reader sees the flow the trajectory
//! rides on.

use ferromotion_learn::NeuralOde;
use wasm_bindgen::prelude::*;

const DT: f64 = 0.12;
const N_FULL: usize = 44;
const N_TRAIN: usize = 26;

fn true_field(x: f64, y: f64) -> (f64, f64) {
    (y, -x - 0.15 * y)
}

#[wasm_bindgen]
pub struct NeuralOdeLab {
    node: NeuralOde,
    obs: Vec<[f64; 2]>,
    pred: Vec<[f64; 2]>,
    epochs: u32,
}

#[wasm_bindgen]
impl NeuralOdeLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> NeuralOdeLab {
        // generate the true trajectory (RK4 on the true field)
        let (mut x, mut y) = (1.7, 0.0);
        let mut obs = vec![[x, y]];
        for _ in 0..N_FULL {
            let f = |x: f64, y: f64| true_field(x, y);
            let (k1x, k1y) = f(x, y);
            let (k2x, k2y) = f(x + 0.5 * DT * k1x, y + 0.5 * DT * k1y);
            let (k3x, k3y) = f(x + 0.5 * DT * k2x, y + 0.5 * DT * k2y);
            let (k4x, k4y) = f(x + DT * k3x, y + DT * k3y);
            x += DT / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += DT / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
            obs.push([x, y]);
        }
        let node = NeuralOde::new(16, 4);
        let pred = node.rollout(&obs[0], N_FULL, DT);
        NeuralOdeLab { node, obs, pred, epochs: 0 }
    }

    /// Train on the first `N_TRAIN` steps for `n` epochs, then refresh the full rollout prediction.
    pub fn train(&mut self, n: u32) -> f64 {
        let loss = self.node.train(&self.obs[..N_TRAIN], DT, n as usize, 5e-3);
        self.pred = self.node.rollout(&self.obs[0], N_FULL, DT);
        self.epochs += n;
        loss
    }

    pub fn epochs(&self) -> u32 {
        self.epochs
    }
    pub fn n_train(&self) -> usize {
        N_TRAIN
    }

    pub fn n_obs(&self) -> usize {
        self.obs.len()
    }
    pub fn obs_x(&self, i: usize) -> f64 {
        self.obs[i][0]
    }
    pub fn obs_y(&self, i: usize) -> f64 {
        self.obs[i][1]
    }

    pub fn n_pred(&self) -> usize {
        self.pred.len()
    }
    pub fn pred_x(&self, i: usize) -> f64 {
        self.pred[i][0]
    }
    pub fn pred_y(&self, i: usize) -> f64 {
        self.pred[i][1]
    }

    /// Learned field components at a point (for drawing the flow arrows).
    pub fn fx(&self, x: f64, y: f64) -> f64 {
        self.node.field(&[x, y])[0]
    }
    pub fn fy(&self, x: f64, y: f64) -> f64 {
        self.node.field(&[x, y])[1]
    }

    /// RMSE of the rollout against the true trajectory beyond the training window (the extrapolation metric).
    pub fn extrapolation_rmse(&self) -> f64 {
        let mut e = 0.0;
        let mut n = 0;
        for i in N_TRAIN..self.obs.len() {
            e += (self.pred[i][0] - self.obs[i][0]).powi(2) + (self.pred[i][1] - self.obs[i][1]).powi(2);
            n += 1;
        }
        (e / n as f64).sqrt()
    }
}

impl Default for NeuralOdeLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_lets_the_rollout_predict_beyond_the_window() {
        let mut lab = NeuralOdeLab::new();
        let before = lab.extrapolation_rmse();
        lab.train(1500);
        let after = lab.extrapolation_rmse();
        assert!(after < before, "training should improve extrapolation: {before} → {after}");
        assert!(after < 0.3, "NODE should predict past the training window: rmse {after}");
    }
}
