//! **MSNN lab** — the rig behind the "interpretable by construction" lesson. It fits a real
//! [`ferromotion_learn::Msnn`] (a partition-of-unity blend of local linear models) to a nonlinear target and
//! exposes the pieces: the blended fit, the individual local lines, and each model's learned slope — which
//! matches the target's true local derivative. Turn up the number of local models and watch the fit tighten
//! while every slope stays a number you can read. Structure buys interpretability a black-box MLP cannot give.

use ferromotion_learn::Msnn;
use wasm_bindgen::prelude::*;

const LO: f64 = -1.5;
const HI: f64 = 1.5;

fn target(x: f64) -> f64 {
    (2.0 * x).sin()
}
fn target_deriv(x: f64) -> f64 {
    2.0 * (2.0 * x).cos()
}

#[wasm_bindgen]
pub struct MsnnLab {
    msnn: Msnn,
    xs: Vec<f64>,
    ys: Vec<f64>,
    centers: usize,
}

#[wasm_bindgen]
impl MsnnLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MsnnLab {
        let xs: Vec<f64> = (0..120).map(|i| LO + (HI - LO) * i as f64 / 119.0).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| target(x)).collect();
        let centers = 7;
        let msnn = Msnn::fit(&xs, &ys, centers, LO, HI);
        MsnnLab { msnn, xs, ys, centers }
    }

    /// Set the number of local models and re-fit (by least squares, instant).
    pub fn set_centers(&mut self, k: usize) {
        self.centers = k.max(2);
        self.msnn = Msnn::fit(&self.xs, &self.ys, self.centers, LO, HI);
    }
    pub fn n_centers(&self) -> usize {
        self.centers
    }

    pub fn lo(&self) -> f64 { LO }
    pub fn hi(&self) -> f64 { HI }
    pub fn predict(&self, x: f64) -> f64 {
        self.msnn.predict(x)
    }
    pub fn target(&self, x: f64) -> f64 {
        target(x)
    }
    pub fn center(&self, i: usize) -> f64 {
        self.msnn.center(i)
    }
    pub fn local_slope(&self, i: usize) -> f64 {
        self.msnn.local_slope(i)
    }
    pub fn true_slope(&self, i: usize) -> f64 {
        target_deriv(self.msnn.center(i))
    }
    pub fn local_line(&self, i: usize, x: f64) -> f64 {
        self.msnn.local_line(i, x)
    }

    /// Fit error over the domain.
    pub fn mse(&self) -> f64 {
        self.xs.iter().zip(&self.ys).map(|(&x, &y)| (self.msnn.predict(x) - y).powi(2)).sum::<f64>() / self.xs.len() as f64
    }
    /// Mean absolute error between the learned local slopes and the true derivative at each center.
    pub fn slope_error(&self) -> f64 {
        let n = self.centers;
        (0..n).map(|i| (self.local_slope(i) - self.true_slope(i)).abs()).sum::<f64>() / n as f64
    }
    /// Total number of parameters (2 per local model) — for the data-efficiency point.
    pub fn n_params(&self) -> usize {
        2 * self.centers
    }
}

impl Default for MsnnLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_local_models_tighten_the_fit_and_slopes_stay_readable() {
        let mut lab = MsnnLab::new();
        lab.set_centers(4);
        let coarse = lab.mse();
        lab.set_centers(12);
        let fine = lab.mse();
        assert!(fine < coarse, "more local models should fit better: {coarse} → {fine}");
        assert!(fine < 1e-3, "should fit sin(2x): mse {fine}");
        assert!(lab.slope_error() < 0.3, "local slopes should match the true derivative: {}", lab.slope_error());
    }
}
