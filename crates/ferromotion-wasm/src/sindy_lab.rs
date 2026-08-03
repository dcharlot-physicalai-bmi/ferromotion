//! **SINDy lab** — the rig behind the "discover the equation" lesson. It runs real
//! [`ferromotion_learn::Sindy`] on data from the unforced Duffing oscillator and lets the reader turn the
//! sparsity knob `λ` and watch the discovered model change: too small and spurious library terms survive; too
//! large and real terms are killed; in the sweet spot SINDy recovers exactly `ẋ = y`, `ẏ = −x − 0.3x³ − 0.1y`
//! from a library of ten candidate terms. Equation discovery, not curve fitting — the output is a formula you
//! can read.

use ferromotion_learn::Sindy;
use wasm_bindgen::prelude::*;

fn duffing(x: f64, y: f64) -> (f64, f64) {
    (y, -x - 0.3 * x * x * x - 0.1 * y)
}

#[wasm_bindgen]
pub struct SindyLab {
    states: Vec<Vec<f64>>,
    derivs: Vec<Vec<f64>>,
    sindy: Sindy,
    lambda: f64,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn trajectory(x0: f64, y0: f64, n: usize, dt: f64, seed: &mut u64, noise: f64, states: &mut Vec<Vec<f64>>, derivs: &mut Vec<Vec<f64>>) {
    let (mut x, mut y) = (x0, y0);
    let nz = |s: &mut u64| ((splitmix64(s) as f64 / u64::MAX as f64) * 2.0 - 1.0) * noise;
    for _ in 0..n {
        let (dx, dy) = duffing(x, y);
        states.push(vec![x, y]);
        // measured derivatives carry noise, as they would if estimated from data
        derivs.push(vec![dx + nz(seed), dy + nz(seed)]);
        let (k1x, k1y) = duffing(x, y);
        let (k2x, k2y) = duffing(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
        let (k3x, k3y) = duffing(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
        let (k4x, k4y) = duffing(x + dt * k3x, y + dt * k3y);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    }
}

#[wasm_bindgen]
impl SindyLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> SindyLab {
        let (mut states, mut derivs) = (vec![], vec![]);
        let mut seed = 4242u64;
        // fewer samples + measurement noise → the sparsity knob genuinely matters
        trajectory(1.5, 0.0, 90, 0.06, &mut seed, 0.25, &mut states, &mut derivs);
        trajectory(0.3, 1.3, 90, 0.06, &mut seed, 0.25, &mut states, &mut derivs);
        let lambda = 0.02; // opens dense (many spurious terms) — the reader tunes λ up to find the true model
        let sindy = Sindy::fit(&states, &derivs, 3, lambda);
        SindyLab { states, derivs, sindy, lambda }
    }

    /// Set the sparsity threshold λ and re-fit.
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
        self.sindy = Sindy::fit(&self.states, &self.derivs, 3, lambda);
    }
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Library size and the term names (the candidate functions).
    pub fn n_terms(&self) -> usize {
        self.sindy.names.len()
    }
    pub fn term_name(&self, i: usize) -> String {
        self.sindy.names.get(i).cloned().unwrap_or_default()
    }
    /// Fitted coefficient of library term `i` in the equation for state dimension `dim` (0 = ẋ, 1 = ẏ).
    pub fn coeff(&self, i: usize, dim: usize) -> f64 {
        self.sindy.coeffs[(i, dim)]
    }

    /// The discovered equation for dimension `dim`, as text.
    pub fn equation(&self, dim: usize) -> String {
        self.sindy.equation(dim)
    }
    /// The true equation for dimension `dim`.
    pub fn true_equation(&self, dim: usize) -> String {
        match dim {
            0 => "ẋ = 1.000 y".to_string(),
            _ => "ẏ = -1.000 x + -0.100 y + -0.300 x^3".to_string(),
        }
    }

    pub fn n_active(&self) -> usize {
        self.sindy.n_active()
    }

    /// Whether the discovered model matches the true Duffing system (right terms, right coefficients).
    pub fn is_correct(&self) -> bool {
        let idx = |name: &str| self.sindy.names.iter().position(|n| n == name);
        let c = |name: &str, dim: usize| idx(name).map(|i| self.sindy.coeffs[(i, dim)]).unwrap_or(0.0);
        self.n_active() == 4
            && (c("y", 0) - 1.0).abs() < 0.05
            && (c("x", 1) + 1.0).abs() < 0.05
            && (c("y", 1) + 0.1).abs() < 0.05
            && (c("x^3", 1) + 0.3).abs() < 0.05
    }
}

impl Default for SindyLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sparsity_knob_finds_the_true_model_in_a_window() {
        let mut lab = SindyLab::new();
        // a small λ keeps many spurious terms (noisy data)
        lab.set_lambda(0.01);
        assert!(lab.n_active() > 4, "small λ should keep spurious terms, got {}", lab.n_active());
        // the sweet spot recovers exactly the Duffing system
        lab.set_lambda(0.08);
        assert!(lab.is_correct(), "λ=0.08 should recover Duffing: {} ; {}", lab.equation(0), lab.equation(1));
        // too-large λ over-sparsifies
        lab.set_lambda(0.3);
        assert!(lab.n_active() < 4, "large λ should drop real terms");
    }
}
