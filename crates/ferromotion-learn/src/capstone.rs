//! **Capstone — model-based control (learn in imagination).** The whole course in one loop: identify a
//! model of an unknown plant from data, tune a controller by differentiating through the *learned* model, and
//! deploy it on the *real* plant. This is the sample-efficient alternative to reinforcement learning on
//! hardware — all the tuning happens offline, in the model, by gradient descent; the real system is only ever
//! run by the finished controller.
//!
//! Three stages: (1) **system identification** — from `(x, v, u) → ẍ` samples of a hidden damped point mass,
//! least squares recovers the coefficients of `ẍ = a·x + b·v + c·u + d`; (2) **control in imagination** — a
//! PID is tuned by rolling the closed loop out through the *learned* model on the autodiff tape and
//! backpropagating the tracking cost; (3) **deployment** — the tuned controller runs on the *true* plant and
//! reaches the setpoint, having never been tuned against it. Verified: the identified coefficients match the
//! plant, and a controller trained entirely in the learned model settles the true plant on its setpoint.

use crate::autodiff::Tape;
use nalgebra::{DMatrix, DVector};

/// Learn a plant model from data, tune a PID inside it, and deploy on the true plant.
pub struct ModelBasedControl {
    // hidden true plant: mass·ẍ = u − damping·v − disturbance
    true_mass: f64,
    true_damping: f64,
    true_disturbance: f64,
    setpoint: f64,
    dt: f64,
    steps: usize,
    // learned model ẍ = a·x + b·v + c·u + d
    learned: [f64; 4],
    identified: bool,
    // controller
    gains: [f64; 3],
    m: [f64; 3],
    v: [f64; 3],
    t: u64,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl ModelBasedControl {
    /// A task on a hidden damped point mass, with the controller starting proportional-only.
    pub fn new(mass: f64, damping: f64, disturbance: f64, setpoint: f64) -> Self {
        ModelBasedControl {
            true_mass: mass,
            true_damping: damping,
            true_disturbance: disturbance,
            setpoint,
            dt: 0.05,
            steps: 90,
            learned: [0.0; 4],
            identified: false,
            gains: [1.0, 0.0, 0.0],
            m: [0.0; 3],
            v: [0.0; 3],
            t: 0,
        }
    }

    fn true_accel(&self, x: f64, v: f64, u: f64) -> f64 {
        let _ = x; // no spring term in the true plant
        (u - self.true_damping * v - self.true_disturbance) / self.true_mass
    }

    /// Stage 1 — identify `ẍ = a·x + b·v + c·u + d` from data by least squares.
    pub fn identify(&mut self) {
        let mut s = 20240101u64;
        let mut rnd = |lo: f64, hi: f64| lo + (splitmix64(&mut s) as f64 / u64::MAX as f64) * (hi - lo);
        let n = 200;
        let mut a = DMatrix::zeros(n, 4);
        let mut y = DVector::zeros(n);
        for i in 0..n {
            let (x, v, u) = (rnd(-2.0, 2.0), rnd(-2.0, 2.0), rnd(-5.0, 5.0));
            a[(i, 0)] = x;
            a[(i, 1)] = v;
            a[(i, 2)] = u;
            a[(i, 3)] = 1.0;
            y[i] = self.true_accel(x, v, u);
        }
        let sol = ferromotion_core::finite_svd(&a, true, true).and_then(|s| s.solve(&y, 1e-12).ok()).unwrap_or_else(|| DVector::zeros(4));
        self.learned = [sol[0], sol[1], sol[2], sol[3]];
        self.identified = true;
    }

    /// The learned model coefficients `[a, b, c, d]`.
    pub fn learned_coeffs(&self) -> [f64; 4] {
        self.learned
    }
    /// The true coefficients (for comparison): `a=0, b=−damping/m, c=1/m, d=−disturbance/m`.
    pub fn true_coeffs(&self) -> [f64; 4] {
        [0.0, -self.true_damping / self.true_mass, 1.0 / self.true_mass, -self.true_disturbance / self.true_mass]
    }
    pub fn gains(&self) -> [f64; 3] {
        self.gains
    }

    /// Roll out the closed loop in plain `f64`; `use_true` selects the true plant or the learned model.
    fn simulate(&self, use_true: bool) -> Vec<f64> {
        let (mut x, mut v, mut integ) = (0.0, 0.0, 0.0);
        let mut out = vec![x];
        for _ in 0..self.steps {
            let e = self.setpoint - x;
            integ += e * self.dt;
            let u = self.gains[0] * e + self.gains[1] * integ + self.gains[2] * (-v);
            let acc = if use_true {
                self.true_accel(x, v, u)
            } else {
                self.learned[0] * x + self.learned[1] * v + self.learned[2] * u + self.learned[3]
            };
            v += acc * self.dt;
            x += v * self.dt;
            out.push(x);
        }
        out
    }

    /// The step response on the true plant.
    pub fn deploy(&self) -> Vec<f64> {
        self.simulate(true)
    }
    /// The step response in the learned model (the "imagination").
    pub fn imagine(&self) -> Vec<f64> {
        self.simulate(false)
    }
    /// Steady-state error on the true plant.
    pub fn deploy_error(&self) -> f64 {
        (self.deploy().last().copied().unwrap_or(0.0) - self.setpoint).abs()
    }

    /// Stage 2 — one Adam step tuning the PID by differentiating through the LEARNED model. Returns the cost.
    pub fn train_step(&mut self, lr: f64) -> f64 {
        let tape = Tape::new();
        let kp = tape.var(self.gains[0]);
        let ki = tape.var(self.gains[1]);
        let kd = tape.var(self.gains[2]);
        let mut x = tape.constant(0.0);
        let mut v = tape.constant(0.0);
        let mut integ = tape.constant(0.0);
        let mut cost = tape.constant(0.0);
        let (sp, dt) = (self.setpoint, self.dt);
        let [a, b, c, d] = self.learned;
        for step in 0..self.steps {
            let e = tape.constant(sp) - x;
            integ = integ + e * dt;
            let u = kp * e + ki * integ + kd * (v * -1.0);
            // learned model dynamics ẍ = a x + b v + c u + d
            let acc = x * a + v * b + u * c + tape.constant(d);
            v = v + acc * dt;
            x = x + v * dt;
            let w = 0.5 + step as f64 / self.steps as f64;
            cost = cost + e * e * (w * dt) + u * u * (1e-4 * dt);
        }
        let g = cost.backward();
        let grad = [g.wrt(kp), g.wrt(ki), g.wrt(kd)];
        self.t += 1;
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8);
        let (bc1, bc2) = (1.0 - b1.powi(self.t as i32), 1.0 - b2.powi(self.t as i32));
        for (i, &gi) in grad.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            self.gains[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + eps);
        }
        cost.value()
    }

    /// Train the controller in the learned model for `epochs` steps.
    pub fn train(&mut self, epochs: usize, lr: f64) -> f64 {
        let mut c = f64::INFINITY;
        for _ in 0..epochs {
            c = self.train_step(lr);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_a_model_then_control_the_real_plant_through_it() {
        // The full arc: identify the plant, tune a PID inside the learned model, deploy on the true plant.
        let mut mbc = ModelBasedControl::new(1.0, 0.6, 2.0, 1.0);
        mbc.identify();
        // stage 1: identification is accurate
        let (lc, tc) = (mbc.learned_coeffs(), mbc.true_coeffs());
        for i in 0..4 {
            assert!((lc[i] - tc[i]).abs() < 1e-6, "coeff {i}: learned {} vs true {}", lc[i], tc[i]);
        }
        // stages 2+3: a controller tuned entirely in the learned model settles the TRUE plant on its setpoint
        let before = mbc.deploy_error();
        assert!(before > 0.3, "proportional-only should miss on the true plant: {before}");
        mbc.train(800, 0.05);
        let after = mbc.deploy_error();
        assert!(after < 0.06, "controller trained in imagination should settle the real plant: {after}");
        assert!(mbc.gains()[1] > 0.1, "it should discover integral action: Ki={}", mbc.gains()[1]);
    }
}
