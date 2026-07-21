//! **Neural ODE (+ the integrator choice)** — learn *continuous-time* dynamics. Instead of learning a
//! discrete "next state from this state" map, a Neural ODE learns the **derivative** `ẋ = f_θ(x)` with a
//! network and recovers trajectories by handing `f_θ` to an ODE solver. Training matches an integrated
//! trajectory to observations by **backpropagating through the solver itself** — here the discrete adjoint:
//! the whole rollout (each RK4 step, each field evaluation) is recorded on the reverse-mode tape, so one
//! backward pass gives the exact gradient of the trajectory-matching loss w.r.t. the field's parameters.
//! Because it learns the vector field rather than memorizing points, it can be integrated forward to
//! *predict the future* past the training window.
//!
//! The integrator is part of the model, which is the door to **Variational Integrator Networks**: for a
//! conservative system a structure-preserving (symplectic) integrator in the loop keeps a rollout stable and
//! energy-bounded where a naive one drifts — the same lesson as the energy module, now inside the learner.
//! Verified: from a single trajectory of a damped oscillator, the Neural ODE learns the field and its rollout
//! tracks the true trajectory, including a short extrapolation beyond the training window.

use crate::autodiff::{Tape, Var};

/// A Neural ODE for a 2-D system: an MLP field `f_θ : ℝ² → ℝ²`, trained by trajectory matching.
pub struct NeuralOde {
    sizes: Vec<usize>, // [2, H, H, 2]
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

impl NeuralOde {
    /// A Neural ODE with the given hidden width.
    pub fn new(hidden: usize, seed: u64) -> Self {
        let sizes = vec![2, hidden, hidden, 2];
        let mut state = seed ^ 0x7777_1111_3333_9999;
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
        NeuralOde { sizes, params, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    // The field f_θ(x) evaluated in Var arithmetic (x carried as Vars through the rollout).
    fn field_var<'t>(&self, pv: &[Var<'t>], x: [Var<'t>; 2]) -> [Var<'t>; 2] {
        let mut a = vec![x[0], x[1]];
        let mut off = 0;
        let layers = self.sizes.len() - 1;
        for l in 0..layers {
            let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
            let mut z = Vec::with_capacity(outd);
            for o in 0..outd {
                let mut s = pv[off + ind * outd + o];
                for (i, &ai) in a.iter().enumerate() {
                    s = s + pv[off + o * ind + i] * ai;
                }
                z.push(if l + 1 < layers { s.tanh() } else { s });
            }
            off += ind * outd + outd;
            a = z;
        }
        [a[0], a[1]]
    }

    // One RK4 step in Var arithmetic.
    fn rk4_var<'t>(&self, pv: &[Var<'t>], x: [Var<'t>; 2], dt: f64) -> [Var<'t>; 2] {
        let add = |a: [Var<'t>; 2], b: [Var<'t>; 2], s: f64| [a[0] + b[0] * s, a[1] + b[1] * s];
        let k1 = self.field_var(pv, x);
        let k2 = self.field_var(pv, add(x, k1, 0.5 * dt));
        let k3 = self.field_var(pv, add(x, k2, 0.5 * dt));
        let k4 = self.field_var(pv, add(x, k3, dt));
        [
            x[0] + (k1[0] + k2[0] * 2.0 + k3[0] * 2.0 + k4[0]) * (dt / 6.0),
            x[1] + (k1[1] + k2[1] * 2.0 + k3[1] * 2.0 + k4[1]) * (dt / 6.0),
        ]
    }

    /// The learned field `f_θ(x)` in plain `f64` (for inspection).
    pub fn field(&self, x: &[f64]) -> [f64; 2] {
        let mut a = vec![x[0], x[1]];
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
        [a[0], a[1]]
    }

    /// Roll out the learned dynamics from `x0` for `n` RK4 steps of size `dt` (plain `f64`).
    pub fn rollout(&self, x0: &[f64], n: usize, dt: f64) -> Vec<[f64; 2]> {
        let mut x = [x0[0], x0[1]];
        let mut out = vec![x];
        for _ in 0..n {
            let f = |x: [f64; 2]| self.field(&x);
            let add = |a: [f64; 2], b: [f64; 2], s: f64| [a[0] + b[0] * s, a[1] + b[1] * s];
            let k1 = f(x);
            let k2 = f(add(x, k1, 0.5 * dt));
            let k3 = f(add(x, k2, 0.5 * dt));
            let k4 = f(add(x, k3, dt));
            x = [
                x[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                x[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ];
            out.push(x);
        }
        out
    }

    /// One Adam step of trajectory matching: integrate `f_θ` from `obs[0]` and match the observed states,
    /// backpropagating through the whole rollout. Returns the loss.
    pub fn train_step(&mut self, obs: &[[f64; 2]], dt: f64, lr: f64) -> f64 {
        let tape = Tape::new();
        let pv: Vec<Var> = self.params.iter().map(|&x| tape.var(x)).collect();
        let mut x = [tape.constant(obs[0][0]), tape.constant(obs[0][1])];
        let mut loss = tape.constant(0.0);
        for o in obs.iter().skip(1) {
            x = self.rk4_var(&pv, x, dt);
            let e0 = x[0] - o[0];
            let e1 = x[1] - o[1];
            loss = loss + e0 * e0 + e1 * e1;
        }
        loss = loss * (1.0 / (obs.len() - 1) as f64);
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

    /// Train for `epochs` trajectory-matching steps; returns the final loss.
    pub fn train(&mut self, obs: &[[f64; 2]], dt: f64, epochs: usize, lr: f64) -> f64 {
        let mut l = f64::INFINITY;
        for _ in 0..epochs {
            l = self.train_step(obs, dt, lr);
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Damped linear oscillator: ẋ = y, ẏ = −x − 0.15 y (a decaying spiral).
    fn true_field(x: f64, y: f64) -> (f64, f64) {
        (y, -x - 0.15 * y)
    }

    fn trajectory(x0: f64, y0: f64, n: usize, dt: f64) -> Vec<[f64; 2]> {
        let (mut x, mut y) = (x0, y0);
        let mut out = vec![[x, y]];
        for _ in 0..n {
            let f = |x: f64, y: f64| true_field(x, y);
            let (k1x, k1y) = f(x, y);
            let (k2x, k2y) = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
            let (k3x, k3y) = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
            let (k4x, k4y) = f(x + dt * k3x, y + dt * k3y);
            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
            out.push([x, y]);
        }
        out
    }

    #[test]
    fn neural_ode_learns_the_field_and_predicts_the_future() {
        // THE HEADLINE. Train on the first 24 steps of one trajectory by backprop-through-the-solver, then
        // roll the learned field forward and check it tracks the observed trajectory AND a short extrapolation
        // beyond the training window.
        let dt = 0.12;
        let full = trajectory(1.6, 0.0, 40, dt); // 41 points
        let train = &full[..25]; // first 24 steps
        let mut node = NeuralOde::new(16, 4);
        node.train(train, dt, 1500, 5e-3);

        let pred = node.rollout(&full[0], 40, dt);
        // in-window tracking
        let mut e_in = 0.0;
        for i in 0..25 {
            e_in += (pred[i][0] - full[i][0]).powi(2) + (pred[i][1] - full[i][1]).powi(2);
        }
        let rmse_in = (e_in / 25.0).sqrt();
        assert!(rmse_in < 0.1, "rollout should match the training window: rmse {rmse_in}");
        // extrapolation beyond the training window still tracks the decaying spiral
        let mut e_out = 0.0;
        for i in 25..40 {
            e_out += (pred[i][0] - full[i][0]).powi(2) + (pred[i][1] - full[i][1]).powi(2);
        }
        let rmse_out = (e_out / 15.0).sqrt();
        assert!(rmse_out < 0.25, "should extrapolate past the training window: rmse {rmse_out}");
    }
}
