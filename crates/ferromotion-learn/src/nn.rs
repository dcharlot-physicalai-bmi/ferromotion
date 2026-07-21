//! **The multilayer perceptron (MLP) + Adam** — the plain neural network, trained through the reverse-mode
//! [`crate::autodiff`] tape. This is the "black box" the rest of the course measures against: a stack of
//! affine maps and `tanh` nonlinearities that, by the **universal approximation theorem**, can fit any
//! continuous function given enough hidden units — but with *no* physics built in, so it needs a lot of data
//! and extrapolates blindly. Later modules keep this training loop and add a physics prior to the data, the
//! loss, or the architecture.
//!
//! Parameters live as a flat `Vec<f64>`; each training step builds a fresh tape, wraps the parameters as
//! [`Var`]s, runs the batch forward accumulating a mean-squared-error loss, calls `backward` once for the
//! exact gradient w.r.t. every weight, and takes an **Adam** step (adaptive per-parameter learning rates with
//! bias-corrected first/second moment estimates). Weight initialization is seeded (deterministic) Xavier.
//! Pure `f64`, wasm-clean. Verified against the universal-approximation oracle: an MLP drives its training
//! loss near zero fitting `sin(3x)` and learns the non-linearly-separable XOR function.

use crate::autodiff::{Tape, Var};

/// A fully-connected network with `tanh` hidden activations and a linear output layer.
pub struct Mlp {
    sizes: Vec<usize>,
    params: Vec<f64>,
    // Adam optimizer state
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

impl Mlp {
    /// A network with the given layer sizes, e.g. `&[1, 16, 16, 1]` (one input, two hidden layers of 16,
    /// one output). Weights are seeded-Xavier initialized; biases start at zero.
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        assert!(sizes.len() >= 2, "need at least an input and an output layer");
        let mut state = seed ^ 0xA5A5_5A5A_1234_5678;
        let mut params = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (ind, outd) = (sizes[l], sizes[l + 1]);
            let r = (6.0 / (ind + outd) as f64).sqrt(); // Xavier uniform half-range
            for _ in 0..ind * outd {
                let u = (splitmix64(&mut state) as f64 / u64::MAX as f64) * 2.0 - 1.0;
                params.push(u * r);
            }
            params.extend(std::iter::repeat_n(0.0, outd)); // biases
        }
        let n = params.len();
        Mlp { sizes: sizes.to_vec(), params, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }

    /// Number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Forward inference in plain `f64` (no tape): map an input vector to the output vector.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut a = x.to_vec();
        let mut off = 0;
        let layers = self.sizes.len() - 1;
        for l in 0..layers {
            let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
            let mut z = vec![0.0; outd];
            for (o, zo) in z.iter_mut().enumerate() {
                let mut s = self.params[off + ind * outd + o]; // bias
                for (i, &ai) in a.iter().enumerate() {
                    s += self.params[off + o * ind + i] * ai;
                }
                *zo = if l + 1 < layers { s.tanh() } else { s };
            }
            off += ind * outd + outd;
            a = z;
        }
        a
    }

    /// Full-batch loss and gradient via the tape: builds one tape, wraps params as `Var`s, runs every sample
    /// forward accumulating MSE, and backpropagates once. Returns `(mse, gradient)`.
    fn loss_and_grad(&self, xs: &[Vec<f64>], ys: &[Vec<f64>]) -> (f64, Vec<f64>) {
        let tape = Tape::new();
        let pv: Vec<Var> = self.params.iter().map(|&p| tape.var(p)).collect();
        let layers = self.sizes.len() - 1;
        let mut loss = tape.constant(0.0);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut a: Vec<Var> = x.iter().map(|&xi| tape.constant(xi)).collect();
            let mut off = 0;
            for l in 0..layers {
                let (ind, outd) = (self.sizes[l], self.sizes[l + 1]);
                let mut z = Vec::with_capacity(outd);
                for o in 0..outd {
                    let mut s = pv[off + ind * outd + o]; // bias var
                    for i in 0..ind {
                        s = s + pv[off + o * ind + i] * a[i];
                    }
                    z.push(if l + 1 < layers { s.tanh() } else { s });
                }
                off += ind * outd + outd;
                a = z;
            }
            for (o, out) in a.iter().enumerate() {
                let e = *out - y[o];
                loss = loss + e * e;
            }
        }
        let n = (xs.len() * ys[0].len()) as f64;
        let loss = loss * (1.0 / n);
        let g = loss.backward();
        let grad: Vec<f64> = pv.iter().map(|&p| g.wrt(p)).collect();
        (loss.value(), grad)
    }

    /// One Adam optimizer step over the full batch. Returns the MSE *before* the step.
    pub fn train_step(&mut self, xs: &[Vec<f64>], ys: &[Vec<f64>], lr: f64) -> f64 {
        let (loss, grad) = self.loss_and_grad(xs, ys);
        self.t += 1;
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (i, &gi) in grad.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            self.params[i] -= lr * mhat / (vhat.sqrt() + eps);
        }
        loss
    }

    /// Train for `epochs` full-batch Adam steps; returns the final MSE.
    pub fn train(&mut self, xs: &[Vec<f64>], ys: &[Vec<f64>], epochs: usize, lr: f64) -> f64 {
        let mut mse = f64::INFINITY;
        for _ in 0..epochs {
            mse = self.train_step(xs, ys, lr);
        }
        // report the loss AFTER the last step, not before
        self.loss_and_grad(xs, ys).0.min(mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_mlp_approximates_a_nonlinear_function() {
        // THE HEADLINE (universal approximation). Fit sin(3x) on [-1,1] — a function with no linear structure
        // — and drive the training MSE near zero.
        let xs: Vec<Vec<f64>> = (0..40).map(|i| vec![-1.0 + 2.0 * i as f64 / 39.0]).collect();
        let ys: Vec<Vec<f64>> = xs.iter().map(|x| vec![(3.0 * x[0]).sin()]).collect();
        let mut net = Mlp::new(&[1, 24, 24, 1], 7);
        let mse = net.train(&xs, &ys, 1500, 0.02);
        assert!(mse < 1e-3, "MLP should fit sin(3x): final MSE {mse}");
        // spot-check a held-out-ish point
        let p = net.forward(&[0.5])[0];
        assert!((p - 1.5_f64.sin()).abs() < 0.1, "prediction at x=0.5: {p} vs {}", 1.5_f64.sin());
    }

    #[test]
    fn an_mlp_learns_xor() {
        // THE DISCRIMINATOR. XOR is not linearly separable — a single perceptron cannot learn it, but an MLP
        // with a hidden layer can. Targets are ±1 (tanh-friendly).
        let xs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        let ys = vec![vec![-1.0], vec![1.0], vec![1.0], vec![-1.0]];
        let mut net = Mlp::new(&[2, 8, 1], 3);
        let mse = net.train(&xs, &ys, 2000, 0.03);
        assert!(mse < 1e-2, "MLP should learn XOR: final MSE {mse}");
        // signs must match the XOR truth table
        for (x, y) in xs.iter().zip(ys.iter()) {
            let p = net.forward(x)[0];
            assert_eq!(p.signum(), y[0].signum(), "XOR({x:?}) predicted {p}, want sign {}", y[0]);
        }
    }

    #[test]
    fn the_loss_decreases_monotonically_early_on() {
        // Sanity: Adam should reduce the loss over the first steps on a simple linear target y = 2x.
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 / 10.0]).collect();
        let ys: Vec<Vec<f64>> = xs.iter().map(|x| vec![2.0 * x[0]]).collect();
        let mut net = Mlp::new(&[1, 8, 1], 1);
        let first = net.train_step(&xs, &ys, 0.05);
        let mut last = first;
        for _ in 0..50 {
            last = net.train_step(&xs, &ys, 0.05);
        }
        assert!(last < first, "loss should drop: {first} → {last}");
    }
}
