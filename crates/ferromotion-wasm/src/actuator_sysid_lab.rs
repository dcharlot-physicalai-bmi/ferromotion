//! **Trajectory-matching system identification** — the rig behind the "the industry agrees" lesson.
//!
//! Fit a *nonlinear* drag coefficient by differentiating a multi-step rollout against recorded data:
//!
//! ```text
//!   m v' = u - c v |v|,      loss(c) = sum_k ( x_k(c) - x_k^data )^2
//! ```
//!
//! Quadratic drag is what makes this a genuine identification problem rather than a linear least-squares one: `c`
//! enters through `v|v|`, so there is no normal equation to solve and the only route is a gradient through the whole
//! rollout. The gradient comes from [`ferromotion_learn::Tape`], exact to the last bit, and the lab shows it agreeing
//! with a finite difference so a reader can see that "exact" is a claim being checked rather than asserted.
//!
//! This lab existed as a lesson before it existed as code: PAI-230 lesson 14 referenced `task.lab = "actuator-sysid"`,
//! which was registered nowhere, so its bench rendered blank. A cross-check test now makes that condition impossible.

use ferromotion_learn::Tape;
use wasm_bindgen::prelude::*;

const MASS: f64 = 1.4;
const DT: f64 = 0.02;
const STEPS: usize = 120;
/// The coefficient the recorded data was generated with, and the answer the reader is fitting toward.
const C_TRUE: f64 = 0.85;

/// The excitation: a chirp, so the rollout visits a range of speeds. A constant drive would leave `c` weakly
/// identified, because quadratic drag is only informative where `|v|` varies.
fn drive(k: usize) -> f64 {
    let t = k as f64 * DT;
    6.0 * (2.4 * t + 0.35 * t * t).sin()
}

/// Roll the model out with coefficient `c`, returning the position trace.
fn rollout(c: f64) -> Vec<f64> {
    let (mut x, mut v) = (0.0, 0.0);
    let mut trace = Vec::with_capacity(STEPS);
    for k in 0..STEPS {
        let a = (drive(k) - c * v * v.abs()) / MASS;
        v += a * DT;
        x += v * DT;
        trace.push(x);
    }
    trace
}

#[wasm_bindgen]
pub struct ActuatorSysIdLab {
    data: Vec<f64>,
    c: f64,
    lr: f64,
    iters: u32,
    loss0: f64,
}

#[wasm_bindgen]
impl ActuatorSysIdLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ActuatorSysIdLab {
        let data = rollout(C_TRUE);
        let mut lab = ActuatorSysIdLab { data, c: 0.20, lr: 0.35, iters: 0, loss0: 0.0 };
        lab.loss0 = lab.loss();
        lab
    }

    /// The starting guess, deliberately far from the truth.
    pub fn reset(&mut self) {
        self.c = 0.20;
        self.iters = 0;
    }

    pub fn set_coefficient(&mut self, c: f64) {
        self.c = c.clamp(0.0, 3.0);
        self.iters = 0;
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr.clamp(0.01, 2.0);
    }

    pub fn coefficient(&self) -> f64 {
        self.c
    }

    pub fn truth(&self) -> f64 {
        C_TRUE
    }

    pub fn iterations(&self) -> u32 {
        self.iters
    }

    pub fn steps(&self) -> usize {
        STEPS
    }

    /// Trajectory mismatch at the current coefficient.
    pub fn loss(&self) -> f64 {
        self.loss_at(self.c)
    }

    pub fn loss_at(&self, c: f64) -> f64 {
        rollout(c).iter().zip(&self.data).map(|(p, d)| (p - d) * (p - d)).sum()
    }

    /// How far the loss has fallen from the starting guess.
    pub fn loss_reduction(&self) -> f64 {
        if self.loss0 > 0.0 { self.loss() / self.loss0 } else { f64::NAN }
    }

    pub fn coefficient_error(&self) -> f64 {
        (self.c - C_TRUE).abs()
    }

    /// **The exact gradient**, by one reverse sweep of the tape through all `STEPS` of the rollout. Quadratic drag
    /// means this is the only route: there is no closed form to fall back on.
    pub fn gradient(&self) -> f64 {
        let tape = Tape::new();
        let c = tape.var(self.c);
        let mut x = tape.constant(0.0);
        let mut v = tape.constant(0.0);
        let mut loss = tape.constant(0.0);
        for k in 0..STEPS {
            // v |v| without an abs() on the tape: v * v is |v|^2, so v*v*sign(v) is v|v|. The sign is a constant
            // within a step, which is exactly how a subgradient at v = 0 is handled anyway.
            let sign = if v.value() >= 0.0 { 1.0 } else { -1.0 };
            let drag = c * v * v * tape.constant(sign);
            let a = (tape.constant(drive(k)) - drag) * tape.constant(1.0 / MASS);
            v = v + a * tape.constant(DT);
            x = x + v * tape.constant(DT);
            let r = x - tape.constant(self.data[k]);
            loss = loss + r * r;
        }
        loss.backward().wrt(c)
    }

    /// The same gradient by central differences, so the reader can check the tape rather than trust it.
    pub fn gradient_finite_difference(&self) -> f64 {
        let h = 1e-6;
        (self.loss_at(self.c + h) - self.loss_at(self.c - h)) / (2.0 * h)
    }

    /// One gradient-descent step on the coefficient.
    pub fn step(&mut self) {
        let g = self.gradient();
        if g.is_finite() {
            // the loss scales with the trace length, so normalise the step by it to keep the rate readable
            self.c = (self.c - self.lr * g / STEPS as f64).clamp(0.0, 3.0);
            self.iters += 1;
        }
    }

    /// Predicted position at step `k` under the current coefficient, for plotting against the data.
    pub fn predicted(&self, k: usize) -> f64 {
        rollout(self.c).get(k).copied().unwrap_or(f64::NAN)
    }

    pub fn measured(&self, k: usize) -> f64 {
        self.data.get(k).copied().unwrap_or(f64::NAN)
    }

    /// The lesson's pass condition: the coefficient is identified to better than 0.05.
    pub fn passed(&self) -> bool {
        self.coefficient_error() < 0.05
    }
}

impl Default for ActuatorSysIdLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tape gradient must match a finite difference, or the lesson's central claim is unsupported.
    #[test]
    fn the_tape_gradient_matches_a_finite_difference() {
        let mut lab = ActuatorSysIdLab::new();
        for c in [0.20, 0.50, 0.85, 1.40] {
            lab.set_coefficient(c);
            let (tape, fd) = (lab.gradient(), lab.gradient_finite_difference());
            // Relative error is meaningless where the reference is zero - and it IS zero at the true coefficient, so
            // a bare `(tape-fd)/fd` reports total disagreement exactly where the two agree perfectly. Fall back to the
            // absolute difference once the reference is negligible.
            let abs = (tape - fd).abs();
            let rel = if fd.abs() > 1e-6 { abs / fd.abs() } else { abs };
            eprintln!("c = {c:.2}: tape {tape:>12.6}, finite difference {fd:>12.6}, error {rel:.2e}{}", if fd.abs() > 1e-6 { " (relative)" } else { " (absolute, reference is zero)" });
            assert!(rel < 1e-5, "the tape must be the gradient: {rel:.3e}");
        }
        // and it vanishes at the truth, which is what makes it the right objective
        lab.set_coefficient(C_TRUE);
        eprintln!("at the true coefficient {C_TRUE}: gradient {:.3e}, loss {:.3e}", lab.gradient(), lab.loss());
        assert!(lab.gradient().abs() < 1e-9 && lab.loss() < 1e-20);
    }

    /// Descent has to actually identify the coefficient, from a start far away.
    #[test]
    fn descent_identifies_the_coefficient() {
        let mut lab = ActuatorSysIdLab::new();
        eprintln!("start: c = {:.4} (truth {C_TRUE}), loss {:.4e}", lab.coefficient(), lab.loss());
        for _ in 0..400 {
            lab.step();
        }
        eprintln!("after {} steps: c = {:.4}, error {:.2e}, loss {:.3e} ({:.1e} of the start)", lab.iterations(), lab.coefficient(), lab.coefficient_error(), lab.loss(), lab.loss_reduction());
        assert!(lab.passed(), "coefficient error {:.4} must be under 0.05", lab.coefficient_error());
        assert!(lab.loss_reduction() < 1e-3, "the trajectory match improves by orders of magnitude");
    }

    /// Quadratic drag is what makes this nonlinear: the loss is not a parabola in `c`, so no normal equation solves it.
    #[test]
    fn the_problem_is_genuinely_nonlinear() {
        let lab = ActuatorSysIdLab::new();
        // a quadratic in c would have a constant second difference; measure it and show it is not constant
        let h = 0.2;
        let second = |c: f64| lab.loss_at(c + h) - 2.0 * lab.loss_at(c) + lab.loss_at(c - h);
        let (a, b) = (second(0.5), second(1.5));
        eprintln!("second difference of the loss at c = 0.5: {a:.4e}, at c = 1.5: {b:.4e}, ratio {:.2}", a / b);
        assert!((a / b - 1.0).abs() > 0.5, "a quadratic loss would give a ratio of 1: {:.3}", a / b);
        eprintln!("   not a parabola, so there is no closed-form least squares and the gradient is the only route");
    }
}
