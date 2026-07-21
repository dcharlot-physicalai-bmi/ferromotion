//! **Autodiff lab** — the rig behind the "how machines learn" lesson. It drives the real
//! [`ferromotion_learn::Tape`] reverse-mode autodiff so the reader can watch gradient descent walk downhill
//! on a 2-D surface, and can confirm that the tape's *exact* gradient matches a finite-difference gradient at
//! every point. The surface is a simple anisotropic bowl `f(x,y) = (x−3)² + 2·(y+1)²` (minimum at `(3,−1)`),
//! valley steeper in `y` than `x` — so the descent path visibly curves, the classic picture of why the
//! gradient (not the value) is what learning follows.

use ferromotion_learn::Tape;
use wasm_bindgen::prelude::*;

/// Evaluate `f(x,y) = (x−3)² + 2(y+1)²` and its gradient via the reverse-mode tape.
fn eval_grad(x: f64, y: f64) -> (f64, f64, f64) {
    let tape = Tape::new();
    let vx = tape.var(x);
    let vy = tape.var(y);
    let dx = vx - tape.constant(3.0);
    let dy = vy + tape.constant(1.0);
    let f = dx * dx + (dy * dy) * 2.0;
    let g = f.backward();
    (f.value(), g.wrt(vx), g.wrt(vy))
}

#[wasm_bindgen]
pub struct AutodiffLab {
    x: f64,
    y: f64,
    lr: f64,
    steps: u32,
}

#[wasm_bindgen]
impl AutodiffLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AutodiffLab {
        AutodiffLab { x: -2.5, y: 2.5, lr: 0.1, steps: 0 }
    }

    /// Place the query/descent point.
    pub fn set_point(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
        self.steps = 0;
    }
    /// Set the gradient-descent learning rate (step size).
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// The surface height `f(x,y)` at an arbitrary point (for drawing contours).
    pub fn value_at(&self, x: f64, y: f64) -> f64 {
        eval_grad(x, y).0
    }

    /// Current value and the exact gradient at the current point.
    pub fn value(&self) -> f64 {
        eval_grad(self.x, self.y).0
    }
    pub fn grad_x(&self) -> f64 {
        eval_grad(self.x, self.y).1
    }
    pub fn grad_y(&self) -> f64 {
        eval_grad(self.x, self.y).2
    }

    /// The finite-difference gradient at the current point — the numerical approximation the exact autodiff
    /// gradient should match (the lesson's "they agree" reveal).
    pub fn fd_grad_x(&self) -> f64 {
        let h = 1e-5;
        (eval_grad(self.x + h, self.y).0 - eval_grad(self.x - h, self.y).0) / (2.0 * h)
    }
    pub fn fd_grad_y(&self) -> f64 {
        let h = 1e-5;
        (eval_grad(self.x, self.y + h).0 - eval_grad(self.x, self.y - h).0) / (2.0 * h)
    }

    /// Take one gradient-descent step `p ← p − lr·∇f(p)` using the exact tape gradient. Returns the new value.
    pub fn step(&mut self) -> f64 {
        let (_, gx, gy) = eval_grad(self.x, self.y);
        self.x -= self.lr * gx;
        self.y -= self.lr * gy;
        self.steps += 1;
        self.value()
    }

    pub fn x(&self) -> f64 {
        self.x
    }
    pub fn y(&self) -> f64 {
        self.y
    }
    pub fn n_steps(&self) -> u32 {
        self.steps
    }
    /// Distance to the known minimum `(3, −1)` — for a "converged?" readout.
    pub fn dist_to_min(&self) -> f64 {
        ((self.x - 3.0).powi(2) + (self.y + 1.0).powi(2)).sqrt()
    }
}

impl Default for AutodiffLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_gradient_matches_finite_difference() {
        let lab = AutodiffLab::new();
        // exact ∇f = (2(x−3), 4(y+1)); check the tape against both the closed form and finite differences
        assert!((lab.grad_x() - 2.0 * (-2.5 - 3.0)).abs() < 1e-9);
        assert!((lab.grad_y() - 4.0 * (2.5 + 1.0)).abs() < 1e-9);
        assert!((lab.grad_x() - lab.fd_grad_x()).abs() < 1e-4, "tape vs finite diff x");
        assert!((lab.grad_y() - lab.fd_grad_y()).abs() < 1e-4, "tape vs finite diff y");
    }

    #[test]
    fn gradient_descent_reaches_the_minimum() {
        let mut lab = AutodiffLab::new();
        lab.set_lr(0.1);
        for _ in 0..200 {
            lab.step();
        }
        assert!(lab.dist_to_min() < 1e-3, "descent should converge to (3,−1): dist {}", lab.dist_to_min());
    }
}
