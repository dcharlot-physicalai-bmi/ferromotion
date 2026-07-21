//! **Forward-mode automatic differentiation** — dual and hyper-dual numbers for *exact* first and second
//! derivatives of a function w.r.t. its **inputs**. Reverse mode ([`crate::autodiff`]) is the right tool for
//! `∂loss/∂parameters` (one scalar out, many params in); forward mode is the right tool for the *other*
//! direction a physics-informed model needs: the derivatives of the model's output w.r.t. a small number of
//! inputs — `∂u/∂t`, `∂u/∂x`, `∂²u/∂x²` for a PINN residual; `∂L/∂q̇`, `∂²L/∂q̇²` (the learned mass matrix)
//! for a Lagrangian net.
//!
//! A [`Dual`] carries `(value, derivative)` and propagates the chain rule automatically through arithmetic:
//! evaluate any function on `Dual::var(x)` and read `.eps` for `f'(x)`. A [`HyperDual`] carries four
//! components `(re, e1, e2, e12)` with two independent infinitesimals `ε₁, ε₂` (`ε₁²=ε₂²=0`); seed `ε₁` on
//! one input and `ε₂` on another and the `e12` slot holds the exact mixed partial `∂²f/∂x∂y` — with **no
//! step-size error**, unlike finite differences (seed both on the *same* input to get `f''`). Pure `f64`,
//! wasm-clean. Verified against central differences.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// A first-order dual number `re + eps·ε` (`ε² = 0`). Evaluating `f` on `Dual::var(x)` yields `f(x)` in `re`
/// and `f'(x)` in `eps`.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub re: f64,
    pub eps: f64,
}

impl Dual {
    /// The active variable `x` (seed derivative 1).
    pub fn var(x: f64) -> Self {
        Dual { re: x, eps: 1.0 }
    }
    /// A constant (derivative 0).
    pub fn constant(c: f64) -> Self {
        Dual { re: c, eps: 0.0 }
    }
    fn map(self, v: f64, d: f64) -> Self {
        Dual { re: v, eps: d * self.eps }
    }
    /// Sine.
    pub fn sin(self) -> Self {
        self.map(self.re.sin(), self.re.cos())
    }
    /// Cosine.
    pub fn cos(self) -> Self {
        self.map(self.re.cos(), -self.re.sin())
    }
    /// Exponential.
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        self.map(e, e)
    }
    /// Natural log.
    pub fn ln(self) -> Self {
        self.map(self.re.ln(), 1.0 / self.re)
    }
    /// Hyperbolic tangent.
    pub fn tanh(self) -> Self {
        let t = self.re.tanh();
        self.map(t, 1.0 - t * t)
    }
    /// Real power.
    pub fn powf(self, n: f64) -> Self {
        self.map(self.re.powf(n), n * self.re.powf(n - 1.0))
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual { re: self.re + o.re, eps: self.eps + o.eps }
    }
}
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual { re: self.re - o.re, eps: self.eps - o.eps }
    }
}
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual { re: self.re * o.re, eps: self.re * o.eps + self.eps * o.re }
    }
}
impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual { re: self.re / o.re, eps: (self.eps * o.re - self.re * o.eps) / (o.re * o.re) }
    }
}
impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { re: -self.re, eps: -self.eps }
    }
}

/// A second-order (hyper-)dual number `re + e1·ε₁ + e2·ε₂ + e12·ε₁ε₂`, with `ε₁² = ε₂² = 0`. Seeding `ε₁`
/// and `ε₂` on two inputs makes `e12` the exact mixed partial `∂²f/∂x∂y`; seeding both on the same input
/// makes `e12 = f''`.
#[derive(Clone, Copy, Debug)]
pub struct HyperDual {
    pub re: f64,
    pub e1: f64,
    pub e2: f64,
    pub e12: f64,
}

impl HyperDual {
    /// A constant.
    pub fn constant(c: f64) -> Self {
        HyperDual { re: c, e1: 0.0, e2: 0.0, e12: 0.0 }
    }
    /// A value seeded active in **both** infinitesimal directions — used to read the pure second derivative
    /// `f''(x)` from `e12`.
    pub fn var2(x: f64) -> Self {
        HyperDual { re: x, e1: 1.0, e2: 1.0, e12: 0.0 }
    }
    /// A value seeded active in the `ε₁` direction only (for mixed partials).
    pub fn var_e1(x: f64) -> Self {
        HyperDual { re: x, e1: 1.0, e2: 0.0, e12: 0.0 }
    }
    /// A value seeded active in the `ε₂` direction only (for mixed partials).
    pub fn var_e2(x: f64) -> Self {
        HyperDual { re: x, e1: 0.0, e2: 1.0, e12: 0.0 }
    }
    /// Apply a scalar function given `h(x)`, `h'(x)`, `h''(x)`.
    fn chain(self, h: f64, dh: f64, ddh: f64) -> Self {
        HyperDual {
            re: h,
            e1: dh * self.e1,
            e2: dh * self.e2,
            e12: dh * self.e12 + ddh * self.e1 * self.e2,
        }
    }
    /// Sine.
    pub fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos(), -self.re.sin())
    }
    /// Cosine.
    pub fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin(), -self.re.cos())
    }
    /// Exponential.
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e, e)
    }
    /// Hyperbolic tangent (`tanh' = 1−tanh²`, `tanh'' = −2·tanh·(1−tanh²)`).
    pub fn tanh(self) -> Self {
        let t = self.re.tanh();
        let d = 1.0 - t * t;
        self.chain(t, d, -2.0 * t * d)
    }
    /// Real power.
    pub fn powf(self, n: f64) -> Self {
        self.chain(self.re.powf(n), n * self.re.powf(n - 1.0), n * (n - 1.0) * self.re.powf(n - 2.0))
    }
}

impl Add for HyperDual {
    type Output = HyperDual;
    fn add(self, o: HyperDual) -> HyperDual {
        HyperDual { re: self.re + o.re, e1: self.e1 + o.e1, e2: self.e2 + o.e2, e12: self.e12 + o.e12 }
    }
}
impl Sub for HyperDual {
    type Output = HyperDual;
    fn sub(self, o: HyperDual) -> HyperDual {
        HyperDual { re: self.re - o.re, e1: self.e1 - o.e1, e2: self.e2 - o.e2, e12: self.e12 - o.e12 }
    }
}
impl Mul for HyperDual {
    type Output = HyperDual;
    fn mul(self, o: HyperDual) -> HyperDual {
        HyperDual {
            re: self.re * o.re,
            e1: self.re * o.e1 + self.e1 * o.re,
            e2: self.re * o.e2 + self.e2 * o.re,
            e12: self.re * o.e12 + self.e1 * o.e2 + self.e2 * o.e1 + self.e12 * o.re,
        }
    }
}
impl Neg for HyperDual {
    type Output = HyperDual;
    fn neg(self) -> HyperDual {
        HyperDual { re: -self.re, e1: -self.e1, e2: -self.e2, e12: -self.e12 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_first_derivative_matches_finite_difference() {
        // THE ORACLE. f(x) = sin(x)·exp(x); f'(x) via dual vs central difference.
        let f = |x: f64| x.sin() * x.exp();
        let x0 = 0.8;
        let d = (Dual::var(x0).sin()) * (Dual::var(x0).exp());
        let fd = (f(x0 + 1e-6) - f(x0 - 1e-6)) / 2e-6;
        assert!((d.re - f(x0)).abs() < 1e-12, "value");
        assert!((d.eps - fd).abs() < 1e-6, "dual f'={} vs fd={fd}", d.eps);
    }

    #[test]
    fn hyperdual_second_derivative_matches_finite_difference() {
        // THE HEADLINE. f(x) = tanh(x²); f''(x) read from e12, vs a central second difference.
        let f = |x: f64| (x * x).tanh();
        let x0 = 0.9;
        let x = HyperDual::var2(x0);
        let y = (x * x).tanh();
        let h = 1e-4;
        let fdd = (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h);
        assert!((y.re - f(x0)).abs() < 1e-12, "value");
        assert!((y.e1 - 2.0 * x0 * (1.0 - f(x0) * f(x0))).abs() < 1e-9, "first derivative in e1");
        assert!((y.e12 - fdd).abs() < 1e-4, "hyperdual f''={} vs fd={fdd}", y.e12);
    }

    #[test]
    fn hyperdual_mixed_partial_is_exact() {
        // THE DISCRIMINATOR. f(x,y) = sin(x)·y²; ∂²f/∂x∂y = 2y·cos(x), read from e12 with x on ε₁, y on ε₂.
        let (x0, y0) = (0.5, 1.7);
        let x = HyperDual::var_e1(x0);
        let y = HyperDual::var_e2(y0);
        let out = x.sin() * (y * y);
        let expect = 2.0 * y0 * x0.cos();
        assert!((out.e12 - expect).abs() < 1e-12, "mixed partial {} vs {expect}", out.e12);
        // and the pure first partials live in e1 (∂/∂x = cos x · y²) and e2 (∂/∂y = sin x · 2y)
        assert!((out.e1 - x0.cos() * y0 * y0).abs() < 1e-12, "∂/∂x");
        assert!((out.e2 - x0.sin() * 2.0 * y0).abs() < 1e-12, "∂/∂y");
    }
}
