//! **Reverse-mode automatic differentiation** — the keystone the whole crate stands on. Every learned
//! physical model in `ferromotion-learn` (PINNs, Lagrangian/Hamiltonian nets, Neural ODEs, model-structured
//! nets) is trained by gradient descent, and every gradient here is computed *exactly* — not by finite
//! differences — by recording the forward computation on a **tape** (a Wengert list) and walking it backward,
//! accumulating `∂output/∂·` via the chain rule. One backward pass yields the gradient w.r.t. *all* inputs at
//! once, which is why reverse mode (a.k.a. backpropagation) is the right tool when there are many parameters
//! and one scalar loss.
//!
//! The design is a shared [`Tape`] that owns the nodes and lightweight [`Var`] handles (`Copy`, carrying a
//! value + a tape index) that record operations as they execute. Each node stores up to two parents and the
//! local partial derivative w.r.t. each; leaves point to themselves with weight 0 so propagation stops. This
//! is pure `std` + `f64` — no BLAS, no `ndarray` — so it compiles to WASM and runs on-device, the whole point
//! of doing this in Rust rather than importing PyTorch.
//!
//! Verified against **finite differences**: for a battery of nonlinear scalar functions of several variables,
//! the tape's exact gradient matches a central-difference gradient to ~1e-6, and the classic identities
//! (`∂(x·y)/∂x = y`, `∂tanh/∂x = 1−tanh²`, product/quotient/chain rules) hold. For higher-order input
//! derivatives (needed by PINNs and Lagrangian nets), see the forward-mode [`crate::dual`] module.

use core::cell::RefCell;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// One entry of the tape: up to two parents and the local partial derivative of this node w.r.t. each.
#[derive(Clone, Copy)]
struct Node {
    dep: [usize; 2],
    weight: [f64; 2],
}

/// The tape (Wengert list) recording the forward computation. Create [`Var`]s from it, compute, then call
/// [`Var::backward`]. Interior mutability lets `Copy` `Var` handles append nodes during ordinary arithmetic.
#[derive(Default)]
pub struct Tape {
    nodes: RefCell<Vec<Node>>,
}

impl Tape {
    /// A fresh, empty tape.
    pub fn new() -> Self {
        Tape { nodes: RefCell::new(Vec::new()) }
    }

    /// Number of recorded nodes (operations + leaves).
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.borrow().is_empty()
    }

    /// Introduce an independent variable (a leaf) with the given value.
    pub fn var(&self, value: f64) -> Var<'_> {
        let index = self.push([usize::MAX, usize::MAX], [0.0, 0.0], true);
        Var { tape: self, index, value }
    }

    /// A constant on this tape (a leaf whose gradient never flows). Same as `var` but semantically read-only.
    pub fn constant(&self, value: f64) -> Var<'_> {
        self.var(value)
    }

    fn push(&self, dep: [usize; 2], weight: [f64; 2], leaf: bool) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        // A leaf refers to itself with weight 0 so reverse propagation is a harmless no-op there.
        let dep = if leaf { [index, index] } else { dep };
        nodes.push(Node { dep, weight });
        index
    }
}

/// A differentiable value: a `Copy` handle carrying its numeric value and its position on the [`Tape`].
#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    index: usize,
    value: f64,
}

/// The result of a backward pass: the gradient of the seed output w.r.t. every node on the tape.
pub struct Grad(Vec<f64>);

impl Grad {
    /// The partial derivative of the backward-seed output w.r.t. the given variable.
    pub fn wrt(&self, v: Var<'_>) -> f64 {
        self.0[v.index]
    }
}

impl<'t> Var<'t> {
    /// The current numeric value.
    pub fn value(&self) -> f64 {
        self.value
    }

    fn unary(self, value: f64, d: f64) -> Var<'t> {
        let index = self.tape.push([self.index, self.index], [d, 0.0], false);
        Var { tape: self.tape, index, value }
    }

    fn binary(self, rhs: Var<'t>, value: f64, da: f64, db: f64) -> Var<'t> {
        let index = self.tape.push([self.index, rhs.index], [da, db], false);
        Var { tape: self.tape, index, value }
    }

    /// Reverse pass seeded at `self`: returns `∂self/∂·` for every node. Cost is O(#nodes), one linear sweep.
    pub fn backward(&self) -> Grad {
        let nodes = self.tape.nodes.borrow();
        let mut grad = vec![0.0; nodes.len()];
        grad[self.index] = 1.0;
        for i in (0..nodes.len()).rev() {
            let g = grad[i];
            if g != 0.0 {
                let n = nodes[i];
                grad[n.dep[0]] += n.weight[0] * g;
                grad[n.dep[1]] += n.weight[1] * g;
            }
        }
        Grad(grad)
    }

    // --- elementary functions, each recording its exact local derivative ---

    /// Sine (`d/dx sin x = cos x`).
    pub fn sin(self) -> Var<'t> {
        self.unary(self.value.sin(), self.value.cos())
    }
    /// Cosine (`d/dx cos x = −sin x`).
    pub fn cos(self) -> Var<'t> {
        self.unary(self.value.cos(), -self.value.sin())
    }
    /// Exponential (`d/dx eˣ = eˣ`).
    pub fn exp(self) -> Var<'t> {
        let e = self.value.exp();
        self.unary(e, e)
    }
    /// Natural log (`d/dx ln x = 1/x`).
    pub fn ln(self) -> Var<'t> {
        self.unary(self.value.ln(), 1.0 / self.value)
    }
    /// Hyperbolic tangent (`d/dx tanh x = 1 − tanh²x`) — the workhorse network activation.
    pub fn tanh(self) -> Var<'t> {
        let t = self.value.tanh();
        self.unary(t, 1.0 - t * t)
    }
    /// Square root (`d/dx √x = 1/(2√x)`).
    pub fn sqrt(self) -> Var<'t> {
        let s = self.value.sqrt();
        self.unary(s, 0.5 / s)
    }
    /// Integer/real power (`d/dx xⁿ = n·xⁿ⁻¹`).
    pub fn powf(self, n: f64) -> Var<'t> {
        self.unary(self.value.powf(n), n * self.value.powf(n - 1.0))
    }
    /// Logistic sigmoid `σ(x) = 1/(1+e⁻ˣ)` (`σ' = σ(1−σ)`).
    pub fn sigmoid(self) -> Var<'t> {
        let s = 1.0 / (1.0 + (-self.value).exp());
        self.unary(s, s * (1.0 - s))
    }
    /// Rectified linear unit (`ReLU' = [x>0]`).
    pub fn relu(self) -> Var<'t> {
        let d = if self.value > 0.0 { 1.0 } else { 0.0 };
        self.unary(self.value.max(0.0), d)
    }
}

impl<'t> Add for Var<'t> {
    type Output = Var<'t>;
    fn add(self, rhs: Var<'t>) -> Var<'t> {
        self.binary(rhs, self.value + rhs.value, 1.0, 1.0)
    }
}
impl<'t> Sub for Var<'t> {
    type Output = Var<'t>;
    fn sub(self, rhs: Var<'t>) -> Var<'t> {
        self.binary(rhs, self.value - rhs.value, 1.0, -1.0)
    }
}
impl<'t> Mul for Var<'t> {
    type Output = Var<'t>;
    fn mul(self, rhs: Var<'t>) -> Var<'t> {
        self.binary(rhs, self.value * rhs.value, rhs.value, self.value)
    }
}
impl<'t> Div for Var<'t> {
    type Output = Var<'t>;
    fn div(self, rhs: Var<'t>) -> Var<'t> {
        let v = self.value / rhs.value;
        self.binary(rhs, v, 1.0 / rhs.value, -self.value / (rhs.value * rhs.value))
    }
}
impl<'t> Neg for Var<'t> {
    type Output = Var<'t>;
    fn neg(self) -> Var<'t> {
        self.unary(-self.value, -1.0)
    }
}

// scalar convenience: Var ∘ f64
impl<'t> Add<f64> for Var<'t> {
    type Output = Var<'t>;
    fn add(self, rhs: f64) -> Var<'t> {
        self.unary(self.value + rhs, 1.0)
    }
}
impl<'t> Mul<f64> for Var<'t> {
    type Output = Var<'t>;
    fn mul(self, rhs: f64) -> Var<'t> {
        self.unary(self.value * rhs, rhs)
    }
}
impl<'t> Sub<f64> for Var<'t> {
    type Output = Var<'t>;
    fn sub(self, rhs: f64) -> Var<'t> {
        self.unary(self.value - rhs, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference gradient of a scalar function of `n` inputs — the oracle.
    fn fd_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        (0..x.len())
            .map(|i| {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[i] += h;
                xm[i] -= h;
                (f(&xp) - f(&xm)) / (2.0 * h)
            })
            .collect()
    }

    #[test]
    fn product_and_quotient_rules_are_exact() {
        // THE ORACLE. ∂(x·y)/∂x = y, ∂(x/y)/∂y = −x/y².
        let t = Tape::new();
        let x = t.var(3.0);
        let y = t.var(4.0);
        let z = x * y;
        let g = z.backward();
        assert!((g.wrt(x) - 4.0).abs() < 1e-12 && (g.wrt(y) - 3.0).abs() < 1e-12);

        let t2 = Tape::new();
        let a = t2.var(3.0);
        let b = t2.var(4.0);
        let q = a / b;
        let gq = q.backward();
        assert!((gq.wrt(a) - 0.25).abs() < 1e-12, "∂(a/b)/∂a = 1/b");
        assert!((gq.wrt(b) - (-3.0 / 16.0)).abs() < 1e-12, "∂(a/b)/∂b = −a/b²");
    }

    #[test]
    fn tanh_derivative_matches_one_minus_tanh_squared() {
        let t = Tape::new();
        let x = t.var(0.7);
        let y = x.tanh();
        let g = y.backward();
        let expect = 1.0 - 0.7_f64.tanh().powi(2);
        assert!((g.wrt(x) - expect).abs() < 1e-12, "tanh' = 1 − tanh²");
    }

    #[test]
    fn gradient_of_a_nonlinear_multivariable_function_matches_finite_differences() {
        // THE HEADLINE. f(x,y,z) = sin(x·y) + exp(z) / (1 + x²) − tanh(y + z). Exact reverse-mode gradient vs
        // central differences at a nontrivial point.
        let f = |v: &[f64]| {
            let (x, y, z) = (v[0], v[1], v[2]);
            (x * y).sin() + z.exp() / (1.0 + x * x) - (y + z).tanh()
        };
        let x0 = [0.6, -1.3, 0.4];
        let t = Tape::new();
        let x = t.var(x0[0]);
        let y = t.var(x0[1]);
        let z = t.var(x0[2]);
        let one = t.constant(1.0);
        let out = (x * y).sin() + z.exp() / (one + x * x) - (y + z).tanh();
        let g = out.backward();
        let fd = fd_grad(f, &x0);
        for (i, &v) in [g.wrt(x), g.wrt(y), g.wrt(z)].iter().enumerate() {
            assert!((v - fd[i]).abs() < 1e-6, "grad[{i}]: autodiff {v} vs fd {}", fd[i]);
        }
    }

    #[test]
    fn a_shared_subexpression_accumulates_both_paths() {
        // THE DISCRIMINATOR. When a variable feeds an output through two paths, reverse mode sums both
        // contributions in ONE sweep. f = x·x + x → ∂f/∂x = 2x + 1.
        let t = Tape::new();
        let x = t.var(5.0);
        let f = x * x + x;
        let g = f.backward();
        assert!((g.wrt(x) - 11.0).abs() < 1e-12, "∂(x²+x)/∂x = 2x+1 = 11");
    }

    #[test]
    fn sigmoid_and_relu_gradients_are_correct() {
        let t = Tape::new();
        let x = t.var(0.5);
        let s = x.sigmoid();
        let gs = s.backward();
        let sv = 1.0 / (1.0 + (-0.5_f64).exp());
        assert!((gs.wrt(x) - sv * (1.0 - sv)).abs() < 1e-12, "σ' = σ(1−σ)");

        let t2 = Tape::new();
        let xp = t2.var(2.0);
        let xn = t2.var(-2.0);
        assert!((xp.relu().backward().wrt(xp) - 1.0).abs() < 1e-12, "ReLU'(+) = 1");
        assert!(xn.relu().backward().wrt(xn).abs() < 1e-12, "ReLU'(−) = 0");
    }
}
