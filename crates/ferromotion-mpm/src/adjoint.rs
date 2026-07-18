//! **The adjoint method for differentiable-physics control** — the technique behind Diff-MPM / DiffTaichi
//! (Hu et al., ICLR 2020): differentiate *through a whole simulation rollout* so a scalar loss at the end
//! yields the gradient w.r.t. an entire per-step control sequence in a single reverse pass. That one
//! backward pass is what makes gradient-based control of a deformable body tractable — the control is
//! high-dimensional (many steps × dimensions), the loss is scalar, and reverse-mode gets all the gradients
//! at once (vs. one forward simulation per parameter for finite differences).
//!
//! This module provides (1) a small, general **reverse-mode automatic-differentiation tape** ([`Tape`],
//! [`Var`]) and (2) a **differentiable 2-D mass-spring soft body** ([`SoftBody`]) built on it. A terminal
//! loss (center-of-mass to a target) is differentiated back to a per-step body-force control; the adjoint
//! gradient is checked against finite differences, and gradient descent then discovers a control that
//! drives the soft body to its target — the canonical differentiable-physics result. Pure Rust → WASM.
//!
//! (The sibling [`crate::MpmSim`] carries the *forward-mode* analytic material gradient `∂KE/∂E`; this is
//! the *reverse-mode* trajectory adjoint, the complementary half of differentiable physics.)

use std::cell::RefCell;
use std::rc::Rc;

/// One tape node: up to two parents and the local partial derivatives `∂self/∂parent`.
#[derive(Clone, Copy)]
struct Node {
    parent: [usize; 2],
    weight: [f64; 2],
}

/// A reverse-mode autodiff tape. Every arithmetic operation on a [`Var`] appends a node recording its
/// parents and local derivatives; [`Tape::grad`] then walks the tape backward to accumulate adjoints.
#[derive(Clone, Default)]
pub struct Tape {
    nodes: Rc<RefCell<Vec<Node>>>,
}

/// A scalar value tracked on a [`Tape`].
#[derive(Clone)]
pub struct Var {
    pub v: f64,
    i: usize,
    tape: Tape,
}

impl Tape {
    pub fn new() -> Tape {
        Tape { nodes: Rc::new(RefCell::new(Vec::new())) }
    }

    /// A leaf variable (an independent input) with value `v`.
    pub fn var(&self, v: f64) -> Var {
        let mut n = self.nodes.borrow_mut();
        let i = n.len();
        n.push(Node { parent: [i, i], weight: [0.0, 0.0] }); // self-referential, zero weight ⇒ no flow
        Var { v, i, tape: self.clone() }
    }

    /// A constant (no gradient flows to it): same as a leaf, callers just never read its grad.
    pub fn constant(&self, v: f64) -> Var {
        self.var(v)
    }

    fn push(&self, v: f64, p0: usize, w0: f64, p1: usize, w1: f64) -> Var {
        let mut n = self.nodes.borrow_mut();
        let i = n.len();
        n.push(Node { parent: [p0, p1], weight: [w0, w1] });
        Var { v, i, tape: self.clone() }
    }

    /// Reverse pass: gradient of `y` w.r.t. every tape node (index with [`Var::index`]).
    pub fn grad(&self, y: &Var) -> Vec<f64> {
        let n = self.nodes.borrow();
        let mut g = vec![0.0; n.len()];
        g[y.i] = 1.0;
        for i in (0..n.len()).rev() {
            let gi = g[i];
            if gi == 0.0 {
                continue;
            }
            let node = n[i];
            // skip the self-referential leaf edge
            if node.parent[0] != i {
                g[node.parent[0]] += gi * node.weight[0];
            }
            if node.parent[1] != i {
                g[node.parent[1]] += gi * node.weight[1];
            }
        }
        g
    }
}

impl Var {
    /// This variable's index into its tape (use with the vector returned by [`Tape::grad`]).
    pub fn index(&self) -> usize {
        self.i
    }
    pub fn ln(&self) -> Var {
        self.tape.push(self.v.ln(), self.i, 1.0 / self.v, self.i, 0.0)
    }
    pub fn sqrt(&self) -> Var {
        let r = self.v.sqrt();
        self.tape.push(r, self.i, 0.5 / r.max(1e-300), self.i, 0.0)
    }
    pub fn sq(&self) -> Var {
        self.tape.push(self.v * self.v, self.i, 2.0 * self.v, self.i, 0.0)
    }
    pub fn recip(&self) -> Var {
        self.tape.push(1.0 / self.v, self.i, -1.0 / (self.v * self.v), self.i, 0.0)
    }
    /// Scale by a constant.
    pub fn scale(&self, c: f64) -> Var {
        self.tape.push(self.v * c, self.i, c, self.i, 0.0)
    }
    /// Add a constant.
    pub fn shift(&self, c: f64) -> Var {
        self.tape.push(self.v + c, self.i, 1.0, self.i, 0.0)
    }
}

impl std::ops::Add for &Var {
    type Output = Var;
    fn add(self, rhs: &Var) -> Var {
        self.tape.push(self.v + rhs.v, self.i, 1.0, rhs.i, 1.0)
    }
}
impl std::ops::Sub for &Var {
    type Output = Var;
    fn sub(self, rhs: &Var) -> Var {
        self.tape.push(self.v - rhs.v, self.i, 1.0, rhs.i, -1.0)
    }
}
impl std::ops::Mul for &Var {
    type Output = Var;
    fn mul(self, rhs: &Var) -> Var {
        self.tape.push(self.v * rhs.v, self.i, rhs.v, rhs.i, self.v)
    }
}

/// A 2-D vector of tape variables (a soft-body node position or velocity).
type V2 = [Var; 2];

fn v_add(a: &V2, b: &V2) -> V2 {
    [&a[0] + &b[0], &a[1] + &b[1]]
}
fn v_sub(a: &V2, b: &V2) -> V2 {
    [&a[0] - &b[0], &a[1] - &b[1]]
}
fn v_norm(a: &V2) -> Var {
    let s = &a[0].sq() + &a[1].sq();
    s.sqrt()
}

/// A differentiable 2-D mass-spring soft body: point masses connected by Hookean springs, integrated with
/// semi-implicit Euler. Its whole rollout is differentiable via the tape, so a terminal loss can be
/// pushed back to a per-step control by the adjoint method.
#[derive(Clone, Debug)]
pub struct SoftBody {
    /// Initial node positions.
    pub x0: Vec<[f64; 2]>,
    /// Springs as `(i, j, rest_length, stiffness)`.
    pub springs: Vec<(usize, usize, f64, f64)>,
    pub mass: f64,
    pub dt: f64,
    pub gravity: [f64; 2],
    /// Ground plane at `y = floor` with a soft penalty stiffness (0 ⇒ no ground).
    pub floor: f64,
    pub floor_k: f64,
}

/// Result of an adjoint solve: the loss and the gradient w.r.t. every control (`steps × 2`).
#[derive(Clone, Debug)]
pub struct AdjointResult {
    pub loss: f64,
    /// `grad[t] = ∂loss/∂u_t` (a 2-vector per step).
    pub grad: Vec<[f64; 2]>,
    /// Final center of mass.
    pub com: [f64; 2],
}

impl SoftBody {
    fn n(&self) -> usize {
        self.x0.len()
    }

    /// Roll the body forward under a per-step uniform body-force control `u` on a fresh tape, returning the
    /// tape, the terminal loss `‖com_T − target‖²`, the control leaf variables (to read their gradients),
    /// and the numeric final center of mass. Reused for both the loss value and the adjoint.
    fn rollout(&self, u: &[[f64; 2]], target: [f64; 2]) -> (Tape, Var, Vec<[Var; 2]>, [f64; 2]) {
        let tape = Tape::new();
        let n = self.n();
        let inv_m = 1.0 / self.mass;
        let uc: Vec<[Var; 2]> = u.iter().map(|ut| [tape.var(ut[0]), tape.var(ut[1])]).collect();
        let mut x: Vec<V2> = self.x0.iter().map(|p| [tape.var(p[0]), tape.var(p[1])]).collect();
        let mut v: Vec<V2> = (0..n).map(|_| [tape.var(0.0), tape.var(0.0)]).collect();

        for ut in &uc {
            // spring forces: F_i += −k(‖d‖ − L)·d/‖d‖ with d = x_i − x_j (fully differentiable)
            let mut f: Vec<V2> = (0..n).map(|_| [tape.constant(0.0), tape.constant(0.0)]).collect();
            for &(i, j, l, k) in &self.springs {
                let d = v_sub(&x[i], &x[j]);
                let len = v_norm(&d);
                let stretch = len.shift(-l);
                let inv_len = len.recip();
                let coeff = &stretch.scale(-k) * &inv_len;
                let fi = [&coeff * &d[0], &coeff * &d[1]];
                f[i] = v_add(&f[i], &fi);
                f[j] = v_sub(&f[j], &fi);
            }
            // one-sided ground penalty (kept off, floor_k=0, for the free-space reach demo)
            if self.floor_k > 0.0 {
                for (node, xn) in x.iter().enumerate() {
                    if xn[1].v < self.floor {
                        let pen = xn[1].shift(-self.floor).scale(-self.floor_k);
                        f[node][1] = &f[node][1] + &pen;
                    }
                }
            }
            // semi-implicit Euler: v += dt·(F/m + g + u); x += dt·v
            for ((vn, xn), fnode) in v.iter_mut().zip(x.iter_mut()).zip(f.iter()) {
                let ax = &fnode[0].scale(inv_m).shift(self.gravity[0]) + &ut[0];
                let ay = &fnode[1].scale(inv_m).shift(self.gravity[1]) + &ut[1];
                *vn = [&vn[0] + &ax.scale(self.dt), &vn[1] + &ay.scale(self.dt)];
                *xn = [&xn[0] + &vn[0].scale(self.dt), &xn[1] + &vn[1].scale(self.dt)];
            }
        }

        // loss = ‖com − target‖²
        let inv_n = 1.0 / n as f64;
        let mut cx = tape.constant(0.0);
        let mut cy = tape.constant(0.0);
        for xn in &x {
            cx = &cx + &xn[0];
            cy = &cy + &xn[1];
        }
        let cx = cx.scale(inv_n);
        let cy = cy.scale(inv_n);
        let dx = cx.shift(-target[0]);
        let dy = cy.shift(-target[1]);
        let loss = &dx.sq() + &dy.sq();
        let com = [cx.v, cy.v];
        (tape, loss, uc, com)
    }

    /// The terminal loss for a control (forward only).
    pub fn loss(&self, u: &[[f64; 2]], target: [f64; 2]) -> f64 {
        self.rollout(u, target).1.v
    }

    /// The **adjoint** gradient: loss and `∂loss/∂u_t` for every step, in one reverse pass.
    pub fn adjoint(&self, u: &[[f64; 2]], target: [f64; 2]) -> AdjointResult {
        let (tape, loss, uc, com) = self.rollout(u, target);
        let g = tape.grad(&loss);
        let grad = uc.iter().map(|c| [g[c[0].index()], g[c[1].index()]]).collect();
        AdjointResult { loss: loss.v, grad, com }
    }

    /// Gradient descent on the per-step control to drive the body's terminal COM to `target`. Returns the
    /// optimized control and the loss history.
    pub fn optimize(&self, steps: usize, target: [f64; 2], iters: usize, lr: f64) -> (Vec<[f64; 2]>, Vec<f64>) {
        let mut u = vec![[0.0, 0.0]; steps];
        let mut hist = Vec::with_capacity(iters);
        for _ in 0..iters {
            let r = self.adjoint(&u, target);
            hist.push(r.loss);
            for (ut, gt) in u.iter_mut().zip(r.grad.iter()) {
                ut[0] -= lr * gt[0];
                ut[1] -= lr * gt[1];
            }
        }
        hist.push(self.loss(&u, target));
        (u, hist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small deformable quad: 4 corner masses + cross-braced springs.
    fn quad() -> SoftBody {
        let x0 = vec![[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]];
        let l = 0.2;
        let d = (2.0f64).sqrt() * 0.2;
        let k = 40.0;
        let springs = vec![
            (0, 1, l, k), (1, 2, l, k), (2, 3, l, k), (3, 0, l, k), // edges
            (0, 2, d, k), (1, 3, d, k), // diagonals (shear stiffness)
        ];
        SoftBody { x0, springs, mass: 1.0, dt: 0.02, gravity: [0.0, 0.0], floor: -1.0, floor_k: 0.0 }
    }

    #[test]
    fn the_tape_gradient_matches_finite_differences_on_a_composite_function() {
        // f(a,b) = ln(a)·b + sqrt(a·b) − (a−b)² ; check ∂f both ways.
        let t = Tape::new();
        let (av, bv) = (1.7, 0.9);
        let a = t.var(av);
        let b = t.var(bv);
        let ab = &a * &b;
        let f = &(&(&a.ln() * &b) + &ab.sqrt()) - &(&a - &b).sq();
        let g = t.grad(&f);
        let (ga, gb) = (g[a.index()], g[b.index()]);
        let eval = |a: f64, b: f64| a.ln() * b + (a * b).sqrt() - (a - b).powi(2);
        let eps = 1e-6;
        let fda = (eval(av + eps, bv) - eval(av - eps, bv)) / (2.0 * eps);
        let fdb = (eval(av, bv + eps) - eval(av, bv - eps)) / (2.0 * eps);
        assert!((ga - fda).abs() < 1e-7 && (gb - fdb).abs() < 1e-7, "tape grad ({ga},{gb}) vs fd ({fda},{fdb})");
    }

    #[test]
    fn the_adjoint_control_gradient_matches_finite_differences() {
        // THE INVARIANT. The reverse-mode ∂loss/∂u_t through the whole rollout must equal central finite
        // differences of a forward-only resimulation — the guarantee that the trajectory adjoint is exact.
        let body = quad();
        let target = [0.6, 0.35];
        let steps = 12;
        // a nontrivial control
        let u: Vec<[f64; 2]> = (0..steps).map(|t| [0.3 * ((t as f64) * 0.5).cos(), 0.2 * ((t as f64) * 0.3).sin()]).collect();
        let r = body.adjoint(&u, target);
        let eps = 1e-6;
        for t in [0usize, 4, 9] {
            for c in 0..2 {
                let mut up = u.clone();
                let mut um = u.clone();
                up[t][c] += eps;
                um[t][c] -= eps;
                let fd = (body.loss(&up, target) - body.loss(&um, target)) / (2.0 * eps);
                assert!((r.grad[t][c] - fd).abs() < 1e-6, "grad[{t}][{c}] adjoint {} vs fd {fd}", r.grad[t][c]);
            }
        }
    }

    #[test]
    fn gradient_descent_drives_the_soft_body_to_the_target() {
        // THE HEADLINE. Starting from zero control, the adjoint gradient discovers a body-force sequence
        // that moves the deformable body's center of mass onto a target — differentiable-physics control.
        let body = quad();
        let target = [0.7, 0.4];
        let steps = 15;
        let (u, hist) = body.optimize(steps, target, 4000, 150.0);
        let start_loss = hist[0];
        let end_loss = *hist.last().unwrap();
        assert!(end_loss < start_loss * 1e-3, "loss should fall by >1000×: {start_loss} → {end_loss}");
        let com = body.adjoint(&u, target).com;
        assert!((com[0] - target[0]).abs() < 1e-2 && (com[1] - target[1]).abs() < 1e-2, "COM should reach target: {com:?}");
    }

    #[test]
    fn the_loss_decreases_monotonically_under_descent() {
        let body = quad();
        let (_, hist) = body.optimize(12, [0.5, 0.3], 120, 0.4);
        for w in hist.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "loss rose under gradient descent: {} → {}", w[0], w[1]);
        }
    }
}
