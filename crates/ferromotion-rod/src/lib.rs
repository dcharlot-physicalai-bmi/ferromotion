//! ferromotion-rod — **Discrete Elastic Rods** (Bergou et al., SIGGRAPH 2008) for cables, tendons, and
//! continuum/soft robots — the 1D-deformable companion to the cloth (2D) and MPM (volumetric) solvers.
//!
//! A rod is a polyline centerline `x₀…x_n`. Two elastic energies act on it: **stretch** (edge springs
//! to rest length) and **isotropic bending**, the discrete curvature living on the curvature binormal
//! `κb_i = 2 e_{i-1}×e_i / (‖e_{i-1}‖‖e_i‖ + e_{i-1}·e_i)` at each interior vertex, with energy
//! `E_i = (EI / 2ℓ_i)‖κb_i‖²` — the discretization that reproduces the continuum `½∫EI κ² ds`. Nodal
//! forces are the exact analytic gradient of the energy (verified against finite differences), so the
//! rod is differentiable. Validated against the analytic **Euler-Bernoulli cantilever**. (Twist via
//! material frames is a further extension.) Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// A discrete elastic rod.
#[derive(Clone, Debug)]
pub struct Rod {
    pub x: Vec<Vector3<f64>>,
    pub v: Vec<Vector3<f64>>,
    pub rest_len: Vec<f64>,
    /// Mass per vertex.
    pub mass: f64,
    /// Stretch stiffness.
    pub ks: f64,
    /// Bending modulus `EI`.
    pub ei: f64,
    pub gravity: Vector3<f64>,
    /// Fixed (clamped) vertices — their forces are zeroed.
    pub clamped: Vec<bool>,
}

impl Rod {
    /// A straight rod of `nseg` segments along +x, length `len`, clamped at the first two vertices.
    pub fn straight(nseg: usize, len: f64, mass: f64, ks: f64, ei: f64, gravity: Vector3<f64>) -> Self {
        let h = len / nseg as f64;
        let x: Vec<_> = (0..=nseg).map(|i| Vector3::new(h * i as f64, 0.0, 0.0)).collect();
        let rest_len = vec![h; nseg];
        let mut clamped = vec![false; nseg + 1];
        clamped[0] = true;
        clamped[1] = true;
        Self { x, v: vec![Vector3::zeros(); nseg + 1], rest_len, mass, ks, ei, gravity, clamped }
    }

    /// Total potential energy (stretch + bending + gravity) of a configuration.
    pub fn energy(&self, x: &[Vector3<f64>]) -> f64 {
        let mut e = 0.0;
        for j in 0..self.rest_len.len() {
            let len = (x[j + 1] - x[j]).norm();
            e += 0.5 * self.ks * (len - self.rest_len[j]).powi(2);
        }
        for i in 1..x.len() - 1 {
            let (e0, e1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
            let l_i = 0.5 * (self.rest_len[i - 1] + self.rest_len[i]); // rest Voronoi length (const weight)
            e += self.ei / (2.0 * l_i) * kb(e0, e1).norm_squared();
        }
        for xi in x {
            e -= self.mass * self.gravity.dot(xi); // gravity PE = −m g·x
        }
        e
    }

    /// Net force on every vertex — the exact analytic `−∇energy`.
    pub fn forces(&self, x: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = x.len();
        let mut g = vec![Vector3::zeros(); n]; // ∇energy
        // Stretch.
        for j in 0..self.rest_len.len() {
            let e = x[j + 1] - x[j];
            let len = e.norm();
            let dir = e / len;
            let s = self.ks * (len - self.rest_len[j]);
            g[j] -= s * dir;
            g[j + 1] += s * dir;
        }
        // Bending.
        for i in 1..n - 1 {
            let (e0, e1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
            let l_i = 0.5 * (self.rest_len[i - 1] + self.rest_len[i]); // rest Voronoi length (const weight)
            let kbv = kb(e0, e1);
            let g_kb = (self.ei / l_i) * kbv; // ∂E_i/∂κb
            let (dde0, dde1) = kb_grads(e0, e1);
            // ∂x_{i-1} = −∂e0, ∂x_{i+1} = ∂e1, ∂x_i = ∂e0 − ∂e1.
            g[i - 1] += (-dde0).transpose() * g_kb;
            g[i + 1] += dde1.transpose() * g_kb;
            g[i] += (dde0 - dde1).transpose() * g_kb;
        }
        // Gravity.
        for gi in g.iter_mut() {
            *gi -= self.mass * self.gravity;
        }
        // Force = −∇energy, zeroed on clamped vertices.
        (0..n).map(|i| if self.clamped[i] { Vector3::zeros() } else { -g[i] }).collect()
    }

    /// One damped semi-implicit step (for dynamics / relaxation).
    pub fn step(&mut self, dt: f64, damping: f64) {
        let f = self.forces(&self.x);
        for i in 0..self.x.len() {
            if self.clamped[i] {
                continue;
            }
            self.v[i] += dt * f[i] / self.mass;
            self.v[i] *= damping;
            self.x[i] += dt * self.v[i];
        }
    }

    /// Relax to static equilibrium (heavy damping); returns the max residual force.
    pub fn relax(&mut self, iters: usize, dt: f64) -> f64 {
        for _ in 0..iters {
            self.step(dt, 0.9);
        }
        self.forces(&self.x).iter().map(|f| f.norm()).fold(0.0, f64::max)
    }

    /// Kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.v.iter().map(|v| v.norm_squared()).sum::<f64>()
    }
}

/// Curvature binormal at a vertex with incoming/outgoing edges `e0, e1`.
fn kb(e0: Vector3<f64>, e1: Vector3<f64>) -> Vector3<f64> {
    let d = e0.norm() * e1.norm() + e0.dot(&e1);
    2.0 * e0.cross(&e1) / d
}

/// `(∂κb/∂e0, ∂κb/∂e1)` as 3×3 matrices.
fn kb_grads(e0: Vector3<f64>, e1: Vector3<f64>) -> (Matrix3<f64>, Matrix3<f64>) {
    let (n0, n1) = (e0.norm(), e1.norm());
    let cr = e0.cross(&e1);
    let d = n0 * n1 + e0.dot(&e1);
    let dd_de0 = (n1 / n0) * e0 + e1; // ∂d/∂e0
    let dd_de1 = (n0 / n1) * e1 + e0; // ∂d/∂e1
    let dde0 = (2.0 / d) * (-skew(e1)) - (2.0 / (d * d)) * cr * dd_de0.transpose();
    let dde1 = (2.0 / d) * skew(e0) - (2.0 / (d * d)) * cr * dd_de1.transpose();
    (dde0, dde1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forces_are_the_exact_energy_gradient() {
        // A wiggly rod: analytic forces must equal −∇energy (finite differences).
        let mut rod = Rod::straight(8, 1.0, 0.01, 500.0, 0.1, Vector3::new(0.0, 0.0, -9.81));
        for (i, xi) in rod.x.iter_mut().enumerate() {
            xi.z += 0.05 * (i as f64 * 0.9).sin();
            xi.y += 0.03 * (i as f64 * 0.6).cos();
        }
        let x = rod.x.clone();
        let f = rod.forces(&x);
        let eps = 1e-6;
        for i in 0..x.len() {
            if rod.clamped[i] {
                continue;
            }
            for d in 0..3 {
                let (mut xp, mut xm) = (x.clone(), x.clone());
                xp[i][d] += eps;
                xm[i][d] -= eps;
                let fd = -(rod.energy(&xp) - rod.energy(&xm)) / (2.0 * eps);
                assert!((f[i][d] - fd).abs() < 1e-4, "force[{i}][{d}]: analytic {} vs fd {fd}", f[i][d]);
            }
        }
    }

    #[test]
    fn cantilever_matches_euler_bernoulli() {
        // A clamped rod sagging under self-weight; small-deflection tip deflection δ = w·L⁴/(8·EI).
        let (nseg, len, ei) = (20, 1.0, 0.5);
        let m = 0.001; // mass per vertex
        let g = 9.81;
        // ks/dt chosen for explicit stability: the bending mode ω ~ √(EI/(m·ℓ³)) is the stiff limiter.
        let mut rod = Rod::straight(nseg, len, m, 200.0, ei, Vector3::new(0.0, 0.0, -g));
        let residual = rod.relax(200000, 1e-4);
        assert!(residual < 1e-4, "did not reach equilibrium: residual {residual}");

        let w = m * nseg as f64 * g / len; // weight per unit length
        let expected = w * len.powi(4) / (8.0 * ei);
        let tip = -rod.x[nseg].z; // downward deflection
        let rel = (tip - expected).abs() / expected;
        eprintln!("cantilever: tip={tip:.5}, Euler-Bernoulli={expected:.5}, rel_err={rel:.3}");
        assert!(rel < 0.08, "cantilever deflection off Euler-Bernoulli by {rel:.3} (tip {tip}, EB {expected})");
    }

    #[test]
    fn free_vibration_conserves_energy() {
        // Undamped, no gravity: a plucked rod's total energy stays bounded/constant.
        let mut rod = Rod::straight(10, 1.0, 0.02, 200.0, 0.05, Vector3::zeros());
        // Clamp only the first vertex (free elsewhere).
        rod.clamped = vec![false; rod.x.len()];
        rod.clamped[0] = true;
        // Pluck: displace the free end.
        let last = rod.x.len() - 1;
        rod.x[last].z += 0.1;
        let e0 = rod.energy(&rod.x) + rod.kinetic_energy();
        let (dt, mut worst) = (2e-4, 0.0f64);
        for _ in 0..3000 {
            rod.step(dt, 1.0); // no damping
            let e = rod.energy(&rod.x) + rod.kinetic_energy();
            worst = worst.max((e - e0).abs() / e0.abs().max(1e-9));
        }
        assert!(worst < 5e-2, "energy not conserved: worst relative drift {worst}");
    }
}
