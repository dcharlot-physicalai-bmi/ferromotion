//! **Incremental Potential Contact (IPC)** — the barrier-based, guaranteed intersection-free contact
//! model of Li et al., *Incremental Potential Contact: Intersection- and Inversion-free Large
//! Deformation Dynamics* (SIGGRAPH 2020). The basis of robust deformable/rigid contact (Taccel, GRIP,
//! Diffclothai).
//!
//! Contact is a smooth **log-barrier** potential on the distance `d` between primitives:
//! `b(d) = −(d − d̂)² ln(d/d̂)` for `0 < d < d̂` (and 0 beyond `d̂`). As `d → 0` the barrier → ∞, so a
//! finite-energy trajectory can never touch, let alone penetrate — contact is intersection-free *by
//! construction*, with no impulses or LCP. A time step is the unconstrained minimization of the
//! incremental potential (inertia + external + barrier), and a **filtered line search** caps every
//! step so no distance crosses zero — the guarantee holds even under stiff, fast impact. The barrier
//! is `C²` on `(0, d̂)`, so the whole model is differentiable. Pure `nalgebra` → WASM-clean.

/// IPC log-barrier `b(d)` with support `d̂` (`0` for `d ≥ d̂`).
pub fn barrier(d: f64, dhat: f64) -> f64 {
    if d <= 0.0 {
        f64::INFINITY
    } else if d < dhat {
        let s = d - dhat;
        -s * s * (d / dhat).ln()
    } else {
        0.0
    }
}

/// `b'(d)`.
pub fn barrier_grad(d: f64, dhat: f64) -> f64 {
    if d <= 0.0 || d >= dhat {
        0.0
    } else {
        let s = d - dhat;
        -2.0 * s * (d / dhat).ln() - s * s / d
    }
}

/// `b''(d)` (always ≥ 0 on `(0, d̂)` — the barrier is convex near contact).
pub fn barrier_hess(d: f64, dhat: f64) -> f64 {
    if d <= 0.0 || d >= dhat {
        0.0
    } else {
        let s = d - dhat;
        -2.0 * (d / dhat).ln() - 4.0 * s / d + s * s / (d * d)
    }
}

/// A set of masses over a floor at `z = 0`, resolved with IPC. Each mass's contact distance is its
/// height `z`; a step minimizes the incremental potential
/// `½ m (z − z̃)²/dt² + κ·b(z)` (with `z̃` the inertia+gravity prediction) by filtered Newton.
#[derive(Clone, Debug)]
pub struct IpcFloor {
    pub z: Vec<f64>,
    pub v: Vec<f64>,
    pub mass: f64,
    pub dhat: f64,
    pub kappa: f64,
    pub dt: f64,
    pub gravity: f64,
}

impl IpcFloor {
    /// Advance one step. Guarantees every `z` stays strictly positive (intersection-free).
    pub fn step(&mut self) {
        let (m, dt, k, dhat) = (self.mass, self.dt, self.kappa, self.dhat);
        for i in 0..self.z.len() {
            let z_n = self.z[i];
            let z_tilde = z_n + dt * self.v[i] - dt * dt * self.gravity; // inertia + gravity prediction
            let mut z = z_n.max(1e-9);
            // Newton on E(z) = m/(2dt²)(z − z̃)² + κ·b(z), with a filtered line search.
            for _ in 0..50 {
                let g = m / (dt * dt) * (z - z_tilde) + k * barrier_grad(z, dhat);
                let h = m / (dt * dt) + k * barrier_hess(z, dhat);
                let step = -g / h;
                if step.abs() < 1e-12 {
                    break;
                }
                // Filter: never let z reach the floor (cap α so z stays ≥ 0.9·(distance to 0)).
                let mut alpha: f64 = 1.0;
                if step < 0.0 {
                    alpha = alpha.min(-0.9 * z / step);
                }
                // Backtracking line search on E.
                let e = |zz: f64| m / (2.0 * dt * dt) * (zz - z_tilde).powi(2) + k * barrier(zz, dhat);
                let e0 = e(z);
                while alpha > 1e-12 && e(z + alpha * step) > e0 {
                    alpha *= 0.5;
                }
                z += alpha * step;
            }
            self.v[i] = (z - z_n) / dt;
            self.z[i] = z;
        }
    }

    /// Smallest height across all masses (must stay > 0).
    pub fn min_height(&self) -> f64 {
        self.z.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barrier_derivatives_match_finite_difference() {
        let dhat = 0.05;
        let eps = 1e-7;
        for &d in &[0.005, 0.01, 0.02, 0.04, 0.049] {
            let fdg = (barrier(d + eps, dhat) - barrier(d - eps, dhat)) / (2.0 * eps);
            assert!((barrier_grad(d, dhat) - fdg).abs() < 1e-3, "b'({d}): {} vs fd {fdg}", barrier_grad(d, dhat));
            let fdh = (barrier_grad(d + eps, dhat) - barrier_grad(d - eps, dhat)) / (2.0 * eps);
            assert!((barrier_hess(d, dhat) - fdh).abs() < 1e-2, "b''({d}): {} vs fd {fdh}", barrier_hess(d, dhat));
        }
        // The barrier diverges at contact and vanishes past d̂.
        assert!(barrier(1e-4, dhat) > barrier(0.02, dhat) && barrier(0.02, dhat) > 0.0);
        assert_eq!(barrier(0.06, dhat), 0.0);
    }

    #[test]
    fn mass_settles_above_the_floor_without_penetrating() {
        let mut f = IpcFloor { z: vec![0.5], v: vec![0.0], mass: 1.0, dhat: 0.05, kappa: 1e5, dt: 0.01, gravity: 9.81 };
        let mut worst = f64::INFINITY;
        for _ in 0..2000 {
            f.step();
            worst = worst.min(f.min_height());
        }
        assert!(worst > 0.0, "penetrated the floor: min height {worst}");
        // Settles inside the barrier layer, at rest, where κ·b'(z*) balances gravity.
        assert!(f.z[0] > 0.0 && f.z[0] < f.dhat, "did not rest in contact: z = {}", f.z[0]);
        assert!(f.v[0].abs() < 1e-3, "did not settle: v = {}", f.v[0]);
    }

    #[test]
    fn stiff_fast_impact_stays_intersection_free() {
        // The IPC guarantee: a fast mass and a large timestep still never penetrate.
        let mut f = IpcFloor { z: vec![0.3], v: vec![-80.0], mass: 1.0, dhat: 0.05, kappa: 1e5, dt: 0.05, gravity: 9.81 };
        for _ in 0..500 {
            f.step();
            assert!(f.min_height() > 0.0, "penetrated under stiff impact: z = {}", f.min_height());
        }
    }
}
