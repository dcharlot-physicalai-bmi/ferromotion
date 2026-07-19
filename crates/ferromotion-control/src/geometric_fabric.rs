//! **Geometric Fabrics** (Ratliff, Van Wyk, Xie, Fox et al., "Optimization Fabrics" / "Generalized
//! Nonlinear Geometry", 2020–2021) — the successor to RMPflow ([`crate::RmpArm`]) for reactive motion
//! generation. The core object is a **geometry**: a second-order policy `ẍ = h(x, ẋ)` whose acceleration is
//! **homogeneous of degree 2 (HD2)** in velocity, `h(x, αẋ) = α² h(x, ẋ)`. HD2 has a remarkable
//! consequence — the integral curves (the *paths*) are **invariant to the speed of traversal**: run the
//! system twice as fast and it draws the exact same geometric path. This is the property RMPs lack and the
//! reason fabrics give predictable, provably-consistent behavior.
//!
//! A geometry is made well-behaved for combination by **energization**: given an energy `Lₑ = ½ ẋᵀ Mₑ ẋ`,
//! the energized geometry `ẍ = h + αₑ ẋ` with `αₑ = −(ẋᵀ Mₑ h)/(ẋᵀ Mₑ ẋ)` **conserves that energy while
//! preserving the path** — because the correction is parallel to `ẋ`, it only reparametrizes time. A
//! complete fabric then *forces* the energized geometry with an attractor potential and damping,
//! `ẍ = hₑ − ∂ψ − β ẋ`, which converges to the goal.
//!
//! Here: a planar point-robot obstacle-avoidance fabric (a radial barrier geometry, `Mₑ = I`). Verified: the
//! geometry is numerically HD2; the pure-geometry closest approach to the obstacle is invariant to initial
//! speed (the defining fabric property); energization conserves energy to integrator precision; and the
//! forced fabric reaches the goal while keeping clearance from the obstacle. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector2;

/// A planar geometric fabric: an obstacle at `obstacle` with barrier gain `k_obs`, an attractor of stiffness
/// `k_goal`, and damping `beta`. The obstacle geometry is `h = (k_obs / d²) ‖ẋ‖² n̂` (radial, HD2 in `ẋ`).
#[derive(Clone, Copy, Debug)]
pub struct GeometricFabric {
    pub obstacle: Vector2<f64>,
    pub k_obs: f64,
    pub k_goal: f64,
    pub beta: f64,
}

impl GeometricFabric {
    /// The bare obstacle geometry `h(x, ẋ)` — HD2 in `ẋ`.
    pub fn geometry(&self, x: &Vector2<f64>, xd: &Vector2<f64>) -> Vector2<f64> {
        let diff = x - self.obstacle;
        let d = diff.norm().max(1e-6);
        let n = diff / d; // unit vector pointing away from the obstacle
        (self.k_obs / (d * d)) * xd.norm_squared() * n
    }

    /// Energization coefficient `αₑ = −(ẋᵀ Mₑ h)/(ẋᵀ Mₑ ẋ)` with `Mₑ = I`. Adds a component parallel to `ẋ`,
    /// so the path is unchanged but the energy `½‖ẋ‖²` is conserved.
    fn alpha_e(xd: &Vector2<f64>, h: &Vector2<f64>) -> f64 {
        let denom = xd.norm_squared();
        if denom < 1e-12 {
            0.0
        } else {
            -xd.dot(h) / denom
        }
    }

    /// The energized geometry acceleration `hₑ = h + αₑ ẋ` (path-preserving, energy-conserving).
    pub fn energized(&self, x: &Vector2<f64>, xd: &Vector2<f64>) -> Vector2<f64> {
        let h = self.geometry(x, xd);
        h + Self::alpha_e(xd, &h) * xd
    }

    /// The full forced-fabric acceleration `hₑ − k_goal (x − goal) − β ẋ` (attractor potential + damping).
    pub fn forced(&self, x: &Vector2<f64>, xd: &Vector2<f64>, goal: &Vector2<f64>) -> Vector2<f64> {
        self.energized(x, xd) - self.k_goal * (x - goal) - self.beta * xd
    }

    /// RK4 rollout of the **pure energized geometry** (no forcing). Returns the path and the closest approach
    /// to the obstacle. Because the geometry is HD2, this closest approach is invariant to `‖ẋ0‖`.
    pub fn rollout_geometry(&self, x0: Vector2<f64>, xd0: Vector2<f64>, dt: f64, steps: usize) -> (Vec<Vector2<f64>>, f64) {
        let accel = |x: &Vector2<f64>, xd: &Vector2<f64>| self.energized(x, xd);
        self.integrate(x0, xd0, dt, steps, accel)
    }

    /// RK4 rollout of the **forced fabric** toward `goal`. Returns the path and the closest approach to the
    /// obstacle.
    pub fn rollout_forced(&self, x0: Vector2<f64>, xd0: Vector2<f64>, goal: Vector2<f64>, dt: f64, steps: usize) -> (Vec<Vector2<f64>>, f64) {
        let accel = |x: &Vector2<f64>, xd: &Vector2<f64>| self.forced(x, xd, &goal);
        self.integrate(x0, xd0, dt, steps, accel)
    }

    fn integrate(&self, x0: Vector2<f64>, xd0: Vector2<f64>, dt: f64, steps: usize, accel: impl Fn(&Vector2<f64>, &Vector2<f64>) -> Vector2<f64>) -> (Vec<Vector2<f64>>, f64) {
        let mut x = x0;
        let mut xd = xd0;
        let mut path = vec![x];
        let mut min_d = (x - self.obstacle).norm();
        for _ in 0..steps {
            // RK4 on the state (x, xd)
            let (k1x, k1v) = (xd, accel(&x, &xd));
            let (k2x, k2v) = (xd + 0.5 * dt * k1v, accel(&(x + 0.5 * dt * k1x), &(xd + 0.5 * dt * k1v)));
            let (k3x, k3v) = (xd + 0.5 * dt * k2v, accel(&(x + 0.5 * dt * k2x), &(xd + 0.5 * dt * k2v)));
            let (k4x, k4v) = (xd + dt * k3v, accel(&(x + dt * k3x), &(xd + dt * k3v)));
            x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            xd += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
            path.push(x);
            min_d = min_d.min((x - self.obstacle).norm());
        }
        (path, min_d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fab() -> GeometricFabric {
        GeometricFabric { obstacle: Vector2::new(2.5, 0.0), k_obs: 0.6, k_goal: 1.0, beta: 2.0 }
    }

    #[test]
    fn the_geometry_is_homogeneous_of_degree_two() {
        // THE FOUNDATION. h(x, αẋ) = α² h(x, ẋ) for all scalings α.
        let f = fab();
        let x = Vector2::new(1.0, 0.3);
        let xd = Vector2::new(0.7, -0.4);
        let base = f.geometry(&x, &xd);
        for &a in &[0.5, 2.0, 3.7] {
            let scaled = f.geometry(&x, &(a * xd));
            assert!((scaled - a * a * base).norm() < 1e-12, "HD2 violated at α={a}: {scaled} vs {}", a * a * base);
        }
    }

    #[test]
    fn the_closest_approach_is_invariant_to_traversal_speed() {
        // THE DEFINING PROPERTY. The pure geometry drives the SAME geometric path regardless of speed, so the
        // closest approach to the obstacle is identical whether we start slow or fast — only the time to
        // traverse it changes. (We compensate dt by the speed factor so both cover the same arc.)
        let f = fab();
        let x0 = Vector2::new(0.0, 0.35);
        let dir = Vector2::new(1.0, 0.0);
        let (_p_slow, d_slow) = f.rollout_geometry(x0, dir, 0.002, 6000);
        let (_p_fast, d_fast) = f.rollout_geometry(x0, 3.0 * dir, 0.002 / 3.0, 6000);
        assert!((d_slow - d_fast).abs() < 1e-3, "closest approach must be speed-invariant: {d_slow} vs {d_fast}");
        // and the geometry actually bends away — it clears the obstacle by a margin the straight line wouldn't
        assert!(d_slow > 0.35 + 1e-3, "the fabric should push away from the obstacle: {d_slow}");
    }

    #[test]
    fn energization_conserves_energy() {
        // Energization adds a term parallel to ẋ, chosen so ½‖ẋ‖² is conserved along the pure geometry.
        let f = fab();
        let x0 = Vector2::new(0.0, 0.35);
        let xd0 = Vector2::new(1.2, 0.0);
        let e0 = 0.5 * xd0.norm_squared();
        // integrate and track energy via re-deriving speed from consecutive path points
        let dt = 0.001;
        let mut x = x0;
        let mut xd = xd0;
        let mut max_dev: f64 = 0.0;
        for _ in 0..4000 {
            let a = f.energized(&x, &xd);
            // symplectic-ish RK4 step (reuse the struct integrator's math inline for one step)
            let (k1x, k1v) = (xd, a);
            let (k2x, k2v) = (xd + 0.5 * dt * k1v, f.energized(&(x + 0.5 * dt * k1x), &(xd + 0.5 * dt * k1v)));
            let (k3x, k3v) = (xd + 0.5 * dt * k2v, f.energized(&(x + 0.5 * dt * k2x), &(xd + 0.5 * dt * k2v)));
            let (k4x, k4v) = (xd + dt * k3v, f.energized(&(x + dt * k3x), &(xd + dt * k3v)));
            x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            xd += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
            max_dev = max_dev.max((0.5 * xd.norm_squared() - e0).abs());
        }
        assert!(max_dev < 1e-3, "energy should be conserved along the energized geometry: max deviation {max_dev}");
    }

    #[test]
    fn the_forced_fabric_reaches_the_goal_while_avoiding_the_obstacle() {
        // THE HEADLINE / application. Start left, goal on the right directly behind the obstacle. The forced
        // fabric (attractor + damping + obstacle geometry) must reach the goal AND keep clearance.
        let f = fab();
        let start = Vector2::new(0.0, 0.05);
        let goal = Vector2::new(5.0, 0.0);
        let (path, min_d) = f.rollout_forced(start, Vector2::zeros(), goal, 0.01, 4000);
        let end = *path.last().unwrap();
        assert!((end - goal).norm() < 0.05, "should converge to the goal: {end}");
        assert!(min_d > 0.15, "should keep clearance from the obstacle: min distance {min_d}");
    }
}
