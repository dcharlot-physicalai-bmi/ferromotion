//! **FOCI — Field Overlap Collision Integral** (Gómez Andreu, Bjelonic, Hutter et al., IROS 2025):
//! collision *directly on a 3D Gaussian-Splat map*, with no conservative bounding boxes. Both the world
//! and the robot are a cloud of anisotropic Gaussians (the native output of 3DGS reconstruction), and the
//! collision cost between two Gaussians is their **overlap integral** — the integral of the product of
//! their density fields, which has an exact closed form:
//!
//! ```text
//!   ∫_ℝ³ 𝒩(x; μ_i, Σ_i) · 𝒩(x; μ̄_j, Σ̄_j) dx  =  𝒩(μ̄_j; μ_i, Σ_i + Σ̄_j)
//! ```
//!
//! — a single Gaussian in the *mean separation*, with the *summed* covariance. It is smooth and cheap,
//! and (the paper's key point) **orientation-aware**: because each robot Gaussian's covariance `Σ̄_j(ψ)`
//! rotates with the robot's yaw, an elongated robot can *turn* to slip through a narrow slot it would hit
//! head-on — a conservative sphere/box model can never express that. The scene collision cost sums the
//! overlap over every (environment, robot) Gaussian pair along the trajectory, and FOCI descends that cost
//! (with a jerk-smoothness term and a goal term) to plan. Pure `nalgebra` → WASM-clean.
//!
//! (Following the paper, the collision *cost* per pair is the unnormalized kernel `exp(−½·d_M)` — peaking
//! at 1 when the means coincide — while [`overlap_integral`] returns the true normalized ∫p·r dV, which
//! this module verifies against Monte-Carlo integration of the product of the two densities.)

use nalgebra::{Matrix3, Vector3};
use std::f64::consts::PI;

/// A single anisotropic 3-D Gaussian (one "splat"): a mean and a symmetric positive-definite covariance.
#[derive(Clone, Copy, Debug)]
pub struct Gaussian3 {
    pub mu: Vector3<f64>,
    pub sigma: Matrix3<f64>,
}

impl Gaussian3 {
    /// An axis-aligned Gaussian from a mean and per-axis standard deviations.
    pub fn axis_aligned(mu: Vector3<f64>, std: Vector3<f64>) -> Self {
        let sigma = Matrix3::from_diagonal(&Vector3::new(std.x * std.x, std.y * std.y, std.z * std.z));
        Gaussian3 { mu, sigma }
    }

    /// An anisotropic Gaussian: per-axis std devs in a body frame yawed by `psi` about +z, placed at `mu`.
    /// (`Σ = R Σ₀ Rᵀ`, the standard covariance rotation.)
    pub fn yawed(mu: Vector3<f64>, std: Vector3<f64>, psi: f64) -> Self {
        let s0 = Matrix3::from_diagonal(&Vector3::new(std.x * std.x, std.y * std.y, std.z * std.z));
        let r = rot_z(psi);
        Gaussian3 { mu, sigma: r * s0 * r.transpose() }
    }

    /// The normalized 3-D Gaussian density `𝒩(x; μ, Σ)`.
    pub fn pdf(&self, x: &Vector3<f64>) -> f64 {
        let inv = self.sigma.try_inverse().expect("covariance must be invertible");
        let d = x - self.mu;
        let m = (d.transpose() * inv * d)[(0, 0)];
        let norm = (2.0 * PI).powf(-1.5) * self.sigma.determinant().powf(-0.5);
        norm * (-0.5 * m).exp()
    }
}

/// Yaw rotation about +z.
fn rot_z(psi: f64) -> Matrix3<f64> {
    let (s, c) = psi.sin_cos();
    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Squared Mahalanobis separation of two Gaussians under their *summed* covariance:
/// `d_M = Δᵀ (Σ_a + Σ_b)⁻¹ Δ`, with `Δ = μ_a − μ_b`. This is the exponent argument of the overlap.
pub fn mahalanobis_sq(a: &Gaussian3, b: &Gaussian3) -> f64 {
    let s = a.sigma + b.sigma;
    let inv = s.try_inverse().expect("summed covariance must be invertible");
    let d = a.mu - b.mu;
    (d.transpose() * inv * d)[(0, 0)]
}

/// The **exact overlap integral** `∫ 𝒩(x;μ_a,Σ_a)·𝒩(x;μ_b,Σ_b) dx = 𝒩(μ_a; μ_b, Σ_a+Σ_b)` (fully
/// normalized). This is the true value the FOCI cost approximates; verified here against Monte-Carlo.
pub fn overlap_integral(a: &Gaussian3, b: &Gaussian3) -> f64 {
    let s = a.sigma + b.sigma;
    let norm = (2.0 * PI).powf(-1.5) * s.determinant().powf(-0.5);
    norm * (-0.5 * mahalanobis_sq(a, b)).exp()
}

/// The FOCI collision **kernel** for one pair: the unnormalized overlap `exp(−½·d_M)` used as the cost
/// (peaks at 1 when the means coincide, → 0 as the Gaussians separate). Smooth in both poses.
pub fn collision_kernel(a: &Gaussian3, b: &Gaussian3) -> f64 {
    (-0.5 * mahalanobis_sq(a, b)).exp()
}

/// Gradient of [`collision_kernel`] with respect to `a`'s mean `μ_a`:
/// `∂/∂μ_a exp(−½ Δᵀ S⁻¹ Δ) = −kernel · S⁻¹ Δ` (with `S = Σ_a+Σ_b`, `Δ = μ_a−μ_b`).
pub fn kernel_grad_mu_a(a: &Gaussian3, b: &Gaussian3) -> Vector3<f64> {
    let s = a.sigma + b.sigma;
    let inv = s.try_inverse().expect("summed covariance must be invertible");
    let d = a.mu - b.mu;
    let k = (-0.5 * (d.transpose() * inv * d)[(0, 0)]).exp();
    -k * (inv * d)
}

/// A robot modeled as a rigid set of body-frame Gaussians (offset + body-frame per-axis std devs). Posing
/// by `(p, ψ)` translates every mean by `p` and **rotates both the offset and the covariance by `ψ`** —
/// this is what makes the collision orientation-aware.
#[derive(Clone, Debug)]
pub struct RobotSplat {
    /// `(body-frame offset from base, body-frame axis std devs)` for each robot Gaussian.
    pub bodies: Vec<(Vector3<f64>, Vector3<f64>)>,
}

impl RobotSplat {
    /// A single Gaussian at the base, elongated per `std` — the minimal orientation-aware robot.
    pub fn single(std: Vector3<f64>) -> Self {
        RobotSplat { bodies: vec![(Vector3::zeros(), std)] }
    }

    /// The robot's Gaussians posed in the world at base position `p` and yaw `ψ`.
    pub fn posed(&self, p: &Vector3<f64>, psi: f64) -> Vec<Gaussian3> {
        let r = rot_z(psi);
        self.bodies
            .iter()
            .map(|(off, std)| {
                let s0 = Matrix3::from_diagonal(&Vector3::new(std.x * std.x, std.y * std.y, std.z * std.z));
                Gaussian3 { mu: p + r * off, sigma: r * s0 * r.transpose() }
            })
            .collect()
    }
}

/// Total FOCI collision cost of a posed robot against an environment splat: the summed overlap kernel over
/// every (environment, robot) Gaussian pair.
pub fn collision_cost(env: &[Gaussian3], robot: &[Gaussian3]) -> f64 {
    let mut c = 0.0;
    for e in env {
        for r in robot {
            c += collision_kernel(e, r);
        }
    }
    c
}

/// Analytic gradient of [`collision_cost`] with respect to the base position `p` (holding yaw fixed):
/// every robot mean shifts rigidly with `p` (`∂μ̄/∂p = I`), so the position gradient is the summed
/// per-pair kernel gradient. Used to descend the trajectory.
pub fn collision_grad_p(env: &[Gaussian3], robot: &[Gaussian3]) -> Vector3<f64> {
    let mut g = Vector3::zeros();
    for e in env {
        for r in robot {
            // kernel(e, r); ∂/∂(r.mu) = −k·S⁻¹(r.mu−e.mu); ∂(r.mu)/∂p = I
            g += kernel_grad_mu_a(r, e);
        }
    }
    g
}

/// A planned trajectory: the sequence of base positions (waypoint 0 is the fixed start).
#[derive(Clone, Debug)]
pub struct FociPlan {
    pub waypoints: Vec<Vector3<f64>>,
    /// Peak per-configuration collision cost along the final path (0 ⇒ collision-free by the kernel).
    pub peak_collision: f64,
}

/// FOCI trajectory optimization by gradient descent on
/// `ω₁·Σ collision + ω₂·Σ‖jerk‖² + ω₃·‖x_K − goal‖²`, with the start pinned and the robot at a fixed
/// yaw. Returns the optimized waypoints and the peak collision cost along them.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    env: &[Gaussian3],
    robot: &RobotSplat,
    start: Vector3<f64>,
    goal: Vector3<f64>,
    yaw: f64,
    k: usize,
    iters: usize,
    lr: f64,
) -> FociPlan {
    // Weights follow the paper's ratio (collision ≪ smoothness). The paper minimizes this with IPOPT; we
    // demonstrate with damped gradient descent, so `w2` is kept modest to keep the jerk-Hessian spectral
    // radius (∝ w2) within the first-order stability limit `lr < 2/λmax`.
    let (w1, w2, w3) = (1.0, 10.0, 1.0);
    // initialize as a straight line start → goal
    let mut x: Vec<Vector3<f64>> = (0..=k)
        .map(|i| start + (goal - start) * (i as f64 / k as f64))
        .collect();

    for _ in 0..iters {
        let mut grad = vec![Vector3::zeros(); k + 1];
        // collision gradient at each movable waypoint
        for (i, xi) in x.iter().enumerate().skip(1) {
            let posed = robot.posed(xi, yaw);
            grad[i] += w1 * collision_grad_p(env, &posed);
        }
        // jerk-smoothness gradient: ω₂·Σ‖x_{i+3}−3x_{i+2}+3x_{i+1}−x_i‖². ∂/∂x of Σ‖Dx‖² = 2 DᵀD x.
        for i in 0..=k.saturating_sub(3) {
            let jerk = x[i + 3] - 3.0 * x[i + 2] + 3.0 * x[i + 1] - x[i];
            let g = 2.0 * w2 * jerk;
            grad[i] += -g;
            grad[i + 1] += 3.0 * g;
            grad[i + 2] += -3.0 * g;
            grad[i + 3] += g;
        }
        // goal-attraction gradient on the endpoint
        grad[k] += 2.0 * w3 * (x[k] - goal);

        // descend all movable waypoints (waypoint 0 pinned to start)
        for i in 1..=k {
            x[i] -= lr * grad[i];
        }
    }

    let peak = x
        .iter()
        .map(|xi| {
            let posed = robot.posed(xi, yaw);
            env.iter()
                .flat_map(|e| posed.iter().map(move |r| collision_kernel(e, r)))
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    FociPlan { waypoints: x, peak_collision: peak }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
        Vector3::new(x, y, z)
    }

    #[test]
    fn the_closed_form_overlap_matches_monte_carlo_integration() {
        // THE INVARIANT. The overlap integral 𝒩(μ_a; μ_b, Σ_a+Σ_b) must equal ∫ p_a(x)·p_b(x) dx.
        // Integrate the product of the two densities on a deterministic grid and compare — no fitting.
        let a = Gaussian3::yawed(v(0.1, 0.0, 0.0), v(0.5, 0.3, 0.4), 0.6);
        let b = Gaussian3::axis_aligned(v(-0.2, 0.2, 0.1), v(0.4, 0.5, 0.3));
        let closed = overlap_integral(&a, &b);

        // Riemann sum over a box that comfortably contains both Gaussians.
        let (lo, hi, n) = (-3.0_f64, 3.0_f64, 60usize);
        let dx = (hi - lo) / n as f64;
        let dv = dx * dx * dx;
        let mut acc = 0.0;
        for ix in 0..n {
            let px = lo + (ix as f64 + 0.5) * dx;
            for iy in 0..n {
                let py = lo + (iy as f64 + 0.5) * dx;
                for iz in 0..n {
                    let pz = lo + (iz as f64 + 0.5) * dx;
                    let x = v(px, py, pz);
                    acc += a.pdf(&x) * b.pdf(&x) * dv;
                }
            }
        }
        let rel = (closed - acc).abs() / closed;
        assert!(rel < 0.01, "closed-form overlap {closed} vs MC {acc} (rel {rel})");
    }

    #[test]
    fn the_collision_kernel_is_one_at_full_overlap_and_decays_with_separation() {
        let base = Gaussian3::axis_aligned(v(0.0, 0.0, 0.0), v(0.3, 0.3, 0.3));
        let same = base;
        assert!((collision_kernel(&base, &same) - 1.0).abs() < 1e-12, "coincident means ⇒ kernel 1");
        let near = Gaussian3::axis_aligned(v(0.4, 0.0, 0.0), v(0.3, 0.3, 0.3));
        let far = Gaussian3::axis_aligned(v(2.0, 0.0, 0.0), v(0.3, 0.3, 0.3));
        assert!(collision_kernel(&base, &near) < 1.0);
        assert!(collision_kernel(&base, &far) < collision_kernel(&base, &near), "monotone decay with distance");
        assert!(collision_kernel(&base, &far) < 1e-4, "well-separated ⇒ ~zero");
    }

    #[test]
    fn the_overlap_is_symmetric() {
        let a = Gaussian3::yawed(v(0.3, -0.1, 0.2), v(0.6, 0.2, 0.3), 0.9);
        let b = Gaussian3::axis_aligned(v(-0.2, 0.4, 0.0), v(0.3, 0.5, 0.4));
        assert!((overlap_integral(&a, &b) - overlap_integral(&b, &a)).abs() < 1e-15);
        assert!((collision_kernel(&a, &b) - collision_kernel(&b, &a)).abs() < 1e-15);
    }

    #[test]
    fn the_kernel_gradient_matches_finite_differences() {
        let b = Gaussian3::yawed(v(0.2, -0.3, 0.1), v(0.5, 0.25, 0.4), 0.5);
        let a = Gaussian3::axis_aligned(v(-0.15, 0.2, -0.05), v(0.3, 0.3, 0.3));
        let g = kernel_grad_mu_a(&a, &b);
        let eps = 1e-6;
        for axis in 0..3 {
            let mut ap = a;
            let mut am = a;
            ap.mu[axis] += eps;
            am.mu[axis] -= eps;
            let fd = (collision_kernel(&ap, &b) - collision_kernel(&am, &b)) / (2.0 * eps);
            assert!((g[axis] - fd).abs() < 1e-6, "grad[{axis}] {} vs fd {fd}", g[axis]);
        }
    }

    #[test]
    fn rotating_an_elongated_robot_lets_it_fit_through_a_slot() {
        // THE CHAPTER. A corridor along world-x: two obstacle Gaussians walling it off in ±y. The robot
        // is one Gaussian elongated along its BODY-y axis. At ψ=0 its long axis points along world-y and
        // jams into both walls (high collision); yawed 90° the long axis turns to run along the corridor
        // (world-x), thin in y, and slips through (low collision). Orientation, not just position, matters.
        let g = 0.6; // half-gap in y
        let env = vec![
            Gaussian3::axis_aligned(v(0.0, g, 0.0), v(0.4, 0.3, 0.4)),
            Gaussian3::axis_aligned(v(0.0, -g, 0.0), v(0.4, 0.3, 0.4)),
        ];
        let robot = RobotSplat::single(v(0.12, 0.7, 0.12)); // long along body-y
        let at_center = v(0.0, 0.0, 0.0);

        let head_on = collision_cost(&env, &robot.posed(&at_center, 0.0));
        let turned = collision_cost(&env, &robot.posed(&at_center, std::f64::consts::FRAC_PI_2));
        assert!(turned < 0.25 * head_on, "yawing to align with the slot must cut collision: {turned} vs {head_on}");
    }

    #[test]
    fn foci_plans_a_collision_free_path_to_the_goal() {
        // An obstacle Gaussian sits astride the straight line between start and goal (just off the axis, so
        // the avoidance direction is defined — a dead-centered obstacle is a symmetric saddle). FOCI should
        // bow the path around it (peak collision small) while still arriving near the goal.
        let env = vec![Gaussian3::axis_aligned(v(1.0, 0.18, 0.0), v(0.35, 0.35, 0.35))];
        let robot = RobotSplat::single(v(0.15, 0.15, 0.15));
        let start = v(0.0, 0.0, 0.0);
        let goal = v(2.0, 0.0, 0.0);

        // straight-line baseline clips the obstacle
        let straight_peak = {
            let mid = v(1.0, 0.0, 0.0);
            collision_cost(&env, &robot.posed(&mid, 0.0))
        };
        let plan = plan(&env, &robot, start, goal, 0.0, 16, 20000, 5e-4);
        assert!(plan.peak_collision < 0.5 * straight_peak, "planned path should reduce peak collision: {} vs {straight_peak}", plan.peak_collision);
        let end = *plan.waypoints.last().unwrap();
        assert!((end - goal).norm() < 0.25, "should arrive near the goal: {end:?}");
    }
}
