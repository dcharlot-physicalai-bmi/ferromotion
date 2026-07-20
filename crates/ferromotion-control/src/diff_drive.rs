//! **Differential-drive / unicycle mobile robots** — kinematics and pose stabilization for the most common
//! ground-robot platform (two independently-driven wheels: TurtleBot, warehouse AMRs, most research
//! rovers). Unlike the car-like bicycle of [`crate::pure_pursuit`]/[`crate::stanley`], a differential drive
//! can **rotate in place**, so it reaches arbitrary poses. The kinematic model is the unicycle
//! `ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω`, with the body velocity `(v, ω)` mapped to/from wheel speeds by the
//! wheel radius and track width.
//!
//! For driving to a goal *pose*, the **Astolfi polar-coordinate controller** (Aicardi, Casalino, Bicchi &
//! Balestrino, 1995; Siegwart) expresses the error in polar form `(ρ, α, β)` — distance, bearing to goal,
//! and final-heading error — and applies `v = k_ρ ρ`, `ω = k_α α + k_β β`, which is globally asymptotically
//! stable for `k_ρ > 0`, `k_β < 0`, `k_α − k_ρ > 0`. Verified: the wheel↔body map round-trips; equal wheel
//! speeds drive a straight line and opposite speeds spin in place; and the controller drives the robot to a
//! goal pose from several starts. Pure `nalgebra` → WASM-clean.

/// A differential-drive robot: wheel radius and track width (distance between wheels).
#[derive(Clone, Copy, Debug)]
pub struct DiffDrive {
    pub wheel_radius: f64,
    pub track: f64,
}

impl DiffDrive {
    /// Body velocity `(v, ω)` from left/right **wheel angular** speeds.
    pub fn body_from_wheels(&self, wl: f64, wr: f64) -> (f64, f64) {
        let v = self.wheel_radius * (wr + wl) / 2.0;
        let omega = self.wheel_radius * (wr - wl) / self.track;
        (v, omega)
    }

    /// Left/right wheel angular speeds from a body velocity `(v, ω)`.
    pub fn wheels_from_body(&self, v: f64, omega: f64) -> (f64, f64) {
        let half = omega * self.track / 2.0;
        ((v - half) / self.wheel_radius, (v + half) / self.wheel_radius)
    }
}

/// Unicycle pose `(x, y, θ)`.
#[derive(Clone, Copy, Debug)]
pub struct Unicycle {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Unicycle {
    /// Exact integration of the unicycle under a constant body velocity `(v, ω)` over `dt` (straight line
    /// when `ω ≈ 0`, otherwise a circular arc).
    pub fn step(&self, v: f64, omega: f64, dt: f64) -> Unicycle {
        if omega.abs() < 1e-9 {
            Unicycle { x: self.x + v * self.theta.cos() * dt, y: self.y + v * self.theta.sin() * dt, theta: self.theta }
        } else {
            let th2 = self.theta + omega * dt;
            Unicycle {
                x: self.x + (v / omega) * (th2.sin() - self.theta.sin()),
                y: self.y - (v / omega) * (th2.cos() - self.theta.cos()),
                theta: th2,
            }
        }
    }
}

fn wrap(a: f64) -> f64 {
    let mut a = a % std::f64::consts::TAU;
    if a > std::f64::consts::PI {
        a -= std::f64::consts::TAU;
    } else if a <= -std::f64::consts::PI {
        a += std::f64::consts::TAU;
    }
    a
}

/// Gains for the Astolfi polar controller. Stable when `k_rho > 0`, `k_beta < 0`, `k_alpha − k_rho > 0`.
#[derive(Clone, Copy, Debug)]
pub struct PolarGains {
    pub k_rho: f64,
    pub k_alpha: f64,
    pub k_beta: f64,
}

impl Default for PolarGains {
    fn default() -> Self {
        PolarGains { k_rho: 3.0, k_alpha: 8.0, k_beta: -1.5 }
    }
}

/// **Astolfi polar-coordinate pose controller**: body velocity `(v, ω)` steering the unicycle `state` to the
/// goal pose `(gx, gy, gtheta)`.
pub fn polar_control(state: &Unicycle, gx: f64, gy: f64, gtheta: f64, gains: &PolarGains) -> (f64, f64) {
    // robot position expressed in the goal frame
    let (dx, dy) = (state.x - gx, state.y - gy);
    let (c, s) = (gtheta.cos(), gtheta.sin());
    let px = c * dx + s * dy;
    let py = -s * dx + c * dy;
    let theta_rel = wrap(state.theta - gtheta);

    let rho = (px * px + py * py).sqrt();
    // bearing from robot to goal (goal at origin of this frame ⇒ direction is −p)
    let alpha = wrap(-theta_rel + (-py).atan2(-px));
    let beta = wrap(-theta_rel - alpha);

    let v = gains.k_rho * rho;
    let omega = gains.k_alpha * alpha + gains.k_beta * beta;
    (v, omega)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wheel_body_map_round_trips() {
        // THE ORACLE (algebraic). wheels → body → wheels is the identity.
        let dd = DiffDrive { wheel_radius: 0.05, track: 0.3 };
        for (wl, wr) in [(2.0, 5.0), (-1.0, 1.0), (3.0, 3.0), (4.0, -2.0)] {
            let (v, w) = dd.body_from_wheels(wl, wr);
            let (wl2, wr2) = dd.wheels_from_body(v, w);
            assert!((wl2 - wl).abs() < 1e-12 && (wr2 - wr).abs() < 1e-12, "round trip failed for ({wl},{wr})");
        }
    }

    #[test]
    fn equal_wheels_go_straight_and_opposite_wheels_spin_in_place() {
        let dd = DiffDrive { wheel_radius: 0.05, track: 0.3 };
        // equal speeds ⇒ pure translation
        let (v, w) = dd.body_from_wheels(4.0, 4.0);
        assert!(w.abs() < 1e-12 && v > 0.0, "equal wheels ⇒ straight");
        let mut u = Unicycle { x: 0.0, y: 0.0, theta: 0.3 };
        let before = u.theta;
        u = u.step(v, w, 0.5);
        assert!((u.theta - before).abs() < 1e-12, "heading unchanged going straight");
        assert!((u.y - u.x * (0.3f64).tan()).abs() < 1e-9, "moves along its heading");
        // opposite speeds ⇒ pure rotation, no translation
        let (v2, w2) = dd.body_from_wheels(-4.0, 4.0);
        assert!(v2.abs() < 1e-12 && w2.abs() > 0.0, "opposite wheels ⇒ spin");
        let u2 = Unicycle { x: 1.0, y: 2.0, theta: 0.0 }.step(v2, w2, 0.3);
        assert!((u2.x - 1.0).abs() < 1e-12 && (u2.y - 2.0).abs() < 1e-12, "position fixed while spinning");
    }

    #[test]
    fn the_polar_controller_drives_to_the_goal_pose() {
        // THE HEADLINE. From several starting poses the Astolfi controller regulates the unicycle to the goal
        // position and orientation.
        let gains = PolarGains::default();
        let (gx, gy, gth) = (2.0, 1.0, 0.5);
        let starts = [(0.0, 0.0, 0.0), (-1.0, 2.0, 1.5), (3.0, -1.0, -2.0), (2.5, 2.5, 3.0)];
        for &(x0, y0, th0) in &starts {
            let mut u = Unicycle { x: x0, y: y0, theta: th0 };
            let dt = 0.01;
            for _ in 0..6000 {
                let (v, w) = polar_control(&u, gx, gy, gth, &gains);
                u = u.step(v, w, dt);
            }
            let dist = ((u.x - gx).powi(2) + (u.y - gy).powi(2)).sqrt();
            assert!(dist < 0.02, "start ({x0},{y0},{th0}) should reach the goal: dist {dist}");
            // ~5° heading tolerance — the Astolfi law's slow orientation tail as ρ→0 (v→0)
            assert!(wrap(u.theta - gth).abs() < 0.1, "final heading off by {}", wrap(u.theta - gth));
        }
    }
}
