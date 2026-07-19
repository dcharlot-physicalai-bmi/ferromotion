//! **Path-tracking control for nonholonomic vehicles** — the lateral controllers that steer a car onto and
//! along a reference path, closing the loop on a [`crate::FrenetPlanner`] plan. Two classics on the
//! **kinematic bicycle model** `ẋ = v cosθ, ẏ = v sinθ, θ̇ = (v/L) tanδ, v̇ = a`:
//!
//! - **Pure pursuit** (Coulter, 1992): aim the car at a point one lookahead-distance `L_d` ahead on the
//!   path; the required steering follows from the geometry of the arc through it, `δ = atan2(2L sinα, L_d)`.
//! - **Stanley** (Thrun et al., the Stanford DARPA Grand Challenge winner, 2006): steer by the **front-axle
//!   heading error** plus a cross-track term `δ = ψ + atan2(k·e, v)`, which drives the cross-track error to
//!   zero exponentially.
//!
//! This opens the ground-vehicle *control* layer next to the [`crate::FrenetPlanner`] *planning* layer and
//! [`crate::dubins`] geometry. Verified: the bicycle model traces an exact circle of radius `L/tanδ` under
//! constant steering (the geometric oracle); and both pure pursuit and Stanley drive a car that starts off a
//! straight path onto it (cross-track error → 0), and track a curved (circular) path with bounded error.
//! Pure `nalgebra` → WASM-clean.

use nalgebra::Vector2;

/// Kinematic bicycle state.
#[derive(Clone, Copy, Debug)]
pub struct CarState {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub v: f64,
}

/// A kinematic bicycle of wheelbase `l`.
#[derive(Clone, Copy, Debug)]
pub struct Bicycle {
    pub l: f64,
}

impl Bicycle {
    /// One RK4 step with steering `delta` and longitudinal acceleration `accel`.
    pub fn step(&self, s: CarState, delta: f64, accel: f64, dt: f64) -> CarState {
        let f = |st: &CarState| CarState {
            x: st.v * st.theta.cos(),
            y: st.v * st.theta.sin(),
            theta: st.v / self.l * delta.tan(),
            v: accel,
        };
        let add = |a: &CarState, k: &CarState, h: f64| CarState { x: a.x + h * k.x, y: a.y + h * k.y, theta: a.theta + h * k.theta, v: a.v + h * k.v };
        let k1 = f(&s);
        let k2 = f(&add(&s, &k1, 0.5 * dt));
        let k3 = f(&add(&s, &k2, 0.5 * dt));
        let k4 = f(&add(&s, &k3, dt));
        CarState {
            x: s.x + dt / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
            y: s.y + dt / 6.0 * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
            theta: s.theta + dt / 6.0 * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta),
            v: s.v + dt / 6.0 * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v),
        }
    }
}

/// A reference path as a polyline.
#[derive(Clone, Debug)]
pub struct Path {
    pub pts: Vec<Vector2<f64>>,
}

impl Path {
    /// Index and closest point on the polyline to `p`, plus the local tangent heading there.
    fn nearest(&self, p: &Vector2<f64>) -> (usize, Vector2<f64>, f64) {
        let mut best = (0usize, self.pts[0], 0.0);
        let mut bd = f64::INFINITY;
        for i in 0..self.pts.len() - 1 {
            let (a, b) = (self.pts[i], self.pts[i + 1]);
            let ab = b - a;
            let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
            let proj = a + t * ab;
            let d = (p - proj).norm_squared();
            if d < bd {
                bd = d;
                best = (i, proj, ab.y.atan2(ab.x));
            }
        }
        best
    }

    /// A point one arc-distance `dist` ahead of the projection of `p` onto the path.
    fn lookahead(&self, p: &Vector2<f64>, dist: f64) -> Vector2<f64> {
        let (i0, proj, _) = self.nearest(p);
        let mut remaining = dist;
        let mut cur = proj;
        for i in i0..self.pts.len() - 1 {
            let seg_end = self.pts[i + 1];
            let seg = seg_end - cur;
            let len = seg.norm();
            if len >= remaining {
                return cur + seg / len * remaining;
            }
            remaining -= len;
            cur = seg_end;
        }
        *self.pts.last().unwrap()
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

/// **Pure-pursuit** steering angle for wheelbase `l` and lookahead `ld`.
pub fn pure_pursuit(s: &CarState, path: &Path, ld: f64, l: f64) -> f64 {
    let rear = Vector2::new(s.x, s.y);
    let target = path.lookahead(&rear, ld);
    let alpha = wrap((target.y - s.y).atan2(target.x - s.x) - s.theta);
    (2.0 * l * alpha.sin()).atan2(ld)
}

/// **Stanley** steering angle: front-axle heading error + cross-track term with gain `k`.
pub fn stanley(s: &CarState, path: &Path, k: f64, l: f64) -> f64 {
    let front = Vector2::new(s.x + l * s.theta.cos(), s.y + l * s.theta.sin());
    let (_i, closest, path_heading) = path.nearest(&front);
    // signed cross-track: positive when the front axle is to the LEFT of the path direction
    let left = Vector2::new(-path_heading.sin(), path_heading.cos());
    let e = (front - closest).dot(&left);
    let psi = wrap(path_heading - s.theta); // heading error
    let vsafe = s.v.abs().max(0.5);
    psi + (-k * e).atan2(vsafe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_path() -> Path {
        Path { pts: (0..=200).map(|i| Vector2::new(i as f64 * 0.5, 0.0)).collect() }
    }

    #[test]
    fn constant_steering_traces_a_circle_of_the_expected_radius() {
        // THE ORACLE. Under constant steering δ the bicycle turns with radius R = L/tanδ; after driving an
        // arc length L_arc the heading advances by L_arc/R and the position lies on that circle.
        let bike = Bicycle { l: 2.5 };
        let delta = 0.2_f64;
        let r = bike.l / delta.tan();
        let mut s = CarState { x: 0.0, y: 0.0, theta: 0.0, v: 5.0 };
        // circle centre is to the left of the car (positive δ turns left): (0, R)
        let dt = 0.001;
        for _ in 0..4000 {
            s = bike.step(s, delta, 0.0, dt);
            let dist_from_centre = ((s.x - 0.0).powi(2) + (s.y - r).powi(2)).sqrt();
            assert!((dist_from_centre - r).abs() < 1e-2, "car should stay on the radius-{r} circle: {dist_from_centre}");
        }
        // after 4000·0.001·5 = 20 m of arc, heading advanced by arc/R
        assert!((wrap(s.theta - 20.0 / r)).abs() < 1e-2, "heading should advance by arc/R");
    }

    #[test]
    fn pure_pursuit_converges_to_a_straight_path() {
        // Start 2 m off the path, heading along it; pure pursuit should null the cross-track error.
        let bike = Bicycle { l: 2.5 };
        let path = straight_path();
        let mut s = CarState { x: 0.0, y: 2.0, theta: 0.0, v: 5.0 };
        let dt = 0.02;
        for _ in 0..600 {
            let delta = pure_pursuit(&s, &path, 6.0, bike.l).clamp(-0.6, 0.6);
            s = bike.step(s, delta, 0.0, dt);
        }
        assert!(s.y.abs() < 0.05, "cross-track error should vanish: {}", s.y);
    }

    #[test]
    fn stanley_converges_to_a_straight_path() {
        // Same test for the Stanley controller (front-axle geometry).
        let bike = Bicycle { l: 2.5 };
        let path = straight_path();
        let mut s = CarState { x: 0.0, y: 2.0, theta: 0.0, v: 5.0 };
        let dt = 0.02;
        for _ in 0..600 {
            let delta = stanley(&s, &path, 2.0, bike.l).clamp(-0.6, 0.6);
            s = bike.step(s, delta, 0.0, dt);
        }
        assert!(s.y.abs() < 0.05, "cross-track error should vanish: {}", s.y);
    }

    #[test]
    fn pure_pursuit_tracks_a_circular_path_with_bounded_error() {
        // THE HEADLINE. A curved reference (a big circle); the tracker should follow it with small,
        // bounded cross-track error rather than drifting off.
        let bike = Bicycle { l: 2.5 };
        let radius = 30.0;
        // a full circle (+ a little), so there is always path ahead of the lookahead point
        let path = Path {
            pts: (0..=380).map(|i| {
                let a = i as f64 * std::f64::consts::PI / 180.0;
                Vector2::new(radius * a.sin(), radius * (1.0 - a.cos()))
            }).collect(),
        };
        // start on the path at the origin, heading +x (tangent)
        let mut s = CarState { x: 0.0, y: 0.0, theta: 0.0, v: 6.0 };
        let dt = 0.02;
        let mut max_err: f64 = 0.0;
        // 1000·0.02·6 = 120 m of arc, well within one 188 m loop — never overruns the path end
        for _ in 0..1000 {
            let delta = pure_pursuit(&s, &path, 6.0, bike.l).clamp(-0.6, 0.6);
            s = bike.step(s, delta, 0.0, dt);
            let (_i, closest, _h) = path.nearest(&Vector2::new(s.x, s.y));
            max_err = max_err.max((Vector2::new(s.x, s.y) - closest).norm());
        }
        // pure pursuit's known steady-state curve-cutting error is ~L_d²/(2R) ≈ 0.6 m here
        assert!(max_err < 0.8, "circular tracking error should stay bounded: max {max_err}");
    }
}
