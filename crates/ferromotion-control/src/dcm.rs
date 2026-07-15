//! **Divergent Component of Motion (DCM)** walking — Englsberger et al.'s bipedal framework, and the
//! reason capture-point control works.
//!
//! Split the LIPM's CoM dynamics `ẍ = ω²(x − v)` (with `ω = √(g/z₀)`, `v` the VRP/ZMP) into two
//! first-order parts via the DCM `ξ = x + ẋ/ω`:
//!
//! ```text
//!   ξ̇ = ω(ξ − v)     ← UNSTABLE: diverges away from the VRP. This is what you must control.
//!   ẋ = −ω(x − ξ)    ← STABLE:   the CoM always converges to the DCM, for free.
//! ```
//!
//! So walking reduces to steering one unstable first-order state with foot placement. Because the DCM
//! is unstable forward in time it is stable *backward*, so a reference is planned **backward** from
//! the final rest state through the footstep VRPs: `ξ_i^ini = v_i + (ξ_i^eos − v_i)·e^{−ω T_i}`. The
//! tracking law `v_cmd = ξ − ξ̇_des/ω + (k/ω)(ξ − ξ_des)` makes the DCM error decay as `ė = −k·e`.
//! Complements the capture-point / ZMP-preview / centroidal-MPC pieces already here. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector2;

/// LIPM natural frequency `ω = √(g / z₀)`.
pub fn lipm_omega(com_height: f64, g: f64) -> f64 {
    (g / com_height).sqrt()
}

/// The DCM (divergent component of motion / capture point) `ξ = x + ẋ/ω`.
pub fn dcm(com: Vector2<f64>, com_vel: Vector2<f64>, omega: f64) -> Vector2<f64> {
    com + com_vel / omega
}

/// One footstep: hold the VRP (≈ the foot / CoP) here for `duration`.
#[derive(Clone, Copy, Debug)]
pub struct DcmStep {
    pub vrp: Vector2<f64>,
    pub duration: f64,
}

/// A DCM reference planned backward through a footstep sequence.
#[derive(Clone, Debug)]
pub struct DcmPlan {
    pub omega: f64,
    pub steps: Vec<DcmStep>,
    /// DCM at the start of each step (from the backward recursion).
    pub xi_ini: Vec<Vector2<f64>>,
}

/// Plan the DCM reference backward from rest over the final footstep.
pub fn plan_dcm(omega: f64, steps: &[DcmStep]) -> DcmPlan {
    let n = steps.len();
    let mut xi_ini = vec![Vector2::zeros(); n];
    // End of the last step: at rest over the last foot ⇒ ξ = v (then ξ̇ = ω(ξ−v) = 0).
    let mut xi_eos = steps[n - 1].vrp;
    for i in (0..n).rev() {
        xi_ini[i] = steps[i].vrp + (xi_eos - steps[i].vrp) * (-omega * steps[i].duration).exp();
        xi_eos = xi_ini[i]; // this step's start is the previous step's end
    }
    DcmPlan { omega, steps: steps.to_vec(), xi_ini }
}

impl DcmPlan {
    pub fn total_duration(&self) -> f64 {
        self.steps.iter().map(|s| s.duration).sum()
    }

    /// Reference `(ξ_des, ξ̇_des)` at time `t`. Past the end, hold the final rest state.
    pub fn reference(&self, t: f64) -> (Vector2<f64>, Vector2<f64>) {
        let mut t0 = 0.0;
        for (i, s) in self.steps.iter().enumerate() {
            if t < t0 + s.duration {
                let tl = t - t0;
                let xi = s.vrp + (self.xi_ini[i] - s.vrp) * (self.omega * tl).exp();
                return (xi, self.omega * (xi - s.vrp));
            }
            t0 += s.duration;
        }
        let last = self.steps.last().unwrap().vrp;
        (last, Vector2::zeros()) // at rest over the final foot
    }
}

/// DCM tracking law: the VRP to command so the DCM error decays as `ė = −k·e`.
/// `v_cmd = ξ − ξ̇_des/ω + (k/ω)(ξ − ξ_des)`.
pub fn dcm_control(xi: Vector2<f64>, xi_des: Vector2<f64>, xi_dot_des: Vector2<f64>, omega: f64, k: f64) -> Vector2<f64> {
    xi - xi_dot_des / omega + (k / omega) * (xi - xi_des)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture_point;

    #[test]
    fn dcm_matches_the_capture_point_and_its_first_order_dynamics() {
        let (z0, g) = (0.9, 9.81);
        let omega = lipm_omega(z0, g);
        let (com, vel) = (Vector2::new(0.1, -0.05), Vector2::new(0.3, 0.1));
        // Same quantity as the existing capture point.
        let cp = capture_point([com.x, com.y], [vel.x, vel.y], z0, g);
        let xi = dcm(com, vel, omega);
        assert!((xi.x - cp[0]).abs() < 1e-12 && (xi.y - cp[1]).abs() < 1e-12);

        // Simulate the LIPM with a fixed VRP and confirm ξ̇ = ω(ξ − v) and ẋ = −ω(x − ξ).
        let v = Vector2::new(0.0, 0.0);
        let (mut x, mut xd, dt) = (com, vel, 1e-5);
        for _ in 0..20_000 {
            let xi0 = dcm(x, xd, omega);
            let acc = omega * omega * (x - v);
            xd += acc * dt;
            x += xd * dt;
            let xi1 = dcm(x, xd, omega);
            let xi_dot_num = (xi1 - xi0) / dt;
            assert!((xi_dot_num - omega * (xi0 - v)).norm() < 1e-3, "DCM dynamics violated");
        }
    }

    #[test]
    fn backward_plan_is_continuous_and_ends_at_rest() {
        let omega = lipm_omega(0.9, 9.81);
        let steps = vec![
            DcmStep { vrp: Vector2::new(0.0, 0.0), duration: 0.6 },
            DcmStep { vrp: Vector2::new(0.25, 0.1), duration: 0.6 },
            DcmStep { vrp: Vector2::new(0.5, -0.1), duration: 0.6 },
            DcmStep { vrp: Vector2::new(0.75, 0.0), duration: 0.6 },
        ];
        let plan = plan_dcm(omega, &steps);
        // Each step's forward DCM flow lands exactly on the next step's initial DCM.
        for i in 0..steps.len() - 1 {
            let s = &steps[i];
            let eos = s.vrp + (plan.xi_ini[i] - s.vrp) * (omega * s.duration).exp();
            assert!((eos - plan.xi_ini[i + 1]).norm() < 1e-9, "DCM reference discontinuous at step {i}");
        }
        // The reference ends at rest over the final foot.
        let (xi_end, xi_dot_end) = plan.reference(plan.total_duration() + 1.0);
        assert!((xi_end - steps.last().unwrap().vrp).norm() < 1e-9 && xi_dot_end.norm() < 1e-12);
    }

    #[test]
    fn closed_loop_dcm_control_walks_and_comes_to_rest() {
        let (z0, g) = (0.9, 9.81);
        let omega = lipm_omega(z0, g);
        let steps = vec![
            DcmStep { vrp: Vector2::new(0.0, 0.0), duration: 0.7 },
            DcmStep { vrp: Vector2::new(0.3, 0.09), duration: 0.7 },
            DcmStep { vrp: Vector2::new(0.6, -0.09), duration: 0.7 },
            DcmStep { vrp: Vector2::new(0.9, 0.09), duration: 0.7 },
            DcmStep { vrp: Vector2::new(1.2, 0.0), duration: 0.7 },
        ];
        let plan = plan_dcm(omega, &steps);

        // Start the CoM at the planned DCM (so the reference is consistent), slightly perturbed.
        let (xi0, _) = plan.reference(0.0);
        let mut com = xi0 - Vector2::new(0.02, 0.01); // deliberate initial DCM error
        let mut vel = Vector2::zeros();

        let (dt, k) = (1e-4, 4.0);
        let mut worst_err = 0.0f64;
        let total = plan.total_duration();
        let steps_n = ((total + 3.0) / dt) as usize; // walk, then settle
        for i in 0..steps_n {
            let t = i as f64 * dt;
            let xi = dcm(com, vel, omega);
            let (xi_des, xi_dot_des) = plan.reference(t);
            if t > 0.2 && t < total {
                worst_err = worst_err.max((xi - xi_des).norm());
            }
            let v_cmd = dcm_control(xi, xi_des, xi_dot_des, omega, k);
            let acc = omega * omega * (com - v_cmd);
            vel += acc * dt;
            com += vel * dt;
        }
        // The DCM tracked the reference (the initial error decayed away) …
        assert!(worst_err < 0.02, "DCM tracking error too large: {worst_err}");
        // … the robot walked forward …
        assert!(com.x > 1.0, "did not walk forward: com.x = {}", com.x);
        // … and came to rest over the final foot.
        let last = steps.last().unwrap().vrp;
        assert!((com - last).norm() < 0.02, "did not settle over the last foot: {com:?} vs {last:?}");
        assert!(vel.norm() < 0.02, "did not come to rest: |v| = {}", vel.norm());
    }
}
