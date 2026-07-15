//! **Series-elastic actuator (SEA) / transmission model** — the actuator physics between a commanded
//! motor torque and the torque a joint actually delivers. Everything else in the stack treats joints
//! as ideal torque sources; real hardware has a rotor inertia reflected through a gearbox (`N²·J`), a
//! deliberate **series spring**, and friction.
//!
//! The spring is the point of an SEA: its deflection `δ = θ_motor − q_load` *is* the torque sensor
//! (`τ = k·δ`), which makes clean force control possible on a stiff geared drive. The motor side obeys
//! `J_reflected·θ̈ = N·τ_motor − τ_spring − τ_friction`, and `τ_spring` is what the load feels — so an
//! SEA is a spring-mass system whose blocked-output natural frequency is `√(k / N²J_rotor)`.
//! Pure Rust → WASM-clean.

/// A series-elastic joint: rotor + gearbox + series spring + friction.
#[derive(Clone, Copy, Debug)]
pub struct SeaJoint {
    /// Rotor inertia (motor side, before the gearbox).
    pub j_rotor: f64,
    /// Gear ratio `N` (output torque = `N·τ_motor`; reflected inertia = `N²·j_rotor`).
    pub ratio: f64,
    pub k_spring: f64,
    /// Series-spring damping.
    pub b_spring: f64,
    /// Viscous friction (output side).
    pub viscous: f64,
    /// Coulomb friction magnitude (smoothed).
    pub coulomb: f64,
    /// Motor angle, referred to the output side.
    pub theta: f64,
    /// Motor velocity, referred to the output side.
    pub omega: f64,
}

impl SeaJoint {
    /// Motor inertia as seen at the output: `N²·J_rotor`.
    pub fn reflected_inertia(&self) -> f64 {
        self.ratio * self.ratio * self.j_rotor
    }

    /// Spring deflection `δ = θ_motor − q_load`.
    pub fn deflection(&self, q_load: f64) -> f64 {
        self.theta - q_load
    }

    /// Torque transmitted to the load: `k·δ + b·δ̇` — the SEA's intrinsic torque measurement.
    pub fn spring_torque(&self, q_load: f64, qd_load: f64) -> f64 {
        self.k_spring * (self.theta - q_load) + self.b_spring * (self.omega - qd_load)
    }

    /// Advance the motor state one step under commanded `tau_motor` against a load at `(q, q̇)`;
    /// returns the torque delivered to the load.
    pub fn step(&mut self, dt: f64, tau_motor: f64, q_load: f64, qd_load: f64) -> f64 {
        let tau_out = self.ratio * tau_motor; // gearbox amplifies torque
        let tau_s = self.spring_torque(q_load, qd_load);
        // Smoothed Coulomb + viscous friction, always opposing motion.
        let fric = self.viscous * self.omega + self.coulomb * (self.omega / 1e-3).tanh();
        let acc = (tau_out - tau_s - fric) / self.reflected_inertia();
        self.omega += acc * dt;
        self.theta += self.omega * dt;
        tau_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sea(k: f64, ratio: f64) -> SeaJoint {
        SeaJoint { j_rotor: 0.01, ratio, k_spring: k, b_spring: 0.0, viscous: 0.0, coulomb: 0.0, theta: 0.0, omega: 0.0 }
    }

    /// Period of oscillation from successive positive-going zero crossings of θ.
    fn measure_period(mut j: SeaJoint, dt: f64) -> f64 {
        j.theta = 0.01; // small initial deflection, released from rest
        let (mut prev, mut first, mut last, mut count) = (j.theta, -1.0f64, -1.0f64, 0);
        for k in 0..400_000 {
            j.step(dt, 0.0, 0.0, 0.0); // blocked output
            if prev < 0.0 && j.theta >= 0.0 {
                let t = k as f64 * dt;
                if first < 0.0 {
                    first = t;
                } else {
                    last = t;
                    count += 1;
                }
            }
            prev = j.theta;
        }
        (last - first) / count as f64
    }

    #[test]
    fn blocked_output_oscillates_at_the_spring_natural_frequency() {
        // With the load blocked: J_reflected·θ̈ = −k·θ ⇒ ω = √(k / N²J_rotor).
        let j = sea(100.0, 1.0);
        let period = measure_period(j, 1e-6);
        let w_true = (j.k_spring / j.reflected_inertia()).sqrt();
        let period_true = 2.0 * std::f64::consts::PI / w_true;
        assert!((period - period_true).abs() / period_true < 1e-3, "period {period} vs analytic {period_true}");
    }

    #[test]
    fn gearbox_reflects_inertia_as_the_square_of_the_ratio() {
        let j1 = sea(100.0, 1.0);
        let j2 = sea(100.0, 2.0);
        assert!((j2.reflected_inertia() - 4.0 * j1.reflected_inertia()).abs() < 1e-12);
        // Doubling the ratio quadruples the inertia ⇒ halves the natural frequency (doubles the period).
        let (p1, p2) = (measure_period(j1, 1e-6), measure_period(j2, 1e-6));
        assert!((p2 / p1 - 2.0).abs() < 5e-3, "period ratio {} (expected 2)", p2 / p1);
    }

    #[test]
    fn spring_deflection_measures_the_transmitted_torque() {
        // Against a blocked load, a constant motor torque settles to δ = τ_out/k, and the spring
        // delivers exactly that torque — the SEA's force-sensing principle.
        let mut j = SeaJoint { viscous: 2.0, ..sea(100.0, 1.0) }; // damping so it settles
        let tau_motor = 3.0;
        let mut tau_s = 0.0;
        for _ in 0..400_000 {
            tau_s = j.step(1e-5, tau_motor, 0.0, 0.0);
        }
        let tau_out = j.ratio * tau_motor;
        assert!((tau_s - tau_out).abs() < 1e-3, "delivered {tau_s} vs commanded {tau_out}");
        assert!((j.deflection(0.0) - tau_out / j.k_spring).abs() < 1e-5, "deflection ≠ τ/k");
        assert!((j.k_spring * j.deflection(0.0) - tau_s).abs() < 1e-9, "τ = k·δ must hold exactly");
    }

    #[test]
    fn torque_control_tracks_a_setpoint_and_friction_dissipates() {
        // Closed-loop SEA force control: PI on the torque error, measured from spring deflection.
        let mut j = SeaJoint { viscous: 1.0, coulomb: 0.05, ..sea(200.0, 1.0) };
        let target = 2.5;
        let (mut integral, dt) = (0.0, 1e-5);
        let mut tau_s = 0.0;
        for _ in 0..400_000 {
            let err = target - tau_s;
            integral += err * dt;
            let cmd = 0.5 * err + 20.0 * integral; // PI
            tau_s = j.step(dt, cmd, 0.0, 0.0);
        }
        assert!((tau_s - target).abs() < 1e-2, "torque control did not converge: {tau_s} vs {target}");

        // Friction dissipates a free oscillation.
        let mut f = SeaJoint { viscous: 0.5, coulomb: 0.02, ..sea(100.0, 1.0) };
        f.theta = 0.05;
        let e0 = 0.5 * f.k_spring * f.theta * f.theta;
        for _ in 0..200_000 {
            f.step(1e-5, 0.0, 0.0, 0.0);
        }
        let e1 = 0.5 * f.k_spring * f.theta * f.theta + 0.5 * f.reflected_inertia() * f.omega * f.omega;
        assert!(e1 < 0.05 * e0, "friction did not dissipate: {e1} vs {e0}");
    }
}
