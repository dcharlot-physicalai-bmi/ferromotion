//! **Differentiable control** — tune a controller by backpropagating a trajectory cost *through* the closed
//! loop. Because the plant is differentiable, you can roll out the controlled system, measure how far it is
//! from the goal, and take the gradient of that cost w.r.t. the controller's parameters — then step them
//! downhill. No trial-and-error over thousands of episodes as in reinforcement learning; one gradient per
//! rollout. This is the payoff of every learned/differentiable model in the course: a model you can
//! differentiate is a model you can control by gradient descent.
//!
//! Here the controller is a **PID** (three interpretable gains) and the plant is a damped point mass under a
//! constant disturbance, so proportional-derivative action alone leaves a steady-state offset and the loop
//! must *discover* integral action to null it. The gains are tuned by rolling the closed loop out on the
//! reverse-mode tape and backpropagating a setpoint-tracking cost. Verified: from a proportional-only start
//! that never reaches the setpoint, training drives the steady-state error to near zero and the integral gain
//! becomes positive — the loop learned it needs an integrator, from the gradient alone.

use crate::autodiff::Tape;

/// A PID controller (gains `[Kp, Ki, Kd]`) tuned by differentiating through the plant rollout.
pub struct PidController {
    gains: [f64; 3],
    m: [f64; 3],
    v: [f64; 3],
    t: u64,
    // plant + task
    mass: f64,
    damping: f64,
    disturbance: f64,
    setpoint: f64,
    dt: f64,
    steps: usize,
}

impl PidController {
    /// A controller for a damped point mass `mass·ẍ = u − damping·ẋ − disturbance`, tasked to hold
    /// `setpoint`. Starts proportional-only (`Kp=1, Ki=0, Kd=0`).
    pub fn new(mass: f64, damping: f64, disturbance: f64, setpoint: f64) -> Self {
        PidController {
            gains: [1.0, 0.0, 0.0],
            m: [0.0; 3],
            v: [0.0; 3],
            t: 0,
            mass,
            damping,
            disturbance,
            setpoint,
            dt: 0.05,
            steps: 90,
        }
    }

    pub fn gains(&self) -> [f64; 3] {
        self.gains
    }

    /// Roll out the closed loop in plain `f64` and return the position trajectory (for plotting).
    pub fn simulate(&self) -> Vec<f64> {
        let (mut x, mut v, mut integ) = (0.0, 0.0, 0.0);
        let mut out = vec![x];
        for _ in 0..self.steps {
            let e = self.setpoint - x;
            integ += e * self.dt;
            let de = -v; // ė = −ẋ (setpoint constant)
            let u = self.gains[0] * e + self.gains[1] * integ + self.gains[2] * de;
            let acc = (u - self.damping * v - self.disturbance) / self.mass;
            v += acc * self.dt;
            x += v * self.dt;
            out.push(x);
        }
        out
    }

    /// Steady-state tracking error (distance of the final position from the setpoint).
    pub fn final_error(&self) -> f64 {
        (self.simulate().last().copied().unwrap_or(0.0) - self.setpoint).abs()
    }

    /// One Adam step of the tracking cost, differentiating through the closed-loop rollout. Returns the cost.
    pub fn train_step(&mut self, lr: f64) -> f64 {
        let tape = Tape::new();
        let kp = tape.var(self.gains[0]);
        let ki = tape.var(self.gains[1]);
        let kd = tape.var(self.gains[2]);
        let mut x = tape.constant(0.0);
        let mut v = tape.constant(0.0);
        let mut integ = tape.constant(0.0);
        let mut cost = tape.constant(0.0);
        let (m, c, d, sp, dt) = (self.mass, self.damping, self.disturbance, self.setpoint, self.dt);
        for step in 0..self.steps {
            let e = tape.constant(sp) - x;
            integ = integ + e * dt;
            let de = v * (-1.0);
            let u = kp * e + ki * integ + kd * de;
            let acc = (u - v * c - tape.constant(d)) * (1.0 / m);
            v = v + acc * dt;
            x = x + v * dt;
            // weight later steps more (settle to the setpoint), plus a small control-effort penalty
            let w = 0.5 + step as f64 / self.steps as f64;
            cost = cost + e * e * (w * dt) + u * u * (1e-4 * dt);
        }
        let g = cost.backward();
        let grad = [g.wrt(kp), g.wrt(ki), g.wrt(kd)];

        self.t += 1;
        let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8);
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for (i, &gi) in grad.iter().enumerate() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            self.gains[i] -= lr * (self.m[i] / bc1) / ((self.v[i] / bc2).sqrt() + eps);
        }
        cost.value()
    }

    /// Train for `epochs` steps; returns the final cost.
    pub fn train(&mut self, epochs: usize, lr: f64) -> f64 {
        let mut c = f64::INFINITY;
        for _ in 0..epochs {
            c = self.train_step(lr);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differentiable_tuning_nulls_the_steady_state_error_by_discovering_integral_action() {
        // Plant: mass 1, damping 0.6, constant disturbance 2.0, setpoint 1.0. Proportional-only leaves a large
        // offset; differentiating the tracking cost through the rollout must find gains (incl. Ki > 0) that
        // reach the setpoint.
        let mut pid = PidController::new(1.0, 0.6, 2.0, 1.0);
        let e_before = pid.final_error();
        assert!(e_before > 0.3, "proportional-only should miss the setpoint: {e_before}");
        pid.train(800, 0.05);
        let e_after = pid.final_error();
        assert!(e_after < 0.05, "differentiable tuning should reach the setpoint: error {e_after}");
        assert!(pid.gains()[1] > 0.1, "it should discover integral action: Ki = {}", pid.gains()[1]);
    }
}
