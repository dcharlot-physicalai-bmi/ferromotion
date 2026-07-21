//! **Differentiable-control lab** — the rig behind the "tune a controller with a gradient" lesson. It drives
//! a real [`ferromotion_learn::PidController`]: a PID whose three gains are tuned by backpropagating a
//! setpoint-tracking cost *through* the closed-loop rollout on a damped point mass under a constant
//! disturbance. The reader presses *Train* and watches the step response go from a proportional-only run that
//! stalls short of the setpoint (steady-state offset) to one that settles exactly on it — because the gradient
//! discovered it needed integral action. No episodes, no trial and error: one gradient per rollout.

use ferromotion_learn::PidController;
use wasm_bindgen::prelude::*;

const SETPOINT: f64 = 1.0;

#[wasm_bindgen]
pub struct PidLab {
    pid: PidController,
    epochs: u32,
}

#[wasm_bindgen]
impl PidLab {
    #[wasm_bindgen(constructor)]
    pub fn new() -> PidLab {
        PidLab { pid: PidController::new(1.0, 0.6, 2.0, SETPOINT), epochs: 0 }
    }

    /// Train the gains for `n` gradient steps (backprop through the rollout).
    pub fn train(&mut self, n: u32) -> f64 {
        let c = self.pid.train(n as usize, 0.05);
        self.epochs += n;
        c
    }
    pub fn reset(&mut self) {
        self.pid = PidController::new(1.0, 0.6, 2.0, SETPOINT);
        self.epochs = 0;
    }

    pub fn epochs(&self) -> u32 {
        self.epochs
    }
    pub fn setpoint(&self) -> f64 {
        SETPOINT
    }
    pub fn kp(&self) -> f64 {
        self.pid.gains()[0]
    }
    pub fn ki(&self) -> f64 {
        self.pid.gains()[1]
    }
    pub fn kd(&self) -> f64 {
        self.pid.gains()[2]
    }
    pub fn final_error(&self) -> f64 {
        self.pid.final_error()
    }

    /// The closed-loop position trajectory (for plotting the step response).
    pub fn n_traj(&self) -> usize {
        self.pid.simulate().len()
    }
    pub fn traj(&self, i: usize) -> f64 {
        self.pid.simulate().get(i).copied().unwrap_or(0.0)
    }
    /// The whole trajectory as a flat array (fewer boundary calls).
    pub fn trajectory(&self) -> Vec<f64> {
        self.pid.simulate()
    }
}

impl Default for PidLab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_drives_the_step_response_to_the_setpoint() {
        let mut lab = PidLab::new();
        let e0 = lab.final_error();
        lab.train(800);
        assert!(lab.final_error() < 0.05, "should reach setpoint: {} → {}", e0, lab.final_error());
        assert!(lab.ki() > 0.1, "should have discovered integral action: Ki={}", lab.ki());
    }
}
