//! **WebGPU batched-RL rollouts** — the on-device-RL axis on the local GPU. Brax / MJX / Newton
//! simulate thousands of environment instances in parallel for gradient-free policy search; this is
//! the browser-deployable counterpart. Each control step samples a batch of candidate action
//! sequences and rolls every one through the cartpole dynamics — independent per candidate, no
//! cross-lane data dependence, so it maps directly to a WebGPU compute kernel: one GPU thread rolls
//! out one candidate future. The controller picks the lowest-cost candidate and executes its first
//! action (receding-horizon random shooting), and the pole swings itself upright. The same rollout on
//! the verified CPU core ([`batch_rollout`]) is the reference the GPU is checked against.

use ferromotion_control::{batch_rollout, Cartpole};
use wasm_bindgen::prelude::*;

const B: usize = 2048; // parallel candidate futures per control step
const HORIZON: usize = 25;
const DT: f64 = 0.02;
const FORCE: f64 = 12.0; // candidate force magnitude

#[wasm_bindgen]
pub struct GpuCartpoleRef {
    cp: Cartpole,
    state: [f64; 4], // [x, ẋ, θ, θ̇]; θ = π is hanging down, θ = 0 is upright
    t: u64,
}

#[wasm_bindgen]
impl GpuCartpoleRef {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GpuCartpoleRef {
        GpuCartpoleRef { cp: Cartpole::default(), state: [0.0, 0.0, std::f64::consts::PI, 0.0], t: 0 }
    }

    pub fn b(&self) -> usize {
        B
    }
    pub fn horizon(&self) -> usize {
        HORIZON
    }
    pub fn cart_x(&self) -> f64 {
        self.state[0]
    }
    pub fn theta(&self) -> f64 {
        self.state[2]
    }
    /// Upright error `1 − cos θ` (0 = balanced) — the receipt the swing-up is judged by.
    pub fn upright_error(&self) -> f64 {
        1.0 - self.state[2].cos()
    }

    /// `[mc, mp, l, g, dt, horizon, b, x0, ẋ0, θ0, θ̇0]` — physical params plus the current state,
    /// packed for the WGSL rollout kernel.
    pub fn params(&self) -> Vec<f32> {
        vec![
            self.cp.mc as f32, self.cp.mp as f32, self.cp.l as f32, self.cp.g as f32, DT as f32,
            HORIZON as f32, B as f32,
            self.state[0] as f32, self.state[1] as f32, self.state[2] as f32, self.state[3] as f32,
        ]
    }

    /// The `B` candidate forces for the current control step (deterministic splitmix64, reseeded by
    /// the step counter) — the same values fed to the GPU kernel and to [`Self::cpu_finals`].
    pub fn candidates(&self) -> Vec<f32> {
        let mut s = 0x1000u64.wrapping_add(self.t);
        (0..B)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                let v = (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
                (FORCE * v) as f32
            })
            .collect()
    }

    /// The VERIFIED CPU render: roll every candidate out from the current state through the cartpole
    /// dynamics for `HORIZON` steps; return the packed final states `[x,ẋ,θ,θ̇, …]` (`4·B`). The GPU
    /// kernel is checked against this.
    pub fn cpu_finals(&self, cands: &[f32]) -> Vec<f32> {
        let x0: Vec<f64> = (0..B).flat_map(|_| self.state).collect();
        let acts: Vec<f64> = cands.iter().map(|&f| f as f64).collect();
        batch_rollout(&self.cp, &x0, &acts, HORIZON, DT).iter().map(|&v| v as f32).collect()
    }

    /// Execute the chosen candidate's force for one real step (receding horizon) and advance the step
    /// counter. Returns the new upright error.
    pub fn apply(&mut self, force: f64) -> f64 {
        self.state = self.cp.step(self.state, force, DT);
        self.t = self.t.wrapping_add(1);
        self.upright_error()
    }

    pub fn reset(&mut self) {
        self.state = [0.0, 0.0, std::f64::consts::PI, 0.0];
        self.t = 0;
    }
}

impl Default for GpuCartpoleRef {
    fn default() -> Self {
        Self::new()
    }
}
