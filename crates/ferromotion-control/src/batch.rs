//! **Batched / data-parallel rigid-body rollouts** — the on-device-RL axis (Brax/MJX/Newton
//! simulate thousands of environment instances in parallel for gradient-free and gradient-based
//! policy search). This is the browser-deployable counterpart: a structure-of-arrays batched
//! cartpole — the canonical underactuated rigid-body RL benchmark — whose step advances all `B`
//! instances through one vectorizable inner loop (each instance independent, no cross-lane data
//! dependence, so it auto-vectorizes on the CPU and maps directly to wasm-SIMD or a WebGPU compute
//! kernel — the throughput "optimize" leg). Verified: every batched instance reproduces the scalar
//! single-instance rollout bit-for-bit, and a batched random-shooting controller swings the pole up.

/// Cartpole parameters (a cart of mass `mc` carrying a pole of mass `mp`, length `l`, gravity `g`).
#[derive(Clone, Copy, Debug)]
pub struct Cartpole {
    pub mc: f64,
    pub mp: f64,
    pub l: f64,
    pub g: f64,
}

impl Default for Cartpole {
    fn default() -> Self {
        Cartpole { mc: 1.0, mp: 0.1, l: 0.5, g: 9.81 }
    }
}

impl Cartpole {
    /// Continuous accelerations `(ẍ, θ̈)` for state `[x, ẋ, θ, θ̇]` under horizontal force `u`.
    #[inline]
    fn accel(&self, s: [f64; 4], u: f64) -> (f64, f64) {
        let (th, thd) = (s[2], s[3]);
        let (st, ct) = (th.sin(), th.cos());
        let total = self.mc + self.mp;
        let temp = (u + self.mp * self.l * thd * thd * st) / total;
        let thdd = (self.g * st - ct * temp) / (self.l * (4.0 / 3.0 - self.mp * ct * ct / total));
        let xdd = temp - self.mp * self.l * thdd * ct / total;
        (xdd, thdd)
    }

    /// One semi-implicit Euler step of a single instance.
    pub fn step(&self, s: [f64; 4], u: f64, dt: f64) -> [f64; 4] {
        let (xdd, thdd) = self.accel(s, u);
        let xd = s[1] + dt * xdd;
        let thd = s[3] + dt * thdd;
        [s[0] + dt * xd, xd, s[2] + dt * thd, thd]
    }

    /// **Batched step**: advance `b = state.len()/4` instances in place. State is packed
    /// `[x,ẋ,θ,θ̇]` per instance (contiguous), `actions[i]` the force on instance `i`. The loop body
    /// touches only instance `i`'s lane — embarrassingly parallel.
    pub fn batch_step(&self, state: &mut [f64], actions: &[f64], dt: f64) {
        let b = state.len() / 4;
        debug_assert_eq!(actions.len(), b);
        for i in 0..b {
            let s = [state[4 * i], state[4 * i + 1], state[4 * i + 2], state[4 * i + 3]];
            let ns = self.step(s, actions[i], dt);
            state[4 * i..4 * i + 4].copy_from_slice(&ns);
        }
    }
}

/// Roll out `b` instances (packed `x0`, length `4b`) for `steps` under a constant per-instance
/// action, returning the final packed state. The batched workhorse behind random-shooting / MPPI.
pub fn batch_rollout(cp: &Cartpole, x0: &[f64], actions: &[f64], steps: usize, dt: f64) -> Vec<f64> {
    let mut state = x0.to_vec();
    for _ in 0..steps {
        cp.batch_step(&mut state, actions, dt);
    }
    state
}

#[cfg(test)]
mod verification {
    use super::*;

    // A deterministic normal-ish sample stream (splitmix64) — no RNG crate, reproducible.
    fn stream(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                (((z ^ (z >> 31)) as f64) / (u64::MAX as f64)) * 2.0 - 1.0
            })
            .collect()
    }

    /// Every batched instance reproduces the scalar single-instance rollout exactly — the batching
    /// is a pure data layout, not an approximation.
    #[test]
    fn batched_matches_scalar_per_instance() {
        let cp = Cartpole::default();
        let b = 256;
        let x0 = stream(4 * b, 0x11);
        let actions: Vec<f64> = stream(b, 0x22).iter().map(|v| 8.0 * v).collect();
        let dt = 0.01;
        let steps = 60;

        let batched = batch_rollout(&cp, &x0, &actions, steps, dt);
        let mut worst = 0.0f64;
        for i in 0..b {
            let mut s = [x0[4 * i], x0[4 * i + 1], x0[4 * i + 2], x0[4 * i + 3]];
            for _ in 0..steps {
                s = cp.step(s, actions[i], dt);
            }
            for d in 0..4 {
                worst = worst.max((batched[4 * i + d] - s[d]).abs());
            }
        }
        eprintln!("batched vs scalar over {b} instances × {steps} steps: worst diff {worst:.2e}");
        assert_eq!(worst, 0.0, "batched rollout diverged from scalar: {worst}");
    }

    /// **The payoff: batched random-shooting swing-up.** Sample `B` action sequences, roll them all
    /// out in parallel (the batched primitive), pick the one whose cost is lowest, execute its first
    /// action, and repeat (receding horizon). The pole swings from hanging-down to upright.
    #[test]
    fn batched_random_shooting_swings_up() {
        let cp = Cartpole::default();
        let dt = 0.02;
        let horizon = 25;
        let b = 400; // parallel candidate action sequences
        let mut x = [0.0, 0.0, std::f64::consts::PI, 0.0]; // pole hanging DOWN
        let mut best_cost_final = f64::INFINITY;
        for t in 0..120 {
            // sample B action sequences (constant-per-sequence force for simplicity)
            let acts: Vec<f64> = stream(b, 0x1000 + t as u64).iter().map(|v| 12.0 * v).collect();
            // batched parallel rollout of all candidates from the current state
            let x0: Vec<f64> = (0..b).flat_map(|_| x).collect();
            let finals = batch_rollout(&cp, &x0, &acts, horizon, dt);
            // cost: upright (θ=0 ⇒ cosθ=1) + small cart offset + low velocities
            let (mut best_i, mut best_c) = (0usize, f64::INFINITY);
            for i in 0..b {
                let s = &finals[4 * i..4 * i + 4];
                let c = (1.0 - s[2].cos()) * 5.0 + 0.1 * s[0] * s[0] + 0.02 * s[3] * s[3];
                if c < best_c {
                    best_c = c;
                    best_i = i;
                }
            }
            // execute the winning action for one real step
            x = cp.step(x, acts[best_i], dt);
            best_cost_final = (1.0 - x[2].cos()).abs();
        }
        eprintln!("batched shooting swing-up: final upright error 1−cosθ = {best_cost_final:.4}");
        assert!(best_cost_final < 0.1, "pole did not swing up: {best_cost_final}");
    }
}
