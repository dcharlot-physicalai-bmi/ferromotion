//! **Central Pattern Generators** (Ijspeert, *Neural Networks* 2008; Righetti & Ijspeert) — networks of
//! coupled nonlinear oscillators that produce smoothly-modulable rhythmic gaits, the cheap robust rhythm
//! layer under many legged robots (and the base signal for RL/CPG hybrids). Each limb is a **Hopf
//! oscillator** with a stable limit cycle of amplitude `√μ` and frequency `ω`,
//!
//! ```text
//!   ẋ = α(μ − r²)x − ω y,   ẏ = α(μ − r²)y + ω x,   r² = x² + y²
//! ```
//!
//! and a **phase-coupling** term drives the oscillators to a prescribed set of relative phases — the gait.
//! Amplitude, frequency, and phase offsets are each an independent knob, so gait transitions are just
//! parameter changes. Verified against the analytic limit-cycle amplitude/period, its global stability,
//! and phase-locking to a commanded pattern. Pure Rust, deterministic → WASM-clean.

/// A Hopf oscillator: limit-cycle amplitude `√μ`, angular frequency `omega`, convergence rate `alpha`.
#[derive(Clone, Copy, Debug)]
pub struct HopfOscillator {
    pub mu: f64,
    pub omega: f64,
    pub alpha: f64,
}

impl HopfOscillator {
    /// The uncoupled state derivative at `(x, y)`.
    pub fn deriv(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let g = self.alpha * (self.mu - r2);
        (g * x - self.omega * y, g * y + self.omega * x)
    }
    /// Limit-cycle amplitude `√μ`.
    pub fn amplitude(&self) -> f64 {
        self.mu.sqrt()
    }
    /// Limit-cycle period `2π/ω`.
    pub fn period(&self) -> f64 {
        std::f64::consts::TAU / self.omega
    }
}

/// A network of coupled Hopf oscillators. `phase_bias[i][j]` is the desired phase of `i` minus that of `j`
/// (antisymmetric); `coupling` is the interaction strength.
#[derive(Clone, Debug)]
pub struct CpgNetwork {
    pub osc: Vec<HopfOscillator>,
    pub states: Vec<[f64; 2]>,
    pub coupling: f64,
    pub phase_bias: Vec<Vec<f64>>,
}

impl CpgNetwork {
    fn n(&self) -> usize {
        self.osc.len()
    }

    /// Full coupled derivative of the stacked state.
    fn deriv(&self, s: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let n = self.n();
        let mut d = vec![[0.0; 2]; n];
        for i in 0..n {
            let (mut dx, mut dy) = self.osc[i].deriv(s[i][0], s[i][1]);
            // phase coupling: pull i toward phase(j) + bias_ij  (rotation of j's state)
            for (j, sj) in s.iter().enumerate() {
                if i == j {
                    continue;
                }
                let (c, sn) = self.phase_bias[i][j].sin_cos();
                // R(bias)·[x_j; y_j]
                dx += self.coupling * (sn * sj[0] - c * sj[1]);
                dy += self.coupling * (c * sj[0] + sn * sj[1]);
            }
            d[i] = [dx, dy];
        }
        d
    }

    /// Advance the network by `dt` with a classic RK4 step.
    pub fn step(&mut self, dt: f64) {
        let s = self.states.clone();
        let add = |a: &[[f64; 2]], b: &[[f64; 2]], h: f64| -> Vec<[f64; 2]> {
            a.iter().zip(b).map(|(u, v)| [u[0] + h * v[0], u[1] + h * v[1]]).collect()
        };
        let k1 = self.deriv(&s);
        let k2 = self.deriv(&add(&s, &k1, dt / 2.0));
        let k3 = self.deriv(&add(&s, &k2, dt / 2.0));
        let k4 = self.deriv(&add(&s, &k3, dt));
        for i in 0..self.n() {
            for c in 0..2 {
                self.states[i][c] += dt / 6.0 * (k1[i][c] + 2.0 * k2[i][c] + 2.0 * k3[i][c] + k4[i][c]);
            }
        }
    }

    /// Phase of oscillator `i` (`atan2(y, x)`).
    pub fn phase(&self, i: usize) -> f64 {
        self.states[i][1].atan2(self.states[i][0])
    }
    /// Amplitude (radius) of oscillator `i`.
    pub fn radius(&self, i: usize) -> f64 {
        (self.states[i][0].powi(2) + self.states[i][1].powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap an angle to `(−π, π]`.
    fn wrap(a: f64) -> f64 {
        let mut x = a % std::f64::consts::TAU;
        if x > std::f64::consts::PI {
            x -= std::f64::consts::TAU;
        }
        if x <= -std::f64::consts::PI {
            x += std::f64::consts::TAU;
        }
        x
    }

    #[test]
    fn a_single_oscillator_reaches_the_analytic_limit_cycle() {
        // From a small perturbation, the amplitude converges to √μ and the period to 2π/ω.
        let osc = HopfOscillator { mu: 0.25, omega: 3.0, alpha: 5.0 };
        let mut net = CpgNetwork { osc: vec![osc], states: vec![[0.05, 0.0]], coupling: 0.0, phase_bias: vec![vec![0.0]] };
        let dt = 1e-3;
        for _ in 0..10000 {
            net.step(dt);
        }
        assert!((net.radius(0) - osc.amplitude()).abs() < 1e-3, "amplitude {} vs √μ {}", net.radius(0), osc.amplitude());
        // measure the period from successive +x-axis crossings (y: − → +)
        let mut prev_y = net.states[0][1];
        let mut t = 0.0;
        let mut crossings = vec![];
        for _ in 0..20000 {
            net.step(dt);
            t += dt;
            let y = net.states[0][1];
            if prev_y < 0.0 && y >= 0.0 && net.states[0][0] > 0.0 {
                crossings.push(t);
            }
            prev_y = y;
        }
        let period = crossings[1] - crossings[0];
        assert!((period - osc.period()).abs() < 2e-3, "period {period} vs 2π/ω {}", osc.period());
    }

    #[test]
    fn the_limit_cycle_is_a_global_attractor() {
        // Trajectories from well inside and well outside both converge to radius √μ.
        let osc = HopfOscillator { mu: 1.0, omega: 2.0, alpha: 4.0 };
        for &r0 in &[0.05_f64, 3.0] {
            let mut net = CpgNetwork { osc: vec![osc], states: vec![[r0, 0.0]], coupling: 0.0, phase_bias: vec![vec![0.0]] };
            for _ in 0..6000 {
                net.step(1e-3);
            }
            assert!((net.radius(0) - 1.0).abs() < 1e-3, "from r0={r0}, radius {} should → 1", net.radius(0));
        }
    }

    #[test]
    fn a_coupled_pair_phase_locks_to_the_commanded_offset() {
        // THE HEADLINE. Two oscillators commanded to anti-phase (π, a biped's legs) lock to Δφ = π; the
        // same network commanded in-phase (0) locks to Δφ = 0. The relative phase is the gait.
        let osc = HopfOscillator { mu: 1.0, omega: 4.0, alpha: 5.0 };
        for &target in &[std::f64::consts::PI, 0.0] {
            let bias = vec![vec![0.0, target], vec![-target, 0.0]]; // phase(0) − phase(1) = target
            let mut net = CpgNetwork {
                osc: vec![osc, osc],
                states: vec![[1.0, 0.0], [0.9, 0.3]], // arbitrary distinct start
                coupling: 2.0,
                phase_bias: bias,
            };
            for _ in 0..8000 {
                net.step(1e-3);
            }
            let dphi = wrap(net.phase(0) - net.phase(1));
            assert!(wrap(dphi - target).abs() < 0.05, "Δφ {dphi} should lock to {target}");
        }
    }

    #[test]
    fn frequency_sets_the_rhythm() {
        // Doubling ω halves the period — the frequency knob is analytic and independent of amplitude.
        let slow = HopfOscillator { mu: 1.0, omega: 2.0, alpha: 5.0 };
        let fast = HopfOscillator { mu: 1.0, omega: 4.0, alpha: 5.0 };
        assert!((slow.period() / fast.period() - 2.0).abs() < 1e-12, "period should scale as 1/ω");
    }
}
