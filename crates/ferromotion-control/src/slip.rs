//! **SLIP — the spring-loaded inverted pendulum**, the canonical model of *running* and hopping
//! (Blickhan; Raibert's hoppers), and the counterpart to the LIPM/[`crate::DcmPlan`] walking models
//! already here. A point mass on a massless springy leg reproduces the ground-reaction forces of
//! running animals across an enormous size range.
//!
//! The gait is a **hybrid system** alternating two phases:
//! * **flight** — ballistic, with the leg held at a fixed *angle of attack* `α`;
//! * **stance** — the foot is pinned and the mass rides a radial spring, `F = k(L₀ − L)`.
//!
//! Touchdown fires when the foot reaches the ground (`z = L₀·cos α`, descending), liftoff when the leg
//! returns to rest length. Passive SLIP is **conservative**, so the **apex return map** (apex → apex,
//! the Poincaré section at `ż = 0`) preserves `E = ½m‖v‖² + mgz` — the invariant that pins down the
//! whole hybrid simulation. Raibert's insight was that **foot placement alone** steers forward speed,
//! trading it against hop height at constant energy. Pure Rust → WASM-clean.

/// SLIP parameters.
#[derive(Clone, Copy, Debug)]
pub struct Slip {
    pub m: f64,
    /// Leg spring stiffness.
    pub k: f64,
    /// Leg rest length.
    pub l0: f64,
    pub g: f64,
}

/// Point-mass state.
#[derive(Clone, Copy, Debug)]
pub struct SlipState {
    pub x: f64,
    pub z: f64,
    pub vx: f64,
    pub vz: f64,
}

/// Gait phase; `Stance` carries the pinned foot's `x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Flight,
    Stance(f64),
}

impl Slip {
    /// Current leg length during stance.
    fn leg_len(&self, s: &SlipState, foot_x: f64) -> f64 {
        ((s.x - foot_x).powi(2) + s.z * s.z).sqrt()
    }

    /// Rate of change of leg length, `dL/dt` (negative while compressing).
    fn leg_rate(&self, s: &SlipState, foot_x: f64) -> f64 {
        let l = self.leg_len(s, foot_x).max(1e-9);
        ((s.x - foot_x) * s.vx + s.z * s.vz) / l
    }

    /// Total mechanical energy (kinetic + gravitational + spring).
    pub fn energy(&self, s: &SlipState, phase: Phase) -> f64 {
        let ke = 0.5 * self.m * (s.vx * s.vx + s.vz * s.vz);
        let pe = self.m * self.g * s.z;
        let spring = match phase {
            Phase::Stance(fx) => {
                let l = self.leg_len(s, fx);
                if l < self.l0 { 0.5 * self.k * (self.l0 - l).powi(2) } else { 0.0 }
            }
            Phase::Flight => 0.0,
        };
        ke + pe + spring
    }

    fn accel(&self, s: &SlipState, phase: Phase) -> (f64, f64) {
        match phase {
            Phase::Flight => (0.0, -self.g),
            Phase::Stance(fx) => {
                let (dx, dz) = (s.x - fx, s.z);
                let l = (dx * dx + dz * dz).sqrt().max(1e-9);
                let f = self.k * (self.l0 - l) / self.m; // radial spring, foot → mass
                (f * dx / l, f * dz / l - self.g)
            }
        }
    }

    /// One semi-implicit Euler step with touchdown/liftoff event detection.
    pub fn step(&self, s: SlipState, phase: Phase, dt: f64, alpha: f64) -> (SlipState, Phase) {
        let (ax, az) = self.accel(&s, phase);
        let vx = s.vx + ax * dt;
        let vz = s.vz + az * dt;
        let n = SlipState { x: s.x + vx * dt, z: s.z + vz * dt, vx, vz };
        let next_phase = match phase {
            Phase::Flight => {
                // Touchdown: the foot (held at angle α) reaches the ground while descending.
                if n.z <= self.l0 * alpha.cos() && n.vz < 0.0 {
                    Phase::Stance(n.x + self.l0 * alpha.sin())
                } else {
                    Phase::Flight
                }
            }
            Phase::Stance(fx) => {
                // Liftoff: the leg has re-extended to rest length *while lengthening*. The extension
                // check matters: at touchdown the leg is exactly L₀ but compressing, so testing
                // `L ≥ L₀` alone would lift off immediately and the hop would never happen.
                if self.leg_len(&n, fx) >= self.l0 && self.leg_rate(&n, fx) > 0.0 {
                    Phase::Flight
                } else {
                    phase
                }
            }
        };
        (n, next_phase)
    }

    /// **Apex return map**: from an apex `(z, vx)` (where `ż = 0`), simulate one flight–stance–flight
    /// cycle at angle of attack `alpha` and return the next apex. `None` if the hop fails (a fall).
    pub fn apex_map(&self, z: f64, vx: f64, alpha: f64, dt: f64) -> Option<(f64, f64)> {
        let mut s = SlipState { x: 0.0, z, vx, vz: 0.0 };
        let mut phase = Phase::Flight;
        let mut had_stance = false;
        let mut prev_vz = 0.0;
        for _ in 0..2_000_000 {
            let (ns, np) = self.step(s, phase, dt, alpha);
            if matches!(np, Phase::Stance(_)) {
                had_stance = true;
            }
            // Apex after the stance: vertical velocity crosses from rising to falling.
            if had_stance && np == Phase::Flight && prev_vz > 0.0 && ns.vz <= 0.0 {
                return Some((ns.z, ns.vx));
            }
            if ns.z <= 0.0 {
                return None; // fell
            }
            prev_vz = ns.vz;
            s = ns;
            phase = np;
        }
        None
    }

    /// Raibert foot placement: the angle of attack that steers forward speed toward `vx_des`.
    /// The neutral point `vx·T_s/2` keeps the current speed; the gain term corrects the error.
    pub fn raibert_alpha(&self, vx: f64, vx_des: f64, k_v: f64) -> f64 {
        let t_stance = std::f64::consts::PI * (self.m / self.k).sqrt(); // spring half-period
        let offset = vx * t_stance / 2.0 + k_v * (vx - vx_des);
        (offset / self.l0).clamp(-0.9, 0.9).asin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slip() -> Slip {
        Slip { m: 1.0, k: 1000.0, l0: 1.0, g: 9.81 }
    }

    #[test]
    fn energy_is_conserved_across_flight_stance_and_the_events() {
        let sp = slip();
        let alpha = 0.25;
        let mut s = SlipState { x: 0.0, z: 1.05, vx: 1.0, vz: 0.0 };
        let mut phase = Phase::Flight;
        let e0 = sp.energy(&s, phase);
        let (dt, mut worst, mut saw_stance) = (1e-6, 0.0f64, false);
        for _ in 0..600_000 {
            let (ns, np) = sp.step(s, phase, dt, alpha);
            if matches!(np, Phase::Stance(_)) {
                saw_stance = true;
            }
            s = ns;
            phase = np;
            worst = worst.max((sp.energy(&s, phase) - e0).abs() / e0);
        }
        assert!(saw_stance, "never touched down — test not exercising stance");
        assert!(worst < 1e-4, "energy not conserved through the hybrid dynamics: {worst:.2e}");
    }

    #[test]
    fn the_apex_return_map_preserves_energy() {
        // Passive SLIP is conservative ⇒ apex energy E = ½m·vx² + m·g·z is invariant apex-to-apex.
        let sp = slip();
        let (z0, vx0) = (1.05, 1.0);
        // Use the *neutral* angle of attack (foot at the non-braking point vx·T_s/2), so the gait is
        // near-symmetric and sustains. A much steeper α brakes hard and the hop degenerates.
        let alpha = sp.raibert_alpha(vx0, vx0, 0.0);
        let e = |z: f64, vx: f64| 0.5 * sp.m * vx * vx + sp.m * sp.g * z;
        let (mut z, mut vx) = (z0, vx0);
        let e0 = e(z, vx);
        let mut hops = 0;
        for _ in 0..5 {
            let Some((nz, nvx)) = sp.apex_map(z, vx, alpha, 1e-6) else { break };
            let rel = (e(nz, nvx) - e0).abs() / e0;
            assert!(rel < 1e-4, "apex energy drifted on hop {hops}: {rel:.2e}");
            z = nz;
            vx = nvx;
            hops += 1;
        }
        assert!(hops >= 3, "expected a sustained hop, only got {hops}");
        // And it really hopped (the state actually changed).
        assert!((z - z0).abs() + (vx - vx0).abs() > 1e-6, "apex map was a no-op");
    }

    #[test]
    fn raibert_foot_placement_regulates_forward_speed() {
        // Foot placement trades forward speed against hop height at constant energy.
        let sp = slip();
        let (mut z, mut vx) = (1.06, 0.4);
        let vx_des: f64 = 1.1;
        let err0: f64 = (vx - vx_des).abs();
        for _ in 0..25 {
            let alpha = sp.raibert_alpha(vx, vx_des, 0.15);
            match sp.apex_map(z, vx, alpha, 1e-5) {
                Some((nz, nvx)) => {
                    z = nz;
                    vx = nvx;
                }
                None => break, // fell; leave the last good apex
            }
        }
        let err: f64 = (vx - vx_des).abs();
        assert!(err < 0.5 * err0, "Raibert control did not steer toward the target speed: {vx} vs {vx_des} (started {err0} off)");
    }
}
