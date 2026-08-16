//! **Field-oriented control** — the layer between a commanded torque and the currents that produce it.
//!
//! [`DcMotor`](crate::DcMotor) models a brushed motor: one winding, `V = R i + L di/dt + k_e ω`, torque
//! `k_t i`. Almost no actuator on a modern robot is that. They are permanent-magnet synchronous machines
//! (brushless), driven by three phase currents through an inverter, and a commanded torque only becomes a
//! torque after a rotor-angle-dependent coordinate transform, a current regulator, and a modulator that has to
//! fit the result inside a fixed DC bus.
//!
//! Three things that layer determines, none of which a `k_t i` model can express:
//!
//! * **The torque constant is not constant.** On a salient (interior-magnet) machine the reluctance term
//!   `(L_d − L_q) i_d i_q` contributes real torque, so the same current magnitude produces different torque
//!   depending on how it is *split* between axes. [`Pmsm::mtpa`] finds the split that costs the least current,
//!   which is the difference between a motor that overheats and one that does not.
//! * **There is a speed past which the commanded torque is unavailable at any current**, because the back-EMF
//!   plus the resistive and inductive drops exceed what the bus can supply. [`Pmsm::base_speed`] is where that
//!   starts and [`Pmsm::field_weakening_id`] is what extends it.
//! * **How much of the bus is usable depends on the modulator.** Space-vector modulation reaches a peak phase
//!   voltage of `V_dc/√3`; naive sine-triangle modulation reaches `V_dc/2`. That is **15.47% more voltage** for
//!   the same hardware, and therefore 15.47% more speed before field weakening is needed, from a change of
//!   arithmetic alone. Measured in [`svpwm_duties`]'s tests rather than asserted.
//!
//! # Conventions, stated because half the errors in this subject are convention errors
//!
//! The Clarke transform here is the **amplitude-invariant** (2/3) form: a balanced set of phase currents with
//! peak `I` maps to a `dq` vector of magnitude `I`. It is *not* power-invariant, and the consequence appears
//! everywhere downstream as a factor of **3/2**:
//!
//! ```text
//!   torque       T    = (3/2) p [λ_m i_q + (L_d − L_q) i_d i_q]
//!   copper loss  P_cu = (3/2) R (i_d² + i_q²)          = R (i_a² + i_b² + i_c²)
//!   power        P    = (3/2) (v_d i_d + v_q i_q)      = v_a i_a + v_b i_b + v_c i_c
//! ```
//!
//! Each of those right-hand equalities is asserted as an identity in the tests, because a missing 3/2 is a
//! 50% torque error that still produces a plausible-looking simulation.
//!
//! `λ_m` is the peak per-phase magnet flux linkage and `p` is **pole pairs**, not poles. Electrical angle is
//! `p` times mechanical angle, and confusing the two is a factor of `p` in the commutation that manifests as
//! a motor that runs rough and produces a fraction of its rated torque.
//!
//! # Connecting to the rest of the stack
//!
//! [`Pmsm::copper_loss`] is in watts and feeds [`MotorThermal`](crate::MotorThermal) directly; the equivalent
//! DC current that module wants is [`Pmsm::thermal_equivalent_current`]. The bus voltage a
//! [`Battery`](crate::Battery) can actually hold under load is `OCV − I R_int`, which is *lower* than its
//! nameplate and drops as the pack drains, so the achievable speed of a joint is a function of state of
//! charge. That coupling is the reason these modules are worth having in one place.

use std::f64::consts::PI;

/// `√3`, and `1/√3`, to `f64` precision.
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// **Clarke transform** (amplitude-invariant, 2/3 convention): three phase quantities to the stationary
/// two-axis `αβ` frame.
///
/// A balanced set with peak `I` maps to `|(α, β)| = I`. The zero-sequence component is discarded, which is
/// correct for a star-connected machine with no neutral connection: it cannot carry zero-sequence current, so
/// any that appears in a command is unrealizable and silently dropping it is the honest behaviour.
pub fn clarke(a: f64, b: f64, c: f64) -> (f64, f64) {
    let alpha = (2.0 / 3.0) * (a - 0.5 * b - 0.5 * c);
    let beta = (b - c) / SQRT3;
    (alpha, beta)
}

/// **Inverse Clarke**: `αβ` back to three phase quantities, with zero zero-sequence (they sum to zero).
pub fn inverse_clarke(alpha: f64, beta: f64) -> (f64, f64, f64) {
    let a = alpha;
    let b = -0.5 * alpha + (SQRT3 / 2.0) * beta;
    let c = -0.5 * alpha - (SQRT3 / 2.0) * beta;
    (a, b, c)
}

/// **Park transform**: stationary `αβ` to the rotor-synchronous `dq` frame at electrical angle `theta_e`.
///
/// This is the transform that makes field-oriented control possible: a balanced sinusoid at the electrical
/// frequency, which is a *rotating* quantity in `αβ`, becomes **constant** in `dq`. A constant is something a
/// PI regulator can drive to zero error; a sinusoid is not.
pub fn park(alpha: f64, beta: f64, theta_e: f64) -> (f64, f64) {
    let (s, c) = theta_e.sin_cos();
    (alpha * c + beta * s, -alpha * s + beta * c)
}

/// **Inverse Park**: `dq` back to `αβ`.
pub fn inverse_park(d: f64, q: f64, theta_e: f64) -> (f64, f64) {
    let (s, c) = theta_e.sin_cos();
    (d * c - q * s, d * s + q * c)
}

/// Duty cycles for **space-vector modulation**, by common-mode (min-max) injection.
///
/// Returns three duties in the ideal case within `[0, 1]`, `0.5` being zero output. Values outside that range
/// mean the commanded vector does not fit in the bus and the inverter will saturate; they are returned
/// **unclamped** so a caller can detect it, and [`modulation_index`] reports how far over.
///
/// The mechanism: adding the same offset to all three phases does not change any line-to-line voltage, and a
/// three-phase load only responds to line-to-line voltage. So the offset is free, and choosing it to centre
/// the phase voltages between the rails buys headroom. Measured: duties stay inside `[0, 1]` up to a peak
/// phase voltage of exactly `V_dc/√3`, against `V_dc/2` without injection — the origin of the 15.47%.
pub fn svpwm_duties(v_alpha: f64, v_beta: f64, v_dc: f64) -> [f64; 3] {
    let (va, vb, vc) = inverse_clarke(v_alpha, v_beta);
    let max = va.max(vb).max(vc);
    let min = va.min(vb).min(vc);
    // Centre the excursion: this is what third-harmonic injection accomplishes, in one line and exactly.
    let offset = -0.5 * (max + min);
    [
        0.5 + (va + offset) / v_dc,
        0.5 + (vb + offset) / v_dc,
        0.5 + (vc + offset) / v_dc,
    ]
}

/// Duty cycles for **sine-triangle modulation** — no common-mode injection.
///
/// Kept because it is the comparison that makes SVPWM's advantage a measurement rather than a claim. It
/// saturates at a peak phase voltage of `V_dc/2`.
pub fn spwm_duties(v_alpha: f64, v_beta: f64, v_dc: f64) -> [f64; 3] {
    let (va, vb, vc) = inverse_clarke(v_alpha, v_beta);
    [0.5 + va / v_dc, 0.5 + vb / v_dc, 0.5 + vc / v_dc]
}

/// Peak phase voltage available in the linear region, for space-vector modulation: `V_dc/√3`.
pub fn svpwm_voltage_limit(v_dc: f64) -> f64 {
    v_dc / SQRT3
}

/// Peak phase voltage available for sine-triangle modulation: `V_dc/2`.
pub fn spwm_voltage_limit(v_dc: f64) -> f64 {
    0.5 * v_dc
}

/// Commanded voltage magnitude as a fraction of the space-vector limit. `> 1.0` means saturation.
pub fn modulation_index(v_d: f64, v_q: f64, v_dc: f64) -> f64 {
    (v_d * v_d + v_q * v_q).sqrt() / svpwm_voltage_limit(v_dc)
}

/// A permanent-magnet synchronous machine in the `dq` frame.
///
/// Set `l_d == l_q` for a surface-magnet machine (no reluctance torque, and MTPA is exactly `i_d = 0`). For an
/// interior-magnet machine `l_d < l_q`, and the reluctance term earns torque at **negative** `i_d`.
#[derive(Clone, Copy, Debug)]
pub struct Pmsm {
    /// Pole **pairs**. Electrical angle is `pole_pairs` times mechanical angle.
    pub pole_pairs: f64,
    /// Per-phase stator resistance (Ω).
    pub r_s: f64,
    /// Direct-axis inductance (H).
    pub l_d: f64,
    /// Quadrature-axis inductance (H).
    pub l_q: f64,
    /// Peak per-phase permanent-magnet flux linkage (Wb).
    pub flux_linkage: f64,
    /// Direct-axis current (A).
    pub i_d: f64,
    /// Quadrature-axis current (A).
    pub i_q: f64,
}

impl Pmsm {
    /// A surface-magnet machine, `l_d == l_q == l`.
    pub fn surface(pole_pairs: f64, r_s: f64, l: f64, flux_linkage: f64) -> Pmsm {
        Pmsm { pole_pairs, r_s, l_d: l, l_q: l, flux_linkage, i_d: 0.0, i_q: 0.0 }
    }

    /// An interior-magnet (salient) machine. `l_d` should be the smaller of the two.
    pub fn interior(pole_pairs: f64, r_s: f64, l_d: f64, l_q: f64, flux_linkage: f64) -> Pmsm {
        Pmsm { pole_pairs, r_s, l_d, l_q, flux_linkage, i_d: 0.0, i_q: 0.0 }
    }

    /// Whether the machine is salient enough for the reluctance term to matter.
    pub fn is_salient(&self) -> bool {
        (self.l_q - self.l_d).abs() > 1e-12
    }

    /// Electromagnetic torque (N·m) from a `dq` current pair.
    ///
    /// `T = (3/2) p [λ_m i_q + (L_d − L_q) i_d i_q]`. The second term is reluctance torque; on an interior
    /// machine `L_d − L_q < 0`, so it is **positive for negative `i_d`**, which is why MTPA and field
    /// weakening both push `i_d` negative rather than positive.
    pub fn torque_at(&self, i_d: f64, i_q: f64) -> f64 {
        1.5 * self.pole_pairs * (self.flux_linkage * i_q + (self.l_d - self.l_q) * i_d * i_q)
    }

    /// Torque at the machine's current state.
    pub fn torque(&self) -> f64 {
        self.torque_at(self.i_d, self.i_q)
    }

    /// Copper loss (W) at a `dq` current pair: `(3/2) R (i_d² + i_q²)`.
    ///
    /// The 3/2 is the amplitude-invariant Clarke convention showing up, and it is exactly
    /// `R (i_a² + i_b² + i_c²)` — asserted, because dropping it understates heating by a third.
    pub fn copper_loss_at(&self, i_d: f64, i_q: f64) -> f64 {
        1.5 * self.r_s * (i_d * i_d + i_q * i_q)
    }

    /// Copper loss (W) at the current state.
    pub fn copper_loss(&self) -> f64 {
        self.copper_loss_at(self.i_d, self.i_q)
    }

    /// The single DC current that dissipates the same power in one phase resistance, for handing to
    /// [`MotorThermal`](crate::MotorThermal), whose model is a single `R`.
    ///
    /// `P_cu = (3/2) R I_dq²` and `MotorThermal::copper_loss(i) = R i²`, so the equivalent is
    /// `I_dq √(3/2)`. Passing the `dq` magnitude directly would understate the heat by a third.
    pub fn thermal_equivalent_current(&self) -> f64 {
        (self.i_d * self.i_d + self.i_q * self.i_q).sqrt() * (1.5f64).sqrt()
    }

    /// Steady-state `dq` terminal voltages at electrical speed `omega_e` (rad/s electrical).
    ///
    /// ```text
    ///   v_d = R i_d − ω_e L_q i_q
    ///   v_q = R i_q + ω_e (L_d i_d + λ_m)
    /// ```
    ///
    /// The cross-coupling terms are why `d` and `q` current loops interact, and the `ω_e λ_m` term is the
    /// back-EMF that eventually consumes the whole bus.
    pub fn steady_voltage(&self, i_d: f64, i_q: f64, omega_e: f64) -> (f64, f64) {
        let v_d = self.r_s * i_d - omega_e * self.l_q * i_q;
        let v_q = self.r_s * i_q + omega_e * (self.l_d * i_d + self.flux_linkage);
        (v_d, v_q)
    }

    /// Advance the `dq` current dynamics one step under applied voltages, by explicit Euler.
    ///
    /// ```text
    ///   L_d di_d/dt = v_d − R i_d + ω_e L_q i_q
    ///   L_q di_q/dt = v_q − R i_q − ω_e (L_d i_d + λ_m)
    /// ```
    ///
    /// Returns the torque produced. `dt` must be below [`max_stable_dt`](Pmsm::max_stable_dt); the electrical
    /// time constant of a robot actuator is often under a millisecond, well below any mechanical step.
    pub fn step(&mut self, dt: f64, v_d: f64, v_q: f64, omega_e: f64) -> f64 {
        let did = (v_d - self.r_s * self.i_d + omega_e * self.l_q * self.i_q) / self.l_d;
        let diq =
            (v_q - self.r_s * self.i_q - omega_e * (self.l_d * self.i_d + self.flux_linkage)) / self.l_q;
        self.i_d += did * dt;
        self.i_q += diq * dt;
        self.torque()
    }

    /// The explicit-Euler stability bound for the **open-loop** machine: `2 L/R` on the slower axis.
    ///
    /// **This is not the bound that matters once a current regulator is closed around it**, and the
    /// difference is not small. This bound describes the plant alone; a PI regulator at bandwidth `ω_bw`
    /// imposes its own, roughly `2/ω_bw`, which for a 500 Hz loop on a 5 ms machine is **15.7 times tighter**.
    /// Stepping at a tenth of *this* bound and closing a 500 Hz loop produced `NaN` on the first attempt at
    /// the test below, from 2 samples per loop period. Use [`PiCurrent::max_stable_dt`] whenever a regulator
    /// is in the loop, and a small fraction of whichever bound is smaller.
    pub fn max_stable_dt(&self) -> f64 {
        2.0 * self.l_d.min(self.l_q) / self.r_s
    }

    /// The `dq` currents of least magnitude that produce `torque`: **maximum torque per amp**.
    ///
    /// For a surface machine this is exactly `i_d = 0` — all current on the `q` axis, no exceptions. For a
    /// salient machine the reluctance term makes a negative `i_d` worth its own copper loss, and the optimum
    /// is found by a one-dimensional search: for each `i_d`, the `i_q` that meets the torque is determined, so
    /// the problem is a scalar minimization of `i_d² + i_q²`.
    ///
    /// Returns `None` if the torque is unreachable, which happens when the required `i_q` would need a torque
    /// coefficient of the wrong sign.
    pub fn mtpa(&self, torque: f64) -> Option<(f64, f64)> {
        if torque == 0.0 {
            return Some((0.0, 0.0));
        }
        if !self.is_salient() {
            let i_q = torque / (1.5 * self.pole_pairs * self.flux_linkage);
            return Some((0.0, i_q));
        }
        // i_q that meets the torque for a given i_d, or None where the coefficient vanishes or flips sign.
        let i_q_for = |i_d: f64| -> Option<f64> {
            let coeff = 1.5 * self.pole_pairs * (self.flux_linkage + (self.l_d - self.l_q) * i_d);
            // Require the coefficient to keep the magnet's sign: past the point where it vanishes the
            // machine is being demagnetised and the branch is not one to operate on.
            if coeff.abs() < 1e-12 || coeff * self.flux_linkage <= 0.0 {
                None
            } else {
                Some(torque / coeff)
            }
        };
        let cost = |i_d: f64| -> Option<f64> { i_q_for(i_d).map(|i_q| i_d * i_d + i_q * i_q) };

        // Bracket: the useful i_d has the sign that makes (l_d − l_q) i_d increase the coefficient. With
        // l_d < l_q that is negative i_d. Scale the search by the current the surface machine would need.
        let scale = (torque / (1.5 * self.pole_pairs * self.flux_linkage)).abs().max(1e-6);
        let sign = if (self.l_d - self.l_q) * self.flux_linkage < 0.0 { -1.0 } else { 1.0 };
        let hi = sign * 8.0 * scale;

        // Golden-section on [0, hi]; the cost is unimodal there because i_q² falls and i_d² rises.
        let phi = 0.618_033_988_749_894_9;
        let (mut lo, mut up) = (0.0f64, hi);
        let mut c1 = up - phi * (up - lo);
        let mut c2 = lo + phi * (up - lo);
        let mut f1 = cost(c1)?;
        let mut f2 = cost(c2)?;
        for _ in 0..200 {
            if f1 < f2 {
                up = c2;
                c2 = c1;
                f2 = f1;
                c1 = up - phi * (up - lo);
                f1 = cost(c1)?;
            } else {
                lo = c1;
                c1 = c2;
                f1 = f2;
                c2 = lo + phi * (up - lo);
                f2 = cost(c2)?;
            }
            if (up - lo).abs() < 1e-14 * (1.0 + up.abs()) {
                break;
            }
        }
        let i_d = 0.5 * (lo + up);
        Some((i_d, i_q_for(i_d)?))
    }

    /// The highest mechanical speed (rad/s) at which `torque` is still reachable within a current limit
    /// `i_max` and a peak phase voltage `v_max`, using the MTPA current split.
    ///
    /// This is **base speed** when called at rated torque: below it, torque is limited by current; above it,
    /// by voltage. Returns `None` if the torque exceeds what `i_max` can produce at any speed.
    pub fn base_speed(&self, torque: f64, i_max: f64, v_max: f64) -> Option<f64> {
        let (i_d, i_q) = self.mtpa(torque)?;
        if (i_d * i_d + i_q * i_q).sqrt() > i_max {
            return None;
        }
        // |v(ω)| grows monotonically in ω for this current, so bisect on the voltage limit.
        let v_mag = |w_mech: f64| -> f64 {
            let (v_d, v_q) = self.steady_voltage(i_d, i_q, w_mech * self.pole_pairs);
            (v_d * v_d + v_q * v_q).sqrt()
        };
        if v_mag(0.0) > v_max {
            return Some(0.0);
        }
        let mut hi = 1.0f64;
        for _ in 0..200 {
            if v_mag(hi) > v_max {
                break;
            }
            hi *= 2.0;
            if hi > 1e12 {
                return None;
            }
        }
        let mut lo = 0.0f64;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if v_mag(mid) > v_max {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// The `i_d` that keeps the voltage magnitude inside `v_max` at mechanical speed `w_mech` while holding
    /// `i_q`: **field weakening**.
    ///
    /// The `ω_e λ_m` back-EMF term is what saturates the bus. Driving `i_d` negative subtracts from the flux
    /// the stator sees (`L_d i_d + λ_m`), buying voltage back at the cost of current that produces no torque
    /// on a surface machine. Returns `None` if no `i_d` within `i_max` fits, which is the honest answer at a
    /// speed the machine cannot reach.
    pub fn field_weakening_id(&self, i_q: f64, w_mech: f64, v_max: f64, i_max: f64) -> Option<f64> {
        let omega_e = w_mech * self.pole_pairs;
        let v_mag = |i_d: f64| -> f64 {
            let (v_d, v_q) = self.steady_voltage(i_d, i_q, omega_e);
            (v_d * v_d + v_q * v_q).sqrt()
        };
        if v_mag(0.0) <= v_max {
            return Some(0.0); // no weakening needed
        }
        // The room left for i_d after i_q has taken its share of the current limit.
        let budget = (i_max * i_max - i_q * i_q).max(0.0).sqrt();
        if budget <= 0.0 {
            return None;
        }
        // v_mag decreases as i_d goes negative, until the flux is over-cancelled. Bisect on [-budget, 0].
        if v_mag(-budget) > v_max {
            return None;
        }
        let (mut lo, mut hi) = (-budget, 0.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if v_mag(mid) > v_max {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }

    /// Three phase currents from the state, at electrical angle `theta_e`. What a current sensor would read.
    pub fn phase_currents(&self, theta_e: f64) -> (f64, f64, f64) {
        let (alpha, beta) = inverse_park(self.i_d, self.i_q, theta_e);
        inverse_clarke(alpha, beta)
    }

    /// Instantaneous electrical input power (W): `(3/2)(v_d i_d + v_q i_q)`.
    pub fn electrical_power(&self, v_d: f64, v_q: f64) -> f64 {
        1.5 * (v_d * self.i_d + v_q * self.i_q)
    }

    /// Mechanical output power (W) at mechanical speed `w_mech`: `T ω`.
    pub fn mechanical_power(&self, w_mech: f64) -> f64 {
        self.torque() * w_mech
    }
}

/// A PI current regulator for one axis, with integral clamping.
///
/// The clamp is not decoration. When the bus saturates the commanded voltage cannot be delivered, the error
/// does not fall, and an unclamped integrator keeps accumulating — so when the machine finally does have
/// headroom, it slams. That is **integrator windup**, and on a robot joint it is a large unexpected motion
/// after a stall.
#[derive(Clone, Copy, Debug)]
pub struct PiCurrent {
    /// Proportional gain (V/A).
    pub kp: f64,
    /// Integral gain (V/(A·s)).
    pub ki: f64,
    /// Symmetric limit on the integral term's contribution, in volts.
    pub v_limit: f64,
    /// Accumulated integral term (V).
    pub integral: f64,
}

impl PiCurrent {
    /// A regulator with gains from a target bandwidth, by pole-zero cancellation: `kp = L ω_bw`,
    /// `ki = R ω_bw`. This makes the closed-loop current response a first-order lag at `ω_bw` exactly,
    /// independent of the machine's own time constant, which is the point of the design.
    pub fn tuned(r_s: f64, l: f64, bandwidth: f64, v_limit: f64) -> PiCurrent {
        PiCurrent { kp: l * bandwidth, ki: r_s * bandwidth, v_limit, integral: 0.0 }
    }

    /// The explicit-Euler stability bound imposed by a regulator at `bandwidth` rad/s: `2/bandwidth`.
    ///
    /// This is the **asymptotic** limit, and it is necessary rather than sufficient. Measured by bisecting on
    /// the divergence threshold across four machines and three bandwidths, the critical `dt·ω_bw` ranged from
    /// **1.22 to 1.94** — always below 2, approaching it as the loop bandwidth outruns the plant pole `R/L`.
    /// The thresholds were identical for machines sharing `L/R`, which is the check that the closed loop
    /// depends only on the bandwidth and the plant pole and not on `R` and `L` separately.
    ///
    /// So a step *at* this bound is not safe; a fifth of it or less is. Note this bound is independent of the
    /// machine, which is the flip side of what makes [`PiCurrent::tuned`] work.
    pub fn max_stable_dt(bandwidth: f64) -> f64 {
        2.0 / bandwidth
    }

    /// One regulator step; returns the commanded voltage before any cross-coupling feedforward.
    pub fn step(&mut self, dt: f64, reference: f64, measured: f64) -> f64 {
        let e = reference - measured;
        self.integral = (self.integral + self.ki * e * dt).clamp(-self.v_limit, self.v_limit);
        self.kp * e + self.integral
    }

    /// Clear the integral, e.g. on re-enable after a fault.
    pub fn reset(&mut self) {
        self.integral = 0.0;
    }
}

/// Electrical angle from mechanical angle: `θ_e = p θ_m`, wrapped to `[0, 2π)`.
///
/// Kept as a named function because the `p` is the most common place this is dropped, and an unwrapped angle
/// loses precision as it grows.
pub fn electrical_angle(theta_mech: f64, pole_pairs: f64) -> f64 {
    let t = (theta_mech * pole_pairs) % (2.0 * PI);
    if t < 0.0 {
        t + 2.0 * PI
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_machine_surface() -> Pmsm {
        // Roughly a mid-size robot joint actuator: 7 pole pairs, 0.1 Ω, 0.5 mH, 0.02 Wb.
        Pmsm::surface(7.0, 0.1, 0.5e-3, 0.02)
    }

    fn test_machine_interior() -> Pmsm {
        Pmsm::interior(7.0, 0.1, 0.3e-3, 0.9e-3, 0.02)
    }

    #[test]
    fn clarke_and_park_round_trip_exactly() {
        for &(a, b, c) in &[(1.0, -0.5, -0.5), (3.0, -1.0, -2.0), (0.0, 1.0, -1.0), (-7.5, 2.5, 5.0)] {
            let (al, be) = clarke(a, b, c);
            let (a2, b2, c2) = inverse_clarke(al, be);
            // Only valid for zero-sequence-free inputs, which these are (they sum to zero).
            assert!((a - a2).abs() < 1e-14, "clarke round trip a: {a} -> {a2}");
            assert!((b - b2).abs() < 1e-14, "clarke round trip b: {b} -> {b2}");
            assert!((c - c2).abs() < 1e-14, "clarke round trip c: {c} -> {c2}");
        }
        for &th in &[0.0, 0.3, 1.7, PI, 4.9, -2.2] {
            for &(d, q) in &[(1.0, 0.0), (0.0, 1.0), (2.5, -3.5), (-1.0, -1.0)] {
                let (al, be) = inverse_park(d, q, th);
                let (d2, q2) = park(al, be, th);
                assert!((d - d2).abs() < 1e-14 && (q - q2).abs() < 1e-14, "park round trip at {th}");
            }
        }
    }

    #[test]
    fn the_zero_sequence_is_discarded_because_a_star_winding_cannot_carry_it() {
        // Adding the same offset to all three phases must not change alpha-beta at all: a star winding with
        // no neutral cannot carry that current, so a command containing it is unrealizable.
        let (a, b, c) = (2.0, -0.5, -1.5);
        let (al0, be0) = clarke(a, b, c);
        for off in [0.5, -3.0, 100.0] {
            let (al, be) = clarke(a + off, b + off, c + off);
            assert!((al - al0).abs() < 1e-13 && (be - be0).abs() < 1e-13, "zero sequence leaked at {off}");
        }
    }

    #[test]
    fn the_clarke_transform_is_amplitude_invariant_not_power_invariant() {
        // Which convention is in force determines whether a 3/2 belongs in the torque equation. Asserting it
        // directly means the factor downstream is a consequence rather than a guess.
        let peak = 12.0;
        for k in 0..64 {
            let th = 2.0 * PI * k as f64 / 64.0;
            let a = peak * th.cos();
            let b = peak * (th - 2.0 * PI / 3.0).cos();
            let c = peak * (th + 2.0 * PI / 3.0).cos();
            let (al, be) = clarke(a, b, c);
            let mag = (al * al + be * be).sqrt();
            assert!((mag - peak).abs() < 1e-12, "amplitude must be preserved: {mag} vs {peak}");
        }
        // And NOT power invariant: the three-phase sum of squares is 3/2 of the alpha-beta sum.
        let th: f64 = 0.7;
        let (a, b, c) = (
            peak * th.cos(),
            peak * (th - 2.0 * PI / 3.0).cos(),
            peak * (th + 2.0 * PI / 3.0).cos(),
        );
        let (al, be) = clarke(a, b, c);
        let three = a * a + b * b + c * c;
        let two = al * al + be * be;
        assert!((three - 1.5 * two).abs() < 1e-11, "expected a factor of exactly 3/2: {three} vs {two}");
    }

    #[test]
    fn park_turns_a_rotating_sinusoid_into_a_constant() {
        // The entire reason field-oriented control exists. If this did not hold, a PI regulator could not
        // drive the current error to zero, because its reference would be moving.
        let m = test_machine_surface();
        let (id_ref, iq_ref) = (-3.0, 11.0);
        let mut worst = 0.0f64;
        for k in 0..1000 {
            let theta_e = 17.3 * k as f64 / 1000.0; // several electrical revolutions
            let mut mm = m;
            mm.i_d = id_ref;
            mm.i_q = iq_ref;
            let (a, b, c) = mm.phase_currents(theta_e);
            // The phases are genuinely moving, or this test proves nothing.
            let (al, be) = clarke(a, b, c);
            let (d, q) = park(al, be, theta_e);
            worst = worst.max((d - id_ref).abs()).max((q - iq_ref).abs());
        }
        assert!(worst < 1e-12, "dq must be constant through rotation, worst deviation {worst:.3e}");

        // And confirm the phase currents really did vary, so the invariance is not vacuous.
        let mut mm = m;
        mm.i_d = id_ref;
        mm.i_q = iq_ref;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for k in 0..1000 {
            let a = mm.phase_currents(17.3 * k as f64 / 1000.0).0;
            lo = lo.min(a);
            hi = hi.max(a);
        }
        assert!(hi - lo > 10.0, "phase A must actually swing, range was {}", hi - lo);
    }

    #[test]
    fn copper_loss_in_dq_equals_the_three_phase_sum() {
        // A missing 3/2 understates heating by a third, and nothing else in a simulation would notice.
        let m = test_machine_surface();
        for &(i_d, i_q) in &[(0.0, 10.0), (-5.0, 20.0), (3.0, -7.0)] {
            let mut mm = m;
            mm.i_d = i_d;
            mm.i_q = i_q;
            let dq = mm.copper_loss_at(i_d, i_q);
            // Average the three-phase loss over an electrical cycle; it is constant for balanced currents,
            // so the average and every instant agree, which is itself worth checking.
            let mut worst_inst = 0.0f64;
            for k in 0..256 {
                let th = 2.0 * PI * k as f64 / 256.0;
                let (a, b, c) = mm.phase_currents(th);
                let inst = m.r_s * (a * a + b * b + c * c);
                worst_inst = worst_inst.max((inst - dq).abs());
            }
            assert!(worst_inst < 1e-10, "dq loss {dq} must equal the phase sum at every angle, off by {worst_inst:.3e}");
        }
    }

    #[test]
    fn the_thermal_equivalent_current_matches_the_single_r_model() {
        // MotorThermal's copper_loss is R i², so handing it the dq magnitude would lose the 3/2.
        let mut m = test_machine_surface();
        m.i_d = -4.0;
        m.i_q = 15.0;
        let ieq = m.thermal_equivalent_current();
        let via_thermal = m.r_s * ieq * ieq;
        assert!(
            (via_thermal - m.copper_loss()).abs() < 1e-12,
            "equivalent current must reproduce the loss: {via_thermal} vs {}",
            m.copper_loss()
        );
        // And it is larger than the dq magnitude by exactly sqrt(3/2).
        let mag = (m.i_d * m.i_d + m.i_q * m.i_q).sqrt();
        assert!((ieq / mag - (1.5f64).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn svpwm_reaches_v_dc_over_sqrt3_and_spwm_only_half() {
        // The 15.47% claim, measured rather than asserted. Sweep the commanded peak and find where each
        // modulator's duties first leave [0, 1].
        let v_dc = 48.0;
        let first_saturation = |duties: fn(f64, f64, f64) -> [f64; 3]| -> f64 {
            let mut lo = 0.0f64;
            let mut hi = v_dc; // certainly saturated
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                // Worst case over a full electrical cycle: saturation is angle-dependent.
                let mut ok = true;
                for k in 0..720 {
                    let th = 2.0 * PI * k as f64 / 720.0;
                    let (va, vb) = (mid * th.cos(), mid * th.sin());
                    if duties(va, vb, v_dc).iter().any(|d| *d < -1e-12 || *d > 1.0 + 1e-12) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };

        let sv = first_saturation(svpwm_duties);
        let sp = first_saturation(spwm_duties);
        assert!(
            (sv - svpwm_voltage_limit(v_dc)).abs() < 1e-6,
            "SVPWM should saturate at V_dc/sqrt(3) = {}, measured {sv}",
            svpwm_voltage_limit(v_dc)
        );
        assert!(
            (sp - spwm_voltage_limit(v_dc)).abs() < 1e-6,
            "SPWM should saturate at V_dc/2 = {}, measured {sp}",
            spwm_voltage_limit(v_dc)
        );
        // The headroom, as a ratio: 2/sqrt(3) = 1.1547, i.e. 15.47% more voltage from arithmetic alone.
        let ratio = sv / sp;
        assert!((ratio - 2.0 / SQRT3).abs() < 1e-6, "the ratio should be 2/sqrt(3), measured {ratio:.6}");
        assert!(ratio > 1.15 && ratio < 1.16, "which is 15.47%, measured {:.2}%", 100.0 * (ratio - 1.0));
    }

    #[test]
    fn the_common_mode_offset_does_not_change_any_line_to_line_voltage() {
        // This is why SVPWM's headroom is free rather than a distortion: the load only sees differences.
        let v_dc = 48.0;
        for k in 0..64 {
            let th = 2.0 * PI * k as f64 / 64.0;
            let (va, vb) = (20.0 * th.cos(), 20.0 * th.sin());
            let sv = svpwm_duties(va, vb, v_dc);
            let sp = spwm_duties(va, vb, v_dc);
            // Line-to-line, in volts, from duties.
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let ll_sv = (sv[i] - sv[j]) * v_dc;
                let ll_sp = (sp[i] - sp[j]) * v_dc;
                assert!(
                    (ll_sv - ll_sp).abs() < 1e-11,
                    "line-to-line {i}{j} must match: {ll_sv} vs {ll_sp}"
                );
            }
        }
    }

    #[test]
    fn mtpa_is_exactly_id_zero_on_a_surface_machine_and_negative_id_on_a_salient_one() {
        // Two different physical statements, and getting the second's sign wrong would DEMAGNETISE the rotor
        // rather than help it.
        let s = test_machine_surface();
        let (i_d, i_q) = s.mtpa(4.0).expect("reachable");
        assert_eq!(i_d, 0.0, "a surface machine has no reluctance torque to earn");
        assert!((s.torque_at(i_d, i_q) - 4.0).abs() < 1e-12, "and the torque must be met exactly");

        let sal = test_machine_interior();
        let (i_d2, i_q2) = sal.mtpa(4.0).expect("reachable");
        assert!(i_d2 < 0.0, "a salient machine earns reluctance torque at NEGATIVE i_d, got {i_d2}");
        assert!((sal.torque_at(i_d2, i_q2) - 4.0).abs() < 1e-9, "the torque must still be met");

        // And it genuinely costs less current than the i_d = 0 solution, which is the whole claim.
        let iq_only = 4.0 / (1.5 * sal.pole_pairs * sal.flux_linkage);
        let mag_mtpa = (i_d2 * i_d2 + i_q2 * i_q2).sqrt();
        assert!(
            mag_mtpa < iq_only,
            "MTPA {mag_mtpa:.4} A should beat q-axis-only {iq_only:.4} A"
        );
    }

    #[test]
    fn mtpa_beats_every_alternative_on_a_brute_force_sweep() {
        // The optimiser is verified against exhaustive search rather than against a closed form I might have
        // mis-remembered. If a grid point beats it, the golden-section bracket is wrong.
        let sal = test_machine_interior();
        for &t in &[0.5, 2.0, 6.0, 12.0] {
            let (i_d, i_q) = sal.mtpa(t).expect("reachable");
            let best = i_d * i_d + i_q * i_q;
            let scale = (t / (1.5 * sal.pole_pairs * sal.flux_linkage)).abs();
            let mut violations = 0;
            for k in 0..=4000 {
                let cand_d = -8.0 * scale * k as f64 / 4000.0;
                let coeff = 1.5 * sal.pole_pairs * (sal.flux_linkage + (sal.l_d - sal.l_q) * cand_d);
                if coeff.abs() < 1e-9 || coeff * sal.flux_linkage <= 0.0 {
                    continue;
                }
                let cand_q = t / coeff;
                // Only count a real improvement, not floating-point noise at the optimum.
                if cand_d * cand_d + cand_q * cand_q < best * (1.0 - 1e-9) {
                    violations += 1;
                }
            }
            assert_eq!(violations, 0, "torque {t}: {violations} grid points beat the MTPA solution");
        }
    }

    #[test]
    fn base_speed_is_where_the_voltage_limit_is_exactly_reached() {
        let m = test_machine_surface();
        let v_max = svpwm_voltage_limit(48.0);
        let torque = 3.0;
        let w = m.base_speed(torque, 40.0, v_max).expect("reachable");
        assert!(w > 0.0, "base speed should be positive, got {w}");

        let (i_d, i_q) = m.mtpa(torque).expect("reachable");
        let (v_d, v_q) = m.steady_voltage(i_d, i_q, w * m.pole_pairs);
        let mag = (v_d * v_d + v_q * v_q).sqrt();
        assert!((mag - v_max).abs() < 1e-6 * v_max, "at base speed |v| must equal v_max: {mag} vs {v_max}");

        // Just above it, the limit is exceeded: the definition, checked from the other side.
        let (v_d2, v_q2) = m.steady_voltage(i_d, i_q, w * 1.01 * m.pole_pairs);
        assert!((v_d2 * v_d2 + v_q2 * v_q2).sqrt() > v_max, "just above base speed must exceed v_max");

        // A higher bus raises base speed, and by the modulator's ratio when back-EMF dominates.
        let w_sp = m.base_speed(torque, 40.0, spwm_voltage_limit(48.0)).expect("reachable");
        assert!(w > w_sp, "SVPWM's extra voltage must raise base speed: {w} vs {w_sp}");

        // A torque beyond the current limit is unreachable at any speed, and says so.
        assert!(m.base_speed(1e6, 40.0, v_max).is_none(), "an impossible torque must return None");
    }

    #[test]
    fn field_weakening_buys_speed_at_the_cost_of_torqueless_current() {
        let m = test_machine_surface();
        let v_max = svpwm_voltage_limit(48.0);
        let (i_d0, i_q) = m.mtpa(3.0).expect("reachable");
        assert_eq!(i_d0, 0.0);
        let w_base = m.base_speed(3.0, 40.0, v_max).expect("reachable");

        // Below base speed, no weakening is needed and the answer is exactly zero.
        assert_eq!(m.field_weakening_id(i_q, 0.5 * w_base, v_max, 40.0), Some(0.0));

        // Above it, a negative i_d is required, and it must actually bring the voltage inside the limit.
        let w = 1.5 * w_base;
        let i_d = m.field_weakening_id(i_q, w, v_max, 40.0).expect("reachable with weakening");
        assert!(i_d < 0.0, "weakening must be negative i_d, got {i_d}");
        let (v_d, v_q) = m.steady_voltage(i_d, i_q, w * m.pole_pairs);
        assert!(
            (v_d * v_d + v_q * v_q).sqrt() <= v_max * (1.0 + 1e-9),
            "weakening must bring |v| inside the limit"
        );

        // It costs copper loss for no extra torque: the torque is unchanged on a surface machine.
        let mut mm = m;
        mm.i_d = i_d;
        mm.i_q = i_q;
        let mut m0 = m;
        m0.i_d = 0.0;
        m0.i_q = i_q;
        assert!((mm.torque() - m0.torque()).abs() < 1e-12, "i_d makes no torque on a surface machine");
        assert!(mm.copper_loss() > m0.copper_loss(), "but it does cost copper loss");

        // Far enough out, no i_d within the current limit suffices, and that is reported rather than faked.
        assert!(
            m.field_weakening_id(i_q, 500.0 * w_base, v_max, 40.0).is_none(),
            "an unreachable speed must return None"
        );
    }

    #[test]
    fn electrical_power_balances_mechanical_output_plus_copper_loss() {
        // The identity that ties this module to the battery and the thermal model. At steady state with no
        // inductive term storing energy, input must equal output plus loss, exactly.
        let m = test_machine_surface();
        for &(i_d, i_q, w_mech) in &[(0.0, 10.0, 20.0), (-6.0, 18.0, 55.0), (2.0, -9.0, -12.0)] {
            let mut mm = m;
            mm.i_d = i_d;
            mm.i_q = i_q;
            let omega_e = w_mech * m.pole_pairs;
            let (v_d, v_q) = mm.steady_voltage(i_d, i_q, omega_e);
            let p_in = mm.electrical_power(v_d, v_q);
            let p_mech = mm.mechanical_power(w_mech);
            let p_cu = mm.copper_loss();
            assert!(
                (p_in - p_mech - p_cu).abs() < 1e-9 * p_in.abs().max(1.0),
                "power balance: in {p_in:.6} vs mech {p_mech:.6} + copper {p_cu:.6}"
            );
        }
    }

    #[test]
    fn the_current_loop_reaches_its_reference_and_the_step_is_stable_below_the_bound() {
        let m = test_machine_surface();
        let bw = 2.0 * PI * 500.0; // 500 Hz current loop
        let v_max = svpwm_voltage_limit(48.0);
        // The REGULATOR's bound, not the plant's. A tenth of the plant's 2L/R gave 2 samples per loop period
        // and the run produced NaN; the plant bound says nothing about the closed loop.
        let dt = 0.05 * PiCurrent::max_stable_dt(bw);
        assert!(dt < PiCurrent::max_stable_dt(bw), "inside the regulator's bound");
        assert!(dt < m.max_stable_dt(), "and inside the plant's, which here is the looser of the two");
        assert!(
            m.max_stable_dt() > 10.0 * PiCurrent::max_stable_dt(bw),
            "the two bounds differ by more than an order of magnitude here, which is the point"
        );

        let mut mm = m;
        let mut reg_d = PiCurrent::tuned(m.r_s, m.l_d, bw, v_max);
        let mut reg_q = PiCurrent::tuned(m.r_s, m.l_q, bw, v_max);
        let (id_ref, iq_ref) = (0.0, 12.0);
        let omega_e = 0.0; // stalled, so the loop is tested without cross-coupling
        for _ in 0..20_000 {
            let v_d = reg_d.step(dt, id_ref, mm.i_d);
            let v_q = reg_q.step(dt, iq_ref, mm.i_q);
            mm.step(dt, v_d, v_q, omega_e);
        }
        assert!((mm.i_q - iq_ref).abs() < 1e-3, "the q current must reach its reference, got {}", mm.i_q);
        assert!(mm.i_d.abs() < 1e-3, "and d must stay at zero, got {}", mm.i_d);
        assert!((mm.torque() - m.torque_at(0.0, iq_ref)).abs() < 1e-2);
    }

    #[test]
    fn the_closed_loop_stability_bound_is_the_one_that_binds() {
        // The finding this module cost me: `Pmsm::max_stable_dt` is an open-loop bound, and trusting it with a
        // regulator closed produced NaN. Measured critical dt*bandwidth ranged 1.22-1.94 across machines and
        // bandwidths, always below 2. This pins both halves so the documented bound cannot drift.
        let diverges = |r: f64, l: f64, bw: f64, dt: f64| -> bool {
            let mut m = Pmsm::surface(7.0, r, l, 0.02);
            let mut reg = PiCurrent::tuned(r, l, bw, 1e9);
            for _ in 0..200_000 {
                let v_q = reg.step(dt, 5.0, m.i_q);
                m.step(dt, 0.0, v_q, 0.0);
                if !m.i_q.is_finite() || m.i_q.abs() > 1e6 {
                    return true;
                }
            }
            !m.i_q.is_finite() || (m.i_q - 5.0).abs() > 2.5
        };

        for &(r, l) in &[(0.1f64, 0.5e-3f64), (0.5, 2e-3)] {
            for &f in &[100.0f64, 500.0, 2000.0] {
                let bw = 2.0 * PI * f;
                // Bisect geometrically on the divergence threshold.
                let (mut lo, mut hi) = (1e-9f64, 1.0f64);
                for _ in 0..80 {
                    let mid = (lo * hi).sqrt();
                    if diverges(r, l, bw, mid) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let crit = 0.5 * (lo + hi) * bw;
                assert!(
                    (1.15..2.0).contains(&crit),
                    "R={r} L={l} f={f}: critical dt*bw {crit:.4} outside the measured band [1.15, 2.0)"
                );
                // The documented bound must never be optimistic: 2/bw is at or above the real threshold.
                assert!(crit < 2.0, "the documented 2/bw bound must not be exceeded, got {crit:.4}");
            }
        }
        // And the two bounds are genuinely different quantities, not the same one twice.
        let m = test_machine_surface();
        assert!(m.max_stable_dt() > 15.0 * PiCurrent::max_stable_dt(2.0 * PI * 500.0));
    }

    #[test]
    fn the_integral_clamp_prevents_windup() {
        // Drive a reference the voltage limit cannot deliver, then release it. Without the clamp the
        // integrator would have accumulated an enormous command and the machine would slam.
        let mut reg = PiCurrent::tuned(0.1, 0.5e-3, 2.0 * PI * 500.0, 10.0);
        for _ in 0..100_000 {
            reg.step(1e-5, 1e6, 0.0); // an impossible reference
        }
        assert!(
            reg.integral.abs() <= 10.0 + 1e-12,
            "the integral must stay clamped, got {}",
            reg.integral
        );
        // And the clamp is actually engaged, or this test proves nothing about winding up.
        assert!((reg.integral - 10.0).abs() < 1e-9, "the clamp should be saturated, got {}", reg.integral);
        reg.reset();
        assert_eq!(reg.integral, 0.0);
    }

    #[test]
    fn the_tuned_gains_give_the_first_order_response_they_claim() {
        // `tuned` claims pole-zero cancellation makes the closed loop a first-order lag at the chosen
        // bandwidth, independent of the machine. That is checkable: the time to 63.2% of a step should be
        // 1/bandwidth, for machines with very different time constants.
        for &(r, l) in &[(0.1f64, 0.5e-3f64), (1.0, 5e-3), (0.02, 0.1e-3)] {
            let bw = 2.0 * PI * 200.0;
            let mut m = Pmsm::surface(7.0, r, l, 0.02);
            let mut reg = PiCurrent::tuned(r, l, bw, 1e9); // no voltage limit, so the linear claim is tested
            let dt = 1e-7;
            let target = 5.0;
            let mut t_63 = None;
            for k in 0..2_000_000 {
                let v_q = reg.step(dt, target, m.i_q);
                m.step(dt, 0.0, v_q, 0.0);
                if t_63.is_none() && m.i_q >= 0.632_120_558_828_557_7 * target {
                    t_63 = Some(k as f64 * dt);
                    break;
                }
            }
            let t = t_63.expect("the loop must reach 63.2%");
            let expected = 1.0 / bw;
            assert!(
                (t - expected).abs() < 0.03 * expected,
                "R={r} L={l}: time to 63.2% was {t:.6e}, expected 1/bw = {expected:.6e}"
            );
        }
    }

    #[test]
    fn the_electrical_angle_carries_the_pole_pair_count_and_stays_wrapped() {
        // Dropping `p` is the single most common commutation error, and its symptom is a motor that runs
        // rough at a fraction of rated torque rather than an error message.
        assert!((electrical_angle(0.0, 7.0) - 0.0).abs() < 1e-15);
        // One mechanical revolution is exactly `p` electrical revolutions, so it wraps back to zero.
        let a = electrical_angle(2.0 * PI, 7.0);
        assert!(a < 1e-12 || (a - 2.0 * PI).abs() < 1e-12, "one mech rev is p elec revs, got {a}");
        // A seventh of a revolution is one full electrical revolution.
        let b = electrical_angle(2.0 * PI / 7.0, 7.0);
        assert!(b < 1e-12 || (b - 2.0 * PI).abs() < 1e-12, "got {b}");
        // Always in range, including for negative and very large angles.
        for th in [-100.0, -1.0, 0.5, 1e6] {
            let e = electrical_angle(th, 7.0);
            assert!((0.0..2.0 * PI).contains(&e), "angle {th} gave out-of-range {e}");
        }
    }

    #[test]
    fn a_salient_machine_produces_more_torque_than_its_magnet_alone_explains() {
        // The operational statement of saliency. If the reluctance term were dropped, this machine would be
        // indistinguishable from a surface one, and the MTPA advantage above would be inexplicable.
        let sal = test_machine_interior();
        let i_q = 15.0;
        let magnet_only = 1.5 * sal.pole_pairs * sal.flux_linkage * i_q;
        let with_reluctance = sal.torque_at(-10.0, i_q);
        assert!(
            with_reluctance > magnet_only,
            "reluctance torque should add: {with_reluctance:.4} vs magnet-only {magnet_only:.4}"
        );
        // And positive i_d SUBTRACTS, which is the sign trap.
        assert!(sal.torque_at(10.0, i_q) < magnet_only, "positive i_d must reduce torque on this machine");
    }
}
