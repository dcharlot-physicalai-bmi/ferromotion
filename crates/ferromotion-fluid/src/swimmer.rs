//! **A self-propelled undulatory swimmer** (Honest Fluids — the robot loop). The immersed-boundary
//! machinery already couples a rigid body to the fluid; this closes the loop into *locomotion*: a
//! flexible filament whose body-frame shape follows a traveling wave, free to translate under the
//! net hydrodynamic reaction. Thrust is emergent — no prescribed forward velocity — so it is a
//! genuine fluid-structure test, not a kinematic animation.
//!
//! The step generalizes [`crate::MacFluid::step_with_disk`] from one disk to an arbitrary marker
//! set with per-marker target velocities (the undulating body), enforced by the same iterated
//! direct forcing; the net force integrates the body's free streamwise DOF (Newton, neutrally
//! buoyant). Verified by the invariants a swimmer must obey: a still body does not drift, a
//! traveling wave produces net motion, and reversing the wave reverses it.

use crate::MacFluid;
use std::f64::consts::PI;

impl MacFluid {
    /// Advance one step with an immersed **marker set** (world positions `markers`, target surface
    /// velocities `vels`), enforcing them by iterated direct forcing exactly as
    /// [`Self::step_with_disk`] does for a single body. Returns the net hydrodynamic force
    /// `(Fx, Fy)` the fluid exerts on the body (the reaction to the injected momentum).
    pub fn step_markers(&mut self, markers: &[(f64, f64)], vels: &[(f64, f64)]) -> (f64, f64) {
        const N_FORCE_ITERS: usize = 6;
        let (h, dt) = (self.h, self.dt);
        let (mut us, mut vs) = self.predict();
        let su: Vec<_> = markers.iter().map(|&(mx, my)| self.stencil_u(mx, my)).collect();
        let sv: Vec<_> = markers.iter().map(|&(mx, my)| self.stencil_v(mx, my)).collect();

        let (mut sum_fx, mut sum_fy) = (0.0f64, 0.0f64);
        for _ in 0..N_FORCE_ITERS {
            let deficits: Vec<(f64, f64)> = (0..markers.len())
                .map(|m| {
                    let ui: f64 = su[m].iter().map(|&(k, w)| us[k] * w).sum();
                    let vi: f64 = sv[m].iter().map(|&(k, w)| vs[k] * w).sum();
                    (vels[m].0 - ui, vels[m].1 - vi)
                })
                .collect();
            for m in 0..markers.len() {
                let (fx, fy) = deficits[m];
                for &(k, w) in &su[m] {
                    us[k] += fx * w;
                    sum_fx += fx * w;
                }
                for &(k, w) in &sv[m] {
                    vs[k] += fy * w;
                    sum_fy += fy * w;
                }
            }
        }
        self.project_and_correct(us, vs);
        (-sum_fx * h * h / dt, -sum_fy * h * h / dt)
    }
}

/// An undulating filament swimmer. The body spans arclength `[0, len]`; its lateral shape follows a
/// traveling wave `y_d(s, t) = amp · sin(k_b·s − ω·t)`. It is free to translate in `x` (streamwise)
/// under the net hydrodynamic force; the lateral center `y0` is held (the wave is symmetric, so net
/// lateral force averages to zero over a cycle).
#[derive(Clone, Debug)]
pub struct Swimmer {
    pub x: f64,   // streamwise position of the body center
    pub y0: f64,  // lateral center (held)
    pub vx: f64,  // streamwise velocity (free DOF)
    pub len: f64,
    pub seg: usize,
    pub mass: f64,
    pub amp: f64,
    pub nwave: f64, // wavelengths along the body
    pub omega: f64, // angular frequency
    pub t: f64,
}

/// Marker world positions (or velocities) along the body — one `(x, y)` per segment.
pub type MarkerField = Vec<(f64, f64)>;

impl Swimmer {
    /// A filament of `seg` markers spanning `len`, centered at `(x, y0)`, at rest.
    #[allow(clippy::too_many_arguments)] // a swimmer genuinely carries this many physical parameters
    pub fn new(x: f64, y0: f64, len: f64, seg: usize, mass: f64, amp: f64, nwave: f64, omega: f64) -> Self {
        Self { x, y0, vx: 0.0, len, seg, mass, amp, nwave, omega, t: 0.0 }
    }

    #[inline]
    fn kb(&self) -> f64 {
        2.0 * PI * self.nwave / self.len
    }

    /// Current marker world positions and target velocities (body translation + undulation).
    pub fn markers_and_vels(&self) -> (MarkerField, MarkerField) {
        let kb = self.kb();
        let (mut pos, mut vel) = (Vec::with_capacity(self.seg), Vec::with_capacity(self.seg));
        for m in 0..self.seg {
            let s = self.len * m as f64 / (self.seg - 1) as f64; // 0..len along the body
            let phase = kb * s - self.omega * self.t;
            let yd = self.amp * phase.sin();
            let yd_dot = -self.amp * self.omega * phase.cos(); // ∂y_d/∂t
            pos.push((self.x - self.len / 2.0 + s, self.y0 + yd));
            vel.push((self.vx, yd_dot));
        }
        (pos, vel)
    }

    /// Advance the coupled system one fluid step: enforce the body on the fluid, integrate the free
    /// streamwise DOF under the net force (explicit/weak coupling), advance the gait clock.
    pub fn advance(&mut self, fluid: &mut MacFluid) {
        let (markers, vels) = self.markers_and_vels();
        let (fx, _fy) = fluid.step_markers(&markers, &vels);
        self.vx += fluid_dt(fluid) * fx / self.mass;
        self.x += fluid_dt(fluid) * self.vx;
        self.t += fluid_dt(fluid);
    }
}

/// Read the fluid timestep (private field; accessible from this descendant module).
fn fluid_dt(f: &MacFluid) -> f64 {
    f.dt
}

#[cfg(test)]
mod verification {
    use super::*;

    fn tank(n: usize, nu: f64, dt: f64) -> MacFluid {
        MacFluid::new(n, n, nu, dt, 0.0).with_free_slip() // open swim tank: free-slip walls, no lid
    }

    /// A still body (zero gait amplitude, starting at rest) injects no momentum and does not drift.
    #[test]
    fn still_body_does_not_drift() {
        let n = 48;
        let dt = 2e-4;
        let mut f = tank(n, 0.01, dt);
        let mut s = Swimmer::new(0.5, 0.5, 0.3, 24, 0.02, 0.0, 1.0, 2.0 * PI * 4.0);
        for _ in 0..200 {
            s.advance(&mut f);
        }
        eprintln!("still swimmer drift: x {:.3e}, vx {:.3e}", s.x - 0.5, s.vx);
        assert!((s.x - 0.5).abs() < 1e-9, "still body drifted: {}", s.x - 0.5);
    }

    /// A traveling-wave gait produces net streamwise motion, and reversing the wave (ω → −ω)
    /// reverses that motion — the defining signature of undulatory swimming. Low-Re regime
    /// (surface speed `amp·ω ≈ 0.09`, mass above the added mass) so the explicit coupling is stable.
    #[test]
    fn traveling_wave_propels_and_reverses() {
        let n = 64;
        let dt = 4e-4;
        let steps = 3000;
        let swim = |omega: f64| -> f64 {
            let mut f = tank(n, 0.006, dt);
            let mut s = Swimmer::new(0.5, 0.5, 0.35, 28, 0.05, 0.015, 1.0, omega);
            for _ in 0..steps {
                s.advance(&mut f);
            }
            assert!(s.x.is_finite() && (s.x - 0.5).abs() < 0.25, "unstable/out-of-tank: x = {}", s.x);
            s.x - 0.5
        };
        let w = 2.0 * PI * 1.0;
        let fwd = swim(w);
        let rev = swim(-w);
        eprintln!("swim displacement: +ω {fwd:.4e}   −ω {rev:.4e}");
        assert!(fwd.abs() > 1e-3, "no net thrust: {fwd}");
        assert!(fwd * rev < 0.0, "reversing the wave did not reverse motion: {fwd} vs {rev}");
        assert!((fwd + rev).abs() < 0.3 * fwd.abs(), "thrust not antisymmetric in ω: {fwd} vs {rev}");
    }

    /// Net streamwise displacement after a fixed swim, as a function of tail-beat frequency ω.
    fn cruise(n: usize, dt: f64, steps: usize, omega: f64) -> f64 {
        let mut f = tank(n, 0.006, dt);
        let mut s = Swimmer::new(0.5, 0.5, 0.35, 28, 0.05, 0.015, 1.0, omega);
        for _ in 0..steps {
            s.advance(&mut f);
        }
        s.x - 0.5
    }

    /// **The payoff: tune the gait to a commanded cruise by gradient.** Displacement is smooth and
    /// monotone in tail-beat frequency, so a secant root-find on `cruise(ω) − target` (the
    /// gradient-driven controller) converges in a handful of coupled fluid rollouts — the loop
    /// closes: fluid → thrust → body motion → objective → gait update.
    #[test]
    fn gait_tunes_to_a_commanded_cruise() {
        let (n, dt, steps) = (48, 5e-4, 1500);
        let target_omega = 2.0 * PI * 1.1;
        let target = cruise(n, dt, steps, target_omega); // the commanded displacement

        // Secant iteration on g(ω) = cruise(ω) − target, from two wrong guesses.
        let (mut a, mut b) = (2.0 * PI * 0.6, 2.0 * PI * 1.6);
        let (mut ga, mut gb) = (cruise(n, dt, steps, a) - target, cruise(n, dt, steps, b) - target);
        let mut omega = b;
        for it in 0..8 {
            omega = b - gb * (b - a) / (gb - ga);
            let g = cruise(n, dt, steps, omega) - target;
            eprintln!("it {it}: ω/2π {:.4} residual {g:.3e}", omega / (2.0 * PI));
            if g.abs() < 1e-4 * target.abs().max(1e-4) {
                break;
            }
            a = b;
            ga = gb;
            b = omega;
            gb = g;
        }
        let rel = (omega - target_omega).abs() / target_omega;
        eprintln!("recovered ω/2π {:.4} (target {:.4}) rel {rel:.2e}", omega / (2.0 * PI), target_omega / (2.0 * PI));
        assert!(rel < 0.02, "gait not tuned to the commanded cruise: {rel}");
    }
}
