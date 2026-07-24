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
use faer::linalg::solvers::Solve;
use faer::Mat;
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

    /// **Differentiable coupled step for a marker set** — the forward-sensitivity generalization of
    /// [`Self::step_markers`], carrying a tangent fluid field `(ut, vt) = ∂(u,v)/∂θ` alongside the
    /// primal. Each marker `m` supplies `dsurf[m] = ∂vels[m]/∂θ` and `dmarker[m] = ∂pos[m]/∂θ`, so a
    /// body that both undulates (velocity tangent) *and* translates with `θ` (position tangent —
    /// its Peskin weights' `∂w/∂X` enter interpolation and spreading) differentiates exactly.
    /// Returns `(Fx, ∂Fx/∂θ)`; `ut`/`vt` persist across steps to carry the fluid's memory (seed 0).
    #[allow(clippy::too_many_arguments)]
    pub fn step_markers_sensitivity(
        &mut self,
        markers: &[(f64, f64)],
        vels: &[(f64, f64)],
        dmarker: &[(f64, f64)],
        dsurf: &[(f64, f64)],
        ut: &mut [f64],
        vt: &mut [f64],
    ) -> (f64, f64) {
        const N_FORCE_ITERS: usize = 6;
        let (nx, ny, h, nu, dt, lid) = (self.nx, self.ny, self.h, self.nu, self.dt, self.lid_u);
        let (inv2h, invh2) = (1.0 / (2.0 * h), 1.0 / (h * h));

        // --- predictor (primal + tangent), product rule on advection ---
        let (mut us, mut vs) = (self.u.clone(), self.v.clone());
        let (mut uts, mut vts) = (ut.to_vec(), vt.to_vec());
        let ug = |i: i32, j: i32| crate::ughost(&self.u, ny, lid, i, j);
        let vg = |i: i32, j: i32| crate::vghost(&self.v, nx, ny, i, j);
        let tg = |i: i32, j: i32| crate::ughost(ut, ny, 0.0, i, j);
        let wg = |i: i32, j: i32| crate::vghost(vt, nx, ny, i, j);
        for i in 1..nx {
            for j in 0..ny {
                let (ii, jj) = (i as i32, j as i32);
                let uc = ug(ii, jj);
                let dudx = (ug(ii + 1, jj) - ug(ii - 1, jj)) * inv2h;
                let dudy = (ug(ii, jj + 1) - ug(ii, jj - 1)) * inv2h;
                let vbar = 0.25 * (vg(ii - 1, jj) + vg(ii, jj) + vg(ii - 1, jj + 1) + vg(ii, jj + 1));
                let lap = (ug(ii + 1, jj) + ug(ii - 1, jj) + ug(ii, jj + 1) + ug(ii, jj - 1) - 4.0 * uc) * invh2;
                us[i * ny + j] = uc + dt * (-(uc * dudx + vbar * dudy) + nu * lap);
                let uct = tg(ii, jj);
                let dudxt = (tg(ii + 1, jj) - tg(ii - 1, jj)) * inv2h;
                let dudyt = (tg(ii, jj + 1) - tg(ii, jj - 1)) * inv2h;
                let vbart = 0.25 * (wg(ii - 1, jj) + wg(ii, jj) + wg(ii - 1, jj + 1) + wg(ii, jj + 1));
                let lapt = (tg(ii + 1, jj) + tg(ii - 1, jj) + tg(ii, jj + 1) + tg(ii, jj - 1) - 4.0 * uct) * invh2;
                uts[i * ny + j] = uct + dt * (-(uct * dudx + uc * dudxt + vbart * dudy + vbar * dudyt) + nu * lapt);
            }
        }
        for i in 0..nx {
            for j in 1..ny {
                let (ii, jj) = (i as i32, j as i32);
                let vc = vg(ii, jj);
                let dvdx = (vg(ii + 1, jj) - vg(ii - 1, jj)) * inv2h;
                let dvdy = (vg(ii, jj + 1) - vg(ii, jj - 1)) * inv2h;
                let ubar = 0.25 * (ug(ii, jj - 1) + ug(ii + 1, jj - 1) + ug(ii, jj) + ug(ii + 1, jj));
                let lap = (vg(ii + 1, jj) + vg(ii - 1, jj) + vg(ii, jj + 1) + vg(ii, jj - 1) - 4.0 * vc) * invh2;
                vs[i * (ny + 1) + j] = vc + dt * (-(ubar * dvdx + vc * dvdy) + nu * lap);
                let vct = wg(ii, jj);
                let dvdxt = (wg(ii + 1, jj) - wg(ii - 1, jj)) * inv2h;
                let dvdyt = (wg(ii, jj + 1) - wg(ii, jj - 1)) * inv2h;
                let ubart = 0.25 * (tg(ii, jj - 1) + tg(ii + 1, jj - 1) + tg(ii, jj) + tg(ii + 1, jj));
                let lapt = (wg(ii + 1, jj) + wg(ii - 1, jj) + wg(ii, jj + 1) + wg(ii, jj - 1) - 4.0 * vct) * invh2;
                vts[i * (ny + 1) + j] = vct + dt * (-(ubart * dvdx + ubar * dvdxt + vct * dvdy + vc * dvdyt) + nu * lapt);
            }
        }

        // --- immersed-boundary forcing (primal + tangent) with per-marker dsurf/dmarker ---
        let su: Vec<_> = markers.iter().map(|&(mx, my)| self.stencil_u_d(mx, my)).collect();
        let sv: Vec<_> = markers.iter().map(|&(mx, my)| self.stencil_v_d(mx, my)).collect();
        let (mut sum_fx, mut sum_fx_dot) = (0.0f64, 0.0f64);
        for _ in 0..N_FORCE_ITERS {
            let defs: Vec<(f64, f64, f64, f64)> = (0..markers.len())
                .map(|m| {
                    let (dmx, dmy) = dmarker[m];
                    let ui: f64 = su[m].iter().map(|&(k, w, ..)| us[k] * w).sum();
                    let vi: f64 = sv[m].iter().map(|&(k, w, ..)| vs[k] * w).sum();
                    let uit: f64 = su[m].iter().map(|&(k, w, dwx, dwy)| (dwx * dmx + dwy * dmy) * us[k] + w * uts[k]).sum();
                    let vit: f64 = sv[m].iter().map(|&(k, w, dwx, dwy)| (dwx * dmx + dwy * dmy) * vs[k] + w * vts[k]).sum();
                    (vels[m].0 - ui, vels[m].1 - vi, dsurf[m].0 - uit, dsurf[m].1 - vit)
                })
                .collect();
            for m in 0..markers.len() {
                let (fx, fy, fxt, fyt) = defs[m];
                let (dmx, dmy) = dmarker[m];
                for &(k, w, dwx, dwy) in &su[m] {
                    let dw = dwx * dmx + dwy * dmy;
                    us[k] += fx * w;
                    uts[k] += fxt * w + fx * dw;
                    sum_fx += fx * w;
                    sum_fx_dot += fxt * w + fx * dw;
                }
                for &(k, w, dwx, dwy) in &sv[m] {
                    let dw = dwx * dmx + dwy * dmy;
                    vs[k] += fy * w;
                    vts[k] += fyt * w + fy * dw;
                }
            }
        }

        // --- projection (primal + tangent share the prefactored Laplacian) ---
        let n = nx * ny - 1;
        let (mut rhs, mut rhs_t) = (Mat::<f64>::zeros(n, 1), Mat::<f64>::zeros(n, 1));
        for i in 0..nx {
            for j in 0..ny {
                let k = i * ny + j;
                if k == 0 {
                    continue;
                }
                let div = (us[(i + 1) * ny + j] - us[i * ny + j]) / h + (vs[i * (ny + 1) + j + 1] - vs[i * (ny + 1) + j]) / h;
                let divt = (uts[(i + 1) * ny + j] - uts[i * ny + j]) / h + (vts[i * (ny + 1) + j + 1] - vts[i * (ny + 1) + j]) / h;
                rhs[(k - 1, 0)] = -h * h * div / dt;
                rhs_t[(k - 1, 0)] = -h * h * divt / dt;
            }
        }
        self.poisson.solve_in_place(&mut rhs);
        self.poisson.solve_in_place(&mut rhs_t);
        let (mut p, mut pt) = (vec![0.0; nx * ny], vec![0.0; nx * ny]);
        for k in 1..nx * ny {
            p[k] = rhs[(k - 1, 0)];
            pt[k] = rhs_t[(k - 1, 0)];
        }

        // --- corrector: write primal and tangent fields back ---
        for i in 1..nx {
            for j in 0..ny {
                self.u[i * ny + j] = us[i * ny + j] - dt * (p[i * ny + j] - p[(i - 1) * ny + j]) / h;
                ut[i * ny + j] = uts[i * ny + j] - dt * (pt[i * ny + j] - pt[(i - 1) * ny + j]) / h;
            }
        }
        for i in 0..nx {
            for j in 1..ny {
                self.v[i * (ny + 1) + j] = vs[i * (ny + 1) + j] - dt * (p[i * ny + j] - p[i * ny + j - 1]) / h;
                vt[i * (ny + 1) + j] = vts[i * (ny + 1) + j] - dt * (pt[i * ny + j] - pt[i * ny + j - 1]) / h;
            }
        }
        (-sum_fx * h * h / dt, -sum_fx_dot * h * h / dt)
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

/// A swimmer advanced with an exact forward-sensitivity `∂/∂amp` alongside the primal — the tangent
/// of the *coupled* system: the body's own `(x, vx)` sensitivity feeds back into every marker's
/// position/velocity tangent each step, so `xd` at the end is the exact `∂(displacement)/∂amp`.
#[derive(Clone, Debug)]
pub struct DiffSwimmer {
    pub s: Swimmer,
    pub xd: f64,  // ∂x/∂amp
    pub vxd: f64, // ∂vx/∂amp
    ut: Vec<f64>,
    vt: Vec<f64>,
}

impl DiffSwimmer {
    /// Seed the swimmer and a zeroed fluid tangent field sized to `fluid`.
    pub fn new(s: Swimmer, fluid: &MacFluid) -> Self {
        Self { s, xd: 0.0, vxd: 0.0, ut: vec![0.0; fluid_nu_len(fluid).0], vt: vec![0.0; fluid_nu_len(fluid).1] }
    }

    /// One coupled step carrying the exact `∂/∂amp` tangent.
    pub fn advance(&mut self, fluid: &mut MacFluid) {
        let (markers, vels) = self.s.markers_and_vels();
        let kb = self.s.kb();
        // Per-marker tangents w.r.t. amp. Position: X = (x − len/2 + s, y0 + amp·sin φ); the body's
        // own ∂x/∂amp = xd couples in. Velocity: Ẋ = (vx, −amp·ω·cos φ); ∂vx/∂amp = vxd.
        let (mut dmarker, mut dsurf) = (Vec::with_capacity(self.s.seg), Vec::with_capacity(self.s.seg));
        for m in 0..self.s.seg {
            let sarc = self.s.len * m as f64 / (self.s.seg - 1) as f64;
            let phase = kb * sarc - self.s.omega * self.s.t;
            dmarker.push((self.xd, phase.sin()));
            dsurf.push((self.vxd, -self.s.omega * phase.cos()));
        }
        let (fx, fxd) = fluid.step_markers_sensitivity(&markers, &vels, &dmarker, &dsurf, &mut self.ut, &mut self.vt);
        let dt = fluid_dt(fluid);
        // Integrate primal and tangent body DOFs together.
        self.s.vx += dt * fx / self.s.mass;
        self.s.x += dt * self.s.vx;
        self.vxd += dt * fxd / self.s.mass;
        self.xd += dt * self.vxd;
        self.s.t += dt;
    }
}

/// `u`/`v` field lengths for the fluid (private fields; accessible from this descendant module).
fn fluid_nu_len(f: &MacFluid) -> (usize, usize) {
    (f.u.len(), f.v.len())
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



    /// Primal displacement at a given amplitude, advanced through the SAME coupled primal the
    /// forward sensitivity linearizes (`step_markers_sensitivity`'s primal, via `DiffSwimmer`) —
    /// so the finite difference and the exact tangent differentiate one and the same function.
    /// (`step_markers` is a slightly different discretization — `stencil_u` vs `stencil_u_d` — so
    /// using it for the FD reference would compare apples to oranges.)
    fn cruise_amp(n: usize, dt: f64, steps: usize, amp: f64) -> f64 {
        let mut f = tank(n, 0.006, dt);
        let s = Swimmer::new(0.5, 0.5, 0.35, 28, 0.05, amp, 1.0, 2.0 * PI * 1.0);
        let mut d = DiffSwimmer::new(s, &f);
        for _ in 0..steps {
            d.advance(&mut f);
        }
        d.s.x - 0.5
    }

    /// The **exact** `∂(displacement)/∂amp` — the coupled forward sensitivity threaded through
    /// every predictor, immersed-boundary forcing, projection, AND the body's own free DOF —
    /// against central finite differences.
    #[test]
    fn swim_displacement_gradient_is_exact() {
        let (n, dt, steps, amp) = (48, 5e-4, 900, 0.015);
        let mut f = tank(n, 0.006, dt);
        let s = Swimmer::new(0.5, 0.5, 0.35, 28, 0.05, amp, 1.0, 2.0 * PI * 1.0);
        let mut d = DiffSwimmer::new(s, &f);
        for _ in 0..steps {
            d.advance(&mut f);
        }
        let exact = d.xd; // ∂x/∂amp

        let eps = 1e-6;
        let fd = (cruise_amp(n, dt, steps, amp + eps) - cruise_amp(n, dt, steps, amp - eps)) / (2.0 * eps);
        let rel = (exact - fd).abs() / fd.abs().max(1e-9);
        eprintln!("∂(disp)/∂amp: exact {exact:.6e}  fd {fd:.6e}  rel {rel:.2e}  (disp {:.4e})", d.s.x - 0.5);
        assert!(rel < 1e-4, "swimmer gradient not exact: rel {rel}");
    }

    /// **Payoff: learn to swim on the exact gradient.** Newton on `disp(amp) − target` using the
    /// coupled sensitivity converges in ~2 iterations — the fluid-structure loop is now
    /// differentiable end to end, not just controllable by sampling.
    #[test]
    fn amplitude_learns_to_hit_a_target_by_exact_gradient() {
        let (n, dt, steps) = (48, 5e-4, 900);
        let target = cruise_amp(n, dt, steps, 0.02); // commanded cruise from a known amplitude

        let mut amp = 0.010; // wrong start
        let mut rel = f64::INFINITY;
        for it in 0..6 {
            let mut f = tank(n, 0.006, dt);
            let s = Swimmer::new(0.5, 0.5, 0.35, 28, 0.05, amp, 1.0, 2.0 * PI * 1.0);
            let mut d = DiffSwimmer::new(s, &f);
            for _ in 0..steps {
                d.advance(&mut f);
            }
            let disp = d.s.x - 0.5;
            let grad = d.xd;
            amp -= (disp - target) / grad; // Newton with the exact derivative
            amp = amp.clamp(1e-3, 0.05);
            rel = (amp - 0.02).abs() / 0.02;
            eprintln!("it {it}: amp {amp:.5} disp {disp:.4e} grad {grad:.3e} rel {rel:.2e}");
            if rel < 1e-4 {
                break;
            }
        }
        assert!(rel < 1e-3, "amplitude not learned: rel {rel}");
    }
}
