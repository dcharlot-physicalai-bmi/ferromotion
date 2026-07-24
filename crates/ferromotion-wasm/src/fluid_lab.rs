//! **Fluids benches** — the wasm rigs behind the verified browser fluid bench. Every 2026 WebGPU
//! fluid demo renders without physics accountability; every verified solver is install-only. These
//! close that gap: the same clean-room, oracle-verified [`ferromotion_fluid`] solvers, rendering in
//! the page *and self-verifying in the page* — the lid-driven cavity checks itself against the Ghia
//! (1982) table as it converges, and the gust bench shows the coefficient model break and the
//! resolved solver reveal the fix, live.

use ferromotion_fluid::gust::{fit_unsteady, run_gust, Gust};
use ferromotion_fluid::sph::Sph;
use ferromotion_fluid::swimmer::{DiffSwimmer, Swimmer};
use ferromotion_fluid::MacFluid;
use wasm_bindgen::prelude::*;

/// Ghia et al. (1982) Re=100 lid-driven cavity — u along the vertical centerline, `(y, u/U)`.
const GHIA_RE100: &[(f64, f64)] = &[
    (0.0547, -0.03717),
    (0.1016, -0.06434),
    (0.2813, -0.15662),
    (0.4531, -0.21090),
    (0.5000, -0.20581),
    (0.6172, -0.13641),
    (0.7344, 0.00332),
    (0.8516, 0.23151),
    (0.9531, 0.68717),
    (0.9766, 0.84123),
];

/// **The lid-driven cavity that grades itself.** Runs the verified MAC projection solver at Re=100
/// and, on demand, reports its own worst deviation from the Ghia reference table — the number that
/// makes this a *verified* browser bench, not eye-candy.
#[wasm_bindgen]
pub struct CavityLab {
    fluid: MacFluid,
    n: usize,
    lid: f64,
    steps: usize,
}

#[wasm_bindgen]
impl CavityLab {
    /// A cavity on an `n × n` grid at the given Reynolds number (unit box, unit lid speed → ν = 1/Re).
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, re: f64) -> CavityLab {
        let lid = 1.0;
        let nu = lid / re; // Re = U·L/ν, L = 1
        let dt = 0.25 / n as f64 / lid; // advective CFL ≈ 0.25
        CavityLab { fluid: MacFluid::new(n, n, nu, dt, lid), n, lid, steps: 0 }
    }

    /// Advance `k` steps; returns the max per-step velocity change (the steady-state monitor).
    pub fn step(&mut self, k: usize) -> f64 {
        let mut last = 0.0;
        for _ in 0..k {
            last = self.fluid.step();
        }
        self.steps += k;
        last
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Cell-centered speed field `|u|`, row-major `n×n` (for rendering).
    pub fn speed(&self) -> Vec<f64> {
        let h = 1.0 / self.n as f64;
        let mut out = vec![0.0; self.n * self.n];
        for j in 0..self.n {
            for i in 0..self.n {
                let (u, v) = self.fluid.velocity_at((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
                out[j * self.n + i] = (u * u + v * v).sqrt();
            }
        }
        out
    }

    /// Max cell divergence — should be ~machine-zero after each projection (a live correctness receipt).
    pub fn divergence(&self) -> f64 {
        self.fluid.max_divergence()
    }

    pub fn energy(&self) -> f64 {
        self.fluid.kinetic_energy()
    }

    /// **The live verification number**: worst deviation of the current centerline profile from the
    /// Ghia Re=100 table. Falls toward ≈0.004 as the cavity converges (only meaningful at Re=100).
    pub fn ghia_deviation(&self) -> f64 {
        let profile = self.fluid.centerline_u();
        let interp = |y: f64| -> f64 {
            for w in profile.windows(2) {
                let ((y0, u0), (y1, u1)) = (w[0], w[1]);
                if (y0..=y1).contains(&y) {
                    return u0 + (u1 - u0) * (y - y0) / (y1 - y0);
                }
            }
            profile.last().map(|p| p.1).unwrap_or(0.0)
        };
        GHIA_RE100
            .iter()
            .map(|&(y, u_ref)| (interp(y) / self.lid - u_ref).abs())
            .fold(0.0f64, f64::max)
    }

    /// The Ghia reference points as a flat `[y0,u0, y1,u1, …]` array (for overlaying on the plot).
    pub fn ghia_reference(&self) -> Vec<f64> {
        GHIA_RE100.iter().flat_map(|&(y, u)| [y, u]).collect()
    }

    /// The current centerline profile as a flat `[y0,u0, …]` array (normalized `u/U`).
    pub fn centerline(&self) -> Vec<f64> {
        self.fluid.centerline_u().into_iter().flat_map(|(y, u)| [y, u / self.lid]).collect()
    }
}

/// **The gust bench** — the deployment-honesty seam, live. Runs the resolved solver through a gust
/// and fits the nested unsteady-force hierarchy, exposing the residual RMS at each level so the page
/// can show the coefficient model breaking and the resolved solver revealing the fix.
#[wasm_bindgen]
pub struct GustLab {
    trace_t: Vec<f64>,
    trace_u: Vec<f64>,
    trace_f: Vec<f64>,
    qs: f64,
    am: f64,
    hi: f64,
    frms: f64,
}

#[wasm_bindgen]
impl GustLab {
    /// Run a gust of relative amplitude `gust_amp` on a base flow, fit the hierarchy. Compact grid
    /// so it finishes in well under a second in the browser.
    #[wasm_bindgen(constructor)]
    pub fn new(gust_amp: f64) -> GustLab {
        let (n, nu, dt, r) = (72, 0.006, 5e-4, 0.08);
        let steps = 1100;
        let t_end = steps as f64 * dt;
        let gust = Gust { u0: 0.05, amp: gust_amp, t0: t_end * 0.5, width: t_end * 0.1, ramp: t_end * 0.12 };
        let trace = run_gust(n, nu, dt, r, steps, gust);
        let win = &trace[steps / 4..3 * steps / 4];
        let fit = fit_unsteady(win, dt);
        GustLab {
            trace_t: win.iter().map(|s| s.t).collect(),
            trace_u: win.iter().map(|s| s.u).collect(),
            trace_f: win.iter().map(|s| s.f).collect(),
            qs: fit.quasi_steady_rms,
            am: fit.added_mass_rms,
            hi: fit.history_rms,
            frms: fit.force_rms,
        }
    }

    /// Residual RMS with quasi-steady drag only (the pure coefficient model).
    pub fn quasi_steady_rms(&self) -> f64 {
        self.qs
    }
    /// Residual RMS after adding the Morison added-mass term.
    pub fn added_mass_rms(&self) -> f64 {
        self.am
    }
    /// Residual RMS after adding the Basset history term.
    pub fn history_rms(&self) -> f64 {
        self.hi
    }
    /// RMS of the resolved force itself (the scale to compare residuals against).
    pub fn force_rms(&self) -> f64 {
        self.frms
    }
    pub fn trace_time(&self) -> Vec<f64> {
        self.trace_t.clone()
    }
    pub fn trace_speed(&self) -> Vec<f64> {
        self.trace_u.clone()
    }
    pub fn trace_force(&self) -> Vec<f64> {
        self.trace_f.clone()
    }
}

/// **The undulatory swimmer** — a self-propelled filament (emergent thrust). Renders the body and
/// the fluid it pushes, and reports the net displacement it earns.
#[wasm_bindgen]
pub struct SwimLab {
    fluid: MacFluid,
    diff: DiffSwimmer,
    n: usize,
}

#[wasm_bindgen]
impl SwimLab {
    /// A swimmer at rest, with the exact `∂displacement/∂amp` carried alongside so the page can show
    /// "learn to swim" on the real gradient.
    #[wasm_bindgen(constructor)]
    pub fn new(amp: f64, freq_hz: f64) -> SwimLab {
        let n = 64;
        let fluid = MacFluid::new(n, n, 0.006, 4e-4, 0.0).with_free_slip();
        let s = Swimmer::new(0.5, 0.5, 0.35, 28, 0.05, amp, 1.0, std::f64::consts::TAU * freq_hz);
        let diff = DiffSwimmer::new(s, &fluid);
        SwimLab { fluid, diff, n }
    }

    /// Advance `k` coupled steps.
    pub fn step(&mut self, k: usize) {
        for _ in 0..k {
            self.diff.advance(&mut self.fluid);
        }
    }

    /// Net streamwise displacement from the start (emergent thrust).
    pub fn displacement(&self) -> f64 {
        self.diff.s.x - 0.5
    }

    /// Exact `∂(displacement)/∂amp` accumulated through the coupled sensitivity.
    pub fn grad_amp(&self) -> f64 {
        self.diff.xd
    }

    /// Body marker world positions as a flat `[x0,y0, x1,y1, …]` array (for drawing the filament).
    pub fn body(&self) -> Vec<f64> {
        let (markers, _) = self.diff.s.markers_and_vels();
        markers.into_iter().flat_map(|(x, y)| [x, y]).collect()
    }

    /// Cell-centered speed field `|u|`, row-major `n×n`.
    pub fn speed(&self) -> Vec<f64> {
        let h = 1.0 / self.n as f64;
        let mut out = vec![0.0; self.n * self.n];
        for j in 0..self.n {
            for i in 0..self.n {
                let (u, v) = self.fluid.velocity_at((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
                out[j * self.n + i] = (u * u + v * v).sqrt();
            }
        }
        out
    }

    pub fn grid(&self) -> usize {
        self.n
    }
}

/// **The dam break** — smoothed particle hydrodynamics, the Lagrangian solver. A column of fluid
/// collapses and surges across the box; the page renders the particles and reads the exact mass
/// conservation and released kinetic energy as it goes.
#[wasm_bindgen]
pub struct SphLab {
    sph: Sph,
    dt: f64,
    m0: f64,
}

#[wasm_bindgen]
impl SphLab {
    #[allow(clippy::new_without_default)]
    #[wasm_bindgen(constructor)]
    pub fn new() -> SphLab {
        let sph = Sph::dam_break(0.02);
        let dt = sph.cfl_dt();
        let m0 = sph.n_fluid as f64 * sph.mass;
        SphLab { sph, dt, m0 }
    }

    /// Advance `k` SPH steps.
    pub fn step(&mut self, k: usize) {
        for _ in 0..k {
            self.sph.step(self.dt);
        }
    }

    /// Fluid particle positions as a flat `[x0,y0, x1,y1, …]` array (fluid only; box is ≈1.0 wide).
    pub fn particles(&self) -> Vec<f64> {
        self.sph.pos[..self.sph.n_fluid].iter().flat_map(|p| [p[0], p[1]]).collect()
    }

    /// Boundary particle positions (the box walls), flat `[x,y,…]`.
    pub fn walls(&self) -> Vec<f64> {
        self.sph.pos[self.sph.n_fluid..].iter().flat_map(|p| [p[0], p[1]]).collect()
    }

    /// Relative mass drift from the start (should stay at machine zero — fixed particle count).
    pub fn mass_drift(&self) -> f64 {
        let m = self.sph.n_fluid as f64 * self.sph.mass;
        (m - self.m0).abs() / self.m0
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.sph.kinetic_energy()
    }

    pub fn n_fluid(&self) -> usize {
        self.sph.n_fluid
    }
}

/// **The honesty harness** — the doctrine flagship, live. Builds a divergence-free ground truth and
/// two predictions with IDENTICAL error budget: one honest, one that cheats (injects divergence or
/// high-frequency noise). Exposes the shared MSE and the physics receipts so the page can show the
/// audit catching what the error metric cannot.
#[wasm_bindgen]
pub struct HarnessLab {
    n: usize,
    truth: ferromotion_fluid::harness::FlowField,
    honest: ferromotion_fluid::harness::FlowField,
    cheat: ferromotion_fluid::harness::FlowField,
    reference: ferromotion_fluid::harness::Receipts,
}

#[wasm_bindgen]
impl HarnessLab {
    /// `mode` 0 = divergence cheat, 1 = spectral (high-frequency) cheat.
    #[wasm_bindgen(constructor)]
    pub fn new(mode: u32) -> HarnessLab {
        use ferromotion_fluid::harness::{audit, FlowField};
        use std::f64::consts::PI;
        let n = 72;
        let k = 2.0 * PI;
        let truth = FlowField::sample(n, |x, y| ((k * x).sin() * (k * y).cos(), -(k * x).cos() * (k * y).sin()));
        let reference = audit(&truth);
        let norm = truth.norm();
        let budget = 0.13 * norm;
        let scaled = |mut f: FlowField, target: f64| {
            let s = target / f.norm();
            for i in 0..f.u.len() {
                f.u[i] *= s;
                f.v[i] *= s;
            }
            f
        };
        let honest_p = scaled(FlowField::sample(n, |x, y| ((2.0 * k * x).sin() * (2.0 * k * y).cos(), -(2.0 * k * x).cos() * (2.0 * k * y).sin())), budget);
        let cheat_p = if mode == 0 {
            // pure gradient — all divergence
            scaled(FlowField::sample(n, |x, y| (-(k * x).sin() * (k * y).cos(), -(k * x).cos() * (k * y).sin())), budget)
        } else {
            // high-frequency mode — all roughness
            scaled(FlowField::sample(n, |x, y| ((10.0 * k * x).sin() * (10.0 * k * y).cos(), -(10.0 * k * x).cos() * (10.0 * k * y).sin())), budget)
        };
        let mut honest = truth.clone();
        honest.add_scaled(&honest_p, 1.0);
        let mut cheat = truth.clone();
        cheat.add_scaled(&cheat_p, 1.0);
        HarnessLab { n, truth, honest, cheat, reference }
    }

    pub fn grid(&self) -> usize {
        self.n
    }

    fn field(&self, which: u32) -> &ferromotion_fluid::harness::FlowField {
        match which {
            0 => &self.truth,
            1 => &self.honest,
            _ => &self.cheat,
        }
    }

    /// Speed field `|u|` for truth(0)/honest(1)/cheat(2), row-major `n×n`.
    pub fn speed(&self, which: u32) -> Vec<f64> {
        let f = self.field(which);
        (0..self.n * self.n).map(|k| (f.u[k] * f.u[k] + f.v[k] * f.v[k]).sqrt()).collect()
    }

    /// RMS error of honest(1)/cheat(2) vs the truth — equal by construction.
    pub fn mse(&self, which: u32) -> f64 {
        self.field(which).rms_diff(&self.truth)
    }

    /// Divergence receipt of a field.
    pub fn divergence(&self, which: u32) -> f64 {
        ferromotion_fluid::harness::audit(self.field(which)).divergence_rms
    }

    /// Roughness receipt of a field.
    pub fn roughness(&self, which: u32) -> f64 {
        ferromotion_fluid::harness::audit(self.field(which)).roughness
    }

    /// Whether a field passes the physics audit (graded against the truth's receipts).
    pub fn passes(&self, which: u32) -> bool {
        ferromotion_fluid::harness::grade(self.field(which), &self.reference, 3.0).passes
    }
}

/// **Environment-scale flow + planning** — the fluids→robot-planning seam, live. A divergence-free
/// wind field with a no-fly obstacle; a Zermelo min-time planner routes around the headwind pockets,
/// and the page shows the wind, the flow-aware path, and the naive straight line it beats.
#[wasm_bindgen]
pub struct EnvLab {
    n: usize,
    wind: ferromotion_fluid::env::Wind,
    obstacle: ferromotion_fluid::env::Obstacle,
    path: Vec<(usize, usize)>,
    opt_time: f64,
    naive_time: f64,
    start: (usize, usize),
    goal: (usize, usize),
}

#[wasm_bindgen]
impl EnvLab {
    #[wasm_bindgen(constructor)]
    pub fn new(amp: f64) -> EnvLab {
        use ferromotion_fluid::env::{Obstacle, Planner, Wind};
        let n = 64;
        let wind = Wind { ux: 0.0, uy: 0.0, amp };
        let obstacle = Obstacle { cx: 0.5, cy: 0.42, r: 0.1 };
        let start = (5, n / 2);
        let goal = (n - 6, n / 2);
        let planner = Planner { n, wind, obstacles: vec![obstacle], airspeed: 3.0 };
        let (opt_time, path) = planner.plan(start, goal).unwrap_or((f64::INFINITY, vec![start, goal]));
        let naive_time = planner.straight_line_time(start, goal, 200);
        EnvLab { n, wind, obstacle, path, opt_time, naive_time, start, goal }
    }

    pub fn grid(&self) -> usize {
        self.n
    }

    /// Wind speed `|w|` field, row-major `n×n` (for the heatmap).
    pub fn wind_speed(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                let (x, y) = ((i as f64 + 0.5) / self.n as f64, (j as f64 + 0.5) / self.n as f64);
                let (u, v) = self.wind.at(x, y);
                out[j * self.n + i] = (u * u + v * v).sqrt();
            }
        }
        out
    }

    /// Flow-aware path as flat normalized `[x0,y0, x1,y1, …]`.
    pub fn path(&self) -> Vec<f64> {
        self.path.iter().flat_map(|&(i, j)| [(i as f64 + 0.5) / self.n as f64, (j as f64 + 0.5) / self.n as f64]).collect()
    }

    /// Straight-line naive path endpoints `[x0,y0, x1,y1]`.
    pub fn straight(&self) -> Vec<f64> {
        let n = self.n as f64;
        vec![
            (self.start.0 as f64 + 0.5) / n,
            (self.start.1 as f64 + 0.5) / n,
            (self.goal.0 as f64 + 0.5) / n,
            (self.goal.1 as f64 + 0.5) / n,
        ]
    }

    /// Obstacle `[cx, cy, r]`.
    pub fn obstacle(&self) -> Vec<f64> {
        vec![self.obstacle.cx, self.obstacle.cy, self.obstacle.r]
    }

    pub fn opt_time(&self) -> f64 {
        self.opt_time
    }
    pub fn naive_time(&self) -> f64 {
        self.naive_time
    }
}
