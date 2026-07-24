//! **Fluids benches** — the wasm rigs behind the verified browser fluid bench. Every 2026 WebGPU
//! fluid demo renders without physics accountability; every verified solver is install-only. These
//! close that gap: the same clean-room, oracle-verified [`ferromotion_fluid`] solvers, rendering in
//! the page *and self-verifying in the page* — the lid-driven cavity checks itself against the Ghia
//! (1982) table as it converges, and the gust bench shows the coefficient model break and the
//! resolved solver reveal the fix, live.

use ferromotion_fluid::gust::{fit_unsteady, run_gust, Gust};
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
