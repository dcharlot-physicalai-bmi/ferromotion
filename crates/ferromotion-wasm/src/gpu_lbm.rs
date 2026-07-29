//! **WebGPU lattice-Boltzmann fluid** — the most GPU-natural solver in the stack on the local GPU.
//! D2Q9 lattice-Boltzmann is embarrassingly parallel: every cell's collide (local BGK relaxation) and
//! stream (push to neighbours) touch only that cell and its lattice links, so the whole grid advances
//! as two WGSL compute dispatches per step over ping-ponged buffers — the exact scheme the crate's
//! native wgpu path ([`ferromotion_fluid::lbm_gpu`]) already verifies against the CPU reference and the
//! Ghia (1982) lid-driven-cavity benchmark. This is the browser counterpart: the same lattice on
//! whatever GPU the page exposes, graded live against the Ghia table. Pure Rust → the config + CPU
//! oracle; the WGSL kernels are shared with the native path.

use ferromotion_fluid::{LbmBc, LbmD2Q9};
use wasm_bindgen::prelude::*;

const N: usize = 96;
const LID: f64 = 0.1; // lid speed (lattice units) → Re = lid·N/ν = 100 at the chosen τ

// D2Q9 rest-equilibrium weights (ρ = 1, u = 0 ⇒ fₖ = Wₖ)
const W: [f32; 9] = [
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
];

// Ghia, Ghia & Shin (1982), Re=100 lid-driven cavity: vertical-centerline u/lid at these heights.
const GHIA: [f32; 20] = [
    0.0547, -0.03717, 0.1016, -0.06434, 0.2813, -0.15662, 0.4531, -0.21090, 0.5000, -0.20581,
    0.6172, -0.13641, 0.7344, 0.00332, 0.8516, 0.23151, 0.9531, 0.68717, 0.9766, 0.84123,
];

#[wasm_bindgen]
pub struct GpuLbmRef {
    nx: usize,
    ny: usize,
    tau: f64,
    omega: f32,
    lid: f64,
}

#[wasm_bindgen]
impl GpuLbmRef {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GpuLbmRef {
        let tau = 3.0 * (LID * N as f64 / 100.0) + 0.5; // ν = (τ−½)/3 ⇒ Re = 100
        GpuLbmRef { nx: N, ny: N, tau, omega: (1.0 / tau) as f32, lid: LID }
    }

    pub fn nx(&self) -> usize {
        self.nx
    }
    pub fn ny(&self) -> usize {
        self.ny
    }
    pub fn omega(&self) -> f32 {
        self.omega
    }
    pub fn lid_u(&self) -> f32 {
        self.lid as f32
    }
    /// Steps for the cavity to reach steady state (`≈ 40·N/lid` — the native test's budget).
    pub fn converge_steps(&self) -> usize {
        (40.0 * self.nx as f64 / self.lid) as usize
    }
    /// Initial distribution: rest equilibrium `fₖ = Wₖ` per cell (ρ = 1, u = 0), flat `nx·ny·9`.
    pub fn init_f(&self) -> Vec<f32> {
        let mut f = vec![0.0f32; self.nx * self.ny * 9];
        for c in 0..self.nx * self.ny {
            for k in 0..9 {
                f[c * 9 + k] = W[k];
            }
        }
        f
    }
    /// The Ghia (1982) Re=100 centerline table, flat `[y, u/lid, …]` (10 points).
    pub fn ghia(&self) -> Vec<f32> {
        GHIA.to_vec()
    }

    /// The VERIFIED CPU render: run the CPU lattice-Boltzmann cavity `steps` and return its
    /// vertical-centerline `u/lid` profile, flat `[y, u/lid, …]` (`2·ny`) — the cross-oracle.
    pub fn cpu_centerline(&self, steps: usize) -> Vec<f32> {
        let mut c = LbmD2Q9::new(self.nx, self.ny, self.tau, LbmBc::Cavity { lid_u: self.lid });
        c.set_velocity(|_, _| (0.0, 0.0)); // rest start, matching the GPU init
        for _ in 0..steps {
            c.step();
        }
        c.centerline_u().iter().flat_map(|&(y, u)| [y as f32, u as f32]).collect()
    }
}

impl Default for GpuLbmRef {
    fn default() -> Self {
        Self::new()
    }
}
