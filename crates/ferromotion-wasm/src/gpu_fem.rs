//! **WebGPU FEM reference** — the data half of stepping the volumetric Neo-Hookean soft body on the
//! *local GPU*. A jelly cube pinned at the top; the mesh, per-tet reference shape (`Dm⁻¹`, rest
//! volume), and a per-vertex → incident-tet adjacency (CSR) are exposed as flat buffers for two WGSL
//! compute kernels: one computes each tet's nodal elastic forces in parallel; the other gathers them
//! per vertex (WebGPU has no f32 atomics, so the scatter is a precomputed gather) and integrates.
//! The same cube stepped by the verified CPU core ([`ferromotion_fem`]) is the reference to check the
//! GPU against. Pure Rust → the data; WGSL → the compute.

use ferromotion_fem::FemSim;
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GpuFemRef {
    fem0: FemSim,      // the initial state; cpu_step(n) replays from here
    live: FemSim,      // an incrementally-stepped copy, for the CPU-fallback animation
    x0: Vec<f32>,
    v0: Vec<f32>,
    tets: Vec<u32>,      // 4 vertex indices per tet
    dm_inv: Vec<f32>,    // 9 per tet (column-major 3×3)
    vol: Vec<f32>,       // |rest volume| per tet
    adj_start: Vec<u32>, // CSR offsets, len n_verts+1
    adj: Vec<u32>,       // packed (tet<<2 | corner) for each (vertex, incident tet)
    pinned: Vec<u32>,    // 1 = fixed
    n_verts: usize,
    n_tets: usize,
}

#[wasm_bindgen]
impl GpuFemRef {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GpuFemRef {
        let mut fem = FemSim::box_grid(3, 3, 3, 0.1, 0.02, 3.0e3, 1.5e3, 2.0e-4);
        fem.damping_rate = 50.505_050_505_050_5; // old per-step 0.01 at dt = 2.0e-4
        fem.gravity = Vector3::new(0.0, 0.0, -9.81);
        fem.floor = None; // pinned-top hanging jelly, no ground contact — the GPU kernel and the CPU
                          // reference then step the exact same equations (elastic + gravity + pins)
        // pin the top layer; give the rest a sideways kick so it swings and wobbles
        let zmax = fem.x.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
        for i in 0..fem.n_verts() {
            if fem.x[i].z > zmax - 1e-6 {
                fem.pinned[i] = true;
            } else {
                fem.v[i] = Vector3::new(0.9, 0.3, 0.0);
            }
        }

        let n_verts = fem.n_verts();
        let n_tets = fem.n_tets();
        let x0: Vec<f32> = fem.x.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
        let v0: Vec<f32> = fem.v.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
        let tets: Vec<u32> = fem.tets().iter().flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32, t[3] as u32]).collect();
        let dm_inv: Vec<f32> = fem.dm_inv().iter().flat_map(|m| m.as_slice().iter().map(|&v| v as f32).collect::<Vec<_>>()).collect();
        let vol: Vec<f32> = fem.vol().iter().map(|&v| v.abs() as f32).collect();
        let pinned: Vec<u32> = fem.pinned.iter().map(|&p| p as u32).collect();

        // per-vertex CSR adjacency of incident (tet, corner)
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
        for (e, t) in fem.tets().iter().enumerate() {
            for (corner, &vtx) in t.iter().enumerate() {
                buckets[vtx].push(((e as u32) << 2) | corner as u32);
            }
        }
        let mut adj_start = Vec::with_capacity(n_verts + 1);
        let mut adj = Vec::new();
        adj_start.push(0u32);
        for b in &buckets {
            adj.extend_from_slice(b);
            adj_start.push(adj.len() as u32);
        }

        let live = fem.clone();
        GpuFemRef { fem0: fem, live, x0, v0, tets, dm_inv, vol, adj_start, adj, pinned, n_verts, n_tets }
    }

    pub fn n_verts(&self) -> usize {
        self.n_verts
    }
    pub fn n_tets(&self) -> usize {
        self.n_tets
    }
    pub fn x0(&self) -> Vec<f32> {
        self.x0.clone()
    }
    pub fn v0(&self) -> Vec<f32> {
        self.v0.clone()
    }
    pub fn tets(&self) -> Vec<u32> {
        self.tets.clone()
    }
    pub fn dm_inv(&self) -> Vec<f32> {
        self.dm_inv.clone()
    }
    pub fn vol(&self) -> Vec<f32> {
        self.vol.clone()
    }
    pub fn adj_start(&self) -> Vec<u32> {
        self.adj_start.clone()
    }
    pub fn adj(&self) -> Vec<u32> {
        self.adj.clone()
    }
    pub fn pinned(&self) -> Vec<u32> {
        self.pinned.clone()
    }
    /// Simulation parameters `[mass, dt, damping, mu, lambda, gx, gy, gz]`.
    pub fn params(&self) -> Vec<f32> {
        let g = self.fem0.gravity;
        vec![self.fem0.mass as f32, self.fem0.dt as f32, self.fem0.damping_rate as f32, self.fem0.mu as f32, self.fem0.lambda as f32, g.x as f32, g.y as f32, g.z as f32]
    }

    /// The verified CPU render: step a fresh copy `n` times from the initial state; return the final
    /// vertex positions (flat `[x,y,z,…]`) for the GPU to be checked against.
    pub fn cpu_step(&self, n: usize) -> Vec<f32> {
        let mut fem = self.fem0.clone();
        for _ in 0..n {
            fem.step();
        }
        fem.x.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect()
    }

    /// Step the incrementally-tracked CPU copy `n` times and return its vertex positions (flat
    /// `[x,y,z,…]`). Used only for the CPU-fallback animation where no local GPU is exposed.
    pub fn step_live(&mut self, n: usize) -> Vec<f32> {
        for _ in 0..n {
            self.live.step();
        }
        self.live.x.iter().flat_map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect()
    }

    /// Reset the incrementally-tracked CPU copy to the initial state.
    pub fn reset_live(&mut self) {
        self.live = self.fem0.clone();
    }

    /// Unique tet edges (flat `[a,b,…]`) for drawing the wireframe.
    pub fn edges(&self) -> Vec<u32> {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for t in self.fem0.tets() {
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let (i, j) = (t[a].min(t[b]), t[a].max(t[b]));
                    set.insert((i as u32, j as u32));
                }
            }
        }
        set.into_iter().flat_map(|(a, b)| [a, b]).collect()
    }
}

impl Default for GpuFemRef {
    fn default() -> Self {
        Self::new()
    }
}
